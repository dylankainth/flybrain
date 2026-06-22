#!/usr/bin/env python3
"""
FLYBRAIN TELLO - REAL BRAIN + REAL CAMERA
==========================================
Uses the actual Drosophila connectome (139,255 neurons) with
REAL optic flow computed from the Tello's camera feed.

This is the full vision-guided neural flight system.

SPACEBAR = Emergency motor kill
ESC = Safe landing

Requirements:
- opencv-python (pip install opencv-python)
- GPU recommended for real-time neural simulation
"""

import os
import time
import threading
import numpy as np
import pandas as pd
import torch
import cv2
from djitellopy import tello
from pynput import keyboard

# Global variables
drone = None
emergency_stop = False
safe_landing = False
running = True
brain = None
current_optic_flow = np.array([0.15, 0.1, 0.1, 0.12], dtype=np.float32)
vision_thread = None

print("=" * 70)
print("FLYBRAIN TELLO - REAL BRAIN + REAL CAMERA")
print("=" * 70)

# ============================================================================
# LOAD REAL FLY BRAIN
# ============================================================================

print("\n[LOADING] Drosophila connectome...", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n", flush=True)

# Load neurons
neurons_df = pd.read_csv('fly_neurons_real.csv')
n_neurons = len(neurons_df)

# Identify key neuron types by EXACT cell type. The old substring regex
# misclassified cells: 'R1|R2|...' matched 'ER3d'/'FR1'/'ExR3' (central-complex
# ring neurons) and 'DN' matched 's-CPDN3A'. Real photoreceptors are
# R1-6/R7/R8; descending neuron types start with 'DN'.
PR_TYPES = ['R1-6', 'R7', 'R8']
photoreceptors = neurons_df[neurons_df['primary_type'].isin(PR_TYPES)]
motion_neurons = neurons_df[neurons_df['primary_type'].str.contains(r'^(?:T4|T5|Tm|Mi|Lo)', na=False, regex=True)]
descending = neurons_df[neurons_df['primary_type'].str.startswith('DN', na=False)]

pr_indices = set(photoreceptors.index)
motion_indices = set(motion_neurons.index)
dn_indices = set(descending.index)

print(f"  Neurons: {n_neurons:,}")
print(f"  Photoreceptors: {len(photoreceptors)}")
print(f"  Motion neurons: {len(motion_neurons)}")
print(f"  Descending neurons: {len(descending)}")

# Load the FULL connectome (~80M synapses, 3.7 GB). Stream the file in chunks
# and keep only the numeric edge arrays so peak memory stays bounded.
# (Previously only every 100th synapse was kept -> ~1% of the wiring.)
# FLYBRAIN_SYNAPSE_STRIDE=N is a hardware fallback (1 = full connectome).
SYNAPSE_STRIDE = int(os.environ.get('FLYBRAIN_SYNAPSE_STRIDE', '1'))
print(f"\n[LOADING] Synaptic connectivity (stride={SYNAPSE_STRIDE})...", flush=True)

root_id_to_idx = {rid: idx for idx, rid in enumerate(neurons_df['root_id'].values)}
is_inhibitory_np = neurons_df['nt_type'].isin(['GABA']).values

pre_list, post_list, size_list = [], [], []
for chunk in pd.read_csv('fly_synapses_real.csv',
                         usecols=['pre_root_id', 'post_root_id', 'size'],
                         chunksize=4_000_000, low_memory=False):
    if SYNAPSE_STRIDE > 1:
        chunk = chunk.iloc[::SYNAPSE_STRIDE]
    pre = chunk['pre_root_id'].map(root_id_to_idx).fillna(-1).to_numpy(dtype=np.int64)
    post = chunk['post_root_id'].map(root_id_to_idx).fillna(-1).to_numpy(dtype=np.int64)
    valid = (pre >= 0) & (post >= 0)
    pre_list.append(pre[valid].astype(np.int32))
    post_list.append(post[valid].astype(np.int32))
    size_list.append(chunk['size'].to_numpy(dtype=np.float32)[valid])

pre_idx = np.concatenate(pre_list)
post_idx = np.concatenate(post_list)
sizes = np.concatenate(size_list)
del pre_list, post_list, size_list

# Normalize weights, then apply Dale's law: presynaptic inhibitory (GABA)
# neurons get a negative sign so inhibition actually inhibits. (Previously
# inhibitory input was added then subtracted, netting exactly zero effect.)
weights = np.clip(sizes / (sizes.max() + 1e-6), 0.1, 2.0).astype(np.float32)
weights *= np.where(is_inhibitory_np[pre_idx], -1.0, 1.0).astype(np.float32)

indices = torch.from_numpy(np.stack([pre_idx, post_idx])).long().to(device)
values = torch.from_numpy(weights).to(device)
connectivity = torch.sparse_coo_tensor(indices, values, (n_neurons, n_neurons),
                                       device=device).coalesce()

print(f"  Synapses: {len(pre_idx):,}")

# ============================================================================
# REAL FLY BRAIN CONTROLLER
# ============================================================================

class RealFlyBrain:
    """Full Drosophila brain with Leaky Integrate-and-Fire neurons"""
    
    def __init__(self, connectivity, neurons_df, pr_indices, dn_indices, motion_indices, n_neurons, device, substeps=20):
        self.connectivity = connectivity
        self.conn_T = connectivity.t().coalesce()  # precompute transpose once
        self.neurons_df = neurons_df
        self.dn_indices = torch.tensor(sorted(list(dn_indices)), dtype=torch.long, device=device)
        self.motion_indices = torch.tensor(sorted(list(motion_indices)), dtype=torch.long, device=device)
        self.n_neurons = n_neurons
        self.device = device
        self.substeps = substeps  # neural steps per control cycle (dt*substeps ~ loop period)

        # Published LIF parameters
        self.V_rest = -0.052
        self.V_thresh = -0.045
        self.R_m = 10.0
        self.C_m = 2.0e-6
        self.tau_syn = 5e-3
        self.W_syn = 0.275e-3
        self.tau_ref = 2.2e-3
        self.dt = 1e-3
        self._syn_decay = float(np.exp(-self.dt / self.tau_syn))

        # Photoreceptor mapping by exact type. NOTE: this dataset has no side /
        # coordinate fields, so a true left/right hemisphere split is not
        # possible; the L/R split below is an explicit arbitrary proxy.
        r16 = sorted(list(neurons_df[neurons_df['primary_type'] == 'R1-6'].index))
        r78 = sorted(list(neurons_df[neurons_df['primary_type'].isin(['R7', 'R8'])].index))
        r16_left = r16[:len(r16) // 2]
        r16_right = r16[len(r16) // 2:]

        # Precompute index tensors for vectorized sensory injection
        self.r16_idx = torch.tensor(r16, dtype=torch.long, device=device)
        self.r16_left_idx = torch.tensor(r16_left, dtype=torch.long, device=device)
        self.r16_right_idx = torch.tensor(r16_right, dtype=torch.long, device=device)
        self.r78_idx = torch.tensor(r78, dtype=torch.long, device=device)

        self.reset_state()

    def reset_state(self):
        """Reset all neurons to resting state"""
        self.v = torch.ones(self.n_neurons, dtype=torch.float32, device=self.device) * self.V_rest
        self.spikes = torch.zeros(self.n_neurons, dtype=torch.bool, device=self.device)
        self.in_refractory = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        self.i_syn = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)

    def _step(self, i_input):
        """One 1ms LIF timestep given a fixed sensory input current."""
        self.in_refractory = torch.clamp(self.in_refractory - self.dt, min=0.0)

        # Signed synaptic transmission (sign baked into weights -> single matmul)
        i_syn_input = torch.sparse.mm(
            self.conn_T, self.spikes.float().unsqueeze(1)
        ).squeeze() * self.W_syn / self.tau_syn
        self.i_syn = self.i_syn * self._syn_decay + i_syn_input
        i_total = i_input + self.i_syn

        # Membrane dynamics (LIF)
        tau_m = self.R_m * self.C_m
        dv = (self.V_rest - self.v + i_total * self.R_m) / tau_m
        self.v = torch.clamp(self.v + dv * self.dt, -0.1, 0.05)

        # Spiking
        can_spike = (self.in_refractory <= 0.0)
        self.spikes = (self.v > self.V_thresh) & can_spike
        self.v[self.spikes] = self.V_rest
        self.in_refractory[self.spikes] = self.tau_ref

    def compute(self, optic_flow):
        """Run `substeps` timesteps of neural simulation and decode motor commands"""
        optic_flow_t = torch.from_numpy(optic_flow).float().to(self.device)

        # Vectorized sensory injection into photoreceptors
        i_input = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        i_input[self.r16_idx] += optic_flow_t[0] * 8e-11
        i_input[self.r16_left_idx] += optic_flow_t[1] * 1.2e-10
        i_input[self.r16_right_idx] += optic_flow_t[2] * 1.2e-10
        i_input[self.r78_idx] += optic_flow_t[3] * 1e-10

        # Run sub-steps; accumulate spikes over the window
        spike_accum = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        for _ in range(self.substeps):
            self._step(i_input)
            spike_accum += self.spikes.float()

        # Decode from mean per-step firing rate (motor scaling stays independent
        # of the number of sub-steps).
        rate = spike_accum / self.substeps
        dn_activity = rate[self.dn_indices]
        n_dn = len(self.dn_indices)

        if n_dn > 0:
            forward_drive = dn_activity.sum() / (n_dn / 100.0)

            motion_rate = rate[self.motion_indices]
            if len(self.motion_indices) > 10:
                n_motion = len(self.motion_indices)
                turn_signal = (motion_rate[:n_motion // 2].sum() - motion_rate[n_motion // 2:].sum()) * 0.1
            else:
                turn_signal = torch.zeros((), device=self.device)

            climb_drive = dn_activity.sum() / (n_dn / 50.0) - 1.0
        else:
            zero = torch.zeros((), device=self.device)
            forward_drive = turn_signal = climb_drive = zero

        # Convert to Tello commands
        motor = {
            'forward': int(torch.clamp(torch.tanh(forward_drive / 100.0) * 100, -100, 100).item()),
            'yaw': int(torch.clamp(torch.tanh(turn_signal) * 100, -100, 100).item()),
            'vertical': int(torch.clamp(torch.tanh(climb_drive / 50.0) * 100, -80, 80).item()),
            'roll': 0
        }

        return motor


print("\n[INIT] Building fly brain controller...", flush=True)
brain = RealFlyBrain(connectivity, neurons_df, pr_indices, dn_indices, motion_indices, n_neurons, device)
print("✓ Brain ready!\n")

# ============================================================================
# VISION SYSTEM - REAL CAMERA OPTIC FLOW
# ============================================================================

class VisionProcessor:
    """Process Tello camera feed to compute optic flow"""
    
    def __init__(self, drone):
        self.drone = drone
        self.prev_frame = None
        self.prev_gray = None
        
    def compute_optic_flow(self, frame):
        """
        Compute optic flow from camera frames.
        Returns [forward, left, right, vertical] motion signals.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.array([0.15, 0.1, 0.1, 0.12], dtype=np.float32)
        
        # Compute dense optical flow (Farneback method)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Split frame into regions
        h, w = gray.shape
        h_mid = h // 2
        w_mid = w // 2
        
        # Extract flow from different regions
        # Forward motion: expansion from center
        center_flow = flow[h_mid-20:h_mid+20, w_mid-20:w_mid+20]
        forward_motion = np.mean(np.abs(center_flow))
        
        # Left/right motion: horizontal flow in left/right halves
        left_region = flow[:, :w_mid]
        right_region = flow[:, w_mid:]
        left_motion = np.mean(np.abs(left_region[:, :, 0]))
        right_motion = np.mean(np.abs(right_region[:, :, 0]))
        
        # Vertical motion: vertical flow in bottom region
        bottom_region = flow[h_mid:, :]
        vertical_motion = np.mean(np.abs(bottom_region[:, :, 1]))
        
        # Normalize to 0-1 range
        forward = np.clip(forward_motion * 0.1, 0.05, 0.3)
        left = np.clip(left_motion * 0.05, 0.05, 0.25)
        right = np.clip(right_motion * 0.05, 0.05, 0.25)
        vertical = np.clip(vertical_motion * 0.08, 0.05, 0.25)
        
        self.prev_gray = gray
        
        return np.array([forward, left, right, vertical], dtype=np.float32)


def vision_processing_thread(drone):
    """Background thread for processing camera feed"""
    global current_optic_flow, running
    
    print("[VISION] Starting camera stream...")
    
    try:
        drone.streamon()
        time.sleep(2)
        print("✓ Camera streaming")
        
        vision_processor = VisionProcessor(drone)
        
        while running:
            try:
                frame = drone.get_frame_read().frame
                
                if frame is not None:
                    # Compute optic flow from camera
                    optic_flow = vision_processor.compute_optic_flow(frame)
                    current_optic_flow = optic_flow
                    
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"[VISION ERROR] {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"[VISION INIT ERROR] {e}")


# ============================================================================
# KEYBOARD CONTROL
# ============================================================================

def on_press(key):
    """Handle keyboard events"""
    global emergency_stop, safe_landing, drone, running
    
    try:
        if key == keyboard.Key.space:
            emergency_stop = True
            running = False
            print("\n" + "=" * 60)
            print("⚠️  EMERGENCY STOP - KILLING MOTORS!")
            print("=" * 60)
            if drone:
                try:
                    drone.emergency()
                except Exception as e:
                    print(f"Error: {e}")
                    
        elif key == keyboard.Key.esc:
            safe_landing = True
            running = False
            print("\n" + "=" * 60)
            print("🛬 SAFE LANDING")
            print("=" * 60)
            if drone:
                try:
                    drone.land()
                except Exception as e:
                    print(f"Error: {e}")
                    
    except AttributeError:
        pass


def setup_keyboard_listener():
    """Setup keyboard listener"""
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("✓ Keyboard listener active")
    print("  SPACEBAR = Kill motors")
    print("  ESC = Safe landing")
    print()
    return listener


# ============================================================================
# MAIN DEPLOYMENT
# ============================================================================

def main():
    """Main deployment loop"""
    global drone, emergency_stop, safe_landing, running, brain, current_optic_flow, vision_thread
    
    print("[DEPLOYMENT] Flybrain Tello - Vision-Guided Neural Flight")
    print("=" * 60)
    
    # Setup keyboard
    listener = setup_keyboard_listener()
    
    # Initialize Tello
    drone = tello.Tello()
    
    try:
        # Connect
        print("[CONNECT] Connecting to Tello...")
        drone.connect()
        print("✓ Connected!")
        
        # Battery check
        battery = drone.get_battery()
        print(f"🔋 Battery: {battery}%")
        
        if battery < 20:
            print("❌ Battery too low!")
            return
        
        # Start vision processing thread
        vision_thread = threading.Thread(target=vision_processing_thread, args=(drone,), daemon=True)
        vision_thread.start()
        time.sleep(3)
        
        print("\n[TAKEOFF] Launching...")
        print("-" * 60)
        
        # Takeoff
        drone.takeoff()
        time.sleep(3)
        
        print("✓ Airborne!")
        print("\n[NEURAL + VISION] Camera-guided brain flight")
        print("-" * 60)
        print()
        
        # Flight loop
        iteration = 0
        start_time = time.time()
        last_command_time = time.time()
        
        while running and not emergency_stop and not safe_landing:
            iteration += 1
            current_time = time.time()
            
            # Get REAL optic flow from camera
            optic_flow = current_optic_flow.copy()
            
            # Neural computation (139,255 neurons)
            commands = brain.compute(optic_flow)
            
            # Send to drone
            if current_time - last_command_time >= 0.05:
                try:
                    drone.send_rc_control(
                        commands['roll'],
                        commands['forward'],
                        commands['vertical'],
                        commands['yaw']
                    )
                    last_command_time = current_time
                except Exception as e:
                    print(f"[ERROR] RC: {e}")
                    break
            
            # Status
            if iteration % 30 == 0:
                n_spikes = brain.spikes.sum().item()
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Iter {iteration:4d} | "
                      f"Spikes: {n_spikes:5d} | "
                      f"Flow:[{optic_flow[0]:.2f},{optic_flow[1]:.2f},"
                      f"{optic_flow[2]:.2f},{optic_flow[3]:.2f}] | "
                      f"V:{commands['vertical']:3d} F:{commands['forward']:3d}")
            
            time.sleep(0.02)
        
        # Exit handling
        print()
        print("-" * 60)
        
        if emergency_stop:
            print("[STATUS] Motors killed")
            time.sleep(1)
        elif safe_landing:
            print("[STATUS] Landing...")
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.land()
            time.sleep(3)
        
        print("[SHUTDOWN] Closing...")
        
        # Stop camera
        try:
            drone.streamoff()
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Landing...")
        if drone:
            try:
                drone.land()
            except:
                pass
                
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if drone:
            try:
                drone.emergency()
            except:
                pass
    
    finally:
        running = False
        if drone:
            try:
                drone.end()
            except:
                pass
        
        print("✓ Connection closed")
        print("[DONE] Vision-guided neural flight complete")


if __name__ == "__main__":
    main()
