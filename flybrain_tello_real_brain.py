#!/usr/bin/env python3
"""
FLYBRAIN TELLO DRONE - REAL NEURAL DEPLOYMENT
==============================================
Deploys the ACTUAL Drosophila melanogaster brain (139,255 neurons)
to control a Tello drone with real-time LIF neural simulation.

This uses the full FlyWire connectome with biophysical parameters.

SPACEBAR = Emergency motor kill (instant)
ESC = Safe landing and exit

Requirements:
- GPU recommended (CUDA) for real-time performance
- ~4GB GPU memory
- Connectome data files (fly_neurons_real.csv, fly_synapses_real.csv)
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
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for threading compatibility
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Global variables
drone = None
emergency_stop = False
safe_landing = False
running = True
brain = None

# Live plotting data buffers
plot_data = {
    'time': deque(maxlen=200),
    'optic_flow_forward': deque(maxlen=200),
    'optic_flow_left': deque(maxlen=200),
    'optic_flow_right': deque(maxlen=200),
    'optic_flow_vertical': deque(maxlen=200),
    'motor_forward': deque(maxlen=200),
    'motor_yaw': deque(maxlen=200),
    'motor_vertical': deque(maxlen=200),
    'spike_count': deque(maxlen=200),
    'dn_activity': deque(maxlen=200),
    'motion_activity': deque(maxlen=200),
}
plot_lock = threading.Lock()
start_time_global = None

print("=" * 70)
print("FLYBRAIN TELLO DEPLOYMENT - REAL NEURAL CONTROLLER")
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
    """
    Full Drosophila brain with 139,255 Leaky Integrate-and-Fire neurons.
    Uses published biophysical parameters from Shiu et al., Nature 2024.
    """
    
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
        # Update refractory period
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
        """
        Run `substeps` timesteps of neural simulation (so simulated time tracks
        the control loop) and decode motor commands.

        Args:
            optic_flow: np.array [forward, left, right, vertical] (0-1 normalized)

        Returns:
            (motor dict for Tello, neural_metrics dict)
        """
        optic_flow_t = torch.from_numpy(optic_flow).float().to(self.device)

        # Vectorized sensory injection into photoreceptors
        i_input = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        i_input[self.r16_idx] += optic_flow_t[0] * 8e-11         # Forward motion
        i_input[self.r16_left_idx] += optic_flow_t[1] * 1.2e-10  # Left motion
        i_input[self.r16_right_idx] += optic_flow_t[2] * 1.2e-10 # Right motion
        i_input[self.r78_idx] += optic_flow_t[3] * 1e-10         # Vertical

        return self._run_and_decode(i_input)

    def compute_retinotopic(self, pr_current, cell_idx):
        """
        Drive R1-6 photoreceptors directly with per-cell luminance current from
        the retinotopic eye map (flybrain_eye_map.EyeMap) instead of optic-flow
        scalars. Motion (T4/T5) is then computed downstream by the connectome
        itself -- the biologically correct direction of information flow.

        Args:
            pr_current: np.array of luminance (0-1) per assigned R1-6 cell
            cell_idx:   np.array of neuron indices aligned with pr_current

        Returns:
            (motor dict for Tello, neural_metrics dict)
        """
        i_input = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        cur = torch.from_numpy(np.asarray(pr_current, dtype=np.float32)).to(self.device)
        idx = torch.from_numpy(np.asarray(cell_idx)).long().to(self.device)
        i_input.index_add_(0, idx, cur * 8e-11)  # same scale as the flow path
        return self._run_and_decode(i_input)

    def _run_and_decode(self, i_input):
        """Run `substeps` LIF steps with a fixed input current and decode motors."""
        # Run sub-steps; accumulate spikes over the window
        spike_accum = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        for _ in range(self.substeps):
            self._step(i_input)
            spike_accum += self.spikes.float()

        # Decode from mean per-step firing rate so motor scaling stays
        # independent of the number of sub-steps.
        rate = spike_accum / self.substeps
        dn_activity = rate[self.dn_indices]
        n_dn = len(self.dn_indices)

        if n_dn > 0:
            # Forward thrust from DN population activity
            forward_drive = dn_activity.sum() / (n_dn / 100.0)

            # Turn from bilateral motion asymmetry
            motion_rate = rate[self.motion_indices]
            if len(self.motion_indices) > 10:
                n_motion = len(self.motion_indices)
                turn_signal = (motion_rate[:n_motion // 2].sum() - motion_rate[n_motion // 2:].sum()) * 0.1
            else:
                turn_signal = torch.zeros((), device=self.device)

            # Altitude from DN activity
            climb_drive = dn_activity.sum() / (n_dn / 50.0) - 1.0
        else:
            zero = torch.zeros((), device=self.device)
            forward_drive = turn_signal = climb_drive = zero

        # Convert to Tello RC commands (-100 to +100)
        motor = {
            'forward': int(torch.clamp(torch.tanh(forward_drive / 100.0) * 100, -100, 100).item()),
            'yaw': int(torch.clamp(torch.tanh(turn_signal) * 100, -100, 100).item()),
            'vertical': int(torch.clamp(torch.tanh(climb_drive / 50.0) * 100, -80, 80).item()),
            'roll': 0  # Can add lateral control if needed
        }

        # Neural activity metrics for visualization (mean per-step over window)
        neural_metrics = {
            'total_spikes': spike_accum.sum().item() / self.substeps,
            'dn_activity': dn_activity.sum().item(),
            'motion_activity': rate[self.motion_indices].sum().item() if len(self.motion_indices) > 0 else 0.0,
        }

        return motor, neural_metrics


print("\n[INIT] Building real fly brain controller...", flush=True)
brain = RealFlyBrain(connectivity, neurons_df, pr_indices, dn_indices, motion_indices, n_neurons, device)
print("[OK] Brain ready!\n")

# ----------------------------------------------------------------------------
# Optional retinotopic eye-map front-end (EXPERIMENTAL; bench-test props-off).
# FLYBRAIN_RETINOTOPIC=1 feeds per-ommatidium luminance (real Buchner-1971 eye
# geometry) into R1-6 instead of optic-flow scalars; motion is then computed by
# the connectome itself. See flybrain_eye_map.py and docs/eye_map.md.
# ----------------------------------------------------------------------------
RETINOTOPIC = os.environ.get('FLYBRAIN_RETINOTOPIC', '0') == '1'
eye_map = cell_idx = omma_of_cell = None
vis_mask = vis_left = vis_right = vis_upper = None
if RETINOTOPIC:
    from flybrain_eye_map import EyeMap
    eye_map = EyeMap()
    cell_idx, omma_of_cell = eye_map.assign_photoreceptors(
        brain.r16_left_idx.cpu().numpy(), brain.r16_right_idx.cpu().numpy())
    vis_mask = eye_map.visible_mask()
    vis_left = vis_mask & eye_map.left_mask
    vis_right = vis_mask & eye_map.right_mask
    vis_upper = vis_mask & (eye_map.elevation > 0)
    print(f"[RETINOTOPIC] {eye_map.n} ommatidia, {vis_mask.sum()} in camera FOV; "
          f"{len(cell_idx)} R1-6 cells assigned")

# ----------------------------------------------------------------------------
# Vertical control (OPEN-LOOP). Closed-loop altitude hold needs a height sensor,
# but this Tello reports height=0 / unreliable values (see the state-decode
# warnings at connect), which made the P-controller saturate and pin the
# throttle to max climb. Instead: a brief climb to clear the ground, then damp
# the brain's vertical toward ~zero-mean and let the Tello's *built-in*
# barometric hover hold altitude (it does that automatically when rc vert ~ 0).
# This keeps the brain's dynamic forward/yaw movement while taming the climb.
# ----------------------------------------------------------------------------
LAUNCH_CLIMB_S = 2.0     # brief gentle climb after takeoff to clear the ground
LAUNCH_CLIMB_CMD = 25    # rc vertical during that climb
VERT_GAIN = 0.3          # fraction of the brain's vertical to keep
VERT_TRIM = 8            # subtract the brain's upward bias (prevents slow climb)
VERT_LIMIT = 25          # cap |vertical| so it can't bolt to the ceiling
YAW_LIMIT = 50           # cap brain yaw (raw decode swings ~+/-70 = spins in place)

# ============================================================================
# KEYBOARD CONTROL
# ============================================================================

def on_press(key):
    """Handle keyboard press events"""
    global emergency_stop, safe_landing, drone, running
    
    try:
        if key == keyboard.Key.space:
            emergency_stop = True
            running = False
            print("\n" + "=" * 60)
            print("[EMERGENCY] STOP - KILLING MOTORS!")
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
            print("[LANDING] SAFE LANDING INITIATED")
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
    print("[OK] Keyboard listener active")
    print("  SPACEBAR = Kill motors (emergency)")
    print("  ESC = Safe landing")
    print()
    return listener


prev_gray = None

def get_gray_frame():
    """Grab the current Tello camera frame as grayscale (or None)."""
    frame = drone.get_frame_read().frame
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def get_optic_flow_from_sensors():
    """
    Compute optic flow from Tello's camera using Farneback optical flow.
    Returns [forward, left, right, vertical] normalized 0-1.
    """
    global prev_gray, drone

    frame = drone.get_frame_read().frame
    if frame is None:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 120))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    global prev_gray
    if prev_gray is None:
        prev_gray = gray
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = gray

    # Decompose flow into motion components
    h, w = flow.shape[:2]
    cx, cy = w // 2, h // 2

    left_half = flow[:, :cx]
    right_half = flow[:, cx:]
    top_half = flow[:cy]
    bottom_half = flow[cy:]

    # Forward: mean outward radial flow magnitude
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    x_centered = x_grid - cx
    y_centered = y_grid - cy
    radial_out = (flow[:, :, 0] * x_centered + flow[:, :, 1] * y_centered) / (np.hypot(x_centered, y_centered) + 1e-6)
    forward = float(np.clip(np.mean(radial_out) * 0.5, 0.0, 1.0))

    # Left/right: horizontal flow asymmetry
    left_flow = float(np.mean(np.abs(left_half[:, :, 0])))
    right_flow = float(np.mean(np.abs(right_half[:, :, 0])))
    total = left_flow + right_flow + 1e-6
    left = float(np.clip(left_flow / total * 2.0, 0.0, 1.0))
    right = float(np.clip(right_flow / total * 2.0, 0.0, 1.0))

    # Vertical: top vs bottom vertical flow
    top_v = float(np.mean(np.abs(top_half[:, :, 1])))
    bot_v = float(np.mean(np.abs(bottom_half[:, :, 1])))
    v_total = top_v + bot_v + 1e-6
    vertical = float(np.clip(bot_v / v_total * 2.0, 0.0, 1.0))

    return np.array([forward, left, right, vertical], dtype=np.float32)


# ============================================================================
# LIVE PLOTTING
# ============================================================================

# Global plot objects
fig = None
axes = None
lines_optic = []
lines_motor = []
lines_neural = []
plot_initialized = False

def init_plots():
    """Initialize plot window (call once from main thread)"""
    global fig, axes, lines_optic, lines_motor, lines_neural, plot_initialized
    
    plt.ion()  # Interactive mode
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.canvas.manager.set_window_title('FlyBrain Live Telemetry')
    
    ax_optic = axes[0]
    ax_motor = axes[1]
    ax_neural = axes[2]
    
    ax_optic.set_title('Optic Flow Sensors', fontsize=12, fontweight='bold')
    ax_optic.set_ylabel('Flow (0-1)')
    ax_optic.set_ylim(0, 1.0)
    ax_optic.set_xlim(0, 20)
    ax_optic.grid(True, alpha=0.3)
    
    ax_motor.set_title('Motor Commands', fontsize=12, fontweight='bold')
    ax_motor.set_ylabel('Command')
    ax_motor.set_ylim(-100, 100)
    ax_motor.set_xlim(0, 20)
    ax_motor.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_motor.grid(True, alpha=0.3)
    
    ax_neural.set_title('Neural Activity', fontsize=12, fontweight='bold')
    ax_neural.set_xlabel('Time (s)')
    ax_neural.set_ylabel('Spikes')
    ax_neural.set_ylim(0, 2000)
    ax_neural.set_xlim(0, 20)
    ax_neural.grid(True, alpha=0.3)
    
    # Initialize lines
    lines_optic = [
        ax_optic.plot([], [], label='Fwd', color='blue', linewidth=1.5)[0],
        ax_optic.plot([], [], label='Left', color='green', linewidth=1.5)[0],
        ax_optic.plot([], [], label='Right', color='red', linewidth=1.5)[0],
        ax_optic.plot([], [], label='Vert', color='purple', linewidth=1.5)[0],
    ]
    
    lines_motor = [
        ax_motor.plot([], [], label='Fwd', color='blue', linewidth=1.5)[0],
        ax_motor.plot([], [], label='Yaw', color='orange', linewidth=1.5)[0],
        ax_motor.plot([], [], label='Vert', color='green', linewidth=1.5)[0],
    ]
    
    lines_neural = [
        ax_neural.plot([], [], label='Total', color='red', linewidth=1.5)[0],
        ax_neural.plot([], [], label='DN', color='blue', linewidth=1.5)[0],
        ax_neural.plot([], [], label='Motion', color='green', linewidth=1.5)[0],
    ]
    
    ax_optic.legend(loc='upper right', fontsize=8, ncol=4)
    ax_motor.legend(loc='upper right', fontsize=8, ncol=3)
    ax_neural.legend(loc='upper right', fontsize=8, ncol=3)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    plot_initialized = True
    print("[OK] Live plotting initialized")


def update_plots():
    """Update plots with latest data (call from main thread periodically)"""
    global plot_data, plot_lock, lines_optic, lines_motor, lines_neural, axes, fig, plot_initialized
    
    if not plot_initialized or fig is None:
        return
    
    try:
        with plot_lock:
            if len(plot_data['time']) < 2:
                return
            
            times = list(plot_data['time'])
            
            # Update all lines
            lines_optic[0].set_data(times, list(plot_data['optic_flow_forward']))
            lines_optic[1].set_data(times, list(plot_data['optic_flow_left']))
            lines_optic[2].set_data(times, list(plot_data['optic_flow_right']))
            lines_optic[3].set_data(times, list(plot_data['optic_flow_vertical']))
            
            lines_motor[0].set_data(times, list(plot_data['motor_forward']))
            lines_motor[1].set_data(times, list(plot_data['motor_yaw']))
            lines_motor[2].set_data(times, list(plot_data['motor_vertical']))
            
            lines_neural[0].set_data(times, list(plot_data['spike_count']))
            lines_neural[1].set_data(times, list(plot_data['dn_activity']))
            lines_neural[2].set_data(times, list(plot_data['motion_activity']))
            
            # Auto-scale x-axis (20 second window)
            if times:
                x_min = max(0, times[-1] - 20)
                x_max = times[-1] + 1
                for ax in axes:
                    ax.set_xlim(x_min, x_max)
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
    except Exception as e:
        pass  # Silently ignore plot errors


# ============================================================================
# MAIN DEPLOYMENT
# ============================================================================

def main():
    """Main deployment loop"""
    global drone, emergency_stop, safe_landing, running, brain, start_time_global
    
    print("[DEPLOYMENT] Flybrain Tello - Real Neural Controller")
    print("=" * 60)
    
    # Setup keyboard
    listener = setup_keyboard_listener()
    
    # Initialize live plotting (from main thread)
    init_plots()
    
    # Initialize Tello
    drone = tello.Tello()
    
    try:
        # Connect
        print("[CONNECT] Connecting to Tello...")
        drone.connect()
        drone.streamon()
        print("[OK] Connected!")
        
        # Battery check
        battery = drone.get_battery()
        print(f"[BATTERY] Battery: {battery}%")
        
        if battery < 20:
            print("[ERROR] Battery too low!")
            return
        
        print("\n[TAKEOFF] Launching...")
        print("-" * 60)
        try:
            print(f"[STATE] battery={drone.get_battery()}%  "
                  f"temp={drone.get_temperature()}C  "
                  f"height={drone.get_height()}cm")
        except Exception:
            pass

        # Takeoff -- if the Tello rejects it (low battery, not level, being held,
        # IMU not ready) it raises. Surface that clearly instead of blindly
        # killing motors via the outer handler.
        try:
            drone.takeoff()
        except Exception as e:
            print("\n" + "=" * 60)
            print(f"[TAKEOFF FAILED] Drone rejected takeoff: {e}")
            print("Common causes: battery too low for flight (recharge -- repeated")
            print("test flights drain it fast), not on a flat/level surface, the")
            print("drone is being held, or the IMU needs calibration.")
            print("=" * 60)
            try:
                drone.land()
            except Exception:
                pass
            return
        time.sleep(3)

        # Confirm it actually got airborne
        try:
            h0 = drone.get_height()
            print(f"[OK] Airborne! height ~{h0} cm")
            if h0 <= 0:
                print("[WARN] Height reads 0 -- drone may not have left the ground.")
        except Exception:
            print("[OK] Airborne! (height read unavailable)")

        # Send immediate hover commands to prevent auto-land
        print("[OK] Sending hover commands...")
        for i in range(10):
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.05)

        print("[OK] Hover stabilized!")
        print("\n[NEURAL CONTROL] 139,255 neurons now flying the drone")
        print("-" * 60)
        print()
        
        # Flight loop
        iteration = 0
        start_time = time.time()
        start_time_global = start_time
        last_command_time = time.time()
        
        while running and not emergency_stop and not safe_landing:
            iteration += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Get sensory input + run neural computation (139,255 neurons)
            if RETINOTOPIC:
                gray = get_gray_frame()
                if gray is None:
                    optic_flow = np.zeros(4, dtype=np.float32)
                    commands, neural_metrics = brain.compute(optic_flow)
                else:
                    intensity = eye_map.sample(gray)
                    commands, neural_metrics = brain.compute_retinotopic(
                        intensity[omma_of_cell], cell_idx)
                    # 4-vector summary for the telemetry plot only
                    optic_flow = np.array([
                        intensity[vis_mask].mean() if vis_mask.any() else 0.0,
                        intensity[vis_left].mean() if vis_left.any() else 0.0,
                        intensity[vis_right].mean() if vis_right.any() else 0.0,
                        intensity[vis_upper].mean() if vis_upper.any() else 0.0,
                    ], dtype=np.float32)
            else:
                optic_flow = get_optic_flow_from_sensors()
                commands, neural_metrics = brain.compute(optic_flow)

            # --- Vertical control (open-loop; no usable height sensor) ---
            if elapsed < LAUNCH_CLIMB_S:
                commands['vertical'] = LAUNCH_CLIMB_CMD   # brief climb to clear ground
            else:
                # Damp toward ~zero-mean; the Tello's built-in hover holds altitude.
                commands['vertical'] = int(np.clip(
                    commands['vertical'] * VERT_GAIN - VERT_TRIM, -VERT_LIMIT, VERT_LIMIT))

            # Cap yaw so the noisy turn decode doesn't just spin it in place
            commands['yaw'] = int(np.clip(commands['yaw'], -YAW_LIMIT, YAW_LIMIT))

            # Update plot data
            with plot_lock:
                plot_data['time'].append(elapsed)
                plot_data['optic_flow_forward'].append(optic_flow[0])
                plot_data['optic_flow_left'].append(optic_flow[1])
                plot_data['optic_flow_right'].append(optic_flow[2])
                plot_data['optic_flow_vertical'].append(optic_flow[3])
                plot_data['motor_forward'].append(commands['forward'])
                plot_data['motor_yaw'].append(commands['yaw'])
                plot_data['motor_vertical'].append(commands['vertical'])
                plot_data['spike_count'].append(neural_metrics['total_spikes'])
                plot_data['dn_activity'].append(neural_metrics['dn_activity'])
                plot_data['motion_activity'].append(neural_metrics['motion_activity'])
            
            # Send to drone (rate limited)
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
                    print(f"[ERROR] RC control: {e}")
                    break
            
            # Status display
            if iteration % 30 == 0:
                n_spikes = neural_metrics['total_spikes']
                print(f"[{elapsed:.1f}s] Iter {iteration:4d} | "
                      f"Spikes: {n_spikes:5.0f} | "
                      f"V:{commands['vertical']:3d} F:{commands['forward']:3d} "
                      f"Y:{commands['yaw']:3d}")
            
            # Update plots periodically (every 5 iterations = ~100ms)
            if iteration % 5 == 0:
                update_plots()
            
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
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Landing...")
        if drone:
            try:
                drone.land()
            except:
                pass
                
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        if drone:
            try:
                drone.emergency()
            except:
                pass
    
    finally:
        if drone:
            try:
                drone.end()
            except:
                pass
        
        print("[OK] Connection closed")
        print("[DONE] Real fly brain deployment complete")


if __name__ == "__main__":
    main()
