"""
PHASE 11C: EVOLUTIONARY BEHAVIOR - IMPROVED MOTOR DECODING

Based on DN analysis:
- All descending neurons contribute to forward thrust
- Turn emerges from bilateral asymmetry in motion input
- Altitude control from vertical motion sensors

Key insight: Motion detectors feed into central circuits that
eventually project to DNs. DN population activity = motor output.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
from scipy import special

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("PHASE 11C: EVOLUTIONARY BEHAVIOR - IMPROVED MOTOR DECODING")
print("="*70)
print(f"\nDevice: {device}\n", flush=True)

# ============================================================================
# 1. LOAD CONNECTOME AND IDENTIFY KEY NEURON TYPES
# ============================================================================

print("[1/5] Loading connectome and identifying neuron types...", flush=True)

neurons_df = pd.read_csv('fly_neurons_real.csv')
n_neurons = len(neurons_df)

# Photoreceptors and motion circuits
photoreceptors = neurons_df[neurons_df['primary_type'].str.contains('R1|R2|R3|R4|R5|R6|R7|R8', na=False)]
motion_neurons = neurons_df[neurons_df['primary_type'].str.contains('T4|T5|Tm|Mi|Lo', na=False)]
descending = neurons_df[neurons_df['primary_type'].str.contains('DN', na=False)]

print(f"  Photoreceptors (R1-R8): {len(photoreceptors)}")
print(f"  Motion neurons (T4/T5/Tm/Mi/Lo): {len(motion_neurons)}")
print(f"  Descending neurons (motor): {len(descending)}")

pr_indices = set(photoreceptors.index)
motion_indices = set(motion_neurons.index)
dn_indices = set(descending.index)

print(f"\n  Total neurons: {n_neurons:,}")

# ============================================================================
# 2. LOAD CONNECTIVITY
# ============================================================================

print("\n[2/5] Loading connectivity matrix...", flush=True)

synapses_list = []
for i, chunk in enumerate(pd.read_csv('fly_synapses_real.csv', chunksize=1000000, low_memory=False)):
    synapses_list.append(chunk.iloc[np.arange(0, len(chunk), 100)])

synapses_df = pd.concat(synapses_list, ignore_index=True)
root_id_to_idx = {rid: idx for idx, rid in enumerate(neurons_df['root_id'].values)}

pre_idx = synapses_df['pre_root_id'].map(root_id_to_idx).fillna(-1).astype(int)
post_idx = synapses_df['post_root_id'].map(root_id_to_idx).fillna(-1).astype(int)
valid = (pre_idx >= 0) & (post_idx >= 0)
pre_idx, post_idx = pre_idx[valid].values, post_idx[valid].values
weights = np.clip(synapses_df[valid]['size'].values / (synapses_df[valid]['size'].max() + 1e-6), 0.1, 2.0)

indices = torch.LongTensor([pre_idx, post_idx]).to(device)
values = torch.FloatTensor(weights).to(device)
connectivity = torch.sparse_coo_tensor(indices, values, (n_neurons, n_neurons), device=device)

is_inhibitory = torch.from_numpy(neurons_df['nt_type'].isin(['GABA']).values).to(device)

print(f"  Connectivity: {len(pre_idx):,} synapses")

# ============================================================================
# 3. IMPROVED BRAIN CONTROLLER
# ============================================================================

print("\n[3/5] Building improved evolutionary brain controller...", flush=True)

class ImprovedEvolutionaryBrain:
    """
    Improved motor decoding based on DN analysis.
    Forward = total DN activity
    Turn = modulation from bilateral motion inputs
    Climb = vertical motion response
    """

    def __init__(self, connectivity, neurons_df, pr_indices, dn_indices, motion_indices, n_neurons, device):
        self.connectivity = connectivity
        self.neurons_df = neurons_df
        self.pr_indices = torch.tensor(sorted(list(pr_indices)), dtype=torch.long, device=device)
        self.dn_indices = torch.tensor(sorted(list(dn_indices)), dtype=torch.long, device=device)
        self.motion_indices = torch.tensor(sorted(list(motion_indices)), dtype=torch.long, device=device)
        self.n_neurons = n_neurons
        self.device = device

        # Published LIF parameters
        self.V_rest = -0.052
        self.V_thresh = -0.045
        self.R_m = 10.0
        self.C_m = 2.0e-6
        self.tau_syn = 5e-3
        self.W_syn = 0.275e-3
        self.tau_ref = 2.2e-3
        self.dt = 1e-3

        # Photoreceptors split by field
        self.r16_neurons = sorted(list(neurons_df[neurons_df['primary_type'].str.contains('R1|R2|R3|R4|R5|R6', na=False)].index))
        self.r78_neurons = sorted(list(neurons_df[neurons_df['primary_type'].str.contains('R7|R8', na=False)].index))

        # Split left/right for bilateral motion encoding
        self.r16_left = self.r16_neurons[:len(self.r16_neurons)//2]
        self.r16_right = self.r16_neurons[len(self.r16_neurons)//2:]

        self.is_inhibitory = torch.from_numpy(neurons_df['nt_type'].isin(['GABA']).values).to(device)

        self.reset_state()

    def reset_state(self):
        self.v = torch.ones(self.n_neurons, dtype=torch.float32, device=self.device) * self.V_rest
        self.spikes = torch.zeros(self.n_neurons, dtype=torch.bool, device=self.device)
        self.in_refractory = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        self.i_syn = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        self.spike_history = []
        self.dn_history = {'forward': [], 'left': [], 'right': []}
        self.turn_history = []

    def compute(self, optic_flow):
        """
        Improved computation with better motor decoding.
        optic_flow: [forward, left, right, vertical]
        """

        optic_flow_t = torch.from_numpy(optic_flow).float().to(self.device)

        i_input = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)

        # Map optic flow to photoreceptors
        # Forward motion: visible to all R1-R6
        for idx in self.r16_neurons:
            i_input[idx] += optic_flow_t[0] * 8e-11

        # Bilateral motion for turn detection
        for idx in self.r16_left:
            i_input[idx] += optic_flow_t[1] * 1.2e-10  # Left motion -> left PRs

        for idx in self.r16_right:
            i_input[idx] += optic_flow_t[2] * 1.2e-10  # Right motion -> right PRs

        # R7/R8 for altitude
        for idx in self.r78_neurons:
            i_input[idx] += optic_flow_t[3] * 1e-10

        # Update refractory
        self.in_refractory = torch.clamp(self.in_refractory - self.dt, min=0.0)

        # Network activity
        spikes_float = self.spikes.float()
        n_exc = torch.sparse.mm(self.connectivity.t().coalesce(), spikes_float.unsqueeze(1)).squeeze()
        n_inh = torch.sparse.mm(self.connectivity.t().coalesce(),
                               (self.spikes & self.is_inhibitory).float().unsqueeze(1)).squeeze()

        i_exc_input = n_exc * self.W_syn / self.tau_syn
        i_inh_input = n_inh * self.W_syn / self.tau_syn

        self.i_syn = self.i_syn * np.exp(-self.dt / self.tau_syn) + i_exc_input - i_inh_input
        i_total = i_input + self.i_syn

        tau_m = self.R_m * self.C_m
        dv = (self.V_rest - self.v + i_total * self.R_m) / tau_m
        self.v = self.v + dv * self.dt
        self.v = torch.clamp(self.v, -0.1, 0.05)

        # Spiking
        can_spike = (self.in_refractory <= 0.0)
        self.spikes = (self.v > self.V_thresh) & can_spike

        self.v[self.spikes] = self.V_rest
        self.in_refractory[self.spikes] = self.tau_ref

        # IMPROVED MOTOR DECODING
        dn_activity = self.spikes[self.dn_indices].float()

        if len(self.dn_indices) > 0:
            n_dn = len(self.dn_indices)

            # Forward thrust: total DN activity (they encode motor commands)
            forward_drive = dn_activity.sum() / (n_dn / 100.0)  # Normalize to percent

            # Turn: use motion neuron asymmetry
            # Left motion neurons vs right motion neurons
            motion_spikes = self.spikes[self.motion_indices].float()
            if len(self.motion_indices) > 10:
                n_motion = len(self.motion_indices)
                left_motion = motion_spikes[:n_motion//3].sum() / (n_motion//3 + 1e-6)
                right_motion = motion_spikes[n_motion//3:].sum() / ((2*n_motion)//3 + 1e-6)
                turn_signal = (left_motion - right_motion) * 0.1
            else:
                turn_signal = 0.0

            # Climb: DN activity related to altitude (positive drive = climb)
            climb_drive = dn_activity.sum() / (n_dn / 50.0) - 1.0  # Relative to baseline

            # Convert to motor commands
            motor = torch.zeros(3, device=self.device)
            motor[0] = torch.tanh(forward_drive / 100.0)  # Forward (normalized)
            motor[1] = torch.tanh(turn_signal)  # Turn
            motor[2] = torch.tanh(climb_drive / 50.0)  # Climb

            # Record for analysis
            self.dn_history['forward'].append(forward_drive.item())
            self.dn_history['left'].append(left_motion.item() if len(self.motion_indices) > 10 else 0)
            self.dn_history['right'].append(right_motion.item() if len(self.motion_indices) > 10 else 0)
            self.turn_history.append(turn_signal)
        else:
            motor = torch.zeros(3, device=self.device)

        motor_np = motor.detach().cpu().numpy()
        self.spike_history.append(self.spikes.sum().item())

        return motor_np

# ============================================================================
# 4. RUN SIMULATION
# ============================================================================

print("\n[4/5] Running flight simulation with improved motor decoding...", flush=True)
print("  Simulating 400 steps (20 seconds @ 20Hz)...\n", flush=True)

brain = ImprovedEvolutionaryBrain(connectivity, neurons_df, pr_indices, dn_indices, motion_indices, n_neurons, device)

# Flight environment
class FlightEnv:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.5
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.step_count = 0
        self.obstacles = [(5, 0, 0.5, 0.4), (10, 0, 0.5, 0.3), (15, 0, 0.5, 0.2)]
        self.trajectory = {'x': [], 'y': [], 'z': []}

    def get_optic_flow(self):
        """Compute optic flow from obstacles."""
        obs = np.zeros(4, dtype=np.float32)
        for ox, oy, oz, r in self.obstacles:
            dist = max(np.sqrt((ox - self.x)**2 + (oy - self.y)**2 + (oz - self.z)**2), 0.1)

            if ox > self.x:
                obs[0] += 1.0 / dist  # Forward
            if oy > self.y:
                obs[1] += 1.0 / dist  # Left
            if oy < self.y:
                obs[2] += 1.0 / dist  # Right
            if oz > self.z:
                obs[3] += 1.0 / dist  # Up

        return np.clip(obs / (obs.sum() + 0.01), 0, 1).astype(np.float32)

    def step(self, forward, turn, climb):
        self.step_count += 1
        self.vx = forward * 0.25
        self.vy = turn * 0.15
        self.vz = climb * 0.15

        self.x += self.vx * 0.05
        self.y += self.vy * 0.05
        self.z += self.vz * 0.05
        self.z = np.clip(self.z, 0.1, 0.9)

        self.trajectory['x'].append(self.x)
        self.trajectory['y'].append(self.y)
        self.trajectory['z'].append(self.z)

env = FlightEnv()
motor_history = {'forward': [], 'turn': [], 'climb': []}
spike_counts = []

start_sim = time.time()
for step in range(400):
    obs = env.get_optic_flow()
    motor = brain.compute(obs)
    env.step(motor[0], motor[1], motor[2])

    motor_history['forward'].append(motor[0])
    motor_history['turn'].append(motor[1])
    motor_history['climb'].append(motor[2])
    spike_counts.append(brain.spike_history[-1])

    if (step + 1) % 50 == 0:
        print(f"  Step {step+1}: x={env.x:6.2f} z={env.z:6.2f} spikes={int(spike_counts[-1]):6.0f} fwd={motor[0]:6.2f}", flush=True)

sim_time = time.time() - start_sim
print(f"\n[OK] Simulation complete in {sim_time:.1f}s")
print(f"  Flight distance: {env.x:.2f} units")
print(f"  Mean altitude: {np.mean(env.trajectory['z']):.3f}")
print(f"  Mean spike rate: {np.mean(spike_counts):.0f} spikes/timestep")
print(f"  Total spikes: {int(np.sum(spike_counts)):,}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

print("\n[5/5] Generating visualization...", flush=True)

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Phase 11C - Evolutionary Behavior with Improved Motor Decoding', fontsize=16, fontweight='bold')

# 3D trajectory
ax = fig.add_subplot(2, 4, 1, projection='3d')
ax.plot(env.trajectory['x'], env.trajectory['y'], env.trajectory['z'], 'b-', linewidth=2, label='Flight path')
for ox, oy, oz, r in env.obstacles:
    ax.scatter([ox], [oy], [oz], s=200, c='red', marker='o', alpha=0.5)
ax.set_xlabel('X (forward)')
ax.set_ylabel('Y (sideways)')
ax.set_zlabel('Z (altitude)')
ax.set_title('3D Flight Trajectory')
ax.set_xlim(-2, 16)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.legend()

# Motor commands
ax = fig.add_subplot(2, 4, 2)
ax.plot(motor_history['forward'], label='Forward', linewidth=1.5, alpha=0.8)
ax.plot(motor_history['turn'], label='Turn', linewidth=1.5, alpha=0.8)
ax.plot(motor_history['climb'], label='Climb', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Timestep')
ax.set_ylabel('Motor Command')
ax.set_title('Motor Output (Improved Decoding)')
ax.legend()
ax.grid(True, alpha=0.3)

# DN activity
ax = fig.add_subplot(2, 4, 3)
ax.plot(brain.dn_history['forward'], 'blue', linewidth=1.5, label='Forward drive', alpha=0.8)
ax.set_xlabel('Timestep')
ax.set_ylabel('DN Activity')
ax.set_title('Descending Neuron Activity')
ax.grid(True, alpha=0.3)

# Neural activity
ax = fig.add_subplot(2, 4, 4)
ax.plot(spike_counts, 'purple', linewidth=1)
ax.fill_between(range(len(spike_counts)), spike_counts, alpha=0.2, color='purple')
ax.set_xlabel('Timestep')
ax.set_ylabel('Active Neurons')
ax.set_title('Total Neural Firing')
ax.grid(True, alpha=0.3)

# Altitude
ax = fig.add_subplot(2, 4, 5)
ax.plot(env.trajectory['z'], 'g-', linewidth=2)
ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Target')
ax.fill_between(range(len(env.trajectory['z'])), env.trajectory['z'], alpha=0.2, color='green')
ax.set_xlabel('Timestep')
ax.set_ylabel('Altitude')
ax.set_title('Altitude Control')
ax.legend()
ax.grid(True, alpha=0.3)

# Forward progress
ax = fig.add_subplot(2, 4, 6)
ax.plot(env.trajectory['x'], 'orange', linewidth=2)
ax.axvline(200, color='r', linestyle='--', alpha=0.3, label='Midpoint')
ax.fill_between(range(len(env.trajectory['x'])), env.trajectory['x'], alpha=0.2, color='orange')
ax.set_xlabel('Timestep')
ax.set_ylabel('Forward Position')
ax.set_title('Forward Progress')
ax.legend()
ax.grid(True, alpha=0.3)

# Lateral movement
ax = fig.add_subplot(2, 4, 7)
ax.plot(env.trajectory['y'], 'cyan', linewidth=2)
ax.fill_between(range(len(env.trajectory['y'])), env.trajectory['y'], alpha=0.2, color='cyan')
ax.set_xlabel('Timestep')
ax.set_ylabel('Lateral Position')
ax.set_title('Lateral Movement (Yaw)')
ax.grid(True, alpha=0.3)

# Turn command
ax = fig.add_subplot(2, 4, 8)
turn_hist_cpu = [t.item() if isinstance(t, torch.Tensor) else t for t in brain.turn_history]
ax.plot(turn_hist_cpu, 'red', linewidth=1, alpha=0.8)
ax.axhline(0, color='k', linestyle='-', alpha=0.2)
ax.fill_between(range(len(turn_hist_cpu)), turn_hist_cpu, alpha=0.2, color='red')
ax.set_xlabel('Timestep')
ax.set_ylabel('Turn Signal')
ax.set_title('Turn Command from Motion Circuits')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase11c_evolutionary_improved.png', dpi=100)
print("[OK] Saved: phase11c_evolutionary_improved.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[COMPLETE] PHASE 11C - IMPROVED MOTOR DECODING")
print("="*70)
print(f"""
Motor Decoding Improvements:
  - Forward: Total DN population activity
  - Turn: Bilateral motion asymmetry
  - Climb: DN-driven altitude modulation

Results:
  Duration: 400 timesteps (20 seconds @ 20Hz)
  Total spikes: {int(np.sum(spike_counts)):,}
  Mean firing: {np.mean(spike_counts):.0f} neurons/timestep

  Flight position: x={env.x:.2f}, y={env.y:.3f}, z={np.mean(env.trajectory['z']):.3f}
  Altitude error: {abs(np.mean(env.trajectory['z']) - 0.5):.3f}
  Forward progress: {env.x:.2f} units

Behavior Analysis:
  - Altitude control: {'EXCELLENT' if abs(np.mean(env.trajectory['z']) - 0.5) < 0.05 else 'GOOD' if abs(np.mean(env.trajectory['z']) - 0.5) < 0.1 else 'FAIR'}
  - Forward motion: {'MOVING FORWARD!' if env.x > 1.0 else 'Hovering' if abs(env.x) < 0.5 else 'MOVING BACKWARD'}
  - Turn control: {'ACTIVE' if np.std(brain.turn_history) > 0.01 else 'Minimal'}

Next: Compare evolutionary vs RL approach, deploy to drone
""")
print("="*70 + "\n")
