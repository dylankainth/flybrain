"""
PHASE 11: EVOLUTIONARY BEHAVIOR - USE CONNECTOME'S NATIVE CIRCUITS

Instead of training artificial sensory gains and readout weights,
use the connectome's evolved sensory and motor pathways directly.

Approach:
1. Identify real sensory input neurons (photoreceptors, motion circuits)
2. Identify real motor output neurons (descending neurons)
3. Map optic flow to photoreceptor spatial layout
4. Run connectome forward without any training
5. Decode flight behavior from motor neuron activity
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
print("PHASE 11: EVOLUTIONARY BEHAVIOR - USING CONNECTOME'S NATIVE CIRCUITS")
print("="*70)
print(f"\nDevice: {device}\n", flush=True)

# ============================================================================
# 1. LOAD CONNECTOME AND IDENTIFY KEY NEURON TYPES
# ============================================================================

print("[1/5] Loading connectome and identifying neuron types...", flush=True)

neurons_df = pd.read_csv('fly_neurons_real.csv')
n_neurons = len(neurons_df)

# Find sensory neurons
photoreceptors = neurons_df[neurons_df['primary_type'].str.contains('R1|R2|R3|R4|R5|R6|R7|R8', na=False)]
print(f"  Photoreceptors (R1-R8): {len(photoreceptors)}")

# Motion detecting neurons
motion_neurons = neurons_df[neurons_df['primary_type'].str.contains('T4|T5|Tm|Mi|Lo', na=False)]
print(f"  Motion neurons (T4/T5/Tm/Mi/Lo): {len(motion_neurons)}")

# Descending neurons (motor output)
descending = neurons_df[neurons_df['primary_type'].str.startswith('DNg', na=False) |
                        neurons_df['primary_type'].str.startswith('DN', na=False)]
if len(descending) == 0:
    # Alternative: look for neurons with specific connectivity patterns
    descending = neurons_df[neurons_df['primary_type'].str.contains('DN', na=False)]

print(f"  Descending neurons (motor): {len(descending)}")

# Get indices for quick lookup
pr_indices = set(neurons_df[neurons_df['primary_type'].str.contains('R1|R2|R3|R4|R5|R6|R7|R8', na=False)].index)
motion_indices = set(motion_neurons.index)
dn_indices = set(descending.index)

print(f"\n  Total neurons: {n_neurons:,}")
print(f"  Sensory pathway: {len(pr_indices)} PR -> {len(motion_indices)} motion -> {len(dn_indices)} DN")

# ============================================================================
# 2. LOAD CONNECTIVITY AND BUILD NETWORK
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
print(f"  Sparsity: {len(pre_idx) / (n_neurons * n_neurons) * 100:.3f}%")

# ============================================================================
# 3. BRAIN CONTROLLER USING NATIVE CIRCUITS
# ============================================================================

print("\n[3/5] Building native circuit controller...", flush=True)

class EvolutionaryBrainController:
    """
    Fly brain using native sensory and motor pathways.

    Uses published biophysical parameters from:
    Shiu et al., Nature (2024): "A Drosophila computational brain model reveals sensorimotor processing"

    LIF Parameters:
    - Resting potential: -52 mV
    - Threshold: -45 mV
    - Membrane resistance: 10 kΩ·cm²
    - Membrane capacitance: 2 µF·cm²
    - Synaptic decay: 5 ms
    - Synaptic weight: 0.275 mV per synapse
    - Refractory period: 2.2 ms
    """

    def __init__(self, connectivity, neurons_df, pr_indices, dn_indices, n_neurons, device):
        self.connectivity = connectivity
        self.neurons_df = neurons_df
        self.pr_indices = torch.tensor(sorted(list(pr_indices)), dtype=torch.long, device=device)
        self.dn_indices = torch.tensor(sorted(list(dn_indices)), dtype=torch.long, device=device)
        self.n_neurons = n_neurons
        self.device = device

        # Published LIF parameters (Shiu et al. 2024)
        self.V_rest = -0.052  # -52 mV
        self.V_thresh = -0.045  # -45 mV
        self.R_m = 10.0  # kΩ·cm²
        self.C_m = 2.0e-6  # 2 µF·cm² -> F/cm²
        self.tau_syn = 5e-3  # 5 ms synaptic decay
        self.W_syn = 0.275e-3  # 0.275 mV per synapse
        self.tau_ref = 2.2e-3  # 2.2 ms refractory period
        self.dt = 1e-3  # 1 ms timestep

        # Photoreceptor field mapping
        # R1-R6: outer photoreceptors (broadband, blue-green ~478nm, motion detection)
        # R7-R8: inner photoreceptors (color-sensitive)
        self.r16_neurons = sorted(list(neurons_df[neurons_df['primary_type'].str.contains('R1|R2|R3|R4|R5|R6', na=False)].index))
        self.r78_neurons = sorted(list(neurons_df[neurons_df['primary_type'].str.contains('R7|R8', na=False)].index))

        # Lateral (L) neurons: feed-forward motion processing
        self.l_neurons = sorted(list(neurons_df[neurons_df['primary_type'].str.startswith('L', na=False)].index))

        # Inhibitory neuron flag
        self.is_inhibitory = torch.from_numpy(neurons_df['nt_type'].isin(['GABA']).values).to(device)

        # Reset state
        self.reset_state()

    def reset_state(self):
        # Initialize at resting potential
        self.v = torch.ones(self.n_neurons, dtype=torch.float32, device=self.device) * self.V_rest
        self.spikes = torch.zeros(self.n_neurons, dtype=torch.bool, device=self.device)
        self.in_refractory = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        self.i_syn = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        self.spike_history = []

    def compute(self, optic_flow):
        """
        One timestep of neural computation using published LIF model.

        optic_flow: [forward, left, right, vertical]
        Maps to bilateral photoreceptor arrays.
        """

        optic_flow_t = torch.from_numpy(optic_flow).float().to(self.device)

        # External input current (from photoreceptors)
        i_input = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)

        # BILATERAL MAPPING: Photoreceptors organized left/right
        # Drosophila has ~800 ommatidia organized in retinotopic columns
        # Each eye ~400 columns; organized by spatial position

        n_r16 = len(self.r16_neurons)
        if n_r16 > 0:
            # Split R1-R6 into left and right visual fields
            r16_left = self.r16_neurons[:n_r16//2]
            r16_right = self.r16_neurons[n_r16//2:]

            # Left eye responds to leftward optic flow (yaw right = left image motion)
            # Right eye responds to rightward optic flow (yaw left = right image motion)
            for idx in r16_left:
                i_input[idx] += optic_flow_t[1] * 1e-10  # Left motion -> left PR

            for idx in r16_right:
                i_input[idx] += optic_flow_t[2] * 1e-10  # Right motion -> right PR

            # Forward/backward motion visible to all R1-R6
            for idx in self.r16_neurons:
                i_input[idx] += optic_flow_t[0] * 5e-11  # Forward motion

        # R7/R8: color vision, also respond to vertical (altitude) motion
        if len(self.r78_neurons) > 0:
            for idx in self.r78_neurons:
                i_input[idx] += optic_flow_t[3] * 8e-11  # Vertical motion

        # Lateral (L) neurons provide local motion opponent signals
        if len(self.l_neurons) > 0:
            for idx in self.l_neurons:
                # L neurons integrate across neighboring columns
                i_input[idx] += (optic_flow_t[0] + optic_flow_t[3]) * 6e-11

        # Update refractory counter
        self.in_refractory = torch.clamp(self.in_refractory - self.dt, min=0.0)

        # Synaptic input from network (spike-driven)
        # Each spike from presynaptic neuron contributes W_syn
        spikes_float = self.spikes.float()
        n_exc = torch.sparse.mm(self.connectivity.t().coalesce(), spikes_float.unsqueeze(1)).squeeze()
        n_inh = torch.sparse.mm(self.connectivity.t().coalesce(),
                               (self.spikes & self.is_inhibitory).float().unsqueeze(1)).squeeze()

        # Synaptic currents: I_syn = W_syn * N_spikes / tau_syn
        i_exc_input = n_exc * self.W_syn / self.tau_syn
        i_inh_input = n_inh * self.W_syn / self.tau_syn

        # Exponential decay of synaptic current
        self.i_syn = self.i_syn * np.exp(-self.dt / self.tau_syn) + i_exc_input - i_inh_input

        # Total input: external + synaptic
        i_total = i_input + self.i_syn

        # Membrane equation: dV/dt = (V_rest - V + I*R_m) / tau_m
        # tau_m = R_m * C_m
        tau_m = self.R_m * self.C_m
        dv = (self.V_rest - self.v + i_total * self.R_m) / tau_m
        self.v = self.v + dv * self.dt

        # Clamp voltage to reasonable range
        self.v = torch.clamp(self.v, -0.1, 0.05)

        # Spiking: threshold crossing + not in refractory
        can_spike = (self.in_refractory <= 0.0)
        self.spikes = (self.v > self.V_thresh) & can_spike

        # Reset voltage and enter refractory period
        self.v[self.spikes] = self.V_rest
        self.in_refractory[self.spikes] = self.tau_ref

        # Decode motor output from descending neurons
        dn_activity = self.spikes[self.dn_indices].float()

        # Motor decoding: population code from DN activity
        # DNg02 regulate wingbeat amplitude (forward thrust)
        # Various DN pairs control turn and altitude
        motor = torch.zeros(3, device=self.device)
        if len(self.dn_indices) > 0:
            n_dn = len(self.dn_indices)

            # Forward control: sum of ~1/3 of DN neurons (flight-related)
            forward_dns = dn_activity[:n_dn//3].sum() / (n_dn//3 + 1e-6)

            # Turn control: bilateral comparison (left vs right DNs)
            turn_dns = (dn_activity[n_dn//3:2*n_dn//3].sum() -
                       dn_activity[2*n_dn//3:].sum()) / (n_dn//3 + 1e-6)

            # Climb/altitude: vertical flight neurons
            climb_dns = dn_activity[2*n_dn//3:].sum() / (n_dn//3 + 1e-6)

            # Convert DN firing to motor commands
            motor[0] = torch.tanh(forward_dns - 0.5)  # Forward
            motor[1] = torch.tanh(turn_dns * 0.5)  # Turn
            motor[2] = torch.tanh(climb_dns - 0.3)  # Climb

        motor_np = motor.detach().cpu().numpy()

        # Record activity
        self.spike_history.append(self.spikes.sum().item())

        return motor_np

# ============================================================================
# 4. RUN SIMULATION
# ============================================================================

print("\n[4/5] Running flight simulation with evolutionary circuits...", flush=True)
print("  Simulating 400 steps (20 seconds @ 20Hz)...\n", flush=True)

brain = EvolutionaryBrainController(connectivity, neurons_df, pr_indices, dn_indices, n_neurons, device)

# Simple flight environment
class FlightEnv:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.5
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.step_count = 0
        self.obstacles = [(5, 0, 0.5, 0.4), (10, 0, 0.5, 0.3)]
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
        self.vx = forward * 0.2
        self.vy = turn * 0.1
        self.vz = climb * 0.1

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
optic_flows = []

start_sim = time.time()
for step in range(400):
    obs = env.get_optic_flow()
    motor = brain.compute(obs)
    env.step(motor[0], motor[1], motor[2])

    motor_history['forward'].append(motor[0])
    motor_history['turn'].append(motor[1])
    motor_history['climb'].append(motor[2])
    spike_counts.append(brain.spike_history[-1])
    optic_flows.append(obs.copy())

    if (step + 1) % 50 == 0:
        print(f"  Step {step+1}: x={env.x:6.2f} z={env.z:6.2f} spikes={int(spike_counts[-1]):6.0f}", flush=True)

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
fig.suptitle('Evolutionary Behavior - Fruit Fly Brain Using Native Circuits', fontsize=16, fontweight='bold')

# 3D trajectory
ax = fig.add_subplot(2, 4, 1, projection='3d')
ax.plot(env.trajectory['x'], env.trajectory['y'], env.trajectory['z'], 'b-', linewidth=2)
for ox, oy, oz, r in env.obstacles:
    ax.scatter([ox], [oy], [oz], s=200, c='red', marker='o', alpha=0.5)
ax.set_xlabel('X (forward)')
ax.set_ylabel('Y (sideways)')
ax.set_zlabel('Z (altitude)')
ax.set_title('3D Flight Trajectory (No Training)')
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)

# Motor commands
ax = fig.add_subplot(2, 4, 2)
ax.plot(motor_history['forward'], label='Forward', linewidth=1, alpha=0.8)
ax.plot(motor_history['turn'], label='Turn', linewidth=1, alpha=0.8)
ax.plot(motor_history['climb'], label='Climb', linewidth=1, alpha=0.8)
ax.set_xlabel('Timestep')
ax.set_ylabel('Motor Command')
ax.set_title('Motor Output (Decoded from DNs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Optic flow input
ax = fig.add_subplot(2, 4, 3)
optic_array = np.array(optic_flows)
ax.plot(optic_array[:, 0], label='Forward', linewidth=1, alpha=0.8)
ax.plot(optic_array[:, 1], label='Left', linewidth=1, alpha=0.8)
ax.plot(optic_array[:, 3], label='Vertical', linewidth=1, alpha=0.8)
ax.set_xlabel('Timestep')
ax.set_ylabel('Optic Flow')
ax.set_title('Sensory Input (Optic Flow)')
ax.legend()
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
ax.set_title('Altitude Dynamics')
ax.legend()
ax.grid(True, alpha=0.3)

# Forward progress
ax = fig.add_subplot(2, 4, 6)
ax.plot(env.trajectory['x'], 'orange', linewidth=2)
ax.fill_between(range(len(env.trajectory['x'])), env.trajectory['x'], alpha=0.2, color='orange')
ax.set_xlabel('Timestep')
ax.set_ylabel('Forward Position')
ax.set_title('Forward Progress')
ax.grid(True, alpha=0.3)

# Lateral movement
ax = fig.add_subplot(2, 4, 7)
ax.plot(env.trajectory['y'], 'cyan', linewidth=2)
ax.fill_between(range(len(env.trajectory['y'])), env.trajectory['y'], alpha=0.2, color='cyan')
ax.set_xlabel('Timestep')
ax.set_ylabel('Lateral Position')
ax.set_title('Lateral Movement')
ax.grid(True, alpha=0.3)

# Spike distribution over time
ax = fig.add_subplot(2, 4, 8)
ax.hist(spike_counts, bins=50, color='purple', alpha=0.7)
ax.set_xlabel('Spikes/timestep')
ax.set_ylabel('Frequency')
ax.set_title(f'Spike Distribution\n(Mean: {np.mean(spike_counts):.0f}, Peak: {max(spike_counts):.0f})')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('phase11_evolutionary_behavior.png', dpi=100)
print("[OK] Saved: phase11_evolutionary_behavior.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[COMPLETE] PHASE 11 - EVOLUTIONARY BEHAVIOR")
print("="*70)
print(f"""
Approach: Use connectome's native sensory and motor pathways
  - No training of artificial weights
  - Photoreceptors receive optic flow directly
  - Descending neurons drive motor output
  - Let evolution's circuits solve the problem

Results:
  Duration: 400 timesteps (20 seconds @ 20Hz)
  Total spikes: {int(np.sum(spike_counts)):,}
  Mean firing: {np.mean(spike_counts):.0f} neurons/timestep

  Flight position: x={env.x:.2f}, y={env.y:.3f}, z={np.mean(env.trajectory['z']):.3f}
  Altitude stability: {abs(np.mean(env.trajectory['z']) - 0.5):.3f} error

Circuit Components Used:
  - {len(pr_indices)} Photoreceptors (R1-R8) -> visual input
  - {len(motion_indices)} Motion neurons (T4/T5/Tm/Mi/Lo) -> processing
  - {len(dn_indices)} Descending neurons -> motor output

Next: Compare with RL approach, analyze emergent behaviors
""")
print("="*70 + "\n")
