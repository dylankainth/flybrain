"""
REAL-TIME BRAIN VISUALIZER

Watch the fruit fly brain compute in real-time:
- Neural activity heatmap (which neurons fire)
- Circuit activity (photoreceptors → motion → descending neurons)
- Sensory input (optic flow)
- Motor output (flight commands)
- Flight trajectory (3D position)

All synchronized as the simulation runs.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("REAL-TIME BRAIN VISUALIZER")
print("="*70)
print(f"\nLoading brain and connectome...\n", flush=True)

# Load connectome
neurons_df = pd.read_csv('fly_neurons_real.csv')
n_neurons = len(neurons_df)

# Load and build connectivity
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

# Key neuron indices
photoreceptors = neurons_df[neurons_df['primary_type'].str.contains('R1|R2|R3|R4|R5|R6|R7|R8', na=False)]
motion_neurons = neurons_df[neurons_df['primary_type'].str.contains('T4|T5|Tm|Mi|Lo', na=False)]
descending = neurons_df[neurons_df['primary_type'].str.contains('DN', na=False)]

pr_indices = sorted(list(photoreceptors.index))
motion_indices = sorted(list(motion_neurons.index))
dn_indices = sorted(list(descending.index))

print(f"Neurons: {n_neurons:,}")
print(f"  Photoreceptors: {len(pr_indices)}")
print(f"  Motion circuits: {len(motion_indices)}")
print(f"  Descending neurons: {len(dn_indices)}")

# ============================================================================
# BRAIN CONTROLLER (same as Phase 11C)
# ============================================================================

class VisualizerBrain:
    def __init__(self):
        self.V_rest = -0.052
        self.V_thresh = -0.045
        self.R_m = 10.0
        self.C_m = 2.0e-6
        self.tau_syn = 5e-3
        self.W_syn = 0.275e-3
        self.tau_ref = 2.2e-3
        self.dt = 1e-3

        self.r16_neurons = sorted(list(neurons_df[neurons_df['primary_type'].str.contains('R1|R2|R3|R4|R5|R6', na=False)].index))
        self.r78_neurons = sorted(list(neurons_df[neurons_df['primary_type'].str.contains('R7|R8', na=False)].index))
        self.r16_left = self.r16_neurons[:len(self.r16_neurons)//2]
        self.r16_right = self.r16_neurons[len(self.r16_neurons)//2:]
        self.is_inhibitory = torch.from_numpy(neurons_df['nt_type'].isin(['GABA']).values).to(device)

        self.reset_state()

    def reset_state(self):
        self.v = torch.ones(n_neurons, dtype=torch.float32, device=device) * self.V_rest
        self.spikes = torch.zeros(n_neurons, dtype=torch.bool, device=device)
        self.in_refractory = torch.zeros(n_neurons, dtype=torch.float32, device=device)
        self.i_syn = torch.zeros(n_neurons, dtype=torch.float32, device=device)

    def compute(self, optic_flow):
        optic_flow_t = torch.from_numpy(optic_flow).float().to(device)
        i_input = torch.zeros(n_neurons, dtype=torch.float32, device=device)

        # Bilateral mapping
        for idx in self.r16_neurons:
            i_input[idx] += optic_flow_t[0] * 8e-11
        for idx in self.r16_left:
            i_input[idx] += optic_flow_t[1] * 1.2e-10
        for idx in self.r16_right:
            i_input[idx] += optic_flow_t[2] * 1.2e-10
        for idx in self.r78_neurons:
            i_input[idx] += optic_flow_t[3] * 1e-10

        self.in_refractory = torch.clamp(self.in_refractory - self.dt, min=0.0)

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

        can_spike = (self.in_refractory <= 0.0)
        self.spikes = (self.v > self.V_thresh) & can_spike
        self.v[self.spikes] = self.V_rest
        self.in_refractory[self.spikes] = self.tau_ref

        dn_activity = self.spikes[dn_indices].float() if len(dn_indices) > 0 else torch.zeros(1, device=device)
        if len(dn_indices) > 0:
            n_dn = len(dn_indices)
            forward_drive = dn_activity.sum() / (n_dn / 100.0)
            motion_spikes = self.spikes[motion_indices].float()
            n_motion = len(motion_indices)
            left_motion = motion_spikes[:n_motion//3].sum() / (n_motion//3 + 1e-6) if n_motion > 10 else 0
            right_motion = motion_spikes[n_motion//3:].sum() / ((2*n_motion)//3 + 1e-6) if n_motion > 10 else 0
            turn_signal = (left_motion - right_motion) * 0.1
            climb_drive = dn_activity.sum() / (n_dn / 50.0) - 1.0

            motor = torch.zeros(3, device=device)
            motor[0] = torch.tanh(forward_drive / 100.0)
            motor[1] = torch.tanh(turn_signal)
            motor[2] = torch.tanh(climb_drive / 50.0)
        else:
            motor = torch.zeros(3, device=device)

        return motor.detach().cpu().numpy()

# ============================================================================
# FLIGHT ENVIRONMENT
# ============================================================================

class FlightEnv:
    def __init__(self):
        self.x, self.y, self.z = 0.0, 0.0, 0.5
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.obstacles = [(5, 0, 0.5, 0.4), (10, 0, 0.5, 0.3), (15, 0, 0.5, 0.2)]
        self.trajectory = {'x': [], 'y': [], 'z': []}

    def get_optic_flow(self):
        obs = np.zeros(4, dtype=np.float32)
        for ox, oy, oz, r in self.obstacles:
            dist = max(np.sqrt((ox - self.x)**2 + (oy - self.y)**2 + (oz - self.z)**2), 0.1)
            if ox > self.x: obs[0] += 1.0 / dist
            if oy > self.y: obs[1] += 1.0 / dist
            if oy < self.y: obs[2] += 1.0 / dist
            if oz > self.z: obs[3] += 1.0 / dist
        return np.clip(obs / (obs.sum() + 0.01), 0, 1).astype(np.float32)

    def step(self, forward, turn, climb):
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

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nInitializing visualization...\n", flush=True)

brain = VisualizerBrain()
env = FlightEnv()

# Create figure with subplots
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Real-Time Fruit Fly Brain Visualization', fontsize=18, fontweight='bold')

# Neuron activity heatmap
ax_neurons = fig.add_subplot(3, 3, 1)
ax_neurons.set_title('Neural Activity Heatmap\n(all neurons, color=firing rate)', fontsize=11, fontweight='bold')
im_neurons = ax_neurons.imshow(np.zeros((20, 7000)), cmap='hot', aspect='auto', vmin=0, vmax=1)
ax_neurons.set_xlabel('Neuron index')
ax_neurons.set_ylabel('Neuron activity bins')
ax_neurons.set_yticks([])

# Circuit activity (stacked bars)
ax_circuit = fig.add_subplot(3, 3, 2)
ax_circuit.set_title('Circuit Activity\n(spikes/frame)', fontsize=11, fontweight='bold')
bars = ax_circuit.bar(['Photoreceptors', 'Motion Circuits', 'Descending Neurons'], [0, 0, 0], color=['red', 'green', 'blue'], alpha=0.7)
ax_circuit.set_ylim(0, 2000)
ax_circuit.set_ylabel('Active neurons')

# Optic flow input
ax_input = fig.add_subplot(3, 3, 3)
ax_input.set_title('Sensory Input\n(optic flow)', fontsize=11, fontweight='bold')
colors_input = ['red', 'blue', 'orange', 'purple']
lines_input = [ax_input.plot([], [], label=['Forward', 'Left', 'Right', 'Vertical'][i], color=colors_input[i], linewidth=2)[0] for i in range(4)]
ax_input.set_xlim(0, 100)
ax_input.set_ylim(0, 1)
ax_input.set_ylabel('Optic flow magnitude')
ax_input.legend(loc='upper right')

# Motor output
ax_motor = fig.add_subplot(3, 3, 4)
ax_motor.set_title('Motor Output', fontsize=11, fontweight='bold')
colors_motor = ['green', 'orange', 'purple']
lines_motor = [ax_motor.plot([], [], label=['Forward', 'Turn', 'Climb'][i], color=colors_motor[i], linewidth=2)[0] for i in range(3)]
ax_motor.set_xlim(0, 100)
ax_motor.set_ylim(-1, 1)
ax_motor.axhline(0, color='k', linestyle='--', alpha=0.3)
ax_motor.set_ylabel('Command magnitude')
ax_motor.legend(loc='upper right')

# Voltage trace (sample neurons)
ax_voltage = fig.add_subplot(3, 3, 5)
ax_voltage.set_title('Sample Neuron Voltages', fontsize=11, fontweight='bold')
lines_voltage = [ax_voltage.plot([], [], label=[f'PR {i}' for i in range(3)][i], alpha=0.7)[0] for i in range(3)]
ax_voltage.set_xlim(0, 100)
ax_voltage.set_ylim(-0.07, -0.04)
ax_voltage.set_ylabel('Voltage (V)')
ax_voltage.legend(loc='lower right', fontsize=8)

# Spike raster (sample neurons)
ax_raster = fig.add_subplot(3, 3, 6)
ax_raster.set_title('Spike Raster\n(100 sample neurons)', fontsize=11, fontweight='bold')
ax_raster.set_xlim(0, 100)
ax_raster.set_ylim(0, 100)
ax_raster.set_xlabel('Time (frames)')
ax_raster.set_ylabel('Neuron #')

# 3D trajectory
ax_3d = fig.add_subplot(3, 3, (7, 9), projection='3d')
ax_3d.set_title('Flight Trajectory', fontsize=11, fontweight='bold')
ax_3d.set_xlabel('X (forward)')
ax_3d.set_ylabel('Y (sideways)')
ax_3d.set_zlabel('Z (altitude)')
ax_3d.set_xlim(-1, 16)
ax_3d.set_ylim(-2, 2)
ax_3d.set_zlim(0, 1)
line_traj, = ax_3d.plot([], [], [], 'b-', linewidth=2, label='Flight path')
scatter_obs = ax_3d.scatter([], [], [], s=200, c='red', marker='o', alpha=0.5, label='Obstacles')
ax_3d.legend()

# Data storage for animation
sim_data = {
    'optic_flow': [],
    'motor': [],
    'spikes_pr': [],
    'spikes_motion': [],
    'spikes_dn': [],
    'neuron_activity': [],
    'sample_voltages': [[] for _ in range(3)],
    'spike_raster': [],
    'step': 0
}

sample_neuron_indices = [pr_indices[0] if pr_indices else 0,
                        pr_indices[1] if len(pr_indices) > 1 else 1,
                        pr_indices[2] if len(pr_indices) > 2 else 2]

print("Running simulation with visualization...")
print("Close the plot window to stop.\n", flush=True)

def animate(frame):
    if frame % 20 == 0:
        print(f"Frame {frame}: x={env.x:.2f} z={env.z:.2f}", flush=True)

    # Simulate
    obs = env.get_optic_flow()
    motor = brain.compute(obs)
    env.step(motor[0], motor[1], motor[2])

    # Collect data
    sim_data['optic_flow'].append(obs)
    sim_data['motor'].append(motor)
    sim_data['spikes_pr'].append(brain.spikes[pr_indices].sum().item() if pr_indices else 0)
    sim_data['spikes_motion'].append(brain.spikes[motion_indices].sum().item() if motion_indices else 0)
    sim_data['spikes_dn'].append(brain.spikes[dn_indices].sum().item() if dn_indices else 0)

    # Neural activity for heatmap
    neuron_activity = brain.spikes.cpu().numpy().astype(float)
    sim_data['neuron_activity'].append(neuron_activity)

    # Sample voltages
    for i, nidx in enumerate(sample_neuron_indices):
        sim_data['sample_voltages'][i].append(brain.v[nidx].item())

    # Spike raster
    sim_data['spike_raster'].append(brain.spikes.cpu().numpy())

    # Update plots
    t = len(sim_data['optic_flow'])
    window = min(100, t)

    # Neuron heatmap
    if t > 0:
        heatmap_data = np.array(sim_data['neuron_activity'][-window:]).T
        binned = np.mean(heatmap_data.reshape(n_neurons, -1, 20), axis=2) if window >= 20 else heatmap_data
        rebinned = np.repeat(binned, max(1, 20 // (window if window > 0 else 1)), axis=1)[:, :window]
        im_neurons.set_data(rebinned[:, -window:])

    # Circuit activity
    if t > 0:
        for i, bar in enumerate(bars):
            vals = [sim_data['spikes_pr'][-1], sim_data['spikes_motion'][-1], sim_data['spikes_dn'][-1]]
            bar.set_height(vals[i])

    # Optic flow
    if len(sim_data['optic_flow']) > 0:
        flow = np.array(sim_data['optic_flow'][-window:])
        for i in range(4):
            lines_input[i].set_data(range(len(flow)), flow[:, i])

    # Motor output
    if len(sim_data['motor']) > 0:
        motor_data = np.array(sim_data['motor'][-window:])
        for i in range(3):
            lines_motor[i].set_data(range(len(motor_data)), motor_data[:, i])

    # Voltage traces
    for i, voltage_list in enumerate(sim_data['sample_voltages']):
        if voltage_list:
            lines_voltage[i].set_data(range(len(voltage_list[-window:])), voltage_list[-window:])

    # Spike raster
    ax_raster.clear()
    ax_raster.set_title('Spike Raster (100 sample neurons)', fontsize=11, fontweight='bold')
    ax_raster.set_xlim(0, window)
    ax_raster.set_ylim(0, 100)
    if len(sim_data['spike_raster']) > 0:
        raster = np.array(sim_data['spike_raster'][-window:]).T
        sample_raster = raster[np.random.choice(n_neurons, 100, replace=False), :]
        for i, neuron_spikes in enumerate(sample_raster):
            spike_times = np.where(neuron_spikes)[0]
            ax_raster.scatter(spike_times, [i]*len(spike_times), s=5, c='blue', alpha=0.8)
    ax_raster.set_xlabel('Time (frames)')
    ax_raster.set_ylabel('Neuron #')

    # 3D trajectory
    ax_3d.clear()
    ax_3d.set_title(f'Flight Trajectory (step {t})', fontsize=11, fontweight='bold')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_xlim(-1, 16)
    ax_3d.set_ylim(-2, 2)
    ax_3d.set_zlim(0, 1)
    ax_3d.plot(env.trajectory['x'], env.trajectory['y'], env.trajectory['z'], 'b-', linewidth=2, label='Path')
    for ox, oy, oz, r in env.obstacles:
        ax_3d.scatter([ox], [oy], [oz], s=200, c='red', marker='o', alpha=0.5)
    ax_3d.scatter([env.x], [env.y], [env.z], s=100, c='green', marker='o', label='Current')
    ax_3d.legend(fontsize=8)

    return [im_neurons, ax_circuit, ax_input, ax_motor, ax_raster, ax_3d]

# Run animation
anim = FuncAnimation(fig, animate, frames=400, interval=50, blit=False, repeat=False)
plt.tight_layout()
plt.show()

print("\n[COMPLETE] Visualization finished")
