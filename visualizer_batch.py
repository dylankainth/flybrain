"""
BATCH BRAIN VISUALIZER

Records and saves visualization frames without needing an interactive display.
Creates an MP4 animation of the brain flying.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import pickle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("BATCH BRAIN VISUALIZER - SAVING ANIMATION")
print("="*70)
print(f"\nLoading brain and connectome...\n", flush=True)

# Load connectome
neurons_df = pd.read_csv('fly_neurons_real.csv')
n_neurons = len(neurons_df)

# Load and build connectivity
synapses_list = []
for i, chunk in enumerate(pd.read_csv('fly_synapses_real.csv', chunksize=1000000, low_memory=False)):
    synapses_list.append(chunk.iloc[np.arange(0, len(chunk), 100)])
    if (i + 1) % 20 == 0:
        print(f"  Loaded {sum(len(s) for s in synapses_list):,} synapses...", flush=True)

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
print(f"  Descending neurons: {len(dn_indices)}\n", flush=True)

# ============================================================================
# SIMPLIFIED BRAIN FOR BATCH MODE
# ============================================================================

class SimpleBrain:
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

        self.reset_state()

    def reset_state(self):
        self.v = torch.ones(n_neurons, dtype=torch.float32, device=device) * self.V_rest
        self.spikes = torch.zeros(n_neurons, dtype=torch.bool, device=device)
        self.in_refractory = torch.zeros(n_neurons, dtype=torch.float32, device=device)
        self.i_syn = torch.zeros(n_neurons, dtype=torch.float32, device=device)

    def compute(self, optic_flow):
        optic_flow_t = torch.from_numpy(optic_flow).float().to(device)
        i_input = torch.zeros(n_neurons, dtype=torch.float32, device=device)

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
        n_exc = torch.sparse.mm(connectivity.t().coalesce(), spikes_float.unsqueeze(1)).squeeze()
        n_inh = torch.sparse.mm(connectivity.t().coalesce(),
                               (self.spikes & is_inhibitory).float().unsqueeze(1)).squeeze()

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
# FLIGHT ENV
# ============================================================================

class FlightEnv:
    def __init__(self):
        self.x, self.y, self.z = 0.0, 0.0, 0.5
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
        self.x += forward * 0.25 * 0.05
        self.y += turn * 0.15 * 0.05
        self.z += climb * 0.15 * 0.05
        self.z = np.clip(self.z, 0.1, 0.9)
        self.trajectory['x'].append(self.x)
        self.trajectory['y'].append(self.y)
        self.trajectory['z'].append(self.z)

# ============================================================================
# RUN SIMULATION AND COLLECT DATA
# ============================================================================

print("Running 200-step simulation...\n", flush=True)

brain = SimpleBrain()
env = FlightEnv()

data = {
    'optic_flow': [],
    'motor': [],
    'spikes_pr': [],
    'spikes_motion': [],
    'spikes_dn': [],
    'total_spikes': [],
    'x': [],
    'y': [],
    'z': []
}

for step in range(200):
    obs = env.get_optic_flow()
    motor = brain.compute(obs)
    env.step(motor[0], motor[1], motor[2])

    data['optic_flow'].append(obs)
    data['motor'].append(motor)
    data['spikes_pr'].append(brain.spikes[pr_indices].sum().item() if pr_indices else 0)
    data['spikes_motion'].append(brain.spikes[motion_indices].sum().item() if motion_indices else 0)
    data['spikes_dn'].append(brain.spikes[dn_indices].sum().item() if dn_indices else 0)
    data['total_spikes'].append(brain.spikes.sum().item())
    data['x'].append(env.x)
    data['y'].append(env.y)
    data['z'].append(env.z)

    if (step + 1) % 50 == 0:
        print(f"  Step {step+1}: x={env.x:.2f} z={env.z:.2f} spikes={int(data['total_spikes'][-1])}", flush=True)

print(f"\n[OK] Simulation complete")
print(f"  Total steps: {len(data['optic_flow'])}")
print(f"  Mean spikes/step: {np.mean(data['total_spikes']):.0f}")

# ============================================================================
# CREATE MULTI-PANEL VISUALIZATION
# ============================================================================

print("\nCreating visualization...\n", flush=True)

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Fruit Fly Brain in Flight - Neural Activity Visualization', fontsize=16, fontweight='bold')

# Subplot layout
ax1 = plt.subplot(3, 3, 1)  # Optic flow
ax2 = plt.subplot(3, 3, 2)  # Circuit activity
ax3 = plt.subplot(3, 3, 3)  # Total spikes
ax4 = plt.subplot(3, 3, 4)  # Motor output
ax5 = plt.subplot(3, 3, 5)  # Position
ax6 = plt.subplot(3, 3, 6)  # Altitude
ax7 = plt.subplot(3, 3, 7)  # Spikes PR
ax8 = plt.subplot(3, 3, 8)  # Spikes Motion
ax9 = plt.subplot(3, 3, 9)  # 3D trajectory

# Helper to update all plots
def update_frame(frame, data, env_obstacles):
    window = min(frame + 1, 100)
    start = max(0, frame - window)

    # Optic flow
    ax1.clear()
    ax1.set_title('Sensory Input (Optic Flow)', fontweight='bold')
    flow = np.array(data['optic_flow'][start:frame+1])
    ax1.plot(flow[:, 0], label='Forward', linewidth=2)
    ax1.plot(flow[:, 1], label='Left', linewidth=2)
    ax1.plot(flow[:, 2], label='Right', linewidth=2)
    ax1.plot(flow[:, 3], label='Vertical', linewidth=2)
    ax1.legend(fontsize=8)
    ax1.set_ylabel('Magnitude')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Circuit activity
    ax2.clear()
    ax2.set_title('Circuit Activity', fontweight='bold')
    ax2.bar([0, 1, 2],
           [data['spikes_pr'][frame], data['spikes_motion'][frame], data['spikes_dn'][frame]],
           color=['red', 'green', 'blue'], alpha=0.7)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['PR', 'Motion', 'DN'], fontsize=9)
    ax2.set_ylabel('Spikes')
    ax2.set_ylim(0, 2000)
    ax2.grid(True, alpha=0.3, axis='y')

    # Total spikes
    ax3.clear()
    ax3.set_title('Total Neural Activity', fontweight='bold')
    spikes_hist = np.array(data['total_spikes'][start:frame+1])
    ax3.plot(spikes_hist, 'purple', linewidth=2)
    ax3.fill_between(range(len(spikes_hist)), spikes_hist, alpha=0.3, color='purple')
    ax3.set_ylabel('Active neurons')
    ax3.grid(True, alpha=0.3)

    # Motor output
    ax4.clear()
    ax4.set_title('Motor Commands', fontweight='bold')
    motor = np.array(data['motor'][start:frame+1])
    ax4.plot(motor[:, 0], label='Forward', linewidth=2)
    ax4.plot(motor[:, 1], label='Turn', linewidth=2)
    ax4.plot(motor[:, 2], label='Climb', linewidth=2)
    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax4.legend(fontsize=8)
    ax4.set_ylabel('Command')
    ax4.set_ylim(-1, 1)
    ax4.grid(True, alpha=0.3)

    # Position
    ax5.clear()
    ax5.set_title('Forward Progress', fontweight='bold')
    x_pos = np.array(data['x'][start:frame+1])
    ax5.plot(x_pos, 'orange', linewidth=2)
    ax5.fill_between(range(len(x_pos)), x_pos, alpha=0.3, color='orange')
    ax5.set_ylabel('X position')
    ax5.grid(True, alpha=0.3)

    # Altitude
    ax6.clear()
    ax6.set_title('Altitude Control', fontweight='bold')
    z_pos = np.array(data['z'][start:frame+1])
    ax6.plot(z_pos, 'green', linewidth=2, label='Altitude')
    ax6.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Target')
    ax6.fill_between(range(len(z_pos)), z_pos, alpha=0.3, color='green')
    ax6.legend(fontsize=8)
    ax6.set_ylabel('Z position')
    ax6.set_ylim(0.1, 0.9)
    ax6.grid(True, alpha=0.3)

    # PR spikes
    ax7.clear()
    ax7.set_title('Photoreceptor Firing', fontweight='bold')
    pr_spikes = np.array(data['spikes_pr'][start:frame+1])
    ax7.plot(pr_spikes, 'red', linewidth=2)
    ax7.fill_between(range(len(pr_spikes)), pr_spikes, alpha=0.3, color='red')
    ax7.set_ylabel('PR spikes')
    ax7.grid(True, alpha=0.3)

    # Motion spikes
    ax8.clear()
    ax8.set_title('Motion Circuit Firing', fontweight='bold')
    motion_spikes = np.array(data['spikes_motion'][start:frame+1])
    ax8.plot(motion_spikes, 'green', linewidth=2)
    ax8.fill_between(range(len(motion_spikes)), motion_spikes, alpha=0.3, color='green')
    ax8.set_ylabel('Motion spikes')
    ax8.grid(True, alpha=0.3)

    # 3D trajectory
    ax9.clear()
    ax9.set_title(f'Flight Path (Step {frame+1}/200)', fontweight='bold')
    ax9.plot(data['x'][:frame+1], data['y'][:frame+1], 'b-', linewidth=2, label='Flight path')
    ax9.scatter(data['x'][frame], data['y'][frame], s=100, c='green', marker='o', label='Current position')
    for ox, oy, oz, r in env_obstacles:
        ax9.scatter(ox, oy, s=200, c='red', marker='x', alpha=0.7, linewidths=2)
    ax9.set_xlabel('X (forward)')
    ax9.set_ylabel('Y (sideways)')
    ax9.set_xlim(-1, 16)
    ax9.set_ylim(-2, 2)
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    ax9.set_aspect('equal')

# Create animation
obstacles = [(5, 0, 0.5, 0.4), (10, 0, 0.5, 0.3), (15, 0, 0.5, 0.2)]
anim = animation.FuncAnimation(
    fig, update_frame, frames=len(data['optic_flow']),
    fargs=(data, obstacles), interval=50, repeat=True
)

# Save
output_file = 'visualizer_output.png'
print(f"Saving animation frames to: {output_file}")
anim.save('visualizer_animation.gif', writer='pillow', fps=10)
print(f"[OK] Saved: visualizer_animation.gif")

plt.savefig(output_file, dpi=100, bbox_inches='tight')
print(f"[OK] Saved: {output_file}\n")
print("Done!")
print("="*70)
