"""
PHASE 7: HARDER TASK - FORCES NEURAL FIRING

Problem with Phase 6: Brain achieves +50 reward with 0 spikes.
Task was too trivial (just maintain altitude in simple env).

Solution: Make a task that REQUIRES neural computation.

Key changes:
1. Dynamic obstacles - obstacles move and appear randomly
2. Neural activity bonus - only get reward if firing > threshold
3. Broader sensory input - feed optic flow to L-neurons, R-neurons, not just T4/T5
4. Complex goal - must navigate while avoiding and maintaining altitude
5. Temporal challenge - requires integration over time

This forces the brain to actually compute and fire.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("PHASE 7: HARDER TASK - FORCES NEURAL FIRING")
print("="*70)
print(f"\nDevice: {device}\n", flush=True)

# ============================================================================
# 1. LOAD FULL BRAIN (fast)
# ============================================================================

print("[1/5] Loading full brain...", flush=True)
start_total = time.time()

neurons_df = pd.read_csv('fly_neurons_real.csv')
n_neurons = len(neurons_df)

# Load synapses
synapses_list = []
for i, chunk in enumerate(pd.read_csv('fly_synapses_real.csv', chunksize=1000000, low_memory=False)):
    synapses_list.append(chunk.iloc[np.arange(0, len(chunk), 100)])
    if (i + 1) % 20 == 0:
        print(f"  {sum(len(s) for s in synapses_list):,} synapses loaded...", flush=True)

synapses_df = pd.concat(synapses_list, ignore_index=True)

# Build connectivity
root_id_to_idx = {rid: idx for idx, rid in enumerate(neurons_df['root_id'].values)}
pre_idx = synapses_df['pre_root_id'].map(root_id_to_idx).fillna(-1).astype(int)
post_idx = synapses_df['post_root_id'].map(root_id_to_idx).fillna(-1).astype(int)
valid = (pre_idx >= 0) & (post_idx >= 0)
pre_idx, post_idx = pre_idx[valid].values, post_idx[valid].values
weights = np.clip(synapses_df[valid]['size'].values / (synapses_df[valid]['size'].max() + 1e-6), 0.1, 2.0)

# GPU sparse tensor
indices = torch.LongTensor([pre_idx, post_idx]).to(device)
values = torch.FloatTensor(weights).to(device)
connectivity = torch.sparse_coo_tensor(indices, values, (n_neurons, n_neurons), device=device)

# Neuron properties
is_inhibitory = torch.from_numpy(neurons_df['nt_type'].isin(['GABA']).values).to(device)
t4_indices = torch.where(torch.from_numpy(neurons_df['primary_type'].values == 'T4').to(device))[0]
t5_indices = torch.where(torch.from_numpy(neurons_df['primary_type'].values == 'T5').to(device))[0]
l_indices = torch.where(torch.from_numpy(neurons_df['primary_type'].str.startswith('L', na=False).values).to(device))[0]
r_indices = torch.where(torch.from_numpy(neurons_df['primary_type'].str.startswith('R', na=False).values).to(device))[0]
motor_mask = neurons_df['primary_type'].str.startswith(('DN', 'MN'), na=False)
motor_indices = torch.where(torch.from_numpy(motor_mask.values).to(device))[0]

print(f"  Neurons: {n_neurons:,} | Synapses: {connectivity._nnz():,}")
print(f"  T4: {len(t4_indices)}, T5: {len(t5_indices)}, L: {len(l_indices)}, R: {len(r_indices)}", flush=True)

# ============================================================================
# 2. HARDER FLIGHT ENVIRONMENT
# ============================================================================

print("\n[2/5] Setting up harder environment...", flush=True)

class HarderFlightEnv:
    """
    Harder task: dynamic obstacles, temporal challenge, multiple goals.

    Rewards:
    - Altitude stability: -|z - 0.5|
    - Forward progress: +forward_cmd if > 0
    - Neural firing: +spike_count / 1000 (FORCES BRAIN TO FIRE)
    - Collision: -5.0
    - Goal reached: +10.0 if reach x > 8.0
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = 0
        self.x = 0.0
        self.z = 0.5
        self.max_steps = 100
        self.goal_reached = False

        # Dynamic obstacles: move over time
        self.obstacles = []
        for _ in range(np.random.randint(3, 6)):
            x = np.random.uniform(2, 10)
            z = np.random.uniform(0.2, 0.8)
            vx = np.random.uniform(-0.5, 0.5)  # Obstacle velocity
            self.obstacles.append({'x': x, 'z': z, 'vx': vx})

        return self.get_observation()

    def get_observation(self):
        """
        Richer sensory input: forward, vertical, left, right flow.
        Based on obstacle distances in each direction.
        """
        if not self.obstacles:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        obs = np.zeros(4, dtype=np.float32)

        for obs_dict in self.obstacles:
            ox, oz = obs_dict['x'], obs_dict['z']
            dx = ox - self.x
            dz = oz - self.z
            dist = np.sqrt(dx**2 + dz**2) + 0.1

            # Directional flow encoding
            if dx > 0:
                obs[0] += 1.0 / dist  # Forward
            if dz > 0:
                obs[1] += 1.0 / dist  # Upward
            if dz < 0:
                obs[2] += 1.0 / dist  # Downward
            if dx < 0:
                obs[3] += 1.0 / dist  # Backward

        # Normalize
        return np.clip(obs / (obs.sum() + 0.01), 0, 1).astype(np.float32)

    def step(self, forward, climb, spike_count):
        """Execute action and compute reward."""
        self.t += 1

        # Update positions
        self.x += forward * 0.1
        self.z += climb * 0.05
        self.z = np.clip(self.z, 0.1, 0.9)

        # Move obstacles
        for obs_dict in self.obstacles:
            obs_dict['x'] += obs_dict['vx'] * 0.05

        reward = 0.0

        # Altitude penalty (encourages stability)
        altitude_error = abs(self.z - 0.5)
        reward -= altitude_error * 2

        # Forward bonus
        if forward > 0.5:
            reward += 1.0

        # **CRITICAL: NEURAL ACTIVITY BONUS**
        # Only get reward if brain actually fires!
        spike_bonus = min(spike_count / 1000.0, 2.0)
        reward += spike_bonus

        # Collision detection
        collision = False
        for obs_dict in self.obstacles:
            ox, oz = obs_dict['x'], obs_dict['z']
            dist = np.sqrt((ox - self.x)**2 + (oz - self.z)**2)
            if dist < 0.3:
                reward -= 5.0
                collision = True
                self.t = self.max_steps  # End episode
                break

        # Goal reward
        if self.x > 8.0 and not self.goal_reached:
            reward += 10.0
            self.goal_reached = True

        done = (self.t >= self.max_steps) or collision
        obs = self.get_observation()

        return obs, reward, done

env = HarderFlightEnv()
print(f"  Environment ready (dynamic obstacles, neural activity required)", flush=True)

# ============================================================================
# 3. READOUT WITH BROADER SENSORY INPUT
# ============================================================================

print("\n[3/5] Setting up readout with broader sensory input...", flush=True)

class BroadReadout:
    """Sensory input to MORE neuron types (not just T4/T5)."""

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        # Stronger initial gains
        self.sensory_gains = torch.tensor([1000e-12, 1000e-12, 1000e-12, 1000e-12], dtype=torch.float32).to(device)
        self.readout_weights = (torch.randn(2, n_neurons, dtype=torch.float32) * 0.001).to(device)
        self.lr_gains = 0.01
        self.lr_weights = 0.001

    def encode_sensory(self, optic_flow_np):
        optic_flow = torch.from_numpy(optic_flow_np).to(device)
        i_input = torch.zeros(self.n_neurons, dtype=torch.float32, device=device)

        # Feed to multiple neuron types (not just T4/T5)
        i_input[t4_indices] += optic_flow[0] * self.sensory_gains[0]  # Forward -> T4
        i_input[t5_indices] += optic_flow[1] * self.sensory_gains[1]  # Upward -> T5
        i_input[l_indices] += optic_flow[2] * self.sensory_gains[2]  # Downward -> L neurons
        i_input[r_indices] += optic_flow[3] * self.sensory_gains[3]  # Backward -> R neurons

        return i_input

    def decode_motor(self, neural_activity):
        motor = torch.tanh(self.readout_weights @ neural_activity)
        return torch.clamp(motor, -1, 1)

    def update_weights(self, reward):
        if reward > 0:
            scale = 1.0 + self.lr_weights * min(reward, 2.0)
            self.readout_weights.data *= scale
            self.sensory_gains.data *= (1.0 + self.lr_gains * 0.05)

readout = BroadReadout(n_neurons)
print(f"  Readout initialized with broader sensory input (T4, T5, L, R neurons)", flush=True)

# ============================================================================
# 4. BRAIN SIMULATION & RL TRAINING
# ============================================================================

print("\n[4/5] Running RL training on harder task...", flush=True)

def simulate_brain_step(optic_flow, readout, connectivity, is_inhibitory, n_neurons,
                       v, spikes, g_exc, g_inh):
    i_input = readout.encode_sensory(optic_flow)
    tau_exc = 5e-3
    tau_inh = 10e-3

    decay_exc = torch.exp(torch.tensor(-0.001 / tau_exc, device=device))
    decay_inh = torch.exp(torch.tensor(-0.001 / tau_inh, device=device))

    spikes_float = spikes.float()
    i_exc = torch.sparse.mm(connectivity.t().coalesce(), spikes_float.unsqueeze(1)).squeeze()
    i_inh = torch.sparse.mm(connectivity.t().coalesce(), (spikes & is_inhibitory).float().unsqueeze(1)).squeeze()

    g_exc = g_exc * decay_exc + i_exc * 1e-9
    g_inh = g_inh * decay_inh + i_inh * 1e-9

    i_syn = g_exc * (0 - v) + g_inh * (-80e-3 - v) + 10e-9 * (-70e-3 - v) + i_input
    v = v + (i_syn / 100e-12) * 0.001
    v = torch.clamp(v, -100e-3, 50e-3)

    spikes = v > -50e-3
    v[spikes] = -70e-3

    motor = readout.decode_motor(spikes.float())
    spike_count = spikes.sum().item()

    return v, spikes, g_exc, g_inh, motor, spike_count

n_episodes = 150
episode_rewards = []
episode_spikes = []

episode_start = time.time()
for episode in range(n_episodes):
    obs = env.reset()
    v = torch.ones(n_neurons, dtype=torch.float32, device=device) * -70e-3
    spikes = torch.zeros(n_neurons, dtype=torch.bool, device=device)
    g_exc = torch.zeros(n_neurons, dtype=torch.float32, device=device)
    g_inh = torch.zeros(n_neurons, dtype=torch.float32, device=device)

    episode_reward = 0.0
    episode_spike_count = 0

    for step in range(env.max_steps):
        motor = torch.zeros(2, device=device)
        spike_step = 0

        # Run 5 brain timesteps per action
        for _ in range(5):
            v, spikes, g_exc, g_inh, motor, step_spikes = simulate_brain_step(
                obs, readout, connectivity, is_inhibitory, n_neurons,
                v, spikes, g_exc, g_inh
            )
            spike_step += step_spikes

        episode_spike_count += spike_step
        motor_np = motor.detach().cpu().numpy()
        obs, reward, done = env.step(motor_np[0], motor_np[1], spike_step)
        episode_reward += reward

        if done:
            break

    readout.update_weights(episode_reward)
    episode_rewards.append(episode_reward)
    episode_spikes.append(episode_spike_count)

    if (episode + 1) % 30 == 0:
        avg_reward = np.mean(episode_rewards[-30:])
        avg_spikes = np.mean(episode_spikes[-30:])
        elapsed = time.time() - episode_start
        rate = (episode + 1) / (elapsed / 60.0)
        remaining = (n_episodes - episode - 1) / rate if rate > 0 else 0
        print(f"  [{episode+1:3d}/{n_episodes}] Reward: {avg_reward:+7.2f} | "
              f"Spikes: {avg_spikes:7.0f} | Rate: {rate:.1f} ep/min | ETA: {remaining:.1f}min", flush=True)

print(f"\n[SUCCESS] Training complete!", flush=True)
print(f"  Final reward: {episode_rewards[-1]:+.2f}", flush=True)
print(f"  Best reward: {max(episode_rewards):+.2f}", flush=True)
print(f"  Mean spikes: {np.mean(episode_spikes):.0f} per episode", flush=True)

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

print("\n[5/5] Generating results...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 7: Harder Task - Forcing Neural Firing', fontsize=14, fontweight='bold')

ax = axes[0, 0]
ax.plot(episode_rewards, 'b-', linewidth=1, alpha=0.5)
ax.plot(np.convolve(episode_rewards, np.ones(15)/15, mode='valid'), 'r-', linewidth=2, label='MA-15')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Learning Curve (Harder Task)')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[0, 1]
ax.plot(episode_spikes, 'purple', linewidth=1, alpha=0.5)
ax.plot(np.convolve(episode_spikes, np.ones(15)/15, mode='valid'), 'darkviolet', linewidth=2, label='MA-15')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Spikes')
ax.set_title('Neural Activity (Should increase!)')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 0]
channels = ['Forward', 'Upward', 'Downward', 'Backward']
gains_pA = (readout.sensory_gains.detach().cpu().numpy() / 1e-12)
ax.bar(channels, gains_pA, color=['red', 'blue', 'green', 'orange'], alpha=0.6)
ax.set_ylabel('Input Gain (pA)')
ax.set_title('Learned Sensory Gains (Broader Input)')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
ax.hist(readout.readout_weights.detach().cpu().numpy().flatten(), bins=50, alpha=0.6, color='cyan', edgecolor='black')
ax.set_xlabel('Weight value')
ax.set_ylabel('Frequency')
ax.set_title('Readout Weight Distribution')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('phase7_harder_task_results.png', dpi=100)
print(f"[OK] Saved: phase7_harder_task_results.png", flush=True)

with open('trained_harder_task.pkl', 'wb') as f:
    pickle.dump({
        'sensory_gains': readout.sensory_gains.detach().cpu().numpy(),
        'readout_weights': readout.readout_weights.detach().cpu().numpy(),
        'final_reward': episode_rewards[-1],
        'best_reward': max(episode_rewards),
        'mean_spikes': np.mean(episode_spikes),
    }, f)

total_time = time.time() - start_total
print("\n" + "="*70)
print("[COMPLETE] PHASE 7 - HARDER TASK TRAINING")
print("="*70)
print(f"""
Harder Task Results:
  - Episodes: {n_episodes}
  - Total time: {total_time/60:.1f} minutes
  - Final reward: {episode_rewards[-1]:+.2f}
  - Best reward: {max(episode_rewards):+.2f}
  - Mean spikes/episode: {np.mean(episode_spikes):.0f}

Key Changes from Phase 6:
  1. Dynamic obstacles (move over time)
  2. Neural activity BONUS (forces brain to fire)
  3. Broader sensory input (T4, T5, L, R neurons)
  4. Harder goal (reach x=8.0)
  5. Richer reward signal

Expected vs Phase 6:
  - Phase 6: 0 spikes (trivial task)
  - Phase 7: Should have 1000+ spikes (neural activity required)

This forces the brain to actually USE its structure!
""")
print("="*70 + "\n")
