"""
Full Brain Simulator: 139,255 neurons, 80M+ synapses

Goal: Infrastructure to run complete FlyWire connectome
Status: Build the platform first, add functionality iteratively

This is the baseline. Later we'll add:
- Sensory input encoding
- Motor output decoding
- Learning/plasticity
- Validation

For now: Just prove the full brain can run.
"""

import numpy as np
import pandas as pd
import time
import pickle
from scipy.sparse import csr_matrix

print("="*70)
print("FULL BRAIN SIMULATOR - Infrastructure Build")
print("="*70)

# ============================================================================
# LOAD CONNECTOME DATA
# ============================================================================

print("\n[1/5] Loading connectome data...")
start_load = time.time()

neurons_df = pd.read_csv('fly_neurons_real.csv')
print(f"  Loaded {len(neurons_df):,} neurons")

# Load synapses in chunks to avoid memory spike
print(f"  Loading {80215790:,} synapses (chunked)...")
chunk_size = 5000000
synapse_chunks = []
total_synapses = 0

for i, chunk in enumerate(pd.read_csv('fly_synapses_real.csv', chunksize=chunk_size, low_memory=False)):
    synapse_chunks.append(chunk)
    total_synapses += len(chunk)
    if (i + 1) % 5 == 0:
        print(f"    Loaded {total_synapses:,} synapses...")

synapses_df = pd.concat(synapse_chunks, ignore_index=True)
print(f"  Total synapses loaded: {len(synapses_df):,}")

load_time = time.time() - start_load
print(f"  Load time: {load_time:.1f}s")

# ============================================================================
# BUILD NEURON INDEX & MAPPING
# ============================================================================

print("\n[2/5] Building neuron index...")
start_index = time.time()

# Create mapping: root_id -> neuron_index
root_id_to_idx = {rid: idx for idx, rid in enumerate(neurons_df['root_id'].values)}
idx_to_root_id = {idx: rid for rid, idx in root_id_to_idx.items()}

print(f"  Created bidirectional mapping for {len(root_id_to_idx):,} neurons")

# ============================================================================
# BUILD SPARSE CONNECTIVITY MATRIX
# ============================================================================

print("\n[3/5] Building sparse connectivity matrix...")

# Convert root_ids to indices
print("  Converting synapse root_ids to indices...")
pre_indices = synapses_df['pre_root_id'].map(root_id_to_idx)
post_indices = synapses_df['post_root_id'].map(root_id_to_idx)

# Filter out synapses with unmapped neurons (shouldn't happen)
valid_mask = (~pre_indices.isna()) & (~post_indices.isna())
pre_indices = pre_indices[valid_mask].astype(int).values
post_indices = post_indices[valid_mask].astype(int).values
synapse_weights = synapses_df[valid_mask]['size'].values if 'size' in synapses_df.columns else np.ones(len(pre_indices))

print(f"  Valid synapses: {len(pre_indices):,} / {len(synapses_df):,}")

# Normalize weights
synapse_weights = np.clip(synapse_weights / (synapse_weights.max() + 1e-6), 0.1, 2.0)

# Build sparse connectivity matrix
print("  Creating sparse CSR matrix...")
connectivity = csr_matrix(
    (synapse_weights, (pre_indices, post_indices)),
    shape=(len(neurons_df), len(neurons_df))
)

print(f"  Connectivity matrix: {connectivity.shape}")
print(f"  Sparsity: {100 * connectivity.nnz / (connectivity.shape[0] * connectivity.shape[1]):.4f}%")
print(f"  Memory: ~{connectivity.data.nbytes / 1e6:.1f} MB")

index_time = time.time() - start_index
print(f"  Index time: {index_time:.1f}s")

# ============================================================================
# INITIALIZE NEURON STATES
# ============================================================================

print("\n[4/5] Initializing neuron states...")
start_init = time.time()

n_neurons = len(neurons_df)

# Neuron states
v = np.ones(n_neurons) * -70e-3  # Voltage (resting)
spikes = np.zeros(n_neurons, dtype=bool)
spike_history = []
firing_rates = np.zeros(n_neurons)

# Synaptic conductances
g_excite = np.zeros(n_neurons)
g_inhibit = np.zeros(n_neurons)

# Parameters
tau_m = 20e-3  # Membrane time constant
tau_exc = 5e-3  # Excitation decay
tau_inh = 10e-3  # Inhibition decay
dt = 0.001  # 1 ms timestep

# Neuron type classification (from primary_type)
neuron_types = neurons_df['primary_type'].values
is_inhibitory = neurons_df['nt_type'].isin(['GABA']).values

print(f"  Neurons initialized:")
print(f"    Voltage: -70 mV (rest)")
print(f"    Inhibitory: {is_inhibitory.sum():,} ({100*is_inhibitory.sum()/n_neurons:.1f}%)")
print(f"    Excitatory: {(~is_inhibitory).sum():,} ({100*(~is_inhibitory).sum()/n_neurons:.1f}%)")

init_time = time.time() - start_init
print(f"  Init time: {init_time:.1f}s")

# ============================================================================
# RUN SIMULATION
# ============================================================================

print("\n[5/5] Running full brain simulation...")
print(f"  Duration: 10 seconds")
print(f"  Timestep: {dt*1000:.1f} ms")
print(f"  Total steps: {int(10 / dt):,}")

start_sim = time.time()

# Baseline input (tonic drive to keep network from being silent)
# In real system, this comes from sensory input
baseline_input = np.ones(n_neurons) * 100e-12  # 100 pA

timesteps = int(10 / dt)
sample_rate = 10  # Save every 10ms
spike_raster = []
membrane_trace = []
step_times = []

for step in range(timesteps):
    t = step * dt

    # 1. Receive sensory input (baseline + noise for now)
    i_input = baseline_input + np.random.randn(n_neurons) * 50e-12

    # 2. Synaptic input from other neurons
    i_exc = connectivity.T @ spikes.astype(float)  # Incoming spikes * weights
    i_inh = connectivity.T @ (spikes & is_inhibitory).astype(float)

    # 3. Voltage integration (simplified Hodgkin-Huxley)
    e_leak = -70e-3
    e_exc = 0e-3
    e_inh = -80e-3
    g_leak = 10e-9

    # Conductances decay exponentially
    decay_exc = np.exp(-dt / tau_exc)
    decay_inh = np.exp(-dt / tau_inh)
    g_excite = g_excite * decay_exc + i_exc * 1e-9
    g_inhibit = g_inhibit * decay_inh + i_inh * 1e-9

    # Current balance
    i_syn = (g_excite * (e_exc - v) + g_inhibit * (e_inh - v) +
             g_leak * (e_leak - v) + i_input)

    # Voltage update
    dv_dt = i_syn / 100e-12  # 100 pF capacitance
    v = v + dv_dt * dt
    v = np.clip(v, -100e-3, 50e-3)

    # 4. Spike detection
    threshold = -50e-3
    spikes = v > threshold
    v[spikes] = -70e-3  # Reset

    # 5. Track activity
    if step % sample_rate == 0:
        spike_raster.append(spikes.copy())
        membrane_trace.append(v.copy())
        step_times.append(t)

        # Print progress
        if (step + 1) % (timesteps // 10) == 0:
            spike_count = spikes.sum()
            firing_rate = spike_count / n_neurons * 1000 / dt
            print(f"  [{t:5.2f}s] Spikes: {spike_count:6,} ({firing_rate:6.1f} Hz avg)")

sim_time = time.time() - start_sim
print(f"  Simulation time: {sim_time:.1f}s (for 10s brain time)")
print(f"  Speedup: {10 / sim_time:.1f}x real-time")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "="*70)
print("FULL BRAIN SIMULATION RESULTS")
print("="*70)

spike_raster = np.array(spike_raster)
membrane_trace = np.array(membrane_trace)

print(f"\nActivity statistics:")
total_spikes = spike_raster.sum()
print(f"  Total spikes: {total_spikes:,}")
print(f"  Average firing rate: {total_spikes / n_neurons / 10 * 1000:.1f} Hz")
print(f"  Max firing rate (any neuron): {spike_raster.sum(axis=0).max() / 10 * 1000:.1f} Hz")

# Identify most active neurons
active_neurons = spike_raster.sum(axis=0)
top_indices = np.argsort(active_neurons)[-20:]
print(f"\nTop 20 most active neurons:")
for rank, idx in enumerate(reversed(top_indices), 1):
    rate = active_neurons[idx] / 10 * 1000
    ntype = neurons_df.iloc[idx]['primary_type']
    print(f"  {rank:2d}. Neuron {idx:6d} ({ntype:10s}): {rate:6.1f} Hz")

# Memory usage
import sys
print(f"\nMemory usage:")
print(f"  Neurons state: {v.nbytes / 1e6:.1f} MB")
print(f"  Synaptic state: {(g_excite.nbytes + g_inhibit.nbytes) / 1e6:.1f} MB")
print(f"  Connectivity matrix: {connectivity.data.nbytes / 1e6:.1f} MB")
print(f"  Spike raster (10s): {spike_raster.nbytes / 1e6:.1f} MB")
print(f"  Total: ~{(v.nbytes + g_excite.nbytes + connectivity.data.nbytes + spike_raster.nbytes) / 1e6:.1f} MB")

# ============================================================================
# SAVE FULL BRAIN STATE FOR LATER USE
# ============================================================================

print("\n" + "="*70)
print("SAVING FULL BRAIN SIMULATOR STATE")
print("="*70)

state = {
    'neurons_df': neurons_df,
    'connectivity': connectivity,
    'root_id_to_idx': root_id_to_idx,
    'idx_to_root_id': idx_to_root_id,
    'spike_raster': spike_raster,
    'membrane_trace': membrane_trace,
    'parameters': {
        'n_neurons': n_neurons,
        'n_synapses': len(pre_indices),
        'dt': dt,
        'simulation_time': 10.0,
        'tau_m': tau_m,
        'tau_exc': tau_exc,
        'tau_inh': tau_inh,
    },
    'metadata': {
        'load_time': load_time,
        'index_time': index_time,
        'init_time': init_time,
        'sim_time': sim_time,
        'total_time': load_time + index_time + init_time + sim_time,
    }
}

# Save as pickle (for later loading)
with open('full_brain_state.pkl', 'wb') as f:
    pickle.dump(state, f)
print(f"Saved: full_brain_state.pkl")

# Save summary as CSV for analysis
summary_df = pd.DataFrame({
    'neuron_id': range(n_neurons),
    'root_id': neurons_df['root_id'].values,
    'primary_type': neurons_df['primary_type'].values,
    'spike_count': spike_raster.sum(axis=0),
    'firing_rate_hz': spike_raster.sum(axis=0) / 10 * 1000,
    'mean_voltage': membrane_trace.mean(axis=0),
    'max_voltage': membrane_trace.max(axis=0),
})

summary_df.to_csv('full_brain_activity_summary.csv', index=False)
print(f"Saved: full_brain_activity_summary.csv")

# ============================================================================
# FINAL STATUS
# ============================================================================

print("\n" + "="*70)
print("[SUCCESS] FULL BRAIN SIMULATOR OPERATIONAL")
print("="*70)

print(f"""
Simulation completed successfully!

Brain specifications:
  Neurons: {n_neurons:,}
  Synapses: {len(pre_indices):,}
  Connectivity: {100 * len(pre_indices) / (n_neurons**2):.3f}%

Performance:
  Wall-clock time: {sim_time:.1f}s (for 10s simulation)
  Speedup: {10/sim_time:.1f}x real-time
  Memory footprint: {(connectivity.data.nbytes + v.nbytes) / 1e6:.0f} MB

Output files:
  - full_brain_state.pkl (complete state for reloading)
  - full_brain_activity_summary.csv (neuron activity analysis)

Next steps:
  1. Load full_brain_state.pkl in future sessions
  2. Add sensory input encoding (vision -> neurons)
  3. Add motor output decoding (neurons -> thrust)
  4. Implement learning/plasticity
  5. Train with RL on flight task
  6. Validate against real fly behavior

The full brain is now available as a platform for:
  - Circuit analysis
  - Learning system development
  - Behavioral simulation
  - Robotics control
""")

print(f"Total pipeline time: {state['metadata']['total_time']:.1f}s")
print(f"Ready for next phase of development.")
print(f"{'='*70}\n")
