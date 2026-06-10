"""
Phase 1: Extract T4/T5 Direction-Selective Circuit from FlyWire FAFB v783

Goal: Extract the core visual motion detection circuit (T4/T5 neurons and their
immediate upstream/downstream connections) to build a proper motion detection model.

Key references:
- Klapoetke et al. 2017 (T4/T5 direction selectivity)
- Takemura et al. 2015 (medulla circuitry)
- Shinomiya et al. 2019 (lamina circuits)
- Joesch et al. 2010 (optic flow processing)
"""

import pandas as pd
import numpy as np
from collections import defaultdict

print("="*70)
print("PHASE 1: EXTRACT T4/T5 DIRECTION-SELECTIVE CIRCUIT")
print("="*70)

# Load FlyWire data
print("\n[1/4] Loading FlyWire neurons...")
neurons_df = pd.read_csv('fly_neurons_real.csv')
print(f"      Total neurons: {len(neurons_df):,}")

# Identify all T4 and T5 neurons
print("\n[2/4] Identifying T4/T5 neurons...")
t4_neurons = neurons_df[neurons_df['primary_type'].str.contains('T4', na=False)]
t5_neurons = neurons_df[neurons_df['primary_type'].str.contains('T5', na=False)]

print(f"      T4 neurons (motion up): {len(t4_neurons):,}")
print(f"      T5 neurons (motion down): {len(t5_neurons):,}")
print(f"      Total direction-selective: {len(t4_neurons) + len(t5_neurons):,}")

# List the T4/T5 subtypes
t4_types = t4_neurons['primary_type'].unique()
t5_types = t5_neurons['primary_type'].unique()
print(f"\n      T4 subtypes: {sorted(t4_types)}")
print(f"      T5 subtypes: {sorted(t5_types)}")

# Create mapping of root_id -> neuron index
root_id_to_idx = {rid: idx for idx, rid in enumerate(neurons_df['root_id'].values)}
t4_indices = [root_id_to_idx[rid] for rid in t4_neurons['root_id'].values]
t5_indices = [root_id_to_idx[rid] for rid in t5_neurons['root_id'].values]
visual_circuit_indices = set(t4_indices + t5_indices)

# Identify upstream neurons (medulla: L, M, C neurons)
print("\n[3/4] Identifying upstream visual circuit neurons...")
upstream_pattern = [
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6',      # Lamina neurons
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6',      # Medulla neurons
    'C1', 'C2', 'C3', 'C4',                  # Columnar neurons
    'T1', 'T2', 'T3'                         # T1-T3 (upstream of T4/T5)
]

upstream_neurons = neurons_df[
    neurons_df['primary_type'].isin(upstream_pattern)
]

print(f"      Upstream neurons (L, M, C, T1-T3): {len(upstream_neurons):,}")
print(f"      Types: {sorted(upstream_neurons['primary_type'].unique())}")

upstream_indices = set(root_id_to_idx[rid] for rid in upstream_neurons['root_id'].values)

# Identify downstream neurons (descending neurons, central complex)
print("\n[4/4] Identifying downstream motor/integration neurons...")
downstream_pattern = [
    'FD', 'FS',          # Flicker-detection neurons
    'H',                 # Horizontal neurons
    'VSN',               # Vertical shift neurons
    'LPLC',              # Lobular plate local circuits
]

# Also include some known descending neuron types
motor_pattern = ['DN', 'MN']

downstream_neurons = neurons_df[
    (neurons_df['primary_type'].isin(downstream_pattern)) |
    (neurons_df['primary_type'].str.startswith(tuple(motor_pattern), na=False))
]

print(f"      Downstream neurons: {len(downstream_neurons):,}")

downstream_indices = set(root_id_to_idx[rid] for rid in downstream_neurons['root_id'].values)

# Combined circuit definition
visual_circuit = {
    'upstream': upstream_indices,
    't4_t5': visual_circuit_indices,
    'downstream': downstream_indices,
}

# Now analyze synapses within and connecting these groups
print("\n" + "="*70)
print("LOADING AND ANALYZING SYNAPSES")
print("="*70)

print("\nLoading synapses (this may take a moment)...")
synapses_df = pd.read_csv('fly_synapses_real.csv', low_memory=False)
print(f"Total synapses in connectome: {len(synapses_df):,}")

# Convert root_ids to neuron indices for synapses
# Create mapping for synapses
print("\nMapping synapse root_ids to neuron indices...")

# Pre-compute the sets for faster lookup
all_t4_root_ids = set(t4_neurons['root_id'].values)
all_t5_root_ids = set(t5_neurons['root_id'].values)
all_upstream_root_ids = set(upstream_neurons['root_id'].values)
all_downstream_root_ids = set(downstream_neurons['root_id'].values)

# Filter synapses to those within our circuit of interest
# Include: upstream->T4/T5, T4/T5->T4/T5, T4/T5->downstream

circuit_synapses = []

for idx, row in synapses_df.iterrows():
    pre_id = row['pre_root_id']
    post_id = row['post_root_id']

    # Include if it's a connection within our circuit
    is_upstream_to_visual = (pre_id in all_upstream_root_ids and
                             (post_id in all_t4_root_ids or post_id in all_t5_root_ids))
    is_visual_to_visual = ((pre_id in all_t4_root_ids or pre_id in all_t5_root_ids) and
                           (post_id in all_t4_root_ids or post_id in all_t5_root_ids))
    is_visual_to_downstream = ((pre_id in all_t4_root_ids or pre_id in all_t5_root_ids) and
                               post_id in all_downstream_root_ids)

    if is_upstream_to_visual or is_visual_to_visual or is_visual_to_downstream:
        circuit_synapses.append({
            'pre_root_id': pre_id,
            'post_root_id': post_id,
            'size': row.get('size', 1),
            'neuropil': row.get('neuropil', 'unknown'),
            'connection_type': (
                'upstream->T4/T5' if is_upstream_to_visual else
                'T4/T5->T4/T5' if is_visual_to_visual else
                'T4/T5->downstream'
            )
        })

    if idx % 10000000 == 0 and idx > 0:
        print(f"  Processed {idx:,} synapses, found {len(circuit_synapses):,} circuit synapses...")

circuit_synapses_df = pd.DataFrame(circuit_synapses)

print(f"\n[OK] Found {len(circuit_synapses_df):,} synapses in visual motion circuit")
print(f"\nSynapse breakdown by connection type:")
print(circuit_synapses_df['connection_type'].value_counts())

# Compute circuit statistics
print("\n" + "="*70)
print("CIRCUIT STATISTICS")
print("="*70)

# Count pre/post connections for each neuron type
def count_connections(synapses, pre_root_ids, post_root_ids, name):
    pre_synapses = synapses[synapses['pre_root_id'].isin(pre_root_ids)]
    post_synapses = synapses[synapses['post_root_id'].isin(post_root_ids)]
    pre_count = len(pre_synapses)
    post_count = len(post_synapses)
    return pre_count, post_count

upstream_pre, upstream_post = count_connections(
    circuit_synapses_df,
    all_upstream_root_ids,
    all_upstream_root_ids,
    "Upstream (L/M/C)"
)

t4t5_pre, t4t5_post = count_connections(
    circuit_synapses_df,
    all_t4_root_ids | all_t5_root_ids,
    all_t4_root_ids | all_t5_root_ids,
    "T4/T5"
)

downstream_pre, downstream_post = count_connections(
    circuit_synapses_df,
    all_downstream_root_ids,
    all_downstream_root_ids,
    "Downstream"
)

print(f"\nUpstream layer (L/M/C neurons):")
print(f"  Sending synapses: {upstream_pre:,}")
print(f"  Receiving synapses: {upstream_post:,}")

print(f"\nT4/T5 layer (direction-selective):")
print(f"  Sending synapses: {t4t5_pre:,}")
print(f"  Receiving synapses: {t4t5_post:,}")

print(f"\nDownstream layer (motor/integration):")
print(f"  Sending synapses: {downstream_pre:,}")
print(f"  Receiving synapses: {downstream_post:,}")

# Analyze T4 vs T5 properties
print(f"\n" + "="*70)
print("T4 vs T5 ANALYSIS")
print("="*70)

t4_ids = set(t4_neurons['root_id'].values)
t5_ids = set(t5_neurons['root_id'].values)

t4_inputs = circuit_synapses_df[circuit_synapses_df['post_root_id'].isin(t4_ids)]
t5_inputs = circuit_synapses_df[circuit_synapses_df['post_root_id'].isin(t5_ids)]

t4_outputs = circuit_synapses_df[circuit_synapses_df['pre_root_id'].isin(t4_ids)]
t5_outputs = circuit_synapses_df[circuit_synapses_df['pre_root_id'].isin(t5_ids)]

print(f"\nT4 neurons ({len(t4_neurons):,}):")
print(f"  Average input synapses: {len(t4_inputs) / len(t4_neurons) if len(t4_neurons) > 0 else 0:.1f}")
print(f"  Average output synapses: {len(t4_outputs) / len(t4_neurons) if len(t4_neurons) > 0 else 0:.1f}")
print(f"  Total inputs: {len(t4_inputs):,}")
print(f"  Total outputs: {len(t4_outputs):,}")

print(f"\nT5 neurons ({len(t5_neurons):,}):")
print(f"  Average input synapses: {len(t5_inputs) / len(t5_neurons) if len(t5_neurons) > 0 else 0:.1f}")
print(f"  Average output synapses: {len(t5_outputs) / len(t5_neurons) if len(t5_neurons) > 0 else 0:.1f}")
print(f"  Total inputs: {len(t5_inputs):,}")
print(f"  Total outputs: {len(t5_outputs):,}")

# Save circuit definition
print(f"\n" + "="*70)
print("SAVING CIRCUIT DATA")
print("="*70)

# Save T4/T5 neurons
t4_neurons.to_csv('visual_circuit_t4_neurons.csv', index=False)
t5_neurons.to_csv('visual_circuit_t5_neurons.csv', index=False)

# Save upstream neurons
upstream_neurons.to_csv('visual_circuit_upstream_neurons.csv', index=False)

# Save circuit synapses
circuit_synapses_df.to_csv('visual_circuit_synapses.csv', index=False)

print(f"\nSaved:")
print(f"  - visual_circuit_t4_neurons.csv ({len(t4_neurons):,} neurons)")
print(f"  - visual_circuit_t5_neurons.csv ({len(t5_neurons):,} neurons)")
print(f"  - visual_circuit_upstream_neurons.csv ({len(upstream_neurons):,} neurons)")
print(f"  - visual_circuit_synapses.csv ({len(circuit_synapses_df):,} synapses)")

print(f"\n[OK] Phase 1 Complete!")
print(f"\nNext steps:")
print(f"  1. Review literature: Klapoetke et al. 2017, Joesch et al. 2010")
print(f"  2. Implement motion opponency model for T4/T5")
print(f"  3. Validate directional tuning against published data")
print(f"  4. Implement optic flow encoding (Lucas-Kanade on video frames)")

print(f"\n{'='*70}\n")
