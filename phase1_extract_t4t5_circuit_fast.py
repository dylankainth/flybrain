"""
Phase 1 OPTIMIZED: Extract T4/T5 Direction-Selective Circuit (Fast Vectorized Version)

Uses pandas vectorized operations instead of row-by-row iteration for 10-100x speedup.
"""

import pandas as pd
import numpy as np
import time

start_time = time.time()

print("="*70)
print("PHASE 1 (OPTIMIZED): EXTRACT T4/T5 DIRECTION-SELECTIVE CIRCUIT")
print("="*70)

# Load FlyWire data
print("\n[1/3] Loading FlyWire neurons...")
neurons_df = pd.read_csv('fly_neurons_real.csv')
print(f"      Total neurons: {len(neurons_df):,}")

# Identify visual system neurons
print("\n[2/3] Identifying visual circuit neuron populations...")

t4_neurons = neurons_df[neurons_df['primary_type'].str.contains('T4', na=False)]
t5_neurons = neurons_df[neurons_df['primary_type'].str.contains('T5', na=False)]

print(f"      T4 neurons (upward motion): {len(t4_neurons):,}")
print(f"      T5 neurons (downward motion): {len(t5_neurons):,}")

# Upstream: L, M, C, T1-T3
upstream_pattern = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6',
                    'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
                    'C1', 'C2', 'C3', 'C4', 'T1', 'T2', 'T3']

upstream_neurons = neurons_df[neurons_df['primary_type'].isin(upstream_pattern)]
print(f"      Upstream (L/M/C/T1-T3): {len(upstream_neurons):,}")

# Create root ID sets for fast lookup
t4_root_ids = set(t4_neurons['root_id'].values)
t5_root_ids = set(t5_neurons['root_id'].values)
upstream_root_ids = set(upstream_neurons['root_id'].values)
visual_root_ids = t4_root_ids | t5_root_ids

print(f"      Total visual circuit neurons: {len(visual_root_ids):,}")

# Load synapses EFFICIENTLY with chunked reading and immediate filtering
print("\n[3/3] Loading and filtering synapses (vectorized)...")
print(f"      Processing 80M+ synapses in chunks...")

circuit_synapses_list = []
chunk_size = 500000
total_processed = 0

for chunk_idx, chunk in enumerate(pd.read_csv('fly_synapses_real.csv',
                                               chunksize=chunk_size,
                                               low_memory=False)):
    # Filter this chunk using vectorized operations
    pre_is_upstream = chunk['pre_root_id'].isin(upstream_root_ids)
    post_is_visual = chunk['post_root_id'].isin(visual_root_ids)

    pre_is_visual = chunk['pre_root_id'].isin(visual_root_ids)
    post_is_visual = chunk['post_root_id'].isin(visual_root_ids)

    pre_is_visual_post_other = chunk['pre_root_id'].isin(visual_root_ids)

    # Find relevant synapses: upstream->visual, visual->visual, visual->other
    mask_upstream_to_visual = pre_is_upstream & post_is_visual
    mask_visual_to_visual = (pre_is_visual) & (post_is_visual)
    mask_visual_to_downstream = (pre_is_visual) & (~post_is_visual)

    relevant = chunk[mask_upstream_to_visual | mask_visual_to_visual | mask_visual_to_downstream].copy()

    if len(relevant) > 0:
        circuit_synapses_list.append(relevant)

    total_processed += len(chunk)
    if (chunk_idx + 1) % 10 == 0:
        elapsed = time.time() - start_time
        print(f"      [{elapsed:6.1f}s] Chunk {chunk_idx+1:3d}: {total_processed:,} synapses processed, "
              f"found {sum(len(df) for df in circuit_synapses_list):,} circuit synapses")

# Combine all chunks
if circuit_synapses_list:
    circuit_synapses_df = pd.concat(circuit_synapses_list, ignore_index=True)
else:
    circuit_synapses_df = pd.DataFrame()

elapsed = time.time() - start_time
print(f"\n[OK] Synapse extraction complete in {elapsed:.1f}s")
print(f"     Found {len(circuit_synapses_df):,} circuit synapses")

# Classify synapse connections
print("\n" + "="*70)
print("ANALYZING CIRCUIT CONNECTIVITY")
print("="*70)

def classify_synapse(row):
    pre = row['pre_root_id']
    post = row['post_root_id']

    if pre in upstream_root_ids:
        return 'upstream->T4T5'
    elif pre in t4_root_ids or pre in t5_root_ids:
        if post in t4_root_ids or post in t5_root_ids:
            return 'T4T5->T4T5'
        else:
            return 'T4T5->downstream'
    return 'other'

if len(circuit_synapses_df) > 0:
    circuit_synapses_df['connection_type'] = circuit_synapses_df.apply(classify_synapse, axis=1)

    print("\nSynapse breakdown:")
    for conn_type in ['upstream->T4T5', 'T4T5->T4T5', 'T4T5->downstream', 'other']:
        count = (circuit_synapses_df['connection_type'] == conn_type).sum()
        if count > 0:
            pct = 100 * count / len(circuit_synapses_df)
            print(f"  {conn_type:20s}: {count:8,} ({pct:5.1f}%)")

    # Save all data
    print("\n" + "="*70)
    print("SAVING CIRCUIT DATA")
    print("="*70)

    t4_neurons.to_csv('visual_circuit_t4_neurons.csv', index=False)
    t5_neurons.to_csv('visual_circuit_t5_neurons.csv', index=False)
    upstream_neurons.to_csv('visual_circuit_upstream_neurons.csv', index=False)
    circuit_synapses_df.to_csv('visual_circuit_synapses.csv', index=False)

    print(f"\nSaved files:")
    print(f"  - visual_circuit_t4_neurons.csv ({len(t4_neurons):,})")
    print(f"  - visual_circuit_t5_neurons.csv ({len(t5_neurons):,})")
    print(f"  - visual_circuit_upstream_neurons.csv ({len(upstream_neurons):,})")
    print(f"  - visual_circuit_synapses.csv ({len(circuit_synapses_df):,})")

    # Compute statistics
    print("\n" + "="*70)
    print("CIRCUIT STATISTICS")
    print("="*70)

    t4_inputs = circuit_synapses_df[circuit_synapses_df['post_root_id'].isin(t4_root_ids)]
    t5_inputs = circuit_synapses_df[circuit_synapses_df['post_root_id'].isin(t5_root_ids)]

    t4_outputs = circuit_synapses_df[circuit_synapses_df['pre_root_id'].isin(t4_root_ids)]
    t5_outputs = circuit_synapses_df[circuit_synapses_df['pre_root_id'].isin(t5_root_ids)]

    print(f"\nT4 Neurons ({len(t4_neurons):,}):")
    print(f"  Avg inputs per cell: {len(t4_inputs) / max(len(t4_neurons), 1):.1f}")
    print(f"  Avg outputs per cell: {len(t4_outputs) / max(len(t4_neurons), 1):.1f}")
    print(f"  Total inputs: {len(t4_inputs):,}")
    print(f"  Total outputs: {len(t4_outputs):,}")

    print(f"\nT5 Neurons ({len(t5_neurons):,}):")
    print(f"  Avg inputs per cell: {len(t5_inputs) / max(len(t5_neurons), 1):.1f}")
    print(f"  Avg outputs per cell: {len(t5_outputs) / max(len(t5_neurons), 1):.1f}")
    print(f"  Total inputs: {len(t5_inputs):,}")
    print(f"  Total outputs: {len(t5_outputs):,}")

    print(f"\nUpstream inputs to T4/T5:")
    upstream_to_visual = circuit_synapses_df[
        (circuit_synapses_df['pre_root_id'].isin(upstream_root_ids)) &
        (circuit_synapses_df['post_root_id'].isin(visual_root_ids))
    ]
    print(f"  Total connections: {len(upstream_to_visual):,}")
    if len(upstream_to_visual) > 0:
        print(f"  Neuropils involved:")
        for neuropil, count in upstream_to_visual['neuropil'].value_counts().head(5).items():
            print(f"    {neuropil}: {count:,}")

total_elapsed = time.time() - start_time
print(f"\n[SUCCESS] Phase 1 complete in {total_elapsed:.1f}s")
print(f"\n{'='*70}\n")
