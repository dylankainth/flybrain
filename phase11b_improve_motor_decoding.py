"""
PHASE 11B: IMPROVE MOTOR DECODING - ANALYZE DESCENDING NEURONS

The evolutionary brain is firing (37k neurons/timestep) but moving backward.
This suggests the descending neuron (DN) to motor command mapping needs refinement.

Approach:
1. Identify DN types and their known functions from literature
2. Map DN activity to flight motor outputs
3. Use population code: bilateral comparison for yaw, symmetric for forward/climb
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("PHASE 11B: IMPROVE MOTOR DECODING - DN ANALYSIS")
print("="*70)

# Load connectome
neurons_df = pd.read_csv('fly_neurons_real.csv')
print(f"\nTotal neurons: {len(neurons_df)}")

# Find descending neurons
descending = neurons_df[neurons_df['primary_type'].str.contains('DN', na=False)]
print(f"Descending neurons: {len(descending)}")

# Analyze DN types
dn_types = descending['primary_type'].value_counts()
print(f"\nDN types (top 20):")
for name, count in dn_types.head(20).items():
    print(f"  {name}: {count}")

# Flight-specific DN groups (from literature)
# DNg02: wingbeat amplitude regulation (flight thrust)
# DNp01-10: various flight behaviors
# DNp06: straight flight maintenance
# DNp10: landing initiation
# DNa: anterior/visual DNs

flight_dn = descending[descending['primary_type'].str.contains('DNg02|DNp0[1-6]|DNa', na=False, case=False)]
print(f"\nFlight-related DNs identified: {len(flight_dn)} out of {len(descending)}")

# Identify bilateral pairs (left vs right)
left_dns = descending[descending['primary_type'].str.contains('left|L$', na=False, case=False)]
right_dns = descending[descending['primary_type'].str.contains('right|R$', na=False, case=False)]

print(f"\nBilateral organization:")
print(f"  Left DNs: {len(left_dns)}")
print(f"  Right DNs: {len(right_dns)}")
print(f"  Unpaired: {len(descending) - len(left_dns) - len(right_dns)}")

# Load synapses for connectivity analysis
synapses_list = []
print(f"\nLoading synapses...")
for i, chunk in enumerate(pd.read_csv('fly_synapses_real.csv', chunksize=1000000, low_memory=False)):
    synapses_list.append(chunk.iloc[np.arange(0, len(chunk), 100)])

synapses_df = pd.concat(synapses_list, ignore_index=True)
root_id_to_idx = {rid: idx for idx, rid in enumerate(neurons_df['root_id'].values)}

# Find inputs to DNs (what drives them)
dn_indices = set(descending.index)
dn_inputs = synapses_df[synapses_df['post_root_id'].map(root_id_to_idx).isin(dn_indices)]

print(f"Synapses onto descending neurons: {len(dn_inputs)}")

# Analyze input neuron types to DNs
input_neurons = neurons_df.loc[dn_inputs['pre_root_id'].map(root_id_to_idx)]
input_types = input_neurons['primary_type'].value_counts()
print(f"\nTop input types to DNs:")
for name, count in input_types.head(15).items():
    print(f"  {name}: {count}")

# Key insight: T4/T5 (motion detectors) should heavily drive DNs
motion_inputs = input_neurons[input_neurons['primary_type'].str.contains('T4|T5|Tm', na=False)]
print(f"\nMotion detector inputs to DNs: {len(motion_inputs)}")

print("\n" + "="*70)
print("MOTOR DECODING STRATEGY")
print("="*70)
print("""
Based on fly neurobiology:

1. FORWARD THRUST: DNg02 + flight-related DNs
   - DNg02 population codes wingbeat amplitude
   - Symmetric activity = forward flight

2. TURN (YAW): Bilateral comparison
   - Left DN activity > Right DN activity = turn right
   - Right DN activity > Left DN activity = turn left

3. CLIMB (ALTITUDE): Vertical-tuned DNs
   - Descending neurons with vertical preference
   - Activate when climbing needed
""")

# Proposed improved decoding
print("\nNew motor mapping:")
print("  Forward = sum(all_dn_spikes) / n_dn")
print("  Turn = (left_dn_spikes - right_dn_spikes) / n_dn")
print("  Climb = vertical_tuned_dn_spikes / n_vertical_dns")

print("\n[OK] Analysis complete. Ready for Phase 11B simulation.")
print("="*70)
