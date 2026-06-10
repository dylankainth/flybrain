"""
Fix root_id prefix mismatch between neurons and synapses
"""

import pandas as pd

print("Fixing root_id prefix in synapses...")
print("Adding prefix 720575940000000000 to synapse root_ids")

# Read synapse file
print("Reading fly_synapses_real.csv...")
df = pd.read_csv('fly_synapses_real.csv')

prefix = 720575940000000000

print(f"Before: sample pre_root_id = {df['pre_root_id'].iloc[0]}")

# Add prefix
df['pre_root_id'] = df['pre_root_id'] + prefix
df['post_root_id'] = df['post_root_id'] + prefix

print(f"After:  sample pre_root_id = {df['pre_root_id'].iloc[0]}")

# Save
df.to_csv('fly_synapses_real.csv', index=False)
print("\n[OK] Fixed and saved fly_synapses_real.csv")
print(f"Total synapses: {len(df):,}")
