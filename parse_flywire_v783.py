"""
Parse FlyWire FAFB v783 data specifically for the fly brain drone simulator
Handles: neurons.csv, consolidated_cell_types.csv, fafb_v783_princeton_synapse_table.csv
"""

import pandas as pd
import numpy as np

def parse_fafb_v783(neurons_file='neurons.csv',
                     cell_types_file='consolidated_cell_types.csv',
                     synapses_file='fafb_v783_princeton_synapse_table.csv',
                     chunksize=50000):
    """
    Parse FAFB v783 data and create simulator-compatible format.
    Uses chunked reading for the massive synapse table.
    """

    print("="*70)
    print("FlyWire FAFB v783 Parser - 139,255 Neurons, 50M+ Synapses")
    print("="*70)

    # 1. Load neurons
    print("\n[1/3] Loading neuron data...")
    neurons_df = pd.read_csv(neurons_file)
    print(f"      Loaded {len(neurons_df):,} neurons")
    print(f"      Columns: {list(neurons_df.columns)}")

    # 2. Load cell types
    print("[2/3] Loading cell type classifications...")
    cell_types_df = pd.read_csv(cell_types_file)
    print(f"      Loaded {len(cell_types_df):,} cell types")

    # Merge cell types into neurons
    neurons_df = neurons_df.merge(cell_types_df[['root_id', 'primary_type', 'additional_type(s)']],
                                   on='root_id', how='left')
    print(f"      Merged cell type info")

    # Classify neurons into functional types
    print(f"\n      Identifying neuron populations...")
    neurons_df['neuron_type'] = 'interneuron'

    # Visual system neurons (rough classification based on types like L1-L6, M1-M6, etc.)
    visual_types = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
                    'C2', 'C3', 'T1', 'T2', 'T3', 'T4', 'T5']
    neurons_df.loc[neurons_df['primary_type'].isin(visual_types), 'neuron_type'] = 'sensory'

    # Motor descending neurons (rough classification)
    motor_types = ['DN', 'MN', 'A1', 'A2', 'A3']  # Descending/motor neurons
    neurons_df.loc[neurons_df['primary_type'].str.startswith(('DN', 'MN'), na=False), 'neuron_type'] = 'motor'

    print(f"      Neuron type breakdown:")
    for ntype in ['sensory', 'motor', 'interneuron']:
        count = (neurons_df['neuron_type'] == ntype).sum()
        pct = 100 * count / len(neurons_df)
        print(f"        {ntype:12s}: {count:6,} ({pct:5.1f}%)")

    # 3. Load and process synapses (chunked for efficiency)
    print("\n[3/3] Loading synapse data (large file, processing in chunks)...")

    all_synapses = []
    total_synapses = 0

    for chunk_num, synapses_chunk in enumerate(pd.read_csv(synapses_file, chunksize=chunksize)):
        if chunk_num % 20 == 0 and chunk_num > 0:
            print(f"      Processed {total_synapses:,} synapses...")

        # Parse root IDs (they have a prefix like "720575940")
        # Format appears to be: "pre_root_id_720575940" where 720575940 is the prefix
        synapses_chunk.rename(columns={
            'pre_root_id_720575940': 'pre_root_id',
            'post_root_id_720575940': 'post_root_id'
        }, inplace=True)

        # Extract neurotransmitter info from neurons table
        synapses_chunk = synapses_chunk.merge(
            neurons_df[['root_id', 'nt_type']].rename(columns={'root_id': 'pre_root_id', 'nt_type': 'neurotransmitter'}),
            on='pre_root_id', how='left'
        )

        all_synapses.append(synapses_chunk)
        total_synapses += len(synapses_chunk)

    synapses_df = pd.concat(all_synapses, ignore_index=True)
    print(f"      Loaded {len(synapses_df):,} synapses total")

    # 4. Prepare output data
    print(f"\n[*] Preparing output files...")

    # Create output neurons file with minimal needed columns
    output_neurons = neurons_df[[
        'root_id', 'primary_type', 'neuron_type', 'nt_type',
        'ach_avg', 'gaba_avg', 'glut_avg'
    ]].copy()

    # Add coordinates from synapses
    pre_coords = synapses_df[['pre_root_id', 'pre_x', 'pre_y', 'pre_z']].drop_duplicates('pre_root_id')
    pre_coords.rename(columns={
        'pre_root_id': 'root_id',
        'pre_x': 'x', 'pre_y': 'y', 'pre_z': 'z'
    }, inplace=True)

    output_neurons = output_neurons.merge(pre_coords, on='root_id', how='left')

    # Create output synapses with needed columns
    output_synapses = synapses_df[[
        'pre_root_id', 'post_root_id', 'neurotransmitter', 'size', 'neuropil'
    ]].copy()

    # Save files
    print(f"    Saving neurons...")
    output_neurons.to_csv('fly_neurons_real.csv', index=False)
    print(f"      -> fly_neurons_real.csv ({len(output_neurons):,} neurons)")

    print(f"    Saving synapses...")
    output_synapses.to_csv('fly_synapses_real.csv', index=False)
    print(f"      -> fly_synapses_real.csv ({len(output_synapses):,} synapses)")

    # 5. Summary statistics
    print(f"\n[OK] Conversion Complete!")
    print(f"\n{'='*70}")
    print(f"REAL FRUIT FLY CONNECTOME LOADED")
    print(f"{'='*70}")
    print(f"Neurons:           {len(output_neurons):,}")
    print(f"Synapses:          {len(output_synapses):,}")
    print(f"Avg synapses/neuron: {len(output_synapses)/len(output_neurons):.1f}")
    print(f"\nNeuron Type Breakdown:")
    for ntype in output_neurons['neuron_type'].unique():
        count = (output_neurons['neuron_type'] == ntype).sum()
        print(f"  {ntype:12s}: {count:6,}")

    if 'neurotransmitter' in output_synapses.columns:
        print(f"\nNeurotransmitter Types:")
        for nt, count in output_synapses['neurotransmitter'].value_counts().head(10).items():
            pct = 100 * count / len(output_synapses)
            print(f"  {nt:12s}: {count:8,} ({pct:5.1f}%)")

    print(f"\n[*] Ready to simulate with REAL FLY BRAIN!")
    print(f"    Next: run simulate_headless.py")
    print(f"{'='*70}\n")

    return output_neurons, output_synapses


if __name__ == "__main__":
    import os

    # Find files
    neurons = 'neurons.csv' if os.path.exists('neurons.csv') else None
    cell_types = 'consolidated_cell_types.csv' if os.path.exists('consolidated_cell_types.csv') else None
    synapses = 'fafb_v783_princeton_synapse_table.csv' if os.path.exists('fafb_v783_princeton_synapse_table.csv') else None

    if neurons and cell_types and synapses:
        print(f"Found all required files!")
        print(f"  neurons.csv: {os.path.getsize(neurons)/1e6:.1f} MB")
        print(f"  consolidated_cell_types.csv: {os.path.getsize(cell_types)/1e6:.1f} MB")
        print(f"  fafb_v783_princeton_synapse_table.csv: {os.path.getsize(synapses)/1e6:.1f} MB")
        print()

        parse_fafb_v783(neurons, cell_types, synapses)
    else:
        print("[!] Missing files:")
        print(f"  neurons.csv: {'✓' if neurons else '✗'}")
        print(f"  consolidated_cell_types.csv: {'✓' if cell_types else '✗'}")
        print(f"  fafb_v783_princeton_synapse_table.csv: {'✓' if synapses else '✗'}")
