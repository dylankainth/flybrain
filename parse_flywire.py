"""
Parse FlyWire FAFB v783 data files into simulator format.
Converts downloaded CSVs to fly_neurons.csv and fly_synapses.csv
"""

import pandas as pd
import numpy as np
import os
import sys

def parse_flywire_data(cell_types_file='cell_types.csv',
                       synapse_table_file='synapse_table.csv',
                       neurotransmitter_file='neurotransmitter_predictions.csv',
                       coordinates_file='marked_neuron_coordinates.csv',
                       output_dir='.'):
    """
    Parse FlyWire data files and create simulator-compatible CSVs

    Args:
        cell_types_file: Cell Types CSV from FlyWire
        synapse_table_file: Synapse Table CSV (the big 2.7GB file)
        neurotransmitter_file: Neurotransmitter predictions
        coordinates_file: Neuron coordinates
        output_dir: Where to save fly_neurons.csv and fly_synapses.csv
    """

    print("="*60)
    print("FlyWire FAFB v783 Data Parser")
    print("="*60)

    # Load neurons
    print("\n[1/4] Loading neuron data...")
    try:
        neurons_df = pd.read_csv(cell_types_file)
        print(f"    Loaded {len(neurons_df)} neurons from {cell_types_file}")
        print(f"    Columns: {list(neurons_df.columns)}")
    except Exception as e:
        print(f"    [!] Error loading cell types: {e}")
        return False

    # Load neuron coordinates if available
    print("[2/4] Loading neuron coordinates...")
    try:
        coords_df = pd.read_csv(coordinates_file)
        print(f"    Loaded coordinates for {len(coords_df)} neurons")
        # Merge with neurons
        neurons_df = neurons_df.merge(coords_df[['root_id', 'x', 'y', 'z']],
                                       on='root_id', how='left')
        print(f"    Merged coordinates into neuron table")
    except Exception as e:
        print(f"    [!] No coordinates file or merge failed: {e}")
        if 'x' not in neurons_df.columns:
            print("    [!] WARNING: No x,y,z coordinates - simulation may not work properly")

    # Load synapses
    print("[3/4] Loading synapse data (this may take a minute)...")
    try:
        synapses_df = pd.read_csv(synapse_table_file,
                                   dtype={'pre_root_id': 'int64',
                                          'post_root_id': 'int64'})
        print(f"    Loaded {len(synapses_df)} synapses")
        print(f"    Columns: {list(synapses_df.columns)}")
    except Exception as e:
        print(f"    [!] Error loading synapses: {e}")
        return False

    # Load neurotransmitter predictions
    print("[4/4] Loading neurotransmitter predictions...")
    try:
        nt_df = pd.read_csv(neurotransmitter_file)
        print(f"    Loaded neurotransmitter predictions for {len(nt_df)} neurons")
        # Merge with synapses if needed
        if 'neurotransmitter' not in synapses_df.columns and 'pre_root_id' in nt_df.columns:
            synapses_df = synapses_df.merge(nt_df[['pre_root_id', 'neurotransmitter']],
                                            on='pre_root_id', how='left')
            print(f"    Merged neurotransmitter predictions")
    except Exception as e:
        print(f"    [!] Warning: neurotransmitter loading failed: {e}")

    # Save neurons
    print(f"\n[*] Saving neurons...")
    neurons_output = os.path.join(output_dir, 'fly_neurons.csv')
    neurons_df.to_csv(neurons_output, index=False)
    print(f"    Saved {len(neurons_df)} neurons to {neurons_output}")

    # Save synapses
    print(f"[*] Saving synapses...")
    synapses_output = os.path.join(output_dir, 'fly_synapses.csv')
    synapses_df.to_csv(synapses_output, index=False)
    print(f"    Saved {len(synapses_df)} synapses to {synapses_output}")

    # Print summary
    print(f"\n[OK] Conversion complete!")
    print(f"\nConnectome Summary:")
    print(f"  Neurons: {len(neurons_df)}")
    print(f"  Synapses: {len(synapses_df)}")
    print(f"  Synapse density: {len(synapses_df) / len(neurons_df):.1f} per neuron")

    if 'neurotransmitter' in synapses_df.columns:
        print(f"\n  Neurotransmitter breakdown:")
        for nt, count in synapses_df['neurotransmitter'].value_counts().items():
            pct = 100 * count / len(synapses_df)
            print(f"    {nt}: {count} ({pct:.1f}%)")

    return True


if __name__ == "__main__":
    # Try to find files in current directory
    import glob

    print("Looking for FlyWire data files...")
    cell_types = glob.glob("*cell*type*.csv") or glob.glob("*Cell*Type*.csv")
    synapses = glob.glob("*synapse*.csv") or glob.glob("*Synapse*.csv")
    neurotrans = glob.glob("*neurotrans*.csv") or glob.glob("*Neurotrans*.csv")
    coords = glob.glob("*coord*.csv") or glob.glob("*Coord*.csv")

    if cell_types and synapses:
        print(f"\nFound files:")
        print(f"  Cell types: {cell_types[0]}")
        print(f"  Synapses: {synapses[0]}")
        if neurotrans:
            print(f"  Neurotransmitter: {neurotrans[0]}")
        if coords:
            print(f"  Coordinates: {coords[0]}")

        parse_flywire_data(
            cell_types_file=cell_types[0],
            synapse_table_file=synapses[0],
            neurotransmitter_file=neurotrans[0] if neurotrans else None,
            coordinates_file=coords[0] if coords else None
        )
    else:
        print("\n[!] FlyWire data files not found in current directory")
        print("Expected files:")
        print("  - *cell*type*.csv (or *Cell*Type*.csv)")
        print("  - *synapse*.csv (or *Synapse*.csv)")
        print("  - *neurotrans*.csv (optional)")
        print("  - *coord*.csv (optional)")
        print("\nPlace downloaded files in this directory and run again")
