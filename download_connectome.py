"""
Download fruit fly connectome from FlyWire.
This creates two CSV files: neurons + synapses with neurotransmitter predictions.
Takes ~10 minutes on decent internet.
"""

from caveclient import CAVEclient
import pandas as pd
import numpy as np

def download_connectome(output_dir='.'):
    """Fetch full adult fly connectome from FlyWire production"""

    print("[1/3] Connecting to FlyWire...")
    client = CAVEclient('flywire_production')

    # Get all valid neurons
    print("[2/3] Fetching 139k neurons...")
    neurons = client.materialize.query_table(
        'nucleus_detection_v1',
        filter_equal_dict={'valid': True},
        split_positions=True
    )
    print(f"  ✓ Downloaded {len(neurons)} neurons")

    # Get all synapses with neurotransmitter predictions (Eckstein et al. 2024)
    print("[3/3] Fetching 50M synapses (this is slow)...")
    synapses = client.materialize.query_table(
        'synapses_nt_v1',  # Includes neurotransmitter predictions
        limit=None  # Get all
    )
    print(f"  ✓ Downloaded {len(synapses)} synapses")

    # Save to disk
    neurons.to_csv(f'{output_dir}/fly_neurons.csv', index=False)
    synapses.to_csv(f'{output_dir}/fly_synapses.csv', index=False)

    print(f"\n✓ Saved to:")
    print(f"  - {output_dir}/fly_neurons.csv ({len(neurons)} rows)")
    print(f"  - {output_dir}/fly_synapses.csv ({len(synapses)} rows)")

    # Quick stats
    print(f"\nConnectome Stats:")
    print(f"  Neurons: {len(neurons)}")
    print(f"  Synapses: {len(synapses)}")
    print(f"  Avg connections per neuron: {len(synapses) / len(neurons):.1f}")

    return neurons, synapses

if __name__ == "__main__":
    neurons, synapses = download_connectome()
