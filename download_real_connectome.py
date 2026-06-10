"""
Download real fruit fly connectome from FlyWire.
This uses the official CAVEclient to fetch production data.
Note: Full download (139k neurons, 50M synapses) takes ~30-60 minutes.
"""

from caveclient import CAVEclient
import pandas as pd
import numpy as np
import sys

def download_connectome(n_neurons_max=None, output_dir='.', test_mode=False):
    """
    Fetch fruit fly connectome from FlyWire production.

    Args:
        n_neurons_max: Limit to first N neurons (None = all 139k)
        output_dir: Where to save CSV files
        test_mode: If True, download only first 1000 neurons for quick test
    """

    print("[*] Connecting to FlyWire...")
    client = CAVEclient('flywire_production')

    # Get all valid neurons
    print("[*] Fetching neuron list...")
    neurons = client.materialize.query_table(
        'nucleus_detection_v1',
        filter_equal_dict={'valid': True},
        split_positions=True
    )
    print(f"    Total neurons available: {len(neurons)}")

    # Limit for testing
    if test_mode:
        n_neurons_max = 5000
        print(f"    [TEST MODE] Limiting to first {n_neurons_max} neurons")

    if n_neurons_max:
        neurons = neurons.iloc[:n_neurons_max]
        print(f"    Using {len(neurons)} neurons")

    print(f"\n[*] Fetching synapses for {len(neurons)} neurons...")
    print(f"    (This may take several minutes for all 139k neurons)")

    neuron_ids = neurons['root_id'].values

    # Query synapses in batches to avoid timeout
    all_synapses = []
    batch_size = 5000

    for i, batch_start in enumerate(range(0, len(neuron_ids), batch_size)):
        batch_end = min(batch_start + batch_size, len(neuron_ids))
        batch_ids = neuron_ids[batch_start:batch_end]

        print(f"    Batch {i+1}/{(len(neuron_ids)-1)//batch_size + 1}: neurons {batch_start}-{batch_end}...", end='', flush=True)

        try:
            # Get synapses where these neurons are presynaptic
            synapses = client.materialize.query_table(
                'synapses_nt_v1',
                filter_in_dict={'pre_root_id': batch_ids},
            )
            all_synapses.append(synapses)
            print(f" {len(synapses)} synapses")
        except Exception as e:
            print(f" ERROR: {e}")
            continue

    if all_synapses:
        synapses = pd.concat(all_synapses, ignore_index=True)
    else:
        synapses = pd.DataFrame()

    print(f"\n[*] Saving data...")
    neurons.to_csv(f'{output_dir}/fly_neurons.csv', index=False)
    synapses.to_csv(f'{output_dir}/fly_synapses.csv', index=False)

    print(f"\n[OK] Download complete!")
    print(f"    fly_neurons.csv: {len(neurons)} neurons")
    print(f"    fly_synapses.csv: {len(synapses)} synapses")

    if len(synapses) > 0:
        print(f"\n[*] Connectome statistics:")
        print(f"    Synapses per neuron: {len(synapses) / len(neurons):.1f}")
        if 'neurotransmitter' in synapses.columns:
            print(f"    Neurotransmitter breakdown:")
            for nt, count in synapses['neurotransmitter'].value_counts().items():
                print(f"      {nt}: {count} ({100*count/len(synapses):.1f}%)")

if __name__ == "__main__":
    # Parse command line args
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    full_mode = '--full' in sys.argv or '-f' in sys.argv

    if test_mode:
        print("="*60)
        print("TEST MODE: Downloading 5,000 neurons")
        print("="*60)
        download_connectome(n_neurons_max=5000, test_mode=True)
    elif full_mode:
        print("="*60)
        print("FULL MODE: Downloading all 139,000 neurons")
        print("This will take 30-60 minutes")
        print("="*60)
        download_connectome()
    else:
        print("="*60)
        print("USAGE: python download_real_connectome.py [--test|--full]")
        print("="*60)
        print("\nOptions:")
        print("  --test, -t   Download 5,000 neurons (quick test)")
        print("  --full, -f   Download all 139,000 neurons (slow)")
        print("\nRunning test mode by default...")
        download_connectome(n_neurons_max=5000, test_mode=True)
