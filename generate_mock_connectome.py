"""
Generate mock fly connectome data for testing.
Creates CSV files matching FlyWire format without internet access.
"""

import pandas as pd
import numpy as np

def generate_mock_connectome(n_neurons=1000, seed=42):
    """Generate synthetic but realistic connectome data"""

    np.random.seed(seed)

    print(f"Generating mock connectome ({n_neurons} neurons)...")

    # Generate neurons
    root_ids = np.arange(10000, 10000 + n_neurons, dtype=np.int64)
    x = np.random.uniform(0, 100, n_neurons)
    y = np.random.uniform(0, 100, n_neurons)
    z = np.random.uniform(0, 100, n_neurons)

    neurons_df = pd.DataFrame({
        'root_id': root_ids,
        'x': x,
        'y': y,
        'z': z,
    })

    # Generate synapses (sparse connectivity ~0.1%)
    n_synapses = int(n_neurons * 50)  # ~50 synapses per neuron on average

    pre_indices = np.random.choice(n_neurons, n_synapses)
    post_indices = np.random.choice(n_neurons, n_synapses)

    pre_root_ids = root_ids[pre_indices]
    post_root_ids = root_ids[post_indices]

    # Assign neurotransmitter types
    transmitter_types = ['acetylcholine', 'GABA', 'glutamate', 'unknown']
    transmitters = np.random.choice(transmitter_types, n_synapses, p=[0.4, 0.3, 0.2, 0.1])

    # Synapse sizes (small positive values)
    sizes = np.random.exponential(scale=2, size=n_synapses) + 1

    synapses_df = pd.DataFrame({
        'pre_root_id': pre_root_ids,
        'post_root_id': post_root_ids,
        'neurotransmitter': transmitters,
        'size': sizes,
    })

    # Save
    neurons_df.to_csv('fly_neurons.csv', index=False)
    synapses_df.to_csv('fly_synapses.csv', index=False)

    print(f"[OK] Generated {len(neurons_df)} neurons")
    print(f"[OK] Generated {len(synapses_df)} synapses")
    print(f"  Connectivity: {len(synapses_df) / len(neurons_df):.1f} synapses per neuron")
    print(f"  Transmitter mix: {dict(synapses_df['neurotransmitter'].value_counts())}")

if __name__ == "__main__":
    generate_mock_connectome(n_neurons=1000)
