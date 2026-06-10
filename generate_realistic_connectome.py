"""
Generate realistic synthetic fly connectome based on Drosophila neuroscience.
Incorporates known structural properties: small-world, modular, sparse connectivity.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def generate_realistic_connectome(n_neurons=10000, seed=42):
    """
    Generate connectome with biological realism:
    - Sparse connectivity (~0.5%)
    - Small-world topology (local clustering + long-range connections)
    - Distance-dependent connectivity (nearby neurons more likely to connect)
    - Neurotransmitter distribution based on fly brain statistics
    - Identify sensory and motor neuron populations
    """

    np.random.seed(seed)
    print(f"Generating realistic connectome ({n_neurons} neurons)...")

    # Generate 3D positions (simulating brain volume)
    print("  Step 1: Generating neuron positions...")
    x = np.random.uniform(0, 100, n_neurons)
    y = np.random.uniform(0, 100, n_neurons)
    z = np.random.uniform(0, 50, n_neurons)

    # Assign neuron types (approximate percentages from literature)
    print("  Step 2: Assigning neuron types...")
    neuron_types = np.random.choice(
        ['sensory', 'interneuron', 'motor', 'modulatory'],
        n_neurons,
        p=[0.05, 0.80, 0.10, 0.05]  # Realistic proportions
    )

    # Create neurons DataFrame
    neurons_df = pd.DataFrame({
        'root_id': np.arange(100000, 100000 + n_neurons, dtype=np.int64),
        'x': x,
        'y': y,
        'z': z,
        'neuron_type': neuron_types
    })

    # Generate synapses with biological constraints
    print("  Step 3: Generating synapses (this may take a moment)...")

    # Overall connectivity: ~0.5% (sparse like real brain)
    n_possible = n_neurons * n_neurons
    sparsity = 0.005
    n_synapses = max(int(n_possible * sparsity), n_neurons * 10)

    pre_indices = []
    post_indices = []
    transmitters = []
    sizes = []

    # Local connectivity (small-world)
    print("    Adding local connections...")
    for i in range(n_neurons):
        # Each neuron connects to ~10-20 nearby neighbors (local clustering)
        n_local = np.random.randint(5, 20)
        distances = np.sqrt((x - x[i])**2 + (y - y[i])**2 + (z - z[i])**2)
        nearest = np.argsort(distances)[1:n_local+1]  # Exclude self

        for j in nearest:
            if np.random.rand() < 0.8:  # 80% of local neighbors connect
                pre_indices.append(i)
                post_indices.append(j)

    # Long-range connections (rewiring for small-world)
    print("    Adding long-range connections...")
    n_longrange = max(len(pre_indices) // 5, 1000)
    for _ in range(n_longrange):
        i = np.random.randint(0, n_neurons)
        j = np.random.randint(0, n_neurons)
        if i != j:
            pre_indices.append(i)
            post_indices.append(j)

    # Assign neurotransmitters based on presynaptic neuron type
    print("    Assigning neurotransmitters...")
    for pre_idx in pre_indices:
        pre_type = neuron_types[pre_idx]

        if pre_type == 'motor':
            # Motor neurons: mostly glutamatergic (excitatory)
            nt = np.random.choice(['glutamate', 'acetylcholine'], p=[0.7, 0.3])
        elif pre_type == 'modulatory':
            # Modulatory: diverse transmitters
            nt = np.random.choice(['acetylcholine', 'GABA', 'dopamine'], p=[0.4, 0.4, 0.2])
        elif pre_type == 'sensory':
            # Sensory: mixed
            nt = np.random.choice(['acetylcholine', 'GABA'], p=[0.6, 0.4])
        else:  # interneuron
            # Interneurons: mostly GABAergic (inhibitory)
            nt = np.random.choice(['GABA', 'acetylcholine'], p=[0.7, 0.3])

        transmitters.append(nt)

    # Synapse sizes (strength)
    print("    Assigning synapse strengths...")
    sizes = np.random.exponential(scale=2, size=len(pre_indices)) + 1

    # Remove duplicates
    synapse_set = set(zip(pre_indices, post_indices))
    pre_indices = [s[0] for s in synapse_set]
    post_indices = [s[1] for s in synapse_set]
    transmitters = transmitters[:len(pre_indices)]
    sizes = sizes[:len(pre_indices)]

    # Create synapses DataFrame
    synapses_df = pd.DataFrame({
        'pre_root_id': neurons_df['root_id'].iloc[pre_indices].values,
        'post_root_id': neurons_df['root_id'].iloc[post_indices].values,
        'neurotransmitter': transmitters[:len(pre_indices)],
        'size': sizes[:len(pre_indices)]
    })

    # Save
    neurons_df.to_csv('fly_neurons.csv', index=False)
    synapses_df.to_csv('fly_synapses.csv', index=False)

    print(f"\n[OK] Connectome generated!")
    print(f"  Neurons: {len(neurons_df)}")
    print(f"  Synapses: {len(synapses_df)}")
    print(f"  Connectivity: {len(synapses_df) / len(neurons_df):.1f} synapses per neuron")
    print(f"  Sparsity: {100 * len(synapses_df) / (len(neurons_df)**2):.3f}%")

    # Print neuron type breakdown
    print(f"\n  Neuron type breakdown:")
    for ntype in ['sensory', 'interneuron', 'motor', 'modulatory']:
        count = np.sum(neuron_types == ntype)
        pct = 100 * count / len(neurons_df)
        print(f"    {ntype:12s}: {count:5d} ({pct:5.1f}%)")

    # Print transmitter breakdown
    print(f"\n  Neurotransmitter breakdown:")
    for nt, count in pd.Series(transmitters).value_counts().items():
        pct = 100 * count / len(synapses_df)
        print(f"    {nt:12s}: {count:6d} ({pct:5.1f}%)")

    # Identify sensory and motor populations
    sensory_neurons = neurons_df[neurons_df['neuron_type'] == 'sensory']['root_id'].values
    motor_neurons = neurons_df[neurons_df['neuron_type'] == 'motor']['root_id'].values

    print(f"\n  Sensory neurons: {len(sensory_neurons)}")
    print(f"  Motor neurons: {len(motor_neurons)}")

    return neurons_df, synapses_df, sensory_neurons, motor_neurons


if __name__ == "__main__":
    neurons, synapses, sensory, motor = generate_realistic_connectome(n_neurons=10000)
    print(f"\n[*] Files saved: fly_neurons.csv, fly_synapses.csv")
