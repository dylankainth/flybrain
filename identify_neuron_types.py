"""
Identify sensory and motor neurons from FlyWire connectome.
Uses heuristics based on connectivity patterns and published neuroscience.
"""

import pandas as pd
import numpy as np

def identify_neuron_types(neurons_file='fly_neurons.csv', synapses_file='fly_synapses.csv'):
    """
    Identify sensory (input) and motor (output) neurons based on connectivity.

    Returns:
        sensory_neurons: list of neuron root IDs with high in-degree (many inputs from sensory)
        motor_neurons: list of neuron root IDs with high out-degree (many outputs to muscles)
    """

    print("[*] Loading connectome data...")
    neurons_df = pd.read_csv(neurons_file)
    synapses_df = pd.read_csv(synapses_file)

    print(f"    {len(neurons_df)} neurons")
    print(f"    {len(synapses_df)} synapses")

    # Get in-degree and out-degree for each neuron
    print("\n[*] Computing network statistics...")

    out_degree = synapses_df['pre_root_id'].value_counts()
    in_degree = synapses_df['post_root_id'].value_counts()

    print(f"    Neurons with outgoing synapses: {len(out_degree)}")
    print(f"    Neurons with incoming synapses: {len(in_degree)}")

    # Sensory neurons: high in-degree (they receive many connections from sensory pathways)
    # Motor neurons: high out-degree (they send many connections downstream)

    in_degree_threshold = in_degree.quantile(0.90)
    out_degree_threshold = out_degree.quantile(0.90)

    sensory_neurons = in_degree[in_degree > in_degree_threshold].index.tolist()
    motor_neurons = out_degree[out_degree > out_degree_threshold].index.tolist()

    print(f"\n[*] Neuron type identification:")
    print(f"    High in-degree (sensory) threshold: {in_degree_threshold:.0f} inputs")
    print(f"    Sensory neurons (top 10%): {len(sensory_neurons)}")
    print(f"\n    High out-degree (motor) threshold: {out_degree_threshold:.0f} outputs")
    print(f"    Motor neurons (top 10%): {len(motor_neurons)}")

    # Additionally, look for neurons mentioned in literature
    # (This would require matching against a database of known neuron types)

    print(f"\n[*] Sample sensory neurons (by in-degree):")
    top_sensory = in_degree.nlargest(5)
    for nid, degree in top_sensory.items():
        print(f"    {nid}: {degree:.0f} inputs")

    print(f"\n[*] Sample motor neurons (by out-degree):")
    top_motor = out_degree.nlargest(5)
    for nid, degree in top_motor.items():
        print(f"    {nid}: {degree:.0f} outputs")

    return sensory_neurons, motor_neurons


def recommend_neuron_subsets(neurons_file='fly_neurons.csv', synapses_file='fly_synapses.csv'):
    """
    Recommend specific neuron subsets for sensory input and motor output.
    Returns ranges suitable for use in SensoryEncoder and MotorDecoder.
    """

    sensory_neurons, motor_neurons = identify_neuron_types(neurons_file, synapses_file)

    print(f"\n[*] Recommended configuration for io_mapping.py:")
    print(f"\n    # Visual/sensory neurons (use top 300):")
    visual_neurons = sensory_neurons[:300]
    print(f"    VISUAL_NEURONS = {visual_neurons}")

    print(f"\n    # Motor/output neurons (use top 100):")
    motor_subset = motor_neurons[:100]
    print(f"    FORWARD_NEURONS = {motor_subset[:25]}")
    print(f"    LEFT_TURN_NEURONS = {motor_subset[25:50]}")
    print(f"    RIGHT_TURN_NEURONS = {motor_subset[50:75]}")
    print(f"    CLIMB_NEURONS = {motor_subset[75:100]}")

    return sensory_neurons, motor_neurons


if __name__ == "__main__":
    try:
        sensory, motor = recommend_neuron_subsets()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure fly_neurons.csv and fly_synapses.csv exist first")
