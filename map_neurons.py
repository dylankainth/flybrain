"""
Extract sensory and motor neuron indices from the generated connectome.
"""

import pandas as pd

neurons = pd.read_csv('fly_neurons.csv')

sensory_neurons = neurons[neurons['neuron_type'] == 'sensory']['root_id'].tolist()
motor_neurons = neurons[neurons['neuron_type'] == 'motor']['root_id'].tolist()

# Convert to ranges if possible
sensory_root_ids = sorted(sensory_neurons)
motor_root_ids = sorted(motor_neurons)

print("# Copy this to io_mapping.py\n")
print("class SensoryEncoder:")
print("    \"\"\"Convert robot sensors -> fly sensory neuron activation\"\"\"")
print(f"\n    # Real sensory neuron root IDs from connectome")
print(f"    SENSORY_ROOT_IDS = {sensory_root_ids[:50]}  # First 50")
print(f"    VISUAL_LEFT_IDX = {sensory_root_ids[:171]}")
print(f"    VISUAL_RIGHT_IDX = {sensory_root_ids[171:342]}")
print(f"    VISUAL_CENTER_IDX = {sensory_root_ids[342:513]}")

print("\n\nclass MotorDecoder:")
print("    \"\"\"Map fly descending neurons -> robot motor commands\"\"\"")
print(f"\n    # Real motor neuron root IDs from connectome")
print(f"    MOTOR_ROOT_IDS = {motor_root_ids[:100]}")
print(f"    FORWARD_NEURONS = {motor_root_ids[:250]}")
print(f"    LEFT_TURN_NEURONS = {motor_root_ids[250:500]}")
print(f"    RIGHT_TURN_NEURONS = {motor_root_ids[500:750]}")
print(f"    CLIMB_NEURONS = {motor_root_ids[750:1004]}")

print(f"\n\nStatistics:")
print(f"Total sensory neurons: {len(sensory_root_ids)}")
print(f"Total motor neurons: {len(motor_root_ids)}")
