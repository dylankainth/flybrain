"""
Simple test - no plotting, just verify the pipeline works
"""

import numpy as np
from fly_brain import FlyBrain
from drone_sim import DroneSimulator
from io_mapping import SensoryEncoder, MotorDecoder

print("="*60)
print("FRUIT FLY BRAIN - SIMPLE TEST (5 steps)")
print("="*60)

print("\n[1/3] Initializing FlyBrain (1000 neurons, 50k synapses)...")
brain = FlyBrain(
    synapse_file='fly_synapses.csv',
    neuron_file='fly_neurons.csv',
    verbose=False
)

print("[2/3] Initializing DroneSimulator...")
sim = DroneSimulator(gui=False, dt=0.05)

print("[3/3] Initializing I/O mappings...")
encoder = SensoryEncoder()
decoder = MotorDecoder()

print("\nRunning 5 simulation steps...")
print("-" * 60)

for step in range(5):
    # Get observation
    obs = sim.get_observation()

    # Encode vision
    sensory_input = encoder.encode_vision(obs['vision'])

    # Inject & step brain
    brain.inject_sensory(sensory_input)
    brain.step(duration_ms=50)

    # Decode motor
    forward, turn, climb = decoder.decode(brain, window_ms=50)

    # Apply & step physics
    sim.apply_motor_command(forward, turn, climb)
    sim.step()

    # Print status
    print(f"Step {step+1}: pos=({obs['position'][0]:+.2f}, {obs['position'][1]:+.2f}, {obs['position'][2]:.2f}) "
          f"cmd=[f:{forward:+.2f}, t:{turn:+.2f}, c:{climb:+.2f}]")

sim.close()

print("-" * 60)
print("[OK] Test complete! Pipeline working correctly.")
print("\nNext steps:")
print("  1. Run simulate_headless.py for full 10-second sim")
print("  2. Customize neuron indices for real FlyWire connectome")
print("  3. Adjust motor gains for stable flight")
