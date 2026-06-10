"""
Diagnose why motor neurons aren't firing
"""

import numpy as np
from fly_brain import FlyBrain
from io_mapping import MotorDecoder

print("="*60)
print("Brain Diagnostic: Checking Neuron Firing")
print("="*60)

print("\n[1] Initializing brain...")
brain = FlyBrain(verbose=False)

print("[2] Setting up motor neurons...")
decoder = MotorDecoder()

print(f"\n[*] Motor neuron populations:")
print(f"    FORWARD: {len(decoder.FORWARD_NEURONS)} neurons")
print(f"    LEFT_TURN: {len(decoder.LEFT_TURN_NEURONS)} neurons")
print(f"    RIGHT_TURN: {len(decoder.RIGHT_TURN_NEURONS)} neurons")
print(f"    CLIMB: {len(decoder.CLIMB_NEURONS)} neurons")

print(f"\n[*] Sample motor neuron root IDs:")
print(f"    Forward: {decoder.FORWARD_NEURONS[:5]}")
print(f"    Climb: {decoder.CLIMB_NEURONS[:5]}")

print(f"\n[3] Testing neuron injection...")
test_neurons = decoder.FORWARD_NEURONS[:50]
print(f"    Injecting 1000 pA into first 50 forward neurons...")

brain.inject_sensory({tuple(test_neurons): 1000})
brain.step(duration_ms=50)

print(f"\n[4] Checking spike monitor...")
print(f"    Total spikes recorded: {len(brain.spikes.i)}")
print(f"    Spike times: {brain.spikes.t[:20] if len(brain.spikes.t) > 0 else 'NO SPIKES'}")

if len(brain.spikes.i) > 0:
    print(f"    Spike neuron IDs: {brain.spikes.i[:20]}")
    unique_spiking = set(brain.spikes.i)
    print(f"    Number of unique neurons that spiked: {len(unique_spiking)}")
else:
    print("    [!] NO NEURONS FIRED - even with 1000 pA injection!")

print(f"\n[5] Testing baseline motor drive...")
brain.set_baseline_motor_drive(decoder.FORWARD_NEURONS, drive_strength=2000)
brain.apply_baseline_motor_drive()
brain.step(duration_ms=50)

forward_spikes_before = len(brain.spikes.i)
print(f"    Spikes after 2000 pA baseline: {forward_spikes_before}")

if forward_spikes_before > 0:
    print(f"    SUCCESS: Motor neurons are firing!")
    rates = []
    for _ in range(10):
        brain.apply_baseline_motor_drive()
        brain.step(duration_ms=50)
        rate = brain.get_firing_rate(decoder.FORWARD_NEURONS, window_ms=50)
        rates.append(rate)
    print(f"    Firing rates over 10 steps: {rates}")
else:
    print(f"    [!] PROBLEM: Even with 2000 pA baseline, no spikes!")
    print(f"\n[DIAGNOSIS]")
    print(f"    The synthetic connectome may not have:")
    print(f"    - Proper input connections to motor neurons")
    print(f"    - Sufficient recurrent/tonic connectivity")
    print(f"    - Realistic neuron parameters for spiking at these currents")
    print(f"\n[SOLUTION]")
    print(f"    1. Use the full FlyWire 139k-neuron connectome (real data)")
    print(f"    2. Manually adjust LIF neuron parameters (El, Vt, tau)")
    print(f"    3. Increase injection currents to extreme levels (5000+ pA)")
