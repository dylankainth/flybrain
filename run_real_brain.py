"""
Run the REAL fruit fly brain (139k neurons) on the drone simulator.
Uses FlyWire FAFB v783 data.
"""

import numpy as np
import matplotlib.pyplot as plt
from fly_brain import FlyBrain
from drone_sim import DroneSimulator
from io_mapping import SensoryEncoder, MotorDecoder

print("="*70)
print("FLYING THE REAL FRUIT FLY BRAIN - FlyWire FAFB v783")
print("="*70)
print(f"\nLoading 139,255 real neurons with 80M real synapses...")
print()

# Initialize with REAL connectome
brain = FlyBrain(
    synapse_file='fly_synapses_real.csv',
    neuron_file='fly_neurons_real.csv',
    verbose=True
)

print("\n[2/4] Initializing drone physics...")
sim = DroneSimulator(gui=False, dt=0.05)

print("[3/4] Setting up sensory/motor mapping...")
encoder = SensoryEncoder()
decoder = MotorDecoder(gain_forward=50, gain_turn=30, gain_climb=20)

print("[4/4] Running 10-second simulation...\n")

timesteps = 200
positions = []
motor_commands = []

for step in range(timesteps):
    # Get observation
    obs = sim.get_observation()

    # Encode vision
    sensory_input = encoder.encode_vision(obs['vision'])

    # Inject sensory input
    brain.inject_sensory(sensory_input)
    brain.step(duration_ms=50)

    # Decode motor output
    forward, turn, climb = decoder.decode(brain, window_ms=50)

    # Apply motor commands
    sim.apply_motor_command(forward, turn, climb)
    sim.step()

    # Log
    positions.append(obs['position'])
    motor_commands.append([forward, turn, climb])

    if (step + 1) % 50 == 0:
        elapsed = (step + 1) * 0.05
        print(f"  [{elapsed:6.1f}s] pos=({obs['position'][0]:+.2f}, {obs['position'][1]:+.2f}, {obs['position'][2]:.2f}) "
              f"cmd=[f:{forward:+.2f}, t:{turn:+.2f}, c:{climb:+.2f}]")

sim.close()

# Analysis
positions = np.array(positions)
motor_commands = np.array(motor_commands)

print(f"\n[OK] Simulation Complete!")
print(f"\n{'='*70}")
print(f"RESULTS WITH REAL FLY BRAIN")
print(f"{'='*70}")
print(f"Duration: 10 seconds")
print(f"Final position: ({positions[-1][0]:.2f}, {positions[-1][1]:.2f}, {positions[-1][2]:.2f})")
print(f"Altitude change: {positions[-1][2] - positions[0][2]:+.2f}m")
print(f"Max motor command: {np.abs(motor_commands).max():.3f}")
print(f"\nAverage by axis:")
print(f"  Forward:  {motor_commands[:, 0].mean():+.3f}")
print(f"  Turn:     {motor_commands[:, 1].mean():+.3f}")
print(f"  Climb:    {motor_commands[:, 2].mean():+.3f}")

if positions[-1][2] > 0.9:
    print(f"\n[SUCCESS] Drone maintained altitude! Neural control working!")
elif positions[-1][2] > positions[0][2]:
    print(f"\n[CLIMBING] Drone generated upward thrust!")
else:
    print(f"\n[FALLING] Motor output insufficient for hover (expected with untrained network)")

print(f"\nGenerating plot...")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Real Fruit Fly Brain Controlling Drone - 139k Neurons', fontsize=14, fontweight='bold')

ax = axes[0, 0]
ax.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=2)
ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
ax.plot(positions[-1, 0], positions[-1, 1], 'rx', markersize=12, label='End')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

ax = axes[0, 1]
ax.plot(positions[:, 2], 'b-', linewidth=2.5, label='Altitude')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Start')
ax.set_xlabel('Timestep')
ax.set_ylabel('Z (m)')
ax.set_title('Altitude')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(motor_commands[:, 0], label='Forward', linewidth=2)
ax.plot(motor_commands[:, 1], label='Turn', linewidth=2)
ax.plot(motor_commands[:, 2], label='Climb', linewidth=2)
ax.set_xlabel('Timestep')
ax.set_ylabel('Command')
ax.set_title('Motor Commands from Real Fly Brain')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)

ax = axes[1, 1]
ax.plot(positions[:, 0], label='X', linewidth=2)
ax.plot(positions[:, 1], label='Y', linewidth=2)
ax.plot(positions[:, 2], label='Z', linewidth=2.5, color='red')
ax.set_xlabel('Timestep')
ax.set_ylabel('Position (m)')
ax.set_title('Position Components')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_real_brain.png', dpi=150)
print("[OK] Saved: simulation_real_brain.png\n")
