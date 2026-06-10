"""
Aggressive tuning: ultra-high baseline motor drive + maximum sensory gain
Tests the extreme limits of the synthetic connectome
"""

import numpy as np
import matplotlib.pyplot as plt
from fly_brain import FlyBrain
from drone_sim import DroneSimulator
from io_mapping import SensoryEncoder, MotorDecoder

def run_simulation(duration_sec=10, dt_ms=50):
    """Run sim with AGGRESSIVE tuning"""

    print("="*60)
    print("AGGRESSIVE TUNING: Maximum Sensory + Motor Drive")
    print("="*60)

    print("\n[1/4] Initializing FlyBrain...")
    brain = FlyBrain(
        synapse_file='fly_synapses.csv',
        neuron_file='fly_neurons.csv',
        verbose=False
    )

    print("[2/4] Initializing DroneSimulator...")
    sim = DroneSimulator(gui=False, dt=dt_ms/1000)

    print("[3/4] Initializing encoders/decoders...")
    encoder = SensoryEncoder()
    decoder = MotorDecoder(
        gain_forward=10,    # VERY sensitive (was 50)
        gain_turn=10,       # VERY sensitive (was 30)
        gain_climb=10       # VERY sensitive (was 20)
    )

    print("[4/4] Aggressive configuration:")
    print("      - Sensory gain: 100x (was 10x)")
    print("      - Motor baseline: 1000 pA (was 300 pA)")
    print("      - Motor decoder gains: 10x more sensitive")

    # AGGRESSIVE baseline drive
    all_motor_neurons = (
        decoder.FORWARD_NEURONS +
        decoder.LEFT_TURN_NEURONS +
        decoder.RIGHT_TURN_NEURONS +
        decoder.CLIMB_NEURONS
    )
    brain.set_baseline_motor_drive(all_motor_neurons, drive_strength=1000)

    print("\n[+] Starting 10-second flight test...")

    timesteps = int(duration_sec * 1000 / dt_ms)
    positions = []
    motor_commands = []

    try:
        for step in range(timesteps):
            brain.apply_baseline_motor_drive()

            obs = sim.get_observation()

            # AGGRESSIVE sensory encoding
            sensory_input = {}
            left, center, right = obs['vision'].astype(float)

            # Ultra-aggressive: always inject at maximum
            sensory_input[tuple(encoder.VISUAL_LEFT_IDX)] = 5000      # was 1000
            sensory_input[tuple(encoder.VISUAL_RIGHT_IDX)] = 5000     # was 1000
            sensory_input[tuple(encoder.VISUAL_CENTER_IDX)] = 7500    # was 1500

            brain.inject_sensory(sensory_input)
            brain.step(duration_ms=dt_ms)

            forward, turn, climb = decoder.decode(brain, window_ms=dt_ms)

            sim.apply_motor_command(forward, turn, climb)
            sim.step()

            positions.append(obs['position'])
            motor_commands.append([forward, turn, climb])

            if (step + 1) % 50 == 0:
                elapsed = (step + 1) * dt_ms / 1000
                print(f"  [{elapsed:6.1f}s] alt={obs['position'][2]:.3f}m "
                      f"cmd=[f:{forward:+.3f}, t:{turn:+.3f}, c:{climb:+.3f}]")

    except KeyboardInterrupt:
        print("\n[!] Interrupted")
    finally:
        sim.close()

    positions = np.array(positions)
    motor_commands = np.array(motor_commands)

    print(f"\n[OK] Simulation complete!")
    print(f"\nRESULTS:")
    print(f"  Final altitude: {positions[-1, 2]:.3f}m (started at 1.0m)")
    if positions[-1, 2] > positions[0, 2]:
        print(f"  STATUS: CLIMBING! (+{positions[-1, 2] - positions[0, 2]:.3f}m)")
    elif positions[-1, 2] < 0.1:
        print(f"  STATUS: CRASHED")
    else:
        print(f"  STATUS: FALLING ({positions[0, 2] - positions[-1, 2]:.3f}m drop)")
    print(f"\n  Max motor command: {np.abs(motor_commands).max():.3f}")
    print(f"  Avg forward: {motor_commands[:, 0].mean():.3f}")
    print(f"  Avg turn: {motor_commands[:, 1].mean():.3f}")
    print(f"  Avg climb: {motor_commands[:, 2].mean():.3f}")

    plot_results(positions, motor_commands)
    return positions, motor_commands


def plot_results(positions, motor_commands):
    """Plot results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fruit Fly Brain - AGGRESSIVE TUNING TEST', fontsize=14, fontweight='bold', color='red')

    # Trajectory
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=2, alpha=0.7)
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'rx', markersize=12, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Altitude
    ax = axes[0, 1]
    ax.plot(positions[:, 2], 'b-', linewidth=2.5)
    ax.fill_between(range(len(positions)), positions[:, 2], alpha=0.2, color='blue')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Start')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='Ground')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Z (m)')
    ax.set_title('Altitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Motor commands
    ax = axes[1, 0]
    ax.plot(motor_commands[:, 0], label='Forward', linewidth=2, alpha=0.8)
    ax.plot(motor_commands[:, 1], label='Turn', linewidth=2, alpha=0.8)
    ax.plot(motor_commands[:, 2], label='Climb', linewidth=2, alpha=0.8)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Command')
    ax.set_title('Motor Commands')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    # Position components
    ax = axes[1, 1]
    ax.plot(positions[:, 0], label='X', linewidth=1.5, alpha=0.8)
    ax.plot(positions[:, 1], label='Y', linewidth=1.5, alpha=0.8)
    ax.plot(positions[:, 2], label='Z', linewidth=2, alpha=0.9, color='red')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_aggressive.png', dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved: simulation_aggressive.png")


if __name__ == "__main__":
    positions, commands = run_simulation(duration_sec=10, dt_ms=50)
