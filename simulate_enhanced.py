"""
Enhanced simulator with:
- Option B: Boosted sensory gains (10x higher)
- Option C: Continuous baseline motor drive
"""

import numpy as np
import matplotlib.pyplot as plt
from fly_brain import FlyBrain
from drone_sim import DroneSimulator
from io_mapping import SensoryEncoder, MotorDecoder

def run_simulation(duration_sec=10, dt_ms=50):
    """Run sim with enhanced sensory drive and baseline motor current"""

    print("="*60)
    print("FRUIT FLY BRAIN DRONE SIMULATOR - ENHANCED")
    print("="*60)

    print("\n[1/4] Initializing FlyBrain...")
    brain = FlyBrain(
        synapse_file='fly_synapses.csv',
        neuron_file='fly_neurons.csv',
        verbose=True
    )

    print("\n[2/4] Initializing DroneSimulator...")
    sim = DroneSimulator(gui=False, dt=dt_ms/1000)

    print("\n[3/4] Initializing I/O mappings...")
    print("      [Option B] Sensory gains: 10x boost, threshold lowered to 50")
    encoder = SensoryEncoder()
    decoder = MotorDecoder(
        gain_forward=30,   # Reduced gain to amplify motor response
        gain_turn=20,
        gain_climb=15
    )

    print("\n[4/4] Setting baseline motor drive...")
    print("      [Option C] Continuous 300pA tonic drive to all motor neurons")
    all_motor_neurons = (
        decoder.FORWARD_NEURONS +
        decoder.LEFT_TURN_NEURONS +
        decoder.RIGHT_TURN_NEURONS +
        decoder.CLIMB_NEURONS
    )
    brain.set_baseline_motor_drive(all_motor_neurons, drive_strength=300)

    print("\n[+] Starting 10-second simulation...")

    timesteps = int(duration_sec * 1000 / dt_ms)
    positions = []
    motor_commands = []
    firing_rates = []

    try:
        for step in range(timesteps):
            # Apply baseline motor drive continuously
            brain.apply_baseline_motor_drive()

            # Get sensor readings
            obs = sim.get_observation()

            # Encode vision (Option B: boosted sensitivity)
            sensory_input = encoder.encode_vision(obs['vision'])

            # Inject sensory input and simulate
            brain.inject_sensory(sensory_input)
            brain.step(duration_ms=dt_ms)

            # Decode motor output
            forward, turn, climb = decoder.decode(brain, window_ms=dt_ms)

            # Apply motor commands
            sim.apply_motor_command(forward, turn, climb)
            sim.step()

            # Logging
            positions.append(obs['position'])
            motor_commands.append([forward, turn, climb])

            if (step + 1) % 50 == 0:
                elapsed = (step + 1) * dt_ms / 1000
                print(f"  [{elapsed:6.1f}s] pos=({obs['position'][0]:+.2f}, {obs['position'][1]:+.2f}, {obs['position'][2]:.2f}) "
                      f"cmd=[f:{forward:+.2f}, t:{turn:+.2f}, c:{climb:+.2f}]")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    finally:
        sim.close()

    print("\n[OK] Simulation complete!")
    print(f"\nFinal Results:")
    print(f"  Duration: {duration_sec}s")
    print(f"  Timesteps: {len(positions)}")
    print(f"  Final position: ({positions[-1][0]:.2f}, {positions[-1][1]:.2f}, {positions[-1][2]:.2f})")
    print(f"  Distance traveled (XY): {np.linalg.norm(np.array(positions[-1][:2]) - np.array(positions[0][:2])):.2f}m")
    print(f"  Max altitude: {np.array(positions)[:, 2].max():.2f}m")
    print(f"  Min altitude: {np.array(positions)[:, 2].min():.2f}m")

    plot_results(positions, motor_commands)
    return positions, motor_commands


def plot_results(positions, motor_commands):
    """Visualize drone trajectory and motor commands"""

    positions = np.array(positions)
    motor_commands = np.array(motor_commands)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fruit Fly Brain Drone - Enhanced Control (B+C)', fontsize=14, fontweight='bold')

    # Trajectory (XY plane)
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=2, alpha=0.7, label='Flight path')
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(positions[-1, 0], positions[-1, 1], 'rx', markersize=12, label='End', zorder=5)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Drone Trajectory (Top-Down)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Altitude
    ax = axes[0, 1]
    ax.plot(positions[:, 2], 'b-', linewidth=2.5, label='Altitude')
    ax.fill_between(range(len(positions)), positions[:, 2], alpha=0.2, color='blue')
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Z (m)', fontsize=11)
    ax.set_title('Altitude Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Start height')
    ax.legend(fontsize=10)

    # Motor commands over time
    ax = axes[1, 0]
    ax.plot(motor_commands[:, 0], label='Forward', alpha=0.8, linewidth=1.5)
    ax.plot(motor_commands[:, 1], label='Turn', alpha=0.8, linewidth=1.5)
    ax.plot(motor_commands[:, 2], label='Climb', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Command (normalized)', fontsize=11)
    ax.set_title('Motor Commands from Fly Brain', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

    # Position components over time
    ax = axes[1, 1]
    ax.plot(positions[:, 0], label='X', alpha=0.8, linewidth=1.5)
    ax.plot(positions[:, 1], label='Y', alpha=0.8, linewidth=1.5)
    ax.plot(positions[:, 2], label='Z (altitude)', alpha=0.8, linewidth=2)
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Position (m)', fontsize=11)
    ax.set_title('Position Components', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_enhanced.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: simulation_enhanced.png")


if __name__ == "__main__":
    positions, commands = run_simulation(
        duration_sec=10,
        dt_ms=50
    )
