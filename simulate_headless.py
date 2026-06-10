"""
Headless fruit fly brain drone simulator (no GUI).
"""

import numpy as np
import matplotlib.pyplot as plt
from fly_brain import FlyBrain
from drone_sim import DroneSimulator
from io_mapping import SensoryEncoder, MotorDecoder

def run_simulation(duration_sec=10, dt_ms=50):
    """Run complete sim with fly brain driving drone (headless)"""

    print("="*60)
    print("FRUIT FLY BRAIN DRONE SIMULATOR (HEADLESS)")
    print("="*60)

    print("\n[1/4] Initializing FlyBrain...")
    brain = FlyBrain(
        synapse_file='fly_synapses.csv',
        neuron_file='fly_neurons.csv',
        verbose=True
    )

    print("\n[2/4] Initializing DroneSimulator (headless)...")
    sim = DroneSimulator(gui=False, dt=dt_ms/1000)

    print("\n[3/4] Initializing I/O mappings...")
    encoder = SensoryEncoder()
    decoder = MotorDecoder(
        gain_forward=50,
        gain_turn=30,
        gain_climb=20
    )

    print("\n[4/4] Starting simulation loop...")

    timesteps = int(duration_sec * 1000 / dt_ms)

    positions = []
    motor_commands = []

    try:
        for step in range(timesteps):
            obs = sim.get_observation()
            sensory_input = encoder.encode_vision(obs['vision'])
            brain.inject_sensory(sensory_input)
            brain.step(duration_ms=dt_ms)
            forward, turn, climb = decoder.decode(brain, window_ms=dt_ms)
            sim.apply_motor_command(forward, turn, climb)
            sim.step()

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

    plot_results(positions, motor_commands)
    return positions, motor_commands


def plot_results(positions, motor_commands):
    """Visualize drone trajectory and motor commands"""

    positions = np.array(positions)
    motor_commands = np.array(motor_commands)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectory (XY plane)
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=1, alpha=0.7)
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'rx', markersize=10, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Drone Trajectory (Top-Down View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Altitude
    ax = axes[0, 1]
    ax.plot(positions[:, 2], 'b-', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Z (m)')
    ax.set_title('Altitude Over Time')
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Start height')
    ax.legend()

    # Motor commands over time
    ax = axes[1, 0]
    ax.plot(motor_commands[:, 0], label='Forward', alpha=0.7, linewidth=1.5)
    ax.plot(motor_commands[:, 1], label='Turn', alpha=0.7, linewidth=1.5)
    ax.plot(motor_commands[:, 2], label='Climb', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Command (normalized)')
    ax.set_title('Motor Commands from Fly Brain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

    # Position components over time
    ax = axes[1, 1]
    ax.plot(positions[:, 0], label='X', alpha=0.7, linewidth=1.5)
    ax.plot(positions[:, 1], label='Y', alpha=0.7, linewidth=1.5)
    ax.plot(positions[:, 2], label='Z', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150)
    print("\n[OK] Saved: simulation_results.png")


if __name__ == "__main__":
    positions, commands = run_simulation(
        duration_sec=10,
        dt_ms=50
    )
