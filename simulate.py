"""
Main loop: FlyBrain → DroneSimulator feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
from fly_brain import FlyBrain
from drone_sim import DroneSimulator
from io_mapping import SensoryEncoder, MotorDecoder

def run_simulation(duration_sec=60, dt_ms=50):
    """
    Run complete sim with fly brain driving drone.

    Args:
        duration_sec: Total simulation duration
        dt_ms: Timestep in milliseconds
    """

    print("="*60)
    print("FRUIT FLY BRAIN DRONE SIMULATOR")
    print("="*60)

    # Initialize
    print("\n[1/4] Initializing FlyBrain...")
    brain = FlyBrain(
        synapse_file='fly_synapses.csv',
        neuron_file='fly_neurons.csv',
        verbose=True
    )

    print("\n[2/4] Initializing DroneSimulator...")
    sim = DroneSimulator(gui=True, dt=dt_ms/1000)

    print("\n[3/4] Initializing I/O mappings...")
    encoder = SensoryEncoder()
    decoder = MotorDecoder(
        gain_forward=50,
        gain_turn=30,
        gain_climb=20
    )

    # Logging
    print("\n[4/4] Starting simulation loop...")

    timesteps = int(duration_sec * 1000 / dt_ms)

    positions = []
    motor_commands = []
    neural_activity = []

    try:
        for step in range(timesteps):
            # 1. Get sensor readings
            obs = sim.get_observation()

            # 2. Encode vision → sensory input
            sensory_input = encoder.encode_vision(obs['vision'])

            # 3. Inject into brain
            brain.inject_sensory(sensory_input)

            # 4. Step brain
            brain.step(duration_ms=dt_ms)

            # 5. Decode motor output
            forward, turn, climb = decoder.decode(brain, window_ms=dt_ms)

            # 6. Apply to drone
            sim.apply_motor_command(forward, turn, climb)

            # 7. Physics step
            sim.step()

            # 8. Log
            positions.append(obs['position'])
            motor_commands.append([forward, turn, climb])
            neural_activity.append({
                'forward_rate': decoder.FORWARD_NEURONS,
                'turn_rate': turn,
            })

            # 9. Status
            if (step + 1) % 100 == 0:
                elapsed = (step + 1) * dt_ms / 1000
                print(f"  [{elapsed:6.1f}s] pos={obs['position'][:2]}, "
                      f"cmd=[f:{forward:+.2f}, t:{turn:+.2f}, c:{climb:+.2f}]")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")

    # Cleanup
    sim.close()

    # Plot results
    print("\n[✓] Simulation complete. Plotting...")
    plot_results(positions, motor_commands)

    return positions, motor_commands


def plot_results(positions, motor_commands):
    """Visualize drone trajectory and motor commands"""

    positions = np.array(positions)
    motor_commands = np.array(motor_commands)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectory (XY plane)
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=1)
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'rx', markersize=10, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Drone Trajectory (Top-Down)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Altitude
    ax = axes[0, 1]
    ax.plot(positions[:, 2], 'b-', linewidth=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Z (m)')
    ax.set_title('Altitude')
    ax.grid(True, alpha=0.3)

    # Motor commands over time
    ax = axes[1, 0]
    ax.plot(motor_commands[:, 0], label='Forward', alpha=0.7)
    ax.plot(motor_commands[:, 1], label='Turn', alpha=0.7)
    ax.plot(motor_commands[:, 2], label='Climb', alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Command (normalized)')
    ax.set_title('Motor Commands from Fly Brain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    # Position over time
    ax = axes[1, 1]
    ax.plot(positions[:, 0], label='X', alpha=0.7)
    ax.plot(positions[:, 1], label='Y', alpha=0.7)
    ax.plot(positions[:, 2], label='Z', alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150)
    print("  Saved: simulation_results.png")
    plt.show()


if __name__ == "__main__":
    positions, commands = run_simulation(
        duration_sec=60,
        dt_ms=50
    )
