"""
Simple quadcopter physics simulator in PyBullet.
"""

import pybullet as p
import pybullet_data
import numpy as np

class DroneSimulator:
    """Simplified quadcopter in PyBullet physics engine"""

    def __init__(self, gui=True, dt=0.01):
        """
        Args:
            gui: Show GUI (True) or headless (False)
            dt: Physics timestep
        """

        # Connect to physics server
        mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Physics parameters
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=dt, numSubSteps=4)
        self.dt = dt

        # Load environment
        p.loadURDF("plane.urdf", [0, 0, 0])

        # Create simple drone body (red box)
        self.drone_mass = 0.5  # kg
        self.drone_size = [0.1, 0.1, 0.05]  # meters

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=self.drone_size
        )

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=self.drone_size,
            rgbaColor=[1, 0, 0, 0.9]
        )

        self.drone_id = p.createMultiBody(
            baseMass=self.drone_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 1.0],
            baseOrientation=[0, 0, 0, 1]
        )

        # Motor parameters
        self.thrust_scale = 0.05  # Force scaling

        # Sensor state
        self.last_obs = None

    def get_observation(self):
        """
        Return current sensor readings.

        Returns:
            dict with:
            - position: [x, y, z]
            - velocity: [vx, vy, vz]
            - orientation: quaternion [x, y, z, w]
            - angular_velocity: [wx, wy, wz]
            - vision: [left_brightness, center_brightness, right_brightness]
            - altitude: height above ground
        """

        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)

        z = pos[2]

        # Mock vision: simulate what drone "sees"
        # In reality this would come from camera feed
        # For now: brightness varies with position (bright near edges)
        vision_left = 50 + int(50 * np.sin(pos[0]))
        vision_right = 50 + int(50 * np.cos(pos[0]))
        vision_center = 100 + int(50 * np.cos(pos[1]))

        obs = {
            'position': np.array(pos),
            'velocity': np.array(vel),
            'orientation': np.array(orn),
            'angular_velocity': np.array(ang_vel),
            'vision': np.array([vision_left, vision_center, vision_right],
                              dtype=np.float32),
            'altitude': z,
        }

        self.last_obs = obs
        return obs

    def apply_motor_command(self, forward, turn, climb):
        """
        Apply motor commands as forces on drone.

        Args:
            forward: [-1, 1] forward/backward thrust
            turn: [-1, 1] yaw rate
            climb: [-1, 1] vertical thrust
        """

        # Clamp inputs
        forward = np.clip(forward, -1, 1)
        turn = np.clip(turn, -1, 1)
        climb = np.clip(climb, -1, 1)

        # Map to 4 motor thrusts
        # Motor layout (viewed from above):
        #   FL -- FR
        #   BL -- BR

        thrust = np.array([
            forward + turn + climb,      # front-left
            forward - turn + climb,      # front-right
            forward + turn - climb,      # back-left
            forward - turn - climb,      # back-right
        ]) * self.thrust_scale

        # Apply upward force at each corner
        for i, t in enumerate(thrust):
            # Small offset to simulate motor positions
            offset = [
                [0.08, 0.08],   # FL
                [0.08, -0.08],  # FR
                [-0.08, 0.08],  # BL
                [-0.08, -0.08]  # BR
            ][i]

            pos = p.getBasePositionAndOrientation(self.drone_id)[0]
            force_pos = [pos[0] + offset[0], pos[1] + offset[1], pos[2]]

            # Apply upward force
            p.applyExternalForce(
                self.drone_id, -1,
                forceObj=[0, 0, t],
                posObj=force_pos,
                flags=p.WORLD_FRAME
            )

    def step(self):
        """Advance physics simulation by one timestep"""
        p.stepSimulation()

    def close(self):
        """Disconnect from physics server"""
        p.disconnect(self.physics_client)

    def reset(self, position=[0, 0, 1.0]):
        """Reset drone to starting position"""
        p.resetBasePositionAndOrientation(
            self.drone_id,
            position,
            [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])
