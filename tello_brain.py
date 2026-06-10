"""
Deploy fly brain to real DJI Tello drone.
This is Step 6 - run only after simulation works well.
"""

import cv2
import numpy as np
from djitellopy import Tello
from fly_brain import FlyBrain
from io_mapping import SensoryEncoder, MotorDecoder
import time

class TelloDrone:
    """Fly brain controller for DJI Tello"""

    def __init__(self):
        """Connect to Tello and initialize brain"""

        print("[1/3] Connecting to Tello...")
        self.tello = Tello()
        self.tello.connect()
        print(f"  Battery: {self.tello.get_battery()}%")

        print("[2/3] Initializing fly brain...")
        self.brain = FlyBrain(verbose=False)

        print("[3/3] Setting up I/O...")
        self.encoder = SensoryEncoder()
        self.decoder = MotorDecoder()

        # Camera
        self.tello.streamon()

        # Command scaling
        self.speed_scale = 50  # cm/s max
        self.turn_scale = 45   # deg/s max

    def encode_camera_frame(self, frame):
        """Convert real camera frame → visual neurons"""

        # Resize for processing
        frame = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Split into left/center/right regions
        left_half = gray[:, :32].mean()
        right_half = gray[:, 32:].mean()
        center = gray[16:48, 16:48].mean()

        # Normalize to [0, 255]
        vision = np.array([left_half, center, right_half])

        return vision

    def send_motor_command(self, forward, turn, climb):
        """Send normalized commands to Tello"""

        # Scale to Tello command ranges
        # Tello: move_forward(cm), turn_clockwise(deg), move_up(cm)
        forward_cm = int(forward * self.speed_scale)
        turn_deg = int(turn * self.turn_scale)
        climb_cm = int(climb * self.speed_scale / 2)

        # Safety: don't send zero commands constantly
        if abs(forward_cm) > 5:
            if forward_cm > 0:
                self.tello.move_forward(forward_cm)
            else:
                self.tello.move_back(-forward_cm)

        if abs(turn_deg) > 3:
            if turn_deg > 0:
                self.tello.turn_clockwise(turn_deg)
            else:
                self.tello.turn_counter_clockwise(-turn_deg)

        if abs(climb_cm) > 5:
            if climb_cm > 0:
                self.tello.move_up(climb_cm)
            else:
                self.tello.move_down(-climb_cm)

    def run(self, duration_sec=30):
        """Fly drone for specified duration"""

        print(f"\n[!] Taking off...")
        self.tello.takeoff()
        time.sleep(2)

        frame_read = self.tello.get_frame_read()
        dt = 0.05  # 50 ms loop

        start_time = time.time()
        step = 0

        try:
            while time.time() - start_time < duration_sec:
                # 1. Capture frame
                frame = frame_read.frame
                if frame is None:
                    continue

                # 2. Encode vision
                vision = self.encode_camera_frame(frame)
                sensory_input = self.encoder.encode_vision(vision)

                # 3. Step brain
                self.brain.inject_sensory(sensory_input)
                self.brain.step(duration_ms=50)

                # 4. Decode motor
                forward, turn, climb = self.decoder.decode(self.brain)

                # 5. Send to Tello
                self.send_motor_command(forward, turn, climb)

                # Status
                if step % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"  [{elapsed:6.1f}s] cmd=[f:{forward:+.2f}, "
                          f"t:{turn:+.2f}, c:{climb:+.2f}]")

                step += 1
                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n[!] Interrupted")

        finally:
            print("\n[!] Landing...")
            self.tello.land()
            self.tello.streamoff()


if __name__ == "__main__":
    drone = TelloDrone()
    drone.run(duration_sec=30)
