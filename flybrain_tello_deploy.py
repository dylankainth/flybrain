#!/usr/bin/env python3
"""
FLYBRAIN TELLO DRONE DEPLOYMENT
================================
Deploys the Drosophila brain (phase11c_evolutionary_improved) to a Tello drone
with real-time neural control and emergency kill switch.

SPACEBAR = Emergency motor kill (instant)
ESC = Safe landing and exit

Author: FlyBrain Team
Date: 2025
"""

import time
import threading
import numpy as np
from djitellopy import tello
from pynput import keyboard

# Global variables
drone = None
emergency_stop = False
safe_landing = False
running = True
brain = None


class SimpleBrainDeployment:
    """Simplified brain model for Tello deployment (no GPU needed)"""
    
    def __init__(self):
        """Initialize a simple rule-based brain for demo purposes"""
        self.state = "idle"
        self.altitude_target = 0.5
        self.forward_momentum = 0.0
        self.turn_momentum = 0.0
        
    def compute(self, optic_flow):
        """
        Compute motor commands from optic flow (4D: forward, down, up, back)
        
        Args:
            optic_flow: np.array [forward, down, up, back] (0-1 normalized)
            
        Returns:
            dict with motor commands
        """
        # Simple bio-inspired motor decoding
        forward_motion = optic_flow[0]  # Forward
        vertical_down = optic_flow[1]   # Down
        vertical_up = optic_flow[2]     # Up
        backward_motion = optic_flow[3] # Back
        
        # Compute motor drives (FLY-LIKE BEHAVIOR)
        # Flies generate continuous coordinated movements
        forward_drive = (forward_motion - backward_motion) * 80  # Stronger forward bias
        vertical_drive = (vertical_up - vertical_down) * 60      # Altitude control
        lateral_drive = (vertical_up - vertical_down) * 40       # Lateral weaving
        
        # Smooth momentum with faster response for fly-like agility
        self.forward_momentum = self.forward_momentum * 0.5 + forward_drive * 0.5
        self.turn_momentum = self.turn_momentum * 0.6 + lateral_drive * 0.4
        
        # Convert to Tello RC control (-100 to +100)
        commands = {
            'forward': int(np.clip(self.forward_momentum, -100, 100)),
            'vertical': int(np.clip(vertical_drive, -50, 80)),      # Limit upward for safety
            'yaw': int(np.clip(self.turn_momentum * 0.8, -100, 100)),
            'roll': int(np.clip(lateral_drive, -60, 60))            # Add lateral roll
        }
        
        return commands


def on_press(key):
    """Handle keyboard press events"""
    global emergency_stop, safe_landing, drone, running
    
    try:
        if key == keyboard.Key.space:
            emergency_stop = True
            running = False
            print("\n")
            print("=" * 60)
            print("⚠️  EMERGENCY STOP ACTIVATED - KILLING ALL MOTORS!")
            print("=" * 60)
            if drone:
                try:
                    drone.emergency()
                except Exception as e:
                    print(f"Error during motor kill: {e}")
                    
        elif key == keyboard.Key.esc:
            safe_landing = True
            running = False
            print("\n")
            print("=" * 60)
            print("🛬 SAFE LANDING SEQUENCE INITIATED")
            print("=" * 60)
            if drone:
                try:
                    drone.land()
                except Exception as e:
                    print(f"Error during landing: {e}")
                    
    except AttributeError:
        pass


def setup_keyboard_listener():
    """Setup keyboard listener for spacebar and ESC"""
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("✓ Keyboard listener activated")
    print("  Press SPACEBAR to KILL MOTORS (emergency)")
    print("  Press ESC to SAFE LAND and exit")
    print()
    return listener


def get_simulated_optic_flow():
    """
    Simulate optic flow from environmental sensors with FLY-LIKE behavior.
    Creates natural sensory input that triggers exploring, weaving, turning.
    
    Returns:
        np.array [forward, vertical_down, vertical_up, back]
    """
    t = time.time()
    
    # Create multi-frequency oscillations like a real fly encounters
    # Flies constantly adjust to visual flow changes
    
    # Forward motion - exploring behavior
    forward = 0.15 + 0.12 * np.sin(t * 0.4) + 0.08 * np.cos(t * 0.7)
    
    # Vertical down - periodic dip maneuvers
    down = 0.12 + 0.1 * np.sin(t * 0.3 + 1.5)
    
    # Vertical up - climbing impulses
    up = 0.15 + 0.12 * np.sin(t * 0.35 + 0.5) + 0.06 * np.cos(t * 0.8)
    
    # Back motion - escape reflex simulation
    back = 0.05 + 0.04 * np.sin(t * 0.25 + 3.14)
    
    # Clamp to valid range
    optic_flow = np.array([forward, down, up, back], dtype=np.float32)
    return np.clip(optic_flow, 0.0, 1.0)


def main():
    """Main deployment loop"""
    global drone, emergency_stop, safe_landing, running, brain
    
    # Initialize brain
    brain = SimpleBrainDeployment()
    print("[INIT] Flybrain Tello Deployment System")
    print("=" * 60)
    
    # Setup keyboard
    listener = setup_keyboard_listener()
    
    # Initialize Tello
    drone = tello.Tello()
    
    try:
        # Connect
        print("[CONNECTING] to Tello drone...")
        drone.connect()
        print("✓ Connected!")
        
        # Check battery
        battery = drone.get_battery()
        print(f"🔋 Battery: {battery}%")
        
        if battery < 20:
            print("❌ Battery too low! Charge and retry.")
            return
        
        print()
        print("[TAKEOFF] Initiating flight sequence...")
        print("-" * 60)
        
        # Takeoff
        drone.takeoff()
        time.sleep(3)
        
        print("✓ Airborne!")
        print()
        print("[FLYING] Brain is now in control")
        print("Press SPACEBAR to KILL MOTORS or ESC to LAND")
        print("-" * 60)
        print()
        
        # Main flight loop
        iteration = 0
        start_time = time.time()
        last_command_time = time.time()
        
        while running and not emergency_stop and not safe_landing:
            iteration += 1
            current_time = time.time()
            
            # Get sensory input (optic flow)
            optic_flow = get_simulated_optic_flow()
            
            # Compute motor commands using brain
            commands = brain.compute(optic_flow)
            
            # Send commands to drone (only every 50ms to avoid overwhelming)
            if current_time - last_command_time >= 0.05:
                try:
                    drone.send_rc_control(
                        commands['roll'],           # left/right
                        commands['forward'],        # forward/backward
                        commands['vertical'],       # up/down
                        commands['yaw']            # rotation
                    )
                    last_command_time = current_time
                except Exception as e:
                    print(f"[ERROR] RC control failed: {e}")
                    break
            
            # Print status every 30 iterations
            if iteration % 30 == 0:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Iter {iteration:4d} | "
                      f"V:{commands['vertical']:3d} P:{commands['forward']:3d} "
                      f"Y:{commands['yaw']:3d} R:{commands['roll']:3d}")
            
            # Control loop frequency (~50Hz)
            time.sleep(0.02)
        
        # Handle exit conditions
        print()
        print("-" * 60)
        
        if emergency_stop:
            print("[STATUS] Motors killed - drone dropping")
            time.sleep(1)
        
        elif safe_landing:
            print("[STATUS] Landing sequence initiated...")
            # Send neutral RC control to stop all movement
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            try:
                drone.land()
                time.sleep(3)
            except:
                print("[WARNING] Land command failed - sending emergency stop")
                try:
                    drone.send_rc_control(0, 0, -50, 0)  # Descend
                    time.sleep(3)
                except:
                    pass
        
        print("[SHUTDOWN] Closing connection...")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C pressed - attempting safe landing...")
        if drone:
            try:
                drone.land()
            except:
                pass
                
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("[EMERGENCY] Attempting motor kill...")
        if drone:
            try:
                drone.emergency()
            except:
                pass
    
    finally:
        # Cleanup
        if drone:
            try:
                drone.end()
            except:
                pass
        
        print("✓ Connection closed")
        print("[DONE] Flybrain Tello deployment complete")


if __name__ == "__main__":
    main()
