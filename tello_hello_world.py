#!/usr/bin/env python3
"""
Tello Drone Hello World - Basic movement example with spacebar emergency stop
This script connects to a Tello drone and performs simple up/down movements.
Press SPACEBAR to emergency stop and land the drone immediately.
"""

import time
import threading
from djitellopy import tello
from pynput import keyboard

# Global variables for drone control
me = None
emergency_stop = False


def on_press(key):
    """Handle keyboard press events"""
    global emergency_stop, me
    try:
        if key == keyboard.Key.space:
            emergency_stop = True
            print("\n⚠️  EMERGENCY STOP ACTIVATED - KILLING MOTORS!")
            if me:
                try:
                    # Send emergency command to kill motors
                    me.emergency()
                except:
                    pass
    except AttributeError:
        pass


def setup_keyboard_listener():
    """Setup keyboard listener for spacebar"""
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def main():
    global me, emergency_stop
    
    # Create a Tello object
    me = tello.Tello()
    
    # Setup keyboard listener
    print("Starting keyboard listener... Press SPACEBAR anytime to emergency stop!")
    listener = setup_keyboard_listener()
    
    try:
        # Connect to the drone
        print("Connecting to Tello drone...")
        me.connect()
        print("Connected!")
        
        # Get battery status
        battery = me.get_battery()
        print(f"Battery: {battery}%")
        
        if battery < 20:
            print("Warning: Battery is low!")
            return
        
        # Take off
        print("Taking off...")
        me.takeoff()
        time.sleep(2)  # Wait for takeoff to complete
        
        if emergency_stop:
            raise Exception("Emergency stop activated during takeoff")
        
        # Move up
        print("Moving up...")
        me.send_rc_control(0, 0, 50, 0)  # up 50
        time.sleep(2)
        me.send_rc_control(0, 0, 0, 0)  # stop
        
        if emergency_stop:
            raise Exception("Emergency stop activated")
        
        # Move down
        print("Moving down...")
        me.send_rc_control(0, 0, -50, 0)  # down 50
        time.sleep(2)
        me.send_rc_control(0, 0, 0, 0)  # stop
        
        if emergency_stop:
            raise Exception("Emergency stop activated")
        
        # Move up again
        print("Moving up again...")
        me.send_rc_control(0, 0, 30, 0)  # up 30
        time.sleep(2)
        me.send_rc_control(0, 0, 0, 0)  # stop
        
        if emergency_stop:
            raise Exception("Emergency stop activated")
        
        # Move down to return to original position
        print("Moving down to landing position...")
        me.send_rc_control(0, 0, -30, 0)  # down 30
        time.sleep(2)
        me.send_rc_control(0, 0, 0, 0)  # stop
        
        # Land
        print("Landing...")
        me.land()
        time.sleep(2)
        
        print("Hello World flight complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Attempting to land drone...")
        try:
            me.land()
        except:
            pass
    
    finally:
        # End connection
        me.end()
        print("Connection closed.")


if __name__ == "__main__":
    main()
