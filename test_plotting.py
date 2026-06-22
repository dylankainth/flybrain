#!/usr/bin/env python3
"""
Quick test of the live plotting functionality
"""

import time
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque

# Test data buffers
plot_data = {
    'time': deque(maxlen=200),
    'optic_flow_forward': deque(maxlen=200),
    'optic_flow_left': deque(maxlen=200),
    'optic_flow_right': deque(maxlen=200),
    'optic_flow_vertical': deque(maxlen=200),
    'motor_forward': deque(maxlen=200),
    'motor_yaw': deque(maxlen=200),
    'motor_vertical': deque(maxlen=200),
    'spike_count': deque(maxlen=200),
    'dn_activity': deque(maxlen=200),
    'motion_activity': deque(maxlen=200),
}
plot_lock = threading.Lock()
running = True

# Global plot objects
fig = None
axes = None
lines_optic = None
lines_motor = None
lines_neural = None

def init_live_plots():
    """Initialize live plotting"""
    global fig, axes, lines_optic, lines_motor, lines_neural, running
    
    try:
        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        fig.canvas.manager.set_window_title('FlyBrain Live Telemetry - TEST')
        
        ax_optic = axes[0]
        ax_motor = axes[1]
        ax_neural = axes[2]
        
        ax_optic.set_title('Optic Flow Sensors', fontsize=12, fontweight='bold')
        ax_optic.set_ylabel('Flow Magnitude (0-1)')
        ax_optic.set_ylim(0, 1.0)
        ax_optic.set_xlim(0, 20)
        ax_optic.grid(True, alpha=0.3)
        
        ax_motor.set_title('Motor Commands', fontsize=12, fontweight='bold')
        ax_motor.set_ylabel('Command Value')
        ax_motor.set_ylim(-100, 100)
        ax_motor.set_xlim(0, 20)
        ax_motor.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_motor.grid(True, alpha=0.3)
        
        ax_neural.set_title('Neural Activity', fontsize=12, fontweight='bold')
        ax_neural.set_xlabel('Time (s)')
        ax_neural.set_ylabel('Spike Count')
        ax_neural.set_ylim(0, 2000)
        ax_neural.set_xlim(0, 20)
        ax_neural.grid(True, alpha=0.3)
        
        lines_optic = [
            ax_optic.plot([], [], label='Forward', color='blue', linewidth=2)[0],
            ax_optic.plot([], [], label='Left', color='green', linewidth=2)[0],
            ax_optic.plot([], [], label='Right', color='red', linewidth=2)[0],
            ax_optic.plot([], [], label='Vertical', color='purple', linewidth=2)[0],
        ]
        
        lines_motor = [
            ax_motor.plot([], [], label='Forward', color='blue', linewidth=2)[0],
            ax_motor.plot([], [], label='Yaw', color='orange', linewidth=2)[0],
            ax_motor.plot([], [], label='Vertical', color='green', linewidth=2)[0],
        ]
        
        lines_neural = [
            ax_neural.plot([], [], label='Total Spikes', color='red', linewidth=2)[0],
            ax_neural.plot([], [], label='DN Activity', color='blue', linewidth=2)[0],
            ax_neural.plot([], [], label='Motion Activity', color='green', linewidth=2)[0],
        ]
        
        ax_optic.legend(loc='upper right', fontsize=9)
        ax_motor.legend(loc='upper right', fontsize=9)
        ax_neural.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        print("[OK] Plot initialized")
        
        # Update loop
        while running:
            try:
                update_plots()
                time.sleep(0.1)
            except Exception as e:
                print(f"[PLOT ERROR] {e}")
                break
                
    except Exception as e:
        print(f"[PLOT INIT ERROR] {e}")


def update_plots():
    """Update plots with latest data"""
    global plot_data, plot_lock, lines_optic, lines_motor, lines_neural, axes, fig
    
    if fig is None or axes is None:
        return
    
    with plot_lock:
        if len(plot_data['time']) < 2:
            return
        
        times = np.array(plot_data['time'])
        
        if lines_optic is not None:
            lines_optic[0].set_data(times, np.array(plot_data['optic_flow_forward']))
            lines_optic[1].set_data(times, np.array(plot_data['optic_flow_left']))
            lines_optic[2].set_data(times, np.array(plot_data['optic_flow_right']))
            lines_optic[3].set_data(times, np.array(plot_data['optic_flow_vertical']))
        
        if lines_motor is not None:
            lines_motor[0].set_data(times, np.array(plot_data['motor_forward']))
            lines_motor[1].set_data(times, np.array(plot_data['motor_yaw']))
            lines_motor[2].set_data(times, np.array(plot_data['motor_vertical']))
        
        if lines_neural is not None:
            lines_neural[0].set_data(times, np.array(plot_data['spike_count']))
            lines_neural[1].set_data(times, np.array(plot_data['dn_activity']))
            lines_neural[2].set_data(times, np.array(plot_data['motion_activity']))
        
        if len(times) > 0:
            x_min = max(0, times[-1] - 20)
            x_max = times[-1] + 1
            for ax in axes:
                ax.set_xlim(x_min, x_max)
    
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    except:
        pass


def generate_test_data():
    """Generate test data to simulate drone flight"""
    global plot_data, plot_lock, running
    
    t = 0
    while running:
        t += 0.05
        
        # Simulate optic flow with sine waves
        with plot_lock:
            plot_data['time'].append(t)
            plot_data['optic_flow_forward'].append(0.5 + 0.3 * np.sin(t * 0.5))
            plot_data['optic_flow_left'].append(0.3 + 0.2 * np.sin(t * 0.8))
            plot_data['optic_flow_right'].append(0.3 + 0.2 * np.cos(t * 0.8))
            plot_data['optic_flow_vertical'].append(0.2 + 0.1 * np.sin(t * 0.3))
            
            # Simulate motor commands
            plot_data['motor_forward'].append(30 * np.sin(t * 0.4))
            plot_data['motor_yaw'].append(20 * np.cos(t * 0.6))
            plot_data['motor_vertical'].append(15 * np.sin(t * 0.2))
            
            # Simulate neural activity
            plot_data['spike_count'].append(800 + 400 * np.sin(t * 0.5))
            plot_data['dn_activity'].append(100 + 50 * np.sin(t * 0.7))
            plot_data['motion_activity'].append(300 + 150 * np.cos(t * 0.9))
        
        time.sleep(0.05)


def main():
    """Test the plotting"""
    global running
    
    print("=" * 60)
    print("TESTING LIVE PLOTTING")
    print("=" * 60)
    print("\nStarting plot thread...")
    
    plot_thread = threading.Thread(target=init_live_plots, daemon=True)
    plot_thread.start()
    time.sleep(0.5)
    
    print("Starting data generation...")
    data_thread = threading.Thread(target=generate_test_data, daemon=True)
    data_thread.start()
    
    print("\nPlot should be displaying simulated data.")
    print("Press Ctrl+C to stop...\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOPPING] Shutting down...")
        running = False
        time.sleep(1)
        print("[DONE]")


if __name__ == "__main__":
    main()
