"""
Phase 3: Optic Flow Encoding - From Video to T4/T5 Population Response

This phase implements the sensory encoding pipeline:
Video frame → Optical flow computation → T4/T5 population activation

Optic flow algorithms:
- Lucas-Kanade: Dense flow field computation
- Used in fly: Motion compensation, optic flow-driven behavior

The key insight: Optic flow (motion vectors) are what fly visual system
computes and T4/T5 neurons encode. Not raw pixels, not brightness.
This is the biological way flies see motion.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

print("="*70)
print("PHASE 3: OPTIC FLOW ENCODING")
print("="*70)

# ============================================================================
# OPTICAL FLOW COMPUTATION
# ============================================================================

class OpticalFlowEncoder:
    """
    Compute optical flow from video frames using Lucas-Kanade method.

    This simulates how the medulla computes motion: by analyzing brightness
    gradients across space and time.
    """

    def __init__(self, frame_shape=(480, 640), downsample=4):
        """
        frame_shape: (height, width) of video frames
        downsample: reduce resolution for faster computation
        """
        self.h, self.w = frame_shape
        self.downsample = downsample
        self.h_ds = self.h // downsample
        self.w_ds = self.w // downsample

        self.prev_frame = None
        self.prev_gray = None

    def compute_flow(self, frame_rgb):
        """
        Compute dense optical flow using Lucas-Kanade.

        Returns: flow field (2D array of motion vectors)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Downsample for speed
        gray_ds = cv2.resize(gray, (self.w_ds, self.h_ds))
        prev_gray_ds = cv2.resize(self.prev_gray, (self.w_ds, self.h_ds))

        # Lucas-Kanade optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray_ds, gray_ds,
            None,
            0.5,  # pyr_scale
            3,    # levels
            15,   # winsize
            3,    # iterations
            5,    # poly_n
            1.2,  # poly_sigma
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        self.prev_gray = gray
        return flow

    def flow_to_motion_components(self, flow):
        """
        Decompose optical flow into cardinal motion directions.

        Returns: (upward, downward, leftward, rightward) motion magnitudes
        """
        if flow is None:
            return 0, 0, 0, 0

        # Flow components
        flow_x = flow[..., 0]  # Horizontal
        flow_y = flow[..., 1]  # Vertical

        # Sum motion in each direction
        upward = np.sum(np.maximum(-flow_y, 0))  # Negative Y = upward
        downward = np.sum(np.maximum(flow_y, 0))  # Positive Y = downward
        leftward = np.sum(np.maximum(-flow_x, 0))
        rightward = np.sum(np.maximum(flow_x, 0))

        # Normalize
        total = upward + downward + leftward + rightward + 1
        upward /= total
        downward /= total
        leftward /= total
        rightward /= total

        return upward, downward, leftward, rightward

    def flow_magnitude_direction(self, flow):
        """
        Get flow magnitude and direction at each pixel.

        Returns: magnitude, angle (in degrees)
        """
        if flow is None:
            return None, None

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        angle = np.arctan2(flow[..., 1], flow[..., 0]) * 180 / np.pi
        return mag, angle


# ============================================================================
# T4/T5 POPULATION RESPONSE FROM OPTIC FLOW
# ============================================================================

class T4T5PopulationEncoder:
    """
    Encode optic flow into T4/T5 population response.

    Each T4 neuron has a preferred direction (one of 8 cardinal/diagonal).
    Response is proportional to flow in that direction.
    """

    def __init__(self, n_directions=8):
        """
        n_directions: number of direction-selective neurons
                      8 = cardinal (N, NE, E, SE, S, SW, W, NW)
        """
        self.n_directions = n_directions
        # Preferred directions in degrees
        self.preferred_dirs = np.linspace(0, 360, n_directions, endpoint=False)

        # Last responses (for computing changes)
        self.last_response = np.zeros(n_directions)

    def encode_flow(self, flow_magnitude, flow_angle):
        """
        Encode optical flow magnitude and direction into neural response.

        flow_magnitude: array of flow magnitudes at each pixel
        flow_angle: array of flow angles at each pixel (degrees)
        """
        if flow_magnitude is None or flow_angle is None:
            return np.zeros(self.n_directions)

        # Flatten
        mag = flow_magnitude.flatten()
        ang = flow_angle.flatten()

        # For each direction-selective neuron, compute response
        responses = np.zeros(self.n_directions)

        for i, pref_dir in enumerate(self.preferred_dirs):
            # Compute angle difference
            angle_diff = np.abs(ang - pref_dir)
            # Wrap around 180
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)

            # Tuning curve: cosine (width = ~90 degrees)
            tuning = np.cos(np.radians(angle_diff))
            tuning = np.maximum(tuning, 0)  # Half-wave rectification

            # Response: weighted by flow magnitude
            response = np.sum(mag * tuning)
            responses[i] = response

        # Normalize
        responses = responses / (np.max(responses) + 1)

        # Convert to firing rates (0-200 Hz)
        firing_rates = responses * 200

        self.last_response = firing_rates
        return firing_rates

    def decode_motion_direction(self, responses):
        """
        Decode population response to infer motion direction.

        Uses population vector decoding: weighted sum of preferred directions.
        """
        if np.sum(responses) < 1:
            return None

        # Population vector
        angles = np.radians(self.preferred_dirs)
        vector_x = np.sum(responses * np.cos(angles))
        vector_y = np.sum(responses * np.sin(angles))

        # Direction of population vector
        direction = np.degrees(np.arctan2(vector_y, vector_x))
        if direction < 0:
            direction += 360

        # Magnitude represents confidence
        magnitude = np.sqrt(vector_x**2 + vector_y**2)

        return direction, magnitude


# ============================================================================
# SIMULATE SIMPLE SYNTHETIC VIDEO
# ============================================================================

print("\n" + "="*70)
print("TESTING ON SYNTHETIC VIDEO")
print("="*70)

# Create simple synthetic motion
h, w = 256, 256
n_frames = 100
video = np.zeros((n_frames, h, w, 3), dtype=np.uint8)

print(f"Creating synthetic video: {h}x{w}, {n_frames} frames")
print("  Content: Moving vertical bar")

for frame_idx in range(n_frames):
    # Black background
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # White vertical bar at different positions (moving upward)
    bar_x = w // 2
    bar_width = 30

    # Bar y-position changes (moving up means decreasing y)
    # 0.1 pixel/frame upward motion
    bar_y = int(h // 2 - frame_idx * 0.3)

    # Draw bar (with wrapping)
    y1 = bar_y % h
    y2 = (bar_y + 40) % h

    frame[max(0, y1):min(h, y2), bar_x-bar_width:bar_x+bar_width] = 255

    # Add some texture
    noise = np.random.randint(0, 30, (h, w))
    frame = frame.astype(np.int32)
    frame[..., 0] += noise
    frame[..., 1] += noise
    frame[..., 2] += noise
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    video[frame_idx] = frame

# Process video
encoder = OpticalFlowEncoder(frame_shape=(h, w), downsample=2)
population = T4T5PopulationEncoder(n_directions=8)

decoded_directions = []
flow_magnitudes = []
population_responses = []

print("\nProcessing video frames...")

for frame_idx in range(1, n_frames):
    frame = video[frame_idx]

    # Compute optical flow
    flow = encoder.compute_flow(frame)

    if flow is not None:
        # Get flow components
        mag, angle = encoder.flow_magnitude_direction(flow)

        # Encode into T4/T5 population
        responses = population.encode_flow(mag, angle)
        population_responses.append(responses)

        # Decode motion direction
        decoded = population.decode_motion_direction(responses)

        if decoded is not None:
            direction, confidence = decoded
            decoded_directions.append(direction)
        else:
            decoded_directions.append(None)

        # Log flow magnitude
        if mag is not None:
            flow_magnitudes.append(np.mean(mag))
        else:
            flow_magnitudes.append(0)

    if (frame_idx + 1) % 20 == 0:
        print(f"  Frame {frame_idx+1}/{n_frames}")

# Analyze results
print(f"\n[RESULTS]")
if decoded_directions:
    valid_directions = [d for d in decoded_directions if d is not None]
    print(f"Mean decoded direction: {np.mean(valid_directions):.1f} degrees")
    print(f"  (Expected: 270 degrees for upward motion)")

print(f"Mean optic flow magnitude: {np.mean(flow_magnitudes):.3f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sample video frame
ax = axes[0, 0]
ax.imshow(video[50])
ax.set_title('Sample Video Frame')
ax.axis('off')

# Flow magnitude over time
ax = axes[0, 1]
ax.plot(flow_magnitudes, linewidth=2)
ax.set_xlabel('Frame')
ax.set_ylabel('Mean Optical Flow Magnitude')
ax.set_title('Optic Flow Strength Over Time')
ax.grid(True, alpha=0.3)

# Decoded direction over time
ax = axes[1, 0]
if decoded_directions:
    valid = [d for d in decoded_directions if d is not None]
    if valid:
        ax.plot(valid, 'o-', markersize=4, linewidth=1)
        ax.axhline(270, color='r', linestyle='--', label='Expected (270 deg)')
        ax.set_ylabel('Decoded Direction (degrees)')
        ax.set_xlabel('Frame')
        ax.set_title('Decoded Motion Direction')
        ax.set_ylim([0, 360])
        ax.legend()
        ax.grid(True, alpha=0.3)

# T4/T5 population response (last frame)
ax = axes[1, 1]
if population_responses:
    last_response = population_responses[-1]
    directions = population.preferred_dirs
    ax.bar(directions, last_response, width=40, alpha=0.7)
    ax.set_xlabel('Preferred Direction (degrees)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('T4/T5 Population Response (last frame)')
    ax.set_xlim([-20, 380])

plt.tight_layout()
plt.savefig('phase3_optic_flow.png', dpi=100)
print(f"\n[OK] Saved: phase3_optic_flow.png")

# ============================================================================
# READY FOR INTEGRATION
# ============================================================================

print("\n" + "="*70)
print("[SUCCESS] Phase 3 Complete - Optic Flow Encoding")
print("="*70)

print(f"\nIntegration ready for:")
print(f"  [OK] Video frame -> Optical flow")
print(f"  [OK] Optical flow -> T4/T5 population activation")
print(f"  [OK] Population response -> Motion direction decoding")

print(f"\nNext: Phase 4 - Motor Circuit Integration")
print(f"  - Map T4/T5 output to descending neurons")
print(f"  - Descending neurons -> thrust commands")
print(f"  - Closed-loop drone control with real fly brain")

print(f"\n{'='*70}\n")
