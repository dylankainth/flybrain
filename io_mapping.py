"""
Map between robot sensors and fly brain neurons.
Also: map fly neural activity to robot motor commands.
"""

import numpy as np

class SensoryEncoder:
    """Convert robot sensors → fly sensory neuron activation"""

    # Mock neuron indices (you'll populate these from FlyWire)
    VISUAL_LEFT_IDX = range(1000, 1500)      # ~500 visual neurons for left
    VISUAL_RIGHT_IDX = range(1500, 2000)     # ~500 for right
    VISUAL_CENTER_IDX = range(2000, 2500)    # ~500 for center/forward

    def encode_vision(self, vision_array, brightness_threshold=120):
        """
        Convert mock vision sensor → visual neuron activation.

        Args:
            vision_array: [left_brightness, center_brightness, right_brightness]
            brightness_threshold: Activation threshold (0-255)

        Returns:
            dict: {neuron_indices: activation (amp)}
        """

        left, center, right = vision_array.astype(float)

        sensory_input = {}

        # Left visual neurons fire if left is bright
        if left > brightness_threshold:
            activation = (left - brightness_threshold) / (255 - brightness_threshold)
            sensory_input[self.VISUAL_LEFT_IDX] = activation * 100  # pA

        # Right visual neurons fire if right is bright
        if right > brightness_threshold:
            activation = (right - brightness_threshold) / (255 - brightness_threshold)
            sensory_input[self.VISUAL_RIGHT_IDX] = activation * 100  # pA

        # Forward neurons fire if center is bright
        if center > brightness_threshold:
            activation = (center - brightness_threshold) / (255 - brightness_threshold)
            sensory_input[self.VISUAL_CENTER_IDX] = activation * 150  # pA

        return sensory_input


class MotorDecoder:
    """Map fly descending neurons → robot motor commands"""

    # Motor neuron indices (from published fly neuroscience)
    # These are "control neurons" that regulate behavior
    FORWARD_NEURONS = range(120000, 120200)   # oDN1 and related
    LEFT_TURN_NEURONS = range(120200, 120400) # DNa02 and related
    RIGHT_TURN_NEURONS = range(120400, 120600) # DNa01 and related
    CLIMB_NEURONS = range(120600, 120800)     # (hypothetical)

    def __init__(self, gain_forward=50, gain_turn=30, gain_climb=20):
        """
        Args:
            gain_*: Scaling from neural activity to motor command
        """
        self.gain_forward = gain_forward
        self.gain_turn = gain_turn
        self.gain_climb = gain_climb

    def decode(self, brain, window_ms=50):
        """
        Read motor neurons, return commands.

        Args:
            brain: FlyBrain instance
            window_ms: Time window for firing rate calculation

        Returns:
            (forward, turn, climb) - each in [-1, 1]
        """

        # Get firing rates (spikes/second)
        forward_rate = brain.get_firing_rate(self.FORWARD_NEURONS, window_ms)
        left_turn_rate = brain.get_firing_rate(self.LEFT_TURN_NEURONS, window_ms)
        right_turn_rate = brain.get_firing_rate(self.RIGHT_TURN_NEURONS, window_ms)
        climb_rate = brain.get_firing_rate(self.CLIMB_NEURONS, window_ms)

        # Convert to motor commands
        # Normalize by expected max firing rate (~100 Hz)
        forward = np.clip(forward_rate / self.gain_forward, -1, 1)
        turn = np.clip((right_turn_rate - left_turn_rate) / self.gain_turn, -1, 1)
        climb = np.clip(climb_rate / self.gain_climb, -1, 1)

        return forward, turn, climb
