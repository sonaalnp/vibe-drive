"""
audio_capture.capture — Audio file loading and multi-mic simulation.

Loads WAV files and simulates a 3-microphone triangular array
by introducing time delays based on sound direction.
"""

import os
import numpy as np
from scipy.io import wavfile


class AudioCapture:
    """Handles audio file I/O and multi-microphone simulation."""

    # Speed of sound in air (m/s)
    SPEED_OF_SOUND = 343.0
    # Distance between microphones in the simulated triangular array (meters)
    MIC_SPACING = 0.15  # 15 cm

    # Microphone positions in a triangular layout (x, y in meters)
    # Mic 0: front-center, Mic 1: rear-left, Mic 2: rear-right
    MIC_POSITIONS = {
        "mic_0": np.array([0.0, 0.075]),     # front
        "mic_1": np.array([-0.065, -0.0375]),  # rear-left
        "mic_2": np.array([0.065, -0.0375]),   # rear-right
    }

    # Direction vectors (unit vectors pointing FROM direction of sound source)
    DIRECTION_VECTORS = {
        "LEFT":  np.array([-1.0, 0.0]),
        "RIGHT": np.array([1.0, 0.0]),
        "FRONT": np.array([0.0, 1.0]),
        "BEHIND": np.array([0.0, -1.0]),
    }

    def load_file(self, filepath):
        """
        Load a WAV file and return signal + sample rate.

        Parameters:
            filepath (str): Path to .wav file

        Returns:
            tuple: (signal as float64 numpy array, sample_rate)
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        sr, data = wavfile.read(filepath)

        # Convert to float64 normalized [-1, 1]
        if data.dtype == np.int16:
            signal = data.astype(np.float64) / 32767.0
        elif data.dtype == np.int32:
            signal = data.astype(np.float64) / 2147483647.0
        elif data.dtype == np.float32 or data.dtype == np.float64:
            signal = data.astype(np.float64)
        else:
            signal = data.astype(np.float64)

        # If stereo, take first channel
        if len(signal.shape) > 1:
            signal = signal[:, 0]

        return signal, sr

    def simulate_multi_mic(self, signal, sr, direction="FRONT"):
        """
        Simulate 3-microphone array capture with time delays.

        Takes a mono signal and creates delayed versions to simulate
        sound arriving from a specific direction.

        Parameters:
            signal  (np.ndarray): Mono audio signal
            sr      (int):        Sample rate
            direction (str):      One of 'LEFT', 'RIGHT', 'FRONT', 'BEHIND'

        Returns:
            dict: {mic_name: delayed_signal} for each of 3 mics
        """
        direction = direction.upper()
        if direction not in self.DIRECTION_VECTORS:
            direction = "FRONT"

        sound_dir = self.DIRECTION_VECTORS[direction]

        # Calculate time-of-arrival delay for each mic
        # Delay = projection of mic position onto sound direction / speed_of_sound
        delays = {}
        for mic_name, mic_pos in self.MIC_POSITIONS.items():
            # Positive delay = sound arrives later at this mic
            delay_seconds = np.dot(mic_pos, sound_dir) / self.SPEED_OF_SOUND
            delays[mic_name] = delay_seconds

        # Normalize delays so the earliest mic has delay = 0
        min_delay = min(delays.values())
        for mic_name in delays:
            delays[mic_name] -= min_delay

        # Apply delays by shifting samples
        mic_signals = {}
        for mic_name, delay_sec in delays.items():
            delay_samples = int(delay_sec * sr)
            delayed = np.zeros(len(signal) + delay_samples)
            delayed[delay_samples:delay_samples + len(signal)] = signal
            # Trim to original length
            delayed = delayed[:len(signal)]
            # Add slight per-mic noise variation
            noise = np.random.normal(0, 0.005, len(delayed))
            mic_signals[mic_name] = delayed + noise

        return mic_signals

    def get_available_samples(self, samples_dir):
        """
        List all .wav files in the samples directory.

        Returns:
            list: List of (filepath, category_name) tuples
        """
        if not os.path.isdir(samples_dir):
            return []

        samples = []
        for filename in sorted(os.listdir(samples_dir)):
            if filename.endswith(".wav"):
                filepath = os.path.join(samples_dir, filename)
                # Extract category from filename like "car_horn_01.wav"
                parts = filename.rsplit("_", 1)
                category = parts[0] if len(parts) > 1 else filename[:-4]
                samples.append((filepath, category))

        return samples
