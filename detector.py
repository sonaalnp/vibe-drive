"""
direction_detector.detector — Simulated TDOA multi-microphone direction detection.

Simulates a 3-microphone triangular array and uses cross-correlation
to estimate the direction of a sound source.
"""

import numpy as np


class DirectionDetector:
    """
    Direction-of-arrival estimator using simulated TDOA
    (Time Difference of Arrival) with cross-correlation.
    """

    # Microphone layout matches AudioCapture's array
    # Mic 0: front-center, Mic 1: rear-left, Mic 2: rear-right
    DIRECTIONS = ["LEFT", "RIGHT", "FRONT", "BEHIND"]

    def __init__(self):
        pass

    def _cross_correlate(self, sig_a, sig_b, max_lag=100):
        """
        Compute cross-correlation between two signals and find peak lag.

        Parameters:
            sig_a   (np.ndarray): First signal
            sig_b   (np.ndarray): Second signal
            max_lag (int):        Maximum lag to search (in samples)

        Returns:
            int: Lag in samples (positive = sig_b arrives later)
        """
        # Ensure same length
        min_len = min(len(sig_a), len(sig_b))
        sig_a = sig_a[:min_len]
        sig_b = sig_b[:min_len]

        # Normalize signals
        norm_a = np.linalg.norm(sig_a)
        norm_b = np.linalg.norm(sig_b)
        if norm_a > 0:
            sig_a = sig_a / norm_a
        if norm_b > 0:
            sig_b = sig_b / norm_b

        # Compute cross-correlation using numpy
        correlation = np.correlate(sig_a, sig_b, mode="full")
        mid = len(correlation) // 2

        # Search only within max_lag window
        search_start = max(0, mid - max_lag)
        search_end = min(len(correlation), mid + max_lag + 1)
        search_region = correlation[search_start:search_end]

        if len(search_region) == 0:
            return 0

        peak_idx = np.argmax(np.abs(search_region))
        lag = peak_idx - (mid - search_start)

        return lag

    def detect_direction(self, mic_signals):
        """
        Detect the direction of a sound source from multi-mic signals.

        Uses TDOA between microphone pairs:
        - mic_0 (front) vs mic_1 (rear-left):  determines front/behind
        - mic_1 (rear-left) vs mic_2 (rear-right): determines left/right

        Parameters:
            mic_signals (dict): Dictionary with keys 'mic_0', 'mic_1', 'mic_2'
                                containing numpy arrays for each mic

        Returns:
            str: Direction string — 'LEFT', 'RIGHT', 'FRONT', or 'BEHIND'
        """
        if not all(k in mic_signals for k in ("mic_0", "mic_1", "mic_2")):
            return "FRONT"  # Default if incomplete data

        sig_0 = mic_signals["mic_0"]  # front
        sig_1 = mic_signals["mic_1"]  # rear-left
        sig_2 = mic_signals["mic_2"]  # rear-right

        # TDOA: front mic vs rear-left mic
        # Positive lag → sound hits front first → sound from FRONT
        lag_front_rear = self._cross_correlate(sig_0, sig_1)

        # TDOA: rear-left mic vs rear-right mic
        # Positive lag → sound hits left first → sound from LEFT
        lag_left_right = self._cross_correlate(sig_1, sig_2)

        # Decision logic
        # Primary: is the source in front or behind?
        # Secondary: is it left or right?

        front_behind_threshold = 1  # samples
        left_right_threshold = 1    # samples

        if abs(lag_left_right) > abs(lag_front_rear):
            # Dominant left/right difference
            if lag_left_right > left_right_threshold:
                return "LEFT"
            elif lag_left_right < -left_right_threshold:
                return "RIGHT"

        # Check front/behind
        if lag_front_rear > front_behind_threshold:
            return "FRONT"
        elif lag_front_rear < -front_behind_threshold:
            return "BEHIND"

        # If delays are very small, check left/right as fallback
        if lag_left_right > 0:
            return "LEFT"
        elif lag_left_right < 0:
            return "RIGHT"

        return "FRONT"  # Default

    def detect_direction_simulated(self, direction_hint="FRONT"):
        """
        Simplified direction detection using a known direction hint.
        Used in demo mode when we know the simulated direction.

        Parameters:
            direction_hint (str): The known direction

        Returns:
            str: Validated direction string
        """
        direction_hint = direction_hint.upper()
        if direction_hint in self.DIRECTIONS:
            return direction_hint
        return "FRONT"
