"""Tests for direction_detector module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from direction_detector.detector import DirectionDetector
from audio_capture.capture import AudioCapture


@pytest.fixture
def detector():
    return DirectionDetector()


@pytest.fixture
def capture():
    return AudioCapture()


class TestDirectionDetector:
    def test_returns_valid_direction(self, detector):
        """Direction detection should return a valid direction string."""
        sr = 22050
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)

        capture = AudioCapture()
        mic_signals = capture.simulate_multi_mic(signal, sr, "LEFT")
        direction = detector.detect_direction(mic_signals)
        assert direction in DirectionDetector.DIRECTIONS

    def test_simulated_direction_validator(self, detector):
        assert detector.detect_direction_simulated("LEFT") == "LEFT"
        assert detector.detect_direction_simulated("RIGHT") == "RIGHT"
        assert detector.detect_direction_simulated("FRONT") == "FRONT"
        assert detector.detect_direction_simulated("BEHIND") == "BEHIND"

    def test_invalid_direction_defaults(self, detector):
        assert detector.detect_direction_simulated("UP") == "FRONT"
        assert detector.detect_direction_simulated("") == "FRONT"

    def test_incomplete_mic_data(self, detector):
        """Should default to FRONT when mic data is incomplete."""
        result = detector.detect_direction({"mic_0": np.zeros(100)})
        assert result == "FRONT"

    def test_cross_correlate_zero_lag(self, detector):
        """Identical signals should have zero lag."""
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 2205))
        lag = detector._cross_correlate(signal, signal)
        assert lag == 0

    def test_cross_correlate_positive_lag(self, detector):
        """Delayed signal should show positive lag."""
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 2205))
        delay = 5
        delayed = np.zeros(len(signal))
        delayed[delay:] = signal[:-delay]
        lag = detector._cross_correlate(signal, delayed, max_lag=50)
        assert lag > 0
