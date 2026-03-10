"""Tests for audio_processing module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_processing.processing import AudioProcessor


@pytest.fixture
def processor():
    return AudioProcessor()


@pytest.fixture
def sample_signal():
    """Generate a simple test signal: 440 Hz sine wave at 22050 Hz SR."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)
    return signal, sr


class TestNoiseFilter:
    def test_output_shape(self, processor, sample_signal):
        signal, sr = sample_signal
        filtered = processor.apply_noise_filter(signal, sr)
        assert filtered.shape == signal.shape

    def test_output_not_all_zeros(self, processor, sample_signal):
        signal, sr = sample_signal
        filtered = processor.apply_noise_filter(signal, sr)
        assert np.any(filtered != 0)

    def test_filters_low_frequency(self, processor):
        """50 Hz signal should be mostly filtered out (below 200 Hz cutoff)."""
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        low_signal = np.sin(2 * np.pi * 50 * t)
        filtered = processor.apply_noise_filter(low_signal, sr)
        # Energy should be significantly reduced
        assert np.sqrt(np.mean(filtered ** 2)) < np.sqrt(np.mean(low_signal ** 2)) * 0.3


class TestSpectrogram:
    def test_output_is_2d(self, processor, sample_signal):
        signal, sr = sample_signal
        spec = processor.generate_spectrogram(signal, sr)
        assert spec.ndim == 2

    def test_output_shape_n_mels(self, processor, sample_signal):
        signal, sr = sample_signal
        n_mels = 64
        spec = processor.generate_spectrogram(signal, sr, n_mels=n_mels)
        assert spec.shape[0] == n_mels


class TestFeatureExtraction:
    def test_feature_vector_length(self, processor, sample_signal):
        signal, sr = sample_signal
        features = processor.extract_features(signal, sr)
        assert len(features) == 32

    def test_features_are_finite(self, processor, sample_signal):
        signal, sr = sample_signal
        features = processor.extract_features(signal, sr)
        assert np.all(np.isfinite(features))

    def test_different_signals_different_features(self, processor):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sig_a = np.sin(2 * np.pi * 300 * t)
        sig_b = np.sin(2 * np.pi * 1000 * t)
        feat_a = processor.extract_features(sig_a, sr)
        feat_b = processor.extract_features(sig_b, sr)
        assert not np.allclose(feat_a, feat_b)
