"""
audio_processing.processing — Noise filtering, spectrogram, and feature extraction.

Pipeline step: raw audio → bandpass filter → mel spectrogram → MFCC features
"""

import numpy as np
from scipy.signal import butter, sosfilt
import warnings


class AudioProcessor:
    """Audio signal processing for the VibeDrive pipeline."""

    def __init__(self, low_freq=200, high_freq=3000):
        """
        Parameters:
            low_freq  (int): Bandpass filter low cutoff (Hz)
            high_freq (int): Bandpass filter high cutoff (Hz)
        """
        self.low_freq = low_freq
        self.high_freq = high_freq

    def apply_noise_filter(self, signal, sr):
        """
        Apply a Butterworth bandpass filter to isolate relevant frequencies.

        Parameters:
            signal (np.ndarray): Input audio signal
            sr     (int):        Sample rate

        Returns:
            np.ndarray: Filtered signal
        """
        nyquist = sr / 2.0
        low = self.low_freq / nyquist
        high = min(self.high_freq / nyquist, 0.99)  # Ensure < 1.0

        sos = butter(N=4, Wn=[low, high], btype="band", output="sos")
        filtered = sosfilt(sos, signal)
        return filtered

    def generate_spectrogram(self, signal, sr, n_mels=64, hop_length=512):
        """
        Generate a mel spectrogram from the signal.

        Uses librosa if available, otherwise falls back to scipy STFT.

        Parameters:
            signal     (np.ndarray): Audio signal
            sr         (int):        Sample rate
            n_mels     (int):        Number of mel bands
            hop_length (int):        Hop length for STFT

        Returns:
            np.ndarray: Mel spectrogram (2D array, shape: n_mels x time_frames)
        """
        try:
            import librosa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mel_spec = librosa.feature.melspectrogram(
                    y=signal, sr=sr, n_mels=n_mels, hop_length=hop_length
                )
                # Convert to log scale (dB)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except ImportError:
            # Fallback: simple STFT-based spectrogram
            from scipy.signal import stft
            _, _, Zxx = stft(signal, fs=sr, nperseg=1024, noverlap=512)
            return np.abs(Zxx)

    def extract_features(self, signal, sr, n_mfcc=13):
        """
        Extract MFCC and spectral features for classification.

        Features extracted:
        - 13 MFCCs (mean + std = 26 features)
        - Spectral centroid (mean + std = 2)
        - Spectral rolloff (mean + std = 2)
        - Zero crossing rate (mean + std = 2)
        Total: 32 features

        Parameters:
            signal (np.ndarray): Audio signal
            sr     (int):        Sample rate
            n_mfcc (int):        Number of MFCC coefficients

        Returns:
            np.ndarray: Feature vector (1D)
        """
        features = []

        try:
            import librosa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # MFCCs
                mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
                features.extend(np.mean(mfccs, axis=1))
                features.extend(np.std(mfccs, axis=1))

                # Spectral centroid
                centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
                features.append(np.mean(centroid))
                features.append(np.std(centroid))

                # Spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
                features.append(np.mean(rolloff))
                features.append(np.std(rolloff))

                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(signal)
                features.append(np.mean(zcr))
                features.append(np.std(zcr))

        except ImportError:
            # Fallback: basic spectral features using numpy/scipy only
            features = self._extract_basic_features(signal, sr)

        return np.array(features, dtype=np.float64)

    def _extract_basic_features(self, signal, sr):
        """Fallback feature extraction without librosa."""
        features = []

        # Simple spectral analysis via FFT
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)

        # Spectral centroid
        if np.sum(fft) > 0:
            centroid = np.sum(freqs * fft) / np.sum(fft)
        else:
            centroid = 0.0
        features.append(centroid)

        # Spectral spread
        if np.sum(fft) > 0:
            spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft) / np.sum(fft))
        else:
            spread = 0.0
        features.append(spread)

        # Spectral energy in bands
        band_edges = [0, 200, 500, 1000, 2000, 4000, sr // 2]
        for i in range(len(band_edges) - 1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            band_energy = np.sum(fft[mask] ** 2)
            features.append(band_energy)

        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal))
        features.append(zcr)

        # RMS energy
        rms = np.sqrt(np.mean(signal ** 2))
        features.append(rms)

        # Pad to 32 features for consistency
        while len(features) < 32:
            features.append(0.0)

        return features[:32]
