"""
sound_classifier.inference — Runtime sound classification.

Loads trained model and classifies audio signals.
"""

import os
import joblib
import numpy as np

from audio_processing.processing import AudioProcessor
from sound_classifier.model import SoundClassifierModel


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trained_model.pkl"
)


class SoundClassifier:
    """High-level classifier interface for the pipeline."""

    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self.processor = AudioProcessor()
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        if os.path.isfile(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def is_ready(self):
        """Check if a trained model is loaded."""
        return self.model is not None and self.model.is_trained

    def classify(self, signal, sr):
        """
        Classify an audio signal.

        Parameters:
            signal (np.ndarray): Audio signal
            sr     (int):        Sample rate

        Returns:
            tuple: (label, confidence)
        """
        if not self.is_ready():
            raise RuntimeError(
                "No trained model found. Run: python main.py --train"
            )

        # Process and extract features
        filtered = self.processor.apply_noise_filter(signal, sr)
        features = self.processor.extract_features(filtered, sr)

        # Predict
        label, confidence = self.model.predict(features)
        return label, confidence
