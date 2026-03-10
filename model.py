"""
sound_classifier.model — ML model definition for sound classification.

Uses scikit-learn RandomForestClassifier (lightweight, no GPU needed).
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Sound categories
LABELS = [
    "car_horn",
    "truck_horn",
    "ambulance_siren",
    "police_siren",
    "background_noise",
]


class SoundClassifierModel:
    """RandomForest-based sound classifier."""

    def __init__(self, n_estimators=100, random_state=42):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABELS)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=20,
            min_samples_split=3,
            class_weight="balanced",
        )
        self.is_trained = False

    def train(self, X, y):
        """
        Train the classifier.

        Parameters:
            X (np.ndarray): Feature matrix (n_samples x n_features)
            y (list):       Labels (string category names)
        """
        y_encoded = self.label_encoder.transform(y)
        self.model.fit(X, y_encoded)
        self.is_trained = True

    def predict(self, features):
        """
        Predict sound category.

        Parameters:
            features (np.ndarray): Feature vector (1D or 2D)

        Returns:
            tuple: (label_string, confidence_float)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run training first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        label = self.label_encoder.inverse_transform([prediction])[0]

        return label, confidence

    def get_accuracy(self, X, y):
        """Calculate accuracy on given data."""
        y_encoded = self.label_encoder.transform(y)
        return self.model.score(X, y_encoded)
