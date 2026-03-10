"""Tests for sound_classifier module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sound_classifier.model import SoundClassifierModel, LABELS


@pytest.fixture
def trained_model():
    """Create and train a model on dummy data."""
    model = SoundClassifierModel()
    n_features = 32
    n_per_class = 10

    X = []
    y = []
    for i, label in enumerate(LABELS):
        # Create distinct feature patterns per class
        np.random.seed(i)
        for _ in range(n_per_class):
            features = np.random.randn(n_features) + i * 2  # Offset per class
            X.append(features)
            y.append(label)

    X = np.array(X)
    model.train(X, y)
    return model, X, y


class TestSoundClassifierModel:
    def test_train_no_error(self, trained_model):
        model, X, y = trained_model
        assert model.is_trained

    def test_predict_returns_valid_label(self, trained_model):
        model, X, y = trained_model
        label, confidence = model.predict(X[0])
        assert label in LABELS

    def test_predict_returns_confidence(self, trained_model):
        model, X, y = trained_model
        label, confidence = model.predict(X[0])
        assert 0.0 <= confidence <= 1.0

    def test_accuracy_above_threshold(self, trained_model):
        model, X, y = trained_model
        acc = model.get_accuracy(X, y)
        # With well-separated dummy data, accuracy should be high
        assert acc > 0.5

    def test_predict_untrained_raises(self):
        model = SoundClassifierModel()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros(32))

    def test_all_labels_encoded(self):
        model = SoundClassifierModel()
        encoded = model.label_encoder.transform(LABELS)
        assert len(encoded) == len(LABELS)
