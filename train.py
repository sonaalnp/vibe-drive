"""
sound_classifier.train — Training script for the VibeDrive sound classifier.

Reads all samples from samples/, extracts features, trains the model,
and saves to trained_model.pkl.
"""

import os
import sys
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sound_classifier.model import SoundClassifierModel
from audio_processing.processing import AudioProcessor
from audio_capture.capture import AudioCapture


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trained_model.pkl"
)


def train_model(samples_dir=None):
    """
    Train the classifier on generated samples.

    Parameters:
        samples_dir (str): Path to samples directory

    Returns:
        SoundClassifierModel: Trained model
    """
    if samples_dir is None:
        samples_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples"
        )

    capture = AudioCapture()
    processor = AudioProcessor()

    samples = capture.get_available_samples(samples_dir)
    if not samples:
        print(f"  ✗ No samples found in {samples_dir}")
        print("    Run: python main.py --generate")
        return None

    print(f"  Found {len(samples)} samples in {samples_dir}")

    # Extract features from all samples
    X = []
    y = []
    for filepath, category in samples:
        try:
            signal, sr = capture.load_file(filepath)
            filtered = processor.apply_noise_filter(signal, sr)
            features = processor.extract_features(filtered, sr)
            X.append(features)
            y.append(category)
        except Exception as e:
            print(f"  ⚠ Skipped {os.path.basename(filepath)}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"  Extracted features: {X.shape[0]} samples × {X.shape[1]} features")

    # Train model
    model = SoundClassifierModel()
    model.train(X, y)

    # Report accuracy
    accuracy = model.get_accuracy(X, y)
    print(f"  Training accuracy: {accuracy * 100:.1f}%")

    # Print per-class breakdown
    from sklearn.metrics import classification_report
    y_encoded = model.label_encoder.transform(y)
    y_pred = model.model.predict(X)
    report = classification_report(
        y_encoded, y_pred,
        target_names=model.label_encoder.classes_,
        zero_division=0
    )
    print(f"\n  Classification Report:\n{report}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"  ✓ Model saved to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    print("=" * 50)
    print("  VibeDrive — Sound Classifier Training")
    print("=" * 50)
    train_model()
