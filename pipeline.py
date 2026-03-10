"""
simulator.pipeline — Full VibeDrive simulation pipeline orchestrator.

Wires all modules together:
Audio Input → Noise Filtering → Feature Extraction → Classification → Direction Detection → Alert
"""

import os
import sys
import random
import time

from audio_capture.capture import AudioCapture
from audio_processing.processing import AudioProcessor
from sound_classifier.inference import SoundClassifier
from direction_detector.detector import DirectionDetector
from alert_system.alerts import AlertSystem, ANSI_COLORS


class SimulatorPipeline:
    """
    Full pipeline orchestrator for VibeDrive simulation.

    Connects all modules in sequence and runs audio through
    the complete detection-to-alert pipeline.
    """

    def __init__(self):
        self.capture = AudioCapture()
        self.processor = AudioProcessor()
        self.classifier = SoundClassifier()
        self.direction_detector = DirectionDetector()
        self.alert_system = AlertSystem()

    def run(self, audio_path, direction="FRONT", verbose=True):
        """
        Run a single audio file through the full pipeline.

        Parameters:
            audio_path (str):  Path to a .wav file
            direction  (str):  Simulated direction (LEFT/RIGHT/FRONT/BEHIND)
            verbose    (bool): Print intermediate steps

        Returns:
            dict: Results containing label, confidence, direction
        """
        bold = ANSI_COLORS["BOLD"]
        reset = ANSI_COLORS["RESET"]
        gray = ANSI_COLORS["GRAY"]

        filename = os.path.basename(audio_path)

        if verbose:
            print(f"\n{gray}{'─' * 56}{reset}")
            print(f"  {bold}Processing:{reset} {filename}")
            print(f"  {bold}Simulated direction:{reset} {direction}")
            print(f"{gray}{'─' * 56}{reset}")

        # Step 1: Load audio
        if verbose:
            print(f"  {gray}[1/5]{reset} Loading audio file...")
        signal, sr = self.capture.load_file(audio_path)

        # Step 2: Simulate multi-mic capture
        if verbose:
            print(f"  {gray}[2/5]{reset} Simulating 3-mic array (direction={direction})...")
        mic_signals = self.capture.simulate_multi_mic(signal, sr, direction)

        # Step 3: Apply noise filter on primary mic signal
        if verbose:
            print(f"  {gray}[3/5]{reset} Applying bandpass noise filter...")
        filtered = self.processor.apply_noise_filter(signal, sr)

        # Step 4: Classify the sound
        if verbose:
            print(f"  {gray}[4/5]{reset} Running ML classification...")
        label, confidence = self.classifier.classify(filtered, sr)

        # Step 5: Detect direction from mic array
        if verbose:
            print(f"  {gray}[5/5]{reset} Detecting direction via TDOA...")
        detected_direction = self.direction_detector.detect_direction(mic_signals)

        # Trigger alert
        self.alert_system.clear()
        self.alert_system.trigger(label, detected_direction, confidence)

        return {
            "file": filename,
            "label": label,
            "confidence": confidence,
            "simulated_direction": direction,
            "detected_direction": detected_direction,
        }

    def run_demo(self, samples_dir=None, delay=1.0):
        """
        Run all sample files through the pipeline as a full demo.

        Parameters:
            samples_dir (str):   Path to samples directory
            delay       (float): Seconds to pause between samples
        """
        if samples_dir is None:
            samples_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "samples",
            )

        samples = self.capture.get_available_samples(samples_dir)
        if not samples:
            print("  ✗ No samples found. Run: python main.py --generate")
            return

        if not self.classifier.is_ready():
            print("  ✗ No trained model. Run: python main.py --train")
            return

        bold = ANSI_COLORS["BOLD"]
        reset = ANSI_COLORS["RESET"]
        white = ANSI_COLORS["WHITE"]

        directions = ["LEFT", "RIGHT", "FRONT", "BEHIND"]

        print()
        print(f"  {bold}{'=' * 56}{reset}")
        print(f"  {bold}  VibeDrive — Full Pipeline Demo{reset}")
        print(f"  {bold}  Processing {len(samples)} audio samples...{reset}")
        print(f"  {bold}{'=' * 56}{reset}")

        results = []
        correct = 0
        total = 0

        for filepath, expected_category in samples:
            # Randomly assign a direction for demo variety
            direction = random.choice(directions)

            result = self.run(filepath, direction=direction, verbose=True)
            results.append(result)

            # Check if classification matches expected
            if result["label"] == expected_category:
                correct += 1
            total += 1

            time.sleep(delay)

        # Print summary
        print(f"\n  {bold}{'=' * 56}{reset}")
        print(f"  {bold}  Demo Summary{reset}")
        print(f"  {bold}{'=' * 56}{reset}")
        print(f"  Total samples processed: {total}")
        print(f"  Correct classifications: {correct}/{total} ({correct/total*100:.1f}%)")
        print()

        # Summary table
        print(f"  {'File':<30} {'Predicted':<20} {'Direction':<10} {'Conf':>6}")
        print(f"  {'─' * 30} {'─' * 20} {'─' * 10} {'─' * 6}")
        for r in results:
            print(
                f"  {r['file']:<30} {r['label']:<20} {r['detected_direction']:<10} "
                f"{r['confidence'] * 100:5.1f}%"
            )

        print(f"\n  {bold}Demo complete.{reset}\n")
        return results
