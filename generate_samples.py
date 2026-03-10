"""
generate_samples.py — Synthetic Audio Sample Generator for VibeDrive

Generates WAV files simulating road sounds:
  - Car horn:        400-500 Hz sine bursts
  - Truck horn:      150-200 Hz low-frequency bursts
  - Ambulance siren: 600-900 Hz alternating wail sweep
  - Police siren:    800-1200 Hz rapid yelp
  - Background noise: white noise + random low tones
"""

import os
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 22050
DURATION = 2.0  # seconds per sample
SAMPLES_PER_CATEGORY = 8
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

CATEGORIES = [
    "car_horn",
    "truck_horn",
    "ambulance_siren",
    "police_siren",
    "background_noise",
]


def _normalize(signal):
    """Normalize signal to [-1, 1] range."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return signal


def _add_noise(signal, noise_level=0.02):
    """Add slight background noise for realism."""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise


def generate_car_horn(variation=0):
    """Short ~400-500 Hz beep bursts."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = 400 + variation * 15  # vary between samples
    # Two short bursts
    envelope = np.zeros_like(t)
    burst_len = int(SAMPLE_RATE * 0.3)
    gap = int(SAMPLE_RATE * 0.15)
    start1 = int(SAMPLE_RATE * 0.2)
    start2 = start1 + burst_len + gap
    envelope[start1 : start1 + burst_len] = 1.0
    if start2 + burst_len < len(t):
        envelope[start2 : start2 + burst_len] = 1.0
    # Apply fade in/out
    fade = int(SAMPLE_RATE * 0.01)
    for s in [start1, start2]:
        if s + burst_len < len(t):
            envelope[s : s + fade] = np.linspace(0, 1, fade)
            envelope[s + burst_len - fade : s + burst_len] = np.linspace(1, 0, fade)

    signal = np.sin(2 * np.pi * freq * t) * envelope
    # Add harmonic
    signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t) * envelope
    return _normalize(_add_noise(signal))


def generate_truck_horn(variation=0):
    """Low ~150-200 Hz sustained horn."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = 150 + variation * 8
    # One long sustained blast
    envelope = np.zeros_like(t)
    start = int(SAMPLE_RATE * 0.15)
    end = int(SAMPLE_RATE * (DURATION - 0.3))
    envelope[start:end] = 1.0
    fade = int(SAMPLE_RATE * 0.05)
    envelope[start : start + fade] = np.linspace(0, 1, fade)
    envelope[end - fade : end] = np.linspace(1, 0, fade)

    signal = np.sin(2 * np.pi * freq * t) * envelope
    # Add sub-harmonics for the deep truck sound
    signal += 0.5 * np.sin(2 * np.pi * (freq * 0.5) * t) * envelope
    signal += 0.2 * np.sin(2 * np.pi * (freq * 3) * t) * envelope
    return _normalize(_add_noise(signal))


def generate_ambulance_siren(variation=0):
    """Wailing siren sweeping 600-900 Hz."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    sweep_rate = 1.2 + variation * 0.1  # cycles per second
    freq_low = 600 + variation * 10
    freq_high = 900 + variation * 10
    freq_center = (freq_low + freq_high) / 2
    freq_range = (freq_high - freq_low) / 2
    instantaneous_freq = freq_center + freq_range * np.sin(2 * np.pi * sweep_rate * t)
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / SAMPLE_RATE
    signal = np.sin(phase)

    # Envelope: fade in and out
    envelope = np.ones_like(t)
    fade = int(SAMPLE_RATE * 0.1)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope
    return _normalize(_add_noise(signal))


def generate_police_siren(variation=0):
    """Rapid yelp alternating 800-1200 Hz."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    yelp_rate = 3.5 + variation * 0.2  # faster alternation
    freq_low = 800 + variation * 15
    freq_high = 1200 + variation * 15
    freq_center = (freq_low + freq_high) / 2
    freq_range = (freq_high - freq_low) / 2
    # Square-ish modulation for yelp effect
    mod = np.sign(np.sin(2 * np.pi * yelp_rate * t))
    instantaneous_freq = freq_center + freq_range * mod * 0.8
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / SAMPLE_RATE
    signal = np.sin(phase)

    envelope = np.ones_like(t)
    fade = int(SAMPLE_RATE * 0.05)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope
    return _normalize(_add_noise(signal))


def generate_background_noise(variation=0):
    """White noise with random low-frequency tones."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    np.random.seed(42 + variation)
    signal = np.random.normal(0, 0.3, len(t))
    # Add a few random tones to simulate distant traffic
    for _ in range(3):
        freq = np.random.uniform(80, 300)
        amp = np.random.uniform(0.05, 0.15)
        signal += amp * np.sin(2 * np.pi * freq * t)
    return _normalize(signal)


GENERATORS = {
    "car_horn": generate_car_horn,
    "truck_horn": generate_truck_horn,
    "ambulance_siren": generate_ambulance_siren,
    "police_siren": generate_police_siren,
    "background_noise": generate_background_noise,
}


def generate_all_samples():
    """Generate all synthetic samples and save to samples/ directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    for category in CATEGORIES:
        generator = GENERATORS[category]
        for i in range(SAMPLES_PER_CATEGORY):
            signal = generator(variation=i)
            # Convert to 16-bit PCM
            pcm = np.int16(signal * 32767)
            filename = f"{category}_{i + 1:02d}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            wavfile.write(filepath, SAMPLE_RATE, pcm)
            total += 1
            print(f"  ✓ Generated: {filename}")

    print(f"\n  Total: {total} samples saved to {OUTPUT_DIR}")
    return total


if __name__ == "__main__":
    print("=" * 50)
    print("  VibeDrive — Synthetic Sample Generator")
    print("=" * 50)
    generate_all_samples()
