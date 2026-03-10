"""
VibeDrive — Smart Assistive Driving Alert System
=================================================

Software simulation of an assistive device for deaf drivers.
Detects road sounds, classifies them via ML, simulates directional
detection, and outputs color-coded console alerts.

Usage:
    python main.py --generate    Generate synthetic audio samples
    python main.py --train       Train the sound classifier
    python main.py --run <file>  Classify a single audio file
    python main.py --demo        Run full pipeline demo
"""

import argparse
import os
import sys

# Ensure project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def print_banner():
    """Print the VibeDrive startup banner."""
    banner = """
\033[1m\033[94m
 ██╗   ██╗██╗██████╗ ███████╗██████╗ ██████╗ ██╗██╗   ██╗███████╗
 ██║   ██║██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██║██║   ██║██╔════╝
 ██║   ██║██║██████╔╝█████╗  ██║  ██║██████╔╝██║██║   ██║█████╗
 ╚██╗ ██╔╝██║██╔══██╗██╔══╝  ██║  ██║██╔══██╗██║╚██╗ ██╔╝██╔══╝
  ╚████╔╝ ██║██████╔╝███████╗██████╔╝██║  ██║██║ ╚████╔╝ ███████╗
   ╚═══╝  ╚═╝╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝
\033[0m
\033[1m  Smart Assistive Driving Alert System for Deaf Drivers\033[0m
\033[90m  Software Simulation v1.0\033[0m
"""
    print(banner)


def cmd_generate():
    """Generate synthetic audio samples."""
    print("\033[1m" + "=" * 56 + "\033[0m")
    print("\033[1m  Step 1: Generating Synthetic Audio Samples\033[0m")
    print("\033[1m" + "=" * 56 + "\033[0m")
    from generate_samples import generate_all_samples
    generate_all_samples()
    print("\n  \033[92m✓ Sample generation complete.\033[0m\n")


def cmd_train():
    """Train the sound classifier."""
    print("\033[1m" + "=" * 56 + "\033[0m")
    print("\033[1m  Step 2: Training Sound Classifier\033[0m")
    print("\033[1m" + "=" * 56 + "\033[0m")
    from sound_classifier.train import train_model
    model = train_model()
    if model:
        print("\n  \033[92m✓ Training complete.\033[0m\n")
    else:
        print("\n  \033[91m✗ Training failed.\033[0m\n")
        sys.exit(1)


def cmd_run(filepath, direction="FRONT"):
    """Classify a single audio file."""
    if not os.path.isfile(filepath):
        print(f"\n  \033[91m✗ File not found: {filepath}\033[0m\n")
        sys.exit(1)

    print("\033[1m" + "=" * 56 + "\033[0m")
    print("\033[1m  Single File Classification\033[0m")
    print("\033[1m" + "=" * 56 + "\033[0m")

    from simulator.pipeline import SimulatorPipeline
    pipeline = SimulatorPipeline()

    if not pipeline.classifier.is_ready():
        print("\n  \033[91m✗ No trained model. Run: python main.py --train\033[0m\n")
        sys.exit(1)

    result = pipeline.run(filepath, direction=direction, verbose=True)
    print(f"\n  Result: {result['label']} ({result['confidence']*100:.1f}% confidence)")
    print(f"  Direction: {result['detected_direction']}\n")


def cmd_demo():
    """Run the full demo pipeline."""
    print("\033[1m" + "=" * 56 + "\033[0m")
    print("\033[1m  Full Pipeline Demo\033[0m")
    print("\033[1m" + "=" * 56 + "\033[0m")

    from simulator.pipeline import SimulatorPipeline
    pipeline = SimulatorPipeline()

    if not pipeline.classifier.is_ready():
        print("\n  \033[91m✗ No trained model found.\033[0m")
        print("  Running auto-setup: generate → train → demo\n")
        cmd_generate()
        cmd_train()

    pipeline = SimulatorPipeline()  # Reinitialize to load new model
    pipeline.run_demo(delay=0.5)


def cmd_all():
    """Run the complete pipeline: generate → train → demo."""
    cmd_generate()
    cmd_train()
    cmd_demo()


def main():
    parser = argparse.ArgumentParser(
        prog="vibedrive",
        description="VibeDrive — Smart Assistive Driving Alert System Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --generate           Generate synthetic audio samples
  python main.py --train              Train the sound classifier
  python main.py --run samples/car_horn_01.wav   Classify a single file
  python main.py --run samples/car_horn_01.wav --direction LEFT
  python main.py --demo               Run full pipeline demo
  python main.py --all                Run generate → train → demo
        """,
    )

    parser.add_argument(
        "--generate", action="store_true",
        help="Generate synthetic audio samples in samples/"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train the ML sound classifier"
    )
    parser.add_argument(
        "--run", type=str, metavar="FILE",
        help="Classify a single audio file"
    )
    parser.add_argument(
        "--direction", type=str, default="FRONT",
        choices=["LEFT", "RIGHT", "FRONT", "BEHIND"],
        help="Simulated direction for --run (default: FRONT)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the full pipeline demo with all samples"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run complete pipeline: generate → train → demo"
    )

    args = parser.parse_args()

    print_banner()

    # If no arguments, show help
    if not any([args.generate, args.train, args.run, args.demo, args.all]):
        parser.print_help()
        print("\n  \033[93mTip: Use --all to run the complete pipeline.\033[0m\n")
        return

    if args.all:
        cmd_all()
        return

    if args.generate:
        cmd_generate()

    if args.train:
        cmd_train()

    if args.run:
        cmd_run(args.run, args.direction)

    if args.demo:
        cmd_demo()


if __name__ == "__main__":
    main()
