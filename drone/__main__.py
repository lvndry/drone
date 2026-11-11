"""Predict musical keys using a trained checkpoint."""

import argparse

import torch

from drone import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+")
    parser.add_argument(
        "--weights", help="Checkpoint path; defaults to DRONE_WEIGHTS or weights/drone-v1.pt"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    predictor = Predictor(args.weights, device=args.device)
    for path in args.audio:
        tonic, mode, probability = predictor.predict(path)
        print(f"{path}: {tonic} {mode} (softmax {100 * probability:.1f}%)")


if __name__ == "__main__":
    main()
