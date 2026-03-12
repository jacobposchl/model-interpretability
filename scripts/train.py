"""
CLI entry point for training.

Examples:
    # Stage 1 — baseline, no consistency loss
    python scripts/train.py --config configs/baseline.yaml

    # Stage 2 — full CTLS objective
    python scripts/train.py --config configs/ctls.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/ctls.yaml --resume experiments/ctls/epoch_50.pt

    # Stage 4 ablation — uniform depth weighting
    python scripts/train.py --config configs/ablations/uniform_weighting.yaml
"""

import argparse
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CTLS model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Config:     {args.config}")
    if args.resume:
        print(f"Resuming:   {args.resume}")

    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
