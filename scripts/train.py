"""
CLI entry point for training.

Examples:
    # Baseline — no consistency loss
    python scripts/train.py --config configs/baseline.yaml

    # CTLS — Option A (fixed weighted-sum meta-encoder)
    python scripts/train.py --config configs/unified_a.yaml

    # CTLS — Option B (transformer CLS meta-encoder)
    python scripts/train.py --config configs/unified_b.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/unified_b.yaml --resume experiments/unified_b/epoch_50.pt
"""

import argparse
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from training.unified_trainer import UnifiedTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CTLS model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint_dir = config["logging"]["checkpoint_dir"]
    print(f"Config:     {args.config}")
    print(f"Output dir: {checkpoint_dir}")
    if args.resume:
        print(f"Resuming:   {args.resume}")

    trainer = UnifiedTrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
