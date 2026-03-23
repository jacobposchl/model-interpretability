"""
Schedules for the two auxiliary hyperparameters in CTLS training:

  λ (lambda) — weight on the consistency loss.
    Warmed up from 0 to its target value over the first N epochs, then held
    constant. A cold start prevents the consistency loss from disrupting the
    backbone before task-relevant representations have formed.

  τ (tau) — soft mask temperature.
    Annealed from a high initial value (fluid, continuous weighting) down to
    a low final value (sharp, near-binary circuits). Cosine schedule gives a
    smooth transition that avoids sudden representation shifts.
"""

import math


class LambdaScheduler:
    """Linear warmup for the consistency loss weight λ."""

    def __init__(self, init_val: float, final_val: float, warmup_epochs: int):
        self.init_val = init_val
        self.final_val = final_val
        self.warmup_epochs = warmup_epochs

    def get(self, epoch: int) -> float:
        if self.warmup_epochs == 0:
            return self.final_val
        if epoch >= self.warmup_epochs:
            return self.final_val
        # Linear interpolation
        progress = epoch / self.warmup_epochs
        return self.init_val + progress * (self.final_val - self.init_val)


class TauScheduler:
    """Cosine annealing for the soft mask temperature τ."""

    def __init__(self, init_val: float, final_val: float, anneal_epochs: int):
        self.init_val = init_val
        self.final_val = final_val
        self.anneal_epochs = anneal_epochs

    def get(self, epoch: int) -> float:
        if epoch >= self.anneal_epochs:
            return self.final_val
        progress = epoch / self.anneal_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_val + cosine_decay * (self.init_val - self.final_val)
