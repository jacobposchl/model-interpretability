"""
Schedule for the geometry loss weight lambda.

Lambda is warmed up from 0 to its target value over the first N epochs, then
held constant. A cold start prevents the geometry loss from disrupting the
meta-encoder before the fidelity term has established baseline representations.
"""


class LambdaScheduler:
    """Linear warmup for the geometry loss weight lambda."""

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
