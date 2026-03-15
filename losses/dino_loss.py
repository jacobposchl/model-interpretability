"""
DINO self-distillation loss and projection head.

The DINO loss trains a student network to match the output distribution of a
teacher network (EMA copy of the student). Two mechanisms prevent collapse:

  1. Centering: teacher outputs are shifted by a running mean, preventing the
     teacher from saturating a single prototype dimension.
  2. Sharpening: the teacher uses a lower temperature than the student, producing
     confident pseudo-label distributions that give meaningful gradient signal.

The center buffer is updated inside forward() automatically — no external call
needed. Center update uses detached teacher outputs so the center accumulation
does not create gradient paths back through the teacher.

DINOProjectionHead applies on top of the MetaEncoder output z [B, 64] and
produces the distribution used for self-distillation. The final linear layer
has no bias (standard DINO practice — prevents trivial bias-based solutions).

Reference: Caron et al. "Emerging Properties in Self-Supervised Vision
Transformers" (ICCV 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_output: [B, out_dim] raw logits from student projection head
            teacher_output: [B, out_dim] raw logits from teacher projection head

        Returns:
            Scalar DINO loss.
        """
        # Teacher: center then sharpen
        teacher_centered = teacher_output.detach() - self.center
        teacher_probs = F.softmax(teacher_centered / self.teacher_temp, dim=-1)  # [B, D]

        # Student: log-softmax with higher temperature
        student_log_probs = F.log_softmax(student_output / self.student_temp, dim=-1)  # [B, D]

        # Cross-entropy: teacher → student
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        # Update center with EMA of teacher outputs (detached — no gradient)
        batch_center = teacher_output.detach().mean(dim=0, keepdim=True)
        self.center = (
            self.center_momentum * self.center
            + (1.0 - self.center_momentum) * batch_center
        )

        return loss


class DINOProjectionHead(nn.Module):
    """
    3-layer MLP projection head applied on top of MetaEncoder output z.

    Architecture: Linear → GELU → Linear → GELU → Linear(bottleneck) →
                  L2-normalise → Linear(out_dim, bias=False)

    The intermediate bottleneck + L2-normalisation before the final layer is
    standard DINO practice: it decouples the projection space from the embedding
    space and prevents the head from relying on the magnitude of z.

    Input: z [B, in_dim] from MetaEncoder (already L2-normalised)
    Output: [B, out_dim] raw logits — NOT normalised (normalisation happens
            implicitly through temperature scaling in DINOLoss)
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 256,
        out_dim: int = 256,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.mlp(z)
        x = F.normalize(x, dim=-1)   # L2-normalise before final linear
        return self.last_layer(x)
