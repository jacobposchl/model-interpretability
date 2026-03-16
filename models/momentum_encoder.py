"""
Momentum (EMA) encoder and embedding bank for DINO and CTLS-SSL.

MomentumEncoder
  Wraps a backbone + meta_encoder (and optionally a projection head) as a
  teacher whose parameters are never optimised directly — they track the student
  via exponential moving average (EMA) updates after each training step.

  Used in two roles:
    1. DINO teacher: produces stable distribution targets for self-distillation.
    2. CTLS-SSL bank encoder: produces stable embeddings for EmbeddingBank
       updates, preventing the "chasing your own tail" instability that would
       occur if the bank were populated with rapidly-changing student embeddings.

EmbeddingBank
  Fixed-size buffer of L2-normalised embeddings indexed by dataset position
  (not insertion order). Each slot corresponds to one training sample; slots
  are updated whenever that sample's batch is processed.

  Nearest-neighbor mining (get_neighbors) uses a full cosine similarity matrix
  — practical for CIFAR-10 (50k × 64 = 3.2M floats ≈ 12 MB on GPU). For
  larger datasets this should be replaced with approximate search (FAISS).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentumEncoder(nn.Module):
    """
    EMA wrapper around backbone + meta_encoder (and optionally a projection head).

    Teacher parameters are initialised as a deep copy of the student and are
    set to requires_grad=False. They are updated exclusively through the EMA
    update rule: θ_t ← m · θ_t + (1 − m) · θ_s.

    forward() runs under torch.no_grad() and returns the same tuple as the
    student (logits, trajectory, z) — and optionally proj_output if a
    projection head was supplied.
    """

    def __init__(
        self,
        student_backbone: nn.Module,
        student_meta_encoder: nn.Module,
        student_proj_head: nn.Module | None = None,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.momentum = momentum

        # Deep-copy student weights; disable gradients for all teacher parameters.
        # _re_register_hooks() is required: deepcopy copies hook closures but
        # closures capture the original instance's self, so without re-registration
        # the teacher's forward hooks would write into the student's _trajectory.
        self.teacher_backbone = copy.deepcopy(student_backbone)
        self.teacher_backbone._re_register_hooks()
        self.teacher_meta_encoder = copy.deepcopy(student_meta_encoder)
        self.teacher_proj_head = (
            copy.deepcopy(student_proj_head) if student_proj_head is not None else None
        )

        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Returns:
            logits:      [B, num_classes]
            trajectory:  list of L tensors, each [B, D_l]
            z:           [B, embedding_dim] L2-normalised circuit embedding
            proj_output: [B, out_dim] if a projection head is present, else None
        """
        logits, trajectory = self.teacher_backbone(x)
        z = self.teacher_meta_encoder(trajectory)
        proj_output = (
            self.teacher_proj_head(z) if self.teacher_proj_head is not None else None
        )
        return logits, trajectory, z, proj_output

    @torch.no_grad()
    def update(
        self,
        student_backbone: nn.Module,
        student_meta_encoder: nn.Module,
        student_proj_head: nn.Module | None = None,
    ) -> None:
        """Apply EMA update: θ_t ← m · θ_t + (1 − m) · θ_s."""
        _ema_update(self.teacher_backbone, student_backbone, self.momentum)
        _ema_update(self.teacher_meta_encoder, student_meta_encoder, self.momentum)
        if self.teacher_proj_head is not None and student_proj_head is not None:
            _ema_update(self.teacher_proj_head, student_proj_head, self.momentum)


class EmbeddingBank(nn.Module):
    """
    Fixed-size ring buffer of L2-normalised embeddings indexed by dataset position.

    Each of the bank_size slots corresponds to one training sample. A slot is
    updated by calling update(indices, embeddings) whenever that sample is
    processed in a batch. is_ready() returns True once every slot has been
    written at least once, signalling that CTLS-SSL Phase 2 neighbor mining
    can begin.

    get_neighbors computes a full [B, bank_size] cosine similarity matrix.
    For CIFAR-10 with embedding_dim=64 this is 50k × 64 = ~12 MB — acceptable
    on GPU. For larger datasets, replace with approximate nearest-neighbor search.
    """

    def __init__(self, bank_size: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.bank_size = bank_size
        self.embedding_dim = embedding_dim

        # Bank is not a learnable parameter — register as a buffer so it
        # moves with the module and is included in state_dict for checkpointing.
        self.register_buffer(
            "bank", torch.zeros(bank_size, embedding_dim)
        )
        # Track which slots have been written at least once
        self.register_buffer(
            "_written", torch.zeros(bank_size, dtype=torch.bool)
        )

    def update(self, indices: torch.Tensor, embeddings: torch.Tensor) -> None:
        """
        Write L2-normalised embeddings into the bank at specified positions.

        Args:
            indices:    [B] int64 — dataset positions (from MultiViewDataset idx)
            embeddings: [B, D]   — embeddings to store (will be L2-normalised)
        """
        normed = F.normalize(embeddings.detach(), dim=-1)
        self.bank[indices] = normed
        self._written[indices] = True

    def get_neighbors(
        self,
        query_embeddings: torch.Tensor,
        k: int = 1,
        exclude_self_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find k nearest neighbors for each query embedding.

        Args:
            query_embeddings:    [B, D] L2-normalised query vectors
            k:                   number of neighbors to return
            exclude_self_indices: [B] int64 — dataset positions to exclude from
                                  each query's neighbor search (prevents a sample
                                  from matching itself in the bank)

        Returns:
            neighbor_embeddings: [B, k, D]
            neighbor_indices:    [B, k] int64
        """
        # Cosine similarity: [B, bank_size]
        sim = torch.mm(query_embeddings, self.bank.t())

        if exclude_self_indices is not None:
            # Mask out the self-entry for each query
            B = query_embeddings.shape[0]
            row_idx = torch.arange(B, device=sim.device)
            sim[row_idx, exclude_self_indices] = -float("inf")

        # Top-k neighbors
        top_sim, top_idx = sim.topk(k, dim=1, largest=True, sorted=True)  # [B, k]
        neighbor_embeddings = self.bank[top_idx]  # [B, k, D]

        return neighbor_embeddings, top_idx

    def is_ready(self) -> bool:
        """Returns True once every bank slot has been written at least once."""
        return bool(self._written.all().item())


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """In-place EMA update of teacher parameters from student parameters."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
