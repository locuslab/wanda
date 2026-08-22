"""Prefix-stage, reordered SparseGPT pruning.

Columns are ranked once per linear layer, then processed left-to-right in
that reordered coordinate system. A target global sparsity ``p`` activates
only the first ``p / 0.8`` of columns; every activated compensation segment
selects an approximately 80% SparseGPT mask from its current weights.
"""

import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT


MAX_SPARSITY = 0.8


def prefix_column_count(
    columns: int,
    sparsity: float,
    max_sparsity: float = MAX_SPARSITY,
) -> int:
    """Return the reordered-column prefix active at a target sparsity."""
    if columns < 1:
        raise ValueError("columns must be positive")
    if not 0.0 <= sparsity <= max_sparsity:
        raise ValueError("sparsity must be in [0, max_sparsity]")
    return round(columns * sparsity / max_sparsity)


def iter_prefix_segments(
    columns: int,
    sparsity: float,
    blocksize: int,
    max_sparsity: float = MAX_SPARSITY,
) -> list[tuple[int, int]]:
    """Split the active prefix into compensation segments of at most blocksize."""
    if blocksize < 1:
        raise ValueError("blocksize must be positive")
    prefix = prefix_column_count(columns, sparsity, max_sparsity)
    return [
        (start, min(start + blocksize, prefix))
        for start in range(0, prefix, blocksize)
    ]


def iter_stage_segments(
    columns: int,
    previous_sparsity: float,
    sparsity: float,
    blocksize: int,
    max_sparsity: float = MAX_SPARSITY,
) -> list[tuple[int, int]]:
    """Split only the newly activated prefix interval into compensation segments."""
    if not 0.0 <= previous_sparsity < sparsity <= max_sparsity:
        raise ValueError("stage sparsities must satisfy 0 <= previous < current <= max")
    if blocksize < 1:
        raise ValueError("blocksize must be positive")

    start = prefix_column_count(columns, previous_sparsity, max_sparsity)
    end = prefix_column_count(columns, sparsity, max_sparsity)
    return [
        (segment_start, min(segment_start + blocksize, end))
        for segment_start in range(start, end, blocksize)
    ]


def allocate_segment_budgets(
    rows: int,
    segments: list[tuple[int, int]],
    total_budget: int,
) -> list[int]:
    """Distribute an exact target count proportionally over active segments."""
    capacities = [rows * (end - start) for start, end in segments]
    total_capacity = sum(capacities)
    if not 0 <= total_budget <= total_capacity:
        raise ValueError("total_budget must fit inside the active prefix")

    budgets = []
    cumulative_capacity = 0
    allocated = 0
    for capacity in capacities:
        cumulative_capacity += capacity
        cumulative_budget = round(total_budget * cumulative_capacity / total_capacity)
        budget = cumulative_budget - allocated
        budgets.append(budget)
        allocated += budget
    return budgets


def invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    """Return indices that undo a one-dimensional permutation."""
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=permutation.device
    )
    return inverse


def column_permutation(
    weight: torch.Tensor,
    hessian_factor: torch.Tensor,
) -> torch.Tensor:
    """Rank all input columns from low to high SparseGPT score."""
    denominator = torch.diag(hessian_factor).reshape(1, -1).square()
    column_scores = (weight.float().square() / denominator).mean(dim=0)
    return torch.argsort(column_scores, stable=True)


def build_block_mask(
    scores: torch.Tensor,
    sparsity: float | None = None,
    budget: int | None = None,
) -> torch.Tensor:
    """Prune an exact stable prefix of the lowest scores in one segment."""
    if (sparsity is None) == (budget is None):
        raise ValueError("provide exactly one of sparsity or budget")
    flat_scores = scores.reshape(-1)
    if sparsity is not None:
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError("sparsity must be in [0, 1]")
        budget = round(sparsity * flat_scores.numel())
    if not 0 <= budget <= flat_scores.numel():
        raise ValueError("budget must fit inside the block")

    order = torch.argsort(flat_scores, stable=True)
    flat_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    flat_mask[order[:budget]] = True
    return flat_mask.reshape_as(scores)


class ElasticSparseGPT(SparseGPT):
    """SparseGPT with full-layer reordering and staged prefix pruning."""

    def fasterprune(
        self,
        sparsity: float,
        previous_sparsity: float = 0.0,
        permutation: torch.Tensor | None = None,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> torch.Tensor:
        """Prune only one newly opened prefix stage and return its permutation."""
        segments = iter_stage_segments(
            self.columns,
            previous_sparsity,
            sparsity,
            blocksize,
        )
        target_budget = (
            round(sparsity * self.rows * self.columns)
            - round(previous_sparsity * self.rows * self.columns)
        )
        segment_budgets = allocate_segment_budgets(
            self.rows,
            segments,
            target_budget,
        )

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diagonal_indices = torch.arange(self.columns, device=self.dev)
        H[diagonal_indices, diagonal_indices] += damp

        hessian_inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
        if permutation is None:
            original_factor = torch.linalg.cholesky(hessian_inverse, upper=True)
            permutation = column_permutation(W, original_factor)
        else:
            permutation = permutation.to(self.dev)
        inverse_permutation = invert_permutation(permutation)

        W = W[:, permutation]
        hessian_inverse = hessian_inverse[permutation][:, permutation]
        hessian_factor = torch.linalg.cholesky(hessian_inverse, upper=True)

        for (block_start, block_end), budget in zip(segments, segment_budgets):
            block_weight = W[:, block_start:block_end].clone()
            quantized = torch.zeros_like(block_weight)
            errors = torch.zeros_like(block_weight)
            block_factor = hessian_factor[block_start:block_end, block_start:block_end]

            scores = block_weight.square() / torch.diag(block_factor).reshape(
                1, -1
            ).square()
            block_mask = build_block_mask(scores, budget=budget)

            for column in range(block_end - block_start):
                weight = block_weight[:, column]
                diagonal = block_factor[column, column]

                pruned_weight = weight.clone()
                pruned_weight[block_mask[:, column]] = 0
                quantized[:, column] = pruned_weight

                error = (weight - pruned_weight) / diagonal
                block_weight[:, column:] -= error.unsqueeze(1).matmul(
                    block_factor[column, column:].unsqueeze(0)
                )
                errors[:, column] = error

            W[:, block_start:block_end] = quantized
            W[:, block_end:] -= errors.matmul(
                hessian_factor[block_start:block_end, block_end:]
            )

        W = W[:, inverse_permutation]
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        return permutation.detach().cpu()
