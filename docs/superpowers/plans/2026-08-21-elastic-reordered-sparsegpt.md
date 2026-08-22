# Elastic Reordered SparseGPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental SparseGPT path that globally reorders input columns, allocates an exact sparsity budget across 5 or 10 ordered groups, builds nested unstructured masks, and retains Hessian error compensation.

**Architecture:** Pure tensor helpers construct deterministic permutations, balanced groups, exact group budgets, and frozen nested masks. A separate `ElasticSparseGPT` class performs reordered fixed-mask compensation without altering the original `SparseGPT`; `lib/prune.py` and `main.py` expose the experiment through an isolated CLI method.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-21-elastic-reordered-sparsegpt-design.md`

## Global Constraints

- Keep `lib/sparsegpt.py` behaviorally unchanged.
- Phase one accepts one `--sparsity_ratio` per invocation.
- Support unstructured sparsity only; reject nonzero `prune_n` or `prune_m`.
- `reorder_blocks` must be at least 1 and no greater than the layer's input-column count.
- Use exact integer mask budgets and stable ranking prefixes; do not use score thresholds.
- Reorder both weight columns and both axes of the full Hessian inverse, then recompute the upper Cholesky factor.
- Restore the original column layout before assigning the pruned tensor to the layer.
- Keep allocation-group boundaries independent of the 128-column SparseGPT compute block size.
- Preserve the pre-existing uncommitted runtime-reporting changes in `main.py`, `main_opt.py`, `lib/reporting.py`, and `tests/test_reporting.py`; the elastic implementation may extend `main.py` but must not remove or rewrite that behavior.

---

## File Structure

- Create `lib/elastic_sparsegpt.py`: pure planning helpers and the experimental compensated pruner.
- Create `tests/test_elastic_sparsegpt.py`: CPU-only behavioral and numerical tests.
- Modify `lib/prune.py`: layerwise calibration orchestration for the experimental class.
- Modify `main.py`: CLI choice, block-count argument, input validation, and dispatch.
- Modify `README.md`: reproducible baseline and experimental commands.

### Task 1: Exact Group Allocation and Nested Mask Helpers

**Files:**
- Create: `lib/elastic_sparsegpt.py`
- Create: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Produces: `balanced_group_slices(columns: int, groups: int) -> list[tuple[int, int]]`
- Produces: `allocate_group_budgets(rows: int, columns: int, sparsity: float, slices: list[tuple[int, int]]) -> list[int]`
- Produces: `invert_permutation(permutation: torch.Tensor) -> torch.Tensor`
- Produces: `column_permutation(weight: torch.Tensor, hessian_factor: torch.Tensor) -> torch.Tensor`
- Produces: `build_grouped_mask(scores: torch.Tensor, slices: list[tuple[int, int]], budgets: list[int]) -> torch.Tensor`

- [ ] **Step 1: Write failing tests for balanced slices and exact sequential budgets**

```python
import unittest
import torch

from lib.elastic_sparsegpt import (
    allocate_group_budgets,
    balanced_group_slices,
)


class GroupAllocationTests(unittest.TestCase):
    def test_balances_non_divisible_columns(self):
        self.assertEqual(
            balanced_group_slices(columns=12, groups=5),
            [(0, 3), (3, 6), (6, 8), (8, 10), (10, 12)],
        )

    def test_five_groups_follow_ten_twenty_thirty_percent_example(self):
        slices = balanced_group_slices(columns=10, groups=5)
        self.assertEqual(allocate_group_budgets(2, 10, 0.10, slices), [2, 0, 0, 0, 0])
        self.assertEqual(allocate_group_budgets(2, 10, 0.20, slices), [4, 0, 0, 0, 0])
        self.assertEqual(allocate_group_budgets(2, 10, 0.30, slices), [4, 2, 0, 0, 0])

    def test_ten_groups_are_supported(self):
        slices = balanced_group_slices(columns=20, groups=10)
        self.assertEqual(allocate_group_budgets(3, 20, 0.10, slices), [6] + [0] * 9)
```

- [ ] **Step 2: Run allocation tests and verify the import fails**

Run: `python -m unittest tests.test_elastic_sparsegpt.GroupAllocationTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'lib.elastic_sparsegpt'`.

- [ ] **Step 3: Implement validated balanced slices and integer budgets**

```python
def balanced_group_slices(columns: int, groups: int) -> list[tuple[int, int]]:
    if groups < 1:
        raise ValueError("reorder_blocks must be at least 1")
    if groups > columns:
        raise ValueError("reorder_blocks cannot exceed the layer column count")
    base, extra = divmod(columns, groups)
    widths = [base + (index < extra) for index in range(groups)]
    slices = []
    start = 0
    for width in widths:
        end = start + width
        slices.append((start, end))
        start = end
    return slices


def allocate_group_budgets(rows, columns, sparsity, slices):
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1]")
    remaining = round(sparsity * rows * columns)
    budgets = []
    for start, end in slices:
        capacity = rows * (end - start)
        budget = min(remaining, capacity)
        budgets.append(budget)
        remaining -= budget
    return budgets
```

- [ ] **Step 4: Run allocation tests and verify they pass**

Run: `python -m unittest tests.test_elastic_sparsegpt.GroupAllocationTests -v`

Expected: 3 tests pass.

- [ ] **Step 5: Write failing tests for stable ordering, inversion, exact masks, ties, and nestedness**

```python
from lib.elastic_sparsegpt import (
    build_grouped_mask,
    column_permutation,
    invert_permutation,
)


class MaskPlanningTests(unittest.TestCase):
    def test_columns_are_ordered_from_low_to_high_sparsegpt_score(self):
        weight = torch.tensor([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
        factor = torch.eye(3)
        self.assertTrue(torch.equal(column_permutation(weight, factor), torch.tensor([1, 2, 0])))

    def test_inverse_permutation_restores_columns(self):
        weight = torch.arange(12).reshape(3, 4)
        permutation = torch.tensor([2, 0, 3, 1])
        restored = weight[:, permutation][:, invert_permutation(permutation)]
        self.assertTrue(torch.equal(restored, weight))

    def test_group_mask_uses_exact_stable_prefixes(self):
        scores = torch.ones(2, 4)
        slices = [(0, 2), (2, 4)]
        mask = build_grouped_mask(scores, slices, [3, 0])
        expected = torch.tensor([[True, True, False, False], [True, False, False, False]])
        self.assertTrue(torch.equal(mask, expected))

    def test_masks_are_nested(self):
        scores = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        slices = balanced_group_slices(10, 5)
        masks = [
            build_grouped_mask(scores, slices, allocate_group_budgets(2, 10, ratio, slices))
            for ratio in (0.10, 0.20, 0.30)
        ]
        self.assertTrue(torch.all(masks[0] <= masks[1]))
        self.assertTrue(torch.all(masks[1] <= masks[2]))
        self.assertEqual([int(mask.sum()) for mask in masks], [2, 4, 6])
```

- [ ] **Step 6: Run mask tests and verify missing functions fail**

Run: `python -m unittest tests.test_elastic_sparsegpt.MaskPlanningTests -v`

Expected: FAIL because the new helper functions are not defined.

- [ ] **Step 7: Implement stable planning helpers**

```python
def invert_permutation(permutation):
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse


def column_permutation(weight, hessian_factor):
    denominator = torch.diag(hessian_factor).reshape(1, -1).square()
    column_scores = (weight.float().square() / denominator).mean(dim=0)
    return torch.argsort(column_scores, stable=True)


def build_grouped_mask(scores, slices, budgets):
    mask = torch.zeros_like(scores, dtype=torch.bool)
    rows = scores.shape[0]
    for (start, end), budget in zip(slices, budgets):
        group_scores = scores[:, start:end].reshape(-1)
        order = torch.argsort(group_scores, stable=True)
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask[order[:budget]] = True
        mask[:, start:end] = group_mask.reshape(rows, end - start)
    return mask
```

- [ ] **Step 8: Run all helper tests**

Run: `python -m unittest tests.test_elastic_sparsegpt -v`

Expected: all Task 1 tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add lib/elastic_sparsegpt.py tests/test_elastic_sparsegpt.py
git commit -m "feat: add nested SparseGPT mask planning"
```

### Task 2: Reordered Fixed-Mask SparseGPT Compensation

**Files:**
- Modify: `lib/elastic_sparsegpt.py`
- Modify: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Consumes: all Task 1 helpers.
- Produces: `ElasticSparseGPT(SparseGPT)`.
- Produces: `ElasticSparseGPT.fasterprune(sparsity: float, reorder_blocks: int = 5, blocksize: int = 128, percdamp: float = 0.01) -> None`.

- [ ] **Step 1: Write a failing CPU compensation test**

```python
from torch import nn
from lib.elastic_sparsegpt import ElasticSparseGPT


class ElasticCompensationTests(unittest.TestCase):
    def test_compensation_is_finite_exact_and_restores_layout(self):
        layer = nn.Linear(4, 2, bias=False)
        layer.weight.data.copy_(torch.tensor([[4.0, 1.0, 3.0, 2.0], [8.0, 5.0, 7.0, 6.0]]))
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(4)

        pruner.fasterprune(sparsity=0.25, reorder_blocks=2, blocksize=2, percdamp=0.01)

        self.assertTrue(torch.isfinite(layer.weight).all())
        self.assertEqual(int((layer.weight == 0).sum()), 2)
        self.assertEqual(tuple(layer.weight.shape), (2, 4))

    def test_rejects_more_groups_than_columns(self):
        layer = nn.Linear(3, 2, bias=False)
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(3)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            pruner.fasterprune(0.5, reorder_blocks=4)
```

- [ ] **Step 2: Run compensation tests and verify the class is missing**

Run: `python -m unittest tests.test_elastic_sparsegpt.ElasticCompensationTests -v`

Expected: FAIL because `ElasticSparseGPT` is not defined.

- [ ] **Step 3: Implement Hessian preparation, synchronized permutation, fixed mask, compensation, and restoration**

Implement `ElasticSparseGPT` as a subclass of `SparseGPT`. Its `fasterprune` must:

```python
H = self.H
self.H = None
dead = torch.diag(H) == 0
H[dead, dead] = 1
W[:, dead] = 0
diag = torch.arange(self.columns, device=self.dev)
H[diag, diag] += percdamp * torch.mean(torch.diag(H))
G = torch.cholesky_inverse(torch.linalg.cholesky(H))
original_factor = torch.linalg.cholesky(G, upper=True)
permutation = column_permutation(W, original_factor)
inverse = invert_permutation(permutation)
W = W[:, permutation]
G = G[permutation][:, permutation]
factor = torch.linalg.cholesky(G, upper=True)
slices = balanced_group_slices(self.columns, reorder_blocks)
budgets = allocate_group_budgets(self.rows, self.columns, sparsity, slices)
scores = W.square() / torch.diag(factor).reshape(1, -1).square()
mask = build_grouped_mask(scores, slices, budgets)
```

Then copy the original SparseGPT 128-column compensation loop, replacing its threshold-created mask with `mask[:, i1:i2]`. After the loop, restore with `W = W[:, inverse]`, transpose only for `transformers.Conv1D`, reshape, cast to the original dtype, and assign to `self.layer.weight.data`. Call `torch.cuda.synchronize()` only when `self.dev.type == "cuda"` so CPU tests remain valid.

- [ ] **Step 4: Run compensation and helper tests**

Run: `python -m unittest tests.test_elastic_sparsegpt -v`

Expected: all tests pass.

- [ ] **Step 5: Confirm the original SparseGPT API remains intact**

Add and run:

```python
from lib.sparsegpt import SparseGPT

def test_original_sparsegpt_entry_point_is_unchanged(self):
    self.assertTrue(callable(SparseGPT.fasterprune))
```

Run: `python -m unittest tests.test_elastic_sparsegpt -v`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add lib/elastic_sparsegpt.py tests/test_elastic_sparsegpt.py
git commit -m "feat: add reordered SparseGPT compensation"
```

### Task 3: Layerwise Calibration Orchestration

**Files:**
- Modify: `lib/prune.py:1-8`
- Modify: `lib/prune.py:223-305`
- Modify: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Consumes: `ElasticSparseGPT` from Task 2.
- Produces: `validate_elastic_sparsegpt_args(args, prune_n: int, prune_m: int) -> None`.
- Produces: `prune_elastic_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0) -> None`.

- [ ] **Step 1: Write failing argument-validation tests**

```python
from types import SimpleNamespace
from lib.prune import validate_elastic_sparsegpt_args


class OrchestrationValidationTests(unittest.TestCase):
    def test_rejects_structured_sparsity(self):
        args = SimpleNamespace(sparsity_ratio=0.5, reorder_blocks=5)
        with self.assertRaisesRegex(ValueError, "unstructured"):
            validate_elastic_sparsegpt_args(args, prune_n=2, prune_m=4)

    def test_rejects_invalid_ratio(self):
        args = SimpleNamespace(sparsity_ratio=1.1, reorder_blocks=5)
        with self.assertRaisesRegex(ValueError, "\[0, 1\]"):
            validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0)

    def test_accepts_five_and_ten_groups(self):
        for groups in (5, 10):
            args = SimpleNamespace(sparsity_ratio=0.3, reorder_blocks=groups)
            validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0)
```

- [ ] **Step 2: Run validation tests and verify the import fails**

Run: `python -m unittest tests.test_elastic_sparsegpt.OrchestrationValidationTests -v`

Expected: FAIL because `validate_elastic_sparsegpt_args` is missing.

- [ ] **Step 3: Implement validation and the experimental orchestration function**

Add `from .elastic_sparsegpt import ElasticSparseGPT`. Implement validation exactly as:

```python
def validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0):
    if not 0.0 <= args.sparsity_ratio <= 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1]")
    if args.reorder_blocks < 1:
        raise ValueError("reorder_blocks must be at least 1")
    if prune_n != 0 or prune_m != 0:
        raise ValueError("elastic_sparsegpt supports unstructured sparsity only")
```

Create `prune_elastic_sparsegpt` by following the existing `prune_sparsegpt` calibration and layer-propagation flow. The only pruning-path differences are:

```python
gpts[name] = ElasticSparseGPT(subset[name])
gpts[name].fasterprune(
    args.sparsity_ratio,
    reorder_blocks=args.reorder_blocks,
    percdamp=0.01,
    blocksize=128,
)
```

Call `validate_elastic_sparsegpt_args` before loading calibration data. Do not refactor or change `prune_sparsegpt` in this task.

- [ ] **Step 4: Run the full CPU test module**

Run: `python -m unittest tests.test_elastic_sparsegpt -v`

Expected: all tests pass.

- [ ] **Step 5: Compile the changed modules**

Run: `python -m py_compile lib/elastic_sparsegpt.py lib/prune.py`

Expected: exit code 0 with no output.

- [ ] **Step 6: Commit Task 3**

```bash
git add lib/prune.py tests/test_elastic_sparsegpt.py
git commit -m "feat: orchestrate elastic SparseGPT pruning"
```

### Task 4: CLI Exposure and Dispatch

**Files:**
- Modify: `main.py:8`
- Modify: `main.py:29-41`
- Modify: `main.py:69-79`
- Modify: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Consumes: `prune_elastic_sparsegpt` from Task 3.
- Produces: CLI method `--prune_method elastic_sparsegpt`.
- Produces: CLI integer option `--reorder_blocks`, default `5`.

- [ ] **Step 1: Write a failing CLI parser behavior test**

```python
class CliExposureTests(unittest.TestCase):
    def test_parser_accepts_elastic_method_and_ten_groups(self):
        from main import build_parser

        args = build_parser().parse_args([
            "--prune_method", "elastic_sparsegpt",
            "--sparsity_type", "unstructured",
            "--reorder_blocks", "10",
        ])

        self.assertEqual(args.prune_method, "elastic_sparsegpt")
        self.assertEqual(args.reorder_blocks, 10)
```

- [ ] **Step 2: Run the CLI smoke test and verify it fails**

Run: `python -m unittest tests.test_elastic_sparsegpt.CliExposureTests -v`

Expected: FAIL because `main.build_parser` does not exist.

- [ ] **Step 3: Add the import, parser option, choice, and dispatch**

Extract parser creation into `build_parser()`, update `main()` to call `build_parser().parse_args()`, and update the pruning import to include `prune_elastic_sparsegpt`. Add `"elastic_sparsegpt"` to the `--prune_method` choices. Add:

```python
parser.add_argument(
    "--reorder_blocks",
    type=int,
    default=5,
    help="Number of ordered column-allocation groups for elastic_sparsegpt.",
)
```

Add the dispatch directly after the existing SparseGPT branch:

```python
elif args.prune_method == "elastic_sparsegpt":
    prune_elastic_sparsegpt(
        args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
    )
```

- [ ] **Step 4: Run tests and compile the entry point**

Run: `python -m unittest tests.test_elastic_sparsegpt -v`

Expected: all tests pass.

Run: `python -m py_compile main.py lib/prune.py lib/elastic_sparsegpt.py`

Expected: exit code 0 with no output.

- [ ] **Step 5: Commit Task 4**

```bash
git add main.py tests/test_elastic_sparsegpt.py
git commit -m "feat: expose elastic SparseGPT CLI"
```

### Task 5: Experiment Documentation and Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the CLI added in Task 4.
- Produces: copyable server commands for baseline, five-group, and ten-group experiments.

- [ ] **Step 1: Add experiment commands and interpretation notes**

Document commands using the same model, seed, sample count, sparsity ratio, and save directory pattern. Include at least:

```bash
python main.py \
  --model <MODEL_PATH> \
  --prune_method sparsegpt \
  --sparsity_type unstructured \
  --sparsity_ratio 0.3 \
  --nsamples 128 \
  --seed 0 \
  --save results/sparsegpt-030

python main.py \
  --model <MODEL_PATH> \
  --prune_method elastic_sparsegpt \
  --sparsity_type unstructured \
  --sparsity_ratio 0.3 \
  --reorder_blocks 5 \
  --nsamples 128 \
  --seed 0 \
  --save results/elastic-k5-030

python main.py \
  --model <MODEL_PATH> \
  --prune_method elastic_sparsegpt \
  --sparsity_type unstructured \
  --sparsity_ratio 0.3 \
  --reorder_blocks 10 \
  --nsamples 128 \
  --seed 0 \
  --save results/elastic-k10-030
```

State explicitly that `K=10` with 10%-spaced targets exhausts whole groups and can behave like grouped column pruning, while `K=5` leaves partial frontier groups at 10%, 30%, and 50%.

- [ ] **Step 2: Run the full verification suite**

Run: `python -m unittest tests.test_elastic_sparsegpt -v`

Expected: all tests pass.

Run: `python -m py_compile main.py lib/prune.py lib/elastic_sparsegpt.py tests/test_elastic_sparsegpt.py`

Expected: exit code 0 with no output.

Run: `git diff --check`

Expected: exit code 0 with no output.

- [ ] **Step 3: Inspect the final diff and repository status**

Run: `git diff --stat HEAD~4..HEAD && git status --short --branch`

Expected: only the design, plan, experimental implementation, tests, CLI integration, and README documentation are changed; no unrelated files are present.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/superpowers/plans/2026-08-21-elastic-reordered-sparsegpt.md
git commit -m "docs: add elastic SparseGPT experiment guide"
```

- [ ] **Step 5: Prepare server smoke commands without claiming quality results**

Run the documented commands first with a small accessible LLaMA-compatible model and reduced `--nsamples`, then run the controlled research model comparison. Report actual perplexity, measured sparsity, runtime, GPU peak memory, and the fraction of fully zero input columns for each result; do not claim the phase-two multi-model speedup from this prototype.
