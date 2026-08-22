# Elastic Reordered SparseGPT: Feasibility Prototype

Date: 2026-08-21
Status: Proposed for implementation

## 1. Objective

Build a controlled prototype on top of SparseGPT to test this hypothesis:

> If input columns are ordered once from least to most important, then a sequence of global sparsity targets can be represented by progressively exhausting column groups while retaining SparseGPT's unstructured, Hessian-aware mask and error compensation inside each group.

The prototype must preserve the model's external tensor layout. Column reordering is only an internal pruning order; weights are restored to their original column positions before being written back to the layer.

## 2. Phase-one scope

Phase one is an effect-validation experiment. It implements:

- per-linear-layer global column ranking;
- configurable allocation groups, normally 5 or 10;
- exact global unstructured sparsity allocation across those groups;
- frozen within-group rankings that produce nested masks across target sparsities;
- the original SparseGPT Hessian error-compensation update;
- a separate CLI pruning method so the original `sparsegpt` path remains unchanged;
- deterministic unit tests for ordering, budgets, nestedness, and column restoration.

Phase one intentionally runs one requested sparsity ratio per model invocation. Multiple invocations can therefore compare quality at 10%, 20%, and later ratios, but they do not yet constitute the final "one computation produces multiple full checkpoints" system.

This boundary is deliberate: it tests whether the proposed mask family has acceptable accuracy before adding batched compensation, multi-branch activation propagation, and checkpoint materialization costs.

## 3. Non-goals

The first prototype does not:

- claim end-to-end multi-model speedup;
- generate several full checkpoints in one invocation;
- share a single compensated weight tensor between different masks;
- support structured `n:m` sparsity in the reordered path;
- change the existing WANDA or SparseGPT implementations;
- add the reordered path to `main_opt.py` or `lib/prune_opt.py`.

## 4. Definitions

For a linear layer with weight matrix

\[
W \in \mathbb{R}^{R \times C},
\]

let the calibration Hessian approximation be

\[
H \in \mathbb{R}^{C \times C}.
\]

After applying SparseGPT's dead-column handling and diagonal damping, compute the full inverse and its original-order upper Cholesky factor

\[
G = H^{-1}, \qquad U_0 = \operatorname{chol}(G, \text{upper}=\text{true}).
\]

The original-order weight-level score used only to rank columns is the same form used by SparseGPT:

\[
S^{order}_{ij} = \frac{W_{ij}^{2}}{(U_0)_{jj}^{2}}.
\]

The column score is the mean weight score in that column:

\[
C_j = \frac{1}{R}\sum_i S^{order}_{ij}.
\]

Columns are sorted in ascending order of `C_j`; lower-scored columns are processed first. Stable sorting is required so tied scores are deterministic.

## 5. Reordering and Hessian handling

Let `perm` be the ascending column permutation and `inv_perm` its inverse.

The pruning workspace uses:

\[
W' = W[:, perm], \qquad G' = G[perm, perm].
\]

SparseGPT's upper-triangular compensation factor is then computed from `G'`:

\[
U = \operatorname{chol}(G', \text{upper}=\text{true}).
\]

It is not valid to permute only `W`, or to permute a previously computed triangular factor. The full inverse must be permuted on both axes before taking the upper Cholesky factor.

The frozen score used for masks after reordering is recomputed in the permuted workspace:

\[
S^{mask}_{ij} = \frac{(W'_{ij})^{2}}{U_{jj}^{2}}.
\]

Thus the column permutation is selected once from the original-order SparseGPT metric, while within-group masks use the SparseGPT metric that corresponds to the actual reordered compensation factor.

After pruning and compensation, restore the original input-column layout:

\[
\widehat{W} = \widehat{W}'[:, inv\_perm].
\]

## 6. Allocation groups and sparsity budget

The reordered columns are divided into `K` contiguous allocation groups. `torch.tensor_split` semantics are used so group widths differ by at most one when `C` is not divisible by `K`.

For target global sparsity `s`, the exact layer budget is

\[
T = \operatorname{round}(s \cdot R \cdot C).
\]

Groups receive that budget from left to right:

1. fully prune a group if the remaining budget is at least its number of weights;
2. assign the remaining budget to the first partially covered group;
3. assign zero pruning budget to all later groups.

For equal-width groups this is equivalent to

\[
p_b(s) = \operatorname{clip}\left(\frac{sC-bB}{B}, 0, 1\right),
\]

where `b` is zero-based and `B` is the group width. With five groups, 10% global sparsity therefore prunes 50% of group 1; 20% prunes all of group 1; 30% prunes all of group 1 plus 50% of group 2.

This is weight sparsity within column groups, not whole-column structured pruning.

## 7. Nested mask construction

Within each allocation group, flatten the frozen scores `S_mask` and compute one stable ascending ranking. For a group budget `q`, mask the first `q` ranked weights.

The score tensor and all rankings are computed before any compensation update and never recomputed from partially pruned weights. Therefore, for two targets `s1 <= s2`, the prototype must satisfy:

\[
M(s_1) \subseteq M(s_2).
\]

The implementation must use integer budgets and ranking prefixes rather than floating thresholds. This avoids tie-induced over-pruning and gives an exact number of zeroed weights before compensation.

## 8. SparseGPT compensation

The original SparseGPT computational block size, default 128 columns, is independent of the allocation-group count. The full fixed mask is constructed first, then sliced by the existing 128-column compensation loop.

For each processing column, the prototype keeps the original SparseGPT operations:

- take the current compensated column `w`;
- set entries selected by the fixed mask to zero to obtain `q`;
- compute the normalized error from the diagonal element of `U`;
- update remaining columns in the current computational block;
- propagate accumulated error to all later computational blocks.

Each target sparsity has a different fixed mask and therefore requires its own compensation trajectory in phase one. Sharing calibration statistics and rankings is valid; sharing a single final compensated weight matrix is not.

## 9. Code organization

The prototype will add `lib/elastic_sparsegpt.py` containing:

- pure helpers for exact group boundaries and budgets;
- pure helpers for the stable column permutation and nested masks;
- `ElasticSparseGPT`, an experimental pruning class that owns the reordered fixed-mask compensation path.

The existing `lib/sparsegpt.py` will remain behaviorally unchanged.

`lib/prune.py` will add a `prune_elastic_sparsegpt` orchestration function that mirrors the current layerwise calibration flow but constructs `ElasticSparseGPT` objects and invokes the experimental method.

`main.py` will add:

- pruning method `elastic_sparsegpt`;
- `--reorder_blocks`, default `5`;
- validation that `0 <= sparsity_ratio <= 1`;
- validation that `reorder_blocks >= 1`;
- rejection of nonzero `prune_n` or `prune_m` for this experimental method.

No existing CLI meaning changes.

## 10. Validation and failure behavior

The experimental method must fail early with a clear error when:

- the sparsity ratio lies outside `[0, 1]`;
- the allocation-group count is below 1;
- structured `n:m` arguments are supplied;
- a layer has fewer columns than requested groups.

Numerical behavior follows SparseGPT:

- dead Hessian diagonal entries are replaced and their matching weight columns are zeroed;
- damping is applied before inversion;
- CUDA cache cleanup remains layer-local;
- the final tensor is cast back to the layer's original dtype.

## 11. Tests

`tests/test_elastic_sparsegpt.py` will use small CPU tensors and cover:

1. stable low-to-high column ordering;
2. correct inverse permutation and restored column layout;
3. balanced group boundaries for divisible and non-divisible column counts;
4. exact global mask counts for 0%, intermediate, and 100% sparsity;
5. the five-group 10%, 20%, and 30% allocation examples;
6. mask nestedness over an increasing list of sparsity ratios;
7. deterministic behavior when scores tie;
8. finite compensated output with the requested mask positions zero;
9. validation errors for unsupported inputs;
10. confirmation that the original `SparseGPT.fasterprune` entry point is untouched and still callable.

A server-side quality experiment will then compare original SparseGPT and `elastic_sparsegpt` at identical sparsity ratios, calibration samples, seed, model, and evaluation dataset. The first decision metric is perplexity degradation at each ratio; pruning time is recorded only as a diagnostic in phase one.

## 12. Acceptance criteria

Phase one is complete when:

- all new CPU unit tests pass;
- the existing pruning method remains selectable and unchanged;
- every generated experimental mask has the exact requested integer budget;
- masks generated from one frozen plan are nested;
- compensated weights are restored to their original column layout;
- one small end-to-end smoke run completes without NaN or shape errors;
- the server experiment can be launched with documented commands for baseline and experimental methods.

## 13. Follow-up decision

Only if the quality curve is competitive will phase two add a true multi-rate execution path. That design will need to choose explicitly between:

- batched independent compensation states, which preserves each fixed-mask trajectory but uses memory proportional to the number of target rates; or
- progressive checkpoint snapshots from one trajectory, which is cheaper but is not equivalent to independently compensated SparseGPT models.

It must also define how calibration activations are propagated across layers for multiple sparsity branches. Those choices are excluded from phase one so its results isolate the mask proposal itself.
