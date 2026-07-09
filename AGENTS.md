# AGENTS.md

## Context

This is the `modern` branch of [Wanda](https://arxiv.org/abs/2306.11695) — pruning LLMs by weights & activations. The branch targets modern decoder-only families: **Qwen2.5/3, Gemma2/3, LLaMA**. OPT, GPT-2, and other legacy models were intentionally removed.

## Architecture

- Single pruning entry point: `main.py` → `lib/prune.py`
- `lib/prune.py` assumes `model.model.layers` (standard decoder-only transformer block)
- `lib/sparsegpt.py` — SparseGPT algorithm; `lib/ablate.py` — ablation variant; `lib/layerwrapper.py` — Wanda helper
- `real_main.py` — batch orchestrator for Gemma2 models
- `single_main.py` — batch orchestrator for Qwen2.5 models
- `image_classifiers/` — separate sub-project (Timm pruning), not related to LLM pipeline
- `dense_ft/trainer.py` — vendored old transformers internals, **broken** on modern transformers

## Environment

```
conda create -n wanda_vision python=3.11
pip install torch==2.5.1 transformers==4.51.0 datasets==3.3.0 accelerate==1.3.0 sentencepiece==0.2.0 wandb
```

This repo has no GPU in development. Target runtime is Colab/Kaggle/Vast.ai with GPU.

## Gotchas

- **`model.seqlen = 4096`** hardcoded in `main.py:26`. Adjust for models with longer context windows.
- **`torch.float16`** hardcoded in all `from_pretrained` calls. Gemma2 is natively bfloat16 — may need `torch.bfloat16` or `"auto"`.
- **`trust_remote_code=True`** is set in `main.py` for Gemma2/Qwen3 support.
- **`device` detection** uses `next(iter(model.parameters())).device` — old `model.hf_device_map` dict access was removed.
- **Manual rotary embedding calls** (`model.model.rotary_emb`) were replaced with `position_ids` kwarg for modern model compatibility.
- **No tests, no linter, no CI** in this repo.

## Key CLI

```bash
python main.py \
    --model Qwen/Qwen2.5-0.5B \
    --pruning_dataset_name modern \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save_model pruned_models/wanda/qwen
```

`--pruning_dataset_name` options: `c4`, `modern` (local parquet files in `../data/for_susi/`), `magic_hq`, `magic_qa`.
`--prune_method` options: `magnitude`, `wanda`, `sparsegpt`.
`--sparsity_type`: `unstructured` for unstructured, or `N:M` like `2:4`.

Batch orchestrators are run as standalone scripts:
```bash
python single_main.py --qwen_model_size 3 --wanda
python real_main.py  # loops over all gemma-3 configs in real_main.py
```

## What was modernized

| Change | Reason |
|---|---|
| PyTorch 1.10 → 2.5.1, transformers 4.28 → 4.51.0 | Qwen3 support |
| Removed `torch._six`, `transformers.Conv1D`, `torch.random.manual_seed` | API removals |
| Added `trust_remote_code=True` | Gemma2/Qwen3 loading |
| Replaced `hf_device_map` with `next(layer.parameters()).device` | accelerate API change |
| Manual RoPE → `position_ids` kwarg | Modern model layer forward compatibility |
| `prepare_model_for_int8_training` → `prepare_model_for_kbit_training` | PEFT API change |
| Removed `main_opt.py`, `lib/prune_opt.py`, `lib/prune_gemma.py` | Unused; Gemma works via `lib/prune.py` |
| `"hf-causal-experimental"` → `"hf"` in lm_eval calls | lm_eval API change |
| TF32 flags → `torch.set_float32_matmul_precision("highest")` | Torch API migration |
