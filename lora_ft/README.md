## LoRA Fine-tuning of pruned LLMs
Here we provide the script for the lora fine-tuning experiments in the paper. The commands for reproducing our experiments are in [script.sh](script.sh).

This codebase is based on [run_clm.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling). Here we adapt this code with LoRA fine-tuning on the C4 training dataset. Some custom changes we make in the code include:
- [loc 1](https://github.com/locuslab/wanda/blob/main/lora_ft/finetune_lm.py#L374): set up LLaMA-7B for LoRA fine-tuning;
- [loc 2](https://github.com/locuslab/wanda/blob/main/lora_ft/finetune_lm.py#L521): set up training arguments for Trainer.
- [loc 3](https://github.com/locuslab/wanda/blob/main/lora_ft/finetune_lm.py#L364): load the tokenizer from vicuna, which are the same as the original LLaMA tokenizer but also fix the issues of some special tokens.
- [loc 4](https://github.com/locuslab/wanda/blob/main/lora_ft/finetune_lm.py#L319): load the c4 training dataset.

To train a LoRA adapter, run the command:
```sh
CUDA_VISIBLE_DEVICES=0 python finetune_lm.py \
    --model_name_or_path [PATH to load sparse pruned LLaMA-7B] \
    --config_name "decapoda-research/llama-7b-hf" \
    --dataset_name c4 \
    --num_train_epochs 1 \
    --block_size 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir [PATH to save the LoRA weights]
```
We provide a quick overview of the arguments:  
- `--model_name_or_path`: The path/directory where pruned LLaMA-7B are saved with `model.save_pretrained(PATH)`.
- `--block_size`: context size, if you have 80GB gpu, you can set it to 2048;
- `--max_train_samples`: the number of training sequences, 30000 would lead to roughly 12 hours of training on 1 GPU;
- `--learning_rate`: the learning rate for LoRA fine-tuning;

We provide the code to evaluate LoRA adapter on WikiText validation dataset in [evaluate_ppl.py](evaluate_ppl.py). For zero shot evaluation, additionally pass the `--eval_zero_shot` argument.