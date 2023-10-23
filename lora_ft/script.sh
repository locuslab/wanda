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

CUDA_VISIBLE_DEVICES=0 python evaluate_ppl.py \
    --model [PATH to load sparse pruned LLaMA-7B] \
    --lora_weights [PATH to load the LoRA weights]