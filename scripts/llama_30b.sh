#!/bin/bash

# Set common variables
model="decapoda-research/llama-30b-hf"
sparsity_ratio=0.5
cuda_devices="0,1"

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_devices

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3
}

# llama-30b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/llama_30b/unstructured/wanda/"
run_python_command "wanda" "2:4" "out/llama_30b/2-4/wanda/"
run_python_command "wanda" "4:8" "out/llama_30b/4-8/wanda/"
echo "Finished wanda pruning method"

# llama-30b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/llama_30b/unstructured/sparsegpt/"
run_python_command "sparsegpt" "2:4" "out/llama_30b/2-4/sparsegpt/"
run_python_command "sparsegpt" "4:8" "out/llama_30b/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

# llama-30b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/llama_30b/unstructured/magnitude/"
run_python_command "magnitude" "2:4" "out/llama_30b/2-4/magnitude/"
run_python_command "magnitude" "4:8" "out/llama_30b/4-8/magnitude/"
echo "Finished magnitude pruning method"