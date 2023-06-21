#!/bin/bash

# Set common variables
model="decapoda-research/llama-65b-hf"
sparsity_ratio=0.5

# Define function to run python command
run_python_command () {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model $model \
    --prune_method $2 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $3 \
    --save $4
}

# llama-65b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "0,1,2,3,4" "wanda" "unstructured" "out/llama_65b/unstructured/wanda/"
run_python_command "0,1,2,3,4" "wanda" "2:4" "out/llama_65b/2-4/wanda/"
run_python_command "0,1,2,3,4" "wanda" "4:8" "out/llama_65b/4-8/wanda/"
echo "Finished wanda pruning method"

# llama-65b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "0,1,2,3,4" "sparsegpt" "unstructured" "out/llama_65b/unstructured/sparsegpt/"
run_python_command "0,1,2,3,4" "sparsegpt" "2:4" "out/llama_65b/2-4/sparsegpt/"
run_python_command "0,1,2,3,4" "sparsegpt" "4:8" "out/llama_65b/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

# llama-65b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "0,1,2,3" "magnitude" "unstructured" "out/llama_65b/unstructured/magnitude/"
run_python_command "0,1,2,3" "magnitude" "2:4" "out/llama_65b/2-4/magnitude/"
run_python_command "0,1,2,3" "magnitude" "4:8" "out/llama_65b/4-8/magnitude/"
echo "Finished magnitude pruning method"