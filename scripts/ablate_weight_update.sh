for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model decapoda-research/llama-7b-hf \
  --nsamples 128 \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --prune_method ${method} \
  --save out/llama_7b_ablation/unstructured/
done 

for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model decapoda-research/llama-7b-hf \
  --nsamples 128 \
  --sparsity_ratio 0.5 \
  --sparsity_type 4:8 \
  --prune_method ${method} \
  --save out/llama_7b_ablation/4:8/
done 

for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model decapoda-research/llama-7b-hf \
  --nsamples 128 \
  --sparsity_ratio 0.5 \
  --sparsity_type 2:4 \
  --prune_method ${method} \
  --save out/llama_7b_ablation/2:4/
done 
