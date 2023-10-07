# Pruning LLMs by Weights and Activations
Official PyTorch implementation of **Wanda** (Pruning by **W**eights **and a**ctivations), as presented in our paper:

[A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695).  
[Mingjie Sun*](https://eric-mingjie.github.io/), [Zhuang Liu*](https://liuzhuang13.github.io/), [Anna Bair](https://annaebair.github.io/), [J. Zico Kolter](http://zicokolter.com/) (* indicates equal contribution)  
Carnegie Mellon University, Meta AI and Bosch Center for AI  

--- 
<p align="center">
<img src="https://user-images.githubusercontent.com/20168304/270077956-5d66e843-9dde-4439-aef6-abf876563ef7.png" width=100% height=100% 
class="center">
</p>

Compared to magnitude pruning which removes weights solely based on their magnitudes, our pruning approach **Wanda** removes weights on a per-output basis, by the product of weight magnitudes and input activation norms.

## Update
- [x] (9.22.2023) Add [support](https://github.com/locuslab/wanda#pruning-llama-2) for LLaMA-2.
- [x] (9.22.2023) Add [code](https://github.com/locuslab/wanda#ablation-on-obs-weight-update) to reproduce the ablation study on OBS weight update in the paper.
- [x] (10.6.2023) Add new [support](https://github.com/locuslab/wanda#ablation-on-obs-weight-update) for the weight update analysis in the ablation study. Feel free to try it out!

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
The [scripts](scripts) directory contains all the bash commands to replicate the main results (Table 2) in our paper.

Below is an example command for pruning LLaMA-7B with Wanda, to achieve unstructured 50% sparsity.
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/ 
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--use_variant`: Whether to use the Wanda variant, default is `False`. 
- `--save`: Specifies the directory where the result will be stored.

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/ 
```

### Pruning LLaMA-2
For [LLaMA-2](https://ai.meta.com/llama/) models, replace `--model` with `meta-llama/Llama-2-7b-hf` (take `7b` as an example):
```sh 
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/
```
LLaMA-2 results: (LLaMA-2-34b is not released as of 9.22.2023)
|sparsity| ppl              | llama2-7b | llama2-13b | llama2-70b |
|------|------------------|----------|------------|------------|
|-| dense            | 5.12     | 4.57       | 3.12     |
|unstructured 50%| magnitude        | 14.89    | 6.37       | 4.98     |
|unstructured 50%| sparsegpt        | 6.51     | 5.63       | **3.98**  |
|unstructured 50%| wanda            | **6.42** | **5.56**   | **3.98**  |
|4:8| magnitude        | 16.48    | 6.76       | 5.58     |
|4:8| sparsegpt        | 8.12     | 6.60      | 4.59     |
|4:8| wanda            | **7.97** | **6.55**  | **4.47**     |
|2:4| magnitude        | 54.59    | 8.33       | 6.33       |
|2:4| sparsegpt        | **10.17** | 8.32       | 5.40      |
|2:4| wanda            | 11.02    | **8.27**   | **5.16**     |

### Ablation on OBS weight update
To reproduce the analysis on weight update, we provide our implementation for this ablation. All commands can be found in [this script](scripts/ablate_weight_update.sh).
```sh
for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model decapoda-research/llama-7b-hf \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --prune_method ${method} \
  --save out/llama_7b_ablation/unstructured/
done 
```
Here `ablate_{mag/wanda}_{seq/iter}` means that we use magnitude pruning or wanda to obtain the pruned mask at each layer, then apply weight update procedure with either a sequential style or an iterative style every 128 input channels. For details, please see Section 5 of our [paper](https://arxiv.org/abs/2306.11695).

For pruning image classifiers, see directory [image_classifiers](image_classifiers) for details.

## Acknowledgement
This repository is build upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Questions
Feel free to discuss papers/code with us through issues/emails!

mingjies at cs.cmu.edu  
liuzhuangthu at gmail.com 

## Citation
If you found this work useful, please consider citing:
```
@article{sun2023simple,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```
