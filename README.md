# Pruning LLMs by Weights and Activations
Official PyTorch implementation of **Wanda** (Pruning by **W**eights **and a**ctivations), as presented in our paper:

[A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695).  
[Mingjie Sun*](https://eric-mingjie.github.io/), [Zhuang Liu*](https://liuzhuang13.github.io/), [Anna Bair](https://annaebair.github.io/), [J. Zico Kolter](http://zicokolter.com/) (* indicates equal contribution)  
Carnegie Mellon University, Meta AI and Bosch Center for AI  

--- 
<p align="center">
<img src="https://user-images.githubusercontent.com/20168304/245999360-f951de47-269d-491d-826a-8e6d85627849.png" width=100% height=100% 
class="center">
</p>

Compared to magnitude pruning which removes weights solely based on their magnitudes, our pruning approach **Wanda** removes weights on a per-output basis, by the product of weight magnitudes and input activation norms.


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
