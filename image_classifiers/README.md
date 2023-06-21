# Pruning Image Classifiers
Here we provide the code for pruning ConvNeXt and ViT. This part is built on the [dropout](https://github.com/facebookresearch/dropout) repository.

## Environment
We additionally install `timm` for loading pretrained image classifiers.
```
pip install timm==0.4.12
```

## Download Weights
Run the script [download_weights.sh](download_weights.sh) to download pretrained weights for ConvNeXt-B, DeiT-B and ViT-L, which we used in the paper.

## Usage
Here is the command for pruning ConvNeXt/ViT models:
```
python main.py --model [ARCH] \
    --data_path [PATH to ImageNet] \
    --resume [PATH to the pretrained weights] \
    --prune_metric wanda \
    --prune_granularity row \
    --sparsity 0.5 
```
where:
- `--model`: network architecture, choices [`convnext_base`, `deit_base_patch16_224`, `vit_large_patch16_224`].
- `--resume`: model path to downloaded pretrained weights.
- `--prune_metric`: [`magnitude`, `wanda`].
- `--prune_granularity`: [`layer`, `row`].