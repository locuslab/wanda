# Installation  
Step 1: Create a new conda environment:
```
conda create -n wanda_vision python=3.11
conda activate wanda_vision
```
Step 2: Install relevant packages
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers==4.51.0 datasets==3.3.0 wandb sentencepiece==0.2.0
pip install accelerate==1.3.0
```