## Dense Fine-tuning

We provide a sparse Trainer in [sparse_trainer.py](sparse_trainer.py), which can be used as a drop-in-replacement of huggingface Trainer. In `training_step` function, when the gradients are obtained, `mask_grad()` will zero out the gradient corresponding to the pruned weights.