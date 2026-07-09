TOKEN = ""

import os
import subprocess
from pprint import pprint

from huggingface_hub import login
from tqdm.auto import tqdm

try:
    import lm_eval
except ImportError:
    subprocess.run(["pip", "install", "lm_eval"], check=True)
    import lm_eval

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.makedirs("pruned_models", exist_ok=True)
login(TOKEN)

prune_n = 2
prune_m = 4

sparsity_ratio: float = float(prune_n) / float(prune_m)
sparsity_type = f"{prune_n}:{prune_m}"

repository = 'wanda_vision'
repo_path = os.getcwd()

main_tests = [(model_name, pruning_dataset_name, prune_method)
              for pruning_dataset_name in ("magic_hq", "magic_qa")
              for prune_method in ("magnitude", "wanda", "sparsegpt")
              for model_name in ("google/gemma-3-270m", "google/gemma-3-1b-pt", "google/gemma-3-4b-pt")]

no_eval = False
for idx, (model_name, pruning_dataset_name, prune_method) in tqdm(enumerate(main_tests), leave=True):
    save_name = model_name.replace("/", "__") + f"-{pruning_dataset_name}" + f"-{prune_n}of{prune_m}"
    model_save_path = f"{repo_path}/pruned_models/{prune_method}/{save_name}"
    print(
        f"{idx}/{len(main_tests)}: Pruning '{model_name}' using '{prune_method}' method on '{pruning_dataset_name}' data"
        f"\n\twith {sparsity_type} sparsity (ratio = {sparsity_ratio}).")
    print(f"Model will be saved at '{model_save_path}'")

    script = "main.py"

    cmd = [
        "python", script,
        "--model", model_name,
        "--pruning_dataset_name", pruning_dataset_name,
        "--prune_method", prune_method,
        "--sparsity_ratio", str(sparsity_ratio),
        "--sparsity_type", sparsity_type,
        "--save_model", model_save_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except:
        os.makedirs(f"{model_save_path}_FAIL", exist_ok=True)

print("Done")

pruned_model_folder = f"{repo_path}/pruned_models/"

all_files = []
for root, dirs, files in os.walk(pruned_model_folder):
    for file in files:
        all_files.append(os.path.join(root, file))

pprint(all_files)

no_eval = True
if not no_eval:
    for model_save_path in all_files:
        if "FAIL" in model_save_path:
            continue
        subprocess.run([
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_save_path},max_position_embeddings=4096",
            "--tasks", "wikitext,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,sciq",
            "--batch_size", "auto",
            "--output_path", "results"
        ], check=True)
else:
    print("No eval")
