from transformers import AutoTokenizer 
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn 
from peft import PeftModel, PeftConfig 
from tqdm import tqdm
import sys 
import json
import time  
import os 

import fnmatch

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    try:
        layers = model.model.layers
    except:
        layers = model.model.model.layers 
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            cur_zeros = (W==0).sum().item()
            cur_total = W.numel()

            count += cur_zeros 
            total_params += cur_total 

            print(f"layer {i} name {name} {W.shape} sparsity {float(cur_zeros)/cur_total}")

    print(f"total number of params {total_params}")
    model.config.use_cache = use_cache
    return float(count)/total_params

def evaluate_ppl(dataset_name, model, tokenizer, ctx_length):
    # max_length = model.seqlen 
    model_seqlen = ctx_length
    max_length = ctx_length
    stride = ctx_length

    if dataset_name == "wikitext":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
    elif dataset_name == "ptb":
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        encodings = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        seq_len = encodings.input_ids.size(1)
    elif dataset_name == "c4":
        valdata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )
        encodings = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        # encodings = encodings.input_ids[:, :(256 * model.seqlen)]
        seq_len = 256 * model_seqlen

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item() 

def eval_llm(model, tokenizer, task_list=["boolq","piqa","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], num_fewshot=0):
    from lm_eval import tasks, evaluator
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args="pretrained=decapoda-research/llama-7b-hf",
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None, 
        # device='cuda:0',
        device=None,
        no_cache=True,
        limit=None,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer 
    )

    return results 

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16, cache_dir=args.cache_dir, low_cpu_mem_usage=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(
        "lmsys/vicuna-13b-delta-v0",
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True,
    )

    model = PeftModel.from_pretrained(model,args.lora_weights,torch_dtype=torch.float16)

    model.eval()

    ppl = evaluate_ppl("wikitext", model, tokenizer, args.ctx_length)
    print(f"perplexity on wikitext {ppl}")

    if args.eval_zero_shot:
        task_list_dict = {0: ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]}
        accelerate=False 
        for num_shot in [0]:
            task_list = task_list_dict[num_shot]
            results = eval_llm(model, tokenizer, task_list, num_shot)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str
    )
    parser.add_argument(
        '--cache_dir', type=str, default="llm_weights"  
    )
    parser.add_argument(
        '--lora_weights', type=str, default=None 
    )
    parser.add_argument(
        '--ctx_length', type=int, default=2048 
    )
    parser.add_argument("--eval_zero_shot", action="store_true")

    args = parser.parse_args()
    main(args)