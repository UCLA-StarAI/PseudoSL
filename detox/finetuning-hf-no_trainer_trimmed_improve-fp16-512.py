#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
os.environ['TRANSFORMERS_CACHE'] = '/space/ahmedk/cache/'
os.environ['HF_DATASETS_CACHE'] ="/space/ahmedk/cache/"
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
torch.set_float32_matmul_precision('high')
import torch._dynamo as dynamo
torch._dynamo.config.verbose=True
dynamo.config.cache_size_limit=10000
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import GPT2Tokenizer
import itertools
def create_constraint():
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=True)
    
    with open("bad_words.txt") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        data.append(line.rstrip())
        data.append(' ' + line.rstrip())

    words = tokenizer(data)['input_ids']
    
    # Get unique words and map them to [0, len(unique_words))
    unique_words = torch.tensor(list(itertools.chain.from_iterable(words))).unique().tolist()

    global num_tokens
    num_tokens = len(unique_words) + 1

    global token_idx
    token_idx = num_tokens - 1
    
    tokenid2varid = dict()
    for i, word in enumerate(unique_words):
        tokenid2varid[word] = i
    
    global idxmap
    idxmap = torch.full((50257,), num_tokens-1)
    idxmap[list(tokenid2varid.keys())] = torch.tensor(list(tokenid2varid.values()))
    
    print("Number of tokens", num_tokens)
    
    return

# Used as a hack to create idxmap
create_constraint()

from transformers import (
    GenerationConfig,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
logging.getLogger("transformers").setLevel(logging.ERROR)


import accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository, create_repo
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

logger = get_logger(__name__)
transformers.logging.set_verbosity_error()

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

##################### start pseudo-sl #####################
import sys
sys.setrecursionlimit(20000)
sys.path.append("pypsdd")

from pypsdd import Vtree, SddManager, PSddManager, SddNode, Inst, io
from pypsdd import UniformSmoothing, Prior
from pysdd import sdd
import time
import torch
from typing import Dict, List, Tuple

#@torch.compile(fullgraph=True, mode='reduce-overhead')
#def log1mexp(x):
#    # Source: https://github.com/wouterkool/estimating-gradients-without-replacement/blob/9d8bf8b/bernoulli/gumbel.py#L7-L11
#    # Computes log(1-exp(-|x|))
#    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
#    x = -x.abs()
#    x = torch.where(
#        x > -0.6931471805599453094,
#        torch.log(-torch.expm1(x)),
#        torch.log1p(-torch.exp(x)),
#    )
#
#    return x

@torch.jit.script
def log1mexp(x):
    lt = (x < 0.6931471805599453094).logical_and(x > 0)
    gt = x >= 0.6931471805599453094
    res = torch.empty_like(x)
    res[lt] = torch.log(-torch.expm1(-x[lt]))
    res[gt] = torch.log1p(-torch.exp(-x[gt]))
    res = res.masked_fill_(x == 0, -float('inf'))
    return res

#@torch.compile(fullgraph=True)
#def compute_layer(level: torch.Tensor, idx2primesub: torch.Tensor, d: torch.Tensor):
#    d[level] = d[idx2primesub].sum(-2).logsumexp(-2)

@torch.compile(fullgraph=True, mode='reduce-overhead')
def levelwiseSL(levels: List[torch.Tensor], idx2primesub: torch.Tensor, data: torch.Tensor):
    for level in levels:
        data[level] = data[idx2primesub[level]].sum(-2).logsumexp(-2)
    return data[levels[-1]]

#def levelwiseSL(levels: List[torch.Tensor], idx2primesub: torch.Tensor, d: torch.Tensor):
#    for i, level in enumerate(levels):
#        compute_layer(level, idx2primesub[level], d)
#    #print(d[levels[-1]])
#    return d[levels[-1]]

def levelOrder(beta):
    """
    :type root: Node
    :rtype: List[List[int]]
    """
    seen = dict()
    nodes = [beta]
    level = []
    answer = []
    result = [[beta]]
    while len(nodes) != 0:
        for a in nodes:
            if not a.is_decomposition():
                continue
            for element in a.positive_elements:
                for e in element:
                    if not e.is_decomposition():
                        continue
                    if seen.get(e) != None: continue
                    seen[e] = True
                    level.append(e)
        nodes = level
        for i in level:
            answer.append(i)
        level = []
        answer = list(dict.fromkeys(answer))
        result.append(answer)
        answer = []
    return result[:-1]

# Creating the words constraint
from pysdd import sdd
pysdd_vtree = sdd.Vtree.from_file('all_words_trimmed-10_spaces_min.vtree')
pysdd_manager = sdd.SddManager.from_vtree(pysdd_vtree)
pysdd_alpha = pysdd_manager.read_sdd_file('all_words_trimmed-10_spaces_min.sdd'.encode())

vtree = Vtree.read('all_words_trimmed-10_spaces_min.vtree')
manager = SddManager(vtree)
alpha = io.sdd_read('all_words_trimmed-10_spaces_min.sdd', manager)
pmanager = PSddManager(vtree)
beta = pmanager.copy_and_normalize_sdd(alpha, vtree)

print("Num. decision nodes: ", beta.count())

max_elements = 0
for node in beta.positive_iter():
    if node.is_decomposition():
        max_elements = max(max_elements, len(node.positive_elements))

levels_nodes = levelOrder(beta)
#del levels_nodes[-1][-1]

# Reset ids
nodes = [node for node in beta.positive_iter()]
nodes = list(dict.fromkeys(nodes))

id = 0
for e in nodes:
    e.id = id
    id += 1 

levels = []
for level in levels_nodes:
    levels.append(torch.tensor([l.id for l in level], dtype=torch.long, device='cuda'))
print("num levels:", len(levels))

levels.reverse()

true_indices = torch.LongTensor([node.id for node in nodes if node.is_true()])

literal_indices = torch.LongTensor([[node.id, node.literal] for node in nodes if node.is_literal()]) 
literal_indices, literal_mask = literal_indices.unbind(-1)

literal_mask = literal_mask.abs() - 1, (literal_mask > 0).long()

idx2primesub = torch.zeros((id, max_elements, 2), dtype=torch.long)
for node in nodes:
    if node.is_decomposition():
        tmp = torch.LongTensor([[p.id, s.id] for p, s in node.positive_elements])
        idx2primesub[node.id] = torch.nn.functional.pad(tmp, (0,0,0, max_elements - len(tmp)), value=id)
idx2primesub = idx2primesub.cuda()

ID = id

# Creating the exactly-one constraint
from pysdd import sdd
pysdd_vtree = sdd.Vtree.from_file('exactly_one_trimmed-10_spaces_min.vtree')
pysdd_manager = sdd.SddManager.from_vtree(pysdd_vtree)
pysdd_alpha = pysdd_manager.read_sdd_file('exactly_one_trimmed-10_spaces_min.sdd'.encode())

eg_vtree = Vtree.read('exactly_one_trimmed-10_spaces_min.vtree')
eg_manager = SddManager(eg_vtree)
eg_alpha = io.sdd_read('exactly_one_trimmed-10_spaces_min.sdd', eg_manager)
eg_pmanager = PSddManager(eg_vtree)
eg_beta = eg_pmanager.copy_and_normalize_sdd(eg_alpha, eg_vtree)

#print("Num. decision nodes: ", eg_beta.count())

max_elements = 0
for node in eg_beta.positive_iter():
    if node.is_decomposition():
        max_elements = max(max_elements, len(node.positive_elements))

levels_nodes = levelOrder(eg_beta)
#del levels_nodes[-1][-1]

# Reset ids
nodes = [node for node in eg_beta.positive_iter()]
nodes = list(dict.fromkeys(nodes))

eg_id = 0
for e in nodes:
    e.id = eg_id
    eg_id += 1 

eg_levels = []
for level in levels_nodes:
    eg_levels.append(torch.tensor([l.id for l in level], dtype=torch.long, device='cuda'))

eg_levels.reverse()

eg_true_indices = torch.LongTensor([node.id for node in nodes if node.is_true()])

eg_literal_indices = torch.LongTensor([[node.id, node.literal] for node in nodes if node.is_literal()]) 
eg_literal_indices, eg_literal_mask = eg_literal_indices.unbind(-1)

eg_literal_mask = eg_literal_mask.abs() - 1, (eg_literal_mask > 0).long()

eg_idx2primesub = torch.zeros((eg_id, max_elements, 2), dtype=torch.long)
for node in nodes:
    if node.is_decomposition():
        tmp = torch.LongTensor([[p.id, s.id] for p, s in node.positive_elements])
        eg_idx2primesub[node.id] = torch.nn.functional.pad(tmp, (0,0,0, max_elements - len(tmp)), value=eg_id)
eg_idx2primesub = eg_idx2primesub.cuda()

eg_ID = eg_id

# END exactly-one constraint

def pseudosl(model, sample=None, num_classes=450, seq_len=10):

    sample, attn_mask = sample['input_ids'].cpu(), sample['attention_mask'].cpu()
    batch_size = sample.size(0)

    # For a given sample x_{i}, ..., x_{n} we can
    # get p(x_{i}, ..., x_{n}). For all i, to
    # calculate p(x_{i}|x_{-i}), we need to
    # calculate p(x_{i}, ..., x_{n}) /
    # p(x_{i}, ..., x_{n}) + p(-x_{i}, ..., x_{n})
    # i.e. the joint divided by the marginal
    

    start_time = time.time()
    # Expand sample
    with torch.no_grad():

        # We'll only consider the top-k tokens under 
        # the pseudolikelihood distribution centered 
        # around the samples
        # Output Shape: [batch_size, seq_len, k]
        logits = model(sample.cuda(), attention_mask=attn_mask.cuda())['logits'].cpu()
        logits = logits.sort(descending=True)
        topk_good = logits.indices[(idxmap[logits.indices] == token_idx)].view(1, seq_len, 50257-token_idx)[:, :, :9]
        topk_bad = logits.indices[(idxmap[logits.indices] != token_idx)].view(1, seq_len, token_idx)[:,:,:num_classes-10]
        topk_tokens = torch.cat((topk_good, topk_bad, sample.unsqueeze(-1)), dim=-1)
        del logits
        del topk_good
        del topk_bad

        # Shape: [batch_size, seq_len, k, seq_len)
        samples = sample.unsqueeze(1).unsqueeze(1).repeat(1, seq_len, num_classes, 1) 

        # We want to modify the samples such that we
        # have batch_size*seq_len*k samples, where for
        # each sample and each position we try one of
        # k tokens
        samples[torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1),
                torch.arange(seq_len).unsqueeze(-1).unsqueeze(0),
                torch.arange(num_classes).unsqueeze(0).unsqueeze(0),
                torch.arange(seq_len).unsqueeze(-1).unsqueeze(0)] = topk_tokens

        # We want to batchc all these sample for a single
        # model evaluation
        samples = samples.reshape(-1, seq_len)
    

    # Compute the likelihood of expanded samples
    log_probs = model(samples.cuda(), attention_mask=attn_mask.unsqueeze(1).\
            unsqueeze(1).expand(batch_size, seq_len, num_classes, seq_len).\
            reshape(-1, seq_len).cuda())['logits']
    log_probs = log_probs.log_softmax(dim=-1)
    
    # Compute the loglikelihood of each sample
    log_probs = log_probs.gather(-1, samples.cuda().unsqueeze(-1)).squeeze().sum(-1) #TODO: FIX by shifting
    
    del samples
    
    start_time = time.time()
    # Compute pseudolikelihoods
    lit_weights = log_probs
    lit_weights = lit_weights.view(batch_size, seq_len, num_classes)
    lit_weights = lit_weights - lit_weights.logsumexp(-1, keepdim=True)
    # Correctness Check: assert torch.close(lit_weights.logsumexp(-1).exp(), 1)

    # We want to map the precicted top-k tokens to bad tokens 0 
    # through 616, and everything else to the "good" catch-all token token_idx
    # Everything that is not in the top-k gets a log-prob of (-inf)

    # tmp is going to hold our literal weights
    tmp = torch.full((batch_size, seq_len, num_tokens), float(-300), device='cuda') #TODO: Check 618
    
    # map the predicted top-k tokens to 0-token_idx
    indices = idxmap[topk_tokens.flatten()].view(batch_size, seq_len, num_classes) #TODO: What do the normal tokens map to?
    # Correctness check: indices[0][19] == idxmap[topk_tokens[0][19]]

    # "bad" tokens get their probabilities straight from the predictions
    # after mapping to the right indices through "indices"
    tmp[torch.arange(batch_size).unsqueeze(1).unsqueeze(1),
        torch.arange(seq_len).unsqueeze(-1), indices] = lit_weights
    # Correctness: tmp[0][19][indices[0][19]] == lit_weights[0][19] -- check needs fixing
    
    # All the probability mass that is not allocated to "bad" tokens
    # is allocated to the catch-all 'good' token token_idx
    lit_weights[(indices!=token_idx).nonzero(as_tuple=True)] = -300#float('inf')
    tmp[:, :, token_idx] = lit_weights.logsumexp(-1)
    # Correctness: lit_weights[0][19][(idxmap[topk_tokens[0][19]]==token_idx).nonzero(as_tuple=False)].logsumexp(0) == tmp[0][19][token_idx]

    lit_weights = tmp.view(batch_size, seq_len, -1)
    
    lit_weights = lit_weights.flatten(start_dim=1)
    lit_weights = torch.stack((log1mexp(-lit_weights), lit_weights), dim=-1).permute(1, 2, 0)
    #print("time", time.time() - start_time)
    
    data = torch.empty(ID+1, batch_size, device='cuda')
    data[true_indices] = 0#1
    data[ID] = -1000#0
    data[literal_indices] = lit_weights[literal_mask[0], literal_mask[1]]
    res_sl = levelwiseSL(levels, idx2primesub, data)
    
    del data

    data_eg = torch.empty(eg_ID+1, batch_size, device='cuda')
    data_eg[eg_true_indices] = 0#1
    data_eg[eg_ID] = -1000#0
    data_eg[eg_literal_indices] = lit_weights[eg_literal_mask[0], eg_literal_mask[1]]
    res_eg = levelwiseSL(eg_levels, eg_idx2primesub, data_eg)
    
    return (-res_sl + res_eg).clamp(min=0).mean()


##################### end pseudo-sl #####################

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=0,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--sl_weight",
        type=float,
        default=0.0,
        help="Semantic Loss Weight",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    print(sys.argv)
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, device_placement=True, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            device_map="auto"
        )
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        train_dataset = tokenized_datasets['train'].map(group_texts, batched=True, num_proc=args.preprocessing_num_workers, load_from_cache_file=not args.overwrite_cache, desc=f"Grouping texts in chunks of {block_size}")
        block_size = 10
        eval_dataset = tokenized_datasets['validation'].map(group_texts, batched=True, num_proc=args.preprocessing_num_workers, load_from_cache_file=not args.overwrite_cache, desc=f"Grouping texts in chunks of {block_size}")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=1
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch
    generation_config = GenerationConfig.from_pretrained("gpt2", do_sample=True,
     max_length=100, num_return_sequences=1, top_p=0.9, top_k=0, output_scores=False,
     return_dict_in_generate=True)

    # Start of training loop
    accelerator.free_memory()
    for epoch in range(starting_epoch, args.num_train_epochs):
        
        # training setup
        model.train()
        optimizer.zero_grad()

        eval_iterator = iter(eval_dataloader)

        accumulate = 4
        tot_loss = 0
        tot_sl = 0
        for step, batch in enumerate(train_dataloader):
            
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            # Compute Cross-Entropy loss and backward
            inputs = batch['input_ids'].cuda()
            attn = batch['attention_mask'].cuda()
            loss = model(inputs, attention_mask=attn, labels=inputs).loss/accumulate
            tot_loss += loss.item()

            loss.backward()

            # Compute Pseudo-Semantic Loss if weight is not 0
            if args.sl_weight != 0:
                accumulation_steps = 10
                for i in range(accumulation_steps):
                    try:
                        sample = next(eval_iterator)
                    except:
                        eval_iterator = iter(eval_dataloader)

                    # We take the average pseudo-semantic loss across
                    # 10 steps and scale it by the the loss weight
                    ssl = (args.sl_weight*pseudosl(model, sample))/(accumulation_steps*accumulate)
                    tot_sl += ssl.item()
                    ssl.backward()

            
            if (step+1) % accumulate == 0:
                
                print("loss", tot_loss)
                print("psl", tot_sl)

                tot_loss = 0
                tot_sl = 0

                # Take optimization step and prepare scaler for next training iteration
                optimizer.step()
                lr_scheduler.step()

                # Update state
                progress_bar.update(1)
                completed_steps += 1

                # Checkpointing
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0 and completed_steps != 0 :
                        output_dir = f"step_{completed_steps}_{args.sl_weight}_{args.learning_rate}_512"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.wait_for_everyone()
                        model.save_pretrained(output_dir)
                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
