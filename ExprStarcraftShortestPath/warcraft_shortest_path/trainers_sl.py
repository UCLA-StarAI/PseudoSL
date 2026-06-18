import random

import time
from abc import ABC, abstractmethod

import torch
torch.set_float32_matmul_precision('high')
import torch._dynamo as dynamo
dynamo.config.cache_size_limit=10000
import torch.nn.functional as F
from comb_modules.losses import HammingLoss
from comb_modules.dijkstra import ShortestPath
from logger import Logger
from models import get_model
from utils import AverageMeter, optimizer_from_string, customdefaultdict
from decorators import to_tensor, to_numpy
from . import metrics
from .metrics import compute_metrics
import numpy as np
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from .visualization import draw_paths_on_image

def get_trainer(trainer_name):
    trainers = {"Baseline": BaselineTrainer,"SL": SLTrainer}
    return trainers[trainer_name]

def get_neighbors(i,j):
    ret = []
    d = [(-1,0), (-1,-1), (0, -1), (1, -1), (1, 0)]
    for x,y in d:
        ii = i+x
        jj = j+y
        if ii >= 0 and jj >= 0 and ii < ROWS and jj < COLS:
            ret.append((ii,jj))
    return ret

################### Circuits ###################
import sys
import os
sys.setrecursionlimit(20000)

#sys.path.append(os.path.join(sys.path[0], '/space/ahmedk/semprola_grids/semprola' ,'grids'))
sys.path.append(os.path.join(sys.path[0], '/space/ahmedk/arsl/autoregressive-semantic-loss/detox/pypsdd'))
sys.path.append(os.path.join(sys.path[0], '/space/ahmedk/arsl/autoregressive-semantic-loss/detox'))

from typing import Dict, List, Tuple
from pypsdd import Vtree, SddManager, PSddManager, SddNode, Inst, io
@torch.compile(fullgraph=True)
def compute_layer(level: torch.Tensor, idx2primesub: torch.Tensor, d: torch.Tensor):
    d[level] = d[idx2primesub].sum(-2).logsumexp(-2)

#@torch.compile(fullgraph=True)
def levelwiseSL(levels: List[torch.Tensor], idx2primesub: torch.Tensor, d: torch.Tensor):
    for i, level in enumerate(levels):
        compute_layer(level, idx2primesub[level], d)
    #print(d[levels[-1]])
    return d[levels[-1]]

#@torch.compile(fullgraph=True, mode='reduce-overhead')
#def levelwiseSL(levels: List[torch.Tensor], idx2primesub: torch.Tensor, data: torch.Tensor):
#    for i, level in enumerate(levels):
#        data[level] = data[idx2primesub[level]].sum(-2).logsumexp(-2)
#    return data[levels[-1]]

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

from pysdd import sdd
pysdd_vtree = sdd.Vtree.from_file('data/warcraft_shortest_path/12x12/constraint_trimmed.vtree')
pysdd_manager = sdd.SddManager.from_vtree(pysdd_vtree)
pysdd_alpha = pysdd_manager.read_sdd_file('data/warcraft_shortest_path/12x12/constraint_trimmed.sdd'.encode())

vtree = Vtree.read('data/warcraft_shortest_path/12x12/constraint_trimmed.vtree')
manager = SddManager(vtree)
alpha = io.sdd_read('data/warcraft_shortest_path/12x12/constraint_trimmed.sdd', manager)
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
print(len(levels))

levels.reverse()

true_indices = torch.LongTensor([node.id for node in nodes if node.is_true()]).cuda()

literal_indices = torch.LongTensor([[node.id, node.literal] for node in nodes if node.is_literal()]).cuda()
literal_indices, literal_mask = literal_indices.unbind(-1)

literal_mask = literal_mask.abs() - 1, (literal_mask > 0).long()

idx2primesub = torch.zeros((id, max_elements, 2), dtype=torch.long)
for node in nodes:
    if node.is_decomposition():
        tmp = torch.LongTensor([[p.id, s.id] for p, s in node.positive_elements])
        idx2primesub[node.id] = torch.nn.functional.pad(tmp, (0,0,0, max_elements - len(tmp)), value=id)
idx2primesub = idx2primesub.cuda()

ID = id
################### Circuits ###################


# Circuit paths
import os
import sys

# Circuit imports
#from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE

cmpe = CircuitMPE(f'data/warcraft_shortest_path/12x12/constraint_trimmed.vtree', f'data/warcraft_shortest_path/12x12/constraint_trimmed.sdd')
#gate = None

e2i = torch.load('e2i.pt').cuda()

@torch.jit.script
def log1mexp(x):
    lt = (x < 0.6931471805599453094).logical_and(x > 0)
    gt = x >= 0.6931471805599453094
    res = torch.empty_like(x)
    res[lt] = torch.log(-torch.expm1(-x[lt]))
    res[gt] = torch.log1p(-torch.exp(-x[gt]))
    res = res.masked_fill_(x == 0, -float('inf'))
    return res

def pseudosl(model, img, sample=None, num_classes=2, seq_len=264):


    # Expand sample
    with torch.no_grad():

        sample = model.sample_stoch(img).cpu()
        batch_size = sample.size(0)

        # For a given sample x_{i}, ..., x_{n} we can
        # get p(x_{i}, ..., x_{n}). For all i, to
        # calculate p(x_{i}|x_{-i}), we need to
        # calculate p(x_{i}, ..., x_{n}) /
        # p(x_{i}, ..., x_{n}) + p(-x_{i}, ..., x_{n})
        # i.e. the joint divided by the marginal

        # Shape: [batch_size, seq_len, k, seq_len)
        samples = sample.unsqueeze(1).unsqueeze(1).repeat(1, seq_len, num_classes, 1)

        # We want to modify the samples such that we
        # have batch_size*seq_len*k samples, where for
        # each sample and each position we try one of
        # num_classes
        samples[torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1),
                torch.arange(seq_len).unsqueeze(-1).unsqueeze(0),
                torch.arange(num_classes).unsqueeze(0).unsqueeze(0),
                torch.arange(seq_len).unsqueeze(-1).unsqueeze(0)] = torch.arange(num_classes)

        #import pdb; pdb.set_trace()
        #img = img.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, samples.size(1), samples.size(2), samples.size(3), -1, -1, -1)

        # We want to batchc all these sample for a single
        # model evaluation
        samples = samples.reshape(-1, seq_len)

    # Compute the likelihood of expanded samples
    log_probs = model(img, F.pad(samples[:, :-1], (1,0), 'constant', 2).cuda())
    log_probs = F.logsigmoid(log_probs).clamp(max=-1e-7)
    log_probs = torch.stack((log1mexp(-log_probs), log_probs), dim=-1)

    #log_probs = log_probs.log_softmax(dim=-1)


    # Compute the loglikelihood of each sample
    log_probs = log_probs.gather(-1, samples.cuda().unsqueeze(-1)).squeeze().sum(-1)
    #log_probs = log_probs.gather(-1, samples.cuda()).squeeze().sum(-1) #TODO: FIX by shifting
    #log_probs[-2][torch.arange(264), samples[-2]].sum()

    # Compute pseudolikelihoods
    lit_weights = log_probs
    lit_weights = lit_weights.view(batch_size, seq_len, num_classes)
    lit_weights = lit_weights - lit_weights.logsumexp(-1, keepdim=True)
    # Correctness Check: assert torch.close(lit_weights.logsumexp(-1).exp(), 1)

    lit_weights = lit_weights.view(batch_size, seq_len, -1)
    lit_weights = lit_weights.flatten(start_dim=1)
    lit_weights = torch.stack((log1mexp(-lit_weights), lit_weights), dim=-1).permute(1, 2, 0)

    data = torch.empty(ID+1, batch_size, device='cuda')
    data[true_indices] = 0#1
    data[ID] = -1000#0
    data[literal_indices] = lit_weights[literal_mask[0], literal_mask[1]]
    res_sl = levelwiseSL(levels, idx2primesub, data)

    return (-res_sl).mean()

class ShortestPathAbstractTrainer(ABC):
    def __init__(
        self,
        *,
        train_iterator,
        test_iterator,
        metadata,
        use_cuda,
        batch_size,
        optimizer_name,
        optimizer_params,
        model_params,
        fast_mode,
        neighbourhood_fn,
        preload_batch,
        lr_milestone_1,
        lr_milestone_2,
        use_lr_scheduling,
        num_layers,
        num_reps,
        num_units,
        S,
        sl_weight
    ):

        self.fast_mode = fast_mode
        self.use_cuda = use_cuda
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.test_iterator = test_iterator
        self.train_iterator = train_iterator
        self.metadata = metadata
        self.grid_dim = 12#int(np.sqrt(self.metadata["output_features"]))
        self.neighbourhood_fn = neighbourhood_fn
        self.preload_batch = preload_batch
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_reps = num_reps
        cmpe.beta.num_reps = num_reps
        self.S = S
        self.sl_weight = sl_weight

        self.best_eval = -1

        self.model = None
        self.build_model(**model_params)

        if self.S > 0:
            cmpe.beta.overparameterize(S=self.S)

        #global gate
        #gate = DenseGatingFunction(cmpe.beta, gate_layers=[self.num_units] + [self.num_units]*self.num_layers, num_reps=self.num_reps).cuda()

        if self.use_cuda:
            self.model.to("cuda")

        #self.optimizer = optimizer_from_string(optimizer_name)(list(self.model.parameters()) + list(gate.parameters()), **optimizer_params)
        self.optimizer = optimizer_from_string(optimizer_name)(list(self.model.parameters()), **optimizer_params)

        self.use_lr_scheduling = use_lr_scheduling
        if use_lr_scheduling:
            self.scheduler = MultiStepLR(self.optimizer, milestones=[lr_milestone_1, lr_milestone_2], gamma=0.1)

        self.epochs = 0
        self.train_logger = Logger(scope="training", default_output="tensorboard")
        self.val_logger = Logger(scope="validation", default_output="tensorboard")

    def train_epoch(self):

        self.epochs += 1
        
        # Training Metrics
        batch_time = AverageMeter("Batch time")
        avg_loss = AverageMeter("Loss")

        # Set model and gating functions to train
        self.model.train()
        #gate.train()

        start = time.time()

        # Training data iterator
        iterator = self.train_iterator.get_epoch_iterator(batch_size=self.batch_size,
                number_of_epochs=1, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)

        # Start epoch
        for i, data in enumerate(iterator):

            #if i == 5:
            #    break

            start_time = time.time()
            input, true_path, true_weights = data["images"], data["labels"],  data["true_weights"]

            loss  = self.forward_pass(input, true_path, train=True, i=i)

            # update batch metrics
            avg_loss.update(loss.item(), input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            #print(batch_time)

        meters = [batch_time, avg_loss]
        meter_str = "\t".join([str(meter) for meter in meters])
        print(f"Epoch: {self.epochs}\t{meter_str}")

        if self.use_lr_scheduling:
            self.scheduler.step()

        #self.train_logger.log(avg_loss.avg, "loss")

        return {
            "train_loss": avg_loss.avg,
        }

    @torch.no_grad()
    def evaluate(self, print_paths=False):
        avg_metrics = defaultdict(AverageMeter)

        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        #gate.eval()

        # Test iterator
        iterator = self.test_iterator.get_epoch_iterator(batch_size=128,
                number_of_epochs=1, shuffle=False, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)

        #iterator = self.test_iterator.get_epoch_iterator(batch_size=64,
        #        number_of_epochs=1, shuffle=False, device='cuda', preload=self.preload_batch)

        a = None
        for i, data in enumerate(iterator):
            input, true_path, true_weights = (
                data["images"],#.contiguous(),
                data["labels"],#.contiguous(),
                data["true_weights"],#.contiguous(),
            )

            #if self.use_cuda:
            #    input = input.cuda()
            #    true_path = true_path.cuda()
            #    true_weights = true_weights.cuda()

            start_time = time.time()
            self.model.sample(input)
            accuracy, last_suggestion = self.forward_pass(input, true_path, train=False, i=i)
            print(time.time() - start_time)
            suggested_path = last_suggestion["suggested_path"]
            data.update(last_suggestion)

            if a is None:
                a = suggested_path.cpu().detach().numpy()
            else:
                a = np.concatenate((a, suggested_path.cpu().detach().numpy()), axis=0)

            evaluated_metrics = metrics.compute_metrics(true_paths=true_path,
            suggested_paths=suggested_path, true_vertex_costs=true_weights, e2i=e2i)
            avg_metrics["accuracy"].update(accuracy, input.size(0))
            for key, value in evaluated_metrics.items():
                avg_metrics[key].update(value, input.size(0))

            if self.fast_mode:
                break

        for key, avg_metric in avg_metrics.items():
            self.val_logger.log(avg_metric.avg, key=key)
        avg_metrics_values = dict([(key, avg_metric.avg) for key, avg_metric in avg_metrics.items()])

        return avg_metrics_values

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @abstractmethod
    def forward_pass(self, input, true_shortest_paths, train, i):
        pass

    def log(self, data, train, k=None, num=None):
        logger = self.train_logger if train else self.val_logger
        if not train:
            image = self.metadata['denormalize'](data["images"][k]).squeeze().astype(np.uint8)
            suggested_path = data["suggested_path"][k].squeeze()
            labels = data["labels"][k].squeeze()

            suggested_path_im = torch.ones((3, *suggested_path.shape))*255*suggested_path.cpu()
            labels_im = torch.ones((3, *labels.shape))*255*labels.cpu()
            image_with_path = draw_paths_on_image(image=image, true_path=labels, suggested_path=suggested_path, scaling_factor=10)

            logger.log(labels_im.data.numpy().astype(np.uint8), key=f"shortest_path_{num}", data_type="image")
            logger.log(suggested_path_im.data.numpy().astype(np.uint8), key=f"suggested_path_{num}", data_type="image")
            logger.log(image_with_path, key=f"full_input_with_path{num}", data_type="image")


class BaselineTrainer(ShortestPathAbstractTrainer):
    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=264, in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, label, train, i):

        # Get embedding
        #output = self.model(input, label[:, :-1])
        output = self.model(input, F.pad(label[:, :-1], (1,0), 'constant', 2))
        output = torch.sigmoid(output)
        flat_target = label.view(label.size()[0], -1)

        if train:

            # Cross-Entropy
            criterion = torch.nn.BCELoss(reduction='none')
            loss = criterion(output, flat_target).mean()
            #print(loss.item())
            return loss

        else:

            bsz = label.size()[0]
            flat_target = label.view(label.size()[0], -1)

            # Get point-wise accuracy
            accuracy = (output.round() * flat_target).sum() / flat_target.sum()

            # Get suggested_path
            suggested_path = output.view(label.shape).round()
            valid_paths = cmpe.get_tf_ac([[1-p, p] for p in suggested_path.unbind(axis=-1)]).bool()
            last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

            return accuracy, last_suggestion, valid_paths

    @torch.no_grad()
    def evaluate(self, print_paths=False):
        avg_metrics = defaultdict(AverageMeter)
        self.model.eval()
        #gate.eval()

        # Test iterator
        iterator = self.test_iterator.get_epoch_iterator(batch_size=128,
                number_of_epochs=1, shuffle=False, device='cuda', preload=self.preload_batch)

        a = None
        for i, data in enumerate(iterator):
            input, true_path, true_weights = (
                data["images"],#.contiguous(),
                data["labels"],#.contiguous(),
                data["true_weights"]#.contiguous(),
            )

            start = time.time()
            flat_target = true_path
            suggested_path = self.model.sample(input).float()
            print("Exact match: ", (suggested_path == flat_target).all(dim=1).sum())
            accuracy = (suggested_path * flat_target).sum() / flat_target.sum()
            print(accuracy)

            ####### SL #########
            suggested_path = suggested_path
            lit_weights = torch.stack((log1mexp(-suggested_path.log()), suggested_path.log()), dim=-1).permute(1, 2, 0)
            #lit_weights = lit_weights.flatten().unsqueeze(-1)
            #lit_weights = torch.stack((log1mexp(-lit_weights), lit_weights), dim=1)

            data = torch.empty(ID+1, true_path.size(0), device='cuda')
            data[true_indices] = 0#1
            data[ID] = -float('inf')#0
            data[literal_indices] = lit_weights[literal_mask[0], literal_mask[1]]
            valid_paths = levelwiseSL(levels, idx2primesub, data).exp().bool().squeeze()
            ####### SL #########

            #valid_paths = cmpe.get_tf_ac([[1-p, p] for p in suggested_path.unbind(axis=-1)]).bool()
            #import pdb; pdb.set_trace()

            # Get suggested_path
            #suggested_path = output.view(label.shape).round()
            #valid_paths = cmpe.get_tf_ac([[1-p, p] for p in suggested_path.unbind(axis=-1)]).bool()
            #accuracy, last_suggestion, valid_paths = self.forward_pass(input, true_path, train=False, i=i)
            print("valid_paths:", valid_paths.sum().item())
            #suggested_path = last_suggestion["suggested_path"]

            if a is None:
                a = suggested_path.cpu().detach().numpy()
            else:
                a = np.concatenate((a, suggested_path.cpu().detach().numpy()), axis=0)

            evaluated_metrics = metrics.compute_metrics(true_paths=true_path,
            suggested_paths=suggested_path, true_vertex_costs=true_weights, e2i=e2i, valid_paths=valid_paths)
            avg_metrics["accuracy"].update(accuracy.item(), input.size(0))
            avg_metrics["valid paths"].update((valid_paths.sum()/len(valid_paths)).item(), input.size(0))
            for key, value in evaluated_metrics.items():
                avg_metrics[key].update(value, input.size(0))

            if self.fast_mode:
                break

        for key, avg_metric in avg_metrics.items():
            self.val_logger.log(avg_metric.avg, key=key)
        avg_metrics_values = dict([(key, avg_metric.avg) for key, avg_metric in avg_metrics.items()])

        return avg_metrics_values

class SLTrainer(BaselineTrainer):
    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=264, in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, label, train, i):

        # Get embedding
        #output = self.model(input, label[:, :-1])
        output = self.model(input, F.pad(label[:, :-1], (1,0), 'constant', 2))
        flat_target = label.view(label.size()[0], -1)

        if train:
            logprobs = F.logsigmoid(output).clamp(max=-1e-7)

            # Cross-Entropy
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss = criterion(output, flat_target).mean()
            #print("CE", loss)

            assert(self.sl_weight != 0)
            #semantic_loss = pseudosl(self.model, input[:10])

            batch_size = 10#logprobs.size(0)
            lit_weights = logprobs[:10]
            lit_weights = torch.stack((log1mexp(-lit_weights), lit_weights), dim=-1).permute(1, 2, 0)

            data = torch.empty(ID+1, batch_size, device='cuda')
            data[true_indices] = 0#1
            data[ID] = -1000#0
            data[literal_indices] = lit_weights[literal_mask[0], literal_mask[1]]
            semantic_loss = levelwiseSL(levels, idx2primesub, data)
            #print("SL", semantic_loss)
            #semantic_loss = -cmpe.get_tf_ac([[log1mexp(-p), p] for p in logprobs.unbind(axis=-1)], log_space=True) 
            loss += (self.sl_weight * semantic_loss.mean())

            #print(f"Loss at iter {i}: {loss}")
            return loss

        else:
            bsz = label.size()[0]
            probs = torch.sigmoid(output)

            # Get point-wise accuracy
            accuracy = (probs.round() * flat_target).sum() / flat_target.sum()

            # Get suggested_path
            suggested_path = probs.view(label.shape).round()
            print("here")
            valid_paths = cmpe.get_tf_ac([[1-p, p] for p in suggested_path.unbind(axis=-1)]).bool()
            last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

            return accuracy, last_suggestion, valid_paths





#def _tup_cpu(tup, force=False):
#    if isinstance(tup,tuple) and isinstance(tup[0],tuple):
#        return _tup_cpu_gpt2(tup)
#    elif force or isinstance(tup, tuple):
#        return tuple([t.cpu() for t in tup])
#    elif tup: return tup.cpu()
#    else: return tup
#
#def forward(model, src, num_classes=9):
#    """Takes in LongTensor `src` of size [batch_size, seq_len] and produces logits
#    for next token prediction of size [batch_size, seq_len, vocab_size]."""
#
#    import pdb; pdb.set_trace()
#    one_hot_src = F.one_hot(src, num_classes=num_classes).float()
#    logits, misc_out = model(one_hot_src)
#
#    return {
#        "logits": logits,
#        "misc_output": misc_out
#    }
#
#def get_next_probs(rnn, x, batch_size=100, return_logits=False):
#    """Computes the probability distribution over the vocabulary for the next
#    term in a sequence. Returns this and resulting hidden state. Can specify a
#    temperature to divide the logits by prior to performing a softmax to change
#    how 'peaked' or 'flat' the distribution is."""
#
#    import pdb; pdb.set_trace()
#    prob_outputs = []
#    step_outputs = []
#
#    step_output = forward(rnn, src=x)
#
#    if not return_logits:
#        probs = torch.softmax(step_output['logits'], dim=-1)
#        prob_outputs.append(probs)
#
#    else:
#        prob_outputs.append(step_output['logits'])
#    step_outputs.append(step_output['misc_output'])
#
#    return torch.cat(prob_outputs,dim = 0), step_output
#
#@torch.no_grad()
#def sample_rnn(rnn, input, batch_size=27, num_steps=81):
#
#    # Set model to evaluation
#    rnn.eval()
#
#    import pdb; pdb.set_trace()
#    # Start token: Currently picket uniformly at random from
#    # all possible tokens TODO: using '<BOS>' token
#    src = input#torch.randint(low=0, high=8, size=(batch_size, 1), device='cuda')
#
#    samples = [src]
#    step_input = src
#    hidden_state = None
#    for _ in range(num_steps - 1):
#        probs, hidden_state = get_next_probs(rnn, step_input)
#        next_token_dist = torch.distributions.categorical.Categorical(probs=probs)
#        step_input = next_token_dist.sample()
#        samples.append(step_input)
#
#    samples = torch.cat(samples, dim=-1)
#
#    return samples.detach().cpu()
#
## @torch.compile
#def pseudosl(model, input, num_classes=2):
#
#    import pdb; pdb.set_trace()
#    sample = sample_rnn(model, input)
#
#    batch_size = sample.size(0)
#    seq_len = sample.size(-1)
#
#    # For a given sample x_{i}, ..., x_{n} we can
#    # get p(x_{i}, ..., x_{n}). For all i, to
#    # calculate p(x_{i}|x_{-i}), we need to
#    # calculate p(x_{i}, ..., x_{n}) /
#    # p(x_{i}, ..., x_{n}) + p(-x_{i}, ..., x_{n})
#    # i.e. the joint divided by the marginal
#
#    # Expand sample
#    with torch.no_grad():
#        samples = sample.unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len, num_classes, seq_len)
#        samples[:, torch.arange(seq_len), :, torch.arange(seq_len)] = torch.arange(num_classes)
#        samples = samples.reshape(-1, seq_len)
#
#
#        samples = samples.chunk(9)
#        batch_size = batch_size // 9
#
#    for i in range(len(samples)):
#
#        # Compute the likelihood of expanded samples
#        sample = samples[i].cuda()
#        probs = forward(model, samples[i].cuda())['logits']
#        log_probs = probs.log_softmax(dim=-1)
#        log_probs = log_probs.gather(-1, sample.unsqueeze(-1)).squeeze().logsumexp(-1)
#        sample = sample.detach().cpu()
#
#        # log_probs[i][j] corresponds to p(X_{1}, ..., X_{i-1}, X_{i} = x_{j}, X_{i+1}, ..., X_{n})
#        log_probs = log_probs.view(batch_size, seq_len, num_classes)
#
#        # Compute the conditional probabilities: log_probs[i][j] = p(X_{i}=x_{j}| X_{1}, ..., X_{i-1}, X_{i+1}, ..., X_{n})
#        log_probs = log_probs - log_probs.logsumexp(-1, keepdim=True)
#
#        probs = log_probs.view(-1, 9, 9, 9).exp()
#
#        row_probs = probs.flatten(end_dim=1)
#        col_probs = probs.permute((0, 2, 1, 3)).flatten(end_dim=1)
#        sqr_probs = torch.cat([probs[:, i:i+3, j:j+3, :].flatten(1, 2) for i in range(0, 9, 3) for j in range(0, 9, 3)])
#
#        probs = torch.cat((row_probs, col_probs, sqr_probs)).flatten(start_dim=1)
#        probs = probs.unbind(dim=1)
#        lit_weights = [[1-p, p] for p in probs]
#
#        sl = (-wmc(constraint, lit_weights, log_space=False).log()).chunk(3)
#        #rows.append(sl[0].view(-1, 9))
#        #cols.append(sl[1].view(-1, 9))
#        #sqrs.append(sl[2].view(-1, 9))
#        row_sl = torch.cat([row_sl, sl[0].view(-1, 9)])
#        col_sl = torch.cat([col_sl, sl[1].view(-1, 9)])
#        sqr_sl = torch.cat([sqr_sl, sl[2].view(-1, 9)])
#
#    return row_sl.mean(dim=0).sum() + col_sl.mean(dim=0).sum() + sqr_sl.mean(dim=0).sum()
