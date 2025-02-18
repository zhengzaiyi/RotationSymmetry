import torch
import copy
from scipy.optimize import linear_sum_assignment
import json, os
import numpy as np

def get_submodule(m, name):
    ns = name.split(".")
    r = m
    for n in ns:
        r = r[n]
    return r
class Struct:
    def __init__(self) -> None:
        pass

    def __repr__(self, space=0) -> str:
        s = ""
        for k, v in self.__dict__.items():
            if type(v) is not Struct:
                s += "{}{}: {}\n".format(" " * space, k, v)
            else:
                s += "{}{}:\n{}".format(" " * space, k, v.__repr__(space=space + 2))
        return s

    def to_dict(self):
        dic = {}
        for k, v in vars(self).items():
            if type(v) is Struct:
                v = v.to_dict()
            dic[k] = v
        return dic

    def save(self, output_dir, filename):
        d = self.to_dict()
        with open(os.path.join(output_dir, filename), "w") as wf:
            json.dump(d, wf)

def dic_to_object(dic):
    obj = Struct()
    for k, v in dic.items():
        if type(v) is dict:
            sub_obj = dic_to_object(v)
            setattr(obj, k, sub_obj)
        else:
            setattr(obj, k, v)

    return obj


def get_QKVO(module, attn_pattern): # for attention
    Q = get_submodule(module, attn_pattern['query'])
    K = get_submodule(module, attn_pattern['key'])
    V = get_submodule(module, attn_pattern['value'])
    O = get_submodule(module, attn_pattern['output'])
        
    W_Q, B_Q = Q['weight'], Q['bias']
    W_K, B_K = K['weight'], K['bias']
    W_V, B_V = V['weight'], V['bias']
    
    W_O, B_O = O['weight'], O['bias']
    return {
        'W_Q': W_Q, 'B_Q': B_Q,
        'W_K': W_K, 'B_K': B_K,
        'W_V': W_V, 'B_V': B_V,
        'W_O': W_O, 'B_O': B_O
    }
    
def get_IO(module, ffn_pattern): # for FFN
    I = get_submodule(module, ffn_pattern['intermediate'])
    O = get_submodule(module, ffn_pattern['output'])
    W_I, B_I = I['weight'], I['bias']
    W_O_FFN, B_O_FFN = O['weight'], O['bias']
    return {
        'W_I': W_I, 'B_I': B_I,
        'W_O_FFN': W_O_FFN, 'B_O_FFN': B_O_FFN
    }
    
def Chunk_SVD_QK(
    local_param, anchor_param, 
    num_attention_heads=12,
    attention_head_size=32,
    use_scaling=False,
):
    W_Q1, W_K1, B_Q1, B_K1 = local_param['W_Q'], local_param['W_K'], local_param['B_Q'], local_param['B_K']
    W_Q2, W_K2, B_Q2, B_K2 = anchor_param['W_Q'], anchor_param['W_K'], anchor_param['B_Q'], anchor_param['B_K']
    W_Q, W_K, B_Q, B_K = torch.zeros_like(W_Q1), torch.zeros_like(W_K1), torch.zeros_like(B_Q1), torch.zeros_like(B_K1)
    if not num_attention_heads:
        num_attention_heads = W_Q1.shape[0] // attention_head_size
    for i in range(num_attention_heads):
        start, end = i * attention_head_size, (i + 1) * attention_head_size
        W_Q1_, W_K1_, B_Q1_, B_K1_ = W_Q1[start:end], W_K1[start:end], B_Q1[start:end], B_K1[start:end]
        W_Q2_, W_K2_, B_Q2_, B_K2_ = W_Q2[start:end], W_K2[start:end], B_Q2[start:end], B_K2[start:end]
        # summary = W_Q1_ @ W_Q2_.T + W_K1_ @ W_K2_.T
        summary = W_Q1_ @ W_Q2_.T + W_K1_ @ W_K2_.T + B_Q1_.unsqueeze(0).T @ B_Q2_.unsqueeze(0) + B_K1_.unsqueeze(0).T @ B_K2_.unsqueeze(0)
        U, _, VT = torch.linalg.svd(summary)
        V = VT.T
        P = V @ U.T
        P = P.T
        W_Q[start:end], W_K[start:end] = P.T @ W_Q1_, P.T @ W_K1_
        B_Q[start:end], B_K[start:end] = B_Q1_ @ P, B_K1_ @ P
        k = 1    
        if use_scaling:
            with torch.no_grad():
                k_roots = np.roots([
                    + np.sum(W_Q[start:end].cpu().numpy() ** 2) + np.sum(B_Q[start:end].cpu().numpy() ** 2),
                    - np.sum(W_Q[start:end].cpu().numpy() * W_Q2_.cpu().numpy()) - np.sum(B_Q[start:end].cpu().numpy() * B_Q2_.cpu().numpy()),
                    0,
                    + np.sum(W_K[start:end].cpu().numpy() * W_K2_.cpu().numpy()) + np.sum(B_K[start:end].cpu().numpy() * B_K2_.cpu().numpy()),
                    - np.sum(W_K[start:end].cpu().numpy() ** 2) - np.sum(B_K[start:end].cpu().numpy() ** 2)
                ])
                k_roots = np.array([np.real(x) for x in k_roots if np.isreal(x)])
                if len(k_roots) == 0:
                    k = 1
                else:
                    MIN_condition = lambda x : np.sum((W_Q[start:end].cpu().numpy() * x - W_Q2_.cpu().numpy()) ** 2) +\
                        np.sum((B_Q[start:end].cpu().numpy() * x - B_Q2_.cpu().numpy()) ** 2) +\
                        np.sum((W_K[start:end].cpu().numpy() / x - W_K2_.cpu().numpy()) ** 2) +\
                        np.sum((B_K[start:end].cpu().numpy() / x - B_K2_.cpu().numpy()) ** 2)
                    
                    k = k_roots[np.argmin([MIN_condition(x) for x in k_roots])]
            W_Q[start:end], W_K[start:end] = W_Q[start:end] * k, W_K[start:end] / k
            B_Q[start:end], B_K[start:end] = B_Q[start:end] * k, B_K[start:end] / k
    
    
    
    return {'W_Q': W_Q.detach(), 'B_Q': B_Q.detach(), 'W_K': W_K.detach(), 'B_K': B_K.detach()}

def Chunk_SVD_VO(
    local_param, anchor_param, 
    num_attention_heads=12,
    attention_head_size=32,
    use_scaling=False,
    use_alpha=True,
    alpha=0.5
):
    W_V1, W_O1, B_V1, B_O1 = local_param['W_V'], local_param['W_O'], local_param['B_V'], local_param['B_O']
    W_V2, W_O2, B_V2, B_O2 = anchor_param['W_V'], anchor_param['W_O'], anchor_param['B_V'], anchor_param['B_O']    
    W_V, W_O = torch.zeros_like(W_V2), torch.zeros_like(W_O2)
    B_V = torch.zeros_like(B_V2)
    if not num_attention_heads:
        num_attention_heads = W_V1.shape[0] // attention_head_size
    for i in range(num_attention_heads):
        start, end = i * attention_head_size, (i + 1) * attention_head_size
        W_V1_, W_O1_, B_V1_ = W_V1[start:end], W_O1[:, start:end], B_V1[start:end]
        W_V2_, W_O2_, B_V2_ = W_V2[start:end], W_O2[:, start:end], B_V2[start:end]
        # summary = W_V1_ @ W_V2_.T + W_O1_.T @ W_O2_
        summary = W_V1_ @ W_V2_.T + W_O1_.T @ W_O2_ + B_V1_.unsqueeze(0).T @ B_V2_.unsqueeze(0)
        
        U, _, VT = torch.linalg.svd(summary)
        V = VT.T
        P = V @ U.T
        P = P.T

        W_V[start:end], W_O[:, start:end] = P.T @ W_V1_, W_O1_ @ P
        B_V[start:end] = B_V1_ @ P
        k = 1
        if use_scaling:
            with torch.no_grad():
                k_roots = np.roots([
                    + np.sum(W_V[start:end].cpu().numpy() ** 2) + np.sum(B_V[start:end].cpu().numpy() ** 2),
                    - np.sum(W_V[start:end].cpu().numpy() * W_V2_.cpu().numpy()) - np.sum(B_V[start:end].cpu().numpy() * B_V2_.cpu().numpy()),
                    0,
                    + np.sum(W_O[:, start:end].cpu().numpy() * W_O2_.cpu().numpy()),
                    - np.sum(W_O[:, start:end].cpu().numpy() ** 2)
                ])
                k_roots = np.array([np.real(x) for x in k_roots if np.isreal(x)])
                if len(k_roots) == 0:
                    k = 1
                else:
                    MIN_condition = lambda x : np.sum((W_V[start:end].cpu().numpy() * x - W_V2_.cpu().numpy()) ** 2) +\
                        np.sum((B_V[start:end].cpu().numpy() * x - B_V2_.cpu().numpy()) ** 2) +\
                        np.sum((W_O[:, start:end].cpu().numpy() / x - W_O2_.cpu().numpy()) ** 2)
                    
                    k = k_roots[np.argmin([MIN_condition(x) for x in k_roots])]
                    
            W_V[start:end], W_O[:, start:end] = W_V[start:end] * k, W_O[:, start:end] / k
            B_V[start:end] = B_V[start:end] * k
    B_O = B_O1
    
    
    
    return {'W_V': W_V.detach(), 'B_V': B_V.detach(), 'W_O': W_O.detach(), 'B_O': B_O.detach()}

def Permute_IO(
    local_param, anchor_param,
):
    W_I1, W_O1, B_I1, B_O1 = local_param['W_I'], local_param['W_O_FFN'], local_param['B_I'], local_param['B_O_FFN']
    W_I2, W_O2, B_I2, B_O2 = anchor_param['W_I'], anchor_param['W_O_FFN'], anchor_param['B_I'], anchor_param['B_O_FFN']
    W_I, B_I, W_O = torch.zeros_like(W_I1), torch.zeros_like(B_I1), torch.zeros_like(W_O1)
    
    # shape W_I1: [3072, 768], W_O1: [768, 3072]
    cost = W_I1 @ W_I2.T + B_I1.unsqueeze(dim=0).T @ B_I2.unsqueeze(dim=0) + W_O1.T @ W_O2
    cost = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    P = torch.zeros_like(torch.tensor(cost))
    P[row_ind, col_ind] = 1
    P = P.to(W_I1.device)
    with torch.no_grad():
        W_I, B_I, W_O = P.T @ W_I1, B_I1 @ P, W_O1 @ P
    
    return {'W_I': W_I.detach(), 'B_I': B_I.detach(), 'W_O_FFN': W_O.detach(), 'B_O_FFN': B_O1.detach()}

def pi_fusion(
    args, local_module, anchor_module, alpha=0.5, use_scaling=False,
    FFN_only=False, Attention_only=False
):
    permuted_params = {}
    if not FFN_only:
        permuted_params.update(
            Chunk_SVD_QK(
                get_QKVO(local_module, args['attn_patterns']), 
                get_QKVO(anchor_module, args['attn_patterns']),
                use_scaling=use_scaling
            )
        )
        permuted_params.update(
            Chunk_SVD_VO(
                get_QKVO(local_module, args['attn_patterns']), 
                get_QKVO(anchor_module, args['attn_patterns']),
                use_scaling=use_scaling
            )
        )
    else:
        permuted_params.update(get_QKVO(local_module, args['attn_patterns']))
        
    if not Attention_only:
        permuted_params.update(
            Permute_IO(
                get_IO(local_module, args['ffn_patterns']),
                get_IO(anchor_module, args['ffn_patterns']),
            )
        )
    else:
        permuted_params.update(get_IO(local_module, args['ffn_patterns']))
    
    permuted_params = {
        'attention': {
            'attention': {
                'query': {
                    'weight': permuted_params['W_Q'], 
                    'bias': permuted_params['B_Q']
                }, 'key': {
                    'weight': permuted_params['W_K'], 
                    'bias': permuted_params['B_K']
                }, 'value': {
                    'weight': permuted_params['W_V'], 
                    'bias': permuted_params['B_V']
                }, 
            }, 'output': {
                'dense': {
                    'weight': permuted_params['W_O'], 
                    'bias': permuted_params['B_O']
                }
            }
        }, 
        'layernorm_before': {
            'weight': local_module['layernorm_before']['weight'], 
            'bias': local_module['layernorm_before']['bias']
        }, 'layernorm_after': {
            'weight': local_module['layernorm_after']['weight'], 
            'bias': local_module['layernorm_after']['bias']
        }, 'intermediate': {
            'dense': {
                'weight': permuted_params['W_I'], 
                'bias': permuted_params['B_I']
            }
        }, 'output': {
            'dense': {
                'weight': permuted_params['W_O_FFN'], 
                'bias': permuted_params['B_O_FFN']
            }
        }
    }
    
    return copy.deepcopy(permuted_params)

import re
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
        valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
        if valid_type and valid_name:
            modules[name] = module
    return modules

def compute_gram(model, train_dataloader):
    grams = {} # gram matrices for each linear layer inputs
    xn = {} # number of examples used for computing gram

    def get_gram(name):
        def hook(module, input, output):
            x = input[0].detach() # $[b,t,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0,1), x) # [h,h]
            if name not in grams:
                grams[name] = xtx / x.size(0)
                xn[name] = x.size(0)
            else:
                grams[name] = (grams[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                xn[name] += x.size(0)
        return hook

    linear_modules = filter_modules_by_regex(model, None, [nn.Linear])
    handles = []
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_gram(name))
        handles.append(handle)

    grams["meta_info"] = {
        "conv1d": [
            n
            for n, m in filter_modules_by_regex(
                model, None, [nn.Conv1d, Conv1D]
            ).items()
        ]
    }
    
    n_step = 1000
    total = n_step if n_step > 0 else len(train_dataloader)
    for step, inputs in tqdm(enumerate(train_dataloader), total=total, desc='Computing gram matrix'):
        if n_step > 0 and step == n_step:
            break
        inputs1 = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs1)

    for handle in handles:
        handle.remove()

    return grams

def compute_fisher(model, train_dataloader):
    fisher = {}
    n_b = 0

    n_step = 1000

    total = n_step if n_step > 0 else len(train_dataloader)

    for step, inputs in tqdm(
        enumerate(train_dataloader), total=total, desc="Computing fisher"
    ):
        if n_step > 0 and step == n_step:
            break
        inputs1 = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs1)
        logits = outputs.logits
        n_b += 1
        # computer empirical fisher

        # if logits.size(-1) == 1 or config.merger.emp_fisher:
        if logits.size(-1) == 1 or True:
            # is regression task. can only compute empiricial fisher
            # assume f(x; theta) is gaussian with fixed var. log likelihood proportional to || f(x) - y ||^2
            loss = outputs.loss
            model.zero_grad()
            loss.backward()
            b_n2fisher = collect_squared_gradients(model)
        else:
            log_probs = torch.log_softmax(logits, -1)
            _, target_labels = logits.max(-1)
            nll_loss = F.nll_loss(log_probs, target_labels)
            model.zero_grad()
            nll_loss.backward()
            b_n2fisher = collect_squared_gradients(model)
        for n, f in b_n2fisher.items():
            if n not in fisher:
                fisher[n] = f
            else:
                fisher[n] += f
    assert n_b
    for n, f in fisher.items():
        fisher[n] = f / n_b
    return fisher

def collect_squared_gradients(model):
    n2fisher = {n: p.grad.detach() ** 2 for n, p in model.named_parameters()}
    return n2fisher