import logging
import re
import numpy as np
import ot
import torch
from torch import nn

from .avg_merger import FedAvgMerger
from .misc import filter_modules_by_regex, check_module_name_by_regex
from .net import create_model
from .ot_utils.ot_ground_metric import GroundMetric
from .pi_utils import (
    Chunk_SVD_QK, 
    get_QKVO, 
    get_submodule, 
    Chunk_SVD_VO, 
    Permute_IO,
    get_IO
)
import random


class TmpLocalModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model




class PermutationInvarianMerger(FedAvgMerger):
    def __init__(
        self, config, merger_config, local_models, global_model, merger_ds=None
    ):
        super().__init__(config, merger_config, local_models, global_model, merger_ds)
        # self.pi_params = merger_config.pi_params
        self.avg_local_model = None

        
    def merge_to_global(self, **kwargs):
        self.avg_local_model = self.avg_local_models(self.local_models)
        self.anchor_index = self.get_anchor_index(self.local_models)

        for i, local_model in enumerate(self.local_models):
            if i == self.anchor_index:
                continue
            self.match_ffns(local_model, self.local_models[self.anchor_index])
            self.match_attentions(local_model, self.local_models[self.anchor_index])
            
        self.avg_merge(self.local_models, self.global_model, **kwargs)
        
    def match_ffns(self, local_model, anchor_model):
        ffn_patterns = [v for v in vars(self.merger_config.ffn_patterns).values()]
        for ffn_pattern in ffn_patterns:
            for local, anchor in zip(
                local_model.named_modules(), anchor_model.named_modules()
            ):
                local_module_name, local_module = local
                anchor_module_name, anchor_module = anchor
                assert local_module_name == anchor_module_name
                if check_module_name_by_regex(local_module_name, ffn_pattern.pi_filter_regex):
                    self.match_ffn(local_module, anchor_module, ffn_pattern)
    
    def match_ffn(self, local_module, anchor_module, ffn_pattern):
        local_param = get_IO(local_module, ffn_pattern)
        anchor_param = get_IO(anchor_module, ffn_pattern)
        matched_IO = Permute_IO(local_param=local_param, anchor_param=anchor_param)
        self.update_ffns(local_module, matched_IO, ffn_pattern)
        
    def update_ffns(self, local_model, matched_IO, ffn_pattern):
        I = get_submodule(local_model, ffn_pattern.intermediate)
        O = get_submodule(local_model, ffn_pattern.output)
        I.weight.data.copy_(matched_IO['W_I'])
        I.bias.data.copy_(matched_IO['B_I'])
        O.weight.data.copy_(matched_IO['W_O_FFN'])
        
            
    def match_attentions(self, local_model, anchor_model):
        attn_patterns = [v for v in vars(self.merger_config.attn_patterns).values()]
        # attn_patterns: pattern1, pattern2, ...
        # pattern1: pi_filter_regex
        for attn_pattern in attn_patterns:
            for local, anchor in zip(
                local_model.named_modules(), anchor_model.named_modules()
            ):
                local_module_name, local_module = local
                anchor_module_name, anchor_module = anchor
                assert local_module_name == anchor_module_name
                if check_module_name_by_regex(local_module_name, attn_pattern.pi_filter_regex):
                    self.match_attention(local_module, anchor_module, attn_pattern)
    
    def match_attention(self, local_module, anchor_module, attn_pattern):
        local_param = get_QKVO(local_module, attn_pattern) # {'W_Q': W_Q, 'B_Q': B_Q, 'W_K': W_K, 'B_K': B_K, 'W_V': W_V, 'B_V': B_V, 'W_O': W_O, 'B_O': B_O}
        anchor_param = get_QKVO(anchor_module, attn_pattern)
        matched_QK = Chunk_SVD_QK(anchor_param=anchor_param, local_param=local_param)
        matched_VO = Chunk_SVD_VO(anchor_param=anchor_param, local_param=local_param)
        self.update_attentions(local_module, matched_QK, matched_VO, attn_pattern)
    
    def update_attentions(self, local_model, matched_QK, matched_VO, attn_pattern):
        
        Q = get_submodule(local_model, attn_pattern.query)    # layer.0.attention.self.query
        K = get_submodule(local_model, attn_pattern.key)      # layer.0.attention.self.key
        V = get_submodule(local_model, attn_pattern.value)    # layer.0.attention.self.value
        O = get_submodule(local_model, attn_pattern.output)   # layer.0.attention.output.dense
        Q.weight.data.copy_(matched_QK['W_Q'])
        Q.bias.data.copy_(matched_QK['B_Q'])
        K.weight.data.copy_(matched_QK['W_K'])
        K.bias.data.copy_(matched_QK['B_K']) 
        if matched_VO is not None:
            V.weight.data.copy_(matched_VO['W_V'])
            V.bias.data.copy_(matched_VO['B_V'])
            O.weight.data.copy_(matched_VO['W_O'])      
        
    def avg_local_models(self, local_models):
        avg_local_model, _ = create_model(
            self.config, self.local_models[0].model_config
        )
        avg_local_model = TmpLocalModel(avg_local_model)
        for avg_param, local_params in zip(
            avg_local_model.parameters(), zip(*[m.parameters() for m in local_models])
        ):
            avg_param.data.copy_(
                sum([p.data for p in local_params]) / len(local_params)
            )
        avg_local_model.to('cuda:0')
        return avg_local_model
            
    def get_anchor_index(self, local_models, method='avg'):
        if method == 'avg':
            diffs = []
            for i in range(len(local_models)):
                diff = 0
                for avg_param, local_param in zip(
                    self.avg_local_model.parameters(), local_models[i].parameters()
                ):
                    diff += torch.norm(avg_param - local_param).cpu().detach().numpy()
                diffs.append(diff)
            return np.argmin(diffs)
        else:
            return random.randint(0, len(local_models)-1)