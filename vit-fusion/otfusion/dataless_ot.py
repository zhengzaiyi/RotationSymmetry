# Copyright 2023 Bloomberg Finance L.P.
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

import logging, re

import numpy as np
import ot
import torch
from torch import nn

from .avg_merger import FedAvgMerger

class GroundMetric:
    """
    Ground Metric object for Wasserstein computations:

    """

    def __init__(self, params, not_squared=False):
        self.params = params
        self.ground_metric_type = params.ground_metric
        self.ground_metric_normalize = params.ground_metric_normalize
        self.reg = params.reg
        if hasattr(params, "not_squared"):
            self.squared = not params.not_squared
        else:
            # so by default squared will be on!
            self.squared = not not_squared
        self.mem_eff = params.ground_metric_eff

    def _clip(self, ground_metric_matrix):
        if self.params.debug:
            print("before clipping", ground_metric_matrix.data)

        percent_clipped = (
            float(
                (ground_metric_matrix >= self.reg * self.params.clip_max)
                .long()
                .sum()
                .data
            )
            / ground_metric_matrix.numel()
        ) * 100
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        setattr(self.params, "percent_clipped", percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(
            min=self.reg * self.params.clip_min, max=self.reg * self.params.clip_max
        )
        if self.params.debug:
            print("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            print(
                "Normalizing by max of ground metric and which is ",
                ground_metric_matrix.max(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            print(
                "Normalizing by median of ground metric and which is ",
                ground_metric_matrix.median(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print(
                "Normalizing by mean of ground metric and which is ",
                ground_metric_matrix.mean(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (torch.isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared=True):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            print("dont leave off the squaring of the ground metric")
            c = c ** (1 / 2)
        # print(c.size())
        if self.params.dist_normalize:
            assert NotImplementedError
        return c

    def _pairwise_distances(self, x, y=None, squared=True):
        """
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)
        if self.params.activation_histograms and self.params.dist_normalize:
            dist = dist / self.params.act_num_samples
            print("Divide squared distances by the num samples")

        if not squared:
            print("dont leave off the squaring of the ground metric")
            dist = dist ** (1 / 2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)

        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1])
                - coordinates,
                p=2,
                dim=2,
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(
                    coordinates, other_coordinates, squared=self.squared
                )
            else:
                matrix = self._cost_matrix_xy(
                    coordinates, other_coordinates, squared=self.squared
                )

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        # print("stats of vecs are: mean {}, min {}, max {}, std {}".format(
        #    norms.mean(), norms.min(), norms.max(), norms.std()
        # ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1)
                @ torch.norm(other_coordinates, dim=1).view(1, -1),
            )
        return matrix.clamp_(min=0)

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            "euclidean": self._get_euclidean,
            "cosine": self._get_cosine,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        # print('Processing the coordinates to form ground_metric')
        if self.params.geom_ensemble_type == "wts" and self.params.normalize_wts:
            # print("In weight mode: normalizing weights to unit norm")
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

        if self.params.debug:
            logging.info("coordinates is {}".format(coordinates))
            if other_coordinates is not None:
                logging.info("other_coordinates is {}".format(other_coordinates))
            logging.info("ground_metric_matrix is {}".format(ground_metric_matrix))

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.debug:
            logging.info(
                "ground_metric_matrix at the end is {}".format(ground_metric_matrix)
            )

        return ground_metric_matrix


def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any(
            [re.match(patt, name) for patt in include_patterns]
        )
        valid_type = not include_type or any(
            [isinstance(module, md_cls) for md_cls in include_type]
        )
        if valid_type and valid_name:
            modules[name] = module
    return modules


class TmpLocalModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model


def get_submodule(m, name):
    ns = name.split(".")
    r = m
    for n in ns:
        r = getattr(r, n)
    return r


def ot_fusion(
    local_model, 
    local_model_copy,
    anchor_model, 
    match_config,
):
    def match_all_ffns(local_models, tgt_local_model, match_config):
        # we assume the input and outputs of the ffn layers are aligned
        local_modules = []

        ot_patterns = [v for v in vars(match_config.ot_patterns).values()]
        for ot_pattern in ot_patterns:
            for local_model_id, local_model in enumerate(local_models):
                modules = filter_modules_by_regex(
                    local_model, ot_pattern.ot_filter_regex, include_type=None
                )
                local_modules.append(modules)

            tgt_modules = filter_modules_by_regex(
                tgt_local_model, ot_pattern.ot_filter_regex, include_type=None
            )

            for ffn_name in tgt_modules:
                logging.info("Matching {}".format(ffn_name))
                single_ffns = [x[ffn_name] for x in local_modules]
                tgt_ffn = tgt_modules[ffn_name]
                ot_match_ffns(single_ffns, tgt_ffn, ot_pattern, match_config)

    def ot_match_ffns(ffns, tgt_ffn, ot_pattern, match_config):
        # input: list of ffns, on for each local model (a and b)
        assert len(ffns) == 2
        eps = 1e-10
        layers = [
            [get_submodule(x, ot_pattern.ot_lin1) for x in ffns]
            + [get_submodule(tgt_ffn, ot_pattern.ot_lin1)],
            [get_submodule(x, ot_pattern.ot_lin2) for x in ffns]
            + [get_submodule(tgt_ffn, ot_pattern.ot_lin2)],
        ]
        ground_metric_object = GroundMetric(match_config.ot_params)

        T_var = None

        for layer_id, (lina, linb, tgt_lin) in enumerate(layers):
            w_a = lina.weight.data
            w_b = linb.weight.data
            w_tgt = tgt_lin.weight.data

            mu_card, nu_card = w_a.shape[0], w_b.shape[0]

            if layer_id == 0:
                M = ground_metric_object.process(w_a, w_b).to(w_a.device)
                aligned_wt = w_a
            else:
                aligned_wt = torch.matmul(w_a, T_var).to(w_a.device)
                M = ground_metric_object.process(aligned_wt, w_b)

            mu = np.ones(mu_card) / mu_card
            nu = np.ones(nu_card) / nu_card

            cpuM = M.data.cpu().numpy()
            if match_config.ot_params.exact:
                T = ot.emd(mu, nu, cpuM)
            else:
                T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=match_config.ot_params.reg)

            T_var = torch.from_numpy(T).float().to(w_a.device)

            if match_config.ot_params.debug:
                logging.info("The trace of T is {}".format(T_var.trace()))

            if match_config.ot_params.correction:
                if not match_config.ot_params.proper_marginals:
                    # think of it as m x 1, scaling weights for m linear combinations of points in X
                    marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                    marginals.to(w_a.device)
                    marginals = torch.diag(1.0 / (marginals + eps)).to(
                        w_a.device
                    )  # take inverse
                    T_var = torch.matmul(T_var, marginals)
                else:
                    marginals_beta = T_var.t() @ torch.ones(
                        T_var.shape[0], dtype=T_var.dtype
                    )

                    marginals = 1 / (marginals_beta + eps)
                    print("shape of inverse marginals beta is ", marginals_beta.shape)
                    print("inverse marginals beta is ", marginals_beta)

                    T_var = T_var * marginals
                    # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                    # this should all be ones, and number equal to number of neurons in 2nd model
                    print(T_var.sum(dim=0))

            if match_config.ot_params.past_correction:
                matched_w_a = torch.matmul(
                    T_var.transpose(0, 1), aligned_wt.reshape(aligned_wt.shape[0], -1)
                )
            else:
                matched_w_a = torch.matmul(
                    T_var.transpose(0, 1), w_a.view(w_a.shape[0], -1)
                )

            w_tgt.copy_(matched_w_a)

    match_all_ffns(
        local_models=[local_model_copy, anchor_model],
        tgt_local_model=local_model,
        match_config=match_config,
    )
  