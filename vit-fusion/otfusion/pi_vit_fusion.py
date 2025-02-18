# Top-level function for fusing two vision transformers using OTFusion
#
# Author: Moritz Imfeld <moimfeld@ethz.ch>

from otfusion_lib import ln_fusion, encoder_fusion, fc_fusion, resid_policy
import copy, logging, torch
from pi_utils import Chunk_SVD_QK, Chunk_SVD_VO, Permute_IO, pi_fusion

#------------#
# VIT Fusion #
#------------#
def pi_match(
    args,
    weights: dict, 
    log_file = None,
    use_scaling = False,
    single_layer_index = None,
    tail_layer_index = None,
    FFN_only = False,
    Attention_only = False,
):

    # init
    number_of_encoders = len(weights['model_0']['vit']['encoder']['layer'])
    w_matched            = {'vit': {'embeddings': {'patch_embeddings': {'projection': {}}},
                                  'encoder': {'layer': {}}}}

    assert not (single_layer_index is not None and tail_layer_index is not None), "Only one of single_layer_index and tail_layer_index can be set."
    w_matched['vit']['embeddings'] = copy.deepcopy(weights['model_0']['vit']['embeddings'])

    for i in range(number_of_encoders):
        encoder_key = str(i)
        if single_layer_index is not None and i != single_layer_index:
            w_matched['vit']['encoder']['layer'][encoder_key] = copy.deepcopy(weights['model_0']['vit']['encoder']['layer'][encoder_key])
            continue
        if tail_layer_index is not None and i < number_of_encoders - tail_layer_index - 1:
            w_matched['vit']['encoder']['layer'][encoder_key] = copy.deepcopy(weights['model_0']['vit']['encoder']['layer'][encoder_key])
            continue
        w_matched['vit']['encoder']['layer'][encoder_key] = pi_fusion(
            args=args,
            local_module=weights['model_0']['vit']['encoder']['layer'][encoder_key],
            anchor_module=weights['model_1']['vit']['encoder']['layer'][encoder_key],
            use_scaling=use_scaling,
            FFN_only=FFN_only,
            Attention_only=Attention_only,
        )
        # w_fused['vit']['encoder']['layer'][encoder_key] = copy.deepcopy(weights['model_1']['vit']['encoder']['layer'][encoder_key])

    # Fuse Layer Normalization at the end of encoder chain
    w_matched['vit']['layernorm'] = copy.deepcopy(weights['model_0']['vit']['layernorm'])

    w_matched['classifier'] = copy.deepcopy(weights['model_0']['classifier'])
    return w_matched
