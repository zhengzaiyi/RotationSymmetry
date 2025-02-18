# Transformer fusion Pipeline

import sys, os, pickle, yaml, torch, logging, time, torchvision, copy, random
sys.path.append(os.path.join(os.path.dirname(__file__), "otfusion"))
from otfusion.hf_vit_fusion import hf_vit_fusion
from otfusion.pi_vit_fusion import pi_match
from otfusion.ot_utils import get_activations, accumulate_nested_dicts, divide_nested_dicts, multi_model_vanilla, model_to_dict, vanilla_fusion_old, model_eq_size_check
sys.path.append(os.path.join(os.path.dirname(__file__), "vit"))
from vit import vit_helper
from otfusion.dataless_ot import ot_fusion
from otfusion.pi_utils import dic_to_object, compute_gram, compute_fisher
from otfusion.avg_merger import FedAvgMerger
import numpy as np

from datasets import config as ds_config

def ot_fusion_pipeline(args, weights, device, LOGGING_LEVEL, log_file, log, models):
    log.info(" ------- Computing Activations -------\n")
    dataloader = get_dataloader(args, device)
    start_time = time.perf_counter()
    acts = get_activations(args = args, models = models, dataloader = dataloader, LOGGING_LEVEL = LOGGING_LEVEL, device = device, log_file = log_file)
    end_time = time.perf_counter()
    log.info(' Time for computing activations: {0:.4f} s'.format(end_time - start_time))
    
    log.info(" ------- Performing OTFusion -------\n")
    start_time = time.perf_counter()
    alpha = 1 / args['fusion']['num_models']
    anker_weights = weights['model_1']
    anker_acts    = acts['model_1']
    w_fused_list  = []
    for i in range(args['fusion']['num_models']-1):
        index = i
        if index > 0:
            log.info(' -------')
            index += 1 
        log.info(' Fusing anker model (model_1) with model_{0}'.format(index))
        w_fused_list.append(do_otfusion(args = args, weights = {'model_1': anker_weights, 'model_0': weights['model_{0}'.format(index)]},
                                        acts = {'model_1': anker_acts, 'model_0': acts['model_{0}'.format(index)]}, alpha = alpha, device = device, LOGGING_LEVEL = LOGGING_LEVEL, log_file = log_file))
    for i in range(args['fusion']['num_models']-1):
        if i == 0:
            w_fused_acc = w_fused_list[0]
        else:
            w_fused_acc = accumulate_nested_dicts(w_fused_acc, w_fused_list[i])
    # divide by num_models - 1
    w_fused = divide_nested_dicts(w_fused_acc, args['fusion']['num_models']-1)
    end_time = time.perf_counter()
    log.info(' Time for OTFusion: {0:.4f} s'.format(end_time - start_time))
    return w_fused

def main(exp = None, exp_mod = None, log_file = None):
    """
    ## Description
    The main function implements a full otfusion, evaluation and finetuning pipeline. The function implements the following steps:
    1. Initialize logger
    2. Read YAML file config.
    3. Modify config (if exp_mod is not None)
    4. Load models
    5. Compute activations
    6. Perform OTFusion
    7. Perform vanilla-fusion
    8. Evaluate one-shot accuracy (pre-finetuning)
    9. Finetuning
    10. Evaluate post-finetuning performance
    ------
    ## Parameters
    `exp`       experiment name string (i.e. `fuse_enc_dec_gen_N1_sort.yaml`)\\
    `exp_mod`   either dictionary containing modifications to the experiment config, or the flag 'is_sweep' indicating a wandb sweep
                Note:    dictionary must have the same structure as the experiment
                Example:    The following exp_mod dict would change the num_samples to 50
                            and the switch off the generator fusion:
                            `exp_mod = {'fusion': {'acts': {'num_samples': 50}}, 'fuse_gen': False}`
    `log_file`  relative or full file path + name of the logfile where the function should write to.
                Note:   Each function call of the main function should have a unique log_file name
                        if they are run in parallel, else the log files can get corrupted.
                Example: `reports/14_03_2023_regression/1.log`
    """
    # Default experiment
    EXPERIMENT_CFG_FILE = 'experiments/fuse_hf_vit_cifar10.yaml'
    LOGGING_LEVEL       = logging.INFO

    # Initialize logger
    if len(sys.argv) > 1:
        if (any('--debug' in string for string in sys.argv)):
            LOGGING_LEVEL = logging.DEBUG
    if log_file != None:
        log = logging.getLogger('{0}_main'.format(log_file))
        fileHandler = logging.FileHandler(log_file, mode='a')
        log.addHandler(fileHandler)
    else:
        log = logging.getLogger('main')
    logging.basicConfig(level=LOGGING_LEVEL)

    # Load Experiment Configuration
    args = load_args(log = log, EXPERIMENT_CFG_FILE = EXPERIMENT_CFG_FILE, exp = exp)


    # Print experiment configuration to commandline
    log_args(log = log, args = args, exp_mod = exp_mod)

    device = torch.device('cpu')

    # Set all seeds
    SEED = args['fusion']['acts']['seed']
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Set a different directory for datasets if specified in the environment variables
    # Defaults to huggingface cache otherwise
    ds_path = os.environ.get("HF_DATASET_PATH")
    if ds_path is not None:
        ds_config.DOWNLOADED_DATASETS_PATH = ds_path
        ds_config.HF_DATASETS_CACHE = ds_path

    # Load Models
    log.info(" ------- Loading Models -------\n")
    weights = {}
    models = []
    for i in range(args['fusion']['num_models']):
        weights['model_{0}'.format(i)], model = load_weights_and_model(args, key = "name_{0}".format(i))
        models.append(model)

    # check wheter models are of same size --> if models are of different size, vanilla fusion cannot be applied
    args['fusion']['heterogeneous'] = not model_eq_size_check(models, log)
    if args['fusion']['heterogeneous']:
        log.info(" Models have different sizes")
    else:
        log.info(" Models are of equal size")
    val_dataloader, test_dataloader = get_test_dataloader(args, device)
    dataloader = get_dataloader(args, device)
    # MATCHING
    running_time = {
        'match': 0,
        'scaling_match': 0,
        'ot': 0,
        'ot_match': 0,
        'ot_scaling_match': 0,
        'vanilla': 0,
        'vanilla_match': 0,
        'vanilla_scaling_match': 0,
        'fisher': 0,
        'fisher_match': 0,
        'fisher_scaling_match': 0,
        'regmean': 0,
        'regmean_match': 0,
        'regmean_scaling_match': 0
    }
    current_time = time.time()
    w_matched = pi_match(args, weights, log_file = log_file)
    running_time['match'] = time.time() - current_time
    
    matched_model = get_model(args, w_matched)
    matched_models = [matched_model, models[1]]
    matched_weights = {
        'model_0': model_to_dict(matched_model), 
        'model_1': weights['model_1']
    }
    
    current_time = time.time()
    w_scaling_matched = pi_match(args, weights, log_file = log_file, use_scaling = True)
    running_time['scaling_match'] = time.time() - current_time
    scaling_matched_model = get_model(args, w_scaling_matched)
    scaling_matched_models = [scaling_matched_model, models[1]]
    scaling_matched_weights = {
        'model_0': model_to_dict(scaling_matched_model), 
        'model_1': weights['model_1']
    }
    models[0].save_pretrained('models/model_0')
    models[1].save_pretrained('models/model_1')
    matched_model.save_pretrained('models/matched_model')
    scaling_matched_model.save_pretrained('models/scaling_matched_model')
    

    # otfusion
    current_time = time.time()
    w_fused = ot_fusion_pipeline(args, weights, device, LOGGING_LEVEL, log_file, log, models=models)
    running_time['ot'] = time.time() - current_time
    model_otfused = get_model(args, w_fused)
    current_time = time.time()
    w_fused_matched = ot_fusion_pipeline(args, matched_weights, device, LOGGING_LEVEL, log_file, log, models=matched_models)
    running_time['ot_match'] = time.time() - current_time
    model_otfused_matched = get_model(args, w_fused_matched)
    current_time = time.time()
    w_fused_scaling_matched = ot_fusion_pipeline(args, scaling_matched_weights, device, LOGGING_LEVEL, log_file, log, models=scaling_matched_models)
    running_time['ot_scaling_match'] = time.time() - current_time
    model_otfused_scaling_matched = get_model(args, w_fused_scaling_matched)


    # # simple fusion
    current_time = time.time()
    model_vanilla_fused = do_vanilla_fusion(args, weights, models[0], models[1])
    running_time['vanilla'] = time.time() - current_time
    current_time = time.time()
    matched_model_vanilla_fused = do_vanilla_fusion(args, matched_weights, matched_models[0], matched_models[1])
    running_time['vanilla_match'] = time.time() - current_time
    current_time = time.time()
    scaling_matched_model_vanilla_fused = do_vanilla_fusion(args, scaling_matched_weights, scaling_matched_models[0], scaling_matched_models[1])
    running_time['vanilla_scaling_match'] = time.time() - current_time
    
    # # TODO: Fisher Fusion
    # 
    fisher_config = yaml.load(open('experiments/fisher.yaml', 'r'), Loader=yaml.FullLoader)
    fisher_config = fisher_config['merger']
    fisher_config = dic_to_object(fisher_config)
    avg_merger = FedAvgMerger(
        config=None,
        merger_config=fisher_config,
        local_models=models,
        global_model=None
    )
    current_time = time.time()
    model_fisher_fused = avg_merger.avg_merge(
        local_models=avg_merger.local_models,
        global_model=copy.deepcopy(models[0]),
        fisher_weights=[
            compute_fisher(models[0], dataloader),
            compute_fisher(models[1], dataloader)
        ]
    )
    running_time['fisher'] = time.time() - current_time
    current_time = time.time()
    matched_model_fisher_fused = avg_merger.avg_merge(
        local_models=matched_models,
        global_model=copy.deepcopy(models[0]),
        fisher_weights=[
            compute_fisher(matched_models[0], dataloader),
            compute_fisher(matched_models[1], dataloader)
        ]
    )
    running_time['fisher_match'] = time.time() - current_time
    current_time = time.time()
    scaling_matched_model_fisher_fused = avg_merger.avg_merge(
        local_models=scaling_matched_models,
        global_model=copy.deepcopy(models[0]),
        fisher_weights=[
            compute_fisher(scaling_matched_models[0], dataloader),
            compute_fisher(scaling_matched_models[1], dataloader)
        ]
    )
    running_time['fisher_scaling_match'] = time.time() - current_time
    del avg_merger
     
    # # TODO: Regmean Fusion
    regmean_config = yaml.load(open('experiments/regmean.yaml', 'r'), Loader=yaml.FullLoader)
    regmean_config = regmean_config['merger']
    regmean_config = dic_to_object(regmean_config)
    avg_merger = FedAvgMerger(
        config=None,
        merger_config=regmean_config,
        local_models=models,
        global_model=None
    )
    current_time = time.time()
    model_regmean_fused = avg_merger.avg_merge(
        local_models=avg_merger.local_models,
        global_model=copy.deepcopy(models[0]),
        all_grams=[
            compute_gram(models[0], dataloader),
            compute_gram(models[1], dataloader)
        ]
    )
    running_time['regmean'] = time.time() - current_time
    current_time = time.time()
    matched_model_regmean_fused = avg_merger.avg_merge(
        local_models=matched_models,
        global_model=copy.deepcopy(models[0]),
        all_grams=[
            compute_gram(matched_models[0], dataloader),
            compute_gram(matched_models[1], dataloader)
        ]
    )
    running_time['regmean_match'] = time.time() - current_time
    current_time = time.time()
    scaling_matched_model_regmean_fused = avg_merger.avg_merge(
        local_models=scaling_matched_models,
        global_model=copy.deepcopy(models[0]),
        all_grams=[
            compute_gram(scaling_matched_models[0], dataloader),
            compute_gram(scaling_matched_models[1], dataloader)
        ]
    )
    running_time['regmean_scaling_match'] = time.time() - current_time
    

    # Delete weights and acts from memory
    # del weights
    # del acts
    torch.cuda.empty_cache()

    # Evaluation
    w_matched = pi_match(args, weights, log_file = log_file)
    w_FFN_matched = pi_match(args, weights, log_file = log_file, FFN_only = True)
    w_Attention_matched = pi_match(args, weights, log_file = log_file, Attention_only = True)
    w_scaling_matched = pi_match(args, weights, log_file = log_file, use_scaling = True)
    w_scaling_FFN_matched = pi_match(args, weights, log_file = log_file, use_scaling = True, FFN_only = True)
    w_scaling_Attention_matched = pi_match(args, weights, log_file = log_file, use_scaling = True, Attention_only = True)
    log.info("Distance before matching: {}".format(get_distance(weights['model_0'], weights['model_1'])[0]))
    log.info("Distance after matching: {}".format(get_distance(w_matched, weights['model_1'])[0]))
    log.info("Distance after FFN matching: {}".format(get_distance(w_FFN_matched, weights['model_1'])[0]))
    log.info("Distance after Attention matching: {}".format(get_distance(w_Attention_matched, weights['model_1'])[0]))
    log.info("Distance after scaling matching: {}".format(get_distance(w_scaling_matched, weights['model_1'])[0]))
    log.info("Distance after FFN scaling matching: {}".format(get_distance(w_scaling_FFN_matched, weights['model_1'])[0]))
    log.info("Distance after Attention scaling matching: {}".format(get_distance(w_scaling_Attention_matched, weights['model_1'])[0]))
    log.info("Distance after OTFusion: {}".format(get_distance(w_fused, weights['model_1'])[0]))

    dataless_ot_config = yaml.load(open('experiments/dataless_ot.yaml', 'r'), Loader=yaml.FullLoader)
    dataless_ot_config = dataless_ot_config['merger']
    dataless_ot_config = dic_to_object(dataless_ot_config)
    tmp_local_model = copy.deepcopy(models[0])
    ot_fusion(
        local_model=tmp_local_model,
        local_model_copy=copy.deepcopy(models[0]),
        anchor_model=copy.deepcopy(models[1]),
        match_config=dataless_ot_config,
    )
    log.info("Distance after Dataless OT: {}".format(get_distance(model_to_dict(tmp_local_model), weights['model_1'])[0]))

    log.info(" ------- Done Distance Analysis -------")
    
    # log.info(" ------- Evaluating Models -------")
    # if args.get("regression", {}).get("only_eval_ot", False) == False:
    #     for i in range(args['fusion']['num_models']):
    #         test_accuracy = get_test_acc(args, models[i], test_dataloader, device)
    #         log.info(" Model {0} Accuracy: {1}".format(i, test_accuracy))
    #         test_accuracy = get_test_acc(args, matched_models[i], test_dataloader, device)
    #         log.info(" Matched Model {0} Accuracy: {1}".format(i, test_accuracy))

    results = {'single_layer': 
               {'match': {
                   'vanilla': [],
                    'fisher': [],
                    'regmean': [],
                    'ot': []
               }, 'scale_match': {
                    'vanilla': [],
                    'fisher': [],
                    'regmean': [],
                    'ot': []
               }}, 
               'tail_layer': 
               {'match': {
                   'vanilla': [],
                    'fisher': [],
                    'regmean': [],
                    'ot': []
               }, 
                'scale_match': {
                   'vanilla': [],
                    'fisher': [],
                    'regmean': [],
                    'ot': []
               }}}
    val_results = copy.deepcopy(results)
    

    test_accuracy = get_test_acc(args, model_otfused, test_dataloader, device)
    log.info("OT Fusion Accuracy: {0}".format(test_accuracy))
    results['ot_fusion'] = test_accuracy
    
    test_accuracy = get_test_acc(args, model_otfused_matched, test_dataloader, device)
    log.info("OT Fusion Matched Accuracy: {0}".format(test_accuracy))
    results['ot_fusion_matched'] = test_accuracy
    
    test_accuracy = get_test_acc(args, model_otfused_scaling_matched, test_dataloader, device)
    log.info("OT Fusion Scaling Matched Accuracy: {0}".format(test_accuracy))
    results['ot_fusion_scaling_matched'] = test_accuracy
    
    test_accuracy = get_test_acc(args, model_vanilla_fused, test_dataloader, device)
    log.info("Vanilla Fusion Accuracy: {0}".format(test_accuracy))
    results['vanilla_fusion'] = test_accuracy
    
    test_accuracy = get_test_acc(args, matched_model_vanilla_fused, test_dataloader, device)
    log.info("Vanilla Fusion Matched Accuracy: {0}".format(test_accuracy))
    results['vanilla_fusion_matched'] = test_accuracy
    
    test_accuracy = get_test_acc(args, scaling_matched_model_vanilla_fused, test_dataloader, device)
    log.info("Vanilla Fusion Scaling Matched Accuracy: {0}".format(test_accuracy))
    results['vanilla_fusion_scaling_matched'] = test_accuracy
    
    test_accuracy = get_test_acc(args, model_fisher_fused, test_dataloader, device)
    log.info("Fisher Fusion Accuracy: {0}".format(test_accuracy))
    results['fisher_fusion'] = test_accuracy
    
    test_accuracy = get_test_acc(args, matched_model_fisher_fused, test_dataloader, device)
    log.info("Fisher Fusion Matched Accuracy: {0}".format(test_accuracy))
    results['fisher_fusion_matched'] = test_accuracy
    
    test_accuracy = get_test_acc(args, scaling_matched_model_fisher_fused, test_dataloader, device)
    log.info("Fisher Fusion Scaling Matched Accuracy: {0}".format(test_accuracy))
    results['fisher_fusion_scaling_matched'] = test_accuracy
    
    test_accuracy = get_test_acc(args, model_regmean_fused, test_dataloader, device)
    log.info("Regmean Fusion Accuracy: {0}".format(test_accuracy))
    results['regmean_fusion'] = test_accuracy
    
    test_accuracy = get_test_acc(args, matched_model_regmean_fused, test_dataloader, device)
    log.info("Regmean Fusion Matched Accuracy: {0}".format(test_accuracy))
    results['regmean_fusion_matched'] = test_accuracy
    
    test_accuracy = get_test_acc(args, scaling_matched_model_regmean_fused, test_dataloader, device)
    log.info("Regmean Fusion Scaling Matched Accuracy: {0}".format(test_accuracy))
    results['regmean_fusion_scaling_matched'] = test_accuracy
        
    log.info(" ------- Done Main Results -------")
    
    log.info(" ------- Distance Analysis -------")
    

    log.info(" ------- Layer Analysis -------")
    
    # layers analysis
    from tqdm import tqdm
    
    for i in tqdm(range(len(weights['model_0']['vit']['encoder']['layer']))):
        # Single Layer
        ## Matching
        w_matched = pi_match(args, weights, log_file = log_file, single_layer_index=i)
        matched_model = get_model(args, w_matched)
        matched_models = [matched_model, models[1]]
        matched_weights = {
            'model_0': model_to_dict(matched_model),
            'model_1': weights['model_1']
        }
        matched_model_vanilla_fused = do_vanilla_fusion(args, matched_weights, matched_models[0], matched_models[1])
        test_accuracy = get_test_acc(args, matched_model_vanilla_fused, test_dataloader, device)
        log.info("Vanilla Fusion Matched Accuracy Single Layer {}: {}".format(i, test_accuracy))
        results['single_layer']['match']['vanilla'].append(test_accuracy)
        val_results['single_layer']['match']['vanilla'].append(get_test_acc(args, matched_model_vanilla_fused, val_dataloader, device))
        w_matched_ot = ot_fusion_pipeline(args, matched_weights, device, LOGGING_LEVEL, log_file, log, matched_models)
        matched_model_ot_fused = get_model(args, w_matched_ot)
        test_accuracy = get_test_acc(args, matched_model_ot_fused, test_dataloader, device)
        results['single_layer']['match']['ot'].append(test_accuracy)
        val_results['single_layer']['match']['ot'].append(get_test_acc(args, matched_model_ot_fused, val_dataloader, device))
        ## Scaling Matching
        scaling_matched_model = get_model(args, pi_match(args, weights, log_file = log_file, single_layer_index=i, use_scaling = True))
        scaling_matched_models = [scaling_matched_model, models[1]]
        scaling_matched_weights = {
            'model_0': model_to_dict(scaling_matched_model),
            'model_1': weights['model_1']
        }
        scaling_matched_model_vanilla_fused = do_vanilla_fusion(args, scaling_matched_weights, scaling_matched_models[0], scaling_matched_models[1])
        test_accuracy = get_test_acc(args, scaling_matched_model_vanilla_fused, test_dataloader, device)
        log.info("Vanilla Fusion Scaling Matched Accuracy Single Layer {}: {}".format(i, test_accuracy))
        results['single_layer']['scale_match']['vanilla'].append(test_accuracy)
        val_results['single_layer']['scale_match']['vanilla'].append(get_test_acc(args, scaling_matched_model_vanilla_fused, val_dataloader, device))
        w_scaling_matched_ot = ot_fusion_pipeline(args, scaling_matched_weights, device, LOGGING_LEVEL, log_file, log, scaling_matched_models)
        scaling_matched_model_ot_fused = get_model(args, w_scaling_matched_ot)
        test_accuracy = get_test_acc(args, scaling_matched_model_ot_fused, test_dataloader, device)
        results['single_layer']['scale_match']['ot'].append(test_accuracy)
        val_results['single_layer']['scale_match']['ot'].append(get_test_acc(args, scaling_matched_model_ot_fused, val_dataloader, device))
        _, matched_model_fisher_fused, scaling_matched_model_fisher_fused = get_fisher_model(
            fisher_config,
            models,
            matched_models,
            scaling_matched_models,
            dataloader
        )
        test_accuracy = get_test_acc(args, matched_model_fisher_fused, test_dataloader, device)
        results['single_layer']['match']['fisher'].append(test_accuracy)
        val_results['single_layer']['match']['fisher'].append(get_test_acc(args, matched_model_fisher_fused, val_dataloader, device))
        test_accuracy = get_test_acc(args, scaling_matched_model_fisher_fused, test_dataloader, device)
        results['single_layer']['scale_match']['fisher'].append(test_accuracy)
        val_results['single_layer']['scale_match']['fisher'].append(get_test_acc(args, scaling_matched_model_fisher_fused, val_dataloader, device))
        _, matched_model_regmean_fused, scaling_matched_model_regmean_fused = get_regmean_model(
            regmean_config,
            models,
            matched_models,
            scaling_matched_models,
            dataloader
        )
        test_accuracy = get_test_acc(args, matched_model_regmean_fused, test_dataloader, device)
        results['single_layer']['match']['regmean'].append(test_accuracy)
        val_results['single_layer']['match']['regmean'].append(get_test_acc(args, matched_model_regmean_fused, val_dataloader, device))
        test_accuracy = get_test_acc(args, scaling_matched_model_regmean_fused, test_dataloader, device)
        results['single_layer']['scale_match']['regmean'].append(test_accuracy)
        val_results['single_layer']['scale_match']['regmean'].append(get_test_acc(args, scaling_matched_model_regmean_fused, val_dataloader, device))
        # Tail Layer
        ## Matching
        w_matched = pi_match(args, weights, log_file = log_file, tail_layer_index=i)
        matched_model = get_model(args, w_matched)
        matched_models = [matched_model, models[1]]
        matched_weights = {
            'model_0': model_to_dict(matched_model),
            'model_1': weights['model_1']
        }
        matched_model_vanilla_fused = do_vanilla_fusion(args, matched_weights, matched_models[0], matched_models[1])
        test_accuracy = get_test_acc(args, matched_model_vanilla_fused, test_dataloader, device)
        log.info("Vanilla Fusion Matched Accuracy Tail Layer {}: {}".format(i, test_accuracy))
        results['tail_layer']['match']['vanilla'].append(test_accuracy)
        val_results['tail_layer']['match']['vanilla'].append(get_test_acc(args, matched_model_vanilla_fused, val_dataloader, device))
        
        w_matched_ot = ot_fusion_pipeline(args, matched_weights, device, LOGGING_LEVEL, log_file, log, matched_models)
        matched_model_ot_fused = get_model(args, w_matched_ot)
        test_accuracy = get_test_acc(args, matched_model_ot_fused, test_dataloader, device)
        results['tail_layer']['match']['ot'].append(test_accuracy)
        val_accuracy = get_test_acc(args, matched_model_ot_fused, val_dataloader, device)
        val_results['tail_layer']['match']['ot'].append(val_accuracy)
        
        ## Scaling Matching
        scaling_matched_model = get_model(args, pi_match(args, weights, log_file = log_file, tail_layer_index=i, use_scaling = True))
        scaling_matched_models = [scaling_matched_model, models[1]]
        scaling_matched_weights = {
            'model_0': model_to_dict(scaling_matched_model),
            'model_1': weights['model_1']
        }
        scaling_matched_model_vanilla_fused = do_vanilla_fusion(args, scaling_matched_weights, scaling_matched_models[0], scaling_matched_models[1])
        test_accuracy = get_test_acc(args, scaling_matched_model_vanilla_fused, test_dataloader, device)
        log.info("Vanilla Fusion Scaling Matched Accuracy Tail Layer {}: {}".format(i, test_accuracy))
        results['tail_layer']['scale_match']['vanilla'].append(test_accuracy)
        val_results['tail_layer']['scale_match']['vanilla'].append(get_test_acc(args, scaling_matched_model_vanilla_fused, val_dataloader, device))
        w_scaling_matched_ot = ot_fusion_pipeline(args, scaling_matched_weights, device, LOGGING_LEVEL, log_file, log, scaling_matched_models)
        scaling_matched_model_ot_fused = get_model(args, w_scaling_matched_ot)
        test_accuracy = get_test_acc(args, scaling_matched_model_ot_fused, test_dataloader, device)
        results['tail_layer']['scale_match']['ot'].append(test_accuracy)
        val_results['tail_layer']['scale_match']['ot'].append(get_test_acc(args, scaling_matched_model_ot_fused, val_dataloader, device))

        _, matched_model_fisher_fused, scaling_matched_model_fisher_fused = get_fisher_model(
            fisher_config,
            models,
            matched_models,
            scaling_matched_models,
            dataloader
        )
        test_accuracy = get_test_acc(args, matched_model_fisher_fused, test_dataloader, device)
        results['tail_layer']['match']['fisher'].append(test_accuracy)
        val_results['tail_layer']['match']['fisher'].append(get_test_acc(args, matched_model_fisher_fused, val_dataloader, device))
        test_accuracy = get_test_acc(args, scaling_matched_model_fisher_fused, test_dataloader, device)
        results['tail_layer']['scale_match']['fisher'].append(test_accuracy)
        val_results['tail_layer']['scale_match']['fisher'].append(get_test_acc(args, scaling_matched_model_fisher_fused, val_dataloader, device))

        _, matched_model_regmean_fused, scaling_matched_model_regmean_fused = get_regmean_model(
            regmean_config,
            models,
            matched_models,
            scaling_matched_models,
            dataloader
        )
        test_accuracy = get_test_acc(args, matched_model_regmean_fused, test_dataloader, device)
        results['tail_layer']['match']['regmean'].append(test_accuracy)
        val_results['tail_layer']['match']['regmean'].append(get_test_acc(args, matched_model_regmean_fused, val_dataloader, device))
        test_accuracy = get_test_acc(args, scaling_matched_model_regmean_fused, test_dataloader, device)
        results['tail_layer']['scale_match']['regmean'].append(test_accuracy)
        val_results['tail_layer']['scale_match']['regmean'].append(get_test_acc(args, scaling_matched_model_regmean_fused, val_dataloader, device))

        
    import json
    with open('results12.json', 'w') as f:
        json.dump(results, f, indent=4)
    with open('val_results12.json', 'w') as f:
        json.dump(val_results, f, indent=4)
    with open('running_time.json', 'w') as f:
        json.dump(running_time, f, indent=4)
    log.info(" ------- Done Layer Analysis -------")

def get_fisher_model(
    fisher_config, 
    models, 
    matched_models,
    scaling_matched_models,
    dataloader
):
    avg_merger = FedAvgMerger(
        config=None,
        merger_config=fisher_config,
        local_models=models,
        global_model=None
    )
    # model_fisher_fused = avg_merger.avg_merge(
    #     local_models=avg_merger.local_models,
    #     global_model=copy.deepcopy(models[0]),
    #     fisher_weights=[
    #         compute_fisher(models[0], dataloader),
    #         compute_fisher(models[1], dataloader)
    #     ]
    # )
    matched_model_fisher_fused = avg_merger.avg_merge(
        local_models=matched_models,
        global_model=copy.deepcopy(models[0]),
        fisher_weights=[
            compute_fisher(matched_models[0], dataloader),
            compute_fisher(matched_models[1], dataloader)
        ]
    )
    scaling_matched_model_fisher_fused = avg_merger.avg_merge(
        local_models=scaling_matched_models,
        global_model=copy.deepcopy(models[0]),
        fisher_weights=[
            compute_fisher(scaling_matched_models[0], dataloader),
            compute_fisher(scaling_matched_models[1], dataloader)
        ]
    )
    return None, matched_model_fisher_fused, scaling_matched_model_fisher_fused

def get_regmean_model(
    regmean_config,
    models,
    matched_models,
    scaling_matched_models,
    dataloader
):
    avg_merger = FedAvgMerger(
        config=None,
        merger_config=regmean_config,
        local_models=models,
        global_model=None
    )
    model_regmean_fused = avg_merger.avg_merge(
        local_models=avg_merger.local_models,
        global_model=copy.deepcopy(models[0]),
        all_grams=[
            compute_gram(models[0], dataloader),
            compute_gram(models[1], dataloader)
        ]
    )
    matched_model_regmean_fused = avg_merger.avg_merge(
        local_models=matched_models,
        global_model=copy.deepcopy(models[0]),
        all_grams=[
            compute_gram(matched_models[0], dataloader),
            compute_gram(matched_models[1], dataloader)
        ]
    )
    scaling_matched_model_regmean_fused = avg_merger.avg_merge(
        local_models=scaling_matched_models,
        global_model=copy.deepcopy(models[0]),
        all_grams=[
            compute_gram(scaling_matched_models[0], dataloader),
            compute_gram(scaling_matched_models[1], dataloader)
        ]
    )
    return model_regmean_fused, matched_model_regmean_fused, scaling_matched_model_regmean_fused

# Loading Arguments from experiment file
def load_args(log, EXPERIMENT_CFG_FILE, exp = None):
    """
    There are three ways in which an experiment can be defined. Below is a list ordered by priority (only experiment with highest priority is carried out)
    1. Main function input parameter 'exp'
    2. Command line specified
    3. Default experiment
    """
    if exp == None:
        if len(sys.argv) > 1:
            indices = [sys.argv.index(string) for string in sys.argv if '.yaml' in string]
            if (len(indices) > 0):
                assert(len(indices) == 1) # cannot specify multiple yaml files!
                EXPERIMENT_CFG_FILE = 'experiments/{0}'.format(sys.argv[indices[0]])
                log.info(" Running command line specified experiment: {0}".format(EXPERIMENT_CFG_FILE))
            else:
                log.info(" Using predefined experiment: {0}".format(EXPERIMENT_CFG_FILE))
        else:
            log.info(" Using predefined experiment: {0}".format(EXPERIMENT_CFG_FILE))
    else:
        EXPERIMENT_CFG_FILE = 'experiments/{0}'.format(exp)
        log.info(" Using experiment file defined by main function input parameter: {0}".format(EXPERIMENT_CFG_FILE))
    log.info(" ------- Reading Experiment Configuration -------\n")
    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXPERIMENT_CFG_FILE.split("/")[0], EXPERIMENT_CFG_FILE.split("/")[1])
    with open(cfg_file, 'r') as f:
        args = yaml.safe_load(f)
    return args    
    

def log_args(log, args, exp_mod):
    log.debug('\n{0}'.format(yaml.dump(exp_mod, indent=4)))
    log.info('\n{0}'.format(yaml.dump(args, indent=4)))

#      _             _     _ _            _                    ____                  _  __ _
#     / \   _ __ ___| |__ (_) |_ ___  ___| |_ _   _ _ __ ___  / ___| _ __   ___  ___(_)/ _(_) ___
#    / _ \ | '__/ __| '_ \| | __/ _ \/ __| __| | | | '__/ _ \ \___ \| '_ \ / _ \/ __| | |_| |/ __|
#   / ___ \| | | (__| | | | | ||  __/ (__| |_| |_| | | |  __/  ___) | |_) |  __/ (__| |  _| | (__
#  /_/   \_\_|  \___|_| |_|_|\__\___|\___|\__|\__,_|_|  \___| |____/| .__/ \___|\___|_|_| |_|\___|
#                                                                   |_|

def load_weights_and_model(args, key):
    """
    ## Description
    Loads either model or model weights from memory and returns both the model and the
    corresponding nested weights dictionary containing all the weights.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `key` Model key to retrive the model that should be loaded from the experiment dictionary (usual values `name_0` and `name_1`)
    ------
    ## Outputs
    `weights` Nested dictionary containing only the weights of the model\\
    `model` Pytorch model object
    """
    if args['model']['type'] == 'hf_vit' or args['model']['type'] == 'pi_vit':
        model = vit_helper.get_model('{0}'.format(args['model'][key]))
        weights = model_to_dict(model)
    else:
        raise NotImplementedError
    return weights, model

def get_model(args, weights):
    """
    ## Description
    Transforms the nested weights dictionary into a pytorch model object
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `weights` Nested dictionary containing only the weights of the model
    ------
    ## Outputs
    `model` Pytorch model object
    """
    if args['model']['type'] == 'hf_vit' or args['model']['type'] == 'pi_vit':
        model = vit_helper.get_model('{0}'.format(args['model']['name_1']))
        for name, _ in model.named_parameters():
            words = name.split('.')
            temp_model = model
            temp_dict = weights
            # if words[-1] == "weight":
            for w in words[:-1]:
                # Navigating the tree
                temp_model = getattr(temp_model, w)
                temp_dict = temp_dict[w]
            setattr(temp_model, words[-1], torch.nn.parameter.Parameter(temp_dict[words[-1]]))
    else:
        raise NotImplementedError
    return model

def get_dataloader(args, device):
    """
    ## Description
    Loads the dataloader from memory.
    Exceptions: For hugginface models not a dataloader is loaded but instead the raw dataset!
    The dataloader generated by this function will be used in the forward_pass() function in the get_activation() function.
    NOTE:   Two get_dataloader functions exist (get_dataloader(), get_test_dataloader()) to allow for different batch sizes
            during testing and in the get_activation() function.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `device` Pytorch device object
    ------
    ## Outputs
    `dataloader` dataloader object
    """
    if args['model']['type'] == 'hf_vit' or args['model']['type'] == 'pi_vit':
        val_ds, test_ds = vit_helper.load_dataset_vit(args['model']['dataset'], args['fusion']['acts']['seed'])
        # Create a Dataloader with torch
        dataloader = torch.utils.data.DataLoader(dataset=val_ds,
                                                collate_fn=vit_helper.collate_fn,
                                                batch_size=1,
                                                shuffle=False)
    else:
        raise NotImplementedError
    return dataloader

def get_test_dataloader(args, device):
    """
    ## Description
    Loads the dataloader from memory.
    Exceptions: For hugginface models not a dataloader is loaded but instead the raw dataset!
    The dataloader generated by this function will be used for testing the base models, the otfused model and the vanilla fused model.
    NOTE:   Two get_dataloader functions exist (get_dataloader(), get_test_dataloader()) to allow for different batch sizes
            during testing and in the get_activation() function.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `device` Pytorch device object
    ------
    ## Outputs
    `test_dataloader` dataloader object
    """
    if args['model']['type'] == 'hf_vit' or args['model']['type'] == 'pi_vit':
        val_dataloader, test_dataloader = vit_helper.load_dataset_vit(args['model']['dataset'])
    else:
        raise NotImplementedError
    return val_dataloader, test_dataloader

def do_otfusion(args, weights, acts, alpha, device, LOGGING_LEVEL, log_file):
    """
    ## Description
    Perform otfusion of two
    models.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `weights` Weight dictionary containing the weights of both models (typical structure: `{model_0: {...}, model_1: {...}}`\\
    `acts` Activations dictionary containing all activations of both models (typical structure: `{model_0: {...}, model_1: {...}}`\\
    `alpha` Weighting parameter for anker model\\
    `device` Pytorch device object\\
    `LOGGING_LEVEL` Logging level\\
    `log_file` Path to logfile
    ------
    ## Outputs
    `w_fused` Nested dictionary containing only the weights of the fused model
    """
    if args['model']['type'] == 'hf_vit' or args['model']['type'] == 'pi_vit':
        w_fused = hf_vit_fusion(args = args, weights = weights, acts = acts, alpha = alpha, device = device, LOGGING_LEVEL = LOGGING_LEVEL, log_file = log_file)
    else:
        raise NotImplementedError
    return w_fused

def do_vanilla_fusion(args, weights, model_0, model_1):
    """
    ## Description
    Perform vanilla fusion of two
    models.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `weights` Weight dictionary containing the weights of both models (typical structure: `{model_0: {...}, model_1: {...}}`\\
    `model_0` Pytorch model object of model 0\\
    `model_1` Pytorch model object of model 1
    ------
    ## Outputs
    `model_vanilla_fused` Pytorch object of vanilla-fused model
    """
    if args['model']['type'] == 'hf_vit' or args['model']['type'] == 'pi_vit':
        if args['fusion']['num_models'] > 2:
            w_vf_fused = multi_model_vanilla(args, weights)
            model_vanilla_fused = get_model(args, w_vf_fused)
        else:
            model_vanilla_fused = vit_helper.get_model('{0}'.format(args['model']['name_0']))
            model_vanilla_fused = vanilla_fusion_old(model_0, model_1, model_vanilla_fused)
    else:
        raise NotImplementedError
    return model_vanilla_fused

def get_test_acc(args, model, dataloader, device):
    """
    ## Description
    Tests model and returns
    accuracy over the test set.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `model` Pytorch model object\\
    `dataloader` Dataloader objet\\
    `device` Pytorch device object
    ------
    ## Outputs
    `acc` Accuracy
    """
    if args['model']['type'] == 'hf_vit' or args['model']['type'] == 'pi_vit':
        acc = vit_helper.evaluate_vit(model, dataloader)
    else:
        raise NotImplementedError
    return acc


def get_distance(w_fused, w_original, key = None):
    """
    ## Description
    Compute the distance between the fused model and the original model.
    ------
    ## Parameters
    `w_fused` Nested dictionary containing only the weights of the fused model\\
    `w_original` Nested dictionary containing only the weights of the original model
    ------
    ## Outputs
    `distance` Distance between the models
    """
    total_distance = 0
    distance = {}
    for key in w_fused.keys():
        if isinstance(w_fused[key], dict):
            distance[key] = {}
            tmp_distance, distance[key] = get_distance(w_fused[key], w_original[key], key)
            total_distance += tmp_distance
        elif isinstance(w_fused[key], torch.Tensor):
            total_distance += torch.dist(w_fused[key].cpu().detach(), w_original[key].cpu().detach()).numpy() ** 2
            distance[key] = torch.dist(w_fused[key].cpu().detach(), w_original[key].cpu().detach()).numpy() ** 2
    return total_distance, distance

if __name__ == '__main__':
    main()
