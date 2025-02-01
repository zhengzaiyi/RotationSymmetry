# Beyond the Permutation Symmetry of Transformers
This repository contains the code used in our experiments, which is based on the code of [Jin et al.](https://github.com/bloomberg/dataless-model-merging)

## Abstract
Symmetry in the parameter space has been found to be useful in many deep learning applications.
A simple example of parameter space symmetry is permutation symmetry where for Multi-Layer Perceptrons (MLPs), permuting the rows of two weight matrices in two adjacent layers can yield a functionally equivalent model. 
Although permutation symmetry has been studied thoroughly in basic neural architectures such as MLPs and Convolutional Neural Networks (CNNs), the discussion of transformers, an advanced and predominant architecture for language modeling, remains nascent.
In this paper, we investigate beyond the permutation symmetry of transformers and propose a novel rotation symmetry of the attention mechanism.
Additionally, we explore the application of rotation symmetry in model fusion and propose an optimal parameter matching algorithm for transformers as a plug-and-play module to improve model fusion.
We conduct extensive experiments with pre-trained language models over different Natural Language Processing (NLP) tasks.
The results show the superiority of rotation symmetry within our proposed parameter matching algorithm and reveal the potential of improving language model fusion with parameter space symmetry.
Our study provides a novel understanding of the parameter space symmetry for transformers and paves the way for enhancing the model fusion on NLP tasks using parameter space symmetry.

## Usage
### Environments
```bash
conda create -n permute python=3.10
conda activate permute
pip install -r requirements.txt
```
### Datasets
Please refer to this [link](https://github.com/bloomberg/dataless-model-merging) for the preparation of Emotion classification and NER (CoNLL2003 and Ontonotes) datasets. 

### Run Experiments
```bash
# An example for running with deberta model on the NER tasks. 
# All the scripts are listed in ./scripts folder. 
python scripts/deberta/ner.py           # baseline
python scripts/deberta/ner.py --use_pi  # our method
```

## Code File Structure

+ `./configs`: Contains `yaml` files for different experimental settings
+ `./src`  
    + `./src/data_manager`: Dataloaders for different datasets
    + `./src/model_merge`: Code of mergers and our match method  
        + `./src/model_merge/pi_merger.py`: Our match pipeline
