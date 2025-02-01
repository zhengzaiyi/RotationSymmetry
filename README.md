# Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion
This repository contains the code used in our experiments, which is based on the code of [Jin et al.](https://github.com/bloomberg/dataless-model-merging)


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
