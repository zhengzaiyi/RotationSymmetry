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

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_pi", action='store_true')
args = parser.parse_args()

pi_config = ' src/configs/exps/deberta/deberta-pi.yaml' if args.use_pi else ''

orders = ["model2", "model5", "model1", "model3", "model0"]
# Emotion MTL

for seed in [1, 2, 3, 4, 5]:
    # multi task learning comparator
    to_merge = " ".join(orders)
    os.system(
        f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_mtl.yaml{pi_config} src/configs/exps/deberta/deberta-large-emotion-mtl.yaml --filter_model {to_merge} --templates seed={seed}"
    )


for seed in [1, 2, 3, 4, 5]:
    # multi task learning comparator, ood
    to_merge = " ".join(orders)
    os.system(
        f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/emotion_mtl.yaml{pi_config} src/configs/exps/deberta/ood/deberta-large-emotion-mtl-ood.yaml --filter_model {to_merge} --templates seed={seed}"
    )
    
# NER MTL
orders = [
    "model4",
    "model3",
    "model0",
    "model1",
    "model2",
    "model5",
] 
for seed in [1, 2, 3, 4, 5]:
    # multi task learning comparator
    to_merge = " ".join(orders)
    os.system(
        f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/ner_mtl.yaml{pi_config} src/configs/exps/deberta/ner/deberta-ner-mtl.yaml --filter_model {to_merge} --templates seed={seed}"
    )
    

for seed in [1, 2, 3, 4, 5]:
    # multi task learning comparator, ood
    to_merge = " ".join(orders)
    os.system(
        f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/ner_mtl.yaml{pi_config} src/configs/exps/deberta/ner/ood/deberta-ner-mtl-ood.yaml --filter_model {to_merge} --templates seed={seed}"
    )
