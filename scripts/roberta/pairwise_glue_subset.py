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

pi_config = ' src/configs/exps/roberta-base/glue/roberta-base-pi.yaml' if args.use_pi else ''

for seed in [1, 2, 3, 4, 5]:
    for task in ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte"]:
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/subsets/glue_partition_1k_niid.yaml{pi_config} src/configs/exps/roberta-base/subset/rb-1k-whead.yaml --templates seed={seed} dataset_name={task}"
        )
        # fisher
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/subsets/glue_partition_1k_niid.yaml{pi_config} src/configs/exps/roberta-base/subset/rb-1k-fisher-whead.yaml --templates seed={seed} dataset_name={task}"
        )
        # regmean
        os.system(
            f"python -m src.run_experiments --config_file src/configs/defaults.yaml src/configs/datasets/subsets/glue_partition_1k_niid.yaml{pi_config} src/configs/exps/roberta-base/subset/rb-1k-regmean-whead.yaml --templates seed={seed} dataset_name={task}"
        )
