# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Modified 2026-10-18: use the released channel configuration; remove machine paths.
import json
from pathlib import Path
import ml_collections
import torch

def get_config():
    root=Path(__file__).resolve().parents[5]
    config=ml_collections.ConfigDict(json.loads((root/'configs/order2.json').read_text(encoding='utf-8')))
    config.device=torch.device('cpu')
    return config
