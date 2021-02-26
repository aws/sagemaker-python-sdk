# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import argparse
import gzip
import json
import logging
import os
import struct

import mxnet as mx
import numpy as np


def model_fn(model_dir):
    import eimx

    def read_data_shapes(path, preferred_batch_size=1):
        with open(path, "r") as f:
            signatures = json.load(f)

        data_names = []
        data_shapes = []

        for s in signatures:
            name = s["name"]
            data_names.append(name)

            shape = s["shape"]

            if preferred_batch_size:
                shape[0] = preferred_batch_size

            data_shapes.append((name, shape))

        return data_names, data_shapes

    shapes_file = os.path.join(model_dir, "model-shapes.json")
    data_names, data_shapes = read_data_shapes(shapes_file)

    ctx = mx.cpu()
    sym, args, aux = mx.model.load_checkpoint(os.path.join(model_dir, "model"), 0)
    sym = sym.optimize_for("EIA")

    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=data_names, label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes)
    mod.set_params(args, aux, allow_missing=True)

    return mod
