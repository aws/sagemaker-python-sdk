# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
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

import gzip
import os
import pickle

from tests.integ import DATA_DIR


def one_p_mnist():
    data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
    with gzip.open(data_path, "rb") as f:
        training_set, _, _ = pickle.load(f, encoding="latin1")

    return training_set
