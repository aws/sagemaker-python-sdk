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
import argparse
import json
import os

from packaging.version import Version


def _parse_args_v1():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1)
    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))

    known, unknown = parser.parse_known_args()
    return known


def _parse_args_v2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-checkpoint-steps", type=int, default=200)
    parser.add_argument("--throttle-secs", type=int, default=60)
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--export-model-during-training", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    import tensorflow as tf

    if Version(tf.__version__) <= Version("2.5"):
        from mnist_v1 import main

        args = _parse_args_v1()
        main(args)
    else:
        from mnist_v2 import main

        args = _parse_args_v2()
        main(args)
