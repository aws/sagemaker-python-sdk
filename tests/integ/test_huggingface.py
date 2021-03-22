# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

from sagemaker.huggingface import HuggingFace
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout


def test_huggingface_training(
    sagemaker_session,
    gpu_instance_type,
    huggingface_training_latest_version,
    huggingface_pytorch_version,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "huggingface")

        hf = HuggingFace(
            py_version="py36",
            entry_point="examples/text-classification/run_glue.py",
            role="SageMakerRole",
            transformers_version=huggingface_training_latest_version,
            pytorch_version=huggingface_pytorch_version,
            instance_count=1,
            instance_type=gpu_instance_type,
            hyperparameters={
                "model_name_or_path": "distilbert-base-cased",
                "task_name": "wnli",
                "do_train": True,
                "do_eval": True,
                "max_seq_length": 128,
                "fp16": True,
                "per_device_train_batch_size": 128,
                "output_dir": "/opt/ml/model",
            },
            sagemaker_session=sagemaker_session,
            git_config={
                "repo": "https://github.com/huggingface/transformers.git",
                "branch": f"v{huggingface_training_latest_version}",
            },
        )

        train_input = hf.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/huggingface/train"
        )

        hf.fit(train_input)
