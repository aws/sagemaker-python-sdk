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

import os
from sagemaker.huggingface import HuggingFace
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES, timeout


def test_huggingface_torch_distributed_g5_glue(
    sagemaker_session,
    huggingface_training_latest_version,
    huggingface_training_pytorch_latest_version,
    huggingface_pytorch_latest_training_py_version,
):
    with timeout.timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator = HuggingFace(
            py_version=huggingface_pytorch_latest_training_py_version,
            source_dir=os.path.join(DATA_DIR, "huggingface"),
            entry_point="run_glue.py",
            role="SageMakerRole",
            transformers_version=huggingface_training_latest_version,
            pytorch_version=huggingface_training_pytorch_latest_version,
            instance_count=1,
            instance_type="ml.g5.4xlarge",
            hyperparameters={
                "model_name_or_path": "distilbert-base-cased",
                "task_name": "wnli",
                "do_train": True,
                "do_eval": True,
                "max_seq_length": 128,
                "fp16": True,
                "per_device_train_batch_size": 32,
                "output_dir": "/opt/ml/model",
            },
            distribution={"torch_distributed": {"enabled": True}},
            sagemaker_session=sagemaker_session,
            disable_profiler=True,
        )
        estimator.fit()
