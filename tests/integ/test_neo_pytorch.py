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

import numpy as np
import os
import json

from sagemaker.pytorch.model import PyTorchModel
from sagemaker.utils import sagemaker_timestamp
from sagemaker.predictor import Predictor
from tests.integ import (
    DATA_DIR,
)
from tests.integ.timeout import timeout_and_delete_endpoint_by_name

NEO_DIR = os.path.join(DATA_DIR, "pytorch_neo")
NEO_MODEL = os.path.join(NEO_DIR, "model.tar.gz")
NEO_INFERENCE_IMAGE = os.path.join(NEO_DIR, "cat.jpg")
NEO_IMAGENET_CLASSES = os.path.join(NEO_DIR, "imagenet1000_clsidx_to_labels.txt")
NEO_CODE_DIR = os.path.join(NEO_DIR, "code")
NEO_SCRIPT = os.path.join(NEO_CODE_DIR, "inference.py")


def test_compile_and_deploy_model_with_neo(
    sagemaker_session,
    neo_pytorch_cpu_instance_type,
    neo_pytorch_latest_version,
    neo_pytorch_latest_py_version,
    neo_pytorch_target_device,
    neo_pytorch_compilation_job_name,
):
    endpoint_name = "test-neo-pytorch-deploy-model-{}".format(sagemaker_timestamp())

    model_data = sagemaker_session.upload_data(path=NEO_MODEL)
    bucket = sagemaker_session.default_bucket()
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = PyTorchModel(
            model_data=model_data,
            predictor_cls=Predictor,
            role="SageMakerRole",
            entry_point=NEO_SCRIPT,
            source_dir=NEO_CODE_DIR,
            framework_version=neo_pytorch_latest_version,
            py_version=neo_pytorch_latest_py_version,
            sagemaker_session=sagemaker_session,
            env={"MMS_DEFAULT_RESPONSE_TIMEOUT": "500"},
        )
        data_shape = '{"input0":[1,3,224,224]}'
        compiled_model_path = "s3://{}/{}/output".format(bucket, neo_pytorch_compilation_job_name)
        compiled_model = model.compile(
            target_instance_family=neo_pytorch_target_device,
            input_shape=data_shape,
            job_name=neo_pytorch_compilation_job_name,
            role="SageMakerRole",
            framework="pytorch",
            framework_version=neo_pytorch_latest_version,
            output_path=compiled_model_path,
        )

        # Load names for ImageNet classes
        object_categories = {}
        with open(NEO_IMAGENET_CLASSES, "r") as f:
            for line in f:
                if line.strip():
                    key, val = line.strip().split(":")
                    object_categories[key] = val

        with open(NEO_INFERENCE_IMAGE, "rb") as f:
            payload = f.read()
            payload = bytearray(payload)

        predictor = compiled_model.deploy(
            1, neo_pytorch_cpu_instance_type, endpoint_name=endpoint_name
        )
        response = predictor.predict(payload)
        result = json.loads(response.decode())

        assert "tiger cat" in object_categories[str(np.argmax(result))]
        assert compiled_model.framework_version == neo_pytorch_latest_version
