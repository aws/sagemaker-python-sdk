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

import tarfile

import botocore.exceptions
import os

import pytest
import sagemaker
import sagemaker.predictor
import sagemaker.utils
import tests.integ
import tests.integ.timeout
import numpy as np
import matplotlib.image as mpimg
from sagemaker.deserializers import JSONDeserializer
from sagemaker.tensorflow.model import TensorFlowModel, TensorFlowPredictor
from sagemaker.serializers import CSVSerializer, IdentitySerializer
from tests.integ import (
    DATA_DIR,
)
from tests.integ.timeout import timeout_and_delete_endpoint_by_name

INPUT_MODEL = os.path.join(DATA_DIR, "tensorflow-serving-test-model.tar.gz")
INFERENCE_IMAGE = os.path.join(DATA_DIR, "cuteCat.jpg")


def test_compile_and_deploy_with_accelerator(
    sagemaker_session,
    tfs_eia_cpu_instance_type, 
    tfs_eia_latest_version,
    tfs_eia_latest_py_version,
    tfs_eia_target_device,
    tfs_eia_compilation_job_name
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-tensorflow-serving")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/compiledmodels",
    )
    bucket = sagemaker_session.default_bucket()
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = TensorFlowModel(
            model_data=model_data,
            role="SageMakerRole",
            framework_version=tfs_eia_latest_version,
            py_version=tfs_eia_latest_py_version,
            sagemaker_session=sagemaker_session,
            name=endpoint_name,
        )
        data_shape = {"input": [1, 224, 224, 3]}
        compiled_model_path = "s3://{}/{}/output".format(bucket, tfs_eia_compilation_job_name)
        compiled_model = model.compile(
            target_instance_family=tfs_eia_target_device,
            input_shape=data_shape,
            output_path=compiled_model_path,
            role="SageMakerRole",
            job_name=tfs_eia_compilation_job_name,
            framework='tensorflow',
            framework_version=tfs_eia_latest_version
        )
        predictor = compiled_model.deploy(
            1, tfs_eia_cpu_instance_type, endpoint_name=endpoint_name, accelerator_type="ml.eia2.large"
        )
   
        image_path = os.path.join(tests.integ.DATA_DIR, "cuteCat.jpg")
        img = mpimg.imread(image_path)
        img = np.resize(img, (224, 224, 3))
        img = np.expand_dims(img, axis=0)
        input_data = {"inputs": img}
        result = predictor.predict(input_data)
