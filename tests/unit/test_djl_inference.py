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

import pytest
from mock import Mock

from sagemaker.djl_inference import (
    DJLModel,
)
from sagemaker.session_settings import SessionSettings
from sagemaker import image_uris

VALID_UNCOMPRESSED_MODEL_DATA = "s3://mybucket/model"
VALID_COMPRESSED_MODEL_DATA = "s3://mybucket/model.tar.gz"
HF_MODEL_ID = "hf_hub_model_id"
ROLE = "dummy_role"
REGION = "us-west-2"
VERSION = "0.28.0"

LMI_IMAGE_URI = image_uris.retrieve(framework="djl-lmi", version=VERSION, region=REGION)
TRT_IMAGE_URI = image_uris.retrieve(framework="djl-tensorrtllm", version=VERSION, region=REGION)
TNX_IMAGE_URI = image_uris.retrieve(framework="djl-neuronx", version=VERSION, region=REGION)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session = Mock(
        "sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_resources=None,
        s3_client=None,
        settings=SessionSettings(),
        create_model=Mock(name="create_model"),
        endpoint_from_production_variants=Mock(name="endpoint_from_production_variants"),
        default_bucket_prefix=None,
    )
    session.default_bucket = Mock(name="default_bucket", return_value="bucket")
    # For tests which doesn't verify config file injection, operate with empty config

    session.sagemaker_config = {}
    return session


def test_create_djl_model_only_model_id(sagemaker_session):
    model = DJLModel(
        model_id=VALID_UNCOMPRESSED_MODEL_DATA,
        sagemaker_session=sagemaker_session,
        role=ROLE,
    )
    assert model.engine == "Python"
    assert model.image_uri == LMI_IMAGE_URI
    assert model.env == {"HF_MODEL_ID": VALID_UNCOMPRESSED_MODEL_DATA, "OPTION_ENGINE": "Python"}


def test_create_djl_model_only_model_data(sagemaker_session):
    model = DJLModel(
        model_data={
            "S3DataSource": {
                "S3Uri": VALID_COMPRESSED_MODEL_DATA,
                "S3DataType": "S3Object",
                "CompressionType": "Gzip",
            }
        },
        sagemaker_session=sagemaker_session,
        role=ROLE,
    )
    assert model.engine == "Python"
    assert model.image_uri == LMI_IMAGE_URI
    assert model.env == {"OPTION_ENGINE": "Python"}


def test_create_djl_model_with_task(sagemaker_session):
    model = DJLModel(
        model_id=VALID_UNCOMPRESSED_MODEL_DATA,
        sagemaker_session=sagemaker_session,
        role=ROLE,
        task="text-generation",
    )
    assert model.engine == "Python"
    assert model.image_uri == LMI_IMAGE_URI
    assert model.env == {
        "HF_MODEL_ID": VALID_UNCOMPRESSED_MODEL_DATA,
        "OPTION_ENGINE": "Python",
        "HF_TASK": "text-generation",
    }

    model = DJLModel(
        model_id=HF_MODEL_ID,
        sagemaker_session=sagemaker_session,
        role=ROLE,
        task="text-embedding",
    )
    assert model.engine == "OnnxRuntime"
    assert model.image_uri == LMI_IMAGE_URI
    assert model.env == {
        "HF_MODEL_ID": HF_MODEL_ID,
        "OPTION_ENGINE": "OnnxRuntime",
        "HF_TASK": "text-embedding",
    }


def test_create_djl_model_with_provided_image(sagemaker_session):
    for img_uri in [LMI_IMAGE_URI, TRT_IMAGE_URI, TNX_IMAGE_URI]:
        model = DJLModel(
            model_id=VALID_UNCOMPRESSED_MODEL_DATA,
            sagemaker_session=sagemaker_session,
            role=ROLE,
            image_uri=img_uri,
        )
        assert model.engine == "Python"
        assert model.image_uri == img_uri
        assert model.env == {
            "HF_MODEL_ID": VALID_UNCOMPRESSED_MODEL_DATA,
            "OPTION_ENGINE": "Python",
        }

    for framework in ["djl-lmi", "djl-tensorrtllm", "djl-neuronx"]:
        model = DJLModel(
            model_id=VALID_UNCOMPRESSED_MODEL_DATA,
            sagemaker_session=sagemaker_session,
            role=ROLE,
            djl_framework=framework,
        )
        assert model.engine == "Python"
        assert model.image_uri == image_uris.retrieve(
            framework=framework, version=VERSION, region=REGION
        )
        assert model.env == {
            "HF_MODEL_ID": VALID_UNCOMPRESSED_MODEL_DATA,
            "OPTION_ENGINE": "Python",
        }


def test_create_djl_model_all_provided_args(sagemaker_session):
    model = DJLModel(
        model_id=HF_MODEL_ID,
        sagemaker_session=sagemaker_session,
        role=ROLE,
        task="text-generation",
        djl_framework="djl-tensorrtllm",
        dtype="fp16",
        tensor_parallel_degree=4,
        min_workers=1,
        max_workers=4,
        job_queue_size=12,
        parallel_loading=True,
        model_loading_timeout=10,
        prediction_timeout=3,
        huggingface_hub_token="token",
    )

    assert model.engine == "Python"
    assert model.image_uri == TRT_IMAGE_URI
    assert model.env == {
        "HF_MODEL_ID": HF_MODEL_ID,
        "OPTION_ENGINE": "Python",
        "HF_TASK": "text-generation",
        "TENSOR_PARALLEL_DEGREE": "4",
        "SERVING_MIN_WORKERS": "1",
        "SERVING_MAX_WORKERS": "4",
        "SERVING_JOB_QUEUE_SIZE": "12",
        "OPTION_PARALLEL_LOADING": "True",
        "OPTION_MODEL_LOADING_TIMEOUT": "10",
        "OPTION_PREDICT_TIMEOUT": "3",
        "HF_TOKEN": "token",
        "OPTION_DTYPE": "fp16",
    }
