from unittest.mock import patch
import pytest

from sagemaker.core.config.config_schema import (
    IMAGE_RETRIEVER,
    MODULES,
    PYTHON_SDK,
    SAGEMAKER,
    _simple_path,
)
from sagemaker.core.image_retriever.image_retriever import ImageRetriever
from sagemaker.core.config.config_manager import SageMakerConfig

@pytest.mark.skip("Disabling this for now, Need to be fixed")
@pytest.mark.integ
def test_retrieve_image_uri():
    image_uri = ImageRetriever.retrieve("clarify", "us-west-2")
    assert (
        image_uri == "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0"
    )

    image_uri = ImageRetriever.retrieve(
        framework="sagemaker-distribution",
        image_scope="inference",
        instance_type="ml.g5.4xlarge",
        region="us-west-1",
    )
    assert (
        image_uri
        == "053634841547.dkr.ecr.us-west-1.amazonaws.com/sagemaker-distribution-prod:3.0.0-gpu"
    )

    image_uri = ImageRetriever.retrieve(
        "xgboost",
        "eu-west-1",
        version="0.90-1",
        instance_type="ml.m5.xlarge",
        image_scope="inference",
    )
    assert (
        image_uri == "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-xgboost:0.90-1-cpu-py3"
    )

    image_uri = ImageRetriever.retrieve(
        framework="tensorflow",
        region="us-west-2",
        version="2.3",
        py_version="py37",
        instance_type="ml.p4d.24xlarge",
        image_scope="training",
    )
    assert (
        image_uri
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3-gpu-py37-cu110-ubuntu18.04-v3"
    )

@pytest.mark.skip("Disabling this for now, Need to be fixed")
@pytest.mark.integ
def test_retrieve_pytorch_uri():
    image_uri = ImageRetriever.retrieve_pytorch_uri(
        region="us-west-2",
        version="1.6",
        py_version="py3",
        instance_type="ml.p4d.24xlarge",
        image_scope="training",
    )
    assert (
        image_uri
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.6-gpu-py3-cu110-ubuntu18.04-v3"
    )

@pytest.mark.skip("Disabling this for now, Need to be fixed")
@pytest.mark.integ
def test_retrieve_hugging_face_uri():
    image_uri = ImageRetriever.retrieve_hugging_face_uri(
        version="4.28.1",
        py_version="py310",
        instance_type="ml.p2.xlarge",
        region="us-east-1",
        image_scope="training",
        base_framework_version="pytorch2.0.0",
        container_version="cu110-ubuntu20.04",
    )
    assert image_uri == "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training"
    ":2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"

@pytest.mark.skip("Disabling this for now, Need to be fixed")
@pytest.mark.integ
def test_retrieve_base_python_image_uri():
    image_uri = ImageRetriever.retrieve_base_python_image_uri()
    assert image_uri == "236514542706.dkr.ecr.us-west-2.amazonaws.com/sagemaker-base-python-310:1.0"

@pytest.mark.skip("Disabling this for now, Need to be fixed")
@pytest.mark.integ
@patch.object(SageMakerConfig, "resolve_value_from_config")
def test_retrieve_image_uri_intelligent_default(mock_load_config):
    def custom_return(config_path):
        if config_path == _simple_path(
            SAGEMAKER, PYTHON_SDK, MODULES, IMAGE_RETRIEVER, "ImageScope"
        ):
            return "inference"
        if config_path == _simple_path(
            SAGEMAKER, PYTHON_SDK, MODULES, IMAGE_RETRIEVER, "InstanceType"
        ):
            return "ml.g5.4xlarge"

    mock_load_config.side_effect = custom_return

    # Will get image_scope="inference" and instance_type="ml.g5.4xlarge" from intelligent default
    image_uri = ImageRetriever.retrieve(
        framework="sagemaker-distribution",
        region="us-west-1",
    )
    assert (
        image_uri
        == "053634841547.dkr.ecr.us-west-1.amazonaws.com/sagemaker-distribution-prod:3.0.0-gpu"
    )
