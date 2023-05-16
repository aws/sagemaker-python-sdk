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

import logging

import json
from json import JSONDecodeError

import pytest
from mock import Mock, MagicMock
from mock import patch, mock_open

from sagemaker.djl_inference import (
    defaults,
    DJLModel,
    DJLPredictor,
    HuggingFaceAccelerateModel,
    DeepSpeedModel,
)
from sagemaker.djl_inference.model import DJLServingEngineEntryPointDefaults
from sagemaker.session_settings import SessionSettings

VALID_UNCOMPRESSED_MODEL_DATA = "s3://mybucket/model"
INVALID_UNCOMPRESSED_MODEL_DATA = "s3://mybucket/model.tar.gz"
HF_MODEL_ID = "hf_hub_model_id"
ENTRY_POINT = "entrypoint.py"
SOURCE_DIR = "source_dir/"
ENV = {"ENV_VAR": "env_value"}
ROLE = "dummy_role"
REGION = "us-west-2"
BUCKET = "mybucket"
IMAGE_URI = "763104351884.dkr.ecr.us-west-2.amazon.com/djl-inference:0.20.0-deepspeed0.7.5-cu116"
GPU_INSTANCE = "ml.g5.12xlarge"


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
    )
    session.default_bucket = Mock(name="default_bucket", return_valie=BUCKET)
    # For tests which doesn't verify config file injection, operate with empty config

    session.sagemaker_config = {}
    return session


def test_create_model_invalid_s3_uri():
    with pytest.raises(ValueError) as invalid_s3_data:
        _ = DJLModel(
            INVALID_UNCOMPRESSED_MODEL_DATA,
            ROLE,
        )
    assert str(invalid_s3_data.value).startswith(
        "DJLModel does not support model artifacts in tar.gz"
    )


@patch("urllib.request.urlopen")
def test_create_model_valid_hf_hub_model_id(
    mock_urlopen,
    sagemaker_session,
):
    model_config = {
        "model_type": "opt",
        "num_attention_heads": 4,
    }

    cm = MagicMock()
    cm.getcode.return_value = 200
    cm.read.return_value = json.dumps(model_config).encode("utf-8")
    cm.__enter__.return_value = cm
    mock_urlopen.return_value = cm
    model = DJLModel(
        HF_MODEL_ID,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=4,
    )
    assert model.engine == DJLServingEngineEntryPointDefaults.DEEPSPEED
    expected_url = f"https://huggingface.co/{HF_MODEL_ID}/raw/main/config.json"
    mock_urlopen.assert_any_call(expected_url)

    serving_properties = model.generate_serving_properties()
    assert serving_properties["option.model_id"] == HF_MODEL_ID
    assert "option.s3url" not in serving_properties


@patch("json.load")
@patch("urllib.request.urlopen")
def test_create_model_invalid_hf_hub_model_id(
    mock_urlopen,
    json_load,
    sagemaker_session,
):
    expected_url = f"https://huggingface.co/{HF_MODEL_ID}/raw/main/config.json"
    with pytest.raises(ValueError) as invalid_model_id:
        cm = MagicMock()
        cm.__enter__.return_value = cm
        mock_urlopen.return_value = cm
        json_load.side_effect = JSONDecodeError("", "", 0)
        _ = DJLModel(
            HF_MODEL_ID,
            ROLE,
            sagemaker_session=sagemaker_session,
            number_of_partitions=4,
        )
        mock_urlopen.assert_any_call(expected_url)
    assert str(invalid_model_id.value).startswith(
        "Did not find a config.json or model_index.json file in huggingface hub"
    )


@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_create_model_automatic_engine_selection(mock_s3_list, mock_read_file, sagemaker_session):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    hf_model_config = {
        "model_type": "t5",
        "num_attention_heads": 4,
    }
    mock_read_file.return_value = json.dumps(hf_model_config)
    hf_model = DJLModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=4,
    )
    assert hf_model.engine == DJLServingEngineEntryPointDefaults.FASTER_TRANSFORMER

    hf_model_config = {
        "model_type": "gpt2",
        "num_attention_heads": 25,
    }
    mock_read_file.return_value = json.dumps(hf_model_config)
    hf_model = DJLModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=4,
    )
    assert hf_model.engine == DJLServingEngineEntryPointDefaults.HUGGINGFACE_ACCELERATE

    for model_type in defaults.DEEPSPEED_RECOMMENDED_ARCHITECTURES:
        ds_model_config = {
            "model_type": model_type,
            "num_attention_heads": 12,
        }
        mock_read_file.return_value = json.dumps(ds_model_config)
        ds_model = DJLModel(
            VALID_UNCOMPRESSED_MODEL_DATA,
            ROLE,
            sagemaker_session=sagemaker_session,
            number_of_partitions=2,
        )
        mock_s3_list.assert_any_call(
            VALID_UNCOMPRESSED_MODEL_DATA, sagemaker_session=sagemaker_session
        )
        if model_type == defaults.STABLE_DIFFUSION_MODEL_TYPE:
            assert ds_model.engine == DJLServingEngineEntryPointDefaults.STABLE_DIFFUSION
        else:
            assert ds_model.engine == DJLServingEngineEntryPointDefaults.DEEPSPEED


@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_create_deepspeed_model(mock_s3_list, mock_read_file, sagemaker_session):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    ds_model_config = {
        "model_type": "opt",
        "n_head": 12,
    }
    mock_read_file.return_value = json.dumps(ds_model_config)
    ds_model = DeepSpeedModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        tensor_parallel_degree=4,
    )
    assert ds_model.engine == DJLServingEngineEntryPointDefaults.DEEPSPEED

    ds_model_config = {
        "model_type": "opt",
        "n_head": 25,
    }
    mock_read_file.return_value = json.dumps(ds_model_config)
    with pytest.raises(ValueError) as invalid_partitions:
        _ = DeepSpeedModel(
            VALID_UNCOMPRESSED_MODEL_DATA,
            ROLE,
            sagemaker_session=sagemaker_session,
            tensor_parallel_degree=4,
        )
    assert str(invalid_partitions.value).startswith("The number of attention heads is not evenly")


@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_create_huggingface_model(mock_s3_list, mock_read_file, sagemaker_session):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    hf_model_config = {
        "model_type": "opt",
        "n_head": 12,
    }
    mock_read_file.return_value = json.dumps(hf_model_config)
    hf_model = HuggingFaceAccelerateModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=4,
    )
    assert hf_model.engine == DJLServingEngineEntryPointDefaults.HUGGINGFACE_ACCELERATE

    hf_model_config = {
        "model_type": "t5",
        "n_head": 13,
    }
    mock_read_file.return_value = json.dumps(hf_model_config)
    hf_model = HuggingFaceAccelerateModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=4,
    )
    assert hf_model.engine == DJLServingEngineEntryPointDefaults.HUGGINGFACE_ACCELERATE


@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_model_unsupported_methods(mock_s3_list, mock_read_file, sagemaker_session):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    model_config = {
        "model_type": "opt",
        "n_head": 12,
    }
    mock_read_file.return_value = json.dumps(model_config)
    model = DJLModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
    )

    with pytest.raises(NotImplementedError) as invalid_method:
        model.package_for_edge()
    assert str(invalid_method.value).startswith("DJLModels do not support Sagemaker Edge")

    with pytest.raises(NotImplementedError) as invalid_method:
        model.compile()
    assert str(invalid_method.value).startswith(
        "DJLModels do not currently support compilation with SageMaker Neo"
    )

    with pytest.raises(NotImplementedError) as invalid_method:
        model.transformer()
    assert str(invalid_method.value).startswith(
        "DJLModels do not currently support Batch Transform inference jobs"
    )


@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_deploy_base_model_invalid_instance(mock_s3_list, mock_read_file, sagemaker_session):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    model_config = {
        "model_type": "gpt-neox",
        "n_head": 25,
    }
    mock_read_file.return_value = json.dumps(model_config)
    model = DJLModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=4,
    )

    with pytest.raises(ValueError) as invalid_instance:
        _ = model.deploy("ml.m5.12xlarge")
    assert str(invalid_instance.value).startswith("Invalid instance type. DJLModels only support")


@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_generate_deepspeed_serving_properties_invalid_configurations(
    mock_s3_list, mock_read_file, sagemaker_session
):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    model_config = {
        "model_type": "bert",
        "n_head": 4,
    }
    mock_read_file.return_value = json.dumps(model_config)
    model = DeepSpeedModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        tensor_parallel_degree=4,
        enable_cuda_graph=True,
    )
    with pytest.raises(ValueError) as invalid_config:
        _ = model.generate_serving_properties()
    assert str(invalid_config.value).startswith("enable_cuda_graph is not supported")


@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_generate_huggingface_serving_properties_invalid_configurations(
    mock_s3_list, mock_read_file, sagemaker_session
):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    model_config = {
        "model_type": "t5",
        "n_head": 4,
    }
    mock_read_file.return_value = json.dumps(model_config)
    model = HuggingFaceAccelerateModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        dtype="fp16",
        load_in_8bit=True,
    )
    with pytest.raises(ValueError) as invalid_config:
        _ = model.generate_serving_properties()
    assert str(invalid_config.value).startswith("Set dtype='int8' to use load_in_8bit")

    model = HuggingFaceAccelerateModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=2,
        device_id=1,
    )
    with pytest.raises(ValueError) as invalid_config:
        _ = model.generate_serving_properties()
    assert str(invalid_config.value).startswith(
        "device_id cannot be set when number_of_partitions is > 1"
    )


@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_generate_serving_properties_with_valid_configurations(
    mock_s3_list, mock_read_file, sagemaker_session
):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    model_config = {
        "model_type": "gpt-neox",
        "n_head": 25,
    }
    mock_read_file.return_value = json.dumps(model_config)
    model = DJLModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=4,
        min_workers=1,
        max_workers=3,
        job_queue_size=4,
        dtype="fp16",
        parallel_loading=True,
        model_loading_timeout=120,
        prediction_timeout=4,
        source_dir=SOURCE_DIR,
        entry_point=ENTRY_POINT,
        task="text-classification",
    )
    serving_properties = model.generate_serving_properties()
    expected_dict = {
        "engine": "Python",
        "option.entryPoint": ENTRY_POINT,
        "option.s3url": VALID_UNCOMPRESSED_MODEL_DATA,
        "option.tensor_parallel_degree": 4,
        "option.task": "text-classification",
        "option.dtype": "fp16",
        "minWorkers": 1,
        "maxWorkers": 3,
        "job_queue_size": 4,
        "option.parallel_loading": True,
        "option.model_loading_timeout": 120,
        "option.prediction_timeout": 4,
    }
    assert serving_properties == expected_dict
    serving_properties.clear()
    expected_dict.clear()

    model_config = {
        "model_type": "opt",
        "n_head": 4,
    }
    mock_read_file.return_value = json.dumps(model_config)
    model = DeepSpeedModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        tensor_parallel_degree=1,
        task="text-generation",
        dtype="bf16",
        max_tokens=2048,
        low_cpu_mem_usage=True,
        enable_cuda_graph=True,
    )
    serving_properties = model.generate_serving_properties()
    expected_dict = {
        "engine": "DeepSpeed",
        "option.entryPoint": "djl_python.deepspeed",
        "option.s3url": VALID_UNCOMPRESSED_MODEL_DATA,
        "option.tensor_parallel_degree": 1,
        "option.task": "text-generation",
        "option.dtype": "bf16",
        "option.max_tokens": 2048,
        "option.enable_cuda_graph": True,
        "option.low_cpu_mem_usage": True,
        "option.triangular_masking": True,
        "option.return_tuple": True,
    }
    assert serving_properties == expected_dict
    serving_properties.clear()
    expected_dict.clear()

    model = HuggingFaceAccelerateModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=1,
        device_id=4,
        device_map="balanced",
        dtype="fp32",
        low_cpu_mem_usage=False,
    )
    serving_properties = model.generate_serving_properties()
    expected_dict = {
        "engine": "Python",
        "option.entryPoint": "djl_python.huggingface",
        "option.s3url": VALID_UNCOMPRESSED_MODEL_DATA,
        "option.tensor_parallel_degree": 1,
        "option.dtype": "fp32",
        "option.device_id": 4,
        "option.device_map": "balanced",
    }
    assert serving_properties == expected_dict


@patch("sagemaker.image_uris.retrieve", return_value=IMAGE_URI)
@patch("shutil.rmtree")
@patch("sagemaker.utils.base_name_from_image")
@patch("tempfile.mkdtemp")
@patch("sagemaker.container_def")
@patch("sagemaker.utils._tmpdir")
@patch("sagemaker.utils._create_or_update_code_dir")
@patch("sagemaker.fw_utils.tar_and_upload_dir")
@patch("os.mkdir")
@patch("os.path.exists")
@patch("sagemaker.s3.S3Downloader.read_file")
@patch("sagemaker.s3.S3Downloader.list")
def test_deploy_model_no_local_code(
    mock_s3_list,
    mock_read_file,
    mock_path_exists,
    mock_mkdir,
    mock_tar_upload,
    mock_create_code_dir,
    mock_tmpdir,
    mock_container_def,
    mock_mktmp,
    mock_name_from_base,
    mock_shutil_rmtree,
    mock_imguri_retrieve,
    sagemaker_session,
):
    mock_s3_list.return_value = [VALID_UNCOMPRESSED_MODEL_DATA + "/config.json"]
    model_config = {
        "model_type": "bloom",
        "n_heads": 120,
    }
    mock_read_file.return_value = json.dumps(model_config)
    model = DJLModel(
        VALID_UNCOMPRESSED_MODEL_DATA,
        ROLE,
        sagemaker_session=sagemaker_session,
        number_of_partitions=4,
        dtype="fp16",
        container_log_level=logging.DEBUG,
        env=ENV,
    )

    assert model.image_uri is None

    mock_path_exists.side_effect = [True, False, True]
    mock_mktmp.return_value = "/tmp/dir"
    mock_tar_upload.return_value = Mock(s3_prefix="s3prefix")
    expected_env = {"ENV_VAR": "env_value", "SERVING_OPTS": '"-Dai.djl.logging.level=debug"'}
    with patch("builtins.open", mock_open()) as fake_serving_properties:
        predictor = model.deploy(GPU_INSTANCE)

        assert isinstance(predictor, DJLPredictor)
        mock_mktmp.assert_called_once_with(prefix="tmp", suffix="", dir=None)
        mock_mkdir.assert_called()
        assert fake_serving_properties.call_count == 2
        fake_serving_properties.assert_any_call("/tmp/dir/code/serving.properties", "w+")
        fake_serving_properties.assert_any_call("/tmp/dir/code/serving.properties", "r")
        model.sagemaker_session.create_model.assert_called_once()
        mock_container_def.assert_called_once_with(
            IMAGE_URI, model_data_url="s3prefix", env=expected_env
        )
