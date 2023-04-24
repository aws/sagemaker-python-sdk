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

import json
import os

import boto3
import pytest
import tests.integ

from botocore.config import Config
from packaging.version import Version

from sagemaker import Session, image_uris, utils, get_execution_role
from sagemaker.local import LocalSession
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession

DEFAULT_REGION = "us-west-2"
CUSTOM_BUCKET_NAME_PREFIX = "sagemaker-custom-bucket"
CUSTOM_S3_OBJECT_KEY_PREFIX = "session-default-prefix"

NO_M4_REGIONS = [
    "eu-west-3",
    "eu-north-1",
    "ap-east-1",
    "ap-northeast-1",  # it has m4.xl, but not enough in all AZs
    "sa-east-1",
    "me-south-1",
]

NO_P3_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-southeast-1",  # it has p3, but not enough
    "ap-southeast-2",  # it has p3, but not enough
    "ca-central-1",  # it has p3, but not enough
    "eu-central-1",  # it has p3, but not enough
    "eu-north-1",
    "eu-west-2",  # it has p3, but not enough
    "eu-west-3",
    "eu-south-1",
    "me-south-1",
    "sa-east-1",
    "us-west-1",
    "ap-south-1",  # no p3 availability
]

NO_T2_REGIONS = ["eu-north-1", "ap-east-1", "me-south-1"]

FRAMEWORKS_FOR_GENERATED_VERSION_FIXTURES = (
    "chainer",
    "coach_mxnet",
    "coach_tensorflow",
    "inferentia_mxnet",
    "inferentia_tensorflow",
    "inferentia_pytorch",
    "mxnet",
    "neo_mxnet",
    "neo_pytorch",
    "neo_tensorflow",
    "pytorch",
    "pytorch_training_compiler",
    "ray_pytorch",
    "ray_tensorflow",
    "sklearn",
    "tensorflow",
    "vw",
    "xgboost",
    "spark",
    "huggingface",
    "autogluon",
    "huggingface_training_compiler",
)

PYTORCH_RENEWED_GPU = "ml.g4dn.xlarge"


def pytest_addoption(parser):
    parser.addoption("--sagemaker-client-config", action="store", default=None)
    parser.addoption("--sagemaker-runtime-config", action="store", default=None)
    parser.addoption("--boto-config", action="store", default=None)
    parser.addoption("--sagemaker-metrics-config", action="store", default=None)


def pytest_configure(config):
    bc = config.getoption("--boto-config")
    parsed = json.loads(bc) if bc else {}
    region = parsed.get("region_name", boto3.session.Session().region_name)
    if region:
        os.environ["TEST_AWS_REGION_NAME"] = region


@pytest.fixture(scope="session")
def sagemaker_client_config(request):
    config = request.config.getoption("--sagemaker-client-config")
    return json.loads(config) if config else dict()


@pytest.fixture(scope="session")
def sagemaker_runtime_config(request):
    config = request.config.getoption("--sagemaker-runtime-config")
    return json.loads(config) if config else None


@pytest.fixture(scope="session")
def sagemaker_metrics_config(request):
    config = request.config.getoption("--sagemaker-metrics-config")
    return json.loads(config) if config else None


@pytest.fixture(scope="session")
def boto_session(request):
    config = request.config.getoption("--boto-config")
    if config:
        return boto3.Session(**json.loads(config))
    else:
        return boto3.Session(region_name=DEFAULT_REGION)


@pytest.fixture(scope="session")
def account(boto_session):
    return boto_session.client("sts").get_caller_identity()["Account"]


@pytest.fixture(scope="session")
def region(boto_session):
    return boto_session.region_name


@pytest.fixture(scope="session")
def sagemaker_session(
    sagemaker_client_config, sagemaker_runtime_config, boto_session, sagemaker_metrics_config
):
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )
    runtime_client = (
        boto_session.client("sagemaker-runtime", **sagemaker_runtime_config)
        if sagemaker_runtime_config
        else None
    )
    metrics_client = (
        boto_session.client("sagemaker-metrics", **sagemaker_metrics_config)
        if sagemaker_metrics_config
        else None
    )

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        sagemaker_metrics_client=metrics_client,
        sagemaker_config={},
        default_bucket_prefix=CUSTOM_S3_OBJECT_KEY_PREFIX,
    )


@pytest.fixture(scope="session")
def sagemaker_local_session(boto_session):
    return LocalSession(boto_session=boto_session)


@pytest.fixture(scope="session")
def pipeline_session(boto_session):
    return PipelineSession(boto_session=boto_session)


@pytest.fixture(scope="session")
def local_pipeline_session(boto_session):
    return LocalPipelineSession(boto_session=boto_session)


@pytest.fixture(scope="session")
def execution_role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture(scope="module")
def custom_bucket_name(boto_session):
    region = boto_session.region_name
    account = boto_session.client(
        "sts", region_name=region, endpoint_url=utils.sts_regional_endpoint(region)
    ).get_caller_identity()["Account"]
    return "{}-{}-{}".format(CUSTOM_BUCKET_NAME_PREFIX, region, account)


@pytest.fixture(scope="module", params=["py2", "py3"])
def chainer_py_version(request):
    return request.param


@pytest.fixture(scope="module", params=["py2", "py3"])
def mxnet_inference_py_version(mxnet_inference_version, request):
    if Version(mxnet_inference_version) < Version("1.7.0"):
        return request.param
    elif Version(mxnet_inference_version) == Version("1.8.0"):
        return "py37"
    elif Version(mxnet_inference_version) == Version("1.9.0"):
        return "py38"
    else:
        return "py3"


@pytest.fixture(scope="module", params=["py2", "py3"])
def mxnet_training_py_version(mxnet_training_version, request):
    if Version(mxnet_training_version) < Version("1.7.0"):
        return request.param
    elif Version(mxnet_training_version) == Version("1.8.0"):
        return "py37"
    elif Version(mxnet_training_version) == Version("1.9.0"):
        return "py38"
    else:
        return "py3"


@pytest.fixture(scope="module", params=["py2", "py3"])
def mxnet_eia_py_version(mxnet_eia_version, request):
    if Version(mxnet_eia_version) < Version("1.7.0"):
        return request.param
    else:
        return "py3"


@pytest.fixture(scope="module")
def mxnet_eia_latest_py_version():
    return "py3"


@pytest.fixture(scope="module", params=["py2", "py3"])
def pytorch_training_py_version(pytorch_training_version, request):
    if Version(pytorch_training_version) >= Version("2.0"):
        return "py310"
    elif Version(pytorch_training_version) >= Version("1.13"):
        return "py39"
    elif Version(pytorch_training_version) >= Version("1.9"):
        return "py38"
    elif Version(pytorch_training_version) >= Version("1.5.0"):
        return "py3"
    else:
        return request.param


@pytest.fixture(scope="module", params=["py2", "py3"])
def pytorch_inference_py_version(pytorch_inference_version, request):
    if Version(pytorch_inference_version) >= Version("2.0"):
        return "py310"
    elif Version(pytorch_inference_version) >= Version("1.13"):
        return "py39"
    elif Version(pytorch_inference_version) >= Version("1.9"):
        return "py38"
    elif Version(pytorch_inference_version) >= Version("1.4.0"):
        return "py3"
    else:
        return request.param


@pytest.fixture(scope="module")
def huggingface_pytorch_training_version(huggingface_training_version):
    return _huggingface_base_fm_version(
        huggingface_training_version, "pytorch", "huggingface_training"
    )[0]


@pytest.fixture(scope="module")
def huggingface_pytorch_training_py_version(huggingface_pytorch_training_version):
    if Version(huggingface_pytorch_training_version) >= Version("1.13"):
        return "py39"
    elif Version(huggingface_pytorch_training_version) >= Version("1.9"):
        return "py38"
    else:
        return "py36"


@pytest.fixture(scope="module")
def huggingface_training_compiler_pytorch_version(
    huggingface_training_compiler_version,
):
    versions = _huggingface_base_fm_version(
        huggingface_training_compiler_version, "pytorch", "huggingface_training_compiler"
    )
    if not versions:
        pytest.skip(
            f"Hugging Face Training Compiler version {huggingface_training_compiler_version} does "
            f"not have a PyTorch release."
        )
    return versions[0]


@pytest.fixture(scope="module")
def huggingface_training_compiler_tensorflow_version(
    huggingface_training_compiler_version,
):
    versions = _huggingface_base_fm_version(
        huggingface_training_compiler_version, "tensorflow", "huggingface_training_compiler"
    )
    if not versions:
        pytest.skip(
            f"Hugging Face Training Compiler version {huggingface_training_compiler_version} "
            f"does not have a TensorFlow release."
        )
    return versions[0]


@pytest.fixture(scope="module")
def huggingface_training_compiler_tensorflow_py_version(
    huggingface_training_compiler_tensorflow_version,
):
    return (
        "py37"
        if Version(huggingface_training_compiler_tensorflow_version) < Version("2.6")
        else "py38"
    )


@pytest.fixture(scope="module")
def huggingface_training_compiler_pytorch_py_version(
    huggingface_training_compiler_pytorch_version,
):
    return "py38"


@pytest.fixture(scope="module")
def huggingface_pytorch_latest_training_py_version(
    huggingface_training_pytorch_latest_version,
):
    if Version(huggingface_training_pytorch_latest_version) >= Version("1.13"):
        return "py39"
    elif Version(huggingface_training_pytorch_latest_version) >= Version("1.9"):
        return "py38"
    else:
        return "py36"


@pytest.fixture(scope="module")
def pytorch_training_compiler_py_version(
    pytorch_training_compiler_version,
):
    return "py39" if Version(pytorch_training_compiler_version) > Version("1.12") else "py38"


# TODO: Create a fixture to get the latest py version from TRCOMP image_uri.


@pytest.fixture(scope="module")
def huggingface_pytorch_latest_inference_py_version(
    huggingface_inference_pytorch_latest_version,
):
    if Version(huggingface_inference_pytorch_latest_version) >= Version("1.13"):
        return "py39"
    elif Version(huggingface_inference_pytorch_latest_version) >= Version("1.9"):
        return "py38"
    else:
        return "py36"


@pytest.fixture(scope="module")
def graviton_tensorflow_version():
    return "2.9.1"


@pytest.fixture(scope="module")
def graviton_pytorch_version():
    return "1.12.1"


@pytest.fixture(scope="module")
def graviton_xgboost_versions():
    return ["1.5-1", "1.3-1"]


@pytest.fixture(scope="module")
def graviton_sklearn_versions():
    return ["1.0-1"]


@pytest.fixture(scope="module")
def graviton_xgboost_unsupported_versions():
    return ["1", "0.90-1", "0.90-2", "1.0-1", "1.2-1", "1.2-2"]


@pytest.fixture(scope="module")
def graviton_sklearn_unsupported_versions():
    return ["0.20.0", "0.23-1"]


@pytest.fixture(scope="module")
def huggingface_tensorflow_latest_training_py_version():
    return "py38"


@pytest.fixture(scope="module")
def huggingface_neuron_latest_inference_pytorch_version():
    return "1.9"


@pytest.fixture(scope="module")
def huggingface_neuron_latest_inference_transformer_version():
    return "4.12"


@pytest.fixture(scope="module")
def huggingface_neuron_latest_inference_py_version():
    return "py37"


@pytest.fixture(scope="module")
def pytorch_neuron_version():
    return "1.11"


@pytest.fixture(scope="module")
def pytorch_eia_py_version():
    return "py3"


@pytest.fixture(scope="module")
def neo_pytorch_latest_py_version():
    return "py3"


@pytest.fixture(scope="module")
def neo_pytorch_compilation_job_name():
    return utils.name_from_base("pytorch-neo-model")


@pytest.fixture(scope="module")
def neo_pytorch_target_device():
    return "ml_c5"


@pytest.fixture(scope="module")
def neo_pytorch_cpu_instance_type():
    return "ml.c5.xlarge"


@pytest.fixture(scope="module")
def xgboost_framework_version(xgboost_version):
    if xgboost_version in ("1", "latest"):
        pytest.skip("Skipping XGBoost algorithm version.")
    return xgboost_version


@pytest.fixture(scope="module")
def xgboost_gpu_framework_version(xgboost_version):
    if xgboost_version in ("1", "latest"):
        pytest.skip("Skipping XGBoost algorithm version.")
    if Version(xgboost_version) < Version("1.2"):
        pytest.skip("Skipping XGBoost cpu-only version.")
    return xgboost_version


@pytest.fixture(scope="module", params=["py2", "py3"])
def tensorflow_training_py_version(tensorflow_training_version, request):
    return _tf_py_version(tensorflow_training_version, request)


@pytest.fixture(scope="module", params=["py2", "py3"])
def tensorflow_inference_py_version(tensorflow_inference_version, request):
    version = Version(tensorflow_inference_version)
    if version == Version("1.15") or Version("1.15.4") <= version < Version("1.16"):
        return "py36"
    return _tf_py_version(tensorflow_inference_version, request)


def _tf_py_version(tf_version, request):
    version = Version(tf_version)
    if version == Version("1.15") or Version("1.15.4") <= version < Version("1.16"):
        return "py3"
    if version < Version("1.11"):
        return "py2"
    if version == Version("2.0") or Version("2.0.3") <= version < Version("2.1"):
        return "py3"
    if version == Version("2.1") or Version("2.1.2") <= version < Version("2.2"):
        return "py3"
    if version < Version("2.2"):
        return request.param
    if Version("2.2") <= version < Version("2.6"):
        return "py37"
    if Version("2.6") <= version < Version("2.8"):
        return "py38"
    if Version("2.8") <= version < Version("2.12"):
        return "py39"
    return "py310"


@pytest.fixture(scope="module")
def tf_full_version(tensorflow_training_latest_version, tensorflow_inference_latest_version):
    """Fixture for TF tests that test both training and inference.

    Fixture exists as such, since TF training and TFS have different latest versions.
    Otherwise, this would simply be a single latest version.
    """
    return str(
        min(
            Version(tensorflow_training_latest_version),
            Version(tensorflow_inference_latest_version),
        )
    )


@pytest.fixture(scope="module")
def tf_full_py_version(tf_full_version):
    """Fixture to match tf_full_version

    Fixture exists as such, since TF training and TFS have different latest versions.
    Otherwise, this would simply be py37 to match the latest version support.
    """
    version = Version(tf_full_version)
    if version < Version("1.11"):
        return "py2"
    if version < Version("2.2"):
        return "py3"
    if version < Version("2.6"):
        return "py37"
    if version < Version("2.8"):
        return "py38"
    return "py39"


@pytest.fixture(scope="module")
def pytorch_ddp_py_version():
    return "py3"


@pytest.fixture(
    scope="module", params=["1.10", "1.10.0", "1.10.2", "1.11", "1.11.0", "1.12", "1.12.0"]
)
def pytorch_ddp_framework_version(request):
    return request.param


@pytest.fixture(scope="module")
def torch_distributed_py_version():
    return "py3"


@pytest.fixture(scope="module", params=["1.11.0"])
def torch_distributed_framework_version(request):
    return request.param


@pytest.fixture(scope="session")
def cpu_instance_type(sagemaker_session, request):
    region = sagemaker_session.boto_session.region_name
    if region in NO_M4_REGIONS:
        return "ml.m5.xlarge"
    else:
        return "ml.m4.xlarge"


@pytest.fixture(scope="session")
def gpu_instance_type(sagemaker_session, request):
    region = sagemaker_session.boto_session.region_name
    if region in NO_P3_REGIONS:
        return "ml.p2.xlarge"
    else:
        return "ml.p3.2xlarge"


@pytest.fixture()
def gpu_pytorch_instance_type(sagemaker_session, request):
    fw_version = None
    for pytorch_version_fixture in [
        "pytorch_inference_version",
        "huggingface_training_pytorch_latest_version",
        "huggingface_inference_pytorch_latest_version",
    ]:
        if pytorch_version_fixture in request.fixturenames:
            fw_version = request.getfixturevalue(pytorch_version_fixture)
    if fw_version is None:
        fw_version = request.param
    region = sagemaker_session.boto_session.region_name
    if region in NO_P3_REGIONS:
        if Version(fw_version) >= Version("1.13"):
            return PYTORCH_RENEWED_GPU
        else:
            return "ml.p2.xlarge"
    else:
        return "ml.p3.2xlarge"


@pytest.fixture(scope="session")
def gpu_instance_type_list(sagemaker_session, request):
    region = sagemaker_session.boto_session.region_name
    if region in NO_P3_REGIONS:
        return ["ml.p2.xlarge"]
    else:
        return ["ml.p3.2xlarge", "ml.p2.xlarge"]


@pytest.fixture(scope="session")
def inf_instance_type(sagemaker_session, request):
    return "ml.inf1.xlarge"


@pytest.fixture(scope="session")
def ec2_instance_type(cpu_instance_type):
    return cpu_instance_type[3:]


@pytest.fixture(scope="session")
def alternative_cpu_instance_type(sagemaker_session, request):
    region = sagemaker_session.boto_session.region_name
    if region in NO_T2_REGIONS:
        # T3 is not supported by hosting yet
        return "ml.c5.xlarge"
    else:
        return "ml.t2.medium"


@pytest.fixture(scope="session")
def cpu_instance_family(cpu_instance_type):
    return "_".join(cpu_instance_type.split(".")[0:2])


@pytest.fixture(scope="session")
def inf_instance_family(inf_instance_type):
    return "_".join(inf_instance_type.split(".")[0:2])


def pytest_generate_tests(metafunc):
    if "instance_type" in metafunc.fixturenames:
        boto_config = metafunc.config.getoption("--boto-config")
        parsed_config = json.loads(boto_config) if boto_config else {}
        region = parsed_config.get("region_name", DEFAULT_REGION)
        cpu_instance_type = "ml.m5.xlarge" if region in NO_M4_REGIONS else "ml.m4.xlarge"

        params = [cpu_instance_type]
        if not (
            region in tests.integ.HOSTING_NO_P3_REGIONS
            or region in tests.integ.TRAINING_NO_P3_REGIONS
        ):
            params.append("ml.p3.2xlarge")
        elif not (
            region in tests.integ.HOSTING_NO_P2_REGIONS
            or region in tests.integ.TRAINING_NO_P2_REGIONS
        ):
            params.append("ml.p2.xlarge")

        metafunc.parametrize("instance_type", params, scope="session")

    _generate_all_framework_version_fixtures(metafunc)


def _generate_all_framework_version_fixtures(metafunc):
    for fw in FRAMEWORKS_FOR_GENERATED_VERSION_FIXTURES:
        config = image_uris.config_for_framework(fw.replace("_", "-"))
        if "scope" in config:
            _parametrize_framework_version_fixtures(metafunc, fw, config)
        else:
            for image_scope in config.keys():
                if fw in ("xgboost", "sklearn"):
                    _parametrize_framework_version_fixtures(metafunc, fw, config[image_scope])
                    # XGB and SKLearn use the same configs for training,
                    # inference, and graviton_inference. Break after first
                    # iteration to avoid duplicate KeyError
                    break
                fixture_prefix = f"{fw}_{image_scope}" if image_scope not in fw else fw
                _parametrize_framework_version_fixtures(
                    metafunc, fixture_prefix, config[image_scope]
                )


def _huggingface_base_fm_version(huggingface_version, base_fw, fixture_prefix):
    config_name = (
        "huggingface-training-compiler" if "training_compiler" in fixture_prefix else "huggingface"
    )
    config = image_uris.config_for_framework(config_name)
    if "training" in fixture_prefix:
        hf_config = config.get("training")
    else:
        hf_config = config.get("inference")
    original_version = huggingface_version
    if "version_aliases" in hf_config:
        huggingface_version = hf_config.get("version_aliases").get(
            huggingface_version, huggingface_version
        )
    version_config = hf_config.get("versions").get(huggingface_version)
    versions = list()

    for key in list(version_config.keys()):
        if key.startswith(base_fw):
            base_fw_version = key[len(base_fw) :]
            if len(original_version.split(".")) == 2:
                base_fw_version = ".".join(base_fw_version.split(".")[:-1])
            versions.append(base_fw_version)
    return sorted(versions, reverse=True)


def _generate_huggingface_base_fw_latest_versions(
    metafunc, fixture_prefix, huggingface_version, base_fw
):
    versions = _huggingface_base_fm_version(huggingface_version, base_fw, fixture_prefix)
    fixture_name = f"{fixture_prefix}_{base_fw}_latest_version"

    if fixture_name in metafunc.fixturenames:
        metafunc.parametrize(fixture_name, versions, scope="session")


def _parametrize_framework_version_fixtures(metafunc, fixture_prefix, config):
    fixture_name = "{}_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        versions = list(config["versions"].keys()) + list(config.get("version_aliases", {}).keys())
        metafunc.parametrize(fixture_name, versions, scope="session")

    latest_version = sorted(config["versions"].keys(), key=lambda v: Version(v))[-1]

    fixture_name = "{}_latest_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        metafunc.parametrize(fixture_name, (latest_version,), scope="session")

    if "huggingface" in fixture_prefix:
        _generate_huggingface_base_fw_latest_versions(
            metafunc, fixture_prefix, latest_version, "pytorch"
        )
        _generate_huggingface_base_fw_latest_versions(
            metafunc, fixture_prefix, latest_version, "tensorflow"
        )

    fixture_name = "{}_latest_py_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        config = config["versions"]
        py_versions = config[latest_version].get("py_versions", config[latest_version].keys())
        if "repository" in py_versions or "registries" in py_versions:
            # Config did not specify `py_versions` and is not arranged by py_version. Assume py3
            metafunc.parametrize(fixture_name, ("py3",), scope="session")
        else:
            metafunc.parametrize(fixture_name, (sorted(py_versions)[-1],), scope="session")
