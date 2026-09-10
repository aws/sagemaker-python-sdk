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

from sagemaker.core import image_uris

# GPU instance used wherever a gpu image is expected.
INSTANCE_TYPE = "ml.g5.2xlarge"

# Instance types that resolve to each processor in image_uris._processor().
# m5 is a general-purpose (CPU) family; g5 is a GPU family.
PROCESSOR_INSTANCE_TYPES = {"cpu": "ml.m5.xlarge", "gpu": "ml.g5.2xlarge"}

# Single-variant configs whose tag_prefix is the full image tag, taken verbatim
# (no processors / processor_in_tag / container_version). Instance type is ignored.
WHOLE_TAG_CONFIG_FILES = [
    "llama-cpp-arm64.json",
]

# GPU-only configs on the processor schema: processors=["gpu"], processor_in_tag=false,
# and the tag tail in container_version["gpu"]. instance_type is optional today (single
# processor) and resolves to the gpu image; a cpu entry can be added later as pure data
# without changing how a gpu caller resolves.
GPU_ONLY_PROCESSOR_FILES = [
    "vllm-server.json",
    "vllm-omni.json",
    "sglang-server.json",
    "whisperx.json",
]

# Configs shipping both a cpu and a gpu image under one repository: processors=["cpu","gpu"],
# processor_in_tag=false, per-processor container_version tail. instance_type selects the
# device and is therefore required.
MULTI_PROCESSOR_FILES = [
    "ray-serve.json",
    "llama-cpp.json",
]


@pytest.mark.parametrize("load_config_and_file_name", WHOLE_TAG_CONFIG_FILES, indirect=True)
def test_serving_framework_uris(load_config_and_file_name):
    """Every (version, region) resolves to the expected repo:tag verbatim."""
    config, file_name = load_config_and_file_name
    framework = file_name[: -len(".json")]
    for version, version_config in config["versions"].items():
        repo = version_config["repository"]
        tag = version_config["tag_prefix"]
        for region, account in version_config["registries"].items():
            uri = image_uris.retrieve(
                framework=framework,
                region=region,
                version=version,
                image_scope="inference",
                instance_type=INSTANCE_TYPE,
            )
            # account (registry), region, repository and tag are config-controlled;
            # the domain suffix is resolved by botocore.
            assert uri.startswith(f"{account}.dkr.ecr.{region}."), uri
            assert uri.endswith(f"/{repo}:{tag}"), uri


@pytest.mark.parametrize("load_config_and_file_name", WHOLE_TAG_CONFIG_FILES, indirect=True)
def test_serving_framework_latest_alias(load_config_and_file_name):
    """The 'latest' alias resolves to its target version's tag."""
    config, file_name = load_config_and_file_name
    framework = file_name[: -len(".json")]
    target = config["version_aliases"]["latest"]
    expected = config["versions"][target]
    uri = image_uris.retrieve(
        framework=framework,
        region="us-west-2",
        version="latest",
        image_scope="inference",
        instance_type=INSTANCE_TYPE,
    )
    assert uri.endswith(f"/{expected['repository']}:{expected['tag_prefix']}"), uri


@pytest.mark.parametrize("load_config_and_file_name", GPU_ONLY_PROCESSOR_FILES, indirect=True)
def test_gpu_only_processor_serving_framework_uris(load_config_and_file_name):
    """GPU-only framework on the processor schema resolves to the gpu tail, and because it
    has a single processor, omitting instance_type still yields the gpu image."""
    config, file_name = load_config_and_file_name
    framework = file_name[: -len(".json")]
    for version, version_config in config["versions"].items():
        repo = version_config["repository"]
        prefix = version_config["tag_prefix"]
        gpu_tail = version_config["container_version"]["gpu"]
        expected_tag = f"{prefix}-{gpu_tail}"
        for region, account in version_config["registries"].items():
            uri = image_uris.retrieve(
                framework=framework,
                region=region,
                version=version,
                image_scope="inference",
                instance_type=INSTANCE_TYPE,
            )
            assert uri.startswith(f"{account}.dkr.ecr.{region}."), uri
            assert uri.endswith(f"/{repo}:{expected_tag}"), uri
        # instance_type is optional for a single-processor config (backward-compatible
        # with the whole-tag form these configs used before the processor-schema change).
        uri_no_instance = image_uris.retrieve(
            framework=framework,
            region="us-west-2",
            version=version,
            image_scope="inference",
        )
        assert uri_no_instance.endswith(f"/{repo}:{expected_tag}"), uri_no_instance


@pytest.mark.parametrize("framework", [f[: -len(".json")] for f in GPU_ONLY_PROCESSOR_FILES])
def test_gpu_only_processor_rejects_cpu_instance(framework):
    """Until a cpu image is added, a cpu instance type is rejected (not silently served gpu)."""
    with pytest.raises(ValueError):
        image_uris.retrieve(
            framework=framework,
            region="us-west-2",
            version="latest",
            image_scope="inference",
            instance_type=PROCESSOR_INSTANCE_TYPES["cpu"],
        )


@pytest.mark.parametrize("load_config_and_file_name", MULTI_PROCESSOR_FILES, indirect=True)
def test_processor_serving_framework_uris(load_config_and_file_name):
    """CPU/GPU share one config: the instance type selects the per-processor tag tail."""
    config, file_name = load_config_and_file_name
    framework = file_name[: -len(".json")]
    for version, version_config in config["versions"].items():
        repo = version_config["repository"]
        prefix = version_config["tag_prefix"]
        for processor, tail in version_config["container_version"].items():
            expected_tag = f"{prefix}-{tail}"
            instance_type = PROCESSOR_INSTANCE_TYPES[processor]
            for region, account in version_config["registries"].items():
                uri = image_uris.retrieve(
                    framework=framework,
                    region=region,
                    version=version,
                    image_scope="inference",
                    instance_type=instance_type,
                )
                assert uri.startswith(f"{account}.dkr.ecr.{region}."), uri
                assert uri.endswith(f"/{repo}:{expected_tag}"), uri


@pytest.mark.parametrize("load_config_and_file_name", MULTI_PROCESSOR_FILES, indirect=True)
def test_processor_serving_framework_latest_alias(load_config_and_file_name):
    """The 'latest' alias resolves to its target version's per-processor tag."""
    config, file_name = load_config_and_file_name
    framework = file_name[: -len(".json")]
    target = config["version_aliases"]["latest"]
    expected = config["versions"][target]
    repo = expected["repository"]
    prefix = expected["tag_prefix"]
    for processor, tail in expected["container_version"].items():
        uri = image_uris.retrieve(
            framework=framework,
            region="us-west-2",
            version="latest",
            image_scope="inference",
            instance_type=PROCESSOR_INSTANCE_TYPES[processor],
        )
        assert uri.endswith(f"/{repo}:{prefix}-{tail}"), uri


@pytest.mark.parametrize("framework", [f[: -len(".json")] for f in MULTI_PROCESSOR_FILES])
def test_processor_serving_framework_requires_instance_type(framework):
    """With both cpu and gpu offered, instance_type is required to disambiguate."""
    with pytest.raises(ValueError):
        image_uris.retrieve(
            framework=framework,
            region="us-west-2",
            version="latest",
            image_scope="inference",
        )


# Exact repo:tag each (framework, version, processor) must resolve to. These pin the
# literal strings independent of the config dict: the gpu rows lock backward compatibility,
# the cpu rows lock the newly added tags. A self-consistent typo in tag_prefix/
# container_version would fail here even though it passes the mechanism tests.
EXPECTED_REPO_TAGS = {
    "ray-serve": {
        ("1", "gpu"): "ray:serve-ml-sagemaker-cuda-v1",
        ("1", "cpu"): "ray:serve-ml-sagemaker-cpu-v1",
        ("1.4", "gpu"): "ray:serve-ml-sagemaker-cuda-v1.4",
        ("1.4", "cpu"): "ray:serve-ml-sagemaker-cpu-v1.4",
    },
    "llama-cpp": {
        ("1", "gpu"): "llama-cpp:server-sagemaker-cuda-v1",
        ("1", "cpu"): "llama-cpp:server-sagemaker-cpu-v1",
        ("1.0", "gpu"): "llama-cpp:server-sagemaker-cuda-v1.0",
        ("1.0", "cpu"): "llama-cpp:server-sagemaker-cpu-v1.0",
    },
}

# gpu-only frameworks: the resolved gpu tag must be byte-identical to the pre-conversion
# whole-tag value, so the processor-schema change is a no-op for existing gpu callers.
GPU_ONLY_EXPECTED_REPO_TAGS = {
    "vllm-server": {
        "2": "vllm:server-sagemaker-cuda-v2",
        "2.4": "vllm:server-sagemaker-cuda-v2.4",
    },
    "vllm-omni": {
        "1": "vllm:omni-sagemaker-cuda-v1",
        "1.6": "vllm:omni-sagemaker-cuda-v1.6",
    },
    "sglang-server": {
        "1": "sglang:server-sagemaker-cuda-v1",
        "1.3": "sglang:server-sagemaker-cuda-v1.3",
    },
    "whisperx": {
        "3.8": "whisperx:3.8-cu128-amzn2023-sagemaker",
    },
}


@pytest.mark.parametrize("framework", list(EXPECTED_REPO_TAGS))
def test_processor_serving_framework_literal_tags(framework):
    """Pin the exact repo:tag per (version, processor), not just the resolution mechanism."""
    for (version, processor), repo_tag in EXPECTED_REPO_TAGS[framework].items():
        uri = image_uris.retrieve(
            framework=framework,
            region="us-west-2",
            version=version,
            image_scope="inference",
            instance_type=PROCESSOR_INSTANCE_TYPES[processor],
        )
        assert uri.startswith("763104351884.dkr.ecr.us-west-2."), uri
        assert uri.endswith(f"/{repo_tag}"), uri


@pytest.mark.parametrize("framework", list(GPU_ONLY_EXPECTED_REPO_TAGS))
def test_gpu_only_processor_literal_tags(framework):
    """The gpu tag is byte-identical to the pre-processor-schema (whole-tag) value."""
    for version, repo_tag in GPU_ONLY_EXPECTED_REPO_TAGS[framework].items():
        uri = image_uris.retrieve(
            framework=framework,
            region="us-west-2",
            version=version,
            image_scope="inference",
            instance_type=INSTANCE_TYPE,
        )
        assert uri.endswith(f"/{repo_tag}"), uri
