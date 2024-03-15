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

ALTERNATE_DOMAINS = {
    "cn-north-1": "amazonaws.com.cn",
    "cn-northwest-1": "amazonaws.com.cn",
    "us-iso-east-1": "c2s.ic.gov",
    "us-isob-east-1": "sc2s.sgov.gov",
}
DOMAIN = "amazonaws.com"
IMAGE_URI_FORMAT = "{}.dkr.ecr.{}.{}/{}:{}"
MONITOR_URI_FORMAT = "{}.dkr.ecr.{}.{}/sagemaker-model-monitor-analyzer"
REGION = "us-west-2"


def framework_uri(repo, fw_version, account, py_version=None, processor="cpu", region=REGION):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    tag = "-".join(x for x in (fw_version, processor, py_version) if x)

    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)


def neuron_framework_uri(
    repo,
    fw_version,
    account,
    py_version=None,
    inference_tool="neuron",
    region=REGION,
    sdk_version="sdk2.4.0",
    container_version="ubuntu20.04",
):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    tag = "-".join(
        x for x in (fw_version, inference_tool, py_version, sdk_version, container_version) if x
    )

    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)


def algo_uri(algo, account, region, version=1):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    return IMAGE_URI_FORMAT.format(account, region, domain, algo, version)


def monitor_uri(account, region=REGION):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    return MONITOR_URI_FORMAT.format(account, region, domain)


def algo_uri_with_tag(algo, account, region, tag):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    return IMAGE_URI_FORMAT.format(account, region, domain, algo, tag)


def graviton_framework_uri(
    repo,
    fw_version,
    account,
    py_version="py38",
    processor="cpu",
    region=REGION,
    container_version="ubuntu20.04-sagemaker",
):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    tag = "-".join(x for x in (fw_version, processor, py_version, container_version) if x)

    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)


def djl_framework_uri(repo, account, tag, region=REGION):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)


def sagemaker_triton_framework_uri(repo, account, tag, processor="gpu", region=REGION):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    if processor == "cpu":
        tag = f"{tag}-cpu"
    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)


def huggingface_llm_framework_uri(
    repo,
    account,
    version,
    tag,
    region=REGION,
):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)


def stabilityai_framework_uri(repo, account, tag, region=REGION):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)


def base_python_uri(repo, account, region=REGION):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    tag = "1.0"
    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)
