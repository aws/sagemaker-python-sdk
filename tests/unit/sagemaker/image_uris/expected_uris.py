# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
}
DOMAIN = "amazonaws.com"
IMAGE_URI_FORMAT = "{}.dkr.ecr.{}.{}/{}:{}"
MONITOR_URI_FORMAT = "{}.dkr.ecr.{}.{}/sagemaker-model-monitor-analyzer"
REGION = "us-west-2"


def framework_uri(repo, fw_version, account, py_version=None, processor="cpu", region=REGION):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    tag = "{}-{}".format(fw_version, processor)
    if py_version:
        tag = "-".join((tag, py_version))

    return IMAGE_URI_FORMAT.format(account, region, domain, repo, tag)


def algo_uri(algo, account, region, version=1):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    return IMAGE_URI_FORMAT.format(account, region, domain, algo, version)


def monitor_uri(account, region=REGION):
    domain = ALTERNATE_DOMAINS.get(region, DOMAIN)
    return MONITOR_URI_FORMAT.format(account, region, domain)
