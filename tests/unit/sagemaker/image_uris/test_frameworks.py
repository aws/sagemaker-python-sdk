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

from sagemaker import image_uris

ACCOUNT = "520713654638"
DOMAIN = "amazonaws.com"
IMAGE_URI_FORMAT = "{}.dkr.ecr.{}.{}/{}:{}-{}-{}"
REGION = "us-west-2"

ALTERNATE_REGION_DOMAIN_AND_ACCOUNTS = (
    ("ap-east-1", DOMAIN, "057415533634"),
    ("cn-north-1", "amazonaws.com.cn", "422961961927"),
    ("cn-northwest-1", "amazonaws.com.cn", "423003514399"),
    ("me-south-1", DOMAIN, "724002660598"),
    ("us-gov-west-1", DOMAIN, "246785580436"),
    ("us-iso-east-1", "c2s.ic.gov", "744548109606"),
)


def _expected_uri(
    repo, fw_version, processor, py_version, account=ACCOUNT, region=REGION, domain=DOMAIN
):
    return IMAGE_URI_FORMAT.format(account, region, domain, repo, fw_version, processor, py_version)


def test_chainer(chainer_version, chainer_py_version):
    for instance_type, processor in (("ml.c4.xlarge", "cpu"), ("ml.p2.xlarge", "gpu")):
        uri = image_uris.retrieve(
            framework="chainer",
            region=REGION,
            version=chainer_version,
            py_version=chainer_py_version,
            instance_type=instance_type,
        )
        expected = _expected_uri(
            "sagemaker-chainer", chainer_version, processor, chainer_py_version
        )
        assert expected == uri

    for region, domain, account in ALTERNATE_REGION_DOMAIN_AND_ACCOUNTS:
        uri = image_uris.retrieve(
            framework="chainer",
            region=region,
            version=chainer_version,
            py_version=chainer_py_version,
            instance_type=instance_type,
        )
        expected = _expected_uri(
            "sagemaker-chainer",
            chainer_version,
            processor,
            chainer_py_version,
            account=account,
            region=region,
            domain=domain,
        )
        assert expected == uri
