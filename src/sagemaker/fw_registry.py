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
"""Placeholder docstring"""
from __future__ import absolute_import
import logging

from sagemaker.utils import get_ecr_image_uri_prefix

image_registry_map = {
    "us-west-1": {
        "sparkml-serving": "746614075791",
        "scikit-learn": "746614075791",
        "xgboost": "746614075791",
    },
    "us-west-2": {
        "sparkml-serving": "246618743249",
        "scikit-learn": "246618743249",
        "xgboost": "246618743249",
    },
    "us-east-1": {
        "sparkml-serving": "683313688378",
        "scikit-learn": "683313688378",
        "xgboost": "683313688378",
    },
    "us-east-2": {
        "sparkml-serving": "257758044811",
        "scikit-learn": "257758044811",
        "xgboost": "257758044811",
    },
    "ap-northeast-1": {
        "sparkml-serving": "354813040037",
        "scikit-learn": "354813040037",
        "xgboost": "354813040037",
    },
    "ap-northeast-2": {
        "sparkml-serving": "366743142698",
        "scikit-learn": "366743142698",
        "xgboost": "366743142698",
    },
    "ap-southeast-1": {
        "sparkml-serving": "121021644041",
        "scikit-learn": "121021644041",
        "xgboost": "121021644041",
    },
    "ap-southeast-2": {
        "sparkml-serving": "783357654285",
        "scikit-learn": "783357654285",
        "xgboost": "783357654285",
    },
    "ap-south-1": {
        "sparkml-serving": "720646828776",
        "scikit-learn": "720646828776",
        "xgboost": "720646828776",
    },
    "eu-west-1": {
        "sparkml-serving": "141502667606",
        "scikit-learn": "141502667606",
        "xgboost": "141502667606",
    },
    "eu-west-2": {
        "sparkml-serving": "764974769150",
        "scikit-learn": "764974769150",
        "xgboost": "764974769150",
    },
    "eu-central-1": {
        "sparkml-serving": "492215442770",
        "scikit-learn": "492215442770",
        "xgboost": "492215442770",
    },
    "ca-central-1": {
        "sparkml-serving": "341280168497",
        "scikit-learn": "341280168497",
        "xgboost": "341280168497",
    },
    "us-gov-west-1": {
        "sparkml-serving": "414596584902",
        "scikit-learn": "414596584902",
        "xgboost": "414596584902",
    },
    "us-iso-east-1": {
        "sparkml-serving": "833128469047",
        "scikit-learn": "833128469047",
        "xgboost": "833128469047",
    },
    "ap-east-1": {
        "sparkml-serving": "651117190479",
        "scikit-learn": "651117190479",
        "xgboost": "651117190479",
    },
    "sa-east-1": {
        "sparkml-serving": "737474898029",
        "scikit-learn": "737474898029",
        "xgboost": "737474898029",
    },
    "eu-north-1": {
        "sparkml-serving": "662702820516",
        "scikit-learn": "662702820516",
        "xgboost": "662702820516",
    },
    "eu-west-3": {
        "sparkml-serving": "659782779980",
        "scikit-learn": "659782779980",
        "xgboost": "659782779980",
    },
    "me-south-1": {
        "sparkml-serving": "801668240914",
        "scikit-learn": "801668240914",
        "xgboost": "801668240914",
    },
    "cn-north-1": {
        "sparkml-serving": "450853457545",
        "scikit-learn": "450853457545",
        "xgboost": "450853457545",
    },
    "cn-northwest-1": {
        "sparkml-serving": "451049120500",
        "scikit-learn": "451049120500",
        "xgboost": "451049120500",
    },
}


def registry(region_name, framework=None):
    """Return docker registry for the given AWS region for the given framework.
    This is only used for SparkML and Scikit-learn for now.

    Args:
        region_name:
        framework:
    """
    try:
        account_id = image_registry_map[region_name][framework]
        return get_ecr_image_uri_prefix(account_id, region_name)
    except KeyError:
        logging.error("The specific image or region does not exist")
        raise


def default_framework_uri(framework, region_name, image_tag):
    """
    Args:
        framework:
        region_name:
        image_tag:
    """
    repository_name = "sagemaker-{}".format(framework)
    account_name = registry(region_name, framework)
    return "{}/{}:{}".format(account_name, repository_name, image_tag)
