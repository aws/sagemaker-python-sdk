# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.utils import get_ecr_image_uri_prefix

image_registry_map = {
    "us-west-1": {"sparkml-serving": "746614075791", "scikit-learn": "746614075791"},
    "us-west-2": {"sparkml-serving": "246618743249", "scikit-learn": "246618743249"},
    "us-east-1": {"sparkml-serving": "683313688378", "scikit-learn": "683313688378"},
    "us-east-2": {"sparkml-serving": "257758044811", "scikit-learn": "257758044811"},
    "ap-northeast-1": {"sparkml-serving": "354813040037", "scikit-learn": "354813040037"},
    "ap-northeast-2": {"sparkml-serving": "366743142698", "scikit-learn": "366743142698"},
    "ap-southeast-1": {"sparkml-serving": "121021644041", "scikit-learn": "121021644041"},
    "ap-southeast-2": {"sparkml-serving": "783357654285", "scikit-learn": "783357654285"},
    "ap-south-1": {"sparkml-serving": "720646828776", "scikit-learn": "720646828776"},
    "eu-west-1": {"sparkml-serving": "141502667606", "scikit-learn": "141502667606"},
    "eu-west-2": {"sparkml-serving": "764974769150", "scikit-learn": "764974769150"},
    "eu-central-1": {"sparkml-serving": "492215442770", "scikit-learn": "492215442770"},
    "ca-central-1": {"sparkml-serving": "341280168497", "scikit-learn": "341280168497"},
    "us-gov-west-1": {"sparkml-serving": "414596584902", "scikit-learn": "414596584902"},
    "us-iso-east-1": {"sparkml-serving": "833128469047", "scikit-learn": "833128469047"},
}


def registry(region_name, framework=None):
    """
    Return docker registry for the given AWS region for the given framework.
    This is only used for SparkML and Scikit-learn for now.
    """
    try:
        account_id = image_registry_map[region_name][framework]
        return get_ecr_image_uri_prefix(account_id, region_name)
    except KeyError:
        logging.error("The specific image or region does not exist")
        raise


def default_framework_uri(framework, region_name, image_tag):
    repository_name = "sagemaker-{}".format(framework)
    account_name = registry(region_name, framework)
    return "{}/{}:{}".format(account_name, repository_name, image_tag)
