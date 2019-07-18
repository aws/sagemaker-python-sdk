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

import pytest

from sagemaker.fw_registry import registry, default_framework_uri
from sagemaker.sklearn import SKLearn


scikit_learn_framework_name = SKLearn.__framework_name__


def test_registry_sparkml_serving():
    assert (
        registry("us-west-1", "sparkml-serving") == "746614075791.dkr.ecr.us-west-1.amazonaws.com"
    )
    assert (
        registry("us-west-2", "sparkml-serving") == "246618743249.dkr.ecr.us-west-2.amazonaws.com"
    )
    assert (
        registry("us-east-1", "sparkml-serving") == "683313688378.dkr.ecr.us-east-1.amazonaws.com"
    )
    assert (
        registry("us-east-2", "sparkml-serving") == "257758044811.dkr.ecr.us-east-2.amazonaws.com"
    )
    assert (
        registry("ap-northeast-1", "sparkml-serving")
        == "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com"
    )
    assert (
        registry("ap-northeast-2", "sparkml-serving")
        == "366743142698.dkr.ecr.ap-northeast-2.amazonaws.com"
    )
    assert (
        registry("ap-southeast-1", "sparkml-serving")
        == "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com"
    )
    assert (
        registry("ap-southeast-2", "sparkml-serving")
        == "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com"
    )
    assert (
        registry("ap-south-1", "sparkml-serving") == "720646828776.dkr.ecr.ap-south-1.amazonaws.com"
    )
    assert (
        registry("eu-west-1", "sparkml-serving") == "141502667606.dkr.ecr.eu-west-1.amazonaws.com"
    )
    assert (
        registry("eu-west-2", "sparkml-serving") == "764974769150.dkr.ecr.eu-west-2.amazonaws.com"
    )
    assert (
        registry("eu-central-1", "sparkml-serving")
        == "492215442770.dkr.ecr.eu-central-1.amazonaws.com"
    )
    assert (
        registry("ca-central-1", "sparkml-serving")
        == "341280168497.dkr.ecr.ca-central-1.amazonaws.com"
    )
    assert (
        registry("us-gov-west-1", "sparkml-serving")
        == "414596584902.dkr.ecr.us-gov-west-1.amazonaws.com"
    )
    assert (
        registry("us-iso-east-1", "sparkml-serving")
        == "833128469047.dkr.ecr.us-iso-east-1.c2s.ic.gov"
    )


def test_registry_sklearn():
    assert (
        registry("us-west-1", scikit_learn_framework_name)
        == "746614075791.dkr.ecr.us-west-1.amazonaws.com"
    )
    assert (
        registry("us-west-2", scikit_learn_framework_name)
        == "246618743249.dkr.ecr.us-west-2.amazonaws.com"
    )
    assert (
        registry("us-east-1", scikit_learn_framework_name)
        == "683313688378.dkr.ecr.us-east-1.amazonaws.com"
    )
    assert (
        registry("us-east-2", scikit_learn_framework_name)
        == "257758044811.dkr.ecr.us-east-2.amazonaws.com"
    )
    assert (
        registry("ap-northeast-1", scikit_learn_framework_name)
        == "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com"
    )
    assert (
        registry("ap-northeast-2", scikit_learn_framework_name)
        == "366743142698.dkr.ecr.ap-northeast-2.amazonaws.com"
    )
    assert (
        registry("ap-southeast-1", scikit_learn_framework_name)
        == "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com"
    )
    assert (
        registry("ap-southeast-2", scikit_learn_framework_name)
        == "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com"
    )
    assert (
        registry("ap-south-1", scikit_learn_framework_name)
        == "720646828776.dkr.ecr.ap-south-1.amazonaws.com"
    )
    assert (
        registry("eu-west-1", scikit_learn_framework_name)
        == "141502667606.dkr.ecr.eu-west-1.amazonaws.com"
    )
    assert (
        registry("eu-west-2", scikit_learn_framework_name)
        == "764974769150.dkr.ecr.eu-west-2.amazonaws.com"
    )
    assert (
        registry("eu-central-1", scikit_learn_framework_name)
        == "492215442770.dkr.ecr.eu-central-1.amazonaws.com"
    )
    assert (
        registry("ca-central-1", scikit_learn_framework_name)
        == "341280168497.dkr.ecr.ca-central-1.amazonaws.com"
    )
    assert (
        registry("us-gov-west-1", scikit_learn_framework_name)
        == "414596584902.dkr.ecr.us-gov-west-1.amazonaws.com"
    )
    assert (
        registry("us-iso-east-1", scikit_learn_framework_name)
        == "833128469047.dkr.ecr.us-iso-east-1.c2s.ic.gov"
    )


def test_default_sklearn_image_uri():
    image_tag = "0.20.0-cpu-py3"
    sklearn_image_uri = default_framework_uri(scikit_learn_framework_name, "us-west-1", image_tag)
    assert (
        sklearn_image_uri
        == "746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3"
    )


def test_framework_invalid():
    with pytest.raises(KeyError):
        registry("us-west-2", "dummy-value")


def test_framework_none():
    with pytest.raises(KeyError):
        registry("us-west-2", None)


def test_region_invalid():
    with pytest.raises(KeyError):
        registry("us-west-5", "scikit-learn")


def test_region_none():
    with pytest.raises(KeyError):
        registry(None, "scikit-learn")
