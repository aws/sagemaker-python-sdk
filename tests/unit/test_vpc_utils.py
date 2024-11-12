# -*- coding: utf-8 -*-

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

from sagemaker.vpc_utils import SUBNETS_KEY, SECURITY_GROUP_IDS_KEY, to_dict, from_dict, sanitize

subnets = ["subnet"]
security_groups = ["sg"]
good_vpc_config = {SUBNETS_KEY: subnets, SECURITY_GROUP_IDS_KEY: security_groups}
foo_vpc_config = {SUBNETS_KEY: subnets, SECURITY_GROUP_IDS_KEY: security_groups, "foo": 1}


def test_to_dict():
    assert to_dict(None, None) is None
    assert to_dict(subnets, None) is None
    assert to_dict(None, security_groups) is None

    assert to_dict(subnets, security_groups) == {
        SUBNETS_KEY: subnets,
        SECURITY_GROUP_IDS_KEY: security_groups,
    }


def test_from_dict():
    assert from_dict(good_vpc_config) == (subnets, security_groups)
    assert from_dict(foo_vpc_config) == (subnets, security_groups)

    assert from_dict(None) == (None, None)
    assert from_dict(None, do_sanitize=True) == (None, None)

    with pytest.raises(KeyError):
        from_dict({})
    with pytest.raises(KeyError):
        from_dict({SUBNETS_KEY: subnets})
    with pytest.raises(KeyError):
        from_dict({SECURITY_GROUP_IDS_KEY: security_groups})

    with pytest.raises(ValueError):
        from_dict({}, do_sanitize=True)


def test_sanitize():
    assert sanitize(good_vpc_config) == good_vpc_config
    assert sanitize(foo_vpc_config) == good_vpc_config

    assert sanitize(None) is None

    with pytest.raises(ValueError):
        sanitize([])
    with pytest.raises(ValueError):
        sanitize({})

    with pytest.raises(ValueError):
        sanitize({SUBNETS_KEY: 1})
    with pytest.raises(ValueError):
        sanitize({SUBNETS_KEY: []})

    with pytest.raises(ValueError):
        sanitize({SECURITY_GROUP_IDS_KEY: 1, SUBNETS_KEY: subnets})
    with pytest.raises(ValueError):
        sanitize({SECURITY_GROUP_IDS_KEY: [], SUBNETS_KEY: subnets})
