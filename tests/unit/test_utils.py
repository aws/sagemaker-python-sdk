# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.utils import get_config_value


def test_get_config_value():

    config = {
        'local': {
            'region_name': 'us-west-2',
            'port': '123'
        },
        'other': {
            'key': 1
        }
    }

    assert get_config_value('local.region_name', config) == 'us-west-2'
    assert get_config_value('local', config) == {'region_name': 'us-west-2', 'port': '123'}

    assert get_config_value('does_not.exist', config) is None
    assert get_config_value('other.key', None) is None
