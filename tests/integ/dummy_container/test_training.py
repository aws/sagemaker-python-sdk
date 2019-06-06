# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import json
import pytest


@pytest.fixture()
def assertions():
    with open('/assert.json') as f:
        return json.load(f)


def test_hyperparameters(assertions):
    if 'expected_hyperparameters' in assertions:
        assert read_config_file('hyperparameters.json') == assertions["expected_hyperparameters"]


def test_inputdataconfig(assertions):
    if 'expected_inputdataconfig' in assertions:
        assert read_config_file('inputdataconfig.json') == assertions["expected_inputdataconfig"]


def test_expected_hosts(assertions):
    if 'expected_hosts' in assertions:
        config = read_config_file('resourceconfig.json')
        assert len(config['hosts']) == assertions['expected_hosts']


def read_config_file(name):
    with open('/opt/ml/input/config/%s' % name, 'r') as f:
        return json.load(f)
