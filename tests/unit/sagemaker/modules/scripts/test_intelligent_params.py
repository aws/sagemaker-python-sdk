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
"""Intelligent Parameter Tests."""
from __future__ import absolute_import

import os
import pytest
from sagemaker.modules.scripts.intelligent_params import (
    rewrite_file,
    rewrite_line,
)


@pytest.fixture()
def hyperparameters():
    return {"n_estimators": 3, "epochs": 4, "state": "In Progress", "list_value": [1, 2, 3]}


@pytest.fixture()
def python_code():
    return """
import sagemaker

def main(args):
    n_estimators = 10 # sm_hp_n_estimators
    state = "Not Started" # sm_hyper_param
    random_seed = 0.1 # sm_hyper_param

    output_dir = "local/dir/" # sm_model_dir
    epochs = 5 # sm_hp_epochs
    input_data = [0, 0, 0] # sm_hp_list_value

    # Load the Iris dataset
    iris = load_iris()
    y = iris.target

    # Make predictions on the test set
    y_pred = clf.predict(input_data)

    accuracy = accuracy_score(y, y_pred) # calculate the accuracy
    print(f"# Model accuracy: {accuracy:.2f}")
"""


@pytest.fixture()
def expected_output_code():
    return """
import sagemaker

def main(args):
    n_estimators = 3 # set by intelligent parameters
    state = "In Progress" # set by intelligent parameters
    random_seed = 0.1 # sm_hyper_param

    output_dir = "/opt/ml/input" # set by intelligent parameters
    epochs = 4 # set by intelligent parameters
    input_data = [1, 2, 3] # set by intelligent parameters

    # Load the Iris dataset
    iris = load_iris()
    y = iris.target

    # Make predictions on the test set
    y_pred = clf.predict(input_data)

    accuracy = accuracy_score(y, y_pred) # calculate the accuracy
    print(f"# Model accuracy: {accuracy:.2f}")
"""


def test_rewrite_line(hyperparameters):
    line = "n_estimators = 4 # sm_hyper_param"
    new_line = rewrite_line(line, hyperparameters)
    assert new_line == "n_estimators = 3 # set by intelligent parameters\n"

    line = "    epochs = 5 # sm_hp_epochs"
    new_line = rewrite_line(line, hyperparameters)
    assert new_line == "    epochs = 4 # set by intelligent parameters\n"

    os.environ["SM_MODEL_DIR"] = "/opt/ml/input"
    line = 'output_dir = "local/dir/" # sm_model_dir    '
    new_line = rewrite_line(line, hyperparameters)
    assert new_line == 'output_dir = "/opt/ml/input" # set by intelligent parameters\n'

    line = "    random_state = 1 # not an intelligent parameter comment    \n"
    new_line = rewrite_line(line, hyperparameters)
    assert new_line == "    random_state = 1 # not an intelligent parameter comment    \n"

    line = "not_an_intelligent_parameter = 4 # sm_hyper_param\n"
    new_line = rewrite_line(line, hyperparameters)
    assert new_line == "not_an_intelligent_parameter = 4 # sm_hyper_param\n"

    line = "not_found_in_hyper_params = 4 # sm_hp_not_found_in_hyper_params\n"
    new_line = rewrite_line(line, hyperparameters)
    assert new_line == "not_found_in_hyper_params = 4 # sm_hp_not_found_in_hyper_params\n"

    line = "list_value = [4, 5, 6] # sm_hyper_param"
    new_line = rewrite_line(line, hyperparameters)
    assert new_line == "list_value = [1, 2, 3] # set by intelligent parameters\n"


def test_rewrite_file(hyperparameters, python_code, expected_output_code):
    test_file_path = "temp_test.py"

    os.environ["SM_MODEL_DIR"] = "/opt/ml/input"
    with open(test_file_path, "w") as f:
        f.write(python_code)
    rewrite_file(test_file_path, hyperparameters)

    with open(test_file_path, "r") as f:
        new_python_code = f.read()
    assert new_python_code == expected_output_code

    os.remove(test_file_path)
