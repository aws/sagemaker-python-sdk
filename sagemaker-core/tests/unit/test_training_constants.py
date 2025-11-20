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
"""Unit tests for sagemaker.core.training.constants module."""
from __future__ import absolute_import

import os
from sagemaker.core.training import constants


class TestTrainingConstants:
    """Test training constants module."""

    def test_default_instance_type(self):
        """Test DEFAULT_INSTANCE_TYPE constant."""
        assert constants.DEFAULT_INSTANCE_TYPE == "ml.m5.xlarge"

    def test_sm_code_constant(self):
        """Test SM_CODE constant."""
        assert constants.SM_CODE == "code"

    def test_sm_code_container_path(self):
        """Test SM_CODE_CONTAINER_PATH constant."""
        assert constants.SM_CODE_CONTAINER_PATH == "/opt/ml/input/data/code"

    def test_sm_drivers_constant(self):
        """Test SM_DRIVERS constant."""
        assert constants.SM_DRIVERS == "sm_drivers"

    def test_sm_drivers_container_path(self):
        """Test SM_DRIVERS_CONTAINER_PATH constant."""
        assert constants.SM_DRIVERS_CONTAINER_PATH == "/opt/ml/input/data/sm_drivers"

    def test_sm_drivers_local_path(self):
        """Test SM_DRIVERS_LOCAL_PATH constant."""
        assert isinstance(constants.SM_DRIVERS_LOCAL_PATH, str)
        assert "container_drivers" in constants.SM_DRIVERS_LOCAL_PATH
        assert os.path.isabs(constants.SM_DRIVERS_LOCAL_PATH)

    def test_source_code_json(self):
        """Test SOURCE_CODE_JSON constant."""
        assert constants.SOURCE_CODE_JSON == "sourcecode.json"

    def test_distributed_json(self):
        """Test DISTRIBUTED_JSON constant."""
        assert constants.DISTRIBUTED_JSON == "distributed.json"

    def test_train_script(self):
        """Test TRAIN_SCRIPT constant."""
        assert constants.TRAIN_SCRIPT == "sm_train.sh"

    def test_default_container_entrypoint(self):
        """Test DEFAULT_CONTAINER_ENTRYPOINT constant."""
        assert constants.DEFAULT_CONTAINER_ENTRYPOINT == ["/bin/bash"]
        assert isinstance(constants.DEFAULT_CONTAINER_ENTRYPOINT, list)

    def test_default_container_arguments(self):
        """Test DEFAULT_CONTAINER_ARGUMENTS constant."""
        assert isinstance(constants.DEFAULT_CONTAINER_ARGUMENTS, list)
        assert len(constants.DEFAULT_CONTAINER_ARGUMENTS) == 2
        assert constants.DEFAULT_CONTAINER_ARGUMENTS[0] == "-c"
        assert "chmod +x" in constants.DEFAULT_CONTAINER_ARGUMENTS[1]
        assert "sm_train.sh" in constants.DEFAULT_CONTAINER_ARGUMENTS[1]

    def test_container_arguments_use_correct_paths(self):
        """Test that DEFAULT_CONTAINER_ARGUMENTS uses correct path constants."""
        args_string = constants.DEFAULT_CONTAINER_ARGUMENTS[1]
        assert constants.SM_DRIVERS_CONTAINER_PATH in args_string
        assert constants.TRAIN_SCRIPT in args_string
