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
"""Unit tests for _utils module."""
from __future__ import absolute_import

import pytest
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch, mock_open

from sagemaker.mlops.workflow._utils import (
    FRAMEWORK_VERSION,
    INSTANCE_TYPE,
    REPACK_SCRIPT,
    REPACK_SCRIPT_LAUNCHER,
    LAUNCH_REPACK_SCRIPT_CMD,
    _RepackModelStep,
)


class TestConstants:
    """Tests for module constants."""

    def test_framework_version(self):
        """Test FRAMEWORK_VERSION constant."""
        assert FRAMEWORK_VERSION == "1.2-1"

    def test_instance_type(self):
        """Test INSTANCE_TYPE constant."""
        assert INSTANCE_TYPE == "ml.m5.large"

    def test_repack_script(self):
        """Test REPACK_SCRIPT constant."""
        assert REPACK_SCRIPT == "_repack_model.py"

    def test_repack_script_launcher(self):
        """Test REPACK_SCRIPT_LAUNCHER constant."""
        assert REPACK_SCRIPT_LAUNCHER == "_repack_script_launcher.sh"

    def test_launch_repack_script_cmd(self):
        """Test LAUNCH_REPACK_SCRIPT_CMD contains expected content."""
        assert "#!/bin/bash" in LAUNCH_REPACK_SCRIPT_CMD
        assert "python _repack_model.py" in LAUNCH_REPACK_SCRIPT_CMD
        assert "--inference_script" in LAUNCH_REPACK_SCRIPT_CMD
        assert "--model_archive" in LAUNCH_REPACK_SCRIPT_CMD
        assert "--source_dir" in LAUNCH_REPACK_SCRIPT_CMD


class TestRepackModelStep:
    """Tests for _RepackModelStep class."""

    @pytest.fixture
    def mock_session(self):
        session = Mock()
        session.boto_region_name = "us-west-2"
        return session

    @pytest.fixture
    def temp_entry_point(self):
        """Create a temporary entry point file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test entry point\n")
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Note: Full integration test of _RepackModelStep.__init__ is complex due to
    # TrainingStep validation. The other tests cover the key functionality.

    def test_init_with_display_name_and_description(self, mock_session, temp_entry_point):
        """Test _RepackModelStep initialization with display name and description."""
        with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
            with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                    mock_retrieve.return_value = "sklearn-image:latest"
                    mock_trainer_instance = Mock()
                    mock_trainer_instance.train = Mock(return_value=Mock())
                    mock_trainer.return_value = mock_trainer_instance
                    mock_super.return_value = None

                    step = _RepackModelStep(
                        name="repack-step",
                        sagemaker_session=mock_session,
                        role="arn:aws:iam::123456789012:role/SageMakerRole",
                        model_data="s3://bucket/model.tar.gz",
                        entry_point=temp_entry_point,
                        display_name="Repack Display",
                        description="Repack Description",
                    )

                    # Verify super().__init__ was called with display_name and description
                    mock_super.assert_called_once()
                    call_kwargs = mock_super.call_args[1]
                    assert call_kwargs['display_name'] == "Repack Display"
                    assert call_kwargs['description'] == "Repack Description"

    def test_init_with_source_dir(self, mock_session, temp_entry_point):
        """Test _RepackModelStep initialization with source_dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
                with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                    with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                        mock_retrieve.return_value = "sklearn-image:latest"
                        mock_trainer_instance = Mock()
                        mock_trainer_instance.train = Mock(return_value=Mock())
                        mock_trainer.return_value = mock_trainer_instance
                        mock_super.return_value = None

                        step = _RepackModelStep(
                            name="repack-step",
                            sagemaker_session=mock_session,
                            role="arn:aws:iam::123456789012:role/SageMakerRole",
                            model_data="s3://bucket/model.tar.gz",
                            entry_point=temp_entry_point,
                            source_dir=temp_dir,
                        )

                        assert step._source_dir == temp_dir

    def test_init_with_requirements(self, mock_session, temp_entry_point):
        """Test _RepackModelStep initialization with requirements."""
        with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
            with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                    mock_retrieve.return_value = "sklearn-image:latest"
                    mock_trainer_instance = Mock()
                    mock_trainer_instance.train = Mock(return_value=Mock())
                    mock_trainer.return_value = mock_trainer_instance
                    mock_super.return_value = None

                    step = _RepackModelStep(
                        name="repack-step",
                        sagemaker_session=mock_session,
                        role="arn:aws:iam::123456789012:role/SageMakerRole",
                        model_data="s3://bucket/model.tar.gz",
                        entry_point=temp_entry_point,
                        requirements="requirements.txt",
                    )

                    assert step._requirements == "requirements.txt"

    def test_init_with_networking(self, mock_session, temp_entry_point):
        """Test _RepackModelStep initialization with networking config."""
        with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
            with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                    mock_retrieve.return_value = "sklearn-image:latest"
                    mock_trainer_instance = Mock()
                    mock_trainer_instance.train = Mock(return_value=Mock())
                    mock_trainer.return_value = mock_trainer_instance
                    mock_super.return_value = None

                    step = _RepackModelStep(
                        name="repack-step",
                        sagemaker_session=mock_session,
                        role="arn:aws:iam::123456789012:role/SageMakerRole",
                        model_data="s3://bucket/model.tar.gz",
                        entry_point=temp_entry_point,
                        subnets=["subnet-123"],
                        security_group_ids=["sg-123"],
                    )

                    # Verify ModelTrainer was called with networking config
                    mock_trainer.assert_called_once()
                    call_kwargs = mock_trainer.call_args[1]
                    assert call_kwargs['networking'] is not None

    def test_init_with_custom_instance_type(self, mock_session, temp_entry_point):
        """Test _RepackModelStep initialization with custom instance type."""
        with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
            with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                    mock_retrieve.return_value = "sklearn-image:latest"
                    mock_trainer_instance = Mock()
                    mock_trainer_instance.train = Mock(return_value=Mock())
                    mock_trainer.return_value = mock_trainer_instance
                    mock_super.return_value = None

                    step = _RepackModelStep(
                        name="repack-step",
                        sagemaker_session=mock_session,
                        role="arn:aws:iam::123456789012:role/SageMakerRole",
                        model_data="s3://bucket/model.tar.gz",
                        entry_point=temp_entry_point,
                        instance_type="ml.p3.2xlarge",
                    )

                    # Verify image_uris.retrieve was called with custom instance type
                    mock_retrieve.assert_called_once()
                    call_kwargs = mock_retrieve.call_args[1]
                    assert call_kwargs['instance_type'] == "ml.p3.2xlarge"

    def test_init_with_depends_on(self, mock_session, temp_entry_point):
        """Test _RepackModelStep initialization with depends_on."""
        with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
            with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                    mock_retrieve.return_value = "sklearn-image:latest"
                    mock_trainer_instance = Mock()
                    mock_trainer_instance.train = Mock(return_value=Mock())
                    mock_trainer.return_value = mock_trainer_instance
                    mock_super.return_value = None

                    step = _RepackModelStep(
                        name="repack-step",
                        sagemaker_session=mock_session,
                        role="arn:aws:iam::123456789012:role/SageMakerRole",
                        model_data="s3://bucket/model.tar.gz",
                        entry_point=temp_entry_point,
                        depends_on=["step1", "step2"],
                    )

                    # Verify super().__init__ was called with depends_on
                    mock_super.assert_called_once()
                    call_kwargs = mock_super.call_args[1]
                    assert call_kwargs['depends_on'] == ["step1", "step2"]

    def test_init_with_retry_policies(self, mock_session, temp_entry_point):
        """Test _RepackModelStep initialization with retry_policies."""
        with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
            with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                    mock_retrieve.return_value = "sklearn-image:latest"
                    mock_trainer_instance = Mock()
                    mock_trainer_instance.train = Mock(return_value=Mock())
                    mock_trainer.return_value = mock_trainer_instance
                    mock_super.return_value = None

                    from sagemaker.mlops.workflow.retry import RetryPolicy
                    retry_policy = RetryPolicy(max_attempts=3)

                    step = _RepackModelStep(
                        name="repack-step",
                        sagemaker_session=mock_session,
                        role="arn:aws:iam::123456789012:role/SageMakerRole",
                        model_data="s3://bucket/model.tar.gz",
                        entry_point=temp_entry_point,
                        retry_policies=[retry_policy],
                    )

                    # Verify super().__init__ was called with retry_policies
                    mock_super.assert_called_once()
                    call_kwargs = mock_super.call_args[1]
                    assert call_kwargs['retry_policies'] == [retry_policy]

    def test_establish_source_dir_creates_temp_dir(self, mock_session, temp_entry_point):
        """Test _establish_source_dir creates temporary directory."""
        with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
            with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                    mock_retrieve.return_value = "sklearn-image:latest"
                    mock_trainer_instance = Mock()
                    mock_trainer_instance.train = Mock(return_value=Mock())
                    mock_trainer.return_value = mock_trainer_instance
                    mock_super.return_value = None

                    step = _RepackModelStep(
                        name="repack-step",
                        sagemaker_session=mock_session,
                        role="arn:aws:iam::123456789012:role/SageMakerRole",
                        model_data="s3://bucket/model.tar.gz",
                        entry_point=temp_entry_point,
                    )

                    # When source_dir is None, it should be created
                    assert step._source_dir is not None
                    assert os.path.exists(step._source_dir)

    def test_inject_repack_script_local_source_dir(self, mock_session, temp_entry_point):
        """Test _inject_repack_script_and_launcher with local source_dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
                with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                    with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                        mock_retrieve.return_value = "sklearn-image:latest"
                        mock_trainer_instance = Mock()
                        mock_trainer_instance.train = Mock(return_value=Mock())
                        mock_trainer.return_value = mock_trainer_instance
                        mock_super.return_value = None

                        step = _RepackModelStep(
                            name="repack-step",
                            sagemaker_session=mock_session,
                            role="arn:aws:iam::123456789012:role/SageMakerRole",
                            model_data="s3://bucket/model.tar.gz",
                            entry_point=temp_entry_point,
                            source_dir=temp_dir,
                        )

                        # Verify repack script and launcher were created
                        repack_script_path = os.path.join(temp_dir, REPACK_SCRIPT)
                        launcher_path = os.path.join(temp_dir, REPACK_SCRIPT_LAUNCHER)

                        assert os.path.exists(repack_script_path)
                        assert os.path.exists(launcher_path)

                        # Verify launcher content
                        with open(launcher_path, 'r') as f:
                            content = f.read()
                            assert "#!/bin/bash" in content
                            assert "python _repack_model.py" in content

    def test_properties_returns_parent_properties(self, mock_session, temp_entry_point):
        """Test properties returns parent class properties."""
        with patch('sagemaker.mlops.workflow._utils.image_uris.retrieve') as mock_retrieve:
            with patch('sagemaker.train.ModelTrainer') as mock_trainer:
                with patch('sagemaker.mlops.workflow._utils.TrainingStep.__init__') as mock_super:
                    mock_retrieve.return_value = "sklearn-image:latest"
                    mock_trainer_instance = Mock()
                    mock_trainer_instance.train = Mock(return_value=Mock())
                    mock_trainer.return_value = mock_trainer_instance
                    mock_super.return_value = None

                    step = _RepackModelStep(
                        name="repack-step",
                        sagemaker_session=mock_session,
                        role="arn:aws:iam::123456789012:role/SageMakerRole",
                        model_data="s3://bucket/model.tar.gz",
                        entry_point=temp_entry_point,
                    )

                    # Set _properties to test
                    step._properties = Mock()
                    step._properties.TrainingJobName = "test-job"

                    assert step.properties.TrainingJobName == "test-job"
