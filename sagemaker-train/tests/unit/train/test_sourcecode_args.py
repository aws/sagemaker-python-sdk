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
"""Tests for the SourceCode ``args`` field (issue #5226).

When ``command`` is used in ``SourceCode`` the SDK runs it verbatim and does NOT
append top-level ``hyperparameters`` as CLI arguments. ``args`` lets the user pass
positional/optional arguments that are appended to ``command`` deterministically,
and a warning points users to the ``SM_HPS`` env var when they combine
``hyperparameters`` with ``command``/``args``.
"""

from __future__ import absolute_import

import os
import logging
from tempfile import TemporaryDirectory

import pytest
from unittest.mock import patch, MagicMock

from sagemaker.core.helper.session_helper import Session
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import Compute, StoppingCondition, OutputDataConfig, SourceCode
from sagemaker.train.constants import TRAIN_SCRIPT

DEFAULT_BASE_NAME = "dummy-image-job"
DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_BUCKET_PREFIX = "sample-prefix"
DEFAULT_REGION = "us-west-2"
DEFAULT_COMPUTE_CONFIG = Compute(instance_type="ml.m5.xlarge", instance_count=1)
DEFAULT_OUTPUT_DATA_CONFIG = OutputDataConfig(
    s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/{DEFAULT_BASE_NAME}",
    compression_type="GZIP",
    kms_key_id=None,
)
DEFAULT_STOPPING_CONDITION = StoppingCondition(
    max_runtime_in_seconds=3600,
    max_pending_time_in_seconds=None,
    max_wait_time_in_seconds=None,
)


@pytest.fixture(autouse=True)
def modules_session():
    with (
        patch("sagemaker.train.Session", spec=Session) as session_mock,
        patch("sagemaker.train.defaults.resolve_and_validate_role", return_value=DEFAULT_ROLE),
    ):
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = DEFAULT_BUCKET
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        session_instance.default_bucket_prefix = DEFAULT_BUCKET_PREFIX
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        session_instance.boto_region_name = DEFAULT_REGION
        yield session_instance


def _make_trainer(source_code, hyperparameters=None):
    return ModelTrainer(
        training_image=DEFAULT_IMAGE,
        role=DEFAULT_ROLE,
        compute=DEFAULT_COMPUTE_CONFIG,
        stopping_condition=DEFAULT_STOPPING_CONDITION,
        output_data_config=DEFAULT_OUTPUT_DATA_CONFIG,
        source_code=source_code,
        hyperparameters=hyperparameters or {},
    )


def _written_train_script(trainer, source_code):
    with TemporaryDirectory() as tmp_name:
        tmp_dir = MagicMock()
        tmp_dir.name = tmp_name
        trainer._prepare_train_script(tmp_dir=tmp_dir, source_code=source_code)
        with open(os.path.join(tmp_name, TRAIN_SCRIPT), "r") as f:
            return f.read()


def test_sourcecode_args_field_defaults_to_none():
    """The new ``args`` field exists and defaults to None (backward compatible)."""
    assert SourceCode(command="python train.py").args is None


def test_sourcecode_args_field_synced_in_modules_configs():
    """``args`` must exist in the legacy ``modules.configs`` mirror as well."""
    from sagemaker.core.modules.configs import SourceCode as ModulesSourceCode

    assert ModulesSourceCode(command="python train.py").args is None
    assert ModulesSourceCode(command="python train.py", args=["--epochs", "5"]).args == [
        "--epochs",
        "5",
    ]


def test_sourcecode_args_accepts_numeric_values():
    """``args`` accepts int/float values (matching the maintainer's proposed API)."""
    # Both SourceCode definitions must accept the issue-thread example without raising.
    from sagemaker.core.modules.configs import SourceCode as ModulesSourceCode

    for cls in (SourceCode, ModulesSourceCode):
        source_code = cls(command="python train.py", args=["--epochs", 25, "--lr", 0.001])
        assert source_code.args == ["--epochs", 25, "--lr", 0.001]


def test_prepare_train_script_appends_args_to_command():
    """``args`` are appended to ``command`` in the executed base command."""
    source_code = SourceCode(
        source_dir="scripts",
        command="python launcher.py -e train.py",
        args=["--epochs", "25", "--learning_rate", "0.001"],
    )
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)
    assert 'CMD="python launcher.py -e train.py --epochs 25 --learning_rate 0.001"' in script


def test_prepare_train_script_appends_numeric_args():
    """Numeric ``args`` are stringified and appended to the executed command."""
    source_code = SourceCode(
        source_dir="scripts",
        command="python launcher.py",
        args=["--epochs", 25, "--learning_rate", 0.001],
    )
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)
    assert 'CMD="python launcher.py --epochs 25 --learning_rate 0.001"' in script


def test_prepare_train_script_shell_quotes_args_with_spaces():
    """Args with spaces/special chars are shell-quoted so they stay single arguments."""
    source_code = SourceCode(
        source_dir="scripts",
        command="python launcher.py",
        args=["--prompt", "hello world"],
    )
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)
    # shlex.quote wraps the value so the space does not re-tokenize the argument.
    assert "'hello world'" in script
    assert "CMD=\"python launcher.py --prompt 'hello world'\"" in script


def test_prepare_train_script_command_only_no_regression():
    """No ``args`` -> command runs verbatim, unchanged from prior behavior."""
    source_code = SourceCode(source_dir="scripts", command="python launcher.py -e train.py")
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)
    assert 'CMD="python launcher.py -e train.py"' in script


def test_warning_when_hyperparameters_combined_with_command(caplog):
    """A warning pointing to SM_HPS fires when hyperparameters + command are combined."""
    source_code = SourceCode(source_dir="scripts", command="python launcher.py -e train.py")
    trainer = _make_trainer(source_code, hyperparameters={"epochs": 25})
    with caplog.at_level(logging.WARNING):
        _written_train_script(trainer, source_code)
    assert any("SM_HPS" in rec.message for rec in caplog.records)


def test_no_warning_when_command_without_hyperparameters(caplog):
    """No SM_HPS warning when command is used but no hyperparameters are set."""
    source_code = SourceCode(source_dir="scripts", command="python launcher.py -e train.py")
    trainer = _make_trainer(source_code)
    with caplog.at_level(logging.WARNING):
        _written_train_script(trainer, source_code)
    assert not any("SM_HPS" in rec.message for rec in caplog.records)
