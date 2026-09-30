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

import json
import os
import logging
import re
import shutil
import subprocess
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


_EVAL_LINE = 'eval "$CMD"'
# Deliberately tolerant of how the heredoc delimiter is written, so that a regression which
# keeps the structure but drops the quotes around the delimiter (re-enabling expansion) is
# caught by the semantic assertions below rather than by a parse error in this helper.
_HEREDOC_OPEN_RE = re.compile(r"^CMD=\$\(cat <<'?(?P<delimiter>\w+)'?$")


def _command_block_bounds(lines):
    """Return (heredoc-open index, body slice, index just past ``eval``)."""
    for index, line in enumerate(lines):
        match = _HEREDOC_OPEN_RE.match(line)
        if match:
            delimiter = match.group("delimiter")
            close = lines.index(delimiter, index + 1)
            end = lines.index(_EVAL_LINE, close) + 1
            return index, slice(index + 1, close), end
    raise AssertionError(
        "no 'CMD=$(cat <<DELIMITER' command block in generated script:\n" + "\n".join(lines)
    )


# A stub "command" that reports the argv it actually received, so a test can assert on real
# shell semantics rather than on the text of the generated script.
_CAPTURE_ARGV_PRELUDE = """
capture_args() {
    python3 -c 'import sys, json; print("ARGV=" + json.dumps(sys.argv[1:]))' "$@"
}
"""


def _base_command(script):
    """Return the command carried inside the generated heredoc."""
    lines = script.splitlines()
    _, body, _ = _command_block_bounds(lines)
    return "\n".join(lines[body])


def _argv_from_running_base_command(script):
    """Execute the generated command block under bash and return the argv it produced.

    This is the assertion that actually pins the quoting contract: the SDK writes a shell
    script, so only running it proves that ``args`` arrive as distinct literal arguments.
    """
    lines = script.splitlines()
    start, _, end = _command_block_bounds(lines)
    fragment = "\n".join(lines[start:end])
    completed = subprocess.run(
        ["bash", "-c", _CAPTURE_ARGV_PRELUDE + fragment],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("ARGV="):
            return json.loads(line[len("ARGV=") :])
    raise AssertionError(
        "command block produced no ARGV line.\nstdout={}\nstderr={}".format(
            completed.stdout, completed.stderr
        )
    )


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
    assert (
        _base_command(script) == "python launcher.py -e train.py --epochs 25 --learning_rate 0.001"
    )


def test_prepare_train_script_appends_numeric_args():
    """Numeric ``args`` are stringified and appended to the executed command."""
    source_code = SourceCode(
        source_dir="scripts",
        command="python launcher.py",
        args=["--epochs", 25, "--learning_rate", 0.001],
    )
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)
    assert _base_command(script) == "python launcher.py --epochs 25 --learning_rate 0.001"


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
    assert _base_command(script) == "python launcher.py --prompt 'hello world'"


def test_prepare_train_script_command_only_no_regression():
    """No ``args`` -> command runs verbatim, unchanged from prior behavior."""
    source_code = SourceCode(source_dir="scripts", command="python launcher.py -e train.py")
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)
    assert _base_command(script) == "python launcher.py -e train.py"


requires_bash = pytest.mark.skipif(
    shutil.which("bash") is None, reason="needs bash to execute the generated command block"
)


@requires_bash
@pytest.mark.parametrize(
    "value",
    [
        "hello world",
        "$HOME",
        "${HOME}",
        "$(id -u)",
        "`id -u`",
        'a"b',
        "it's",
        "; echo INJECTED",
        "&& echo INJECTED",
        "| cat",
        "*",
        "a\tb",
        "a\nb",
        "back\\slash",
        "",
    ],
)
def test_args_reach_the_command_as_literal_single_arguments(value):
    """Each arg must arrive verbatim as one argument, whatever shell metacharacters it holds.

    Regression test for the quoting bug in the original implementation: ``shlex.quote`` was
    defeated because the generated script assigned the command inside a double-quoted
    ``CMD="..."``, so the shell expanded ``$VAR``/``$(...)``/backticks at assignment time and
    an embedded double quote terminated the string early. Asserting on the text of the script
    could not catch that -- only running it can.
    """
    source_code = SourceCode(source_dir="scripts", command="capture_args", args=["--prompt", value])
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)

    assert _argv_from_running_base_command(script) == ["--prompt", value]


@requires_bash
def test_multiple_args_preserve_order_and_boundaries():
    """Adjacent args stay distinct even when each contains spaces."""
    values = ["--a", "one two", "--b", "three  four", "--c", "5"]
    source_code = SourceCode(source_dir="scripts", command="capture_args", args=values)
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)

    assert _argv_from_running_base_command(script) == values


@requires_bash
def test_numeric_args_reach_the_command_as_strings():
    """int/float args are stringified without gaining quotes or losing precision."""
    source_code = SourceCode(
        source_dir="scripts",
        command="capture_args",
        args=["--epochs", 25, "--learning_rate", 0.001],
    )
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)

    assert _argv_from_running_base_command(script) == ["--epochs", "25", "--learning_rate", "0.001"]


@requires_bash
def test_environment_variables_in_command_still_expand():
    """``command`` itself is still shell-evaluated, so env vars in it keep working.

    The heredoc stops expansion at *assignment* time only; ``eval`` still parses the command
    once, which is the behavior users of ``command`` rely on.
    """
    source_code = SourceCode(source_dir="scripts", command="capture_args $HOME")
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)

    argv = _argv_from_running_base_command(script)
    assert argv == [os.environ["HOME"]]


@requires_bash
def test_shell_operators_in_command_still_work():
    """``&&`` in a user-supplied ``command`` still chains, unchanged from prior behavior."""
    source_code = SourceCode(source_dir="scripts", command="capture_args one && capture_args two")
    trainer = _make_trainer(source_code)
    script = _written_train_script(trainer, source_code)

    lines = script.splitlines()
    start, _, end = _command_block_bounds(lines)
    completed = subprocess.run(
        ["bash", "-c", _CAPTURE_ARGV_PRELUDE + "\n".join(lines[start:end])],
        capture_output=True,
        text=True,
        check=True,
    )
    emitted = [
        json.loads(line[len("ARGV=") :])
        for line in completed.stdout.splitlines()
        if line.startswith("ARGV=")
    ]
    assert emitted == [["one"], ["two"]]


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
