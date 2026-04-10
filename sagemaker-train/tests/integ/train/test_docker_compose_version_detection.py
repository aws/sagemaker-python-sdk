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
"""Integration tests for Docker Compose version detection fix (issue #5739).

These tests verify that _get_compose_cmd_prefix correctly accepts Docker Compose
versions >= 2 (including v3, v4, v5, etc.) rather than only accepting v2.

The tests run against the real Docker Compose installation on the machine — no mocking.
Requires: Docker with Compose plugin installed (any version >= 2).
"""
from __future__ import absolute_import

import re
import subprocess
import tempfile

import pytest

from sagemaker.core.local.image import _SageMakerContainer
from sagemaker.core.modules.local_core.local_container import (
    _LocalContainer as CoreModulesLocalContainer,
)
from sagemaker.core.shapes import Channel, DataSource, S3DataSource
from sagemaker.train.local.local_container import (
    _LocalContainer as TrainLocalContainer,
)


def _get_installed_compose_major_version():
    """Return the major version int of the installed Docker Compose, or None."""
    try:
        output = subprocess.check_output(
            ["docker", "compose", "version"],
            stderr=subprocess.DEVNULL,
            encoding="UTF-8",
        )
        match = re.search(r"v(\d+)", output.strip())
        if match:
            return int(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


# Skip the entire module if Docker Compose >= 2 is not available
_compose_major = _get_installed_compose_major_version()
pytestmark = pytest.mark.skipif(
    _compose_major is None or _compose_major < 2,
    reason=f"Docker Compose >= 2 required (found: v{_compose_major})",
)


def _make_basic_channel():
    """Create a minimal Channel for constructing _LocalContainer instances."""
    data_source = DataSource(
        s3_data_source=S3DataSource(
            s3_uri="s3://bucket/data",
            s3_data_type="S3Prefix",
            s3_data_distribution_type="FullyReplicated",
        )
    )
    return Channel(channel_name="training", data_source=data_source)


def _make_local_container(container_cls):
    """Construct a _LocalContainer with minimal valid args.

    sagemaker_session is None since _get_compose_cmd_prefix doesn't use it,
    and the Pydantic model rejects Mock objects.
    """
    container_root = tempfile.mkdtemp(prefix="sagemaker-integ-compose-")
    return container_cls(
        training_job_name="integ-test-compose-detection",
        instance_type="local",
        instance_count=1,
        image="test-image:latest",
        container_root=container_root,
        input_data_config=[_make_basic_channel()],
        environment={},
        hyper_parameters={},
        container_entrypoint=[],
        container_arguments=[],
        sagemaker_session=None,
    )


@pytest.fixture
def _core_modules_container():
    return _make_local_container(CoreModulesLocalContainer)


@pytest.fixture
def _train_container():
    return _make_local_container(TrainLocalContainer)


class TestDockerComposeVersionDetection:
    """Integration tests for _get_compose_cmd_prefix across all three code locations.

    Validates the fix for https://github.com/aws/sagemaker-python-sdk/issues/5739
    where Docker Compose v3+ was incorrectly rejected.
    """

    def test_sagemaker_core_image_accepts_installed_compose(self):
        """sagemaker-core local/image.py _SageMakerContainer._get_compose_cmd_prefix
        should accept the installed Docker Compose version."""
        result = _SageMakerContainer._get_compose_cmd_prefix()

        assert result == ["docker", "compose"], (
            f"Expected ['docker', 'compose'] but got {result}. "
            f"Installed Docker Compose is v{_compose_major}."
        )

    def test_sagemaker_core_modules_local_container_accepts_installed_compose(
        self, _core_modules_container
    ):
        """sagemaker-core modules/local_core/local_container.py
        _LocalContainer._get_compose_cmd_prefix should accept the installed version."""
        result = _core_modules_container._get_compose_cmd_prefix()

        assert result == ["docker", "compose"], (
            f"Expected ['docker', 'compose'] but got {result}. "
            f"Installed Docker Compose is v{_compose_major}."
        )

    def test_sagemaker_train_local_container_accepts_installed_compose(
        self, _train_container
    ):
        """sagemaker-train local/local_container.py
        _LocalContainer._get_compose_cmd_prefix should accept the installed version."""
        result = _train_container._get_compose_cmd_prefix()

        assert result == ["docker", "compose"], (
            f"Expected ['docker', 'compose'] but got {result}. "
            f"Installed Docker Compose is v{_compose_major}."
        )

    def test_returned_command_is_functional(self):
        """The command returned by _get_compose_cmd_prefix should actually work."""
        cmd = _SageMakerContainer._get_compose_cmd_prefix()

        # Run the returned command with "version" to prove it's functional
        result = subprocess.run(
            cmd + ["version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"Command {cmd + ['version']} failed: {result.stderr}"
        )
        assert "version" in result.stdout.lower(), (
            f"Unexpected output from {cmd + ['version']}: {result.stdout}"
        )

    @pytest.mark.skipif(
        _compose_major is not None and _compose_major < 3,
        reason="This test specifically validates v3+ acceptance (installed is v2)",
    )
    def test_v3_plus_specifically_accepted(self):
        """When Docker Compose v3+ is installed, it must be accepted — not rejected.

        This is the core regression test for issue #5739.
        """
        result = _SageMakerContainer._get_compose_cmd_prefix()
        assert result == ["docker", "compose"], (
            f"Docker Compose v{_compose_major} was rejected. "
            "This is the exact bug described in issue #5739."
        )
