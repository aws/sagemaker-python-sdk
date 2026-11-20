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
"""LocalContainerMode.create_server must honor the requested container timeout."""

from __future__ import absolute_import

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

import sagemaker.serve.mode.local_container_mode as local_container_mode
from sagemaker.serve.mode.local_container_mode import LocalContainerMode
from sagemaker.serve.utils.exceptions import LocalDeepPingException
from sagemaker.serve.utils.types import ModelServer


@pytest.mark.parametrize("timeout_seconds", [60, 1800])
def test_create_server_gives_up_after_requested_timeout(timeout_seconds):
    clock = {"now": datetime(2026, 1, 1)}

    class FakeDatetime:
        @staticmethod
        def now():
            return clock["now"]

    def fake_sleep(seconds):
        clock["now"] += timedelta(seconds=seconds)

    mode = LocalContainerMode(
        ModelServer.TORCHSERVE, None, None, MagicMock(), model_path="/tmp/model"
    )
    container = MagicMock()

    with (
        patch.object(mode, "_pull_image"),
        patch.object(
            mode,
            "_start_torch_serve",
            side_effect=lambda **_: setattr(mode, "container", container),
        ),
        patch.object(local_container_mode, "datetime", FakeDatetime),
        patch.object(local_container_mode.time, "sleep", side_effect=fake_sleep),
        patch.object(local_container_mode, "pull_logs"),
    ):
        start = clock["now"]
        with pytest.raises(LocalDeepPingException):
            mode.create_server(
                image="image",
                container_timeout_seconds=timeout_seconds,
                secret_key="",
                container_config={},
                ping_fn=lambda: (False, None),
            )

    waited = (clock["now"] - start).total_seconds()
    # 10s initial grace period plus at most one 10s polling interval past the deadline.
    assert timeout_seconds <= waited <= timeout_seconds + 30
