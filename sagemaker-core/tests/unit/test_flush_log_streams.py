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
"""_flush_log_streams must page through DescribeLogStreams with nextToken."""

from __future__ import absolute_import

import pytest

from sagemaker.core import common_utils
from sagemaker.core.helper import session_helper
from sagemaker.core.remote_function import job


class _PaginatingLogsClient:
    """Minimal CloudWatch Logs client that pages log streams with nextToken."""

    def __init__(self, stream_count):
        self.stream_names = [f"job/algo-{i}" for i in range(stream_count)]
        self.describe_calls = 0

    def describe_log_streams(
        self, logGroupName, logStreamNamePrefix, orderBy, limit, nextToken=None
    ):
        self.describe_calls += 1
        if self.describe_calls > 10:
            raise AssertionError("describe_log_streams keeps returning the same page")
        start = int(nextToken) if nextToken else 0
        response = {
            "logStreams": [{"logStreamName": s} for s in self.stream_names[start : start + limit]]
        }
        if start + limit < len(self.stream_names):
            response["nextToken"] = str(start + limit)
        return response

    def get_log_events(self, **_kwargs):
        return {"events": [], "nextForwardToken": "end"}


@pytest.mark.parametrize("module", [common_utils, session_helper, job])
def test_flush_log_streams_pages_through_all_log_streams(module):
    client = _PaginatingLogsClient(stream_count=60)
    positions = {}

    module._flush_log_streams(
        [], 60, client, "/aws/sagemaker/TrainingJobs", "job", positions, False, lambda *_: None
    )

    assert sorted(positions) == sorted(client.stream_names)
    assert client.describe_calls == 2
