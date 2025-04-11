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
from __future__ import absolute_import

import fcntl
import os
import time
import tempfile
from contextlib import contextmanager

DEFAULT_LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_test_lock")


@contextmanager
def lock(path=DEFAULT_LOCK_PATH):
    """Create a file lock to control concurrent test execution. Certain tests or
    test operations need to limit concurrency to work reliably. Examples include
    local mode endpoint tests and vpc creation tests.
    """
    f = open(path, "w")
    fd = f.fileno()

    fcntl.lockf(fd, fcntl.LOCK_EX)

    try:
        yield
    finally:
        time.sleep(5)
        fcntl.lockf(fd, fcntl.LOCK_UN)
