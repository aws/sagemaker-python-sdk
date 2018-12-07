# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from contextlib import contextmanager

import tests.integ

LOCK_PATH = os.path.join(tests.integ.DATA_DIR, 'local_mode_lock')


@contextmanager
def lock():
    # Since Local Mode uses the same port for serving, we need a lock in order
    # to allow concurrent test execution.
    local_mode_lock_fd = open(LOCK_PATH, 'w')
    local_mode_lock = local_mode_lock_fd.fileno()

    fcntl.lockf(local_mode_lock, fcntl.LOCK_EX)

    try:
        yield
    finally:
        time.sleep(5)
        fcntl.lockf(local_mode_lock, fcntl.LOCK_UN)
