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
"""Module containing classes necessary for capturing output. For use with tests."""
from __future__ import absolute_import

import sys
from contextlib import contextmanager
from six import StringIO


@contextmanager
def captured_output():
    """Use this when capturing print output for tests.

    Example:
        >>>>    with captured_output() as (out, err):
        >>>>        method_that_prints_hello_sagemaker()
        >>>>    output = out.getvalue().strip()
        >>>>    assert output == "Hello SageMaker!"
    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err
