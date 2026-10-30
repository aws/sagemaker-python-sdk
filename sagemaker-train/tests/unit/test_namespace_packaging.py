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
"""Regression guard for GH #5795.

sagemaker-train must not ship its own top-level ``sagemaker/__init__.py``:
sagemaker-core owns that file, and having two distributions install the same
path makes strict installers / OS package managers refuse to co-install them.
The ``sagemaker`` package is a namespace shared across the sub-distributions.
"""

from __future__ import absolute_import

from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"


def test_does_not_ship_top_level_sagemaker_init():
    top_level_init = _SRC / "sagemaker" / "__init__.py"
    assert not top_level_init.exists(), (
        f"{top_level_init} must not exist: sagemaker-core owns "
        "sagemaker/__init__.py (see GH #5795)."
    )


def test_still_ships_the_subpackage_init():
    # The distribution's own sub-package must remain a real package.
    assert (_SRC / "sagemaker" / "train" / "__init__.py").exists()
