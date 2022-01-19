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
"""This module stores exceptions related to SageMaker JumpStart."""

from typing import List, Optional


class VulnerableJumpStartModelError(Exception):
    """Exception raised for errors with vulnerable JumpStart models."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        vulnerabilities: Optional[List[str]] = None,
        inference: Optional[bool] = None,
        message: Optional[str] = None,
    ):
        if message:
            self.message = message
        else:
            if None in [model_id, version, vulnerabilities, inference]:
                raise ValueError(
                    "Must specify `model_id`, `version`, `vulnerabilities`, "
                    "and inference arguments."
                )
            if inference is True:
                self.message = (
                    f"JumpStart model '{model_id}' and version '{version}' has at least 1 "
                    "vulnerable dependency in the inference scripts. "
                    f"List of vulnerabilities: {', '.join(vulnerabilities)}"
                )
            else:
                self.message = (
                    f"JumpStart model '{model_id}' and version '{version}' has at least 1 "
                    "vulnerable dependency in the training scripts. "
                    f"List of vulnerabilities: {', '.join(vulnerabilities)}"
                )

        super().__init__(self.message)


class DeprecatedJumpStartModelError(Exception):
    """Exception raised for errors with deprecated JumpStart models."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        message: Optional[str] = None,
    ):
        if message:
            self.message = message
        else:
            if None in [model_id, version]:
                raise ValueError("Must specify `model_id` and `version` arguments.")
            self.message = f"JumpStart model '{model_id}' and version '{version}' is deprecated."

        super().__init__(self.message)
