# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This module contains code to create and manage SageMaker ``ImageConfig``"""


class ImageConfig(object):
    """Configuration of Docker image used in Model."""

    def __init__(
        self, repository_access_mode="Platform",
    ):
        """Initialize an ``ImageConfig``.

        Args:
            repository_access_mode (str): Set this to one of the following values (default: "Platform"):
                * Platform: The model image is hosted in Amazon ECR.
                * Vpc: The model image is hosted in a private Docker registry in your VPC.
        """
        self.repository_access_mode = repository_access_mode

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        req = {
            "RepositoryAccessMode": "Platform",
        }
        if self.repository_access_mode is not None:
            req["RepositoryAccessMode"] = self.repository_access_mode

        return req
