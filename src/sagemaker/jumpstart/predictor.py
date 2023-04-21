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
"""This module contains a Predictor class implementation specific for JumpStart models."""

from __future__ import absolute_import

from typing import Optional
from sagemaker.jumpstart.artifacts import (
    _retrieve_default_deserializer,
    _retrieve_default_serializer,
)

from sagemaker.predictor import Predictor
from sagemaker.session import Session
from sagemaker import content_types, accept_types
from sagemaker.jumpstart.enums import MIMEType


class JumpStartPredictor(Predictor):
    """Predictor to be used with JumpStart models.

    This predictor uses model-specific values for the serializer, deserializer,
    content type, and accept type.
    """

    def __init__(
        self,
        endpoint_name: str = None,
        sagemaker_session: Optional[Session] = None,
        model_id: str = None,
        model_version: str = "*",
        region: Optional[str] = None,
        tolerate_vulnerable_model: bool = False,
        tolerate_deprecated_model: bool = False,
    ):

        if model_id is None or endpoint_name is None:
            raise ValueError(
                "Must supply endpoint name and model id as input to JumpStart Predictor!"
            )

        serializer = _retrieve_default_serializer(
            model_id=model_id,
            model_version=model_version,
            region=region,
            tolerate_vulnerable_model=tolerate_vulnerable_model,
            tolerate_deprecated_model=tolerate_deprecated_model,
        )
        deserializer = _retrieve_default_deserializer(
            model_id=model_id,
            model_version=model_version,
            region=region,
            tolerate_vulnerable_model=tolerate_vulnerable_model,
            tolerate_deprecated_model=tolerate_deprecated_model,
        )

        super(JumpStartPredictor, self).__init__(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )

        self.model_id = model_id
        self.model_version = model_version
        self.region = region
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.tolerate_deprecated_model = tolerate_deprecated_model

    @property
    def content_type(self) -> MIMEType:
        """The MIME type of the data sent to the inference endpoint."""
        return content_types.retrieve_default(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
        )

    @property
    def accept(self) -> MIMEType:
        """The content type that is expected from the inference endpoint."""
        return accept_types.retrieve_default(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
        )
