# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This file contains code related to model metrics, including metric source."""
from __future__ import absolute_import


class ModelMetrics(object):
    """Accepts model metrics parameters for conversion to request dict."""

    def __init__(
        self,
        model_statistics=None,
        model_constraints=None,
        model_data_statistics=None,
        model_data_constraints=None,
        bias=None,
        explainability=None,
    ):
        """Initialize a ``ModelMetrics`` instance and turn parameters into dict.

        # TODO: flesh out docstrings
        Args:
            model_constraints (MetricsSource):
            model_data_constraints (MetricsSource):
            model_data_statistics (MetricsSource):
            bias (MetricsSource):
            explainability (MetricsSource):
        """
        self.model_statistics = model_statistics
        self.model_constraints = model_constraints
        self.model_data_statistics = model_data_statistics
        self.model_data_constraints = model_data_constraints
        self.bias = bias
        self.explainability = explainability

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        model_metrics_request = {}

        model_quality = {}
        if self.model_statistics is not None:
            model_quality["Statistics"] = self.model_statistics._to_request_dict()
        if self.model_constraints is not None:
            model_quality["Constraints"] = self.model_constraints._to_request_dict()
        if model_quality:
            model_metrics_request["ModelQuality"] = model_quality

        model_data_quality = {}
        if self.model_data_statistics is not None:
            model_data_quality["Statistics"] = self.model_data_statistics._to_request_dict()
        if self.model_data_constraints is not None:
            model_data_quality["Constraints"] = self.model_data_constraints._to_request_dict()
        if model_data_quality:
            model_metrics_request["ModelDataQuality"] = model_data_quality

        if self.bias is not None:
            model_metrics_request["Bias"] = self.bias._to_request_dict()
        if self.explainability is not None:
            model_metrics_request["Explainability"] = self.explainability._to_request_dict()
        return model_metrics_request


class MetricsSource(object):
    """Accepts metrics source parameters for conversion to request dict."""

    def __init__(
        self,
        content_type,
        s3_uri,
        content_digest=None,
    ):
        """Initialize a ``MetricsSource`` instance and turn parameters into dict.

        # TODO: flesh out docstrings
        Args:
            content_type (str):
            s3_uri (str):
            content_digest (str):
        """
        self.content_type = content_type
        self.s3_uri = s3_uri
        self.content_digest = content_digest

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        metrics_source_request = {"ContentType": self.content_type, "S3Uri": self.s3_uri}
        if self.content_digest is not None:
            metrics_source_request["ContentDigest"] = self.content_digest
        return metrics_source_request
