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
"""Job subclasses that add ``show_result`` to the inference recommender resources."""
from __future__ import absolute_import

from typing import List

from sagemaker.core.resources import AIBenchmarkJob, AIRecommendationJob
from sagemaker.core.telemetry.constants import Feature
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter


class BenchmarkJob(AIBenchmarkJob):
    """``AIBenchmarkJob`` with a one-shot result reader.

    All standard lifecycle methods (``refresh``, ``wait``, ``stop``,
    ``delete``) are inherited from the underlying resource; ``show_result``
    is the only addition.
    """

    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION, func_name="BenchmarkJob.show_result"
    )
    def show_result(self):
        """Download the benchmark output from S3 and return a parsed result.

        Returns:
            BenchmarkResult: parsed metrics and run metadata. The job must be
            in a terminal state; ``show_result`` calls ``refresh()`` once but
            does not poll.
        """
        from sagemaker.serve.ai_inference_recommender.result import BenchmarkResult

        return BenchmarkResult.from_job(self)


class RecommendationJob(AIRecommendationJob):
    """``AIRecommendationJob`` with a one-shot result reader.

    All standard lifecycle methods (``refresh``, ``wait``, ``stop``,
    ``delete``) are inherited from the underlying resource; ``show_result``
    is the only addition.
    """

    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION, func_name="RecommendationJob.show_result"
    )
    def show_result(self) -> List:
        """Return the ranked recommendation rows produced by the job.

        Each row is a read-only view that pretty-prints when ``repr()``'d;
        attribute access forwards to the underlying service shape. The job
        must be in a terminal state; ``show_result`` calls ``refresh()`` once
        but does not poll.
        """
        from sagemaker.serve.ai_inference_recommender._recommendation_view import (
            _RecommendationView,
        )

        self.refresh()
        rows = list(self.recommendations or [])
        return [_RecommendationView(row, index=i) for i, row in enumerate(rows)]


__all__ = ["BenchmarkJob", "RecommendationJob"]
