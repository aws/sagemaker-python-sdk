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
"""SageMaker GenAI inference benchmarking and recommendation."""
from __future__ import absolute_import

from sagemaker.serve.ai_inference_recommender._constants import (
    InferenceFramework,
    PerformanceTarget,
)
from sagemaker.serve.ai_inference_recommender.exceptions import (
    FeatureGatedError,
    WorkloadValidationError,
)
from sagemaker.serve.ai_inference_recommender.jobs import (
    BenchmarkJob,
    RecommendationJob,
)
from sagemaker.serve.ai_inference_recommender.result import (
    BenchmarkMetric,
    BenchmarkMetrics,
    BenchmarkResult,
)
from sagemaker.serve.ai_inference_recommender.secrets import Secret
from sagemaker.serve.ai_inference_recommender.workload import Workload
from sagemaker.serve.ai_inference_recommender._model_builder_methods import (
    start_benchmark,
)


__all__ = [
    "BenchmarkJob",
    "BenchmarkMetric",
    "BenchmarkMetrics",
    "BenchmarkResult",
    "FeatureGatedError",
    "InferenceFramework",
    "PerformanceTarget",
    "RecommendationJob",
    "Secret",
    "Workload",
    "WorkloadValidationError",
    "start_benchmark",
]
