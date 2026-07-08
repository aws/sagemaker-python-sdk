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
"""Shared utilities for resolving MLflow tracking fields across trainers and evaluators."""

from typing import Optional, Tuple


def resolve_mlflow_tracking_fields(
    mlflow_tracking_uri: Optional[str],
    mlflow_experiment_name: Optional[str],
    mlflow_run_name: Optional[str],
    base_job_name: str,
) -> Tuple[str, str, str]:
    """Resolve MLflow tracking fields with safe defaults.

    When a tracking URI is set but experiment/run names are not provided,
    defaults them to ``base_job_name``. This prevents recipe validation
    failures in the OSS training/evaluation container (which requires
    non-empty experiment and run names alongside a tracking URI) and
    provides a meaningful default for Nova containers as well.

    This utility is shared across BaseTrainer (serverful SMTJ and HyperPod)
    and BaseEvaluator to ensure consistent MLflow behavior.

    Args:
        mlflow_tracking_uri: The MLflow tracking server ARN or URI.
            ``None`` or empty string means tracking is disabled.
        mlflow_experiment_name: User-provided experiment name.
            ``None`` or empty means "not set".
        mlflow_run_name: User-provided run name.
            ``None`` or empty means "not set".
        base_job_name: Fallback experiment/run name when a tracking URI is
            set but no name was provided.

    Returns:
        Tuple of ``(tracking_uri, experiment_name, run_name)``. All values
        are guaranteed non-None strings. When tracking is disabled, all
        three are empty strings.
    """
    tracking_uri = mlflow_tracking_uri or ""
    experiment_name = mlflow_experiment_name or ""
    run_name = mlflow_run_name or ""

    if tracking_uri:
        if not experiment_name:
            experiment_name = base_job_name
        if not run_name:
            run_name = base_job_name

    return tracking_uri, experiment_name, run_name
