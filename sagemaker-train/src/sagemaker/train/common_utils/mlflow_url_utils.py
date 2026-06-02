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
"""Shared MLflow presigned URL utilities."""

import logging
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode

logger = logging.getLogger(__name__)


def _build_mlflow_deep_link(
    authorized_url: str, experiment_id: str, run_id: Optional[str] = None
) -> str:
    """Build MLflow deep link URL with fragment-based routing.

    The MLflow SPA uses hash-based routing. The /auth endpoint consumes the
    authToken, sets a session cookie, and redirects to '/'. The browser
    preserves the URL fragment (#/experiments/...) across the redirect, so
    the SPA lands on the correct page.

    Format: {auth_url}#/experiments/{id}/runs/{run_id}
    """
    if not authorized_url:
        return ""

    parsed = urlparse(authorized_url)
    params = parse_qs(parsed.query)
    auth_token = params.get("authToken", [""])[0]
    if not auth_token:
        parts = authorized_url.split("authToken=", 1)
        if len(parts) > 1:
            auth_token = parts[1].split("&")[0]

    root_url = authorized_url.split("?")[0]
    fragment = f"/experiments/{experiment_id}"
    if run_id:
        fragment = f"{fragment}/runs/{run_id}?workspace=default"
    else:
        fragment = f"{fragment}?workspace=default"

    return f"{root_url}?authToken={auth_token}#{fragment}"


def _build_mlflow_deep_link_by_name(
    authorized_url: str, experiment_name: str
) -> str:
    """Build MLflow deep link URL by resolving experiment name to ID.

    Authenticates via the presigned URL to get a session, then queries the
    MLflow experiments API to resolve the experiment name to an ID.
    Falls back to experiment name search filter if resolution fails.
    """
    if not authorized_url:
        return ""

    experiment_id = _resolve_experiment_id(authorized_url, experiment_name)
    if experiment_id:
        return _build_mlflow_deep_link(authorized_url, experiment_id)

    # Fallback: use search filter in fragment
    parsed = urlparse(authorized_url)
    params = parse_qs(parsed.query)
    auth_token = params.get("authToken", [""])[0]
    if not auth_token:
        parts = authorized_url.split("authToken=", 1)
        if len(parts) > 1:
            auth_token = parts[1].split("&")[0]

    root_url = authorized_url.split("?")[0]
    from urllib.parse import quote
    return f"{root_url}?authToken={auth_token}#/experiments?searchFilter={quote(experiment_name)}"


def _resolve_experiment_id(
    authorized_url: str, experiment_name: str
) -> Optional[str]:
    """Resolve MLflow experiment name to ID by authenticating via presigned URL."""
    try:
        import requests

        s = requests.Session()
        s.get(authorized_url, allow_redirects=True, timeout=10)

        base = authorized_url.split("/auth")[0]
        resp = s.get(
            f"{base}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": experiment_name},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("experiment", {}).get("experiment_id")
    except Exception as e:
        logger.debug(f"Failed to resolve experiment ID for '{experiment_name}': {e}")
    return None


def _resolve_run_id(
    authorized_url: str, experiment_id: str, run_name: Optional[str] = None
) -> Optional[str]:
    """Resolve the latest MLflow run ID for an experiment, optionally filtered by run name."""
    try:
        import requests

        s = requests.Session()
        s.get(authorized_url, allow_redirects=True, timeout=10)

        base = authorized_url.split("/auth")[0]
        body = {
            "experiment_ids": [experiment_id],
            "max_results": 1,
            "order_by": ["start_time DESC"],
        }
        if run_name:
            body["filter"] = f'tags.mlflow.runName = "{run_name}"'

        resp = s.post(
            f"{base}/api/2.0/mlflow/runs/search",
            json=body,
            timeout=10,
        )
        if resp.status_code == 200:
            runs = resp.json().get("runs", [])
            if runs:
                return runs[0].get("info", {}).get("run_id")
    except Exception as e:
        logger.debug(f"Failed to resolve run ID for experiment {experiment_id}: {e}")
    return None


def get_presigned_mlflow_experiment_url(
    mlflow_resource_arn: str,
    mlflow_experiment_name: Optional[str] = None,
) -> Optional[str]:
    """Generate a presigned MLflow URL, optionally deep-linked to an experiment.

    Args:
        mlflow_resource_arn: MLflow tracking server or app ARN.
        mlflow_experiment_name: Optional experiment name for deep-linking.

    Returns:
        Presigned URL with experiment fragment, or base URL, or None on failure.
    """
    try:
        from sagemaker.core.utils.utils import SageMakerClient

        sm_client = SageMakerClient().sagemaker_client
        response = sm_client.create_presigned_mlflow_app_url(Arn=mlflow_resource_arn)
        base_url = response.get("AuthorizedUrl")
        if not base_url:
            return None

        if mlflow_experiment_name:
            experiment_id = _resolve_experiment_id(base_url, mlflow_experiment_name)
            if experiment_id:
                return _build_mlflow_deep_link(base_url, experiment_id)

        return base_url
    except Exception as e:
        logger.debug(f"Failed to generate MLflow experiment URL: {e}")
        return None


def get_presigned_mlflow_url(
    mlflow_resource_arn: str,
    experiment_id: Optional[str] = None,
    run_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Optional[str]:
    """Generate a presigned MLflow URL with the deepest link possible.

    Resolution order:
    1. If experiment_id is provided, deep-link to the experiment (and run if run_id given).
    2. Else if experiment_name is provided, resolve to experiment_id via REST API.
    3. Else return the base presigned URL.

    Args:
        mlflow_resource_arn: MLflow tracking server or app ARN.
        experiment_id: MLflow experiment ID for direct deep-linking.
        run_id: MLflow run ID for run-level deep-linking.
        experiment_name: Experiment name to resolve if experiment_id is not available.

    Returns:
        Presigned URL string, or None on failure.
    """
    try:
        from sagemaker.core.utils.utils import SageMakerClient

        sm_client = SageMakerClient().sagemaker_client
        response = sm_client.create_presigned_mlflow_app_url(Arn=mlflow_resource_arn)
        base_url = response.get("AuthorizedUrl")
        if not base_url:
            return None

        if experiment_id:
            return _build_mlflow_deep_link(base_url, experiment_id, run_id)

        if experiment_name:
            resolved_id = _resolve_experiment_id(base_url, experiment_name)
            if resolved_id:
                return _build_mlflow_deep_link(base_url, resolved_id)

        return base_url
    except Exception as e:
        logger.debug(f"Failed to generate presigned MLflow URL: {e}")
        return None
