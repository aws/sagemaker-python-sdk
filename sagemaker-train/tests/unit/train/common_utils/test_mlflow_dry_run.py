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
"""Unit tests for MLflow dry_run behavior in _resolve_mlflow_resource_arn."""
import logging
from unittest.mock import Mock, patch, MagicMock

import pytest

from sagemaker.train.common_utils.finetune_utils import (
    _resolve_mlflow_resource_arn,
    _create_mlflow_config,
)


class TestResolveMlflowDryRunSkipsCreation:
    """When dry_run=True, _resolve_mlflow_resource_arn must never create or wait."""

    @patch("sagemaker.train.common_utils.finetune_utils._create_mlflow_app")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_no_apps_dry_run_skips_creation(
        self, mock_client, mock_domain, mock_create_app
    ):
        """dry_run=True with zero apps returns None without calling _create_mlflow_app."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Summaries": []}]
        mock_client.return_value.get_paginator.return_value = mock_paginator
        mock_domain.return_value = None

        mock_session = Mock()

        result = _resolve_mlflow_resource_arn(mock_session, dry_run=True)

        assert result is None
        mock_create_app.assert_not_called()

    @patch("sagemaker.train.common_utils.finetune_utils._create_mlflow_app")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_no_apps_non_dry_run_creates_app(
        self, mock_client, mock_domain, mock_create_app
    ):
        """Without dry_run, zero apps triggers _create_mlflow_app."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Summaries": []}]
        mock_client.return_value.get_paginator.return_value = mock_paginator
        mock_domain.return_value = None
        mock_create_app.return_value = "arn:aws:sagemaker:us-east-1:123:mlflow-app/new"

        mock_session = Mock()

        result = _resolve_mlflow_resource_arn(mock_session, dry_run=False)

        assert result == "arn:aws:sagemaker:us-east-1:123:mlflow-app/new"
        mock_create_app.assert_called_once()

    @patch("sagemaker.train.common_utils.finetune_utils._create_mlflow_app")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_creating_app_dry_run_skips_wait(
        self, mock_client, mock_domain, mock_create_app
    ):
        """dry_run=True with an app in 'Creating' state returns ARN without waiting."""
        creating_app = {
            "Arn": "arn:aws:sagemaker:us-east-1:123:mlflow-app/creating",
            "Status": "Creating",
            "AccountDefaultStatus": "ENABLED",
            "MlflowVersion": "3.4.0",
        }
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Summaries": [creating_app]}]
        mock_client.return_value.get_paginator.return_value = mock_paginator
        mock_domain.return_value = None

        mock_session = Mock()

        result = _resolve_mlflow_resource_arn(mock_session, dry_run=True)

        assert result == "arn:aws:sagemaker:us-east-1:123:mlflow-app/creating"
        mock_create_app.assert_not_called()

    @patch("sagemaker.train.common_utils.finetune_utils._create_mlflow_app_as_upgrade")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_version_below_minimum_dry_run_skips_upgrade(
        self, mock_client, mock_domain, mock_upgrade
    ):
        """dry_run=True with app below min version returns ARN without upgrading."""
        old_app = {
            "Arn": "arn:aws:sagemaker:us-east-1:123:mlflow-app/old",
            "Status": "Created",
            "AccountDefaultStatus": "ENABLED",
            "MlflowVersion": "2.0.0",
        }
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Summaries": [old_app]}]
        mock_client.return_value.get_paginator.return_value = mock_paginator
        mock_domain.return_value = None

        mock_session = Mock()

        result = _resolve_mlflow_resource_arn(
            mock_session, min_mlflow_version="3.10", dry_run=True
        )

        assert result == "arn:aws:sagemaker:us-east-1:123:mlflow-app/old"
        mock_upgrade.assert_not_called()

    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_ready_app_dry_run_returns_normally(self, mock_client, mock_domain):
        """dry_run=True with an app in 'Updated' state returns ARN normally."""
        ready_app = {
            "Arn": "arn:aws:sagemaker:us-east-1:123:mlflow-app/ready",
            "Status": "Updated",
            "AccountDefaultStatus": "ENABLED",
            "MlflowVersion": "3.4.0",
        }
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Summaries": [ready_app]}]
        mock_client.return_value.get_paginator.return_value = mock_paginator
        mock_domain.return_value = None

        mock_session = Mock()

        result = _resolve_mlflow_resource_arn(mock_session, dry_run=True)

        assert result == "arn:aws:sagemaker:us-east-1:123:mlflow-app/ready"


class TestResolveMlflowDryRunWarnings:
    """Verify appropriate warnings are logged during dry_run."""

    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_no_apps_logs_warning(self, mock_client, mock_domain, caplog):
        """Warns that job submission would create an app."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Summaries": []}]
        mock_client.return_value.get_paginator.return_value = mock_paginator
        mock_domain.return_value = None

        mock_session = Mock()

        with caplog.at_level(logging.WARNING):
            _resolve_mlflow_resource_arn(mock_session, dry_run=True)

        assert "No MLflow app exists" in caplog.text
        assert "would create a new app" in caplog.text

    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.train.common_utils.finetune_utils._get_prod_sm_client")
    def test_creating_app_logs_warning(self, mock_client, mock_domain, caplog):
        """Warns that job submission would block on a Creating app."""
        creating_app = {
            "Arn": "arn:aws:sagemaker:us-east-1:123:mlflow-app/creating",
            "Status": "Creating",
            "AccountDefaultStatus": "ENABLED",
            "MlflowVersion": "3.4.0",
        }
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Summaries": [creating_app]}]
        mock_client.return_value.get_paginator.return_value = mock_paginator
        mock_domain.return_value = None

        mock_session = Mock()

        with caplog.at_level(logging.WARNING):
            _resolve_mlflow_resource_arn(mock_session, dry_run=True)

        assert "Creating" in caplog.text
        assert "would block" in caplog.text


class TestCreateMlflowConfigDryRun:
    """Verify _create_mlflow_config passes dry_run through."""

    @patch("sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn")
    def test_passes_dry_run_to_resolve(self, mock_resolve):
        """dry_run flag is forwarded to _resolve_mlflow_resource_arn."""
        mock_resolve.return_value = None
        mock_session = Mock()

        _create_mlflow_config(mock_session, dry_run=True)

        mock_resolve.assert_called_once_with(mock_session, None, dry_run=True)

    @patch("sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn")
    def test_dry_run_false_by_default(self, mock_resolve):
        """dry_run defaults to False."""
        mock_resolve.return_value = None
        mock_session = Mock()

        _create_mlflow_config(mock_session)

        mock_resolve.assert_called_once_with(mock_session, None, dry_run=False)
