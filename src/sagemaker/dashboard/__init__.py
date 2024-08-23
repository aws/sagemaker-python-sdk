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
"""Imports the classes in this module to simplify customer imports

Example:
    >>> from sagemaker.dashboard import AutomaticDataQualityDashboard

"""
from __future__ import absolute_import

from sagemaker.dashboard.data_quality_dashboard import AutomaticDataQualityDashboard  # noqa: F401
from sagemaker.dashboard.model_quality_dashboard import AutomaticModelQualityDashboard  # noqa: F401
from sagemaker.dashboard.dashboard_variables import DashboardVariable  # noqa: F401
from sagemaker.dashboard.dashboard_widgets import (
    DashboardWidget, # noqa: F401
    DashboardWidgetProperties, # noqa: F401
)  
