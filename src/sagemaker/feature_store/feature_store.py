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
"""Feature Store.

Amazon SageMaker Feature Store is a fully managed, purpose-built repository to store, share, and
manage features for machine learning (ML) models.
"""
from __future__ import absolute_import

import datetime
from typing import Dict, Any

import attr

from sagemaker import Session


@attr.s
class FeatureStore:
    """FeatureStore definition.

    This class instantiates a FeatureStore object that comprises a SageMaker session instance.

    Attributes:
        sagemaker_session (Session): session instance to perform boto calls.
    """

    sagemaker_session: Session = attr.ib(default=Session)

    def list_feature_groups(
        self,
        name_contains: str = None,
        feature_group_status_equals: str = None,
        offline_store_status_equals: str = None,
        creation_time_after: datetime.datetime = None,
        creation_time_before: datetime.datetime = None,
        sort_order: str = None,
        sort_by: str = None,
        max_results: int = None,
        next_token: str = None,
    ) -> Dict[str, Any]:
        """List all FeatureGroups satisfying given filters.

        Args:
            name_contains (str): A string that partially matches one or more FeatureGroups' names.
                Filters FeatureGroups by name.
            feature_group_status_equals (str): A FeatureGroup status.
                Filters FeatureGroups by FeatureGroup status.
            offline_store_status_equals (str): An OfflineStore status.
                Filters FeatureGroups by OfflineStore status.
            creation_time_after (datetime.datetime): Use this parameter to search for FeatureGroups
                created after a specific date and time.
            creation_time_before (datetime.datetime): Use this parameter to search for FeatureGroups
                created before a specific date and time.
            sort_order (str): The order in which FeatureGroups are listed.
            sort_by (str): The value on which the FeatureGroup list is sorted.
            max_results (int): The maximum number of results returned by ListFeatureGroups.
            next_token (str): A token to resume pagination of ListFeatureGroups results.
        Returns:
            Response dict from service.
        """
        return self.sagemaker_session.list_feature_groups(
            name_contains=name_contains,
            feature_group_status_equals=feature_group_status_equals,
            offline_store_status_equals=offline_store_status_equals,
            creation_time_after=creation_time_after,
            creation_time_before=creation_time_before,
            sort_order=sort_order,
            sort_by=sort_by,
            max_results=max_results,
            next_token=next_token,
        )
