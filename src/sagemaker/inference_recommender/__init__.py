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
"""Classes for using Inference Recommender with Amazon SageMaker."""
from __future__ import absolute_import


class Phase:
    """Used to store phases of a traffic pattern to perform endpoint load testing.

    Required for an Advanced Inference Recommendations Job
    """

    def __init__(self, duration_in_seconds: int, initial_number_of_users: int, spawn_rate: int):
        """Initialze a `Phase`"""
        self.to_json = {
            "DurationInSeconds": duration_in_seconds,
            "InitialNumberOfUsers": initial_number_of_users,
            "SpawnRate": spawn_rate,
        }


class ModelLatencyThreshold:
    """Used to store inference request/response latency to perform endpoint load testing.

    Required for an Advanced Inference Recommendations Job
    """

    def __init__(self, percentile: str, value_in_milliseconds: int):
        """Initialze a `ModelLatencyThreshold`"""
        self.to_json = {"Percentile": percentile, "ValueInMilliseconds": value_in_milliseconds}
