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
from __future__ import absolute_import

import os


# TODO-experiment-plus: Remove this line, which loads the internal boto models.
# The corresponding model jsons were generated from the coral model package and should
# be updated regularly.
normal_json = "file://./tests/data/experiment/resources/sagemaker-2017-07-24.normal.json"
os.system(f"aws configure add-model --service-model {normal_json} --service-name sagemaker")

public_metrics_model_json = (
    "file://./tests/data/experiment/resources/sagemaker-metrics-2022-09-30.normal.json"
)
os.system(
    f"aws configure add-model --service-model {public_metrics_model_json} --service-name sagemaker-metrics"
)
