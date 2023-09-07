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

import pandas as pd

from sagemaker.remote_function import remote


if __name__ == "__main__":

    @remote(
        role="SageMakerRole",
        instance_type="ml.m5.xlarge",
        dependencies="auto_capture",
    )
    def multiply(dataframe: pd.DataFrame, factor: float):
        return dataframe * factor

    df = pd.DataFrame(
        {
            "A": [14, 4, 5, 4, 1],
            "B": [5, 2, 54, 3, 2],
            "C": [20, 20, 7, 3, 8],
            "D": [14, 3, 6, 2, 6],
        }
    )
    multiply(df, 10.0)
