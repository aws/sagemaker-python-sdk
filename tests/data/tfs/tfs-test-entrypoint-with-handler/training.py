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

"""Exports a toy TensorFlow model.
Exports a TensorFlow model to /opt/ml/model/
This graph calculates,
  y = a*x + b
where a and b are variables with a=0.5 and b=2.
"""
import shutil


def save_model():
    shutil.copytree("/opt/ml/code/123", "/opt/ml/model/123")


if __name__ == "__main__":
    save_model()
