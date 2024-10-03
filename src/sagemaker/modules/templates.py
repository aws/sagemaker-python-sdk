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
"""Templates module."""
from __future__ import absolute_import

TRAIN_SCRIPT_TEMPLATE = """
#!/bin/bash
echo "Starting training script"

echo "/opt/ml/input/config/resourceconfig.json:"
cat /opt/ml/input/config/resourceconfig.json
echo

echo "/opt/ml/input/config/inputdataconfig.json:"
cat /opt/ml/input/config/inputdataconfig.json
echo

echo "/opt/ml/input/config/hyperparameters.json:"
cat /opt/ml/input/config/hyperparameters.json
echo

python --version
{working_dir}
{install_requirements}
CMD="{command}"
echo "Running command: $CMD"
eval $CMD
EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "Command failed with exit status $EXIT_STATUS"
    if [ ! -s /opt/ml/output/failure ]; then
        echo "Command failed with exit code $EXIT_STATUS.
For more details, see CloudWatch logs at 'aws/sagemaker/TrainingJobs'.
TrainingJob - $TRAINING_JOB_NAME" >> /opt/ml/output/failure
    fi
    exit $EXIT_STATUS
else
    echo "Command succeeded"
fi
"""
