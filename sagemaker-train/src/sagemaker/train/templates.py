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

EXECUTE_BASE_COMMANDS = """
CMD="{base_command}"
echo "Executing command: $CMD"
eval $CMD
"""

EXECUTE_BASIC_SCRIPT_DRIVER = """
echo "Running Basic Script driver"
$SM_PYTHON_CMD /opt/ml/input/data/sm_drivers/distributed_drivers/basic_script_driver.py
"""

INSTALL_AUTO_REQUIREMENTS = """
if [ -f requirements.txt ]; then
    echo "Installing requirements"
    cat requirements.txt
    $SM_PIP_CMD install -r requirements.txt
else
    echo "No requirements.txt file found. Skipping installation."
fi
"""

INSTALL_REQUIREMENTS = """
echo "Installing requirements"
$SM_PIP_CMD install -r {requirements_file}
"""

EXEUCTE_DISTRIBUTED_DRIVER = """
echo "Running {driver_name} Driver"
$SM_PYTHON_CMD /opt/ml/input/data/sm_drivers/distributed_drivers/{driver_script}
"""

TRAIN_SCRIPT_TEMPLATE = """
#!/bin/bash
set -e
echo "Starting training script"

handle_error() {{
    EXIT_STATUS=$?
    echo "An error occurred with exit code $EXIT_STATUS"
    if [ ! -s /opt/ml/output/failure ]; then
        echo "Training Execution failed. For more details, see CloudWatch logs at 'aws/sagemaker/TrainingJobs'.
TrainingJob - $TRAINING_JOB_NAME" >> /opt/ml/output/failure
    fi
    exit $EXIT_STATUS
}}

check_python() {{
    SM_PYTHON_CMD=$(command -v python3 || command -v python)
    SM_PIP_CMD=$(command -v pip3 || command -v pip)

    # Check if Python is found
    if [[ -z "$SM_PYTHON_CMD" || -z "$SM_PIP_CMD" ]]; then
        echo "Error: The Python executable was not found in the system path."
        return 1
    fi

    return 0
}}

trap 'handle_error' ERR

check_python

set -x
$SM_PYTHON_CMD --version

echo "/opt/ml/input/config/resourceconfig.json:"
cat /opt/ml/input/config/resourceconfig.json
echo

echo "/opt/ml/input/config/inputdataconfig.json:"
cat /opt/ml/input/config/inputdataconfig.json
echo

echo "Setting up environment variables"
$SM_PYTHON_CMD /opt/ml/input/data/sm_drivers/scripts/environment.py

set +x
source /opt/ml/input/sm_training.env
set -x

{working_dir}
{install_requirements}
{execute_driver}

echo "Training Container Execution Completed"
"""
