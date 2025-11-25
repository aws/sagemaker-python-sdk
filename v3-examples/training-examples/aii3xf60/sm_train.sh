
#!/bin/bash
set -e
echo "Starting training script"

handle_error() {
    EXIT_STATUS=$?
    echo "An error occurred with exit code $EXIT_STATUS"
    if [ ! -s /opt/ml/output/failure ]; then
        echo "Training Execution failed. For more details, see CloudWatch logs at 'aws/sagemaker/TrainingJobs'.
TrainingJob - $TRAINING_JOB_NAME" >> /opt/ml/output/failure
    fi
    exit $EXIT_STATUS
}

check_python() {
    SM_PYTHON_CMD=$(command -v python3 || command -v python)
    SM_PIP_CMD=$(command -v pip3 || command -v pip)

    # Check if Python is found
    if [[ -z "$SM_PYTHON_CMD" || -z "$SM_PIP_CMD" ]]; then
        echo "Error: The Python executable was not found in the system path."
        return 1
    fi

    return 0
}

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

cd /opt/ml/input/data/code 



echo "Running Basic Script driver"
$SM_PYTHON_CMD /opt/ml/input/data/sm_drivers/distributed_drivers/basic_script_driver.py


echo "Training Container Execution Completed"
