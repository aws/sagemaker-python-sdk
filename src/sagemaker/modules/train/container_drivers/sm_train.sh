
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
    if command -v python3 &>/dev/null; then
        SM_PYTHON_CMD="python3"
        SM_PIP_CMD="pip3"
        echo "Found python3"
    elif command -v python &>/dev/null; then
        SM_PYTHON_CMD="python"
        SM_PIP_CMD="pip"
        echo "Found python"
    else
        echo "Python may not be installed"
        return 1
    fi
}

trap 'handle_error' ERR

check_python

$SM_PYTHON_CMD --version

echo "/opt/ml/input/config/resourceconfig.json:"
cat /opt/ml/input/config/resourceconfig.json
echo

echo "/opt/ml/input/config/inputdataconfig.json:"
cat /opt/ml/input/config/inputdataconfig.json
echo

echo "/opt/ml/input/data/sm_drivers/sourcecode.json"
cat /opt/ml/input/data/sm_drivers/sourcecode.json
echo

echo "/opt/ml/input/data/sm_drivers/distributed_runner.json"
cat /opt/ml/input/data/sm_drivers/distributed_runner.json
echo

echo "Setting up environment variables"
$SM_PYTHON_CMD /opt/ml/input/data/sm_drivers/scripts/environment.py
source /opt/ml/input/data/sm_drivers/scripts/sm_training.env

cd /opt/ml/input/data/sm_code
$SM_PIP_CMD install -r requirements.txt

echo "Running MPI driver"
$SM_PYTHON_CMD /opt/ml/input/data/sm_drivers/mpi_driver.py


echo "Training Container Execution Completed"
