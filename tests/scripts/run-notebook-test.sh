#!/bin/bash
#
# Run a test against a SageMaker notebook
# Only runs within the SDK's CI/CD environment

function CreateLifeCycleConfig ()
{

echo "Creating life cycle config...."
LIFECYCLE_CONFIG_NAME=$1
LIFECYCLE_CONFIG_CONTENT=$2
aws sagemaker create-notebook-instance-lifecycle-config --notebook-instance-lifecycle-config-name "$LIFECYCLE_CONFIG_NAME" --on-create Content="$LIFECYCLE_CONFIG_CONTENT"

}

function DeleteLifeCycleConfig ()
{

echo "Deleting the existing life cycle config...."
LIFECYCLE_CONFIG_NAME=$1
aws sagemaker delete-notebook-instance-lifecycle-config --notebook-instance-lifecycle-config-name  "$LIFECYCLE_CONFIG_NAME"

}


function CreateLifeCycleConfigContent ()
{
ACCOUNT_ID=$1
COMMIT_ID=$2

TARBALL_DIRECTORY=/tmp/sdk-tarballs
LIFECYCLE_CONFIG_1=$(cat << 'EOF'
#!/bin/bash

set -e
set -x

mkdir "$HOME/.dlami"
touch "$HOME/.dlami/dlami_build_in_progress"
TARBALL_DIRECTORY=/tmp/sdk-tarballs
mkdir -p "$TARBALL_DIRECTORY"

EOF
)

LIFECYCLE_CONFIG_2=$(cat << EOF

aws s3 --region us-west-2 cp "s3://sagemaker-python-sdk-$ACCOUNT_ID/notebook_test/sagemaker-$COMMIT_ID.tar.gz" "$TARBALL_DIRECTORY/sagemaker.tar.gz"

EOF
)

LIFECYCLE_CONFIG_3=$(cat << 'EOF'

# Include "base" separately since it's not a subdirectory.
for env in base /home/ec2-user/anaconda3/envs/*; do
    echo "Updating SageMaker vended software in $env from pre-release SDKs..."

    sudo -u ec2-user -E sh -c 'source /home/ec2-user/anaconda3/bin/activate "$env"'

    if [ $env = base ]; then
        ENV_PYTHON=python
    else
        ENV_PYTHON="$env/bin/python"
    fi

    PYTHON_VERSION=$($ENV_PYTHON -c"import sys; print('{}{}'.format(sys.version_info[0], sys.version_info[1]))")
    if [ $PYTHON_VERSION -lt 38 ]; then
        echo "Skipping because $env uses py$PYTHON_VERSION which is incompatible with SageMaker Python SDK"
        sudo -u ec2-user -E sh -c 'source /home/ec2-user/anaconda3/bin/deactivate'
        continue
    fi

    echo "Updating SageMaker Python SDK..."
    $ENV_PYTHON -m pip install --upgrade pip
    $ENV_PYTHON -m pip install "$TARBALL_DIRECTORY/sagemaker.tar.gz"

    sudo -u ec2-user -E sh -c 'source /home/ec2-user/anaconda3/bin/deactivate'

    echo "Update of $env is complete."
done

sudo rm -rf "$MODELS_SOURCE_DIRECTORY"
sudo rm -rf "$TARBALL_DIRECTORY"
rm -rf "$HOME/.dlami"

EOF
)

LIFECYCLE_CONFIG_CONTENT=$((echo "$LIFECYCLE_CONFIG_1$LIFECYCLE_CONFIG_2$LIFECYCLE_CONFIG_3"|| echo "")| base64)

echo "$LIFECYCLE_CONFIG_CONTENT"

}

set -euo pipefail

# git doesn't work in codepipeline, use CODEBUILD_RESOLVED_SOURCE_VERSION to get commit id
codebuild_initiator="${CODEBUILD_INITIATOR:-0}"
if [ "${codebuild_initiator:0:12}" == "codepipeline" ]; then
    COMMIT_ID="${CODEBUILD_RESOLVED_SOURCE_VERSION}"
else
    COMMIT_ID=$(git rev-parse --short HEAD)
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LIFECYCLE_CONFIG_NAME="install-python-sdk-$COMMIT_ID"

python setup.py sdist

aws s3 --region us-west-2 cp ./dist/sagemaker-*.tar.gz s3://sagemaker-python-sdk-$ACCOUNT_ID/notebook_test/sagemaker-$COMMIT_ID.tar.gz
aws s3 cp s3://sagemaker-python-sdk-cli-$ACCOUNT_ID/mead-nb-test.tar.gz mead-nb-test.tar.gz
tar -xzf mead-nb-test.tar.gz


LIFECYCLE_CONFIG_CONTENT=$(CreateLifeCycleConfigContent "$ACCOUNT_ID" "$COMMIT_ID" )

if !(CreateLifeCycleConfig "$LIFECYCLE_CONFIG_NAME" "$LIFECYCLE_CONFIG_CONTENT")  ; then
(DeleteLifeCycleConfig "$LIFECYCLE_CONFIG_NAME")
(CreateLifeCycleConfig "$LIFECYCLE_CONFIG_NAME" "$LIFECYCLE_CONFIG_CONTENT")
fi

if [ -d amazon-sagemaker-examples ]; then rm -Rf amazon-sagemaker-examples; fi
git clone --depth 1 https://github.com/aws/amazon-sagemaker-examples.git

export JAVA_HOME=$(get-java-home)
echo "set JAVA_HOME=$JAVA_HOME"
export SAGEMAKER_ROLE_ARN=$(aws iam list-roles --output text --query "Roles[?RoleName == 'SageMakerRole'].Arn")
echo "set SAGEMAKER_ROLE_ARN=$SAGEMAKER_ROLE_ARN"

./runtime/bin/mead-run-nb-test \
--instance-type ml.c4.8xlarge \
--region us-west-2 \
--lifecycle-config-name $LIFECYCLE_CONFIG_NAME \
--notebook-instance-role-arn $SAGEMAKER_ROLE_ARN \
--platformIdentifier notebook-al2-v2 \
--consider-skips-failures \
./amazon-sagemaker-examples/sagemaker_processing/spark_distributed_data_processing/sagemaker-spark-processing.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/1P_kmeans_highlevel/kmeans_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.ipynb \
./amazon-sagemaker-examples/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.ipynb \

# Skipping test until fix in example notebook to install docker-compose is complete
#./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_moving_from_framework_mode_to_script_mode/tensorflow_moving_from_framework_mode_to_script_mode.ipynb \

# Skipping this test until we fix the notebook to use the correct version of TensorFlow for training and inference
# ./amazon-sagemaker-examples/advanced_functionality/tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb \

(DeleteLifeCycleConfig "$LIFECYCLE_CONFIG_NAME")
