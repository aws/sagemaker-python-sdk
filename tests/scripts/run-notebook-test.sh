#!/bin/bash
#
# Run a test against a SageMaker notebook
# Only runs within the SDK's CI/CD environment

set -euo pipefail

python setup.py sdist
aws s3 --region us-west-2 cp ./dist/sagemaker-*.tar.gz s3://sagemaker-python-sdk-pr/sagemaker.tar.gz
aws s3 cp s3://sagemaker-mead-cli/mead-nb-test.tar.gz mead-nb-test.tar.gz
tar -xzf mead-nb-test.tar.gz
git clone --depth 1 https://github.com/awslabs/amazon-sagemaker-examples.git
export JAVA_HOME=$(get-java-home)
echo "set JAVA_HOME=$JAVA_HOME"
export SAGEMAKER_ROLE_ARN=$(aws iam list-roles --output text --query "Roles[?RoleName == 'SageMakerRole'].Arn")
echo "set SAGEMAKER_ROLE_ARN=$SAGEMAKER_ROLE_ARN"
./runtime/bin/mead-run-nb-test \
--instance-type ml.c4.8xlarge \
--region us-west-2 \
--lifecycle-config-name install-python-sdk \
--notebook-instance-role-arn $SAGEMAKER_ROLE_ARN \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_batch_transform_mnist.ipynb
