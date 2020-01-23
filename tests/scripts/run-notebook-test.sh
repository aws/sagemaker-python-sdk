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
./amazon-sagemaker-examples/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb \
./amazon-sagemaker-examples/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/1P_kmeans_highlevel/kmeans_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/1P_kmeans_lowlevel/kmeans_mnist_lowlevel.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_gluon_cifar10/mxnet_cifar10_local_mode.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_gluon_cifar10/mxnet_cifar10_with_gluon.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_gluon_mnist/mxnet_mnist_with_gluon.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_gluon_mnist/mxnet_mnist_with_gluon_local_mode.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_gluon_sentiment/mxnet_sentiment_analysis_with_gluon.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist_elastic_inference.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist_with_batch_transform.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_onnx_superresolution/mxnet_onnx.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/pytorch_lstm_word_language_model/pytorch_rnn.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/pytorch_mnist/pytorch_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_keras/tensorflow_abalone_age_predictor_using_keras.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_layers/tensorflow_abalone_age_predictor_using_layers.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_batch_transform_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_distributed_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators_elastic_inference.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators_elastic_inference_local.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_keras_cifar10/tensorflow_keras_CIFAR10.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_pipemode_example/tensorflow_pipemode_example.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_resnet_cifar10_with_tensorboard/tensorflow_resnet_cifar10_with_tensorboard.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/pytorch_mnist/pytorch_mnist.ipynb
