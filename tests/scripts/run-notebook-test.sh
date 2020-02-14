#!/bin/bash
#
# Run a test against a SageMaker notebook
# Only runs within the SDK's CI/CD environment

set -euo pipefail

python setup.py sdist
aws s3 --region us-west-2 cp ./dist/sagemaker-*.tar.gz s3://sagemaker-python-sdk-pr/sagemaker.tar.gz
aws s3 cp s3://sagemaker-mead-cli/mead-nb-test.tar.gz mead-nb-test.tar.gz
tar -xzf mead-nb-test.tar.gz
git clone https://github.com/awslabs/amazon-sagemaker-examples.git
export JAVA_HOME=$(get-java-home)
echo "set JAVA_HOME=$JAVA_HOME"
export SAGEMAKER_ROLE_ARN=$(aws iam list-roles --output text --query "Roles[?RoleName == 'SageMakerRole'].Arn")
echo "set SAGEMAKER_ROLE_ARN=$SAGEMAKER_ROLE_ARN"
./runtime/bin/mead-run-nb-test \
--instance-type ml.c4.8xlarge \
--region us-west-2 \
--lifecycle-config-name install-python-sdk \
--notebook-instance-role-arn $SAGEMAKER_ROLE_ARN \
./amazon-sagemaker-examples/advanced_functionality/kmeans_bring_your_own_model/kmeans_bring_your_own_model.ipynb \
./amazon-sagemaker-examples/advanced_functionality/tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/1P_kmeans_highlevel/kmeans_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/1P_kmeans_lowlevel/kmeans_mnist_lowlevel.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/chainer_sentiment_analysis/chainer_sentiment_analysis.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/keras_script_mode_pipe_mode_horovod/tensorflow_keras_CIFAR10.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/managed_spot_training_mxnet/managed_spot_training_mxnet.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/managed_spot_training_tensorflow_estimator/managed_spot_training_tensorflow_estimator.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_gluon_sentiment/mxnet_sentiment_analysis_with_gluon.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist_with_batch_transform.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_onnx_eia/mxnet_onnx_eia.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_onnx_export/mxnet_onnx_export.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/mxnet_onnx_superresolution/mxnet_onnx.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/pytorch_lstm_word_language_model/pytorch_rnn.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow-eager-script-mode/tf-eager-sm-scriptmode.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_keras/tensorflow_abalone_age_predictor_using_keras.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_layers/tensorflow_abalone_age_predictor_using_layers.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_batch_transform_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_distributed_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators_elastic_inference.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_keras_cifar10/tensorflow_keras_CIFAR10.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_moving_from_framework_mode_to_script_mode/tensorflow_moving_from_framework_mode_to_script_mode.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_pipemode_example/tensorflow_pipemode_example.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_resnet_cifar10_with_tensorboard/tensorflow_resnet_cifar10_with_tensorboard.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_script_mode_pipe_mode/tensorflow_script_mode_pipe_mode.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_script_mode_quickstart/tensorflow_script_mode_quickstart.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_script_mode_training_and_serving/tensorflow_script_mode_training_and_serving.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_script_mode_using_shell_commands/tensorflow_script_mode_using_shell_commands.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_serving_container/tensorflow_serving_container.ipynb \
./amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow_serving_using_elastic_inference_with_your_own_model/tensorflow_serving_pretrained_model_elastic_inference.ipynb \
./amazon-sagemaker-examples/sagemaker-spark/pyspark_mnist/pyspark_mnist_custom_estimator.ipynb \
./amazon-sagemaker-examples/sagemaker-spark/pyspark_mnist/pyspark_mnist_pca_kmeans.ipynb \
./amazon-sagemaker-examples/sagemaker-spark/pyspark_mnist/pyspark_mnist_pca_mllib_kmeans.ipynb \
./amazon-sagemaker-examples/sagemaker-spark/pyspark_mnist/pyspark_mnist_xgboost.ipynb