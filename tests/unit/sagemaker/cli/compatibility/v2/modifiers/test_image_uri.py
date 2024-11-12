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

import pasta

from sagemaker.cli.compatibility.v2.modifiers import renamed_params
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call

ESTIMATORS = {
    "Chainer": ("sagemaker.chainer", "sagemaker.chainer.estimator"),
    "Estimator": ("sagemaker.estimator",),
    "Framework": ("sagemaker.estimator",),
    "MXNet": ("sagemaker.mxnet", "sagemaker.mxnet.estimator"),
    "PyTorch": ("sagemaker.pytorch", "sagemaker.pytorch.estimator"),
    "RLEstimator": ("sagemaker.rl", "sagemaker.rl.estimator"),
    "SKLearn": ("sagemaker.sklearn", "sagemaker.sklearn.estimator"),
    "TensorFlow": ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator"),
    "XGBoost": ("sagemaker.xgboost", "sagemaker.xgboost.estimator"),
}

MODELS = {
    "ChainerModel": ("sagemaker.chainer", "sagemaker.chainer.model"),
    "Model": ("sagemaker.model",),
    "MultiDataModel": ("sagemaker.multidatamodel",),
    "FrameworkModel": ("sagemaker.model",),
    "MXNetModel": ("sagemaker.mxnet", "sagemaker.mxnet.model"),
    "PyTorchModel": ("sagemaker.pytorch", "sagemaker.pytorch.model"),
    "SKLearnModel": ("sagemaker.sklearn", "sagemaker.sklearn.model"),
    "TensorFlowModel": ("sagemaker.tensorflow", "sagemaker.tensorflow.model"),
    "XGBoostModel": ("sagemaker.xgboost", "sagemaker.xgboost.model"),
}


def test_estimator_node_should_be_modified():
    modifier = renamed_params.EstimatorImageURIRenamer()

    for estimator, namespaces in ESTIMATORS.items():
        call = "{}(image_name='my-image:latest')".format(estimator)
        assert modifier.node_should_be_modified(ast_call(call))

        for namespace in namespaces:
            call = "{}.{}(image_name='my-image:latest')".format(namespace, estimator)
            assert modifier.node_should_be_modified(ast_call(call))


def test_estimator_node_should_be_modified_no_distribution():
    modifier = renamed_params.EstimatorImageURIRenamer()

    for estimator, namespaces in ESTIMATORS.items():
        call = "{}()".format(estimator)
        assert not modifier.node_should_be_modified(ast_call(call))

        for namespace in namespaces:
            call = "{}.{}()".format(namespace, estimator)
            assert not modifier.node_should_be_modified(ast_call(call))


def test_estimator_node_should_be_modified_random_function_call():
    modifier = renamed_params.EstimatorImageURIRenamer()
    assert not modifier.node_should_be_modified(ast_call("Session()"))


def test_estimator_modify_node():
    node = ast_call("TensorFlow(image_name=my_image)")
    modifier = renamed_params.EstimatorImageURIRenamer()
    modifier.modify_node(node)

    expected = "TensorFlow(image_uri=my_image)"
    assert expected == pasta.dump(node)


def test_model_node_should_be_modified():
    modifier = renamed_params.ModelImageURIRenamer()

    for model, namespaces in MODELS.items():
        call = "{}(image='my-image:latest')".format(model)
        assert modifier.node_should_be_modified(ast_call(call))

        for namespace in namespaces:
            call = "{}.{}(image='my-image:latest')".format(namespace, model)
            assert modifier.node_should_be_modified(ast_call(call))


def test_model_node_should_be_modified_no_distribution():
    modifier = renamed_params.ModelImageURIRenamer()

    for model, namespaces in MODELS.items():
        call = "{}()".format(model)
        assert not modifier.node_should_be_modified(ast_call(call))

        for namespace in namespaces:
            call = "{}.{}()".format(namespace, model)
            assert not modifier.node_should_be_modified(ast_call(call))


def test_model_node_should_be_modified_random_function_call():
    modifier = renamed_params.ModelImageURIRenamer()
    assert not modifier.node_should_be_modified(ast_call("Session()"))


def test_model_modify_node():
    node = ast_call("TensorFlowModel(image=my_image)")
    modifier = renamed_params.ModelImageURIRenamer()
    modifier.modify_node(node)

    expected = "TensorFlowModel(image_uri=my_image)"
    assert expected == pasta.dump(node)
