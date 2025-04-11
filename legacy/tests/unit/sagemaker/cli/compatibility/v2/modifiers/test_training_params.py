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

import itertools

import pasta

from sagemaker.cli.compatibility.v2.modifiers import training_params
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call

ESTIMATORS_TO_NAMESPACES = {
    "AlgorithmEstimator": ("sagemaker", "sagemaker.algorithm"),
    "AmazonAlgorithmEstimatorBase": ("sagemaker.amazon.amazon_estimator",),
    "Chainer": ("sagemaker.chainer", "sagemaker.chainer.estimator"),
    "Estimator": ("sagemaker.estimator",),
    "EstimatorBase": ("sagemaker.estimator",),
    "FactorizationMachines": ("sagemaker", "sagemaker.amazon.factorization_machines"),
    "Framework": ("sagemaker.estimator",),
    "IPInsights": ("sagemaker", "sagemaker.amazon.ipinsights"),
    "KMeans": ("sagemaker", "sagemaker.amazon.kmeans"),
    "KNN": ("sagemaker", "sagemaker.amazon.knn"),
    "LDA": ("sagemaker", "sagemaker.amazon.lda"),
    "LinearLearner": ("sagemaker", "sagemaker.amazon.linear_learner"),
    "MXNet": ("sagemaker.mxnet", "sagemaker.mxnet.estimator"),
    "NTM": ("sagemaker", "sagemaker.amazon.ntm"),
    "Object2Vec": ("sagemaker", "sagemaker.amazon.object2vec"),
    "PCA": ("sagemaker", "sagemaker.amazon.pca"),
    "PyTorch": ("sagemaker.pytorch", "sagemaker.pytorch.estimator"),
    "RandomCutForest": ("sagemaker", "sagemaker.amazon.randomcutforest"),
    "RLEstimator": ("sagemaker.rl", "sagemaker.rl.estimator"),
    "SKLearn": ("sagemaker.sklearn", "sagemaker.sklearn.estimator"),
    "TensorFlow": ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator"),
    "XGBoost": ("sagemaker.xgboost", "sagemaker.xgboost.estimator"),
}

PARAMS_WITH_VALUES = (
    "train_instance_count=1",
    "train_instance_type='ml.c4.xlarge'",
    "train_max_run=8 * 60 * 60",
    "train_max_wait=1 * 60 * 60",
    "train_use_spot_instances=True",
    "train_volume_size=30",
    "train_volume_kms_key='key'",
)


def _estimators():
    for estimator, namespaces in ESTIMATORS_TO_NAMESPACES.items():
        yield estimator

        for namespace in namespaces:
            yield ".".join((namespace, estimator))


def test_node_should_be_modified():
    modifier = training_params.TrainPrefixRemover()

    for estimator in _estimators():
        for param in PARAMS_WITH_VALUES:
            call = ast_call("{}({})".format(estimator, param))
            assert modifier.node_should_be_modified(call)


def test_node_should_be_modified_no_params():
    modifier = training_params.TrainPrefixRemover()

    for estimator in _estimators():
        call = ast_call("{}()".format(estimator))
        assert not modifier.node_should_be_modified(call)


def test_node_should_be_modified_random_function_call():
    modifier = training_params.TrainPrefixRemover()
    assert not modifier.node_should_be_modified(ast_call("Session()"))


def test_modify_node():
    modifier = training_params.TrainPrefixRemover()

    for params in _parameter_combinations():
        node = ast_call("Estimator({})".format(params))
        modifier.modify_node(node)

        expected = "Estimator({})".format(params).replace("train_", "")
        assert expected == pasta.dump(node)


def _parameter_combinations():
    for subset_length in range(1, len(PARAMS_WITH_VALUES) + 1):
        for subset in itertools.combinations(PARAMS_WITH_VALUES, subset_length):
            yield ", ".join(subset)
