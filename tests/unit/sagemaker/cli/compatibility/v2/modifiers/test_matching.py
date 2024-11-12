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

from sagemaker.cli.compatibility.v2.modifiers import matching
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call


def test_matches_any():
    name_to_namespaces_dict = {
        "KMeansPredictor": ("sagemaker", "sagemaker.amazon.kmeans"),
        "Predictor": ("sagemaker.tensorflow.serving",),
    }

    matches = (
        "KMeansPredictor()",
        "sagemaker.KMeansPredictor()",
        "sagemaker.amazon.kmeans.KMeansPredictor()",
        "Predictor()",
        "sagemaker.tensorflow.serving.Predictor()",
    )

    for call in matches:
        assert matching.matches_any(ast_call(call), name_to_namespaces_dict)

    non_matches = ("MXNet()", "sagemaker.mxnet.MXNet()")
    for call in non_matches:
        assert not matching.matches_any(ast_call(call), name_to_namespaces_dict)


def test_matches_name_or_namespaces():
    name = "KMeans"
    namespaces = ("sagemaker", "sagemaker.amazon.kmeans")

    matches = ("KMeans()", "sagemaker.KMeans()")
    for call in matches:
        assert matching.matches_name_or_namespaces(ast_call(call), name, namespaces)

    non_matches = ("MXNet()", "sagemaker.mxnet.MXNet()")
    for call in non_matches:
        assert not matching.matches_name_or_namespaces(ast_call(call), name, namespaces)


def test_matches_name():
    assert matching.matches_name(ast_call("KMeans()"), "KMeans")
    assert not matching.matches_name(ast_call("sagemaker.KMeans()"), "KMeans")
    assert not matching.matches_name(ast_call("MXNet()"), "KMeans")


def test_matches_attr():
    assert matching.matches_attr(ast_call("sagemaker.amazon.kmeans.KMeans()"), "KMeans")
    assert not matching.matches_attr(ast_call("KMeans()"), "KMeans")
    assert not matching.matches_attr(ast_call("sagemaker.mxnet.MXNet()"), "KMeans")


def test_matches_namespace():
    assert matching.matches_namespace(ast_call("sagemaker.mxnet.MXNet()"), "sagemaker.mxnet")
    assert not matching.matches_namespace(ast_call("sagemaker.KMeans()"), "sagemaker.mxnet")


def test_has_arg():
    assert matching.has_arg(ast_call("MXNet(framework_version=mxnet_version)"), "framework_version")
    assert not matching.has_arg(ast_call("MXNet()"), "framework_version")
