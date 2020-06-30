# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Classes to modify Predictor code to be compatible
with version 2.0 and later of the SageMaker Python SDK.
"""
from __future__ import absolute_import

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

ESTIMATORS_WITH_DISTRIBUTION_PARAM = {
    "TensorFlow": ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator"),
    "MXNet": ("sagemaker.mxnet", "sagemaker.mxnet.estimator"),
}


class DistributionParameterRenamer(Modifier):
    """A class to rename the ``distributions`` attribute in MXNet and TensorFlow estimators."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates an MXNet or TensorFlow estimator and
        contains the ``distributions`` parameter.

        This looks for the following calls:

        - ``<Framework>``
        - ``sagemaker.<framework>.<Framework>``
        - ``sagemaker.<framework>.estimator.<Framework>``

        where ``<Framework>`` is either ``TensorFlow`` or ``MXNet``.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates an MXNet or TensorFlow estimator with
                the ``distributions`` parameter.
        """
        return matching.matches_any(
            node, ESTIMATORS_WITH_DISTRIBUTION_PARAM
        ) and self._has_distribution_arg(node)

    def _has_distribution_arg(self, node):
        """Checks if the node has the ``distributions`` parameter in its keywords."""
        for kw in node.keywords:
            if kw.arg == "distributions":
                return True

        return False

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to rename the ``distributions`` attribute to
        ``distribution``.

        Args:
            node (ast.Call): a node that represents an MXNet or TensorFlow constructor.
        """
        for kw in node.keywords:
            if kw.arg == "distributions":
                kw.arg = "distribution"
                break
