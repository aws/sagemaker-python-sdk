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

import ast
from abc import abstractmethod

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier


class ParamRenamer(Modifier):
    """Abstract class to take in an AST node, check if it is a function call with
    an argument that needs to be renamed, and rename the argument if needed.
    """

    @property
    @abstractmethod
    def calls_to_modify(self):
        """A dictionary mapping function names to possible namespaces."""

    @property
    @abstractmethod
    def old_param_name(self):
        """The parameter name used in previous versions of the SageMaker Python SDK."""

    @property
    @abstractmethod
    def new_param_name(self):
        """The parameter name used in version 2.0 and later of the SageMaker Python SDK."""

    def node_should_be_modified(self, node):
        """Checks if the node matches any of the relevant functions and
        contains the parameter to be renamed.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` matches the relevant function calls and
                contains the parameter to be renamed.
        """
        return matching.matches_any(node, self.calls_to_modify) and self._has_param_to_rename(node)

    def _has_param_to_rename(self, node):
        """Checks if the call has the argument that needs to be renamed."""
        return _keyword_from_keywords(node, self.old_param_name) is not None

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to rename the attribute.

        Args:
            node (ast.Call): a node that represents the relevant function call.
        """
        keyword = _keyword_from_keywords(node, self.old_param_name)
        keyword.arg = self.new_param_name


def _keyword_from_keywords(node, param_name):
    """Retrieves a keyword argument from the node's keywords.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        param_name (str): the name of the argument.

    Returns:
        ast.keyword: the keyword argument if it is present. Otherwise, this returns ``None``.
    """
    for kw in node.keywords:
        if kw.arg == param_name:
            return kw

    return None


class DistributionParameterRenamer(ParamRenamer):
    """A class to rename the ``distributions`` attribute to ``distrbution`` in
    MXNet and TensorFlow estimators.

    This looks for the following calls:

    - ``<Framework>``
    - ``sagemaker.<framework>.<Framework>``
    - ``sagemaker.<framework>.estimator.<Framework>``

    where ``<Framework>`` is either ``TensorFlow`` or ``MXNet``.
    """

    @property
    def calls_to_modify(self):
        """A dictionary mapping ``MXNet`` and ``TensorFlow`` to their respective namespaces."""
        return {
            "TensorFlow": ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator"),
            "MXNet": ("sagemaker.mxnet", "sagemaker.mxnet.estimator"),
        }

    @property
    def old_param_name(self):
        """The previous name for the distribution argument."""
        return "distributions"

    @property
    def new_param_name(self):
        """The new name for the distribution argument."""
        return "distribution"


class S3SessionRenamer(ParamRenamer):
    """A class to rename the ``session`` attribute to ``sagemaker_session`` in
    ``S3Uploader`` and ``S3Downloader``.

    This looks for the following calls:

    - ``sagemaker.s3.S3Uploader.<function>``
    - ``s3.S3Uploader.<function>``
    - ``S3Uploader.<function>``

    where ``S3Uploader`` is either ``S3Uploader`` or ``S3Downloader``, and where
    ``<function>`` is any of the functions belonging to those two classes.
    """

    @property
    def calls_to_modify(self):
        """A dictionary mapping S3 utility functions to their respective namespaces."""
        return {
            "download": ("sagemaker.s3.S3Downloader", "s3.S3Downloader", "S3Downloader"),
            "list": ("sagemaker.s3.S3Downloader", "s3.S3Downloader", "S3Downloader"),
            "read_file": ("sagemaker.s3.S3Downloader", "s3.S3Downloader", "S3Downloader"),
            "upload": ("sagemaker.s3.S3Uploader", "s3.S3Uploader", "S3Uploader"),
            "upload_string_as_file_body": (
                "sagemaker.s3.S3Uploader",
                "s3.S3Uploader",
                "S3Uploader",
            ),
        }

    @property
    def old_param_name(self):
        """The previous name for the SageMaker session argument."""
        return "session"

    @property
    def new_param_name(self):
        """The new name for the SageMaker session argument."""
        return "sagemaker_session"

    def node_should_be_modified(self, node):
        """Checks if the node is one of the S3 utility functions and
        contains the ``session`` parameter.
        """
        if isinstance(node.func, ast.Name):
            return False

        return super(S3SessionRenamer, self).node_should_be_modified(node)
