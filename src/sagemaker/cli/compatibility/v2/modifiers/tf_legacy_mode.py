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
"""Classes to modify TensorFlow legacy mode code to be compatible
with version 2.0 and later of the SageMaker Python SDK.
"""
from __future__ import absolute_import

import ast

import boto3
import six

from sagemaker.cli.compatibility.v2.modifiers import framework_version, matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier
from sagemaker import fw_utils

TF_NAMESPACES = ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator")
LEGACY_MODE_PARAMETERS = (
    "checkpoint_path",
    "evaluation_steps",
    "requirements_file",
    "training_steps",
)


class TensorFlowLegacyModeConstructorUpgrader(Modifier):
    """A class to turn legacy mode parameters into hyperparameters, disable the ``model_dir``
    hyperparameter, and set the image URI when instantiating a TensorFlow estimator.
    """

    def __init__(self):
        """Initializes a ``TensorFlowLegacyModeConstructorUpgrader``."""
        self._region = None

    @property
    def region(self):
        """Returns the AWS region for constructing an ECR image URI."""
        if self._region is None:
            self._region = boto3.Session().region_name

        return self._region

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a TensorFlow estimator with legacy mode.

        This looks for the following formats:

        - ``TensorFlow``
        - ``sagemaker.tensorflow.TensorFlow``
        - ``sagemaker.tensorflow.estimator.TensorFlow``

        Legacy mode is enabled if (1) ``script_mode`` is ``False``, ``None``, or not specified,
        and (2) if ``py_version`` is ``py2`` or not specified.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is instantiating a TensorFlow estimator with legacy mode.
        """
        is_tf_constructor = matching.matches_name_or_namespaces(node, "TensorFlow", TF_NAMESPACES)
        return is_tf_constructor and self._is_legacy_mode(node)

    def _is_legacy_mode(self, node):
        """Checks if the ``ast.Call`` node's keywords signal using legacy mode."""
        script_mode = False
        py_version = "py2"

        for kw in node.keywords:
            if kw.arg == "script_mode":
                script_mode = bool(kw.value.value)
            if kw.arg == "py_version":
                py_version = kw.value.s

        return not (py_version.startswith("py3") or script_mode)

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node's keywords to turn TensorFlow legacy mode parameters
        into hyperparameters and sets ``model_dir=False``.

        The parameters that are converted into hyperparameters:

        - ``training_steps``
        - ``evaluation_steps``
        - ``checkpoint_path``
        - ``requirements_file``

        Args:
            node (ast.Call): a node that represents a TensorFlow constructor.
        """
        base_hps = {}
        additional_hps = {}
        kw_to_remove = []  # remove keyword args after so that none are skipped during iteration

        add_image_uri = True

        for kw in node.keywords:
            if kw.arg in ("script_mode", "model_dir"):
                # model_dir is removed so that it can be set to False later
                kw_to_remove.append(kw)
            if kw.arg == "hyperparameters" and kw.value:
                base_hps = dict(zip(kw.value.keys, kw.value.values))
                kw_to_remove.append(kw)
            if kw.arg in LEGACY_MODE_PARAMETERS and kw.value:
                hp_key = self._hyperparameter_key_for_param(kw.arg)
                additional_hps[hp_key] = kw.value
                kw_to_remove.append(kw)
            if kw.arg == "image_name":
                add_image_uri = False

        self._remove_keywords(node, kw_to_remove)
        self._add_updated_hyperparameters(node, base_hps, additional_hps)

        if add_image_uri:
            image_uri = self._image_uri_from_args(node.keywords)
            node.keywords.append(ast.keyword(arg="image_name", value=ast.Str(s=image_uri)))

        node.keywords.append(ast.keyword(arg="model_dir", value=ast.NameConstant(value=False)))

    def _hyperparameter_key_for_param(self, arg):
        """Returns an ``ast.Str`` for a hyperparameter key replacing a legacy mode parameter."""
        name = "sagemaker_requirements" if arg == "requirements_file" else arg
        return ast.Str(s=name)

    def _remove_keywords(self, node, keywords):
        """Removes the keywords from the ``ast.Call`` node."""
        for kw in keywords:
            node.keywords.remove(kw)

    def _add_updated_hyperparameters(self, node, base_hps, additional_hps):
        """Combines and adds the hyperparameters to the ``ast.Call`` node's keywords."""
        base_hps.update(additional_hps)
        updated_hp_keyword = self._to_ast_keyword(base_hps)

        if updated_hp_keyword:
            node.keywords.append(updated_hp_keyword)

    def _to_ast_keyword(self, hps):
        """Returns an ``ast.keyword`` for the ``hyperparameters`` kwarg if there are any."""
        if hps:
            keys, values = zip(*six.iteritems(hps))
            return ast.keyword(arg="hyperparameters", value=ast.Dict(keys=keys, values=values))

        return None

    def _image_uri_from_args(self, keywords):
        """Returns a legacy TensorFlow image URI based on the estimator arguments."""
        tf_version = framework_version.FRAMEWORK_DEFAULTS["TensorFlow"]
        instance_type = "ml.m4.xlarge"  # CPU default (exact type doesn't matter)

        for kw in keywords:
            if kw.arg == "framework_version":
                tf_version = kw.value.s
            if kw.arg == "train_instance_type":
                instance_type = kw.value.s

        return fw_utils.create_image_uri(
            self.region, "tensorflow", instance_type, tf_version, "py2"
        )


class TensorBoardParameterRemover(Modifier):
    """A class for removing the ``run_tensorboard_locally`` parameter from ``fit()``."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node invokes a function named "fit" and
        contains a keyword argument named "run_tensorboard_locally".

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is invoking a function named "fit" with
                a parameter named "run_tensorboard_locally".
        """
        is_fit_call = isinstance(node.func, ast.Attribute) and node.func.attr == "fit"
        if is_fit_call:
            for kw in node.keywords:
                if kw.arg == "run_tensorboard_locally":
                    return True

        return False

    def modify_node(self, node):
        """Removes ``run_tensorboard_locally`` from the ``ast.Call`` node's keywords.

        Args:
            node (ast.Call): a node that represents ``fit`` being called with
                ``run_tensorboard_locally`` set.
        """
        for kw in node.keywords:
            if kw.arg == "run_tensorboard_locally":
                node.keywords.remove(kw)
