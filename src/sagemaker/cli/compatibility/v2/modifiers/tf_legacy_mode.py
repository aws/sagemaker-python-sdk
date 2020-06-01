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
"""Classes to modify TensorFlow legacy mode code to be compatible with SageMaker Python SDK v2."""
# TODO: handle fit(run_tensorboard_locally=True)
from __future__ import absolute_import

import ast

import six

from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier


class TensorFlowLegacyModeConstructorUpgrader(Modifier):
    """A class to turn legacy mode parameters into hyperparameters when
    instantiating a TensorFlow estimator.
    """

    LEGACY_MODE_PARAMETERS = (
        "checkpoint_path",
        "evaluation_steps",
        "requirements_file",
        "training_steps",
    )

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a TensorFlow estimator with legacy mode.

        This looks for the following formats:

        - ``TensorFlow``
        - ``sagemaker.tensorflow.TensorFlow``

        Legacy mode is enabled if (1) ``script_mode`` is ``False``, ``None``, or not specified,
        and (2) if ``py_version`` is ``py2`` or not specified.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is instantiating a TensorFlow estimator with legacy mode.
        """
        return self._is_tf_constructor(node) and self._is_legacy_mode(node)

    def _is_tf_constructor(self, node):
        """Checks if the ``ast.Call`` node represents a call of the form
        ``TensorFlow`` or ``sagemaker.tensorflow.TensorFlow``.
        """
        # Check for TensorFlow()
        if isinstance(node.func, ast.Name):
            return node.func.id == "TensorFlow"

        # Check for sagemaker.tensorflow.TensorFlow()
        ends_with_tensorflow_constructor = (
            isinstance(node.func, ast.Attribute) and node.func.attr == "TensorFlow"
        )

        is_in_tensorflow_module = (
            isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "tensorflow"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "sagemaker"
        )

        return ends_with_tensorflow_constructor and is_in_tensorflow_module

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
        into hyperparameters and set ``script_mode=False``.

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

        for kw in node.keywords:
            if kw.arg == "script_mode":
                # remove here because is set to False later regardless of current value
                kw_to_remove.append(kw)
            if kw.arg == "hyperparameters" and kw.value:
                base_hps = dict(zip(kw.value.keys, kw.value.values))
                kw_to_remove.append(kw)
            if kw.arg in self.LEGACY_MODE_PARAMETERS and kw.value:
                hp_key = self._hyperparameter_key_for_param(kw.arg)
                additional_hps[hp_key] = kw.value
                kw_to_remove.append(kw)

        self._remove_keywords(node, kw_to_remove)
        self._add_updated_hyperparameters(node, base_hps, additional_hps)

        node.keywords.append(ast.keyword(arg="script_mode", value=ast.NameConstant(value=False)))

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
