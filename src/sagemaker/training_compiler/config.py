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
"""Configuration for the SageMaker Training Compiler."""
from __future__ import absolute_import
import logging

logger = logging.getLogger(__name__)


class TrainingCompilerConfig(object):
    """The configuration class for accelerating SageMaker training jobs through compilation.

    SageMaker Training Compiler speeds up training by optimizing the model execution graph.

    """

    DEBUG_PATH = "/opt/ml/output/data/compiler/"
    SUPPORTED_INSTANCE_CLASS_PREFIXES = ["p3", "g4dn", "p4"]

    HP_ENABLE_COMPILER = "sagemaker_training_compiler_enabled"
    HP_ENABLE_DEBUG = "sagemaker_training_compiler_debug_mode"

    def __init__(
        self,
        enabled=True,
        debug=False,
    ):
        """This class initializes a ``TrainingCompilerConfig`` instance.

        Pass the output of it to the ``compiler_config``
        parameter of the :class:`~sagemaker.huggingface.HuggingFace`
        class.

        Args:
            enabled (bool): Optional. Switch to enable SageMaker Training Compiler.
                The default is ``True``.
            debug (bool): Optional. Whether to dump detailed logs for debugging.
                This comes with a potential performance slowdown.
                The default is ``False``.

        **Example**: The following example shows the basic ``compiler_config``
        parameter configuration, enabling compilation with default parameter values.

        .. code-block:: python

            from sagemaker.huggingface import TrainingCompilerConfig
            compiler_config = TrainingCompilerConfig()

        """

        self.enabled = enabled
        self.debug = debug

        self.disclaimers_and_warnings()

    def __nonzero__(self):
        """Evaluates to 0 if SM Training Compiler is disabled."""
        return self.enabled

    def disclaimers_and_warnings(self):
        """Disclaimers and warnings.

        Logs disclaimers and warnings about the
        requested configuration of SageMaker Training Compiler.

        """

        if self.enabled and self.debug:
            logger.warning(
                "Debugging is enabled."
                "This will dump detailed logs from compilation to %s"
                "This might impair training performance.",
                self.DEBUG_PATH,
            )

    def _to_hyperparameter_dict(self):
        """Converts configuration object into hyperparameters.

        Returns:
            dict: A portion of the hyperparameters passed to the training job as a dictionary.

        """

        compiler_config_hyperparameters = {
            self.HP_ENABLE_COMPILER: self.enabled,
            self.HP_ENABLE_DEBUG: self.debug,
        }

        return compiler_config_hyperparameters

    @classmethod
    def validate(
        cls,
        image_uri,
        instance_type,
        distribution,
    ):
        """Checks if SageMaker Training Compiler is configured correctly.

        Args:
            image_uri (str): A string of a Docker image URI that's specified
                to :class:`~sagemaker.huggingface.HuggingFace`.
                If SageMaker Training Compiler is enabled, the HuggingFace estimator
                automatically chooses the right image URI. You cannot specify and override
                the image URI.
            instance_type (str): A string of the training instance type that's specified
                to :class:`~sagemaker.huggingface.HuggingFace`.
                The `validate` classmethod raises error
                if an instance type not in the ``SUPPORTED_INSTANCE_CLASS_PREFIXES`` list
                or ``local`` is passed to the `instance_type` parameter.
            distribution (dict): A dictionary of the distributed training option that's specified
                to :class:`~sagemaker.huggingface.HuggingFace`.
                SageMaker's distributed data parallel and model parallel libraries
                are currently not compatible
                with SageMaker Training Compiler.

        Raises:
            ValueError: Raised if the requested configuration is not compatible
                        with SageMaker Training Compiler.
        """

        if "local" not in instance_type:
            requested_instance_class = instance_type.split(".")[1]  # Expecting ml.class.size
            if not any(
                [
                    requested_instance_class.startswith(i)
                    for i in cls.SUPPORTED_INSTANCE_CLASS_PREFIXES
                ]
            ):
                error_helper_string = (
                    "Unsupported Instance class {}. SageMaker Training Compiler only supports {}"
                )
                error_helper_string = error_helper_string.format(
                    requested_instance_class, cls.SUPPORTED_INSTANCE_CLASS_PREFIXES
                )
                raise ValueError(error_helper_string)
        elif instance_type == "local":
            error_helper_string = (
                "The local mode is not supported by SageMaker Training Compiler."
                "It only supports the following GPU instances: p3, g4dn, and p4."
            )
            raise ValueError(error_helper_string)

        if image_uri:
            error_helper_string = (
                "Overriding the image URI is currently not supported "
                "for SageMaker Training Compiler."
                "Specify the following parameters to run the Hugging Face training job "
                "with SageMaker Training Compiler enabled: "
                "transformer_version, tensorflow_version or pytorch_version, and compiler_config."
            )
            raise ValueError(error_helper_string)

        if distribution and "smdistributed" in distribution:
            raise ValueError(
                "SageMaker distributed training configuration is currently not compatible with "
                "SageMaker Training Compiler."
            )
