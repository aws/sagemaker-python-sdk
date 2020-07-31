# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Amazon SageMaker Debugger is a service that provides full visibility
into the training of machine learning (ML) models, enabling customers
to automatically detect several classes of errors. Customers can configure
Debugger when starting their training jobs by specifying debug level, models,
and location where debug output will be stored. Optionally, customers can
also specify custom error conditions that they want to be alerted on.
Debugger automatically collects model specific data, monitors for errors,
and alerts when it detects errors during training.
"""
from __future__ import absolute_import

import smdebug_rulesconfig as rule_configs  # noqa: F401 # pylint: disable=unused-import

from sagemaker import image_uris

framework_name = "debugger"


def get_rule_container_image_uri(region):
    """
    Returns the rule image uri for the given AWS region and rule type

    Args:
        region: AWS Region

    Returns:
        str: Formatted image uri for the given region and the rule container type
    """
    return image_uris.retrieve(framework_name, region)


class Rule(object):
    """Rules analyze tensors emitted during the training of a model. They
    monitor conditions that are critical for the success of a training job.

    For example, they can detect whether gradients are getting too large or
    too small or if a model is being overfit. Debugger comes pre-packaged with
    certain built-in rules (created using the Rule.sagemaker classmethod).
    You can use these rules or write your own rules using the Amazon SageMaker
    Debugger APIs. You can also analyze raw tensor data without using rules in,
    for example, an Amazon SageMaker notebook, using Debugger's full set of APIs.
    """

    def __init__(
        self,
        name,
        image_uri,
        instance_type,
        container_local_output_path,
        s3_output_path,
        volume_size_in_gb,
        rule_parameters,
        collections_to_save,
    ):
        """Do not use this initialization method. Instead, use either the
        ``Rule.sagemaker`` or ``Rule.custom`` class method.

        Initialize a ``Rule`` instance. The Rule analyzes tensors emitted
        during the training of a model and monitors conditions that are critical
        for the success of a training job.

        Args:
            name (str): The name of the debugger rule.
            image_uri (str): The URI of the image to be used by the debugger rule.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            container_local_output_path (str): The local path to store the Rule output.
            s3_output_path (str): The location in S3 to store the output.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data.
            rule_parameters (dict): A dictionary of parameters for the rule.
            collections_to_save ([sagemaker.debugger.CollectionConfig]): A list
                of CollectionConfig objects to be saved.
        """
        self.name = name
        self.instance_type = instance_type
        self.container_local_output_path = container_local_output_path
        self.s3_output_path = s3_output_path
        self.volume_size_in_gb = volume_size_in_gb
        self.rule_parameters = rule_parameters
        self.collection_configs = collections_to_save
        self.image_uri = image_uri

    @classmethod
    def sagemaker(
        cls,
        base_config,
        name=None,
        container_local_output_path=None,
        s3_output_path=None,
        other_trials_s3_input_paths=None,
        rule_parameters=None,
        collections_to_save=None,
    ):
        """Initialize a ``Rule`` instance for a built-in SageMaker Debugging
        Rule. The Rule analyzes tensors emitted during the training of a model
        and monitors conditions that are critical for the success of a training
        job.

        Args:
            base_config (dict): This is the base rule config returned from the
                built-in list of rules. For example, 'rule_configs.dead_relu()'.
            name (str): The name of the debugger rule. If one is not provided,
                the name of the base_config will be used.
            container_local_output_path (str): The path in the container.
            s3_output_path (str): The location in S3 to store the output.
            other_trials_s3_input_paths ([str]): S3 input paths for other trials.
            rule_parameters (dict): A dictionary of parameters for the rule.
            collections_to_save ([sagemaker.debugger.CollectionConfig]): A list
                of CollectionConfig objects to be saved.

        Returns:
            sagemaker.debugger.Rule: The instance of the built-in Rule.
        """
        merged_rule_params = {}

        if rule_parameters is not None and rule_parameters.get("rule_to_invoke") is not None:
            raise RuntimeError(
                """You cannot provide a 'rule_to_invoke' for SageMaker rules.
                Please either remove the rule_to_invoke or use a custom rule.
                """
            )

        if other_trials_s3_input_paths is not None:
            for index, s3_input_path in enumerate(other_trials_s3_input_paths):
                merged_rule_params["other_trial_{}".format(str(index))] = s3_input_path

        default_rule_params = base_config["DebugRuleConfiguration"].get("RuleParameters", {})
        merged_rule_params.update(default_rule_params)
        merged_rule_params.update(rule_parameters or {})

        base_config_collections = []
        for config in base_config.get("CollectionConfigurations", []):
            collection_name = None
            collection_parameters = {}
            for key, value in config.items():
                if key == "CollectionName":
                    collection_name = value
                if key == "CollectionParameters":
                    collection_parameters = value
            base_config_collections.append(
                CollectionConfig(name=collection_name, parameters=collection_parameters)
            )

        return cls(
            name=name or base_config["DebugRuleConfiguration"].get("RuleConfigurationName"),
            image_uri="DEFAULT_RULE_EVALUATOR_IMAGE",
            instance_type=None,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=None,
            rule_parameters=merged_rule_params,
            collections_to_save=collections_to_save or base_config_collections,
        )

    @classmethod
    def custom(
        cls,
        name,
        image_uri,
        instance_type,
        volume_size_in_gb,
        source=None,
        rule_to_invoke=None,
        container_local_output_path=None,
        s3_output_path=None,
        other_trials_s3_input_paths=None,
        rule_parameters=None,
        collections_to_save=None,
    ):
        """Initialize a ``Rule`` instance for a custom rule. The Rule
        analyzes tensors emitted during the training of a model and
        monitors conditions that are critical for the success of a
        training job.

        Args:
            name (str): The name of the debugger rule.
            image_uri (str): The URI of the image to be used by the debugger rule.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data.
            source (str): A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str): The name of the rule to invoke within the source.
                If provided, you must also provide source.
            container_local_output_path (str): The path in the container.
            s3_output_path (str): The location in S3 to store the output.
            other_trials_s3_input_paths ([str]): S3 input paths for other trials.
            rule_parameters (dict): A dictionary of parameters for the rule.
            collections_to_save ([sagemaker.debugger.CollectionConfig]): A list
                of CollectionConfig objects to be saved.

        Returns:
            sagemaker.debugger.Rule: The instance of the custom Rule.
        """
        if bool(source) ^ bool(rule_to_invoke):
            raise ValueError(
                "If you provide a source, you must also provide a rule to invoke (and vice versa)."
            )

        merged_rule_params = {}

        if source is not None and rule_to_invoke is not None:
            merged_rule_params["source_s3_uri"] = source
            merged_rule_params["rule_to_invoke"] = rule_to_invoke

        other_trials_params = {}
        if other_trials_s3_input_paths is not None:
            for index, s3_input_path in enumerate(other_trials_s3_input_paths):
                other_trials_params["other_trial_{}".format(str(index))] = s3_input_path

        merged_rule_params.update(other_trials_params)
        merged_rule_params.update(rule_parameters or {})

        return cls(
            name=name,
            image_uri=image_uri,
            instance_type=instance_type,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=volume_size_in_gb,
            rule_parameters=merged_rule_params,
            collections_to_save=collections_to_save or [],
        )

    def to_debugger_rule_config_dict(self):
        """Generates a request dictionary using the parameters provided
        when initializing the object.

        Returns:
            dict: An portion of an API request as a dictionary.
        """
        debugger_rule_config_request = {
            "RuleConfigurationName": self.name,
            "RuleEvaluatorImage": self.image_uri,
        }

        if self.instance_type is not None:
            debugger_rule_config_request["InstanceType"] = self.instance_type

        if self.volume_size_in_gb is not None:
            debugger_rule_config_request["VolumeSizeInGB"] = self.volume_size_in_gb

        if self.container_local_output_path is not None:
            debugger_rule_config_request["LocalPath"] = self.container_local_output_path

        if self.s3_output_path is not None:
            debugger_rule_config_request["S3OutputPath"] = self.s3_output_path

        if self.rule_parameters:
            debugger_rule_config_request["RuleParameters"] = self.rule_parameters

        return debugger_rule_config_request


class DebuggerHookConfig(object):
    """DebuggerHookConfig provides options to customize how debugging
    information is emitted.
    """

    def __init__(
        self,
        s3_output_path=None,
        container_local_output_path=None,
        hook_parameters=None,
        collection_configs=None,
    ):
        """Initialize an instance of ``DebuggerHookConfig``.
        DebuggerHookConfig provides options to customize how debugging
        information is emitted.

        Args:
            s3_output_path (str): The location in S3 to store the output.
            container_local_output_path (str): The path in the container.
            hook_parameters (dict): A dictionary of parameters.
            collection_configs ([sagemaker.debugger.CollectionConfig]): A list
                of CollectionConfig objects to be provided to the API.
        """
        self.s3_output_path = s3_output_path
        self.container_local_output_path = container_local_output_path
        self.hook_parameters = hook_parameters
        self.collection_configs = collection_configs

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided
        when initializing the object.

        Returns:
            dict: An portion of an API request as a dictionary.
        """
        debugger_hook_config_request = {"S3OutputPath": self.s3_output_path}

        if self.container_local_output_path is not None:
            debugger_hook_config_request["LocalPath"] = self.container_local_output_path

        if self.hook_parameters is not None:
            debugger_hook_config_request["HookParameters"] = self.hook_parameters

        if self.collection_configs is not None:
            debugger_hook_config_request["CollectionConfigurations"] = [
                collection_config._to_request_dict()
                for collection_config in self.collection_configs
            ]

        return debugger_hook_config_request


class TensorBoardOutputConfig(object):
    """TensorBoardOutputConfig provides options to customize
    debugging visualization using TensorBoard."""

    def __init__(self, s3_output_path, container_local_output_path=None):
        """Initialize an instance of TensorBoardOutputConfig.
        TensorBoardOutputConfig provides options to customize
        debugging visualization using TensorBoard.

        Args:
            s3_output_path (str): The location in S3 to store the output.
            container_local_output_path (str): The path in the container.
        """
        self.s3_output_path = s3_output_path
        self.container_local_output_path = container_local_output_path

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided
        when initializing the object.

        Returns:
            dict: An portion of an API request as a dictionary.
        """
        tensorboard_output_config_request = {"S3OutputPath": self.s3_output_path}

        if self.container_local_output_path is not None:
            tensorboard_output_config_request["LocalPath"] = self.container_local_output_path

        return tensorboard_output_config_request


class CollectionConfig(object):
    """CollectionConfig object for SageMaker Debugger."""

    def __init__(self, name, parameters=None):
        """Initialize a ``CollectionConfig`` object.

        Args:
            name (str): The name of the collection configuration.
            parameters (dict): The parameters for the collection
                configuration.
        """
        self.name = name
        self.parameters = parameters

    def __eq__(self, other):
        if not isinstance(other, CollectionConfig):
            raise TypeError(
                "CollectionConfig is only comparable with other CollectionConfig objects."
            )

        return self.name == other.name and self.parameters == other.parameters

    def __ne__(self, other):
        if not isinstance(other, CollectionConfig):
            raise TypeError(
                "CollectionConfig is only comparable with other CollectionConfig objects."
            )

        return self.name != other.name or self.parameters != other.parameters

    def __hash__(self):
        return hash((self.name, tuple(sorted((self.parameters or {}).items()))))

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided
        when initializing the object.

        Returns:
            dict: An portion of an API request as a dictionary.
        """
        collection_config_request = {"CollectionName": self.name}

        if self.parameters is not None:
            collection_config_request["CollectionParameters"] = self.parameters

        return collection_config_request
