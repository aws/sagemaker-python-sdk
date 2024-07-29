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
"""This module stores JumpStart implementation of Model class."""

from __future__ import absolute_import

from typing import Dict, List, Optional, Any, Union
import pandas as pd
from botocore.exceptions import ClientError

from sagemaker import payloads
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.enums import EndpointType
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.jumpstart.accessors import JumpStartModelsAccessor
from sagemaker.jumpstart.hub.utils import generate_hub_arn_for_init_kwargs
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.exceptions import (
    INVALID_MODEL_ID_ERROR_MSG,
    get_proprietary_model_subscription_error,
    get_proprietary_model_subscription_msg,
)
from sagemaker.jumpstart.factory.model import (
    get_default_predictor,
    get_deploy_kwargs,
    get_init_kwargs,
    get_register_kwargs,
)
from sagemaker.jumpstart.session_utils import get_model_info_from_endpoint
from sagemaker.jumpstart.types import (
    JumpStartSerializablePayload,
    DeploymentConfigMetadata,
)
from sagemaker.jumpstart.utils import (
    validate_model_id_and_get_type,
    verify_model_region_and_return_specs,
    get_jumpstart_configs,
    get_metrics_from_deployment_configs,
    add_instance_rate_stats_to_benchmark_metrics,
    deployment_config_response_data,
    _deployment_config_lru_cache,
)
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION, JUMPSTART_LOGGER
from sagemaker.jumpstart.enums import JumpStartModelType
from sagemaker.model_card import (
    ModelCard,
    ModelPackageModelCard,
)
from sagemaker.utils import stringify_object, format_tags, Tags
from sagemaker.model import (
    Model,
    ModelPackage,
)
from sagemaker.model_monitor.data_capture_config import DataCaptureConfig
from sagemaker.predictor import PredictorBase
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.session import Session
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.model_metrics import ModelMetrics
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements


class JumpStartModel(Model):
    """JumpStartModel class.

    This class sets defaults based on the model ID and version.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        hub_name: Optional[str] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        region: Optional[str] = None,
        instance_type: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        model_data: Optional[Union[str, PipelineVariable, dict]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None,
        sagemaker_session: Optional[Session] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Optional[Union[int, PipelineVariable]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        model_package_arn: Optional[str] = None,
        resources: Optional[ResourceRequirements] = None,
        config_name: Optional[str] = None,
        additional_model_data_sources: Optional[Dict[str, Any]] = None,
    ):
        """Initializes a ``JumpStartModel``.

        This method sets model-specific defaults for the ``Model.__init__`` method.

        Only model ID is required to instantiate this class, however any field can be overriden.

        Any field set to ``None`` does not get passed to the parent class method.

        Args:
            model_id (Optional[str]): JumpStart model ID to use. See
                https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html
                for list of model IDs.
            model_version (Optional[str]): Version for JumpStart model to use (Default: None).
            hub_name (Optional[str]): Hub name or arn where the model is stored (Default: None).
            tolerate_vulnerable_model (Optional[bool]): True if vulnerable versions of model
                specifications should be tolerated (exception not raised). If False, raises an
                exception if the script used by this version of the model has dependencies with
                known security vulnerabilities. (Default: None).
            tolerate_deprecated_model (Optional[bool]): True if deprecated models should be
                tolerated (exception not raised). False if these models should raise an exception.
                (Default: None).
            region (Optional[str]): The AWS region in which to launch the model. (Default: None).
            instance_type (Optional[str]): The EC2 instance type to use when provisioning a hosting
                endpoint. (Default: None).
            image_uri (Optional[Union[str, PipelineVariable]]): A Docker image URI. (Default: None).
            model_data (Optional[Union[str, PipelineVariable, dict]]): Location
                of SageMaker model data. (Default: None).
            role (Optional[str]): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role if it needs to access some AWS resources.
                It can be null if this is being used to create a Model to pass
                to a ``PipelineModel`` which has its own Role field. (Default:
                None).
            predictor_cls (Optional[callable[string, sagemaker.session.Session]]): A
                function to call to create a predictor (Default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name. (Default: None).
            env (Optional[dict[str, str] or dict[str, PipelineVariable]]): Environment variables
                to run with ``image_uri`` when hosted in SageMaker. (Default: None).
            name (Optional[str]): The model name. If None, a default model name will be
                selected on each ``deploy``. (Default: None).
            vpc_config (Optional[Union[dict[str, list[str]],dict[str, list[PipelineVariable]]]]):
                The VpcConfig set on the model (Default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids. (Default: None).
            sagemaker_session (Optional[sagemaker.session.Session]): A SageMaker Session
                object, used for SageMaker interactions (Default: None). If not
                specified, one is created using the default AWS configuration
                chain. (Default: None).
            enable_network_isolation (Optional[Union[bool, PipelineVariable]]): If True,
                enables network isolation in the endpoint, isolating the model
                container. No inbound or outbound network calls can be made to
                or from the model container. (Default: None).
            model_kms_key (Optional[str]): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked. (Default: None).
            image_config (Optional[Union[dict[str, str], dict[str, PipelineVariable]]]): Specifies
                whether the image of model container is pulled from ECR, or private
                registry in your VPC. By default it is set to pull model container
                image from ECR. (Default: None).
            source_dir (Optional[str]): The absolute, relative, or S3 URI Path to a directory
                with any other training source code dependencies aside from the entry
                point file (Default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory is preserved
                when training on Amazon SageMaker. If 'git_config' is provided,
                'source_dir' should be a relative location to a directory in the Git repo.
                If the directory points to S3, no code is uploaded and the S3 location
                is used instead. (Default: None).

                .. admonition:: Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='inference.py', source_dir='src'.
            code_location (Optional[str]): Name of the S3 bucket where custom code is
                uploaded (Default: None). If not specified, the default bucket
                created by ``sagemaker.session.Session`` is used. (Default: None).
            entry_point (Optional[str]): The absolute or relative path to the local Python
                source file that should be executed as the entry point to
                model hosting. (Default: None). If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo. (Default: None).

                .. admonition:: Example
                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='src/inference.py'.
            container_log_level (Optional[Union[int, PipelineVariable]]): Log level to use
                within the container. Valid values are defined in the Python
                logging module. (Default: None).
            dependencies (Optional[list[str]]): A list of absolute or relative paths to directories
                with any additional libraries that should be exported
                to the container (default: []). The library folders are
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo. If the ```source_dir``` points
                to S3, code will be uploaded and the S3 location will be used
                instead. This is not supported with "local code" in Local Mode.
                (Default: None).

                .. admonition:: Example

                    The following call

                    >>> Model(entry_point='inference.py',
                    ...       dependencies=['my/libs/common', 'virtual-env'])

                    results in the following structure inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ inference.py
                    >>>     |------ common
                    >>>     |------ virtual-env
            git_config (Optional[dict[str, str]]): Git configurations used for cloning
                files, including ``repo``, ``branch``, ``commit``,
                ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                ``repo`` field is required. All other fields are optional.
                ``repo`` specifies the Git repository where your training script
                is stored. If you don't provide ``branch``, the default value
                'master' is used. If you don't provide ``commit``, the latest
                commit in the specified branch is used.

                ``2FA_enabled``, ``username``, ``password`` and ``token`` are
                used for authentication. For GitHub (or other Git) accounts, set
                ``2FA_enabled`` to 'True' if two-factor authentication is
                enabled for the account, otherwise set it to 'False'. If you do
                not provide a value for ``2FA_enabled``, a default value of
                'False' is used. CodeCommit does not support two-factor
                authentication, so do not provide "2FA_enabled" with CodeCommit
                repositories.

                For GitHub and other Git repos, when SSH URLs are provided, it
                doesn't matter whether 2FA is enabled or disabled. You should
                either have no passphrase for the SSH key pairs or have the
                ssh-agent configured so that you will not be prompted for the SSH
                passphrase when you run the 'git clone' command with SSH URLs. When
                HTTPS URLs are provided, if 2FA is disabled, then either ``token``
                or ``username`` and ``password`` are be used for authentication if provided.
                ``Token`` is prioritized. If 2FA is enabled, only ``token`` is used
                for authentication if provided. If required authentication info
                is not provided, the SageMaker Python SDK attempts to use local credentials
                to authenticate. If that fails, an error message is thrown.

                For CodeCommit repos, 2FA is not supported, so ``2FA_enabled``
                should not be provided. There is no token in CodeCommit, so
                ``token`` should also not be provided. When ``repo`` is an SSH URL,
                the requirements are the same as GitHub  repos. When ``repo``
                is an HTTPS URL, ``username`` and ``password`` are used for
                authentication if they are provided. If they are not provided,
                the SageMaker Python SDK attempts to use either the CodeCommit
                credential helper or local credential storage for authentication.
                (Default: None).

                .. admonition:: Example

                    The following config results in cloning the repo specified in 'repo', then
                    checking out the 'master' branch, and checking out the specified
                    commit.

                    >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                    >>>               'branch': 'test-branch-git-config',
                    >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

            model_package_arn (Optional[str]): An existing SageMaker Model Package arn,
                can be just the name if your account owns the Model Package.
                ``model_data`` is not required. (Default: None).
            resources (Optional[ResourceRequirements]): The compute resource requirements
                for a model to be deployed to an endpoint.
                Only EndpointType.INFERENCE_COMPONENT_BASED supports this feature.
                (Default: None).
            config_name (Optional[str]): The name of the JumpStart config that can be
                optionally applied to the model.
            additional_model_data_sources (Optional[Dict[str, Any]]): Additional location
                of SageMaker model data (default: None).
        Raises:
            ValueError: If the model ID is not recognized by JumpStart.
        """

        hub_arn = None
        if hub_name:
            hub_arn = generate_hub_arn_for_init_kwargs(
                hub_name=hub_name, region=region, session=sagemaker_session
            )

        def _validate_model_id_and_type():
            return validate_model_id_and_get_type(
                model_id=model_id,
                model_version=model_version,
                region=region or getattr(sagemaker_session, "boto_region_name", None),
                script=JumpStartScriptScope.INFERENCE,
                sagemaker_session=sagemaker_session,
                hub_arn=hub_arn,
            )

        self.model_type = _validate_model_id_and_type()
        if not self.model_type:
            JumpStartModelsAccessor.reset_cache()
            self.model_type = _validate_model_id_and_type()
            if not self.model_type and not hub_arn:
                raise ValueError(INVALID_MODEL_ID_ERROR_MSG.format(model_id=model_id))

        self._model_data_is_set = model_data is not None
        model_init_kwargs = get_init_kwargs(
            model_id=model_id,
            model_from_estimator=False,
            model_type=self.model_type,
            model_version=model_version,
            hub_arn=hub_arn,
            instance_type=instance_type,
            tolerate_vulnerable_model=tolerate_vulnerable_model,
            tolerate_deprecated_model=tolerate_deprecated_model,
            region=region,
            image_uri=image_uri,
            model_data=model_data,
            source_dir=source_dir,
            entry_point=entry_point,
            env=env,
            predictor_cls=predictor_cls,
            role=role,
            name=name,
            vpc_config=vpc_config,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=enable_network_isolation,
            model_kms_key=model_kms_key,
            image_config=image_config,
            code_location=code_location,
            container_log_level=container_log_level,
            dependencies=dependencies,
            git_config=git_config,
            model_package_arn=model_package_arn,
            resources=resources,
            config_name=config_name,
            additional_model_data_sources=additional_model_data_sources,
        )

        self.orig_predictor_cls = predictor_cls

        self.model_id = model_init_kwargs.model_id
        self.model_version = model_init_kwargs.model_version
        self.hub_arn = model_init_kwargs.hub_arn
        self.instance_type = model_init_kwargs.instance_type
        self.resources = model_init_kwargs.resources
        self.tolerate_vulnerable_model = model_init_kwargs.tolerate_vulnerable_model
        self.tolerate_deprecated_model = model_init_kwargs.tolerate_deprecated_model
        self.region = model_init_kwargs.region
        self.sagemaker_session = model_init_kwargs.sagemaker_session
        self.role = role
        self.config_name = model_init_kwargs.config_name
        self.additional_model_data_sources = model_init_kwargs.additional_model_data_sources
        self.model_reference_arn = model_init_kwargs.model_reference_arn

        if self.model_type == JumpStartModelType.PROPRIETARY:
            self.log_subscription_warning()

        model_init_kwargs_dict = model_init_kwargs.to_kwargs_dict()

        super(JumpStartModel, self).__init__(**model_init_kwargs_dict)

        self.model_package_arn = model_init_kwargs.model_package_arn
        self.init_kwargs = model_init_kwargs.to_kwargs_dict(False)

        self._metadata_configs = get_jumpstart_configs(
            region=self.region,
            model_id=self.model_id,
            model_version=self.model_version,
            sagemaker_session=self.sagemaker_session,
            model_type=self.model_type,
        )

    def log_subscription_warning(self) -> None:
        """Log message prompting the customer to subscribe to the proprietary model."""
        subscription_link = verify_model_region_and_return_specs(
            region=self.region,
            model_id=self.model_id,
            version=self.model_version,
            hub_arn=self.hub_arn,
            model_type=self.model_type,
            scope=JumpStartScriptScope.INFERENCE,
            sagemaker_session=self.sagemaker_session,
        ).model_subscription_link
        JUMPSTART_LOGGER.warning(
            get_proprietary_model_subscription_msg(self.model_id, subscription_link)
        )

    def retrieve_all_examples(self) -> Optional[List[JumpStartSerializablePayload]]:
        """Returns all example payloads associated with the model.

        Raises:
            NotImplementedError: If the scope is not supported.
            ValueError: If the combination of arguments specified is not supported.
            VulnerableJumpStartModelError: If any of the dependencies required by the script have
                known security vulnerabilities.
            DeprecatedJumpStartModelError: If the version of the model is deprecated.
        """
        return payloads.retrieve_all_examples(
            model_id=self.model_id,
            model_version=self.model_version,
            hub_arn=self.hub_arn,
            region=self.region,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            sagemaker_session=self.sagemaker_session,
            model_type=self.model_type,
        )

    def retrieve_example_payload(self) -> JumpStartSerializablePayload:
        """Returns the example payload associated with the model.

        Payload can be directly used with the `sagemaker.predictor.Predictor.predict(...)` function.

        Raises:
            NotImplementedError: If the scope is not supported.
            ValueError: If the combination of arguments specified is not supported.
            VulnerableJumpStartModelError: If any of the dependencies required by the script have
                known security vulnerabilities.
            DeprecatedJumpStartModelError: If the version of the model is deprecated.
        """
        return payloads.retrieve_example(
            model_id=self.model_id,
            model_version=self.model_version,
            model_type=self.model_type,
            region=self.region,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            sagemaker_session=self.sagemaker_session,
        )

    def set_deployment_config(self, config_name: str, instance_type: str) -> None:
        """Sets the deployment config to apply to the model.

        Args:
            config_name (str):
                The name of the deployment config to apply to the model.
                Call list_deployment_configs to see the list of config names.
            instance_type (str):
                The instance_type that the model will use after setting
                the config.
        """
        self.__init__(
            model_id=self.model_id,
            model_version=self.model_version,
            instance_type=instance_type,
            config_name=config_name,
            sagemaker_session=self.sagemaker_session,
            role=self.role,
        )

    @property
    def deployment_config(self) -> Optional[Dict[str, Any]]:
        """The deployment config that will be applied to ``This`` model.

        Returns:
            Optional[Dict[str, Any]]: Deployment config.
        """
        if self.config_name is None:
            return None
        for config in self.list_deployment_configs():
            if config.get("DeploymentConfigName") == self.config_name:
                return config
        return None

    @property
    def benchmark_metrics(self) -> pd.DataFrame:
        """Benchmark Metrics for deployment configs.

        Returns:
            Benchmark Metrics: Pandas DataFrame object.
        """
        df = pd.DataFrame(self._get_deployment_configs_benchmarks_data())
        blank_index = [""] * len(df)
        df.index = blank_index
        return df

    def display_benchmark_metrics(self, **kwargs) -> None:
        """Display deployment configs benchmark metrics."""
        df = self.benchmark_metrics

        instance_type = kwargs.get("instance_type")
        if instance_type:
            df = df[df["Instance Type"].str.contains(instance_type)]

        print(df.to_markdown(index=False, floatfmt=".2f"))

    def list_deployment_configs(self) -> List[Dict[str, Any]]:
        """List deployment configs for ``This`` model.

        Returns:
            List[Dict[str, Any]]: A list of deployment configs.
        """
        return deployment_config_response_data(
            self._get_deployment_configs(self.config_name, self.instance_type)
        )

    @classmethod
    def attach(
        cls,
        endpoint_name: str,
        inference_component_name: Optional[str] = None,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        sagemaker_session=DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
        hub_name: Optional[str] = None,
    ) -> "JumpStartModel":
        """Attaches a JumpStartModel object to an existing SageMaker Endpoint.

        The model id, version (and inference component name) can be inferred from the tags.
        """

        inferred_model_id = inferred_model_version = inferred_inference_component_name = None

        if inference_component_name is None or model_id is None or model_version is None:
            inferred_model_id, inferred_model_version, inferred_inference_component_name, _, _ = (
                get_model_info_from_endpoint(
                    endpoint_name=endpoint_name,
                    inference_component_name=inference_component_name,
                    sagemaker_session=sagemaker_session,
                )
            )

        model_id = model_id or inferred_model_id
        model_version = model_version or inferred_model_version or "*"
        inference_component_name = inference_component_name or inferred_inference_component_name

        model = JumpStartModel(
            model_id=model_id,
            model_version=model_version,
            sagemaker_session=sagemaker_session,
            hub_name=hub_name,
        )
        model.endpoint_name = endpoint_name
        model.inference_component_name = inference_component_name

        return model

    def _create_sagemaker_model(
        self,
        instance_type=None,
        accelerator_type=None,
        tags=None,
        serverless_inference_config=None,
        **kwargs,
    ):
        """Create a SageMaker Model Entity

        Args:
            instance_type (str): Optional. The EC2 instance type that this Model will be
                used for, this is only used to determine if the image needs GPU
                support or not. (Default: None).
            accelerator_type (str): Optional. Type of Elastic Inference accelerator to
                attach to an endpoint for model loading and inference, for
                example, 'ml.eia1.medium'. If not specified, no Elastic
                Inference accelerator will be attached to the endpoint. (Default: None).
            tags (Optional[Tags]): Optional. The list of tags to add to
                the model. Example: >>> tags = [{'Key': 'tagname', 'Value':
                'tagvalue'}] For more information about tags, see
                https://boto3.amazonaws.com/v1/documentation
                /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
                (Default: None).
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Optional. Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.
                (Default: None).
            kwargs: Keyword arguments coming from the caller. This class does not require
                any so they are ignored.
        """

        tags = format_tags(tags)

        # if the user inputs a model artifact uri, do not use model package arn to create
        # inference endpoint.
        if self.model_package_arn and not self._model_data_is_set:
            # When a ModelPackageArn is provided we just create the Model
            model_package = ModelPackage(
                role=self.role,
                model_data=self.model_data,
                model_package_arn=self.model_package_arn,
                sagemaker_session=self.sagemaker_session,
                predictor_cls=self.predictor_cls,
                vpc_config=self.vpc_config,
            )
            if self.name is not None:
                model_package.name = self.name
            if self.env is not None:
                model_package.env = self.env
            model_package._create_sagemaker_model(
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                tags=tags,
                serverless_inference_config=serverless_inference_config,
                **kwargs,
            )
            if self._base_name is None and model_package._base_name is not None:
                self._base_name = model_package._base_name
            if self.name is None and model_package.name is not None:
                self.name = model_package.name
        else:
            super(JumpStartModel, self)._create_sagemaker_model(
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                tags=tags,
                serverless_inference_config=serverless_inference_config,
                **kwargs,
            )

    def deploy(
        self,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        serializer: Optional[BaseSerializer] = None,
        deserializer: Optional[BaseDeserializer] = None,
        accelerator_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        inference_component_name: Optional[str] = None,
        tags: Optional[Tags] = None,
        kms_key: Optional[str] = None,
        wait: Optional[bool] = True,
        data_capture_config: Optional[DataCaptureConfig] = None,
        async_inference_config: Optional[AsyncInferenceConfig] = None,
        serverless_inference_config: Optional[ServerlessInferenceConfig] = None,
        volume_size: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        inference_recommendation_id: Optional[str] = None,
        explainer_config: Optional[ExplainerConfig] = None,
        accept_eula: Optional[bool] = None,
        endpoint_logging: Optional[bool] = False,
        resources: Optional[ResourceRequirements] = None,
        managed_instance_scaling: Optional[str] = None,
        endpoint_type: EndpointType = EndpointType.MODEL_BASED,
        routing_config: Optional[Dict[str, Any]] = None,
    ) -> PredictorBase:
        """Creates endpoint by calling base ``Model`` class `deploy` method.

        Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an
        ``Endpoint`` from this ``Model``.

        Any field set to ``None`` does not get passed to the parent class method.


        Args:
            initial_instance_count (Optional[int]): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. If not using
                serverless inference or the model has not called ``right_size()``,
                then it need to be a number larger or equals
                to 1. (Default: None)
            instance_type (Optional[str]): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference or the model has not called ``right_size()``,
                then it is required to deploy a model.
                (Default: None)
            serializer (Optional[:class:`~sagemaker.serializers.BaseSerializer`]): A
                serializer object, used to encode data for an inference endpoint
                (Default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``. (Default: None).
            deserializer (Optional[:class:`~sagemaker.deserializers.BaseDeserializer`]): A
                deserializer object, used to decode data from an inference
                endpoint (Default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``. (Default: None).
            accelerator_type (Optional[str]): Type of Elastic Inference accelerator to
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
                (Default: None).
            endpoint_name (Optional[str]): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
                (Default: None).
            tags (Optional[Tags]): Tags to attach to this
                specific endpoint. (Default: None).
            kms_key (Optional[str]): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint. (Default: None).
            wait (Optional[bool]): Whether the call should wait until the deployment of
                this model completes. (Default: True).
            data_capture_config (Optional[sagemaker.model_monitor.DataCaptureConfig]): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. (Default: None).
            async_inference_config (Optional[sagemaker.model_monitor.AsyncInferenceConfig]):
                Specifies configuration related to async endpoint. Use this configuration when
                trying to create async endpoint and make async inference. If empty config object
                passed through, will use default config to deploy async endpoint. Deploy a
                real-time endpoint if it's None. (Default: None)
            serverless_inference_config (Optional[sagemaker.serverless.ServerlessInferenceConfig]):
                Specifies configuration related to serverless endpoint. Use this configuration
                when trying to create serverless endpoint and make serverless inference. If
                empty object passed through, will use pre-defined values in
                ``ServerlessInferenceConfig`` class to deploy serverless endpoint. Deploy an
                instance based endpoint if it's None. (Default: None)
            volume_size (Optional[int]): The size, in GB, of the ML storage volume attached to
                individual inference instance associated with the production variant. Currenly only
                Amazon EBS gp2 storage volumes are supported. (Default: None).
            model_data_download_timeout (Optional[int]): The timeout value, in seconds, to download
                and extract model data from Amazon S3 to the individual inference instance
                associated with this production variant. (Default: None).
            container_startup_health_check_timeout (Optional[int]): The timeout value, in seconds,
                for your inference container to pass health check by SageMaker Hosting. For more
                information about health check see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-algo-ping-requests
                (Default: None).
            inference_recommendation_id (Optional[str]): The recommendation id which specifies the
                recommendation you picked from inference recommendation job results and
                would like to deploy the model and endpoint with recommended parameters.
                (Default: None).
            explainer_config (Optional[sagemaker.explainer.ExplainerConfig]): Specifies online
                explainability configuration for use with Amazon SageMaker Clarify. (Default: None).
            accept_eula (bool): For models that require a Model Access Config, specify True or
                False to indicate whether model terms of use have been accepted.
                The `accept_eula` value must be explicitly defined as `True` in order to
                accept the end-user license agreement (EULA) that some
                models require. (Default: None).
            endpoint_logging (Optiona[bool]): If set to true, live logging will be emitted as
                the SageMaker Endpoint starts up. (Default: False).
            resources (Optional[ResourceRequirements]): The compute resource requirements
                for a model to be deployed to an endpoint. Only
                EndpointType.INFERENCE_COMPONENT_BASED supports this feature.
                (Default: None).
            managed_instance_scaling (Optional[Dict]): Managed intance scaling options,
                if configured Amazon SageMaker will manage the instance number behind the
                endpoint.
            endpoint_type (EndpointType): The type of endpoint used to deploy models.
                (Default: EndpointType.MODEL_BASED).
            routing_config (Optional[Dict]): Settings the control how the endpoint routes
                incoming traffic to the instances that the endpoint hosts.

        Raises:
            MarketplaceModelSubscriptionError: If the caller is not subscribed to the model.
        """

        deploy_kwargs = get_deploy_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            hub_arn=self.hub_arn,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            initial_instance_count=initial_instance_count,
            instance_type=instance_type or self.instance_type,
            serializer=serializer,
            deserializer=deserializer,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
            inference_component_name=inference_component_name,
            tags=format_tags(tags),
            kms_key=kms_key,
            wait=wait,
            data_capture_config=data_capture_config,
            async_inference_config=async_inference_config,
            serverless_inference_config=serverless_inference_config,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
            inference_recommendation_id=inference_recommendation_id,
            explainer_config=explainer_config,
            sagemaker_session=self.sagemaker_session,
            accept_eula=accept_eula,
            model_reference_arn=self.model_reference_arn,
            endpoint_logging=endpoint_logging,
            resources=resources,
            managed_instance_scaling=managed_instance_scaling,
            endpoint_type=endpoint_type,
            model_type=self.model_type,
            config_name=self.config_name,
            routing_config=routing_config,
        )
        if (
            self.model_type == JumpStartModelType.PROPRIETARY
            and endpoint_type == EndpointType.INFERENCE_COMPONENT_BASED
        ):
            raise ValueError(
                f"{EndpointType.INFERENCE_COMPONENT_BASED} is not supported for Proprietary models."
            )

        try:
            predictor = super(JumpStartModel, self).deploy(**deploy_kwargs.to_kwargs_dict())
        except ClientError as e:
            subscription_link = verify_model_region_and_return_specs(
                region=self.region,
                model_id=self.model_id,
                version=self.model_version,
                model_type=self.model_type,
                scope=JumpStartScriptScope.INFERENCE,
                sagemaker_session=self.sagemaker_session,
                config_name=self.config_name,
                hub_arn=self.hub_arn,
            ).model_subscription_link
            get_proprietary_model_subscription_error(e, subscription_link)
            raise

        # If no predictor class was passed, add defaults to predictor
        if self.orig_predictor_cls is None and async_inference_config is None:
            return get_default_predictor(
                predictor=predictor,
                model_id=self.model_id,
                model_version=self.model_version,
                hub_arn=self.hub_arn,
                region=self.region,
                tolerate_deprecated_model=self.tolerate_deprecated_model,
                tolerate_vulnerable_model=self.tolerate_vulnerable_model,
                sagemaker_session=self.sagemaker_session,
                model_type=self.model_type,
                config_name=self.config_name,
            )

        # If a predictor class was passed, do not mutate predictor
        return predictor

    def register(
        self,
        content_types: List[Union[str, PipelineVariable]] = None,
        response_types: List[Union[str, PipelineVariable]] = None,
        inference_instances: Optional[List[Union[str, PipelineVariable]]] = None,
        transform_instances: Optional[List[Union[str, PipelineVariable]]] = None,
        model_package_group_name: Optional[Union[str, PipelineVariable]] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        model_metrics: Optional[ModelMetrics] = None,
        metadata_properties: Optional[MetadataProperties] = None,
        approval_status: Optional[Union[str, PipelineVariable]] = None,
        description: Optional[str] = None,
        drift_check_baselines: Optional[DriftCheckBaselines] = None,
        customer_metadata_properties: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        validation_specification: Optional[Union[str, PipelineVariable]] = None,
        domain: Optional[Union[str, PipelineVariable]] = None,
        task: Optional[Union[str, PipelineVariable]] = None,
        sample_payload_url: Optional[Union[str, PipelineVariable]] = None,
        framework: Optional[Union[str, PipelineVariable]] = None,
        framework_version: Optional[Union[str, PipelineVariable]] = None,
        nearest_model_name: Optional[Union[str, PipelineVariable]] = None,
        data_input_configuration: Optional[Union[str, PipelineVariable]] = None,
        skip_model_validation: Optional[Union[str, PipelineVariable]] = None,
        source_uri: Optional[Union[str, PipelineVariable]] = None,
        model_card: Optional[Union[ModelPackageModelCard, ModelCard]] = None,
        accept_eula: Optional[bool] = None,
    ):
        """Creates a model package for creating SageMaker models or listing on Marketplace.

        Args:
            content_types (list[str] or list[PipelineVariable]): The supported MIME types
                for the input data.
            response_types (list[str] or list[PipelineVariable]): The supported MIME types
                for the output data.
            inference_instances (list[str] or list[PipelineVariable]): A list of the instance
                types that are used to generate inferences in real-time (default: None).
            transform_instances (list[str] or list[PipelineVariable]): A list of the instance types
                on which a transformation job can be run or on which an endpoint can be deployed
                (default: None).
            model_package_group_name (str or PipelineVariable): Model Package Group name,
                exclusive to `model_package_name`, using `model_package_group_name` makes the
                Model Package versioned. Defaults to ``None``.
            image_uri (str or PipelineVariable): Inference image URI for the container. Model class'
                self.image will be used if it is None. Defaults to ``None``.
            model_metrics (ModelMetrics): ModelMetrics object. Defaults to ``None``.
            metadata_properties (MetadataProperties): MetadataProperties object.
                Defaults to ``None``.
            approval_status (str or PipelineVariable): Model Approval Status, values can be
                "Approved", "Rejected", or "PendingManualApproval". Defaults to
                ``PendingManualApproval``.
            description (str): Model Package description. Defaults to ``None``.
            drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
            customer_metadata_properties (dict[str, str] or dict[str, PipelineVariable]):
                A dictionary of key-value paired metadata properties (default: None).
            domain (str or PipelineVariable): Domain values can be "COMPUTER_VISION",
                "NATURAL_LANGUAGE_PROCESSING", "MACHINE_LEARNING" (default: None).
            sample_payload_url (str or PipelineVariable): The S3 path where the sample payload
                is stored (default: None).
            task (str or PipelineVariable): Task values which are supported by Inference Recommender
                are "FILL_MASK", "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION",
                "IMAGE_SEGMENTATION", "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
            framework (str or PipelineVariable): Machine learning framework of the model package
                container image (default: None).
            framework_version (str or PipelineVariable): Framework version of the Model Package
                Container Image (default: None).
            nearest_model_name (str or PipelineVariable): Name of a pre-trained machine learning
                benchmarked by Amazon SageMaker Inference Recommender (default: None).
            data_input_configuration (str or PipelineVariable): Input object for the model
                (default: None).
            skip_model_validation (str or PipelineVariable): Indicates if you want to skip model
                validation. Values can be "All" or "None" (default: None).
            source_uri (str or PipelineVariable): The URI of the source for the model package
                (default: None).
            model_card (ModeCard or ModelPackageModelCard): document contains qualitative and
                quantitative information about a model (default: None).
            accept_eula (bool): For models that require a Model Access Config, specify True or
                False to indicate whether model terms of use have been accepted.
                The `accept_eula` value must be explicitly defined as `True` in order to
                accept the end-user license agreement (EULA) that some
                models require. (Default: None).
        Returns:
            A `sagemaker.model.ModelPackage` instance.
        """

        if model_package_group_name is None:
            model_package_group_name = self.model_id
        if self.model_type is JumpStartModelType.PROPRIETARY:
            source_uri = self.model_package_arn

        register_kwargs = get_register_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            hub_arn=self.hub_arn,
            model_type=self.model_type,
            region=self.region,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            sagemaker_session=self.sagemaker_session,
            supported_content_types=content_types,
            response_types=response_types,
            inference_instances=inference_instances,
            transform_instances=transform_instances,
            model_package_group_name=model_package_group_name,
            image_uri=image_uri,
            model_metrics=model_metrics,
            metadata_properties=metadata_properties,
            approval_status=approval_status,
            description=description,
            drift_check_baselines=drift_check_baselines,
            customer_metadata_properties=customer_metadata_properties,
            validation_specification=validation_specification,
            domain=domain,
            task=task,
            sample_payload_url=sample_payload_url,
            framework=framework,
            framework_version=framework_version,
            nearest_model_name=nearest_model_name,
            data_input_configuration=data_input_configuration,
            skip_model_validation=skip_model_validation,
            source_uri=source_uri,
            config_name=self.config_name,
            model_card=model_card,
            accept_eula=accept_eula,
        )

        model_package = super(JumpStartModel, self).register(**register_kwargs.to_kwargs_dict())

        def register_deploy_wrapper(*args, **kwargs):
            if self.model_package_arn is not None:
                return self.deploy(*args, **kwargs)

            self.model_package_arn = model_package.model_package_arn
            predictor = self.deploy(*args, **kwargs)
            self.model_package_arn = None
            return predictor

        model_package.deploy = register_deploy_wrapper

        return model_package

    @_deployment_config_lru_cache
    def _get_deployment_configs_benchmarks_data(self) -> Dict[str, Any]:
        """Deployment configs benchmark metrics.

        Returns:
            Dict[str, List[str]]: Deployment config benchmark data.
        """
        return get_metrics_from_deployment_configs(
            self._get_deployment_configs(None, None),
        )

    @_deployment_config_lru_cache
    def _get_deployment_configs(
        self, selected_config_name: Optional[str], selected_instance_type: Optional[str]
    ) -> List[DeploymentConfigMetadata]:
        """Retrieve deployment configs metadata.

        Args:
            selected_config_name (Optional[str]): The name of the selected deployment config.
            selected_instance_type (Optional[str]): The selected instance type.
        """
        deployment_configs = []
        if not self._metadata_configs:
            return deployment_configs

        err = None
        for config_name, metadata_config in self._metadata_configs.items():
            if selected_config_name == config_name:
                instance_type_to_use = selected_instance_type
            else:
                instance_type_to_use = metadata_config.resolved_config.get(
                    "default_inference_instance_type"
                )

            if metadata_config.benchmark_metrics:
                err, metadata_config.benchmark_metrics = (
                    add_instance_rate_stats_to_benchmark_metrics(
                        self.region, metadata_config.benchmark_metrics
                    )
                )

            config_components = metadata_config.config_components.get(config_name)
            image_uri = (
                (
                    config_components.hosting_instance_type_variants.get("regional_aliases", {})
                    .get(self.region, {})
                    .get("alias_ecr_uri_1")
                )
                if config_components
                else self.image_uri
            )

            init_kwargs = get_init_kwargs(
                config_name=config_name,
                model_id=self.model_id,
                instance_type=instance_type_to_use,
                sagemaker_session=self.sagemaker_session,
                image_uri=image_uri,
                region=self.region,
                model_version=self.model_version,
            )
            deploy_kwargs = get_deploy_kwargs(
                model_id=self.model_id,
                instance_type=instance_type_to_use,
                sagemaker_session=self.sagemaker_session,
                region=self.region,
                model_version=self.model_version,
            )

            deployment_config_metadata = DeploymentConfigMetadata(
                config_name,
                metadata_config,
                init_kwargs,
                deploy_kwargs,
            )
            deployment_configs.append(deployment_config_metadata)

        if err and err["Code"] == "AccessDeniedException":
            error_message = "Instance rate metrics will be omitted. Reason: %s"
            JUMPSTART_LOGGER.warning(error_message, err["Message"])

        return deployment_configs

    def __str__(self) -> str:
        """Overriding str(*) method to make more human-readable."""
        return stringify_object(self)
