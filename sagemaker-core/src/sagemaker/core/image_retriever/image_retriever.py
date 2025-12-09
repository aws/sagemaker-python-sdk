import re
from typing import Optional
from graphene.utils.str_converters import to_camel_case

from sagemaker.core.inference_config import ServerlessInferenceConfig
from sagemaker.core.training_compiler.config import TrainingCompilerConfig
from sagemaker.core.common_utils import _botocore_resolver
from sagemaker.core.workflow import is_pipeline_variable
from sagemaker.core.image_retriever.image_retriever_utils import (
    _config_for_framework_and_scope,
    _get_final_image_scope,
    _get_image_tag,
    _get_inference_tool,
    _get_latest_versions,
    _processor,
    _registry_from_region,
    _retrieve_latest_pytorch_training_uri,
    _retrieve_pytorch_uri_inputs_are_all_default,
    _validate_arg,
    _validate_for_suppported_frameworks_and_instance_type,
    _validate_instance_deprecation,
    _validate_py_version_and_set_if_needed,
    _validate_version_and_set_if_needed,
    _version_for_config,
    config_for_framework,
)
from sagemaker.core.workflow.utilities import override_pipeline_parameter_var
from sagemaker.core.config.config_schema import IMAGE_RETRIEVER, MODULES, SAGEMAKER, _simple_path
from sagemaker.core.config.config_manager import SageMakerConfig

ECR_URI_TEMPLATE = "{registry}.dkr.{hostname}/{repository}"
HUGGING_FACE_FRAMEWORK = "huggingface"
PYTORCH_FRAMEWORK = "pytorch"

CONFIGURABLE_ATTRIBUTES = [
    "version",
    "py_version",
    "instance_type",
    "accelerator_type",
    "image_scope",
    "container_version",
    "distributed",
    "smp",
    "base_framework_version",
    "training_compiler_config",
    "model_id",
    "model_version",
    "sdk_version",
    "inference_tool",
    "serverless_inference_config",
]


class ImageRetriever:
    @staticmethod
    def retrieve_hugging_face_uri(
        region: str,
        version: Optional[str] = None,
        py_version: Optional[str] = None,
        instance_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        image_scope: Optional[str] = None,
        container_version: str = None,
        distributed: bool = False,
        base_framework_version: Optional[str] = None,
        training_compiler_config: TrainingCompilerConfig = None,
        sdk_version: Optional[str] = None,
        inference_tool: Optional[str] = None,
        serverless_inference_config: ServerlessInferenceConfig = None,
    ):
        """Retrieves the ECR URI for the Docker image of HuggingFace models.

        Args:
            region (str): The AWS region.
            version (Optional[str]): The framework or algorithm version. This is required if there is
                more than one supported version for the given framework or algorithm.
            py_version (Optional[str]): The Python version. This is required if there is
                more than one supported Python version for the given framework version.
            instance_type (Optional[str]): The SageMaker instance type. For supported types, see
                https://aws.amazon.com/sagemaker/pricing. This is required if
                there are different images for different processor types.
            accelerator_type (Optional[str]): Elastic Inference accelerator type. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
            image_scope (Optional[str]): The image type, i.e. what it is used for.
                Valid values: "training", "inference", "inference_graviton", "eia".
                If ``accelerator_type`` is set, ``image_scope`` is ignored.
            container_version (Optional[str]): the version of docker image.
                Ideally the value of parameter should be created inside the framework.
                For custom use, see the list of supported container versions:
                https://github.com/aws/deep-learning-containers/blob/master/available_images.md
                (default: None).
            distributed (bool): If the image is for running distributed training.
            base_framework_version (Optional[str]): The base version number of PyTorch or Tensorflow.
                (default: None).
            training_compiler_config (:class:`~sagemaker.training_compiler.TrainingCompilerConfig`):
                A configuration class for the SageMaker Training Compiler
                (default: None).
            sdk_version (Optional[str]): the version of python-sdk that will be used in the image retrieval.
                (default: None).
            inference_tool (Optional[str]): the tool that will be used to aid in the inference.
                Valid values: "neuron, neuronx, None"
                (default: None).
            serverless_inference_config (sagemaker.serve.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to determine processor type..

        Returns:
            str: The ECR URI for the corresponding SageMaker Docker image.
        """
        args = dict(locals())
        for name, val in args.items():
            if name in CONFIGURABLE_ATTRIBUTES and not val:
                default_value = SageMakerConfig.resolve_value_from_config(
                    config_path=_simple_path(
                        SAGEMAKER, MODULES, IMAGE_RETRIEVER, to_camel_case(name)
                    )
                )
                if default_value is not None:
                    locals()[name] = default_value

        if training_compiler_config:
            final_image_scope = image_scope
            config = _config_for_framework_and_scope(
                HUGGING_FACE_FRAMEWORK + "-training-compiler", final_image_scope, accelerator_type
            )
        else:
            _framework = HUGGING_FACE_FRAMEWORK
            inference_tool = _get_inference_tool(inference_tool, instance_type)
            if inference_tool in ["neuron", "neuronx"]:
                _framework = f"{HUGGING_FACE_FRAMEWORK}-{inference_tool}"
            final_image_scope = _get_final_image_scope(
                HUGGING_FACE_FRAMEWORK, instance_type, image_scope
            )
            _validate_for_suppported_frameworks_and_instance_type(
                HUGGING_FACE_FRAMEWORK, instance_type
            )
            config = _config_for_framework_and_scope(
                _framework, final_image_scope, accelerator_type
            )

        if base_framework_version is not None:
            processor = _processor(instance_type, ["cpu", "gpu"])
            is_native_huggingface_gpu = processor == "gpu" and not training_compiler_config
            container_version = "cu110-ubuntu18.04" if is_native_huggingface_gpu else None

        original_version = version
        version = _validate_version_and_set_if_needed(
            version, config, HUGGING_FACE_FRAMEWORK, image_scope
        )
        version_config = config["versions"][_version_for_config(version, config)]

        if version_config.get("version_aliases"):
            full_base_framework_version = version_config["version_aliases"].get(
                base_framework_version, base_framework_version
            )
        _validate_arg(full_base_framework_version, list(version_config.keys()), "base framework")
        version_config = version_config.get(full_base_framework_version)

        py_version = _validate_py_version_and_set_if_needed(
            py_version, version_config, HUGGING_FACE_FRAMEWORK
        )
        version_config = version_config.get(py_version) or version_config
        registry = _registry_from_region(region, version_config["registries"])
        endpoint_data = _botocore_resolver().construct_endpoint("ecr", region)
        if region == "il-central-1" and not endpoint_data:
            endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(region)}
        hostname = endpoint_data["hostname"]

        repo = version_config["repository"]

        processor = _processor(
            instance_type,
            config.get("processors") or version_config.get("processors"),
            serverless_inference_config,
        )

        # if container version is available in .json file, utilize that
        if version_config.get("container_version"):
            container_version = version_config["container_version"][processor]

        # Append sdk version in case of trainium instances
        if repo in ["pytorch-training-neuron", "pytorch-training-neuronx"]:
            if not sdk_version:
                sdk_version = _get_latest_versions(version_config["sdk_versions"])
            container_version = sdk_version + "-" + container_version

        pt_or_tf_version = (
            re.compile("^(pytorch|tensorflow)(.*)$").match(base_framework_version).group(2)
        )
        _version = original_version

        if repo in [
            "huggingface-pytorch-trcomp-training",
            "huggingface-tensorflow-trcomp-training",
        ]:
            _version = version
        if repo in [
            "huggingface-pytorch-inference-neuron",
            "huggingface-pytorch-inference-neuronx",
        ]:
            if not sdk_version:
                sdk_version = _get_latest_versions(version_config["sdk_versions"])
            container_version = sdk_version + "-" + container_version
            if config.get("version_aliases").get(original_version):
                _version = config.get("version_aliases")[original_version]
            if (
                config.get("versions", {})
                .get(_version, {})
                .get("version_aliases", {})
                .get(base_framework_version, {})
            ):
                _base_framework_version = config.get("versions")[_version]["version_aliases"][
                    base_framework_version
                ]
                pt_or_tf_version = (
                    re.compile("^(pytorch|tensorflow)(.*)$").match(_base_framework_version).group(2)
                )

        tag_prefix = f"{pt_or_tf_version}-transformers{_version}"

        if repo == f"{HUGGING_FACE_FRAMEWORK}-inference-graviton":
            container_version = f"{container_version}-sagemaker"
        _validate_instance_deprecation(HUGGING_FACE_FRAMEWORK, instance_type, version)

        tag = _get_image_tag(
            container_version,
            distributed,
            final_image_scope,
            HUGGING_FACE_FRAMEWORK,
            inference_tool,
            instance_type,
            processor,
            py_version,
            tag_prefix,
            version,
        )

        if tag:
            repo += ":{}".format(tag)

        return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo)

    @staticmethod
    def retrieve_jumpstart_uri():
        """Retrieves the container image URI for JumpStart models.

        TODO: We are changing the Jumpstart utilities and moving from S3 based metadata to
        API based metadata. Implement this function after utility changes are finished.
        """

    @staticmethod
    def retrieve_pytorch_uri(
        region: str,
        version: Optional[str] = None,
        py_version: Optional[str] = None,
        instance_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        image_scope: Optional[str] = None,
        container_version: str = None,
        distributed: bool = False,
        smp: bool = False,
        training_compiler_config: TrainingCompilerConfig = None,
        sdk_version: Optional[str] = None,
        inference_tool: Optional[str] = None,
        serverless_inference_config: ServerlessInferenceConfig = None,
    ):
        """Retrieves the ECR URI for the Docker image of PyTorch models.

        Args:
            region (str): The AWS region.
            version (Optional[str]): The framework or algorithm version. This is required if there is
                more than one supported version for the given framework or algorithm.
            py_version (Optional[str]): The Python version. This is required if there is
                more than one supported Python version for the given framework version.
            instance_type (Optional[str]): The SageMaker instance type. For supported types, see
                https://aws.amazon.com/sagemaker/pricing. This is required if
                there are different images for different processor types.
            accelerator_type (Optional[str]): Elastic Inference accelerator type. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
            image_scope (Optional[str]): The image type, i.e. what it is used for.
                Valid values: "training", "inference", "inference_graviton", "eia".
                If ``accelerator_type`` is set, ``image_scope`` is ignored.
            container_version (Optional[str]): the version of docker image.
                Ideally the value of parameter should be created inside the framework.
                For custom use, see the list of supported container versions:
                https://github.com/aws/deep-learning-containers/blob/master/available_images.md
                (default: None).
            distributed (bool): If the image is for running distributed training.
            smp (bool): If using the SMP library for distributed training.
            training_compiler_config (:class:`~sagemaker.training_compiler.TrainingCompilerConfig`):
                A configuration class for the SageMaker Training Compiler
                (default: None).
            sdk_version (Optional[str]): the version of python-sdk that will be used in the image retrieval.
                (default: None).
            inference_tool (Optional[str]): the tool that will be used to aid in the inference.
                Valid values: "neuron, neuronx, None"
                (default: None).
            serverless_inference_config (sagemaker.serve.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to determine processor type.

        Returns:
            str: The ECR URI for the corresponding SageMaker Docker image.
        """
        if _retrieve_pytorch_uri_inputs_are_all_default(
            version,
            py_version,
            instance_type,
            accelerator_type,
            image_scope,
            container_version,
            distributed,
            smp,
            training_compiler_config,
            sdk_version,
            inference_tool,
            serverless_inference_config,
        ):
            # if all inputs are not set, return the latest pytorch training uri
            return _retrieve_latest_pytorch_training_uri(region)

        framework = PYTORCH_FRAMEWORK
        if training_compiler_config:
            final_image_scope = image_scope
            config = _config_for_framework_and_scope(
                PYTORCH_FRAMEWORK + "-training-compiler", final_image_scope, accelerator_type
            )
        else:
            _framework = PYTORCH_FRAMEWORK
            inference_tool = _get_inference_tool(inference_tool, instance_type)
            if inference_tool in ["neuron", "neuronx"]:
                _framework = f"{PYTORCH_FRAMEWORK}-{inference_tool}"
            final_image_scope = _get_final_image_scope(
                PYTORCH_FRAMEWORK, instance_type, image_scope
            )
            _validate_for_suppported_frameworks_and_instance_type(PYTORCH_FRAMEWORK, instance_type)
            config = _config_for_framework_and_scope(
                _framework, final_image_scope, accelerator_type
            )

        version = _validate_version_and_set_if_needed(
            version, config, PYTORCH_FRAMEWORK, image_scope
        )
        version_config = config["versions"][_version_for_config(version, config)]

        if distributed and smp:
            framework = "pytorch-smp"
            supported_smp_pt_versions_cu124 = ("2.5",)
            supported_smp_pt_versions_cu121 = ("2.1", "2.2", "2.3", "2.4")
            if any(pt_version in version for pt_version in supported_smp_pt_versions_cu124):
                container_version = "cu124"
            elif "p5" in instance_type or any(
                pt_version in version for pt_version in supported_smp_pt_versions_cu121
            ):
                container_version = "cu121"
            else:
                container_version = "cu118"

        py_version = _validate_py_version_and_set_if_needed(py_version, version_config, framework)
        version_config = version_config.get(py_version) or version_config
        registry = _registry_from_region(region, version_config["registries"])
        endpoint_data = _botocore_resolver().construct_endpoint("ecr", region)
        if region == "il-central-1" and not endpoint_data:
            endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(region)}
        hostname = endpoint_data["hostname"]

        repo = version_config["repository"]

        processor = _processor(
            instance_type,
            config.get("processors") or version_config.get("processors"),
            serverless_inference_config,
        )

        # if container version is available in .json file, utilize that
        if version_config.get("container_version"):
            container_version = version_config["container_version"][processor]

        # Append sdk version in case of trainium instances
        if repo in ["pytorch-training-neuron", "pytorch-training-neuronx"]:
            if not sdk_version:
                sdk_version = _get_latest_versions(version_config["sdk_versions"])
            container_version = sdk_version + "-" + container_version

        tag_prefix = version_config.get("tag_prefix", version)

        if repo == f"{framework}-inference-graviton":
            container_version = f"{container_version}-sagemaker"
        _validate_instance_deprecation(framework, instance_type, version)

        tag = _get_image_tag(
            container_version,
            distributed,
            final_image_scope,
            framework,
            inference_tool,
            instance_type,
            processor,
            py_version,
            tag_prefix,
            version,
        )

        if tag:
            repo += ":{}".format(tag)

        return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo)

    @staticmethod
    @override_pipeline_parameter_var
    def retrieve(
        framework: str,
        region: str,
        version: Optional[str] = None,
        py_version: Optional[str] = None,
        instance_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        image_scope: Optional[str] = None,
        container_version: str = None,
        distributed: bool = False,
        smp: bool = False,
        base_framework_version: Optional[str] = None,
        training_compiler_config: TrainingCompilerConfig = None,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        sdk_version: Optional[str] = None,
        inference_tool: Optional[str] = None,
        serverless_inference_config: ServerlessInferenceConfig = None,
    ):
        """Retrieves the ECR URI for the Docker image matching the given arguments.

        Args:
            framework (str): The name of the framework or algorithm.
            region (str): The AWS region.
            version (Optional[str]): The framework or algorithm version. This is required if there is
                more than one supported version for the given framework or algorithm.
            py_version (Optional[str]): The Python version. This is required if there is
                more than one supported Python version for the given framework version.
            instance_type (Optional[str]): The SageMaker instance type. For supported types, see
                https://aws.amazon.com/sagemaker/pricing. This is required if
                there are different images for different processor types.
            accelerator_type (Optional[str]): Elastic Inference accelerator type. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
            image_scope (Optional[str]): The image type, i.e. what it is used for.
                Valid values: "training", "inference", "inference_graviton", "eia".
                If ``accelerator_type`` is set, ``image_scope`` is ignored.
            container_version (Optional[str]): the version of docker image.
                Ideally the value of parameter should be created inside the framework.
                For custom use, see the list of supported container versions:
                https://github.com/aws/deep-learning-containers/blob/master/available_images.md
                (default: None).
            distributed (bool): If the image is for running distributed training.
            smp (bool): If using the SMP library for distributed training.
            base_framework_version (Optional[str]): The base version number of PyTorch or Tensorflow.
                (default: None).
            training_compiler_config (:class:`~sagemaker.training_compiler.TrainingCompilerConfig`):
                A configuration class for the SageMaker Training Compiler
                (default: None).
            model_id (Optional[str]): The JumpStart model ID for which to retrieve the image URI
                (default: None).
            model_version (Optional[str]): The version of the JumpStart model for which to retrieve the
                image URI (default: None).
            hub_arn (Optional[str]): The arn of the SageMaker Hub for which to retrieve
                model details from. (Default: None).
            tolerate_vulnerable_model (bool): ``True`` if vulnerable versions of model specifications
                should be tolerated without an exception raised. If ``False``, raises an exception if
                the script used by this version of the model has dependencies with known security
                vulnerabilities. (Default: False).
            tolerate_deprecated_model (bool): True if deprecated versions of model specifications
                should be tolerated without an exception raised. If False, raises an exception
                if the version of the model is deprecated. (Default: False).
            sdk_version (Optional[str]): the version of python-sdk that will be used in the image retrieval.
                (default: None).
            inference_tool (Optional[str]): the tool that will be used to aid in the inference.
                Valid values: "neuron, neuronx, None"
                (default: None).
            serverless_inference_config (sagemaker.serve.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to determine processor type.
            sagemaker_session (sagemaker.core.helper.Session): A SageMaker Session
                object, used for SageMaker interactions. If not
                specified, one is created using the default AWS configuration
                chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
            config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
            model_type (JumpStartModelType): The type of the model, can be open weights model
                or proprietary model. (Default: JumpStartModelType.OPEN_WEIGHTS).

        Returns:
            str: The ECR URI for the corresponding SageMaker Docker image.

        Raises:
            NotImplementedError: If the scope is not supported.
            ValueError: If the combination of arguments specified is not supported or
                any PipelineVariable object is passed in.
            VulnerableJumpStartModelError: If any of the dependencies required by the script have
                known security vulnerabilities.
            DeprecatedJumpStartModelError: If the version of the model is deprecated.
        """
        args = dict(locals())
        for name, val in args.items():
            if name in CONFIGURABLE_ATTRIBUTES and not val:
                default_value = SageMakerConfig.resolve_value_from_config(
                    config_path=_simple_path(
                        SAGEMAKER, MODULES, IMAGE_RETRIEVER, to_camel_case(name)
                    )
                )
                if default_value is not None:
                    locals()[name] = default_value

        for name, val in args.items():
            if is_pipeline_variable(val):
                raise ValueError(
                    "When retrieving the image_uri, the argument %s should not be a pipeline variable "
                    "(%s) since pipeline variables are only interpreted in the pipeline execution time."
                    % (name, type(val))
                )
        if model_id:
            return ImageRetriever.retrieve_jumpstart_uri()

        if framework == HUGGING_FACE_FRAMEWORK:
            return ImageRetriever.retrieve_hugging_face_uri(
                region=region,
                version=version,
                py_version=py_version,
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                image_scope=image_scope,
                container_version=container_version,
                distributed=distributed,
                base_framework_version=base_framework_version,
                training_compiler_config=training_compiler_config,
                sdk_version=sdk_version,
                inference_tool=inference_tool,
                serverless_inference_config=serverless_inference_config,
            )

        if framework == PYTORCH_FRAMEWORK:
            return ImageRetriever.retrieve_pytorch_uri(
                region=region,
                version=version,
                py_version=py_version,
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                image_scope=image_scope,
                container_version=container_version,
                distributed=distributed,
                smp=smp,
                training_compiler_config=training_compiler_config,
                sdk_version=sdk_version,
                inference_tool=inference_tool,
                serverless_inference_config=serverless_inference_config,
            )

        final_image_scope = _get_final_image_scope(framework, instance_type, image_scope)
        _validate_for_suppported_frameworks_and_instance_type(framework, instance_type)
        config = _config_for_framework_and_scope(framework, final_image_scope, accelerator_type)

        version = _validate_version_and_set_if_needed(version, config, framework, image_scope)
        version_config = config["versions"][_version_for_config(version, config)]

        py_version = _validate_py_version_and_set_if_needed(py_version, version_config, framework)
        version_config = version_config.get(py_version) or version_config
        registry = _registry_from_region(region, version_config["registries"])
        endpoint_data = _botocore_resolver().construct_endpoint("ecr", region)
        if region == "il-central-1" and not endpoint_data:
            endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(region)}
        hostname = endpoint_data["hostname"]

        repo = version_config["repository"]

        processor = _processor(
            instance_type,
            config.get("processors") or version_config.get("processors"),
            serverless_inference_config,
        )

        # if container version is available in .json file, utilize that
        if version_config.get("container_version"):
            container_version = version_config["container_version"][processor]

        # Append sdk version in case of trainium instances
        if repo in ["pytorch-training-neuron", "pytorch-training-neuronx"]:
            if not sdk_version:
                sdk_version = _get_latest_versions(version_config["sdk_versions"])
            container_version = sdk_version + "-" + container_version

        tag_prefix = version_config.get("tag_prefix", version)

        if repo == f"{framework}-inference-graviton":
            container_version = f"{container_version}-sagemaker"
        _validate_instance_deprecation(framework, instance_type, version)

        tag = _get_image_tag(
            container_version,
            distributed,
            final_image_scope,
            framework,
            inference_tool,
            instance_type,
            processor,
            py_version,
            tag_prefix,
            version,
        )

        if tag:
            repo += ":{}".format(tag)

        return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo)

    @staticmethod
    def retrieve_base_python_image_uri(region: str, py_version: str = "310") -> str:
        """Retrieves the image URI for base python image.

        Args:
            region (str): The AWS region to use for image URI.
            py_version (str): The python version to use for the image.
            Default to 310

        Returns:
            str: The image URI string.
        """

        framework = "sagemaker-base-python"
        version = "1.0"
        endpoint_data = _botocore_resolver().construct_endpoint("ecr", region)
        if region == "il-central-1" and not endpoint_data:
            endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(region)}
        hostname = endpoint_data["hostname"]
        config = config_for_framework(framework)
        version_config = config["versions"][_version_for_config(version, config)]

        registry = _registry_from_region(region, version_config["registries"])

        repo = version_config["repository"] + "-" + py_version
        repo_and_tag = repo + ":" + version

        return ECR_URI_TEMPLATE.format(
            registry=registry, hostname=hostname, repository=repo_and_tag
        )
