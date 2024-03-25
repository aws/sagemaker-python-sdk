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
"""This module stores CuratedHub converter utilities for JumpStart."""
from __future__ import absolute_import

from typing import Any, Dict, List
from sagemaker.jumpstart.curated_hub.sync.request import HubSyncRequest
from sagemaker.jumpstart.curated_hub.types import (
    FileInfo,
    HubContentReferenceType,
    S3ObjectLocation,
)
from sagemaker.jumpstart.enums import ModelSpecKwargType, NamingConventionType, JumpStartScriptScope
from sagemaker import image_uris
from sagemaker.s3 import s3_path_join, parse_s3_url
from sagemaker.jumpstart.types import (
    JumpStartModelSpecs,
    HubContentType,
    JumpStartDataHolderType,
)
from sagemaker.jumpstart.curated_hub.interfaces import (
    DescribeHubContentResponse,
    HubModelDocument,
    HubContentDependency,
)
from sagemaker.jumpstart.curated_hub.parser_utils import (
    camel_to_snake,
    snake_to_upper_camel,
    walk_and_apply_json,
)


def _to_json(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    """Convert a complex nested dictionary of JumpStartDataHolderType into json"""
    for key, value in dictionary.items():
        if issubclass(type(value), JumpStartDataHolderType):
            dictionary[key] = walk_and_apply_json(value.to_json(), snake_to_upper_camel)
        elif isinstance(value, list):
            new_value = []
            for value_in_list in value:
                new_value_in_list = value_in_list
                if issubclass(type(value_in_list), JumpStartDataHolderType):
                    new_value_in_list = walk_and_apply_json(
                        value_in_list.to_json(), snake_to_upper_camel
                    )
                new_value.append(new_value_in_list)
            dictionary[key] = new_value
        elif isinstance(value, dict):
            for key_in_dict, value_in_dict in value.items():
                if issubclass(type(value_in_dict), JumpStartDataHolderType):
                    value[key_in_dict] = walk_and_apply_json(
                        value_in_dict.to_json(), snake_to_upper_camel
                    )
    return dictionary


def get_model_spec_arg_keys(
    arg_type: ModelSpecKwargType,
    naming_convention: NamingConventionType = NamingConventionType.DEFAULT,
) -> List[str]:
    """Returns a list of arg keys for a specific model spec arg type.

    Args:
        arg_type (ModelSpecKwargType): Type of the model spec's kwarg.
        naming_convention (NamingConventionType): Type of naming convention to return.

    Raises:
        ValueError: If the naming convention is not valid.
    """
    arg_keys = []
    if arg_type == ModelSpecKwargType.DEPLOY:
        arg_keys = ["ModelDataDownloadTimeout", "ContainerStartupHealthCheckTimeout"]
    elif arg_type == ModelSpecKwargType.ESTIMATOR:
        arg_keys = [
            "EncryptInterContainerTraffic",
            "MaxRuntimeInSeconds",
            "DisableOutputCompression",
            "ModelDir",
        ]
    elif arg_type == ModelSpecKwargType.MODEL:
        arg_keys = []
    elif arg_type == ModelSpecKwargType.FIT:
        arg_keys = []

    if naming_convention == NamingConventionType.SNAKE_CASE:
        return [camel_to_snake(key) for key in arg_keys]
    elif naming_convention == NamingConventionType.UPPER_CAMEL_CASE:
        return arg_keys
    else:
        raise ValueError("Please provide a valid naming convention.")


def get_model_spec_kwargs_from_hub_model_document(
    arg_type: ModelSpecKwargType,
    hub_content_document: Dict[str, Any],
    naming_convention: NamingConventionType = NamingConventionType.UPPER_CAMEL_CASE,
) -> Dict[str, Any]:
    """Returns a map of arg type to arg keys for a given hub content document.

    Args:
        arg_type (ModelSpecKwargType): Type of the model spec's kwarg.
        hub_content_document: A dictionary representation of hub content document.
        naming_convention (NamingConventionType): Type of naming convention to return.

    """
    kwargs = dict()
    keys = get_model_spec_arg_keys(arg_type, naming_convention=naming_convention)
    for k in keys:
        kwarg_value = hub_content_document.get(k, None)
        if kwarg_value is not None:
            kwargs[k] = kwarg_value
    return kwargs


def make_model_specs_from_describe_hub_content_response(
    response: DescribeHubContentResponse,
) -> JumpStartModelSpecs:
    """Sets fields in JumpStartModelSpecs based on values in DescribeHubContentResponse

    Args:
        response (Dict[str, any]): parsed DescribeHubContentResponse returned
            from SageMaker:DescribeHubContent
    """
    if response.hub_content_type != HubContentType.MODEL:
        raise AttributeError("Please make sure hub_content_type is model.")
    region = response.get_hub_region()
    specs = {}
    model_id = response.hub_content_name
    specs["model_id"] = model_id
    specs["version"] = response.hub_content_version
    hub_model_document: HubModelDocument = response.hub_content_document
    specs["url"] = hub_model_document.url
    specs["min_sdk_version"] = hub_model_document.min_sdk_version
    specs["training_supported"] = hub_model_document.training_supported
    specs["incremental_training_supported"] = bool(
        hub_model_document.incremental_training_supported
    )
    specs["hosting_ecr_uri"] = hub_model_document.hosting_ecr_uri

    hosting_artifact_key = hub_model_document.hosting_artifact_uri

    specs["hosting_artifact_key"] = hosting_artifact_key
    hosting_script_key = parse_s3_url(hub_model_document.hosting_script_uri)
    specs["hosting_script_key"] = hosting_script_key
    specs["inference_environment_variables"] = hub_model_document.inference_environment_variables
    specs["inference_vulnerable"] = False
    specs["inference_dependencies"] = hub_model_document.inference_dependencies
    specs["inference_vulnerabilities"] = []
    specs["training_vulnerable"] = False
    specs["training_dependencies"] = hub_model_document.training_dependencies
    specs["training_vulnerabilities"] = []
    specs["deprecated"] = False
    specs["deprecated_message"] = None
    specs["deprecate_warn_message"] = None
    specs["usage_info_message"] = None
    specs["default_inference_instance_type"] = hub_model_document.default_inference_instance_type
    specs["default_training_instance_type"] = hub_model_document.default_training_instance_type
    specs[
        "supported_inference_instance_types"
    ] = hub_model_document.supported_inference_instance_types
    specs[
        "supported_training_instance_types"
    ] = hub_model_document.supported_training_instance_types
    specs[
        "dynamic_container_deployment_supported"
    ] = hub_model_document.dynamic_container_deployment_supported
    specs["hosting_resource_requirements"] = hub_model_document.hosting_resource_requirements
    specs["metrics"] = hub_model_document.training_metrics
    specs["training_prepacked_script_key"] = None
    if hub_model_document.training_prepacked_script_uri is not None:
        training_prepacked_script_key = hub_model_document.training_prepacked_script_uri

        specs["training_prepacked_script_key"] = training_prepacked_script_key

    specs["hosting_prepacked_artifact_key"] = None
    if hub_model_document.hosting_prepacked_artifact_uri is not None:
        hosting_prepacked_artifact_key = hub_model_document.hosting_prepacked_artifact_uri

        specs["hosting_prepacked_artifact_key"] = hosting_prepacked_artifact_key

    specs["fit_kwargs"] = get_model_spec_kwargs_from_hub_model_document(
        ModelSpecKwargType.FIT, hub_model_document.to_json()
    )
    specs["model_kwargs"] = get_model_spec_kwargs_from_hub_model_document(
        ModelSpecKwargType.MODEL, hub_model_document.to_json()
    )
    specs["deploy_kwargs"] = get_model_spec_kwargs_from_hub_model_document(
        ModelSpecKwargType.DEPLOY, hub_model_document.to_json()
    )
    specs["estimator_kwargs"] = get_model_spec_kwargs_from_hub_model_document(
        ModelSpecKwargType.ESTIMATOR, hub_model_document.to_json()
    )

    specs["predictor_specs"] = hub_model_document.sage_maker_sdk_predictor_specifications
    default_payloads = {}
    for alias, payload in hub_model_document.default_payloads.items():
        default_payloads[alias] = walk_and_apply_json(payload.to_json(), camel_to_snake)
    specs["default_payloads"] = default_payloads
    specs["gated_bucket"] = hub_model_document.gated_bucket
    specs["inference_volume_size"] = hub_model_document.inference_volume_size
    specs[
        "inference_enable_network_isolation"
    ] = hub_model_document.inference_enable_network_isolation
    specs["resource_name_base"] = hub_model_document.resource_name_base

    specs["hosting_eula_key"] = None
    if hub_model_document.hosting_eula_uri is not None:
        specs["hosting_eula_key"] = hub_model_document.hosting_eula_uri

    if hub_model_document.hosting_model_package_arn:
        specs["hosting_model_package_arns"] = {region: hub_model_document.hosting_model_package_arn}

    specs["hosting_use_script_uri"] = hub_model_document.hosting_use_script_uri

    specs["hosting_instance_type_variants"] = hub_model_document.hosting_instance_type_variants

    if specs["training_supported"]:
        specs["training_ecr_uri"] = hub_model_document.training_ecr_uri
        training_artifact_key = hub_model_document.training_artifact_uri
        specs["training_artifact_key"] = training_artifact_key
        training_script_key = hub_model_document.training_script_uri
        specs["training_script_key"] = training_script_key

        specs["hyperparameters"] = hub_model_document.hyperparameters
        specs["training_volume_size"] = hub_model_document.training_volume_size
        specs[
            "training_enable_network_isolation"
        ] = hub_model_document.training_enable_network_isolation
        if hub_model_document.training_model_package_artifact_uri:
            specs["training_model_package_artifact_uris"] = {
                region: hub_model_document.training_model_package_artifact_uri
            }
        specs[
            "training_instance_type_variants"
        ] = hub_model_document.training_instance_type_variants
    return JumpStartModelSpecs(_to_json(specs), is_hub_content=True)


def make_hub_model_document_from_specs(
    model_specs: JumpStartModelSpecs,
    studio_manifest_entry: Dict[str, Any],
    studio_specs: Dict[str, Any],
    files: List[FileInfo],
    dest_location: S3ObjectLocation,
    hub_content_dependencies: List[HubContentDependency],
    region: str,
) -> HubModelDocument:
    """Sets fields in HubModelDocument based on model specs, studio specs,
    and hub content dependencies.
    """
    document = {}
    document["Url"] = model_specs.url
    document["MinSdkVersion"] = model_specs.min_sdk_version
    document["HostingEcrUri"] = image_uris.retrieve(
        model_id=model_specs.model_id,
        model_version=model_specs.version,
        framework=model_specs.hosting_ecr_specs.framework,
        instance_type=model_specs.default_inference_instance_type,
        image_scope=JumpStartScriptScope.INFERENCE,
        region=region,
    )
    document["GatedBucket"] = model_specs.gated_bucket
    document["HostingArtifactUri"] = next(
        (
            f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
                if not model_specs.gated_bucket else f"s3://{file.location.bucket}/{file.location.key}"
            for file in files
            if file.reference_type is HubContentReferenceType.INFERENCE_ARTIFACT
        ),
        None,
    )
    document["HostingArtifactS3DataType"] = studio_specs.get("inferenceArtifactS3DataType")
    document["HostingArtifactCompressionType"] = studio_specs.get(
        "inferenceArtifactCompressionType"
    )

    document["HostingScriptUri"] = next(
        (
            f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
            for file in files
            if file.reference_type is HubContentReferenceType.INFERENCE_SCRIPT
        ),
        None,
    )
    document["InferenceDependencies"] = model_specs.inference_dependencies
    document["InferenceEnvironmentVariables"] = model_specs.inference_environment_variables
    document["TrainingSupported"] = model_specs.training_supported
    document["IncrementalTrainingSupported"] = model_specs.incremental_training_supported
    document[
        "DynamicContainerDeploymentSupported"
    ] = model_specs.dynamic_container_deployment_supported
    document["HostingPrepackedArtifactUri"] = next(
        (
            f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
            for file in files
            if file.reference_type is HubContentReferenceType.INFERENCE_ARTIFACT
        ),
        None,
    )
    # document["HostingPrepackedArtifactVersion"] = model_specs.hosting_prepacked_artifact_version
    document["HostingUseScriptUri"] = model_specs.hosting_use_script_uri
    document["HostingEulaUri"] = next(
        (
            f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
            for file in files
            if file.reference_type is HubContentReferenceType.EULA
        ),
        None,
    )
    if model_specs.hosting_model_package_arns and region in model_specs.hosting_model_package_arns:
        document["HostingModelPackageArn"] = model_specs.hosting_model_package_arns[region]
    document["DefaultInferenceInstanceType"] = model_specs.default_inference_instance_type
    document["SupportedInferenceInstanceTypes"] = model_specs.supported_inference_instance_types
    document["SageMakerSdkPredictorSpecifications"] = model_specs.predictor_specs
    document["InferenceVolumeSize"] = model_specs.inference_volume_size
    document["InferenceEnableNetworkIsolation"] = model_specs.inference_enable_network_isolation
    document["ResourceNameBase"] = model_specs.resource_name_base
    document["DefaultPayloads"] = {
        alias: walk_and_apply_json(payload.to_json(), snake_to_upper_camel)
        for alias, payload in model_specs.default_payloads.items()
    }
    document["HostingResourceRequirements"] = model_specs.hosting_resource_requirements
    document[
        "HostingInstanceTypeVariants"
    ] = model_specs.hosting_instance_type_variants.regionalize(region)
    document["DefaultTrainingDatasetUri"] = next(
        (
            f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
            for file in files
            if file.reference_type is HubContentReferenceType.DEFAULT_TRAINING_DATASET
        ),
        None,
    )
    document["FineTuningSupported"] = studio_specs.get("fineTuningSupported")
    document["ValidationSupported"] = studio_specs.get("validationSupported")
    document["MinStudioSdkVersion"] = studio_specs.get("minServerVersion")
    if studio_specs.get("notebookLocations"):
        notebook_location_uris = {}
        # if notebook_locations.get("demoNotebook"):
        #     notebook_location_uris["demo_notebook"] = s3_path_join(
        #         "s3://", content_bucket, notebook_locations.get("demoNotebook")
        #     )
        # if notebook_locations.get("modelFit"):
        #     notebook_location_uris["model_fit"] = s3_path_join(
        #         "s3://", content_bucket, notebook_locations.get("modelFit")
        #     )
        notebook_location_uris["model_deploy"] = next(
            (
                f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
                for file in files
                if file.reference_type is HubContentReferenceType.INFERENCE_NOTEBOOK
            ),
            None,
        )
        document["NotebookLocationUris"] = notebook_location_uris

    document["ModelProviderIconUri"] = None  # Not needed for private beta.

    document["Task"] = studio_manifest_entry.get("problemType")
    document["Framework"] = studio_manifest_entry.get("framework")
    document["Datatype"] = studio_manifest_entry.get("dataType")
    document["License"] = studio_manifest_entry.get("license")
    document["ContextualHelp"] = studio_specs.get("contextualHelp")

    # Deploy kwargs
    document["ModelDataDownloadTimeout"] = model_specs.deploy_kwargs.get(
        "model_data_download_timeout"
    )
    document["ContainerStartupHealthCheckTimeout"] = model_specs.deploy_kwargs.get(
        "container_startup_health_check_timeout"
    )

    if document["TrainingSupported"]:
        if model_specs.training_model_package_artifact_uris:
            document[
                "TrainingModelPackageArtifactUri"
            ] = model_specs.training_model_package_artifact_uris.get(region)
        document["TrainingArtifactCompressionType"] = studio_specs.get(
            "trainingArtifactCompressionType"
        )
        document["TrainingArtifactS3DataType"] = studio_specs.get("trainingArtifactS3DataType")
        document["Hyperparameters"] = [
            walk_and_apply_json(param.to_json(), snake_to_upper_camel)
            for param in model_specs.hyperparameters
        ]
        document["TrainingScriptUri"] = next(
            (
                f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
                for file in files
                if file.reference_type is HubContentReferenceType.TRAINING_SCRIPT
            ),
            None,
        )
        document["TrainingPrepackedScriptUri"] = next(
            (
                f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
                for file in files
                if file.reference_type is HubContentReferenceType.TRAINING_SCRIPT
            ),
            None,
        )
        # document["TrainingPrepackedScriptVersion"] = model_specs.training_prepacked_script_version
        document["TrainingEcrUri"] = image_uris.retrieve(
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            framework=model_specs.training_ecr_specs.framework,
            instance_type=model_specs.default_training_instance_type,
            image_scope=JumpStartScriptScope.TRAINING,
            region=region,
        )
        document["TrainingMetrics"] = model_specs.metrics
        document["TrainingArtifactUri"] = next(
            (
                f"s3://{dest_location.bucket}/{dest_location.key}/{file.location.key}"
                for file in files
                if file.reference_type is HubContentReferenceType.TRAINING_ARTIFACT
            ),
            None,
        )
        document["Training_dependencies"] = model_specs.training_dependencies
        document["DefaultTrainingInstanceType"] = model_specs.default_training_instance_type
        document["SupportedTrainingInstanceTypes"] = model_specs.supported_training_instance_types
        document["TrainingVolumeSize"] = model_specs.training_volume_size
        document["TrainingEnableNetworkIsolation"] = model_specs.training_enable_network_isolation
        document[
            "TrainingInstanceTypeVariants"
        ] = model_specs.training_instance_type_variants.regionalize(region)

        # Estimator kwargs
        document["Encrypt_inter_container_traffic"] = model_specs.estimator_kwargs.get(
            "encrypt_inter_container_traffic"
        )
        document["MaxRuntimeInSeconds"] = model_specs.estimator_kwargs.get("max_run")
        document["DisableOutputCompression"] = model_specs.estimator_kwargs.get(
            "disable_output_compression"
        )
        document["ModelDir"] = model_specs.estimator_kwargs.get("model_dir")
    return HubModelDocument(_to_json(document), region, hub_content_dependencies)
