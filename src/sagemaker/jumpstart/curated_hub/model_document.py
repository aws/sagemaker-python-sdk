import json
from dataclasses import asdict
from typing import Optional

from sagemaker import environment_variables as env_vars
from sagemaker.jumpstart.curated_hub.content_copy import ContentCopier
from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import (
    HubModelSpec_v1_0_0,
    DefaultDeploymentConfig,
    DefaultTrainingConfig,
    DefaultDeploymentSdkArgs,
    DefaultTrainingSdkArgs,
    FrameworkImageConfig,
    ModelArtifactConfig,
    ScriptConfig,
    InstanceConfig,
    InferenceNotebookConfig,
    convert_public_model_hyperparameter_to_hub_hyperparameter, SdkArgs, DatasetConfig, )
from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import ModelCapabilities
from sagemaker.jumpstart.curated_hub.utils import construct_s3_uri, base_framework
from sagemaker.jumpstart.types import JumpStartModelSpecs


class ModelDocumentCreator:
    """Makes HubContentDocument for Hub Models."""

    def __init__(self, region: str, content_copier: ContentCopier) -> None:
        """Sets up basic info."""
        self._region = region
        self._content_copier = content_copier

    def make_hub_content_document(self, model_specs: JumpStartModelSpecs) -> str:
        """Converts the provided JumpStartModelSpecs into a Hub Content Document."""
        capabilities = []
        if model_specs.training_supported:
            capabilities.append(ModelCapabilities.TRAINING)
        if model_specs.incremental_training_supported:
            capabilities.append(ModelCapabilities.INCREMENTAL_TRAINING)

        hub_model_spec = HubModelSpec_v1_0_0(
            Capabilities=capabilities,  # TODO add inference if needed?
            DataType="",  # TODO not in SDK metadata
            MlTask="",  # TODO not in SDK metadata
            Framework=model_specs.hosting_ecr_specs.framework,
            Origin=None,
            Dependencies=[],  # TODO add references to copied artifacts
            DatasetConfig=self._dataset_config(model_specs=model_specs),
            DefaultTrainingConfig=self._make_hub_content_default_training_config(
                model_specs=model_specs
            ),
            DefaultDeploymentConfig=self._make_hub_content_default_deployment_config(
                model_specs=model_specs,
            ),
        )

        hub_model_spec_dict = asdict(hub_model_spec)
        if not model_specs.training_supported:
            # Remove keys in the document that would be null and cause an FE validation failure
            # Python dataclass forces us to add these kwargs initially
            hub_model_spec_dict.pop("DefaultTrainingConfig")
            hub_model_spec_dict.pop("DatasetConfig")

        if model_specs.supports_prepacked_inference():
            hub_model_spec_dict["DefaultDeploymentConfig"].pop("ScriptConfig")

        return json.dumps(hub_model_spec_dict)

    def _make_hub_content_default_deployment_config(
        self, model_specs: JumpStartModelSpecs
    ) -> DefaultDeploymentConfig:
        environment_variables = env_vars.retrieve_default(
            region=self._region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            include_aws_sdk_env_vars=False,
        )
        return DefaultDeploymentConfig(
            SdkArgs=DefaultDeploymentSdkArgs(
                MinSdkVersion=model_specs.min_sdk_version,
                SdkModelArgs=SdkArgs(
                    EntryPoint=None,  # TODO check correct way to determine this
                    EnableNetworkIsolation=model_specs.inference_enable_network_isolation,
                    Environment=environment_variables
                ),
            ),
            FrameworkImageConfig=FrameworkImageConfig(
                Framework=model_specs.hosting_ecr_specs.framework,
                FrameworkVersion=model_specs.hosting_ecr_specs.framework_version,
                PythonVersion=model_specs.hosting_ecr_specs.py_version,
                TransformersVersion=getattr(
                    model_specs.hosting_ecr_specs, "huggingface_transformers_version", None
                ),
                BaseFramework=base_framework(model_specs=model_specs),
            ),
            ModelArtifactConfig=ModelArtifactConfig(
                ArtifactLocation=construct_s3_uri(
                    self._content_copier.dst_bucket(),
                    self._content_copier.dst_inference_artifact_key(model_specs=model_specs)
                ),
            ),
            ScriptConfig=ScriptConfig(
                ScriptLocation=construct_s3_uri(
                    self._content_copier.dst_bucket(),
                    self._content_copier.dst_inference_script_key(model_specs=model_specs)
                ),
            ),
            InstanceConfig=InstanceConfig(
                DefaultInstanceType=model_specs.default_inference_instance_type,
                InstanceTypeOptions=model_specs.supported_inference_instance_types or [],
            ),
            InferenceNotebookConfig=InferenceNotebookConfig(
                NotebookLocation=construct_s3_uri(
                    self._content_copier.dst_bucket(),
                    self._content_copier.dst_notebook_key(model_specs=model_specs)
                )
            ),
            CustomImageConfig=None,
        )

    def _make_hub_content_default_training_config(
        self, model_specs: JumpStartModelSpecs
    ) -> Optional[DefaultTrainingConfig]:
        if not model_specs.training_supported:
            return None

        return DefaultTrainingConfig(
            SdkArgs=DefaultTrainingSdkArgs(
                MinSdkVersion=model_specs.min_sdk_version,
                SdkEstimatorArgs=None,  # TODO add env variables
            ),
            CustomImageConfig=None,
            FrameworkImageConfig=FrameworkImageConfig(
                Framework=model_specs.training_ecr_specs.framework,
                FrameworkVersion=model_specs.training_ecr_specs.framework_version,
                PythonVersion=model_specs.training_ecr_specs.py_version,
                TransformersVersion=getattr(
                    model_specs.training_ecr_specs, "huggingface_transformers_version", None
                ),
                BaseFramework=base_framework(model_specs=model_specs),
            ),
            ModelArtifactConfig=ModelArtifactConfig(
                ArtifactLocation=construct_s3_uri(
                    self._content_copier.dst_bucket(),
                    self._content_copier.dst_training_artifact_key(model_specs=model_specs)
                ),
            ),
            ScriptConfig=ScriptConfig(
                ScriptLocation=construct_s3_uri(
                    self._content_copier.dst_bucket(),
                    self._content_copier.dst_training_script_key(model_specs=model_specs)
                ),
            ),
            InstanceConfig=InstanceConfig(
                DefaultInstanceType=model_specs.default_training_instance_type,
                InstanceTypeOptions=model_specs.supported_training_instance_types or [],
            ),
            Hyperparameters=list(
                map(
                    convert_public_model_hyperparameter_to_hub_hyperparameter,
                    model_specs.hyperparameters,
                )
            ),
            ExtraChannels=[],  # TODO: I can't seem to find these
        )

    def _dataset_config(self, model_specs: JumpStartModelSpecs) -> Optional[DatasetConfig]:
        if not model_specs.training_supported:
            return None
        return DatasetConfig(
            TrainingDatasetLocation=self._content_copier.dst_training_dataset_location(model_specs=model_specs),
            ValidationDatasetLocation=None,
            DataFormatLocation=None,
            PredictColumn=None,
        )
