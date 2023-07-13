import json
from dataclasses import asdict
from typing import Optional, Dict, Any

from sagemaker import environment_variables as env_vars
from sagemaker.jumpstart.curated_hub.filesystem.curated_hub_s3_filesystem import CuratedHubS3Filesystem
from sagemaker.jumpstart.curated_hub.filesystem.public_hub_s3_filesystem import PublicHubS3Filesystem
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
    Dependency,
    DependencyType,
    convert_public_model_hyperparameter_to_hub_hyperparameter, SdkArgs, DatasetConfig, )
from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import ModelCapabilities
from sagemaker.jumpstart.curated_hub.utils import construct_s3_uri, base_framework
from sagemaker.jumpstart.types import JumpStartModelSpecs


class ModelDocumentCreator:
    """Makes HubContentDocument for Hub Models."""

    def __init__(self, region: str, src_s3_filesystem: PublicHubS3Filesystem, palatine_hub_s3_filesystem: CuratedHubS3Filesystem, studio_metadata_map: Dict[str, Any]) -> None:
        """Sets up basic info."""
        self._region = region
        self._src_s3_filesystem = src_s3_filesystem
        self._palatine_hub_s3_filesystem = palatine_hub_s3_filesystem
        self.studio_metadata_map = studio_metadata_map

    def make_hub_content_document(self, model_specs: JumpStartModelSpecs) -> str:
        """Converts the provided JumpStartModelSpecs into a Hub Content Document."""
        capabilities = []
        if model_specs.training_supported:
            capabilities.append(ModelCapabilities.TRAINING)
        if model_specs.incremental_training_supported:
            capabilities.append(ModelCapabilities.INCREMENTAL_TRAINING)

        hub_model_spec = HubModelSpec_v1_0_0(
            Capabilities=capabilities,
            DataType=self.studio_metadata_map[model_specs.model_id]["dataType"],
            MlTask=self.studio_metadata_map[model_specs.model_id]["problemType"],
            Framework=model_specs.hosting_ecr_specs.framework,
            Origin=None,
            Dependencies=self._make_hub_dependency_list(model_specs),  # TODO add references to copied artifacts
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

    def _make_hub_dependency_list(self, model_specs: JumpStartModelSpecs):
        dependencies = []

        dependencies.append(
            Dependency(
              DependencyOriginPath=self._src_s3_filesystem.get_inference_artifact_s3_reference(model_specs).get_uri(),
              DependencyCopyPath=self._palatine_hub_s3_filesystem.get_inference_artifact_s3_reference(model_specs).get_uri(),
              DependencyType=DependencyType.ARTIFACT
            )
        )
        dependencies.append(
              Dependency(
                DependencyOriginPath=self._src_s3_filesystem.get_inference_script_s3_reference(model_specs).get_uri(),
                DependencyCopyPath=self._palatine_hub_s3_filesystem.get_inference_script_s3_reference(model_specs).get_uri(),
                DependencyType=DependencyType.SCRIPT
              )
          )
        dependencies.append(
              Dependency(
                DependencyOriginPath=self._src_s3_filesystem.get_demo_notebook_s3_reference(model_specs).get_uri(),
                DependencyCopyPath=self._palatine_hub_s3_filesystem.get_demo_notebook_s3_reference(model_specs).get_uri(),
                DependencyType=DependencyType.NOTEBOOK
              )
          )
        dependencies.append(
              Dependency(
                DependencyOriginPath=self._src_s3_filesystem.get_markdown_s3_reference(model_specs).get_uri(),
                DependencyCopyPath=self._palatine_hub_s3_filesystem.get_markdown_s3_reference(model_specs).get_uri(),
                DependencyType=DependencyType.OTHER
              )
          )

        if model_specs.training_supported:
          dependencies.append(
              Dependency(
                DependencyOriginPath=self._src_s3_filesystem.get_training_artifact_s3_reference(model_specs).get_uri(),
                DependencyCopyPath=self._palatine_hub_s3_filesystem.get_training_artifact_s3_reference(model_specs).get_uri(),
                DependencyType=DependencyType.ARTIFACT
              )
          )
          dependencies.append(
              Dependency(
                DependencyOriginPath=self._src_s3_filesystem.get_training_script_s3_reference(model_specs).get_uri(),
                DependencyCopyPath=self._palatine_hub_s3_filesystem.get_training_script_s3_reference(model_specs).get_uri(),
                DependencyType=DependencyType.SCRIPT
              )
          )
          dependencies.append(
              Dependency(
                DependencyOriginPath=self._src_s3_filesystem.get_default_training_dataset_s3_reference(model_specs).get_uri(),
                DependencyCopyPath=self._palatine_hub_s3_filesystem.get_default_training_dataset_s3_reference(model_specs).get_uri(),
                DependencyType=DependencyType.DATASET
              )
          )

        return dependencies

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
                ArtifactLocation=self._palatine_hub_s3_filesystem.get_inference_artifact_s3_reference(model_specs).get_uri()
            ),
            ScriptConfig=ScriptConfig(
                ScriptLocation=self._palatine_hub_s3_filesystem.get_inference_script_s3_reference(model_specs).get_uri()
            ),
            InstanceConfig=InstanceConfig(
                DefaultInstanceType=model_specs.default_inference_instance_type,
                InstanceTypeOptions=model_specs.supported_inference_instance_types or [],
            ),
            InferenceNotebookConfig=InferenceNotebookConfig(
                NotebookLocation=self._palatine_hub_s3_filesystem.get_demo_notebook_s3_reference(model_specs).get_uri()
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
                ArtifactLocation=self._palatine_hub_s3_filesystem.get_training_artifact_s3_reference(model_specs).get_uri()
            ),
            ScriptConfig=ScriptConfig(
                ScriptLocation=self._palatine_hub_s3_filesystem.get_training_script_s3_reference(model_specs).get_uri()
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
            TrainingDatasetLocation=self._palatine_hub_s3_filesystem.get_default_training_dataset_s3_reference(model_specs).get_uri(),
            ValidationDatasetLocation=None,
            DataFormatLocation=None,
            PredictColumn=None,
        )
