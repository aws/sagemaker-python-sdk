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
"""This module contains helpers to generate Private Hub content documents."""
from __future__ import absolute_import

import json
from dataclasses import asdict
from typing import Optional, Dict, Any, List

from sagemaker import environment_variables as env_vars
from sagemaker.jumpstart.curated_hub.accessors.model_dependency_s3_accessor import (
    ModelDependencyS3Accessor,
)
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
    SdkArgs,
    DatasetConfig,
)
from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import ModelCapabilities
from sagemaker.jumpstart.curated_hub.utils import (
    hosting_base_framework,
    training_base_framework,
    convert_public_model_hyperparameter_to_hub_hyperparameter,
)
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.accessors.constants import (
    UNCOMPRESSED_ARTIFACTS_VALUE
)


class ModelDocumentCreator:
    """Makes HubContentDocument for Hub Models."""

    def __init__(
        self,
        region: str,
        src_s3_accessor: ModelDependencyS3Accessor,
        hub_s3_accessor: ModelDependencyS3Accessor,
        studio_metadata_map: Dict[str, Any],
    ) -> None:
        """Sets up basic info."""
        self._region = region
        self._src_s3_accessor = src_s3_accessor
        self._dst_s3_accessor = hub_s3_accessor
        self.studio_metadata_map = studio_metadata_map

    def make_hub_content_document(self, model_specs: JumpStartModelSpecs) -> str:
        """Converts the provided JumpStartModelSpecs into a Hub Content Document."""
        hub_model_spec_dict = self._make_hub_content_document_json(model_specs)
        return json.dumps(hub_model_spec_dict)

    def _make_hub_content_document_json(self, model_specs: JumpStartModelSpecs) -> Dict[str, Any]:
        """Creates hub content document in json format"""
        capabilities: List[str] = []
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
            Dependencies=self._make_hub_dependency_list(model_specs),
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

        print(f"Generated HubContentDocument: {hub_model_spec_dict}")
        return hub_model_spec_dict
    
    def _get_uncompressed_inference_dependencies(self, model_specs: JumpStartModelSpecs) -> List[Dependency]:
        dependencies: List[Dependency] = []

        dependencies.append(
            Dependency(
                DependencyOriginPath=self._src_s3_accessor.get_uncompresssed_inference_artifact_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyCopyPath=self._dst_s3_accessor.get_uncompresssed_inference_artifact_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyType=DependencyType.ARTIFACT,
            )
        )
        dependencies.append(
            Dependency(
                DependencyOriginPath=self._src_s3_accessor.get_uncompresssed_inference_artifact_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyCopyPath=self._dst_s3_accessor.get_uncompresssed_inference_artifact_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyType=DependencyType.SCRIPT,
            )
        )

        return dependencies 
    
    def _get_inference_dependencies(self, model_specs: JumpStartModelSpecs) -> List[Dependency]:
        dependencies: List[Dependency] = []

        dependencies.append(
            Dependency(
                DependencyOriginPath=self._src_s3_accessor.get_inference_artifact_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyCopyPath=self._dst_s3_accessor.get_inference_artifact_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyType=DependencyType.ARTIFACT,
            )
        )
        dependencies.append(
            Dependency(
                DependencyOriginPath=self._src_s3_accessor.get_inference_script_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyCopyPath=self._dst_s3_accessor.get_inference_script_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyType=DependencyType.SCRIPT,
            )
        )

        return dependencies
    
    def _get_inference_artifact_and_script_reference(self, model_specs: JumpStartModelSpecs) -> List[Dependency]:
        if model_specs.hosting_artifact_s3_data_type == UNCOMPRESSED_ARTIFACTS_VALUE:
            return self._get_uncompressed_inference_dependencies(model_specs)
        else:
            return self._get_inference_dependencies(model_specs)
            

    def _make_hub_dependency_list(self, model_specs: JumpStartModelSpecs):
        """Creates hub content dependencies"""
        dependencies: List[Dependency] = []

        dependencies.extend(self._get_inference_artifact_and_script_reference(model_specs=model_specs))
        dependencies.append(
            Dependency(
                DependencyOriginPath=self._src_s3_accessor.get_demo_notebook_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyCopyPath=self._dst_s3_accessor.get_demo_notebook_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyType=DependencyType.NOTEBOOK,
            )
        )
        dependencies.append(
            Dependency(
                DependencyOriginPath=self._src_s3_accessor.get_markdown_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyCopyPath=self._dst_s3_accessor.get_markdown_s3_reference(
                    model_specs
                ).get_uri(),
                DependencyType=DependencyType.OTHER,
            )
        )

        if model_specs.training_supported:
            dependencies.append(self._get_training_artifact_s3_reference_dependency(model_specs))
            dependencies.append(self._get_training_script_s3_reference_dependency(model_specs))
            dependencies.append(self._get_default_training_dataset_dependency(model_specs))

        print(f"Generated dependencies: {dependencies}")
        return dependencies

    def _get_training_artifact_s3_reference_dependency(
        self, model_specs: JumpStartModelSpecs
    ) -> Dependency:
        """Returns HubContent dependency for training artifacts."""

        if model_specs.training_artifact_s3_data_type == UNCOMPRESSED_ARTIFACTS_VALUE:
            return self._get_uncompressed_training_artifact_s3_reference_dependency(model_specs)
        else:
            return self._get_training_artifact_dependency(model_specs)
    
    def _get_training_artifact_dependency(self, model_specs: JumpStartModelSpecs) -> Dependency:
        return Dependency(
            DependencyOriginPath=self._src_s3_accessor.get_training_artifact_s3_reference(
                model_specs
            ).get_uri(),
            DependencyCopyPath=self._dst_s3_accessor.get_training_artifact_s3_reference(
                model_specs
            ).get_uri(),
            DependencyType=DependencyType.ARTIFACT,
        )
    
    def _get_uncompressed_training_artifact_s3_reference_dependency(self, model_specs: JumpStartModelSpecs) -> Dependency:
        return Dependency(
            DependencyOriginPath=self._src_s3_accessor.get_uncompresssed_training_artifact_s3_reference(
                model_specs
            ).get_uri(),
            DependencyCopyPath=self._dst_s3_accessor.get_uncompresssed_training_artifact_s3_reference(
                model_specs
            ).get_uri(),
            DependencyType=DependencyType.ARTIFACT,
        )
        

    def _get_training_script_s3_reference_dependency(
        self, model_specs: JumpStartModelSpecs
    ) -> Dependency:
        """Returns HubContent dependency for training scripts."""
        return Dependency(
            DependencyOriginPath=self._src_s3_accessor.get_training_script_s3_reference(
                model_specs
            ).get_uri(),
            DependencyCopyPath=self._dst_s3_accessor.get_training_script_s3_reference(
                model_specs
            ).get_uri(),
            DependencyType=DependencyType.SCRIPT,
        )

    def _get_default_training_dataset_dependency(
        self, model_specs: JumpStartModelSpecs
    ) -> Dependency:
        """Returns HubContent dependency for training datasets."""
        return Dependency(
            DependencyOriginPath=self._src_s3_accessor.get_default_training_dataset_s3_reference(
                model_specs
            ).get_uri(),
            DependencyCopyPath=self._dst_s3_accessor.get_default_training_dataset_s3_reference(
                model_specs
            ).get_uri(),
            DependencyType=DependencyType.DATASET,
        )
    
    def _get_model_artifact_config(self, model_specs: JumpStartModelSpecs) -> ModelArtifactConfig:
        if model_specs.hosting_artifact_s3_data_type == UNCOMPRESSED_ARTIFACTS_VALUE: 
            return ModelArtifactConfig(
                  ArtifactLocation=self._dst_s3_accessor.get_uncompresssed_inference_artifact_s3_reference(
                      model_specs
                  ).get_uri()
            )
        else:
            return ModelArtifactConfig(
                ArtifactLocation=self._dst_s3_accessor.get_inference_artifact_s3_reference(
                    model_specs
                ).get_uri()
            )
        
    def _get_model_script_config(self, model_specs: JumpStartModelSpecs) -> ModelArtifactConfig:
        if model_specs.hosting_artifact_s3_data_type == UNCOMPRESSED_ARTIFACTS_VALUE: 
            return ScriptConfig(
                  ScriptLocation=self._dst_s3_accessor.get_uncompresssed_inference_artifact_s3_reference(
                    model_specs
                ).get_uri()
            )
        else:
            return ScriptConfig(
                ScriptLocation=self._dst_s3_accessor.get_inference_script_s3_reference(
                    model_specs
                ).get_uri()
            )

    def _make_hub_content_default_deployment_config(
        self, model_specs: JumpStartModelSpecs
    ) -> DefaultDeploymentConfig:
        """Creates deployment config for hub content"""
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
                    Environment=environment_variables,
                ),
            ),
            FrameworkImageConfig=FrameworkImageConfig(
                Framework=model_specs.hosting_ecr_specs.framework,
                FrameworkVersion=model_specs.hosting_ecr_specs.framework_version,
                PythonVersion=model_specs.hosting_ecr_specs.py_version,
                TransformersVersion=getattr(
                    model_specs.hosting_ecr_specs, "huggingface_transformers_version", None
                ),
                BaseFramework=hosting_base_framework(model_specs=model_specs),
            ),
            ModelArtifactConfig=self._get_model_artifact_config(model_specs),
            ScriptConfig=self._get_model_script_config(model_specs),
            InstanceConfig=InstanceConfig(
                DefaultInstanceType=model_specs.default_inference_instance_type,
                InstanceTypeOptions=model_specs.supported_inference_instance_types or [],
            ),
            InferenceNotebookConfig=InferenceNotebookConfig(
                NotebookLocation=self._dst_s3_accessor.get_demo_notebook_s3_reference(
                    model_specs
                ).get_uri()
            ),
            CustomImageConfig=None,
        )

    def _get_training_artifact_config(self, model_specs: JumpStartModelSpecs) -> ModelArtifactConfig:
        if model_specs.training_artifact_s3_data_type == UNCOMPRESSED_ARTIFACTS_VALUE: 
            return ModelArtifactConfig(
                  ArtifactLocation=self._dst_s3_accessor.get_uncompresssed_training_artifact_s3_reference(
                      model_specs
                  ).get_uri()
            )
        else:
            return ModelArtifactConfig(
                ArtifactLocation=self._dst_s3_accessor.get_training_artifact_s3_reference(
                    model_specs
                ).get_uri()
            )

    def _make_hub_content_default_training_config(
        self, model_specs: JumpStartModelSpecs
    ) -> Optional[DefaultTrainingConfig]:
        """Creates training config for hub content"""
        if not model_specs.training_supported:
            return None

        return DefaultTrainingConfig(
            SdkArgs=DefaultTrainingSdkArgs(
                MinSdkVersion=model_specs.min_sdk_version,
                SdkEstimatorArgs=None,  # TODO: Current Palatine schema doesn't have needed values
            ),
            CustomImageConfig=None,
            FrameworkImageConfig=FrameworkImageConfig(
                Framework=model_specs.training_ecr_specs.framework,
                FrameworkVersion=model_specs.training_ecr_specs.framework_version,
                PythonVersion=model_specs.training_ecr_specs.py_version,
                TransformersVersion=getattr(
                    model_specs.training_ecr_specs, "huggingface_transformers_version", None
                ),
                BaseFramework=training_base_framework(model_specs=model_specs),
            ),
            ModelArtifactConfig=self._get_training_artifact_config(model_specs),
            ScriptConfig=ScriptConfig(
                ScriptLocation=self._dst_s3_accessor.get_training_script_s3_reference(
                    model_specs
                ).get_uri()
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
        """Retrieves the DatasetConfig for JumpStartModelSpecs"""
        if not model_specs.training_supported:
            return None
        return DatasetConfig(
            TrainingDatasetLocation=self._dst_s3_accessor.get_default_training_dataset_s3_reference(
                model_specs
            ).get_uri(),
            ValidationDatasetLocation=None,
            DataFormatLocation=None,
            PredictColumn=None,
        )
