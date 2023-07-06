import json
from dataclasses import dataclass
from typing import List

import boto3

from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs_v1_0_0 import HubModelSpec_v1_0_0, \
    DefaultDeploymentConfig, SdkArgs, FrameworkImageConfig, ModelArtifactConfig, ScriptConfig, \
    InstanceConfig, InferenceNotebookConfig
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.types import (
    JumpStartModelSpecs,
)
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


@dataclass
class PublicModelId:
    id: str
    version: str


class JumpStartCuratedPublicHub:
    """JumpStartCuratedPublicHub class.

    This class helps users create a new curated hub.
    """

    def __init__(self, curated_hub_s3_prefix: str):
        self.curated_hub_s3_prefix = curated_hub_s3_prefix
        self.curated_hub_name = f"{curated_hub_s3_prefix}-sagemaker-testing"  # TODO verify name

        self._region = "us-west-2"
        self._s3_client = boto3.client("s3", region_name=self._region)
        self._sm_client = boto3.client("sagemaker", region_name=self._region)
        self._sagemaker_session = Session()

    def create(self):
        """
       Workflow:
       1. Verifications
        a) S3 location exists
        b) if not creates one
       2. CreateHub
       """
        # TODO chrstfu@ if you could take this piece please
        self._s3_client.create_bucket(bucket=self.curated_hub_name)  # TODO create bucket if not exists, verify args

        hub_bucket_s3_uri = f"s3://{self.curated_hub_name}"
        response = self._sm_client.create_hub(
            HubName=self.curated_hub_name,
            HubDescription="This is a curated hub.",  # TODO verify description
            HubDisplayName=self.curated_hub_name,  # TODO verify display name
            HubSearchKeywords=[],
            S3StorageConfig={
                'S3OutputPath': hub_bucket_s3_uri,
            },
            Tags=[]
        )

    def import_models(self, model_ids: List[PublicModelId]):
        for model_id in model_ids:
            self.import_model(model_id)
        """
       Workflow:
       1. Fo reach model_id in list
       2. Pull model metadata
       3. Convert to "similar metadata". Might need new metadata version for Palatine
        a) HubContentDocument
        b) HubContentMetadata: https://quip-amazon.com/8Q9nAjsiVcqs/Palatine-API-service-side-schema-validation
        c) HubContentMarkdown
       4. Downloads public bucket S3 model data
        a) Finds the regional bucket
        b) Add 
       5. Copy over to hub bucket
        a) Checks if model ID already exists in Hub
       6. 
       """

    def import_model(self, public_js_model: PublicModelId) -> None:
        model_specs = verify_model_region_and_return_specs(
            model_id=public_js_model.id,
            version=public_js_model.version,
            scope=JumpStartScriptScope.INFERENCE,
            region=self._region
        )

        # TODO verify ideal naming (not urgent)
        hub_content_name = f"{model_specs.model_id}-copy"

        # TODO Several fields are not present in SDK specs as they are only in Studio specs right now (not urgent)
        hub_content_display_name = model_specs.model_id
        hub_content_description = f"This is the very informative {model_specs.model_id} description"
        hub_content_markdown = f"This is the {model_specs.model_id} markdown"  # TODO markdown file needs loading

        self.copy_hub_content_dependencies_to_hub_bucket(model_specs=model_specs)

        hub_content_document = self.make_hub_content_document(model_specs=model_specs)
        response = self._sm_client.import_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=hub_content_name,
            HubContentType="Model",
            DocumentSchemaVersion="1.0.0",
            HubContentDisplayName=hub_content_display_name,
            HubContentDescription=hub_content_description,
            HubContentMarkdown=hub_content_markdown,
            HubContentDocument=hub_content_document,
            HubContentSearchKeywords=[],
            Tags=[]
        )

    def copy_hub_content_dependencies_to_hub_bucket(self, model_specs: JumpStartModelSpecs) -> None:
        """Copies artifact and script tarballs into the hub bucket.

        Unfortunately, this logic is duplicated/inconsistent with what is in Studio."""
        pass  # TODO

    def make_hub_content_document(self, model_specs: JumpStartModelSpecs) -> str:
        """Converts the provided JumpStartModelSpecs into a Hub Content Document."""
        # TODO copy artifact and script tarballs
        copied_inference_artifact_location = "s3://foo"
        copied_inference_script_location = "s3://foo"

        hub_model_spec = HubModelSpec_v1_0_0(
            capabilities=[],  # TODO add inference if needed?
            DataType="",  # TODO not in SDK metadata
            MlTask="",  # TODO not in SDK metadata
            Framework=model_specs.hosting_ecr_specs.framework,
            Origin=None,
            Dependencies=[],  # TODO add references to copied artifacts
            DatasetConfig=None,  # Out of scope in p0
            DefaultTrainingConfig=None,  # Out of scope in p0
            DefaultDeploymentConfig=self.make_hub_content_deployment_config(
                model_specs=model_specs,
                copied_artifact_location=copied_inference_artifact_location,
                copied_script_location=copied_inference_script_location,
            )
        )
        return json.dumps(hub_model_spec)  # TODO verify/fix string representation

    def make_hub_content_deployment_config(
        self, model_specs: JumpStartModelSpecs, copied_artifact_location: str, copied_script_location: str
    ) -> DefaultDeploymentConfig:
        """Creates a DefaultDeploymentConfig from the provided JumpStartModelSpecs."""
        return DefaultDeploymentConfig(
            SdkArgs=SdkArgs(
                MinSdkVersion=model_specs.min_sdk_version,
                SdkEstimatorArgs=None,  # Out of scope in p0
            ),
            FrameworkImageConfig=FrameworkImageConfig(
                Framework=model_specs.hosting_ecr_specs.framework,
                FrameworkVersion=model_specs.hosting_ecr_specs.framework_version,
                PythonVersion=model_specs.hosting_ecr_specs.py_version,
                TransformersVersion=getattr(model_specs.hosting_ecr_specs, "huggingface_transformers_version", None),
                BaseFramework=None,  # TODO verify necessity
            ),
            ModelArtifactConfig=ModelArtifactConfig(
                ArtifactLocation=copied_artifact_location,
            ),
            ScriptConfig=ScriptConfig(
                ScriptLocation=copied_script_location,
            ),
            InstanceConfig=InstanceConfig(
                DefaultInstanceType=model_specs.default_inference_instance_type,
                InstanceTypeOptions=model_specs.supported_inference_instance_types or []
            ),
            InferenceNotebookConfig=InferenceNotebookConfig(
                NotebookLocation="s3://foo"  # TODO not present in SDK metadata
            ),
            CustomImageConfig=None,
        )
