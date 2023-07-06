from __future__ import absolute_import

import json
from dataclasses import dataclass, asdict
import time

from typing import List

import boto3
from botocore.client import ClientError

from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import (
    HubModelSpec_v1_0_0,
    DefaultDeploymentConfig,
    DefaultDeploymentSdkArgs,
    FrameworkImageConfig,
    ModelArtifactConfig,
    ScriptConfig,
    InstanceConfig,
    InferenceNotebookConfig,
)
from sagemaker import model_uris, script_uris
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
from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import ModelCapabilities


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
        self.curated_hub_name = f"{curated_hub_s3_prefix}"  # TODO verify name

        self._region = "us-west-2"
        self._s3_client = boto3.client("s3", region_name=self._region)
        self._sm_client = boto3.client("sagemaker", region_name=self._region)
        self._sagemaker_session = Session()

    def create(self):
        """Creates a curated hub in the caller AWS account.
        
        If the S3 bucket does not exist, this will create a new one. 
        If the curated hub does not exist, this will create a new one."""
        self._get_or_create_s3_bucket(self.curated_hub_name)
        self._get_or_create_curated_hub()


    def _get_or_create_curated_hub(self):
        try:
            return self._create_curated_hub()
        except ClientError as ex:
            if ex.response['Error']['Code'] != 'ResourceLimitExceeded':
              raise ex

    def _create_curated_hub(self):
      hub_bucket_s3_uri = f"s3://{self.curated_hub_name}"
      self._sm_client.create_hub(
          HubName=self.curated_hub_name,
          HubDescription="This is a curated hub.",  # TODO verify description
          HubDisplayName=self.curated_hub_name,
          HubSearchKeywords=[],
          S3StorageConfig={
              "S3OutputPath": hub_bucket_s3_uri,
          },
          Tags=[],
      )

    def _get_or_create_s3_bucket(self, bucket_name: str):
        try:
            return self._call_create_bucket(bucket_name)
        except ClientError as ex:
            if ex.response['Error']['Code'] != 'BucketAlreadyExists':
                raise ex

    def _call_create_bucket(self, bucket_name: str):
        self._s3_client.create_bucket(Bucket=bucket_name)

    def import_models(self, model_ids: List[PublicModelId]):
        """Imports models in list to curated hub
        
        If the model already exists in the curated hub, it will skip the upload."""
        for model_id in model_ids:
            self._import_model(model_id)

    def _import_model(self, public_js_model: PublicModelId) -> None:
        model_specs = verify_model_region_and_return_specs(
            model_id=public_js_model.id,
            version=public_js_model.version,
            scope=JumpStartScriptScope.INFERENCE,
            region=self._region,
        )

        # TODO verify ideal naming (not urgent)
        hub_content_name = f"{model_specs.model_id}-copy"

        # TODO Several fields are not present in SDK specs as they are only in Studio specs right now (not urgent)
        hub_content_display_name = model_specs.model_id
        hub_content_description = f"This is the very informative {model_specs.model_id} description"
        hub_content_markdown = (
            f"This is the {model_specs.model_id} markdown"  # TODO markdown file needs loading
        )

        self._copy_hub_content_dependencies_to_hub_bucket(model_specs=model_specs)

        hub_content_document = self._make_hub_content_document(model_specs=model_specs)
        self._sm_client.import_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=hub_content_name,
            HubContentType="Model",
            DocumentSchemaVersion="1.0.0",
            HubContentDisplayName=hub_content_display_name,
            HubContentDescription=hub_content_description,
            HubContentMarkdown=hub_content_markdown,
            HubContentDocument=hub_content_document,
            HubContentSearchKeywords=[],
            Tags=[],
        )

    def _copy_hub_content_dependencies_to_hub_bucket(
        self, model_specs: JumpStartModelSpecs
    ) -> None:
        """Copies artifact and script tarballs into the hub bucket.

        Unfortunately, this logic is duplicated/inconsistent with what is in Studio."""
        src_inference_artifact_location = self._src_inference_artifact_location(model_specs=model_specs)
        src_inference_script_location = self._src_inference_script_location(model_specs=model_specs)
        dst_bucket = self._dst_bucket()

        artifact_copy_source = {
            'Bucket': src_inference_artifact_location.lstrip("s3://").split('/')[0],
            'Key': '/'.join(src_inference_artifact_location.lstrip("s3://").split('/')[1:])
        }
        script_copy_source = {
            'Bucket': src_inference_script_location.lstrip("s3://").split('/')[0],
            'Key': '/'.join(src_inference_script_location.lstrip("s3://").split('/')[1:])
        }
        self._s3_client.copy(
            artifact_copy_source, dst_bucket, self._dst_inference_artifact_key(model_specs=model_specs)
        )
        self._s3_client.copy(script_copy_source, dst_bucket, self._dst_inference_script_key(model_specs=model_specs))


    def _make_hub_content_document(self, model_specs: JumpStartModelSpecs) -> str:
        """Converts the provided JumpStartModelSpecs into a Hub Content Document."""
        hub_model_spec = HubModelSpec_v1_0_0(
            Capabilities=[ModelCapabilities.VALIDATION],  # TODO add inference if needed?
            DataType="",  # TODO not in SDK metadata
            MlTask="",  # TODO not in SDK metadata
            Framework=model_specs.hosting_ecr_specs.framework,
            Origin=None,
            Dependencies=[],  # TODO add references to copied artifacts
            DatasetConfig=None,  # Out of scope in p0
            DefaultTrainingConfig=None,  # Out of scope in p0
            DefaultDeploymentConfig=self.make_hub_content_deployment_config(
                model_specs=model_specs,
            )
        )

        return json.dumps(asdict(hub_model_spec))  # TODO verify/fix string representation

    def make_hub_content_deployment_config(self, model_specs: JumpStartModelSpecs) -> DefaultDeploymentConfig:
        """Creates a DefaultDeploymentConfig from the provided JumpStartModelSpecs."""
        return DefaultDeploymentConfig(
            SdkArgs=DefaultDeploymentSdkArgs(
                MinSdkVersion=model_specs.min_sdk_version,
                SdkModelArgs=None,  # Out of scope in p0
            ),
            FrameworkImageConfig=FrameworkImageConfig(
                Framework=model_specs.hosting_ecr_specs.framework,
                FrameworkVersion=model_specs.hosting_ecr_specs.framework_version,
                PythonVersion=model_specs.hosting_ecr_specs.py_version,
                TransformersVersion=getattr(
                    model_specs.hosting_ecr_specs, "huggingface_transformers_version", None
                ),
                BaseFramework=None,  # TODO verify necessity
            ),
            ModelArtifactConfig=ModelArtifactConfig(
                ArtifactLocation=self._dst_inference_artifact_key(model_specs=model_specs),
            ),
            ScriptConfig=ScriptConfig(
                ScriptLocation=self._dst_inference_script_key(model_specs=model_specs),
            ),
            InstanceConfig=InstanceConfig(
                DefaultInstanceType=model_specs.default_inference_instance_type,
                InstanceTypeOptions=model_specs.supported_inference_instance_types or [],
            ),
            InferenceNotebookConfig=InferenceNotebookConfig(
                NotebookLocation="s3://foo/notebook"  # TODO not present in SDK metadata
            ),
            CustomImageConfig=None,
        )

    def _src_inference_artifact_location(self, model_specs: JumpStartModelSpecs) -> str:
        return model_uris.retrieve(
            region=self._region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            model_scope="inference",
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )

    def _src_inference_script_location(self, model_specs: JumpStartModelSpecs) -> str:
        return script_uris.retrieve(
            region=self._region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            script_scope="inference",
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )

    def _dst_bucket(self) -> str:
        # TODO sync with create hub bucket logic
        return self.curated_hub_name

    def _dst_inference_artifact_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{time.time()}/infer.tar.gz"

    def _dst_inference_script_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{time.time()}/sourcedir.tar.gz"
