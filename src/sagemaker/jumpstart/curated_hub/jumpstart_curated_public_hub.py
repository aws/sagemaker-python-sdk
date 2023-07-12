from __future__ import absolute_import

import json
import time
from dataclasses import asdict
from typing import List, Optional

import boto3
from botocore.client import ClientError

from sagemaker import model_uris, script_uris
from sagemaker import environment_variables as env_vars
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
from sagemaker.jumpstart.curated_hub.hub_client import CuratedHubClient
from sagemaker.jumpstart.curated_hub.utils import get_studio_model_metadata_map_from_region, find_objects_under_prefix, \
    PublicModelId
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs, get_jumpstart_content_bucket,
)
from sagemaker.session import Session

EXTRA_S3_COPY_ARGS = {"ACL": "bucket-owner-full-control", "Tagging": "SageMaker=true"}


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
        self._hub_client = CuratedHubClient(curated_hub_name=self.curated_hub_name, region=self._region)
        self._sagemaker_session = Session()
        self._disambiguator = time.time()

        self.studio_metadata_map = get_studio_model_metadata_map_from_region(region=self._region)

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
            if ex.response["Error"]["Code"] != "ResourceInUse":
                raise ex

    def _get_curated_hub(self):
        self._sm_client.describe_hub(HubName=self.curated_hub_name)

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
            if ex.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise ex

    def _call_create_bucket(self, bucket_name: str):
        # TODO make sure bucket policy permits PutObjectTagging so bucket-to-bucket copy will work
        self._s3_client.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": self._region}
        )

    def import_models(self, model_ids: List[PublicModelId]):
        """Imports models in list to curated hub

        If the model already exists in the curated hub, it will skip the upload."""
        print(f"Importing {len(model_ids)} models to curated private hub...")
        for model_id in model_ids:
            self._import_model(model_id)

    def _import_model(self, public_js_model: PublicModelId) -> None:
        print(
            f"Importing model {public_js_model.id} version {public_js_model.version} to curated private hub..."
        )
        model_specs = verify_model_region_and_return_specs(
            model_id=public_js_model.id,
            version=public_js_model.version,
            scope=JumpStartScriptScope.INFERENCE,
            region=self._region,
        )

        self._copy_hub_content_dependencies_to_hub_bucket(model_specs=model_specs)

        # self._import_public_model_to_hub_no_overwrite(model_specs=model_specs)
        self._import_public_model_to_hub(model_specs=model_specs)
        print(
            f"Importing model {public_js_model.id} version {public_js_model.version} to curated private hub complete!"
        )

    def _import_public_model_to_hub_no_overwrite(self, model_specs: JumpStartModelSpecs):
        try:
            self._import_public_model_to_hub(model_specs)
        except ClientError as ex:
            if ex.response["Error"]["Code"] != "ResourceInUse":
                raise ex

    def _import_public_model_to_hub(self, model_specs: JumpStartModelSpecs):
        # TODO Several fields are not present in SDK specs as they are only in Studio specs right now (not urgent)
        hub_content_display_name = model_specs.model_id
        hub_content_description = f"This is the very informative {model_specs.model_id} description"
        hub_content_markdown = self._construct_s3_uri(self._dst_bucket(), self._dst_markdown_key(model_specs))

        hub_content_document = self._make_hub_content_document(model_specs=model_specs)

        self._sm_client.import_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=model_specs.model_id,
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
        self._copy_inference_dependencies(model_specs)
        self._copy_demo_notebook_dependencies(model_specs)
        self._copy_markdown_dependencies(model_specs)

        if model_specs.training_supported:
            self._copy_training_dependencies(model_specs)

    def _copy_inference_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        src_inference_artifact_location = self._src_inference_artifact_location(
            model_specs=model_specs
        )
        src_inference_script_location = self._src_inference_script_location(model_specs=model_specs)
        dst_bucket = self._dst_bucket()

        print(
            f"Copying model {model_specs.model_id} version {model_specs.version} to curated hub bucket {dst_bucket}..."
        )

        artifact_copy_source = {
            "Bucket": src_inference_artifact_location.lstrip("s3://").split("/")[0],
            "Key": "/".join(src_inference_artifact_location.lstrip("s3://").split("/")[1:]),
        }
        script_copy_source = {
            "Bucket": src_inference_script_location.lstrip("s3://").split("/")[0],
            "Key": "/".join(src_inference_script_location.lstrip("s3://").split("/")[1:]),
        }

        self._s3_client.copy(
            artifact_copy_source,
            dst_bucket,
            self._dst_inference_artifact_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        if not model_specs.supports_prepacked_inference():
            # Need to also copy script if prepack not enabled
            self._s3_client.copy(
                script_copy_source,
                dst_bucket,
                self._dst_inference_script_key(model_specs=model_specs),
                ExtraArgs=EXTRA_S3_COPY_ARGS,
            )

    def _copy_training_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        src_training_artifact_location = self._src_training_artifact_location(
            model_specs=model_specs
        )
        src_training_script_location = self._src_training_script_location(model_specs=model_specs)
        src_bucket = self._src_bucket()
        dst_bucket = self._dst_bucket()

        training_artifact_copy_source = {
            "Bucket": src_training_artifact_location.lstrip("s3://").split("/")[0],
            "Key": "/".join(src_training_artifact_location.lstrip("s3://").split("/")[1:]),
        }
        print(
            f"Copy artifact from {training_artifact_copy_source} to {dst_bucket} / {self._dst_training_artifact_key(model_specs=model_specs)}"
        )
        self._s3_client.copy(
            training_artifact_copy_source,
            dst_bucket,
            self._dst_training_artifact_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        training_script_copy_source = {
            "Bucket": src_training_script_location.lstrip("s3://").split("/")[0],
            "Key": "/".join(src_training_script_location.lstrip("s3://").split("/")[1:]),
        }
        print(
            f"Copy script from {training_script_copy_source} to {dst_bucket} / {self._dst_training_script_key(model_specs=model_specs)}"
        )
        self._s3_client.copy(
            training_script_copy_source,
            dst_bucket,
            self._dst_training_script_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        print(
            f"Copy model {model_specs.model_id} version {model_specs.version} to curated hub bucket {dst_bucket} complete!"
        )

        src_dataset_prefix = self._src_training_dataset_prefix(model_specs=model_specs)
        training_dataset_keys = find_objects_under_prefix(
            bucket=src_bucket,
            prefix=src_dataset_prefix,
            s3_client=self._s3_client,
        )

        # TODO performance: copy in parallel
        for s3_key in training_dataset_keys:
            training_dataset_copy_source = {
                "Bucket": src_bucket,
                "Key": s3_key,
            }
            dst_key = s3_key  # Use same dataset key in the hub bucket as notebooks may expect this location
            print(
                f"Copy dataset file from {training_dataset_copy_source} to {dst_bucket} / {dst_key}"
            )
            self._s3_client.copy(
                training_script_copy_source,
                dst_bucket,
                dst_key,
                ExtraArgs=EXTRA_S3_COPY_ARGS,
            )

    def _copy_demo_notebook_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        src_inference_artifact_location = self._src_inference_artifact_location(
            model_specs=model_specs
        )
        artifact_copy_source = {
            "Bucket": src_inference_artifact_location.lstrip("s3://").split("/")[0],
            "Key": self._src_notebook_key(model_specs),
        }
        dst_bucket = self._dst_bucket()

        print(
            f"Copying notebook for {model_specs.model_id} at {artifact_copy_source} to curated hub bucket {dst_bucket}..."
        ) 

        self._s3_client.copy(
            artifact_copy_source,
            dst_bucket,
            self._dst_notebook_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        print(
            f"Copying notebook for {model_specs.model_id} at {artifact_copy_source} to curated hub bucket successful!"
        ) 

    def _copy_markdown_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        src_inference_artifact_location = self._src_inference_artifact_location(
            model_specs=model_specs
        )
        artifact_copy_source = {
            "Bucket": src_inference_artifact_location.lstrip("s3://").split("/")[0],
            "Key": self._src_markdown_key(model_specs),
        }
        dst_bucket = self._dst_bucket()
        extra_args = {"ACL": "bucket-owner-full-control", "Tagging": "SageMaker=true"}

        print(
            f"Copying notebook for {model_specs.model_id} at {artifact_copy_source} to curated hub bucket {dst_bucket}..."
        ) 

        self._s3_client.copy(
            artifact_copy_source,
            dst_bucket,
            self._dst_markdown_key(model_specs=model_specs),
            ExtraArgs=extra_args,
        )

        print(
            f"Copying notebook for {model_specs.model_id} at {artifact_copy_source} to curated hub bucket successful!"
        ) 

    def _make_hub_content_document(self, model_specs: JumpStartModelSpecs) -> str:
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

    def _dataset_config(self, model_specs: JumpStartModelSpecs) -> Optional[DatasetConfig]:
        if not model_specs.training_supported:
            return None
        return DatasetConfig(
            TrainingDatasetLocation=self._dst_training_dataset_location(model_specs=model_specs),
            ValidationDatasetLocation=None,
            DataFormatLocation=None,
            PredictColumn=None,
        )

    def _src_training_artifact_location(self, model_specs: JumpStartModelSpecs) -> Optional[str]:
        return self._src_artifact_location(JumpStartScriptScope.TRAINING, model_specs)

    def _src_inference_artifact_location(self, model_specs: JumpStartModelSpecs) -> str:
        return self._src_artifact_location(JumpStartScriptScope.INFERENCE, model_specs)

    def _src_artifact_location(self, model_scope: str, model_specs: JumpStartModelSpecs) -> str:
        return model_uris.retrieve(
            region=self._region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            model_scope=model_scope,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )

    def _src_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        studio_model_metadata = self.studio_metadata_map[model_specs.model_id]
        return studio_model_metadata["defaultDataKey"]

    def _src_training_dataset_location(self, model_specs: JumpStartModelSpecs) -> str:
        return self._construct_s3_uri(
            bucket=self._src_bucket(), key=self._src_training_dataset_prefix(model_specs=model_specs)
        )

    def _dst_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        return self._src_training_dataset_prefix(model_specs=model_specs)  # TODO determine best way to copy datasets

    def _dst_training_dataset_location(self, model_specs: JumpStartModelSpecs) -> str:
        return self._construct_s3_uri(
            bucket=self._dst_bucket(), key=self._dst_training_dataset_prefix(model_specs=model_specs)
        )

    def _src_training_script_location(self, model_specs: JumpStartModelSpecs) -> str:
        return self._src_script_location(JumpStartScriptScope.TRAINING, model_specs)

    def _src_inference_script_location(self, model_specs: JumpStartModelSpecs) -> str:
        return self._src_script_location(JumpStartScriptScope.INFERENCE, model_specs)

    def _src_script_location(self, model_scope: str, model_specs: JumpStartModelSpecs) -> str:
        return script_uris.retrieve(
            region=self._region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            script_scope=model_scope,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )

    def _src_bucket(self) -> str:
        return get_jumpstart_content_bucket(self._region)

    def _dst_bucket(self) -> str:
        # TODO sync with create hub bucket logic
        return self.curated_hub_name

    def _dst_inference_artifact_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/infer.tar.gz"

    def _dst_training_artifact_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/train.tar.gz"

    def _dst_inference_script_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/sourcedir.tar.gz"

    def _dst_training_script_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/training-sourcedir.tar.gz"
    
    def _src_notebook_key(self, model_specs: JumpStartModelSpecs) -> str:
        framework = self._get_model_framework(model_specs)
        
        return f"{framework}-notebooks/{model_specs.model_id}-inference.ipynb"
    
    def _src_markdown_key(self, model_specs: JumpStartModelSpecs) -> str:
        framework = self._get_model_framework(model_specs)

        return f"{framework}-metadata/{model_specs.model_id}.md"
    
    def _get_model_framework(self, model_specs: JumpStartModelSpecs) -> str:
        return model_specs.model_id.split("-")[0]
    
    def _dst_notebook_key(self, model_specs: JumpStartModelSpecs) -> str:
        return f"{model_specs.model_id}/{self._disambiguator}/demo-notebook.ipynb"
    
    def _dst_markdown_key(self, model_specs: JumpStartModelSpecs) -> str:
        # Studio excepts the same key format as the bucket
        return self._src_markdown_key(model_specs)

    def _construct_s3_uri(self, bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key}"

    def _base_framework(self, model_specs: JumpStartModelSpecs) -> Optional[str]:
        if model_specs.hosting_ecr_specs.framework == "huggingface":
            return f"pytorch{model_specs.hosting_ecr_specs.framework_version}"
        return None

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
                BaseFramework=self._base_framework(model_specs=model_specs),
            ),
            ModelArtifactConfig=ModelArtifactConfig(
                ArtifactLocation=self._construct_s3_uri(
                    self._dst_bucket(), self._dst_inference_artifact_key(model_specs=model_specs)
                ),
            ),
            ScriptConfig=ScriptConfig(
                ScriptLocation=self._construct_s3_uri(
                    self._dst_bucket(), self._dst_inference_script_key(model_specs=model_specs)
                ),
            ),
            InstanceConfig=InstanceConfig(
                DefaultInstanceType=model_specs.default_inference_instance_type,
                InstanceTypeOptions=model_specs.supported_inference_instance_types or [],
            ),
            InferenceNotebookConfig=InferenceNotebookConfig(
                NotebookLocation=self._construct_s3_uri(self._dst_bucket(), self._dst_notebook_key(model_specs=model_specs))
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
                BaseFramework=self._base_framework(model_specs=model_specs),
            ),
            ModelArtifactConfig=ModelArtifactConfig(
                ArtifactLocation=self._construct_s3_uri(
                    self._dst_bucket(), self._dst_training_artifact_key(model_specs=model_specs)
                ),
            ),
            ScriptConfig=ScriptConfig(
                ScriptLocation=self._construct_s3_uri(
                    self._dst_bucket(), self._dst_training_script_key(model_specs=model_specs)
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

    def delete_models(self, model_ids: List[PublicModelId]):
        for model_id in model_ids:
            self._hub_client.delete_model(model_id)
