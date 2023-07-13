import time
from typing import Optional, Dict, Any

from botocore.client import BaseClient

from sagemaker import model_uris, script_uris
from sagemaker.jumpstart.curated_hub.utils import get_model_framework, find_objects_under_prefix, construct_s3_uri, get_bucket_and_key_from_s3_uri
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.filesystem.public_hub_s3_filesystem import PublicHubS3Filesystem
from sagemaker.jumpstart.curated_hub.filesystem.curated_hub_s3_filesystem import CuratedHubS3Filesystem
from sagemaker.jumpstart.curated_hub.filesystem.s3_object_reference import S3ObjectReference, create_s3_object_reference_from_bucket_and_key, create_s3_object_reference_from_uri

from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

EXTRA_S3_COPY_ARGS = {"ACL": "bucket-owner-full-control", "Tagging": "SageMaker=true"}

# TODO: Delete after the refactor
def _src_markdown_key(model_specs: JumpStartModelSpecs) -> str:
    framework = get_model_framework(model_specs)

    return f"{framework}-metadata/{model_specs.model_id}.md"


def dst_markdown_key(model_specs: JumpStartModelSpecs) -> str:
    
    return _src_markdown_key(model_specs)


class ContentCopier:
    """Copies content from JS source bucket to hub bucket."""

    def __init__(
        self, region: str, s3_client: BaseClient, curated_hub_s3_bucket_name: str, studio_metadata_map: Dict[str, Any]
    ) -> None:
        """Sets up basic info."""
        self._region = region
        self._s3_client = s3_client
        self._curated_hub_name = curated_hub_s3_bucket_name
        self.studio_metadata_map = studio_metadata_map
        self._disambiguator = time.time()

        self._src_s3_filesystem = PublicHubS3Filesystem(region) # TODO: pass this one in
        self._dst_s3_filesystem = CuratedHubS3Filesystem(region, curated_hub_s3_bucket_name) # TODO: only init once the actual destination is found

    def copy_hub_content_dependencies_to_hub_bucket(self, model_specs: JumpStartModelSpecs) -> None:
        """Copies artifact and script tarballs into the hub bucket.

        Unfortunately, this logic is duplicated/inconsistent with what is in Studio."""
        self._copy_inference_dependencies(model_specs)
        self._copy_demo_notebook_dependencies(model_specs)
        self._copy_markdown_dependencies(model_specs)

        if model_specs.training_supported:
            self._copy_training_dependencies(model_specs)

    def _copy_inference_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        src_inference_artifact_location = self._src_s3_filesystem.get_inference_artifact_s3_reference(model_specs)
        dst_artifact_reference = self._dst_s3_filesystem.get_inference_artifact_s3_reference(model_specs)
        self._copy_s3_reference("inference artifact", src_inference_artifact_location, dst_artifact_reference)

        if not model_specs.supports_prepacked_inference():
            # Need to also copy script if prepack not enabled
            src_inference_script_location = self._src_s3_filesystem.get_inference_script_s3_reference(model_specs)
            dst_inference_script_reference = self._dst_s3_filesystem.get_inference_script_s3_reference(model_specs)

            self._copy_s3_reference("inference script", src_inference_script_location, dst_inference_script_reference)

    def _copy_training_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        dst_bucket = self.dst_bucket()

        src_training_artifact_location = self._src_s3_filesystem.get_training_artifact_s3_reference(model_specs)
        print(
            f"Copy training artifact from {src_training_artifact_location.key} to {dst_bucket} / {self.dst_training_artifact_key(model_specs=model_specs)}"
        )
        self._s3_client.copy(
            src_training_artifact_location.format_for_s3_copy(),
            dst_bucket,
            self.dst_training_artifact_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        src_training_script_location = self._src_s3_filesystem.get_training_script_s3_reference(model_specs)
        print(
            f"Copy training script from {src_training_script_location.key} to {dst_bucket} / {self.dst_training_script_key(model_specs=model_specs)}"
        )
        self._s3_client.copy(
            src_training_script_location.format_for_s3_copy(),
            dst_bucket,
            self.dst_training_script_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        print(
            f"Copy training dependencies for {model_specs.model_id} version {model_specs.version} to curated hub bucket {dst_bucket} complete!"
        )

        self._copy_training_dataset_dependencies(model_specs=model_specs)

    def _copy_training_dataset_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        # TODO performance: copy in parallel
        training_dataset_s3_references = self._src_s3_filesystem.get_default_training_dataset_s3_reference(model_specs)
        dst_bucket = self.dst_bucket()
        for s3_reference in training_dataset_s3_references:
            dst_key = s3_reference.key  # Use same dataset key in the hub bucket as notebooks may expect this location
            print(
                f"Copy dataset file from {s3_reference.key} to {dst_bucket} / {dst_key}"
            )
            self._s3_client.copy(
                s3_reference.format_for_s3_copy(),
                dst_bucket,
                dst_key,
                ExtraArgs=EXTRA_S3_COPY_ARGS,
            )

    def _copy_demo_notebook_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        notebook_s3_reference = self._src_s3_filesystem.get_demo_notebook_s3_reference(model_specs)

        print(
            f"Copying notebook for {model_specs.model_id} at {notebook_s3_reference.key} "
            f"to curated hub bucket {self.dst_bucket()}..."
        )

        self._s3_client.copy(
            notebook_s3_reference.format_for_s3_copy(),
            self.dst_bucket(),
            self.dst_notebook_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        print(
            f"Copying notebook for {model_specs.model_id} at {notebook_s3_reference.key} to curated hub bucket successful!"
        )

    def _copy_markdown_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        markdown_s3_reference = self._src_s3_filesystem.get_markdown_s3_reference(model_specs)

        print(
            f"Copying markdown for {model_specs.model_id} at {markdown_s3_reference.key} to"
            f" curated hub bucket {self.dst_bucket()}..."
        )

        self._s3_client.copy(
            markdown_s3_reference.format_for_s3_copy(),
            self.dst_bucket(),
            markdown_s3_reference.key, # Studio expects the same key format as the bucket
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        print(
            f"Copying markdown for {model_specs.model_id} at {markdown_s3_reference.key} to curated hub bucket successful!"
        )

    # TODO: determine if safe to delete after refactor
    def _src_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        studio_model_metadata = self.studio_metadata_map[model_specs.model_id]
        return studio_model_metadata["defaultDataKey"]


    def _dst_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        return self._src_training_dataset_prefix(model_specs=model_specs)  # TODO determine best way to copy datasets

    def dst_training_dataset_location(self, model_specs: JumpStartModelSpecs) -> str:
        return construct_s3_uri(
            bucket=self.dst_bucket(), key=self._dst_training_dataset_prefix(model_specs=model_specs)
        )
    
    def dst_bucket(self) -> str:
        # TODO sync with create hub bucket logic
        return self._curated_hub_name

    def dst_inference_artifact_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/infer.tar.gz"
    
    def dst_inference_script_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/sourcedir.tar.gz"

    def dst_training_artifact_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/train.tar.gz"

    def dst_training_script_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/training/sourcedir.tar.gz"

    def dst_notebook_key(self, model_specs: JumpStartModelSpecs) -> str:
        return f"{model_specs.model_id}/{self._disambiguator}/demo-notebook.ipynb"


    def _copy_s3_reference(self, resource_name: str, src: S3ObjectReference, dst: S3ObjectReference):
        print(
            f"Copying {resource_name} from {src.bucket}/{src.key} to {dst.bucket}/{dst.key}..."
        )

        self._s3_client.copy(
            src.format_for_s3_copy(),
            dst.bucket,
            dst.key,
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        print(
            f"Copying {resource_name} from {src.bucket}/{src.key} to {dst.bucket}/{dst.key} complete!"
        )