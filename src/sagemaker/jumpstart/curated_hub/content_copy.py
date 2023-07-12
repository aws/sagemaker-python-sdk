import time
from typing import Optional, Dict, Any

from botocore.client import BaseClient

from sagemaker import model_uris, script_uris
from sagemaker.jumpstart.curated_hub.utils import get_model_framework, find_objects_under_prefix, construct_s3_uri
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

EXTRA_S3_COPY_ARGS = {"ACL": "bucket-owner-full-control", "Tagging": "SageMaker=true"}


def _src_notebook_key(model_specs: JumpStartModelSpecs) -> str:
    framework = get_model_framework(model_specs)

    return f"{framework}-notebooks/{model_specs.model_id}-inference.ipynb"


def _src_markdown_key(model_specs: JumpStartModelSpecs) -> str:
    framework = get_model_framework(model_specs)

    return f"{framework}-metadata/{model_specs.model_id}.md"


def dst_markdown_key(model_specs: JumpStartModelSpecs) -> str:
    # Studio excepts the same key format as the bucket
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

    def copy_hub_content_dependencies_to_hub_bucket(self, model_specs: JumpStartModelSpecs) -> None:
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
        dst_bucket = self.dst_bucket()

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
            self.dst_inference_artifact_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        if not model_specs.supports_prepacked_inference():
            # Need to also copy script if prepack not enabled
            self._s3_client.copy(
                script_copy_source,
                dst_bucket,
                self.dst_inference_script_key(model_specs=model_specs),
                ExtraArgs=EXTRA_S3_COPY_ARGS,
            )

    def _copy_training_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        src_training_artifact_location = self._src_training_artifact_location(
            model_specs=model_specs
        )
        src_training_script_location = self._src_training_script_location(model_specs=model_specs)
        dst_bucket = self.dst_bucket()

        training_artifact_copy_source = {
            "Bucket": src_training_artifact_location.lstrip("s3://").split("/")[0],
            "Key": "/".join(src_training_artifact_location.lstrip("s3://").split("/")[1:]),
        }
        print(
            f"Copy artifact from {training_artifact_copy_source} to {dst_bucket} / {self.dst_training_artifact_key(model_specs=model_specs)}"
        )
        self._s3_client.copy(
            training_artifact_copy_source,
            dst_bucket,
            self.dst_training_artifact_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        training_script_copy_source = {
            "Bucket": src_training_script_location.lstrip("s3://").split("/")[0],
            "Key": "/".join(src_training_script_location.lstrip("s3://").split("/")[1:]),
        }
        print(
            f"Copy script from {training_script_copy_source} to {dst_bucket} / {self.dst_training_script_key(model_specs=model_specs)}"
        )
        self._s3_client.copy(
            training_script_copy_source,
            dst_bucket,
            self.dst_training_script_key(model_specs=model_specs),
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        print(
            f"Copy model {model_specs.model_id} version {model_specs.version} to curated hub bucket {dst_bucket} complete!"
        )

        self._copy_training_dataset_dependencies(model_specs=model_specs)

    def _copy_training_dataset_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        # TODO performance: copy in parallel
        src_bucket = self.src_bucket()
        dst_bucket = self.dst_bucket()
        src_dataset_prefix = self._src_training_dataset_prefix(model_specs=model_specs)
        training_dataset_keys = find_objects_under_prefix(
            bucket=src_bucket,
            prefix=src_dataset_prefix,
            s3_client=self._s3_client,
        )
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
                training_dataset_copy_source,
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
            "Key": _src_notebook_key(model_specs),
        }

        print(
            f"Copying notebook for {model_specs.model_id} at {artifact_copy_source} "
            f"to curated hub bucket {self.dst_bucket()}..."
        )

        self._s3_client.copy(
            artifact_copy_source,
            self.dst_bucket(),
            self.dst_notebook_key(model_specs=model_specs),
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
            "Key": _src_markdown_key(model_specs),
        }
        extra_args = {"ACL": "bucket-owner-full-control", "Tagging": "SageMaker=true"}

        print(
            f"Copying notebook for {model_specs.model_id} at {artifact_copy_source} to"
            f" curated hub bucket {self.dst_bucket()}..."
        )

        self._s3_client.copy(
            artifact_copy_source,
            self.dst_bucket(),
            dst_markdown_key(model_specs=model_specs),
            ExtraArgs=extra_args,
        )

        print(
            f"Copying notebook for {model_specs.model_id} at {artifact_copy_source} to curated hub bucket successful!"
        )

    def _src_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        studio_model_metadata = self.studio_metadata_map[model_specs.model_id]
        return studio_model_metadata["defaultDataKey"]

    def _src_training_dataset_location(self, model_specs: JumpStartModelSpecs) -> str:
        return construct_s3_uri(
            bucket=self.src_bucket(), key=self._src_training_dataset_prefix(model_specs=model_specs)
        )

    def _dst_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str:
        return self._src_training_dataset_prefix(model_specs=model_specs)  # TODO determine best way to copy datasets

    def dst_training_dataset_location(self, model_specs: JumpStartModelSpecs) -> str:
        return construct_s3_uri(
            bucket=self.dst_bucket(), key=self._dst_training_dataset_prefix(model_specs=model_specs)
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

    def src_bucket(self) -> str:
        return get_jumpstart_content_bucket(self._region)

    def dst_bucket(self) -> str:
        # TODO sync with create hub bucket logic
        return self._curated_hub_name

    def dst_inference_artifact_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/infer.tar.gz"

    def dst_training_artifact_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/train.tar.gz"

    def dst_inference_script_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/sourcedir.tar.gz"

    def dst_training_script_key(self, model_specs: JumpStartModelSpecs) -> str:
        # TODO sync with Studio copy logic
        return f"{model_specs.model_id}/{self._disambiguator}/training/sourcedir.tar.gz"

    def dst_notebook_key(self, model_specs: JumpStartModelSpecs) -> str:
        return f"{model_specs.model_id}/{self._disambiguator}/demo-notebook.ipynb"

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
