import time
from typing import Optional, Dict, Any, List

from botocore.client import BaseClient
from dataclasses import dataclass
from concurrent import futures

from sagemaker import model_uris, script_uris
from sagemaker.jumpstart.curated_hub.utils import (
    get_model_framework,
    find_objects_under_prefix,
    construct_s3_uri,
    get_bucket_and_key_from_s3_uri,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.filesystem.public_hub_s3_filesystem import (
    PublicHubS3Filesystem,
)
from sagemaker.jumpstart.curated_hub.filesystem.curated_hub_s3_filesystem import (
    CuratedHubS3Filesystem,
)
from sagemaker.jumpstart.curated_hub.filesystem.s3_object_reference import (
    S3ObjectReference,
    create_s3_object_reference_from_bucket_and_key,
    create_s3_object_reference_from_uri,
)

from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

EXTRA_S3_COPY_ARGS = {"ACL": "bucket-owner-full-control", "Tagging": "SageMaker=true"}


@dataclass
class CopyContentConfig:
    src: S3ObjectReference
    dst: S3ObjectReference
    display_name: str
    is_dir: bool = False


class ContentCopier:
    """Copies content from JS source bucket to hub bucket."""

    def __init__(
        self,
        region: str,
        s3_client: BaseClient,
        src_s3_filesystem: PublicHubS3Filesystem,
        dst_s3_filesystem: CuratedHubS3Filesystem,  # TODO: abstract this
    ) -> None:
        """Sets up basic info."""
        self._region = region
        self._s3_client = s3_client
        self._disambiguator = time.time()
        self._thread_pool_size = 20

        self._src_s3_filesystem = src_s3_filesystem
        self._dst_s3_filesystem = dst_s3_filesystem

    def copy_hub_content_dependencies_to_hub_bucket(self, model_specs: JumpStartModelSpecs) -> None:
        """Copies artifact and script tarballs into the hub bucket.

        Unfortunately, this logic is duplicated/inconsistent with what is in Studio."""
        copy_configs = []

        copy_configs.extend(self._get_copy_configs_for_inference_dependencies(model_specs))
        copy_configs.extend(self._get_copy_configs_for_demo_notebook_dependencies(model_specs))
        copy_configs.extend(self._get_copy_configs_for_markdown_dependencies(model_specs))

        if model_specs.training_supported:
            copy_configs.extend(self._get_copy_configs_for_training_dependencies(model_specs))

        self._parallel_execute_copy_configs(copy_configs)

    def _get_copy_configs_for_inference_dependencies(
        self, model_specs: JumpStartModelSpecs
    ) -> List[CopyContentConfig]:
        copy_configs = []

        src_inference_artifact_location = (
            self._src_s3_filesystem.get_inference_artifact_s3_reference(model_specs)
        )
        dst_artifact_reference = self._dst_s3_filesystem.get_inference_artifact_s3_reference(
            model_specs
        )
        copy_configs.append(
            CopyContentConfig(
                src=src_inference_artifact_location,
                dst=dst_artifact_reference,
                display_name="inference artifact",
            )
        )

        if not model_specs.supports_prepacked_inference():
            # Need to also copy script if prepack not enabled
            src_inference_script_location = (
                self._src_s3_filesystem.get_inference_script_s3_reference(model_specs)
            )
            dst_inference_script_reference = (
                self._dst_s3_filesystem.get_inference_script_s3_reference(model_specs)
            )

            copy_configs.append(
                CopyContentConfig(
                    src=src_inference_script_location,
                    dst=dst_inference_script_reference,
                    display_name="inference script",
                )
            )

        return copy_configs

    def _get_copy_configs_for_training_dependencies(
        self, model_specs: JumpStartModelSpecs
    ) -> List[CopyContentConfig]:
        copy_configs = []

        src_training_artifact_location = self._src_s3_filesystem.get_training_artifact_s3_reference(
            model_specs
        )
        dst_artifact_reference = self._dst_s3_filesystem.get_training_artifact_s3_reference(
            model_specs
        )
        copy_configs.append(
            CopyContentConfig(
                src=src_training_artifact_location,
                dst=dst_artifact_reference,
                display_name="training artifact",
            )
        )

        src_training_script_location = self._src_s3_filesystem.get_training_script_s3_reference(
            model_specs
        )
        dst_training_script_reference = self._dst_s3_filesystem.get_training_script_s3_reference(
            model_specs
        )
        copy_configs.append(
            CopyContentConfig(
                src=src_training_script_location,
                dst=dst_training_script_reference,
                display_name="training script",
            )
        )

        training_dataset_s3_prefix_reference = (
            self._src_s3_filesystem.get_default_training_dataset_s3_reference(model_specs)
        )
        training_dataset_s3_prefix_reference_dst = (
            self._dst_s3_filesystem.get_default_training_dataset_s3_reference(model_specs)
        )
        copy_configs.append(
            CopyContentConfig(
                src=training_dataset_s3_prefix_reference,
                dst=training_dataset_s3_prefix_reference_dst,
                display_name="training dataset",
                is_dir=True,
            )
        )

        return copy_configs

    def _get_copy_configs_for_demo_notebook_dependencies(
        self, model_specs: JumpStartModelSpecs
    ) -> None:
        copy_configs = []

        notebook_s3_reference = self._src_s3_filesystem.get_demo_notebook_s3_reference(model_specs)
        notebook_s3_reference_dst = self._dst_s3_filesystem.get_demo_notebook_s3_reference(
            model_specs
        )
        copy_configs.append(
            CopyContentConfig(
                src=notebook_s3_reference,
                dst=notebook_s3_reference_dst,
                display_name="demo notebook",
            )
        )

        return copy_configs

    def _get_copy_configs_for_markdown_dependencies(self, model_specs: JumpStartModelSpecs) -> None:
        copy_configs = []

        markdown_s3_reference = self._src_s3_filesystem.get_markdown_s3_reference(model_specs)
        markdown_s3_reference_dst = self._dst_s3_filesystem.get_markdown_s3_reference(model_specs)
        copy_configs.append(
            CopyContentConfig(
                src=markdown_s3_reference, dst=markdown_s3_reference_dst, display_name="markdown"
            )
        )

        return copy_configs

    def _parallel_execute_copy_configs(self, copy_configs: List[CopyContentConfig]):
        tasks = []
        with futures.ThreadPoolExecutor(
            max_workers=self._thread_pool_size, thread_name_prefix="import-models-to-curated-hub"
        ) as deploy_executor:
            for copy_config in copy_configs:
                if copy_config.is_dir:
                    tasks.append(
                        deploy_executor.submit(
                            self._copy_s3_dir,
                            copy_config.display_name,
                            copy_config.src,
                            copy_config.dst,
                        )
                    )
                else:
                    tasks.append(
                        deploy_executor.submit(
                            self._copy_s3_reference,
                            copy_config.display_name,
                            copy_config.src,
                            copy_config.dst,
                        )
                    )

        results = futures.wait(tasks)
        failed_copies: List[BaseException] = []
        for result in results.done:
            exception = result.exception()
            if exception:
                failed_copies.append(exception)
            if failed_copies:
                raise RuntimeError(
                    f"Failures when importing models to curated hub in parallel: {failed_copies}"
                )

    def _copy_s3_dir(self, resource_name: str, src: S3ObjectReference, dst: S3ObjectReference):
        keys_in_dir = find_objects_under_prefix(
            bucket=src.bucket,
            prefix=src.key,
            s3_client=self._s3_client,
        )

        for key in keys_in_dir:
            src_reference = create_s3_object_reference_from_bucket_and_key(src.bucket, key)
            dst_reference = create_s3_object_reference_from_bucket_and_key(
                dst.bucket, key.replace(src.key, dst.key, 1)
            )

            self._copy_s3_reference(resource_name, src_reference, dst_reference)

    def _copy_s3_reference(
        self, resource_name: str, src: S3ObjectReference, dst: S3ObjectReference
    ):
        print(f"Copying {resource_name} from {src.bucket}/{src.key} to {dst.bucket}/{dst.key}...")

        self._s3_client.copy(
            src.format_for_s3_copy(),
            dst.bucket,
            dst.key,
            ExtraArgs=EXTRA_S3_COPY_ARGS,
        )

        print(
            f"Copying {resource_name} from {src.bucket}/{src.key} to {dst.bucket}/{dst.key} complete!"
        )
