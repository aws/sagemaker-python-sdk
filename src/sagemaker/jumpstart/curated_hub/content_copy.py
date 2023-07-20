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
"""This module contains utilities related to SageMaker JumpStart."""
from __future__ import absolute_import
import time
from typing import List, Set

from botocore.client import BaseClient
from dataclasses import dataclass
from concurrent import futures

from sagemaker.jumpstart.curated_hub.utils import (
    find_objects_under_prefix,
)
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.accessors.jumpstart_s3_accessor import JumpstartS3Accessor
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectReference,
    create_s3_object_reference_from_bucket_and_key,
)


EXTRA_S3_COPY_ARGS = {"ACL": "bucket-owner-full-control", "Tagging": "SageMaker=true"}


@dataclass
class CopyContentConfig:
    """Property class to assist S3 copy"""

    src: S3ObjectReference
    dst: S3ObjectReference
    display_name: str


class ContentCopier:
    """Copies content from JS source bucket to hub bucket."""

    def __init__(
        self,
        region: str,
        s3_client: BaseClient,
        src_s3_filesystem: JumpstartS3Accessor,
        dst_s3_filesystem: JumpstartS3Accessor,  # TODO: abstract this
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

        Unfortunately, this logic is duplicated/inconsistent with what is in Studio.
        """
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

        copy_configs.extend(self._get_copy_configs_for_training_dataset(model_specs))

        return copy_configs

    def _get_copy_configs_for_training_dataset(self, model_specs: JumpStartModelSpecs) -> None:
        src_prefix = self._src_s3_filesystem.get_default_training_dataset_s3_reference(model_specs)
        dst_prefix = self._dst_s3_filesystem.get_default_training_dataset_s3_reference(model_specs)

        keys_in_src_dir = self._get_s3_object_keys_under_prefix(src_prefix)

        copy_configs = []
        for full_key in keys_in_src_dir:
            src_reference = create_s3_object_reference_from_bucket_and_key(
                src_prefix.bucket, full_key
            )
            dst_reference = create_s3_object_reference_from_bucket_and_key(
                dst_prefix.bucket, full_key.replace(src_prefix.key, dst_prefix.key, 1)
            )  # Replacing s3 key prefix with expected destination prefix

            copy_configs.append(
                CopyContentConfig(
                    src=src_reference, dst=dst_reference, display_name="training dataset"
                )
            )

        return copy_configs

    def _get_s3_object_keys_under_prefix(self, prefix_reference: S3ObjectReference) -> Set[str]:
        try:
            return find_objects_under_prefix(
                bucket=prefix_reference.bucket,
                prefix=prefix_reference.key,
                s3_client=self._s3_client,
            )
        except Exception as ex:
            print(
                "ERROR: encountered an exception when finding objects"
                + f" under prefix {prefix_reference.bucket}/{prefix_reference.key}: {str(ex)}"
            )
            raise ex

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

    def _copy_s3_reference(
        self, resource_name: str, src: S3ObjectReference, dst: S3ObjectReference
    ):
        if not self._is_s3_object_different(src, dst):
            print(
                f"Detected that {resource_name} is the same in destination bucket. Skipping copy."
            )
            return

        print(f"Copying {resource_name} from {src.bucket}/{src.key} to {dst.bucket}/{dst.key}...")
        try:
            self._s3_client.copy(
                src.format_for_s3_copy(),
                dst.bucket,
                dst.key,
                ExtraArgs=EXTRA_S3_COPY_ARGS,
            )
        except Exception as ex:
            print(
                "ERROR: encountered an exception when calling copy from"
                + f" {src.bucket}/{src.key} to {dst.bucket}/{dst.key}: {str(ex)}"
            )
            raise ex

        print(
            f"Copying {resource_name} from {src.bucket}/{src.key} to {dst.bucket}/{dst.key} complete!"
        )

    def _is_s3_object_different(self, src: S3ObjectReference, dst: S3ObjectReference) -> bool:
        src_etag = self._get_s3_object_etag(src)
        dst_etag = self._get_s3_object_etag(dst)

        return src_etag != dst_etag

    def _get_s3_object_etag(self, object: S3ObjectReference) -> str:
        try:
            response = self._s3_client.head_object(Bucket=object.bucket, Key=object.key)
            return response.pop("ETag")
        except Exception:
            return ""
