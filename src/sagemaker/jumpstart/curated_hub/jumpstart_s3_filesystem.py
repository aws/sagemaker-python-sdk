import time
from typing import Optional, Dict, Any, List

from botocore.client import BaseClient

import boto3
from sagemaker import model_uris, script_uris
from sagemaker.jumpstart.curated_hub.utils import get_model_framework, find_objects_under_prefix, construct_s3_uri
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.jumpstart.curated_hub.utils import PublicModelId, \
    construct_s3_uri, get_studio_model_metadata_map_from_region
from functools import partial

class JumpStartS3Filesystem:
    @staticmethod
    def get_bucket(region: str) -> str:
        return get_jumpstart_content_bucket(region)
    
    @staticmethod
    def get_inference_artifact_s3_uri(region: str, model_specs: JumpStartModelSpecs) -> str:
        return JumpStartS3Filesystem._jumpstart_artifact_location(region, JumpStartScriptScope.INFERENCE, model_specs)
    
    @staticmethod
    def get_training_artifact_s3_uri(region: str, model_specs: JumpStartModelSpecs) -> str:
        return JumpStartS3Filesystem._jumpstart_artifact_location(region, JumpStartScriptScope.TRAINING, model_specs)
    
    @staticmethod
    def get_inference_script_s3_uri(region: str, model_specs: JumpStartModelSpecs) -> str:
        return JumpStartS3Filesystem._jumpstart_script_location(region, JumpStartScriptScope.INFERENCE, model_specs)
    
    @staticmethod
    def get_training_script_s3_uri(region: str, model_specs: JumpStartModelSpecs) -> str:
        return JumpStartS3Filesystem._jumpstart_script_location(region, JumpStartScriptScope.TRAINING, model_specs)
    
    @staticmethod
    def get_default_training_dataset_s3_uris(region: str, model_specs: JumpStartModelSpecs) -> List[str]:
        studio_metadata_map = get_studio_model_metadata_map_from_region(region)
        studio_model_metadata = studio_metadata_map[model_specs.model_id]
        src_dataset_prefix = studio_model_metadata["defaultDataKey"]

        s3_client = boto3.client("s3", region_name=region)

        training_dataset_keys = find_objects_under_prefix(
            bucket=JumpStartS3Filesystem.get_bucket(region),
            prefix=src_dataset_prefix,
            s3_client=s3_client,
        )

        return list(map(partial(construct_s3_uri, JumpStartS3Filesystem.get_bucket(region)), training_dataset_keys))

    @staticmethod
    def get_bucket_and_key_from_s3_uri(s3_uri: str) -> Dict[str, str]:
      uri_with_s3_prefix_removed = s3_uri.replace("s3://", "", 1)
      uri_split = uri_with_s3_prefix_removed.split("/")

      return {
          "Bucket": uri_split[0],
          "Key": "/".join(uri_split[1:]),
      }        
    
    @staticmethod
    def _jumpstart_script_location(region: str, model_scope: str, model_specs: JumpStartModelSpecs) -> str:
        return script_uris.retrieve(
            region=region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            script_scope=model_scope,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )

    @staticmethod
    def _jumpstart_artifact_location(region: str, model_scope: str, model_specs: JumpStartModelSpecs) -> str:
        return model_uris.retrieve(
            region=region,
            model_id=model_specs.model_id,
            model_version=model_specs.version,
            model_scope=model_scope,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )