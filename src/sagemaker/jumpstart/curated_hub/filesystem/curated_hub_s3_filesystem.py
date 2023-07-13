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
from sagemaker.jumpstart.curated_hub.filesystem.s3_object_reference import S3ObjectReference, create_s3_object_reference_from_bucket_and_key, create_s3_object_reference_from_uri

class CuratedHubS3Filesystem:
    def __init__(self, region: str, bucket: str):
        self._region = region
        self._bucket = bucket
        self._studio_metadata_map = get_studio_model_metadata_map_from_region(region) # Necessary for SDK - Studio metadata drift
        self._disambiguator = time.time()

    def get_bucket(self) -> str:
        return self._bucket
    
    def get_inference_artifact_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        return create_s3_object_reference_from_bucket_and_key(self.get_bucket(), f"{model_specs.model_id}/{self._disambiguator}/infer.tar.gz")
    
    def get_inference_script_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        return create_s3_object_reference_from_bucket_and_key(self.get_bucket(), f"{model_specs.model_id}/{self._disambiguator}/sourcedir.tar.gz")

    def get_training_artifact_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        return create_s3_object_reference_from_bucket_and_key(self.get_bucket(), f"{model_specs.model_id}/{self._disambiguator}/train.tar.gz")
    
    def get_training_script_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        return create_s3_object_reference_from_bucket_and_key(self.get_bucket(), f"{model_specs.model_id}/{self._disambiguator}/training/sourcedir.tar.gz")
    
    def get_demo_notebook_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        return create_s3_object_reference_from_bucket_and_key(self.get_bucket(), f"{model_specs.model_id}/{self._disambiguator}/demo-notebook.ipynb")
    
    def get_default_training_dataset_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        return create_s3_object_reference_from_bucket_and_key(self.get_bucket(), self._get_training_dataset_prefix(model_specs))
    
    def _get_training_dataset_prefix(self, model_specs: JumpStartModelSpecs) -> str: # Studio expects the same format as public hub bucket
        studio_model_metadata = self._studio_metadata_map[model_specs.model_id] # TODO: verify this
        return studio_model_metadata["defaultDataKey"]
    
    def get_markdown_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference: # Studio expects the same format as public hub bucket
        framework = get_model_framework(model_specs)
        key = f"{framework}-metadata/{model_specs.model_id}.md"
        return create_s3_object_reference_from_bucket_and_key(self.get_bucket(), key)
    