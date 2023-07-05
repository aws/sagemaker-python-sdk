
from typing import Any, Dict, Union, Optional, List
from sagemaker.session import Session
import boto3
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.types import (
    JumpStartModelHeader,
    JumpStartModelSpecs,
    JumpStartVersionedModelId,
)
from dataclasses import dataclass

@dataclass
class JumpStartModel:
    id: str
    version: str

class JumpStartCuratedPublicHub:
    """JumpStartCuratedPublicHub class.

    This class helps users create a new curated hub.
    """

    def __init__(self, curated_hub_s3_prefix: str):
      self.curated_hub_s3_prefix = curated_hub_s3_prefix

      self._region = "us-west-2"
      self._s3_client = boto3.client("s3", region_name=self._region)
      self._sagemaker_session = Session()

    def create(self):
       """
       Workflow:
       1. Verifications
        a) S3 location exists
        b) if not creates one
       2. CreateHub
       """
       


    def import_models(self, model_ids: List[JumpStartModel]):
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
      
       

    def import_model(self, jumpstart_model: JumpStartModel) -> JumpStartModelSpecs:
       model_spec = verify_model_region_and_return_specs(
          model_id=jumpstart_model.id,
          version=jumpstart_model.version,
          scope=JumpStartScriptScope.INFERENCE,
          region=self._region
      )
       