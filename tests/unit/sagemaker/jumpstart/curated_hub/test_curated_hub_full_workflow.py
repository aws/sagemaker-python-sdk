from __future__ import absolute_import
import unittest

from mock.mock import patch
import uuid

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec
from botocore.client import ClientError


from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import JumpStartCuratedPublicHub
from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import PublicModelId

class JumpStartCuratedPublicHubTest(unittest.TestCase):

    test_s3_prefix = f"test-curated-hub-chrstfu"
    test_public_js_model = PublicModelId(id="autogluon-classification-ensemble", version="1.1.1")
    test_second_public_js_model = PublicModelId(id="catboost-classification-model", version="1.2.7")
    test_nonexistent_public_js_model = PublicModelId(id="fail", version="1.0.0")

    def setUp(self):
        self.test_curated_hub = JumpStartCuratedPublicHub(self.test_s3_prefix)

    """Testing client calls"""
    def test_full_workflow(self):
      self.test_curated_hub.create()
      self.test_curated_hub.import_models([self.test_public_js_model, self.test_second_public_js_model])
