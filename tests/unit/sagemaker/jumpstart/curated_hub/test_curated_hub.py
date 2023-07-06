from __future__ import absolute_import
import unittest

from mock.mock import patch

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec

from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import JumpStartCuratedPublicHub
from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import PublicModelId


class JumpStartCuratedPublicHubTest(unittest.TestCase):

    test_s3_prefix = "testCuratedHub"
    test_public_js_model = PublicModelId(id="pytorch", version="1.0.0")

    def setUp(self):
        self.test_curated_hub = JumpStartCuratedPublicHub(self.test_s3_prefix)

    # def test_create_curated_hub_valid_s3_prefix_should_succeed(self):
    #    self.test_curated_hub.create()

    # def test_create_curated_hub_none_s3_prefix_should_fail(self):
    #    null_s3_prefix_curated_hub = JumpStartCuratedPublicHub(None)
    #    null_s3_prefix_curated_hub.create()

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_import_model(self, patched_get_model_specs):
        patched_get_model_specs.side_effect = get_spec_from_base_spec

        self.test_curated_hub._import_model(public_js_model=self.test_public_js_model)
        pass
