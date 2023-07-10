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
    tests_models = [
        PublicModelId(
            id="autogluon-classification-ensemble", version="1.1.1"
        ),  # test base functionality (deploy + train)
        PublicModelId(id="huggingface-translation-t5-small", version="1.1.0"),  # no training
        PublicModelId(
            id="huggingface-text2text-flan-t5-base", version="1.2.2"
        ),  # test prepack + train
    ]

    def setUp(self):
        self.test_curated_hub = JumpStartCuratedPublicHub(self.test_s3_prefix)

    """Testing client calls"""

    def test_full_workflow(self):
        self.test_curated_hub.create()
        # self.test_curated_hub.import_models(self.tests_models)

        self.test_curated_hub.delete_models([PublicModelId(id="huggingface-translation-t5-small", version="1.1.0")])
