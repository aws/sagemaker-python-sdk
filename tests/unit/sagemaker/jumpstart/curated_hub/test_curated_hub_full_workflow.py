from __future__ import absolute_import
import unittest

from mock.mock import patch

from sagemaker.jumpstart.curated_hub.jumpstart_curated_public_hub import JumpStartCuratedPublicHub
from sagemaker.jumpstart.curated_hub.utils import PublicModelId


class JumpStartCuratedPublicHubTest(unittest.TestCase):

    custom_hub_name = f"test-curated-hub-chrstfu"

    test_models = [
        PublicModelId(id="autogluon-classification-ensemble", version="*"),
        PublicModelId(id="autogluon-regression-ensemble", version="*"),
        PublicModelId(id="catboost-classification-model", version="*"),
        PublicModelId(id="catboost-regression-model", version="*"),
        PublicModelId(id="huggingface-eqa-bert-base-cased", version="*"),
        PublicModelId(id="huggingface-eqa-bert-base-multilingual-cased", version="*"),
        PublicModelId(id="huggingface-eqa-bert-base-multilingual-uncased", version="*"),
        PublicModelId(id="huggingface-eqa-bert-base-uncased", version="*"),
        PublicModelId(id="huggingface-eqa-bert-large-cased", version="*"),
        PublicModelId(id="huggingface-eqa-bert-large-cased-whole-word-masking", version="*"),
        PublicModelId(id="huggingface-llm-falcon-7b-bf16", version="*"),
    ]

    test_delete_models = [PublicModelId(id="autogluon-classification-ensemble", version="*")]

    def setUp(self):
        self.test_curated_hub = JumpStartCuratedPublicHub(self.custom_hub_name, True)

    """Testing client calls"""

    def test_full_workflow(self):
        self.test_curated_hub.get_or_create()
        self.test_curated_hub.sync(self.test_models)
        # self.test_curated_hub.import_models(self.test_models)
        # self.test_curated_hub.delete_models(self.test_models)
