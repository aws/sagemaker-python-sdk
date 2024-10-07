from __future__ import absolute_import
import unittest

from sagemaker.modules.image_spec import ImageSpec, Framework, Processor


class TestImageSpec(unittest.TestCase):

    def test_image_spec_update(self):
        image_spec = ImageSpec(framework=Framework.HUGGING_FACE)
        assert image_spec.version is None
        image_spec.update_image_spec(version="v3")
        assert image_spec.version == "v3"

    def test_image_spec_retrive(self):
        # Asserting substrings because full string uri can change with newer versions
        image_spec = ImageSpec(framework=Framework.XGBOOST)
        xgboost_uri = image_spec.retrieve()
        assert "dkr.ecr.us-west-2.amazonaws.com" in xgboost_uri
        assert "sagemaker-xgboost" in xgboost_uri

        image_spec = ImageSpec(
            framework=Framework.HUGGING_FACE,
            processor=Processor.GPU,
            base_framework_version="pytorch2.1.0",
        )
        hugging_face_uri = image_spec.retrieve()
        assert "dkr.ecr.us-west-2.amazonaws.com" in hugging_face_uri
        assert "huggingface-pytorch-training" in hugging_face_uri

        image_spec = ImageSpec(framework=Framework.PYTORCH)
        pytorch_uri = image_spec.retrieve()
        assert "dkr.ecr.us-west-2.amazonaws.com" in pytorch_uri
        assert "pytorch-training" in pytorch_uri

        image_spec = ImageSpec(framework=Framework.SKLEARN)
        sklearn_uri = image_spec.retrieve()
        assert "dkr.ecr.us-west-2.amazonaws.com" in sklearn_uri
        assert "sagemaker-scikit-learn" in sklearn_uri

    def test_image_spec_retrive_with_version(self):
        image_spec = ImageSpec(framework=Framework.XGBOOST, version="0.90-1")
        xgboost_uri = image_spec.retrieve()
        assert (
            xgboost_uri
            == "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:0.90-1-cpu-py3"
        )
