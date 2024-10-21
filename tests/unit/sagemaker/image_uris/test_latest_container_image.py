import json
import unittest
from unittest.mock import patch

import pytest

from src.sagemaker.image_uris import get_latest_container_image


class TestImageUtils(unittest.TestCase):

    @patch('src.sagemaker.image_uris.config_for_framework')
    def test_get_latest_container_image_invalid_framework(self,
                                                          mock_config_for_framework):
        mock_config_for_framework.side_effect = FileNotFoundError

        with self.assertRaises(ValueError) as e:
            get_latest_container_image("xgboost", "inference")
            assert "No framework config for framework" in str(e.exception)

    @patch('src.sagemaker.image_uris.config_for_framework')
    def test_get_latest_container_image_no_framework(self,
                                                     mock_config_for_framework):
        mock_config_for_framework.return_value = {}

        with self.assertRaises(ValueError) as e:
            get_latest_container_image("xgboost", "inference")
            assert "No framework config for framework" in str(e.exception)


@pytest.mark.parametrize("framework", ['object-detection',
                                       'instance_gpu_info',
                                       'object2vec',
                                       'pytorch',
                                       'djl-lmi',
                                       'mxnet',
                                       'debugger',
                                       'data-wrangler',
                                       'spark',
                                       'blazingtext',
                                       'pytorch-neuron',
                                       'forecasting-deepar',
                                       'huggingface-neuron',
                                       'ntm',
                                       'neo-mxnet',
                                       'image-classification',
                                       'xgboost',
                                       'autogluon',
                                       'sparkml-serving',
                                       'clarify',
                                       'inferentia-pytorch',
                                       'neo-tensorflow',
                                       'huggingface-tei-cpu',
                                       'huggingface',
                                       'sagemaker-tritonserver',
                                       'pytorch-smp',
                                       'knn',
                                       'linear-learner',
                                       'model-monitor',
                                       'ray-tensorflow',
                                       'djl-neuronx',
                                       'huggingface-llm-neuronx',
                                       'image-classification-neo',
                                       'lda',
                                       'stabilityai',
                                       'ray-pytorch',
                                       'chainer',
                                       'coach-mxnet',
                                       'pca',
                                       'sagemaker-geospatial',
                                       'djl-tensorrtllm',
                                       'huggingface-training-compiler',
                                       'pytorch-training-compiler',
                                       'vw',
                                       'huggingface-neuronx',
                                       'ipinsights',
                                       'detailed-profiler',
                                       'inferentia-tensorflow',
                                       'semantic-segmentation',
                                       'inferentia-mxnet',
                                       'xgboost-neo',
                                       'neo-pytorch',
                                       'djl-deepspeed',
                                       'djl-fastertransformer',
                                       'sklearn',
                                       'tensorflow',
                                       'randomcutforest',
                                       'huggingface-llm',
                                       'factorization-machines',
                                       'huggingface-tei',
                                       'coach-tensorflow',
                                       'seq2seq',
                                       'kmeans',
                                       'sagemaker-base-python'])
@patch('src.sagemaker.image_uris.config_for_framework')
@patch('src.sagemaker.image_uris.retrieve')
def test_get_latest_container_image_parameterized(mock_image_retrieve,
                                                  mock_config_for_framework,
                                                  framework):
    file_path = f"src/sagemaker/image_uri_config/{framework}.json"
    with open(file_path, 'r') as json_file:
        config_for_framework = json.load(json_file)

    mock_config_for_framework.return_value = config_for_framework
    mock_image_retrieve.return_value = "latest-image"
    image, version = get_latest_container_image(framework=framework,
                                                image_scope="inference",
                                                instance_type="ml.c4.xlarge",
                                                region="us-east-1")
    assert image == "latest-image"
