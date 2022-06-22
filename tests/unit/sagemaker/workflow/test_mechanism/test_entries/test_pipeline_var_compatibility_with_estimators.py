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
from __future__ import absolute_import

from random import getrandbits

from unittest.mock import patch, MagicMock

from sagemaker.estimator import Estimator
from sagemaker.huggingface import TrainingCompilerConfig as HF_TrainingCompilerConfig
from sagemaker.tensorflow import TensorFlow
from sagemaker.amazon.lda import LDA
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.rl.estimator import RLEstimator
from sagemaker.amazon.pca import PCA
from sagemaker.mxnet.estimator import MXNet
from sagemaker.amazon.randomcutforest import RandomCutForest
from sagemaker.amazon.factorization_machines import FactorizationMachines
from sagemaker.algorithm import AlgorithmEstimator
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.amazon.ipinsights import IPInsights
from sagemaker.huggingface.estimator import HuggingFace
from sagemaker.tensorflow.estimator import TrainingCompilerConfig as TF_TrainingCompilerConfig
from sagemaker.amazon.ntm import NTM
from sagemaker.pytorch import PyTorch
from sagemaker.chainer import Chainer
from sagemaker.amazon.linear_learner import LinearLearner
from sagemaker.amazon.knn import KNN
from tests.unit.sagemaker.workflow.test_mechanism.test_code.test_pipeline_var_compatibility_template import (
    PipelineVarCompatiTestTemplate,
)
from sagemaker.amazon.object2vec import Object2Vec
from sagemaker.amazon.kmeans import KMeans
from tests.unit.sagemaker.workflow.test_mechanism.test_code import IMAGE_URI, MockProperties
from tests.unit.sagemaker.workflow.test_mechanism.test_code.utilities import (
    mock_tar_and_upload_dir,
    mock_image_uris_retrieve,
)

_IS_TRUE = bool(getrandbits(1))


# These tests provide the incomplete default arg dict
# within which some class or target func parameters are missing.
# The test template will fill in those missing args
# Note: the default args should not include PipelineVariable objects
@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=Estimator,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_tensorflow_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(
            compiler_config=TF_TrainingCompilerConfig() if _IS_TRUE else None,
            image_uri=IMAGE_URI if not _IS_TRUE else None,
            instance_type="ml.p3.2xlarge",
            framework_version="2.9",
            py_version="py39",
        ),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=TensorFlow,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_lda_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(
            instance_count=1,
        ),
        func_args=dict(
            mini_batch_size=128,
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=LDA,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_pca_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=PCA,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_mxnet_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(framework_version="1.4.0"),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=MXNet,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_rl_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=RLEstimator,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_ntm_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(clip_gradient=1e-3),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=NTM,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_rcf_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=RandomCutForest,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_xgboost_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(framework_version="1.2-1"),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=XGBoost,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_object2vec_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=Object2Vec,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_fm_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=FactorizationMachines,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_ae_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=AlgorithmEstimator,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_sklearn_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(
            py_version="py3",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            framework_version="0.20.0",
        ),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=SKLearn,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_ipinsights_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=IPInsights,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_huggingface_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(
            instance_type="ml.p3.2xlarge",
            transformers_version="4.11",
            tensorflow_version="2.5" if _IS_TRUE else None,
            pytorch_version="1.9" if not _IS_TRUE else None,
            compiler_config=HF_TrainingCompilerConfig() if _IS_TRUE else None,
            image_uri=IMAGE_URI if not _IS_TRUE else None,
            py_version="py37" if _IS_TRUE else "py38",
        ),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=HuggingFace,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_pytorch_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(framework_version="1.8.0", py_version="py3"),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=PyTorch,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_kmeans_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=KMeans,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_knn_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(
            dimension_reduction_target=6,
            dimension_reduction_type="sign",
        ),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=KNN,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_chainer_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(framework_version="4.0"),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=Chainer,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
def test_ll_estimator_compatibility():
    default_args = dict(
        clazz_args=dict(init_method="normal"),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=LinearLearner,
        default_args=default_args,
    )
    test_template.check_compatibility()
