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

from sagemaker import (
    Model,
    KNNModel,
    KMeansModel,
    PCAModel,
    LDAModel,
    NTMModel,
    Object2VecModel,
    FactorizationMachinesModel,
    IPInsightsModel,
    RandomCutForestModel,
    LinearLearnerModel,
    PipelineModel,
)
from sagemaker.chainer import ChainerModel
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.model import FrameworkModel
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.mxnet import MXNetModel
from sagemaker.pytorch import PyTorchModel
from sagemaker.sklearn import SKLearnModel
from sagemaker.sparkml import SparkMLModel
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.workflow._utils import _RepackModelStep
from sagemaker.xgboost import XGBoostModel
from tests.unit.sagemaker.workflow.test_mechanism.test_code.test_pipeline_var_compatibility_template import (
    PipelineVarCompatiTestTemplate,
)
from tests.unit.sagemaker.workflow.test_mechanism.test_code import BUCKET, MockProperties
from tests.unit.sagemaker.workflow.test_mechanism.test_code.utilities import (
    mock_image_uris_retrieve,
    mock_tar_and_upload_dir,
)

_IS_TRUE = bool(getrandbits(1))
_mock_properties = MockProperties(step_name="MyStep")
_mock_properties.__dict__["ModelArtifacts"] = MockProperties(
    step_name="MyStep", path="ModelArtifacts"
)
_mock_properties.ModelArtifacts.__dict__["S3ModelArtifacts"] = MockProperties(
    step_name="MyStep", path="ModelArtifacts.S3ModelArtifacts"
)


# These tests provide the incomplete default arg dict
# within which some class or target func parameters are missing.
# The test template will fill in those missing args
# Note: the default args should not include PipelineVariable objects
@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(
                inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
                transform_instances=["ml.t2.medium", "ml.m5.xlarge"],
                model_package_group_name="my-model-pkg-group" if not _IS_TRUE else None,
                model_package_name="my-model-pkg" if _IS_TRUE else None,
            ),
            create=dict(
                instance_type="ml.t2.medium",
            ),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=Model,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_framework_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=FrameworkModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
# Skip validate source dir because we skip _inject_repack_script
# Thus, repack script is not in the dir and the validation fails
@patch("sagemaker.estimator.validate_source_dir", MagicMock(return_value=True))
def test_tensorflow_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=TensorFlowModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_knn_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=KNNModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_sparkml_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=SparkMLModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_kmeans_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=KMeansModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_pca_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=PCAModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_lda_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=LDAModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_ntm_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=NTMModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_object2vec_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=Object2VecModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_factorizationmachines_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=FactorizationMachinesModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_ipinsights_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=IPInsightsModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_randomcutforest_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=RandomCutForestModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_linearlearner_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=LinearLearnerModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
@patch("sagemaker.estimator.validate_source_dir", MagicMock(return_value=True))
def test_sklearn_model_compatibility():
    default_args = dict(
        clazz_args=dict(framework_version="0.20.0"),
        func_args=dict(
            register=dict(),
            create=dict(accelerator_type=None),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=SKLearnModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
@patch("sagemaker.estimator.validate_source_dir", MagicMock(return_value=True))
def test_pytorch_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=PyTorchModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
@patch("sagemaker.estimator.validate_source_dir", MagicMock(return_value=True))
def test_xgboost_model_compatibility():
    default_args = dict(
        clazz_args=dict(
            framework_version="1",
        ),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=XGBoostModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_chainer_model_compatibility():
    default_args = dict(
        clazz_args=dict(framework_version="4.0.0"),
        func_args=dict(
            register=dict(),
            create=dict(accelerator_type=None),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=ChainerModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
@patch("sagemaker.estimator.validate_source_dir", MagicMock(return_value=True))
def test_huggingface_model_compatibility():
    default_args = dict(
        clazz_args=dict(
            tensorflow_version="2.4.1" if _IS_TRUE else None,
            pytorch_version="1.7.1" if not _IS_TRUE else None,
            transformers_version="4.6.1",
            py_version="py37" if _IS_TRUE else "py36",
        ),
        func_args=dict(
            register=dict(),
            create=dict(accelerator_type=None),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=HuggingFaceModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
@patch("sagemaker.estimator.validate_source_dir", MagicMock(return_value=True))
def test_mxnet_model_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=MXNetModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_multidata_model_compatibility():
    default_args = dict(
        clazz_args=dict(
            model_data_prefix=f"s3://{BUCKET}",
            model=None,
        ),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=MultiDataModel,
        default_args=default_args,
    )
    test_template.check_compatibility()


@patch("sagemaker.workflow.steps.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.workflow._utils.Properties", MagicMock(return_value=_mock_properties))
@patch("sagemaker.image_uris.retrieve", MagicMock(side_effect=mock_image_uris_retrieve))
@patch("sagemaker.estimator.tar_and_upload_dir", MagicMock(side_effect=mock_tar_and_upload_dir))
@patch("sagemaker.utils.repack_model", MagicMock())
@patch.object(_RepackModelStep, "_inject_repack_script", MagicMock())
def test_pipelinemodel_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(
            register=dict(),
            create=dict(),
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=PipelineModel,
        default_args=default_args,
    )
    test_template.check_compatibility()
