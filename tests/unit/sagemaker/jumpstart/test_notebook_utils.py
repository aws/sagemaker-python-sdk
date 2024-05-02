from __future__ import absolute_import
import json

from unittest import TestCase
from unittest.mock import Mock, patch

import pytest
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.filters import And, Identity, Not, Or
from tests.unit.sagemaker.jumpstart.constants import PROTOTYPICAL_MODEL_SPECS_DICT
from tests.unit.sagemaker.jumpstart.utils import (
    get_header_from_base_header,
    get_prototype_manifest,
    get_prototype_model_spec,
)
from sagemaker.jumpstart.enums import JumpStartModelType
from sagemaker.jumpstart.notebook_utils import (
    _generate_jumpstart_model_versions,
    get_model_url,
    list_jumpstart_frameworks,
    list_jumpstart_models,
    list_jumpstart_scripts,
    list_jumpstart_tasks,
    _is_valid_version,
)


@patch("sagemaker.jumpstart.notebook_utils.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.read_s3_file")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
@patch("sagemaker.jumpstart.notebook_utils.accessors.JumpStartModelsAccessor.get_model_specs")
@patch("sagemaker.jumpstart.notebook_utils._generate_jumpstart_model_versions")
def test_list_jumpstart_scripts(
    patched_generate_jumpstart_models: Mock,
    patched_get_model_specs: Mock,
    patched_get_manifest: Mock,
    patched_read_s3_file: Mock,
):
    patched_get_model_specs.side_effect = get_prototype_model_spec
    patched_get_manifest.side_effect = (
        lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
    )
    patched_generate_jumpstart_models.side_effect = _generate_jumpstart_model_versions
    patched_read_s3_file.side_effect = lambda *args, **kwargs: json.dumps(
        get_prototype_model_spec(None, "pytorch-eqa-bert-base-cased").to_json()
    )

    assert list_jumpstart_scripts() == sorted(["inference", "training"])
    patched_get_model_specs.assert_not_called()
    patched_get_manifest.assert_not_called()
    patched_generate_jumpstart_models.assert_not_called()

    patched_get_model_specs.reset_mock()
    patched_get_manifest.reset_mock()
    patched_generate_jumpstart_models.reset_mock()

    kwargs = {
        "filter": "framework==tensorflow",
        "region": "sa-east-1",
    }
    assert list_jumpstart_scripts(**kwargs) == sorted(["inference", "training"])
    patched_generate_jumpstart_models.assert_called_once_with(
        **kwargs,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    )
    assert patched_get_manifest.call_count == 2
    assert patched_get_model_specs.call_count == 1

    patched_get_model_specs.reset_mock()
    patched_get_manifest.reset_mock()
    patched_generate_jumpstart_models.reset_mock()

    kwargs = {
        "filter": "training_supported is False",
        "region": "sa-east-1",
    }
    num_specs = len(PROTOTYPICAL_MODEL_SPECS_DICT)
    assert list_jumpstart_scripts(**kwargs) == []
    patched_generate_jumpstart_models.assert_called_once_with(
        **kwargs,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    )
    assert patched_get_manifest.call_count == 2
    assert patched_read_s3_file.call_count == num_specs


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
@patch("sagemaker.jumpstart.notebook_utils._generate_jumpstart_model_versions")
def test_list_jumpstart_tasks(
    patched_generate_jumpstart_models: Mock,
    patched_get_model_specs: Mock,
    patched_get_manifest: Mock,
):
    patched_get_model_specs.side_effect = get_prototype_model_spec
    patched_get_manifest.side_effect = (
        lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
    )
    patched_generate_jumpstart_models.side_effect = _generate_jumpstart_model_versions

    assert list_jumpstart_tasks() == sorted(
        [
            "classification",
            "eqa",
            "ic",
            "semseg",
            "spc",
        ]
    )  # incomplete list, based on mocked metadata

    patched_generate_jumpstart_models.assert_called_once()
    assert patched_get_manifest.call_count == 2
    patched_get_model_specs.assert_not_called()

    patched_get_model_specs.reset_mock()
    patched_get_manifest.reset_mock()
    patched_generate_jumpstart_models.reset_mock()

    kwargs = {
        "filter": "framework==tensorflow",
        "region": "sa-east-1",
    }
    assert list_jumpstart_tasks(**kwargs) == ["ic"]
    patched_generate_jumpstart_models.assert_called_once_with(
        **kwargs,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    )
    assert patched_get_manifest.call_count == 2
    patched_get_model_specs.assert_not_called()


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
@patch("sagemaker.jumpstart.notebook_utils._generate_jumpstart_model_versions")
def test_list_jumpstart_frameworks(
    patched_generate_jumpstart_models: Mock,
    patched_get_model_specs: Mock,
    patched_get_manifest: Mock,
):
    patched_get_model_specs.side_effect = get_prototype_model_spec
    patched_get_manifest.side_effect = (
        lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
    )
    patched_generate_jumpstart_models.side_effect = _generate_jumpstart_model_versions

    assert list_jumpstart_frameworks() == sorted(
        [
            "catboost",
            "huggingface",
            "lightgbm",
            "mxnet",
            "pytorch",
            "sklearn",
            "tensorflow",
            "xgboost",
        ]
    )

    patched_generate_jumpstart_models.assert_called_once()
    assert patched_get_manifest.call_count == 2
    patched_get_model_specs.assert_not_called()

    patched_get_model_specs.reset_mock()
    assert patched_get_manifest.call_count == 2
    patched_generate_jumpstart_models.reset_mock()

    kwargs = {
        "filter": "task is not ic",
        "region": "sa-east-1",
    }
    assert list_jumpstart_frameworks(**kwargs) == sorted(
        [
            "catboost",
            "huggingface",
            "lightgbm",
            "mxnet",
            "pytorch",
            "sklearn",
            "xgboost",
        ]
    )

    patched_generate_jumpstart_models.assert_called_once_with(
        **kwargs,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session=DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    )
    assert patched_get_manifest.call_count == 4
    patched_get_model_specs.assert_not_called()


def test_is_valid_version():
    valid_version_strs = ["1.0", "1.0.0", "2012.4", "1!1.0", "1.dev0", "1.2.3+abc.dev1"]
    invalid_version_strs = ["1.1.053_m", "invalid version", "v1-1.0-v2", "@"]
    assert all(_is_valid_version(v) for v in valid_version_strs)
    assert not any(_is_valid_version(v) for v in invalid_version_strs)


class ListJumpStartModels(TestCase):
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_list_jumpstart_models_simple_case(
        self, patched_get_model_specs: Mock, patched_get_manifest: Mock
    ):
        patched_get_model_specs.side_effect = get_prototype_model_spec
        patched_get_manifest.side_effect = lambda region, *args, **kwargs: get_prototype_manifest(
            region
        )
        assert list_jumpstart_models(list_versions=True) == [
            ("catboost-classification-model", "1.0.0"),
            ("huggingface-spc-bert-base-cased", "1.0.0"),
            ("lightgbm-classification-model", "1.0.0"),
            ("mxnet-semseg-fcn-resnet50-ade", "1.0.0"),
            ("pytorch-eqa-bert-base-cased", "1.0.0"),
            ("sklearn-classification-linear", "1.0.0"),
            ("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0"),
            ("xgboost-classification-model", "1.0.0"),
        ]

        patched_get_manifest.assert_called()
        patched_get_model_specs.assert_not_called()

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.notebook_utils.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.read_s3_file")
    def test_list_jumpstart_models_script_filter(
        self, patched_read_s3_file: Mock, patched_get_manifest: Mock
    ):
        patched_read_s3_file.side_effect = lambda *args, **kwargs: json.dumps(
            get_prototype_model_spec(None, "pytorch-eqa-bert-base-cased").to_json()
        )
        patched_get_manifest.side_effect = (
            lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
        )

        manifest_length = len(get_prototype_manifest())
        vals = [True, False]
        for val in vals:
            kwargs = {"filter": And(f"training_supported == {val}", "model_type is open_weights")}
            list_jumpstart_models(**kwargs)
            assert patched_read_s3_file.call_count == manifest_length
            assert patched_get_manifest.call_count == 2

            patched_get_manifest.reset_mock()
            patched_read_s3_file.reset_mock()

            kwargs = {"filter": And(f"training_supported != {val}", "model_type is open_weights")}
            list_jumpstart_models(**kwargs)
            assert patched_read_s3_file.call_count == manifest_length
            assert patched_get_manifest.call_count == 2

            patched_get_manifest.reset_mock()
            patched_read_s3_file.reset_mock()
        kwargs = {
            "filter": And(f"training_supported != {val}", "model_type is open_weights"),
            "list_versions": True,
        }
        assert list_jumpstart_models(**kwargs) == [
            ("catboost-classification-model", "1.0.0"),
            ("huggingface-spc-bert-base-cased", "1.0.0"),
            ("lightgbm-classification-model", "1.0.0"),
            ("mxnet-semseg-fcn-resnet50-ade", "1.0.0"),
            ("pytorch-eqa-bert-base-cased", "1.0.0"),
            ("sklearn-classification-linear", "1.0.0"),
            ("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0"),
            ("xgboost-classification-model", "1.0.0"),
        ]
        assert patched_read_s3_file.call_count == manifest_length
        assert patched_get_manifest.call_count == 2

        patched_get_manifest.reset_mock()
        patched_read_s3_file.reset_mock()

        kwargs = {"filter": And(f"training_supported not in {vals}", "model_type is open_weights")}
        models = list_jumpstart_models(**kwargs)
        assert [] == models
        assert patched_read_s3_file.call_count == manifest_length
        assert patched_get_manifest.call_count == 2

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_list_jumpstart_models_task_filter(
        self, patched_get_model_specs: Mock, patched_get_manifest: Mock
    ):
        patched_get_model_specs.side_effect = get_prototype_model_spec
        patched_get_manifest.side_effect = lambda region, *args, **kwargs: get_prototype_manifest(
            region
        )

        vals = [
            "classification",
            "eqa",
            "ic",
            "semseg",
            "spc",
        ]
        for val in vals:
            kwargs = {"filter": f"task == {val}"}
            list_jumpstart_models(**kwargs)
            patched_get_model_specs.assert_not_called()
            assert patched_get_manifest.call_count == 2

            patched_get_manifest.reset_mock()
            patched_get_model_specs.reset_mock()

            kwargs = {"filter": f"task != {val}"}
            list_jumpstart_models(**kwargs)
            patched_get_model_specs.assert_not_called()
            assert patched_get_manifest.call_count == 2

            patched_get_manifest.reset_mock()
            patched_get_model_specs.reset_mock()

        kwargs = {"filter": f"task in {vals}", "list_versions": True}
        assert list_jumpstart_models(**kwargs) == [
            ("catboost-classification-model", "1.0.0"),
            ("huggingface-spc-bert-base-cased", "1.0.0"),
            ("lightgbm-classification-model", "1.0.0"),
            ("mxnet-semseg-fcn-resnet50-ade", "1.0.0"),
            ("pytorch-eqa-bert-base-cased", "1.0.0"),
            ("sklearn-classification-linear", "1.0.0"),
            ("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0"),
            ("xgboost-classification-model", "1.0.0"),
        ]
        patched_get_model_specs.assert_not_called()
        assert patched_get_manifest.call_count == 2

        patched_get_manifest.reset_mock()
        patched_get_model_specs.reset_mock()

        kwargs = {"filter": f"task not in {vals}"}
        models = list_jumpstart_models(**kwargs)
        assert [] == models
        patched_get_model_specs.assert_not_called()
        assert patched_get_manifest.call_count == 2

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.notebook_utils.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.read_s3_file")
    def test_list_jumpstart_models_framework_filter(
        self, patched_read_s3_file: Mock, patched_get_manifest: Mock
    ):
        patched_read_s3_file.side_effect = lambda *args, **kwargs: json.dumps(
            get_prototype_model_spec(None, "pytorch-eqa-bert-base-cased").to_json()
        )
        patched_get_manifest.side_effect = lambda region, *args, **kwargs: get_prototype_manifest(
            region
        )

        vals = [
            "catboost",
            "huggingface",
            "lightgbm",
            "mxnet",
            "pytorch",
            "sklearn",
            "xgboost",
        ]
        for val in vals:
            kwargs = {"filter": f"framework == {val}"}
            list_jumpstart_models(**kwargs)
            patched_read_s3_file.assert_not_called()
            assert patched_get_manifest.call_count == 2

            patched_get_manifest.reset_mock()
            patched_read_s3_file.reset_mock()

            kwargs = {"filter": f"framework != {val}"}
            list_jumpstart_models(**kwargs)
            patched_read_s3_file.assert_not_called()
            assert patched_get_manifest.call_count == 2

            patched_get_manifest.reset_mock()
            patched_read_s3_file.reset_mock()

        kwargs = {"filter": f"framework in {vals}", "list_versions": True}
        assert list_jumpstart_models(**kwargs) == [
            ("catboost-classification-model", "1.0.0"),
            ("huggingface-spc-bert-base-cased", "1.0.0"),
            ("lightgbm-classification-model", "1.0.0"),
            ("mxnet-semseg-fcn-resnet50-ade", "1.0.0"),
            ("pytorch-eqa-bert-base-cased", "1.0.0"),
            ("sklearn-classification-linear", "1.0.0"),
            ("xgboost-classification-model", "1.0.0"),
        ]
        patched_read_s3_file.assert_not_called()
        assert patched_get_manifest.call_count == 2

        patched_get_manifest.reset_mock()
        patched_read_s3_file.reset_mock()

        kwargs = {"filter": f"framework not in {vals}", "list_versions": True}
        models = list_jumpstart_models(**kwargs)
        assert [("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0")] == models

        patched_get_manifest.reset_mock()
        patched_read_s3_file.reset_mock()

        kwargs = {
            "filter": And(f"framework not in {vals}", "training_supported is True"),
            "list_versions": True,
        }
        models = list_jumpstart_models(**kwargs)
        assert [("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0")] == models
        assert patched_read_s3_file.call_count == 2
        assert patched_get_manifest.call_count == 2

        patched_get_manifest.reset_mock()
        patched_read_s3_file.reset_mock()

        kwargs = {
            "filter": And(
                f"framework not in {vals + ['tensorflow']}", "training_supported is True"
            ),
            "list_versions": True,
        }
        models = list_jumpstart_models(**kwargs)
        assert [] == models
        patched_read_s3_file.assert_not_called()
        assert patched_get_manifest.call_count == 2

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_list_jumpstart_models_region(
        self, patched_get_model_specs: Mock, patched_get_manifest: Mock
    ):
        patched_get_model_specs.side_effect = get_prototype_model_spec

        patched_get_manifest.side_effect = lambda region, *args, **kwargs: get_prototype_manifest(
            region="us-west-2"
        )

        list_jumpstart_models(region="some-region")

        patched_get_manifest.assert_called_with(
            region="some-region",
            s3_client=DEFAULT_JUMPSTART_SAGEMAKER_SESSION.s3_client,
            model_type=JumpStartModelType.OPEN_WEIGHTS,
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_list_jumpstart_models_old_models(
        self,
        patched_get_model_specs: Mock,
        patched_get_manifest: Mock,
    ):
        def get_manifest_more_versions(region: str = JUMPSTART_DEFAULT_REGION_NAME):
            return [
                get_header_from_base_header(region=region, model_id=model_id, version=version)
                for model_id in PROTOTYPICAL_MODEL_SPECS_DICT.keys()
                for version in ["2.400.0", "1.4.0", "2.5.1", "1.300.0"]
            ]

        patched_get_manifest.side_effect = (
            lambda region, *args, **kwargs: get_manifest_more_versions(region)
        )

        assert [
            ("catboost-classification-model", "2.400.0"),
            ("catboost-classification-model", "2.5.1"),
            ("catboost-classification-model", "1.300.0"),
            ("catboost-classification-model", "1.4.0"),
            ("huggingface-spc-bert-base-cased", "2.400.0"),
            ("huggingface-spc-bert-base-cased", "2.5.1"),
            ("huggingface-spc-bert-base-cased", "1.300.0"),
            ("huggingface-spc-bert-base-cased", "1.4.0"),
            ("lightgbm-classification-model", "2.400.0"),
            ("lightgbm-classification-model", "2.5.1"),
            ("lightgbm-classification-model", "1.300.0"),
            ("lightgbm-classification-model", "1.4.0"),
            ("mxnet-semseg-fcn-resnet50-ade", "2.400.0"),
            ("mxnet-semseg-fcn-resnet50-ade", "2.5.1"),
            ("mxnet-semseg-fcn-resnet50-ade", "1.300.0"),
            ("mxnet-semseg-fcn-resnet50-ade", "1.4.0"),
            ("pytorch-eqa-bert-base-cased", "2.400.0"),
            ("pytorch-eqa-bert-base-cased", "2.5.1"),
            ("pytorch-eqa-bert-base-cased", "1.300.0"),
            ("pytorch-eqa-bert-base-cased", "1.4.0"),
            ("sklearn-classification-linear", "2.400.0"),
            ("sklearn-classification-linear", "2.5.1"),
            ("sklearn-classification-linear", "1.300.0"),
            ("sklearn-classification-linear", "1.4.0"),
            ("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "2.400.0"),
            ("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "2.5.1"),
            ("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.300.0"),
            ("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.4.0"),
            ("xgboost-classification-model", "2.400.0"),
            ("xgboost-classification-model", "2.5.1"),
            ("xgboost-classification-model", "1.300.0"),
            ("xgboost-classification-model", "1.4.0"),
        ] == list_jumpstart_models(list_old_models=True, list_versions=True)

        patched_get_model_specs.assert_not_called()
        assert patched_get_manifest.call_count == 2

        patched_get_manifest.reset_mock()
        patched_get_model_specs.reset_mock()

        assert [
            ("catboost-classification-model", "2.400.0"),
            ("huggingface-spc-bert-base-cased", "2.400.0"),
            ("lightgbm-classification-model", "2.400.0"),
            ("mxnet-semseg-fcn-resnet50-ade", "2.400.0"),
            ("pytorch-eqa-bert-base-cased", "2.400.0"),
            ("sklearn-classification-linear", "2.400.0"),
            ("tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "2.400.0"),
            ("xgboost-classification-model", "2.400.0"),
        ] == list_jumpstart_models(list_old_models=False, list_versions=True)
        assert list_jumpstart_models(
            list_old_models=False, list_versions=True
        ) == list_jumpstart_models(list_versions=True)

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.notebook_utils.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.read_s3_file")
    def test_list_jumpstart_models_vulnerable_models(
        self,
        patched_read_s3_file: Mock,
        patched_get_manifest: Mock,
    ):

        patched_get_manifest.side_effect = (
            lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
        )

        def vulnerable_inference_model_spec(bucket, key, *args, **kwargs) -> str:
            spec = get_prototype_model_spec(None, "pytorch-eqa-bert-base-cased")
            spec.inference_vulnerable = True
            return json.dumps(spec.to_json())

        def vulnerable_training_model_spec(bucket, key, *args, **kwargs):
            spec = get_prototype_model_spec(None, "pytorch-eqa-bert-base-cased")
            spec.training_vulnerable = True
            return json.dumps(spec.to_json())

        patched_read_s3_file.side_effect = vulnerable_inference_model_spec

        num_specs = len(PROTOTYPICAL_MODEL_SPECS_DICT)
        assert [] == list_jumpstart_models(
            And(
                "inference_vulnerable is false",
                "training_vulnerable is false",
                "model_type is open_weights",
            )
        )

        assert patched_read_s3_file.call_count == num_specs
        assert patched_get_manifest.call_count == 2

        patched_get_manifest.reset_mock()
        patched_read_s3_file.reset_mock()

        patched_read_s3_file.side_effect = vulnerable_training_model_spec

        assert [] == list_jumpstart_models(
            And(
                "inference_vulnerable is false",
                "training_vulnerable is false",
                "model_type is open_weights",
            )
        )

        assert patched_read_s3_file.call_count == num_specs
        assert patched_get_manifest.call_count == 2

        patched_get_manifest.reset_mock()
        patched_read_s3_file.reset_mock()

        assert [] != list_jumpstart_models()

        assert patched_read_s3_file.call_count == 0

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.notebook_utils.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.read_s3_file")
    def test_list_jumpstart_models_deprecated_models(
        self,
        patched_read_s3_file: Mock,
        patched_get_manifest: Mock,
    ):

        patched_get_manifest.side_effect = (
            lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
        )

        def deprecated_model_spec(bucket, key, *args, **kwargs) -> str:
            spec = get_prototype_model_spec(None, "pytorch-eqa-bert-base-cased")
            spec.deprecated = True
            return json.dumps(spec.to_json())

        patched_read_s3_file.side_effect = deprecated_model_spec

        num_specs = len(PROTOTYPICAL_MODEL_SPECS_DICT)
        assert [] == list_jumpstart_models(
            And("deprecated equals false", "model_type is open_weights")
        )

        assert patched_read_s3_file.call_count == num_specs
        assert patched_get_manifest.call_count == 2

        patched_get_manifest.reset_mock()
        patched_read_s3_file.reset_mock()

        assert [] != list_jumpstart_models()

        assert patched_read_s3_file.call_count == 0

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_list_jumpstart_models_no_versions(
        self,
        patched_get_model_specs: Mock,
        patched_get_manifest: Mock,
    ):
        patched_get_model_specs.side_effect = get_prototype_model_spec
        patched_get_manifest.side_effect = lambda region, *args, **kwargs: get_prototype_manifest(
            region
        )

        all_model_ids = [
            "catboost-classification-model",
            "huggingface-spc-bert-base-cased",
            "lightgbm-classification-model",
            "mxnet-semseg-fcn-resnet50-ade",
            "pytorch-eqa-bert-base-cased",
            "sklearn-classification-linear",
            "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1",
            "xgboost-classification-model",
        ]

        assert list_jumpstart_models() == all_model_ids

        assert list_jumpstart_models(list_versions=False) == all_model_ids

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_list_jumpstart_proprietary_models(
        self,
        patched_get_model_specs: Mock,
        patched_get_manifest: Mock,
    ):
        patched_get_model_specs.side_effect = get_prototype_model_spec
        patched_get_manifest.side_effect = (
            lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
        )

        all_prop_model_ids = [
            "ai21-paraphrase",
            "ai21-summarization",
            "lighton-mini-instruct40b",
            "nc-soft-model-1",
        ]

        all_open_weight_model_ids = [
            "catboost-classification-model",
            "huggingface-spc-bert-base-cased",
            "lightgbm-classification-model",
            "mxnet-semseg-fcn-resnet50-ade",
            "pytorch-eqa-bert-base-cased",
            "sklearn-classification-linear",
            "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1",
            "xgboost-classification-model",
        ]

        assert list_jumpstart_models("model_type == proprietary") == all_prop_model_ids
        assert list_jumpstart_models("model_type == marketplace") == all_prop_model_ids
        assert list_jumpstart_models("model_type == open_weights") == all_open_weight_model_ids

        assert list_jumpstart_models(list_versions=False) == sorted(
            all_prop_model_ids + all_open_weight_model_ids
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.notebook_utils.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.read_s3_file")
    def test_list_jumpstart_models_complex_queries(
        self,
        patched_read_s3_file: Mock,
        patched_get_manifest: Mock,
    ):
        patched_read_s3_file.side_effect = lambda *args, **kwargs: json.dumps(
            get_prototype_model_spec(None, "pytorch-eqa-bert-base-cased").to_json()
        )
        patched_get_manifest.side_effect = (
            lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
        )

        assert list_jumpstart_models(
            Or(
                And(
                    "task is ic",
                    "framework is not huggingface",
                    And("training_supported is true", Not("false")),
                    "true",
                ),
                "false",
                "unknown",
            )
        ) == ["tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1"]

        assert list_jumpstart_models(
            Or(
                And(
                    "task is ic",
                    "framework==tensorflow",
                    Identity(
                        And(
                            And("incremental_training_supported==falSE"),
                            "true",
                            Or("unknown", "version equals 1.0.0"),
                        )
                    ),
                    And("training_supported is true", Not("false")),
                    "true",
                ),
                "false",
                "unknown",
            )
        ) == ["tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1"]

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_list_jumpstart_models_multiple_level_index(
        self,
        patched_get_model_specs: Mock,
        patched_get_manifest: Mock,
    ):
        patched_get_model_specs.side_effect = get_prototype_model_spec
        patched_get_manifest.side_effect = (
            lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
        )

        with pytest.raises(NotImplementedError):
            list_jumpstart_models("hosting_ecr_specs.py_version == py3")


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._get_manifest")
@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_get_model_url(
    patched_get_model_specs: Mock,
    patched_validate_model_id_and_get_type: Mock,
    patched_get_manifest: Mock,
):

    patched_get_model_specs.side_effect = get_prototype_model_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS
    patched_get_manifest.side_effect = (
        lambda region, model_type, *args, **kwargs: get_prototype_manifest(region, model_type)
    )

    model_id, version = "xgboost-classification-model", "1.0.0"
    assert "https://xgboost.readthedocs.io/en/latest/" == get_model_url(model_id, version)

    model_id, version = "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0"
    assert "https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1" == get_model_url(
        model_id, version
    )

    model_id, version = "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0"

    patched_get_model_specs.reset_mock()
    patched_get_model_specs.side_effect = lambda *largs, **kwargs: get_prototype_model_spec(
        *largs,
        region="us-west-2",
        **{key: value for key, value in kwargs.items() if key != "region"},
    )

    get_model_url(model_id, version, region="us-west-2")

    patched_get_model_specs.assert_called_once_with(
        model_id=model_id,
        version=version,
        region="us-west-2",
        s3_client=DEFAULT_JUMPSTART_SAGEMAKER_SESSION.s3_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    )
