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
import unittest
from unittest.mock import Mock


from mock.mock import patch
import pytest

import copy
from sagemaker.jumpstart import artifacts
from sagemaker.jumpstart.artifacts.environment_variables import (
    _retrieve_default_environment_variables,
)
from sagemaker.jumpstart.artifacts.hyperparameters import _retrieve_default_hyperparameters
from sagemaker.jumpstart.artifacts.image_uris import _retrieve_image_uri
from sagemaker.jumpstart.artifacts.incremental_training import _model_supports_incremental_training
from sagemaker.jumpstart.artifacts.instance_types import _retrieve_default_instance_type
from sagemaker.jumpstart.artifacts.metric_definitions import (
    _retrieve_default_training_metric_definitions,
)
from sagemaker.jumpstart.artifacts.model_uris import (
    _retrieve_hosting_prepacked_artifact_key,
    _retrieve_hosting_artifact_key,
    _retrieve_training_artifact_key,
)
from sagemaker.jumpstart.artifacts.script_uris import _retrieve_script_uri
from sagemaker.jumpstart.types import JumpStartModelSpecs
from tests.unit.sagemaker.jumpstart.constants import (
    BASE_SPEC,
)

from sagemaker.jumpstart.artifacts.model_packages import _retrieve_model_package_arn
from sagemaker.jumpstart.artifacts.model_uris import _retrieve_model_uri
from sagemaker.jumpstart.enums import JumpStartScriptScope

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec, get_special_model_spec
from tests.unit.sagemaker.workflow.conftest import mock_client


class ModelArtifactVariantsTest(unittest.TestCase):
    def test_retrieve_hosting_prepacked_artifact_key(self):

        test_spec = copy.deepcopy(BASE_SPEC)

        test_spec["hosting_prepacked_artifact_key"] = "some/thing"

        test_spec["hosting_instance_type_variants"] = {
            "regional_aliases": {
                "us-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.ama"
                    "zonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                }
            },
            "variants": {
                "c4": {
                    "regional_properties": {
                        "image_uri": "$alias_ecr_uri_1",
                    },
                    "properties": {
                        "prepacked_artifact_key": "in/the/way",
                    },
                }
            },
        }

        self.assertEqual(
            _retrieve_hosting_prepacked_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "in/the/way",
        )

        test_spec["hosting_prepacked_artifact_key"] = None

        self.assertEqual(
            _retrieve_hosting_prepacked_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "in/the/way",
        )

        test_spec["hosting_instance_type_variants"] = None

        self.assertEqual(
            _retrieve_hosting_prepacked_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            None,
        )

        test_spec["hosting_prepacked_artifact_key"] = "shemoves"

        self.assertEqual(
            _retrieve_hosting_prepacked_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "shemoves",
        )

    def test_retrieve_hosting_artifact_key(self):

        test_spec = copy.deepcopy(BASE_SPEC)

        test_spec["hosting_artifact_key"] = "some/thing"

        test_spec["hosting_instance_type_variants"] = {
            "regional_aliases": {
                "us-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.ama"
                    "zonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                }
            },
            "variants": {
                "c4": {
                    "regional_properties": {
                        "image_uri": "$alias_ecr_uri_1",
                    },
                    "properties": {
                        "artifact_key": "in/the/way",
                    },
                }
            },
        }

        self.assertEqual(
            _retrieve_hosting_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "in/the/way",
        )

        test_spec["hosting_artifact_key"] = None

        self.assertEqual(
            _retrieve_hosting_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "in/the/way",
        )

        test_spec["hosting_instance_type_variants"] = None

        self.assertEqual(
            _retrieve_hosting_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            None,
        )

        test_spec["hosting_artifact_key"] = "shemoves"

        self.assertEqual(
            _retrieve_hosting_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "shemoves",
        )

    def test_retrieve_training_artifact_key(self):

        test_spec = copy.deepcopy(BASE_SPEC)

        test_spec["training_artifact_key"] = "some/thing"

        test_spec["training_instance_type_variants"] = {
            "regional_aliases": {
                "us-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-2."
                    "amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                }
            },
            "variants": {
                "c4": {
                    "regional_properties": {
                        "image_uri": "$alias_ecr_uri_1",
                    },
                    "properties": {
                        "artifact_key": "in/the/way",
                    },
                }
            },
        }

        self.assertEqual(
            _retrieve_training_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "in/the/way",
        )

        test_spec["training_artifact_key"] = None

        self.assertEqual(
            _retrieve_training_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "in/the/way",
        )

        test_spec["training_instance_type_variants"] = None

        self.assertEqual(
            _retrieve_training_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            None,
        )

        test_spec["training_artifact_key"] = "shemoves"

        self.assertEqual(
            _retrieve_training_artifact_key(
                JumpStartModelSpecs(test_spec), instance_type="ml.c4.xlarge"
            ),
            "shemoves",
        )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
class RetrieveKwargsTest(unittest.TestCase):

    model_id, model_version = "pytorch-eqa-bert-base-cased", "*"
    region = "us-west-2"

    def test_model_kwargs(self, patched_get_model_specs):

        patched_get_model_specs.side_effect = get_spec_from_base_spec

        kwargs = artifacts._retrieve_model_init_kwargs(
            region=self.region,
            model_id=self.model_id,
            model_version=self.model_version,
        )

        assert kwargs == {
            "some-model-kwarg-key": "some-model-kwarg-value",
            "enable_network_isolation": True,
        }

    @patch("sagemaker.jumpstart.artifacts.kwargs.volume_size_supported")
    def test_estimator_kwargs(self, patched_volume_size_supported, patched_get_model_specs):

        patched_volume_size_supported.return_value = False
        patched_get_model_specs.side_effect = get_spec_from_base_spec

        kwargs = artifacts._retrieve_estimator_init_kwargs(
            region=self.region,
            model_id=self.model_id,
            model_version=self.model_version,
            instance_type="blah",
        )

        assert kwargs == {
            "encrypt_inter_container_traffic": True,
            "enable_network_isolation": False,
        }

    @patch("sagemaker.jumpstart.artifacts.kwargs.volume_size_supported")
    def test_estimator_kwargs_with_volume_size(
        self, patched_volume_size_supported, patched_get_model_specs
    ):

        patched_volume_size_supported.return_value = True
        patched_get_model_specs.side_effect = get_spec_from_base_spec

        kwargs = artifacts._retrieve_estimator_init_kwargs(
            region=self.region,
            model_id=self.model_id,
            model_version=self.model_version,
            instance_type="blah",
        )

        assert kwargs == {
            "encrypt_inter_container_traffic": True,
            "volume_size": 456,
            "enable_network_isolation": False,
        }

    @patch("sagemaker.jumpstart.artifacts.kwargs.volume_size_supported")
    def test_model_deploy_kwargs(self, patched_volume_size_supported, patched_get_model_specs):

        patched_volume_size_supported.return_value = False

        patched_get_model_specs.side_effect = get_spec_from_base_spec

        kwargs = artifacts._retrieve_model_deploy_kwargs(
            region=self.region,
            model_id=self.model_id,
            model_version=self.model_version,
            instance_type="blah",
        )

        assert kwargs == {"some-model-deploy-kwarg-key": "some-model-deploy-kwarg-value"}

    @patch("sagemaker.jumpstart.artifacts.kwargs.volume_size_supported")
    def test_model_deploy_kwargs_with_volume_size(
        self, patched_volume_size_supported, patched_get_model_specs
    ):

        patched_volume_size_supported.return_value = True

        patched_get_model_specs.side_effect = get_spec_from_base_spec

        kwargs = artifacts._retrieve_model_deploy_kwargs(
            region=self.region,
            model_id=self.model_id,
            model_version=self.model_version,
            instance_type="blah",
        )

        assert kwargs == {
            "some-model-deploy-kwarg-key": "some-model-deploy-kwarg-value",
            "volume_size": 123,
        }

    def test_estimator_fit_kwargs(self, patched_get_model_specs):

        patched_get_model_specs.side_effect = get_spec_from_base_spec

        kwargs = artifacts._retrieve_estimator_fit_kwargs(
            region=self.region,
            model_id=self.model_id,
            model_version=self.model_version,
        )

        assert kwargs == {"some-estimator-fit-key": "some-estimator-fit-value"}


class RetrieveModelPackageArnTest(unittest.TestCase):

    mock_session = Mock(s3_client=mock_client)

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_retrieve_model_package_arn(self, patched_get_model_specs):
        patched_get_model_specs.side_effect = get_special_model_spec

        model_id = "variant-model"
        region = "us-west-2"

        assert (
            _retrieve_model_package_arn(
                region=region,
                model_id=model_id,
                scope=JumpStartScriptScope.INFERENCE,
                model_version="*",
                sagemaker_session=self.mock_session,
                instance_type="ml.p2.48xlarge",
            )
            == "us-west-2/blah/blah/blah/gpu"
        )

        assert (
            _retrieve_model_package_arn(
                region=region,
                model_id=model_id,
                scope=JumpStartScriptScope.INFERENCE,
                model_version="*",
                sagemaker_session=self.mock_session,
                instance_type="ml.p4.2xlarge",
            )
            == "us-west-2/blah/blah/blah/gpu"
        )

        assert (
            _retrieve_model_package_arn(
                region=region,
                model_id=model_id,
                scope=JumpStartScriptScope.INFERENCE,
                model_version="*",
                sagemaker_session=self.mock_session,
                instance_type="ml.inf1.2xlarge",
            )
            == "us-west-2/blah/blah/blah/inf"
        )

        assert (
            _retrieve_model_package_arn(
                region=region,
                model_id=model_id,
                scope=JumpStartScriptScope.INFERENCE,
                model_version="*",
                sagemaker_session=self.mock_session,
                instance_type="ml.inf2.12xlarge",
            )
            == "us-west-2/blah/blah/blah/inf"
        )

        assert (
            _retrieve_model_package_arn(
                region=region,
                model_id=model_id,
                scope=JumpStartScriptScope.INFERENCE,
                model_version="*",
                sagemaker_session=self.mock_session,
                instance_type="ml.afasfasf.12xlarge",
            )
            == "arn:aws:sagemaker:us-west-2:594846645681:model-package/llama2-7b-v3-740347e540da35b4ab9f6fc0ab3fed2c"
        )

        assert (
            _retrieve_model_package_arn(
                region=region,
                model_id=model_id,
                scope=JumpStartScriptScope.INFERENCE,
                model_version="*",
                sagemaker_session=self.mock_session,
                instance_type="ml.m2.12xlarge",
            )
            == "arn:aws:sagemaker:us-west-2:594846645681:model-package/llama2-7b-v3-740347e540da35b4ab9f6fc0ab3fed2c"
        )

        assert (
            _retrieve_model_package_arn(
                region=region,
                model_id=model_id,
                scope=JumpStartScriptScope.INFERENCE,
                model_version="*",
                sagemaker_session=self.mock_session,
                instance_type="nobodycares",
            )
            == "arn:aws:sagemaker:us-west-2:594846645681:model-package/llama2-7b-v3-740347e540da35b4ab9f6fc0ab3fed2c"
        )

        with pytest.raises(ValueError):
            _retrieve_model_package_arn(
                region="cn-north-1",
                model_id=model_id,
                scope=JumpStartScriptScope.INFERENCE,
                model_version="*",
                sagemaker_session=self.mock_session,
                instance_type="ml.p2.12xlarge",
            )


class PrivateJumpStartBucketTest(unittest.TestCase):

    mock_session = Mock(s3_client=mock_client)

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_retrieve_uri_from_gated_bucket(self, patched_get_model_specs):
        patched_get_model_specs.side_effect = get_special_model_spec

        model_id = "private-model"
        region = "us-west-2"

        self.assertEqual(
            _retrieve_model_uri(
                model_id=model_id, model_version="*", model_scope="inference", region=region
            ),
            "s3://jumpstart-private-cache-prod-us-west-2/pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
        )
        self.assertEqual(
            _retrieve_model_uri(
                model_id=model_id, model_version="*", model_scope="training", region=region
            ),
            "s3://jumpstart-private-cache-prod-us-west-2/pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
        )


class HubModelTest(unittest.TestCase):
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
    def test_retrieve_default_environment_variables(self, mock_cache):
        mock_cache.get_hub_model.return_value = JumpStartModelSpecs(spec=copy.deepcopy(BASE_SPEC))

        model_id, version = "pytorch-ic-mobilenet-v2", "1.0.2"
        hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/my-cool-hub"

        self.assertEqual(
            _retrieve_default_environment_variables(
                model_id=model_id,
                model_version=version,
                hub_arn=hub_arn,
                script=JumpStartScriptScope.INFERENCE,
            ),
            {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
        )
        mock_cache.get_hub_model.assert_called_once_with(
            hub_model_arn=(
                f"arn:aws:sagemaker:us-west-2:000000000000:hub-content/my-cool-hub/Model/{model_id}/{version}"
            )
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
    def test_retrieve_image_uri(self, mock_cache):
        mock_cache.get_hub_model.return_value = JumpStartModelSpecs(spec=copy.deepcopy(BASE_SPEC))

        model_id, version = "pytorch-ic-mobilenet-v2", "1.0.2"
        hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/my-cool-hub"

        self.assertEqual(
            _retrieve_image_uri(
                model_id=model_id,
                model_version=version,
                hub_arn=hub_arn,
                instance_type="ml.p3.2xlarge",
                image_scope=JumpStartScriptScope.TRAINING,
            ),
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.5.0-gpu-py3",
        )
        mock_cache.get_hub_model.assert_called_once_with(
            hub_model_arn=(
                f"arn:aws:sagemaker:us-west-2:000000000000:hub-content/my-cool-hub/Model/{model_id}/{version}"
            )
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
    def test_retrieve_default_hyperparameters(self, mock_cache):
        mock_cache.get_hub_model.return_value = JumpStartModelSpecs(spec=copy.deepcopy(BASE_SPEC))

        model_id, version = "pytorch-ic-mobilenet-v2", "1.0.2"
        hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/my-cool-hub"

        self.assertEqual(
            _retrieve_default_hyperparameters(
                model_id=model_id, model_version=version, hub_arn=hub_arn
            ),
            {
                "epochs": "3",
                "adam-learning-rate": "0.05",
                "batch-size": "4",
            },
        )
        mock_cache.get_hub_model.assert_called_once_with(
            hub_model_arn=(
                f"arn:aws:sagemaker:us-west-2:000000000000:hub-content/my-cool-hub/Model/{model_id}/{version}"
            )
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
    def test_model_supports_incremental_training(self, mock_cache):
        mock_cache.get_hub_model.return_value = JumpStartModelSpecs(spec=copy.deepcopy(BASE_SPEC))

        model_id, version = "pytorch-ic-mobilenet-v2", "1.0.2"
        hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/my-cool-hub"

        self.assertEqual(
            _model_supports_incremental_training(
                model_id=model_id, model_version=version, hub_arn=hub_arn, region="us-west-2"
            ),
            True,
        )
        mock_cache.get_hub_model.assert_called_once_with(
            hub_model_arn=(
                f"arn:aws:sagemaker:us-west-2:000000000000:hub-content/my-cool-hub/Model/{model_id}/{version}"
            )
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
    def test_retrieve_default_instance_type(self, mock_cache):
        mock_cache.get_hub_model.return_value = JumpStartModelSpecs(spec=copy.deepcopy(BASE_SPEC))

        model_id, version = "pytorch-ic-mobilenet-v2", "1.0.2"
        hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/my-cool-hub"

        self.assertEqual(
            _retrieve_default_instance_type(
                model_id=model_id,
                model_version=version,
                hub_arn=hub_arn,
                scope=JumpStartScriptScope.TRAINING,
            ),
            "ml.p3.2xlarge",
        )
        mock_cache.get_hub_model.assert_called_once_with(
            hub_model_arn=(
                f"arn:aws:sagemaker:us-west-2:000000000000:hub-content/my-cool-hub/Model/{model_id}/{version}"
            )
        )

        self.assertEqual(
            _retrieve_default_instance_type(
                model_id=model_id,
                model_version=version,
                hub_arn=hub_arn,
                scope=JumpStartScriptScope.INFERENCE,
            ),
            "ml.p2.xlarge",
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
    def test_retrieve_default_training_metric_definitions(self, mock_cache):
        mock_cache.get_hub_model.return_value = JumpStartModelSpecs(spec=copy.deepcopy(BASE_SPEC))

        model_id, version = "pytorch-ic-mobilenet-v2", "1.0.2"
        hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/my-cool-hub"

        self.assertEqual(
            _retrieve_default_training_metric_definitions(
                model_id=model_id, model_version=version, hub_arn=hub_arn, region="us-west-2"
            ),
            [{"Regex": "val_accuracy: ([0-9\\.]+)", "Name": "pytorch-ic:val-accuracy"}],
        )
        mock_cache.get_hub_model.assert_called_once_with(
            hub_model_arn=(
                f"arn:aws:sagemaker:us-west-2:000000000000:hub-content/my-cool-hub/Model/{model_id}/{version}"
            )
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
    def test_retrieve_model_uri(self, mock_cache):
        mock_cache.get_hub_model.return_value = JumpStartModelSpecs(spec=copy.deepcopy(BASE_SPEC))

        model_id, version = "pytorch-ic-mobilenet-v2", "1.0.2"
        hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/my-cool-hub"

        self.assertEqual(
            _retrieve_model_uri(
                model_id=model_id, model_version=version, hub_arn=hub_arn, model_scope="training"
            ),
            "s3://jumpstart-cache-prod-us-west-2/pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
        )
        mock_cache.get_hub_model.assert_called_once_with(
            hub_model_arn=(
                f"arn:aws:sagemaker:us-west-2:000000000000:hub-content/my-cool-hub/Model/{model_id}/{version}"
            )
        )

        self.assertEqual(
            _retrieve_model_uri(
                model_id=model_id, model_version=version, hub_arn=hub_arn, model_scope="inference"
            ),
            "s3://jumpstart-cache-prod-us-west-2/pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor._cache")
    def test_retrieve_script_uri(self, mock_cache):
        mock_cache.get_hub_model.return_value = JumpStartModelSpecs(spec=copy.deepcopy(BASE_SPEC))

        model_id, version = "pytorch-ic-mobilenet-v2", "1.0.2"
        hub_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub/my-cool-hub"

        self.assertEqual(
            _retrieve_script_uri(
                model_id=model_id,
                model_version=version,
                hub_arn=hub_arn,
                script_scope=JumpStartScriptScope.TRAINING,
            ),
            "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/pytorch/"
            "transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        )
        mock_cache.get_hub_model.assert_called_once_with(
            hub_model_arn=(
                f"arn:aws:sagemaker:us-west-2:000000000000:hub-content/my-cool-hub/Model/{model_id}/{version}"
            )
        )

        self.assertEqual(
            _retrieve_script_uri(
                model_id=model_id,
                model_version=version,
                hub_arn=hub_arn,
                script_scope=JumpStartScriptScope.INFERENCE,
            ),
            "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/pytorch/"
            "inference/ic/v1.0.0/sourcedir.tar.gz",
        )
