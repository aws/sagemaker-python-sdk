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
from sagemaker.jumpstart.artifacts.model_uris import (
    _retrieve_hosting_prepacked_artifact_key,
    _retrieve_hosting_artifact_key,
    _retrieve_training_artifact_key,
)
from sagemaker.jumpstart.types import JumpStartModelSpecs
from tests.unit.sagemaker.jumpstart.constants import (
    BASE_SPEC,
)

from sagemaker.jumpstart.artifacts.model_packages import _retrieve_model_package_arn
from sagemaker.jumpstart.artifacts.model_uris import _retrieve_model_uri
from sagemaker.jumpstart.enums import JumpStartScriptScope, JumpStartModelType

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

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_retrieve_model_package_arn(
        self, patched_get_model_specs: Mock, patched_validate_model_id_and_get_type: Mock
    ):
        patched_get_model_specs.side_effect = get_special_model_spec
        patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

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

    @patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_retrieve_uri_from_gated_bucket(
        self, patched_get_model_specs, patched_validate_model_id_and_get_type
    ):
        patched_get_model_specs.side_effect = get_special_model_spec
        patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

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
