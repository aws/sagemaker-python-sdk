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


from mock.mock import patch

from sagemaker.jumpstart import artifacts

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec


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
