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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from typing import Callable

import pytest
from mock import Mock

from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)
from sagemaker.feature_store.feature_processor._udf_arg_provider import UDFArgProvider
from sagemaker.feature_store.feature_processor._udf_output_receiver import (
    UDFOutputReceiver,
)
from sagemaker.feature_store.feature_processor._udf_wrapper import UDFWrapper


@pytest.fixture
def udf_arg_provider():
    udf_arg_provider = Mock(UDFArgProvider)
    udf_arg_provider.provide_input_args.return_value = {"input": Mock()}
    udf_arg_provider.provide_params_arg.return_value = {"params": Mock()}
    udf_arg_provider.provide_additional_kwargs.return_value = {"kwarg": Mock()}

    return udf_arg_provider


@pytest.fixture
def udf_output_receiver():
    udf_output_receiver = Mock(UDFOutputReceiver)
    udf_output_receiver.ingest_udf_output.return_value = Mock()
    return udf_output_receiver


@pytest.fixture
def udf_output():
    udf_output = Mock(Callable)
    return udf_output


@pytest.fixture
def udf(udf_output):
    udf = Mock(Callable)
    udf.return_value = udf_output
    return udf


@pytest.fixture
def fp_config():
    fp_config = Mock(FeatureProcessorConfig)
    return fp_config


def test_wrap(fp_config, udf_output, udf_arg_provider, udf_output_receiver):
    def test_udf(input, params, kwarg):
        # Verify wrapped function is called with auto-loaded arguments.
        assert input is udf_arg_provider.provide_input_args.return_value["input"]
        assert params is udf_arg_provider.provide_params_arg.return_value["params"]
        assert kwarg is udf_arg_provider.provide_additional_kwargs.return_value["kwarg"]
        return udf_output

    udf_wrapper = UDFWrapper(udf_arg_provider, udf_output_receiver)

    # Execute decorator function and the decorated function.
    wrapped_udf = udf_wrapper.wrap(test_udf, fp_config)
    wrapped_udf()

    # Verify interactions with dependencies.
    udf_arg_provider.provide_input_args.assert_called_with(test_udf, fp_config)
    udf_arg_provider.provide_params_arg.assert_called_with(test_udf, fp_config)
    udf_arg_provider.provide_additional_kwargs.assert_called_with(test_udf)
    udf_output_receiver.ingest_udf_output.assert_called_with(udf_output, fp_config)
