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
import test_data_helpers as tdh
from mock import Mock, patch

from sagemaker.mlops.feature_store.feature_processor._enums import FeatureProcessorMode
from sagemaker.mlops.feature_store.feature_processor._factory import (
    UDFWrapperFactory,
    ValidatorFactory,
)
from sagemaker.mlops.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)
from sagemaker.mlops.feature_store.feature_processor._udf_wrapper import UDFWrapper
from sagemaker.mlops.feature_store.feature_processor._validation import ValidatorChain
from sagemaker.mlops.feature_store.feature_processor.feature_processor import (
    feature_processor,
)


@pytest.fixture
def udf():
    return Mock(Callable)


@pytest.fixture
def wrapped_udf():
    return Mock()


@pytest.fixture
def udf_wrapper(wrapped_udf):
    mock = Mock(UDFWrapper)
    mock.wrap.return_value = wrapped_udf
    return mock


@pytest.fixture
def validator_chain():
    return Mock(ValidatorChain)


@pytest.fixture
def fp_config():
    mock = Mock(FeatureProcessorConfig)
    mock.mode = FeatureProcessorMode.PYSPARK
    return mock


def test_feature_processor(udf, udf_wrapper, validator_chain, fp_config, wrapped_udf):
    with patch.object(
        FeatureProcessorConfig, "create", return_value=fp_config
    ) as fp_config_create_method:
        with patch.object(
            UDFWrapperFactory, "get_udf_wrapper", return_value=udf_wrapper
        ) as get_udf_wrapper:
            with patch.object(
                ValidatorFactory,
                "get_validation_chain",
                return_value=validator_chain,
            ) as get_validation_chain:
                decorated_udf = feature_processor(
                    inputs=[tdh.FEATURE_GROUP_DATA_SOURCE],
                    output="",
                )(udf)

                assert decorated_udf == wrapped_udf

                fp_config_create_method.assert_called()
                get_udf_wrapper.assert_called_with(fp_config)
                get_validation_chain.assert_called()

                validator_chain.validate.assert_called_with(fp_config=fp_config, udf=udf)
                udf_wrapper.wrap.assert_called_with(fp_config=fp_config, udf=udf)

                assert decorated_udf.feature_processor_config == fp_config


def test_feature_processor_validation_fails(udf, udf_wrapper, validator_chain, fp_config):
    with patch.object(
        FeatureProcessorConfig, "create", return_value=fp_config
    ) as fp_config_create_method:
        with patch.object(
            UDFWrapperFactory, "get_udf_wrapper", return_value=udf_wrapper
        ) as get_udf_wrapper:
            with patch.object(
                ValidatorFactory,
                "get_validation_chain",
                return_value=validator_chain,
            ) as get_validation_chain:
                validator_chain.validate.side_effect = ValueError()

                # Verify validation error is raised to user.
                with pytest.raises(ValueError):
                    feature_processor(
                        inputs=tdh.FEATURE_PROCESSOR_INPUTS,
                        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
                    )(udf)

                # Verify validation failure causes execution to terminate early.
                # Verify FeatureProcessorConfig interactions.
                fp_config_create_method.assert_called()
                get_udf_wrapper.assert_called_once()
                get_validation_chain.assert_called_once()
                validator_chain.validate.assert_called_with(fp_config=fp_config, udf=udf)
                udf_wrapper.wrap.assert_not_called()
