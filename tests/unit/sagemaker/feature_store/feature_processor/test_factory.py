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

import pytest
import test_data_helpers as tdh
from mock import Mock, patch

from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode
from sagemaker.feature_store.feature_processor._factory import (
    UDFWrapperFactory,
    ValidatorFactory,
)
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)
from sagemaker.feature_store.feature_processor._udf_wrapper import UDFWrapper
from sagemaker.feature_store.feature_processor._validation import (
    FeatureProcessorArgValidator,
    InputValidator,
    SparkUDFSignatureValidator,
    InputOffsetValidator,
    BaseDataSourceValidator,
)
from sagemaker.session import Session


def test_get_validation_chain():
    fp_config = tdh.create_fp_config(mode=FeatureProcessorMode.PYSPARK)
    result = ValidatorFactory.get_validation_chain(fp_config)

    assert result.validators is not None
    assert {
        InputValidator,
        FeatureProcessorArgValidator,
        InputOffsetValidator,
        BaseDataSourceValidator,
        SparkUDFSignatureValidator,
    } == {type(instance) for instance in result.validators}


def test_get_udf_wrapper():
    fp_config = tdh.create_fp_config(mode=FeatureProcessorMode.PYSPARK)
    udf_wrapper = Mock(UDFWrapper)

    with patch.object(
        UDFWrapperFactory, "_get_spark_udf_wrapper", return_value=udf_wrapper
    ) as get_udf_wrapper_method:
        result = UDFWrapperFactory.get_udf_wrapper(fp_config)

        assert result == udf_wrapper
        get_udf_wrapper_method.assert_called_with(fp_config)


def test_get_udf_wrapper_invalid_mode():
    fp_config = Mock(FeatureProcessorConfig)
    fp_config.mode = FeatureProcessorMode.PYTHON
    fp_config.sagemaker_session = Mock(Session)

    with pytest.raises(
        ValueError,
        match=r"FeatureProcessorMode FeatureProcessorMode.PYTHON is not supported\.",
    ):
        UDFWrapperFactory.get_udf_wrapper(fp_config)
