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

import attr
import pytest
import test_data_helpers as tdh

from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)


def test_feature_processor_config_is_immutable():
    fp_config = FeatureProcessorConfig.create(
        inputs=[tdh.FEATURE_GROUP_DATA_SOURCE, tdh.S3_DATA_SOURCE],
        output=tdh.OUTPUT_FEATURE_GROUP_ARN,
        mode=FeatureProcessorMode.PYSPARK,
        target_stores=None,
        enable_ingestion=True,
        parameters=None,
        spark_config=None,
    )

    with pytest.raises(attr.exceptions.FrozenInstanceError):
        # Only attempting one field, as FrozenInstanceError indicates all fields are frozen
        # (as opposed to FrozenAttributeError).
        fp_config.inputs = []

    with pytest.raises(
        TypeError,
        match="'FeatureProcessorConfig' object does not support item assignment",
    ):
        fp_config["inputs"] = []
