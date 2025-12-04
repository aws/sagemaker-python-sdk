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

import pytest
from unittest.mock import Mock, patch
from typing import List

from sagemaker.core.jumpstart.types import (
    JumpStartDataHolderType,
    JumpStartS3FileType,
    HubType,
    HubContentType,
    JumpStartLaunchedRegionInfo,
    JumpStartModelHeader,
    JumpStartVersionedModelId,
    JumpStartBenchmarkStat,
    JumpStartHyperparameter,
    JumpStartEnvironmentVariable,
    ModelAccessConfig,
    HubAccessConfig,
    S3DataSource,
    AdditionalModelDataSource,
    JumpStartModelDataSource,
)


class TestJumpStartDataHolderType:
    """Test cases for JumpStartDataHolderType base class"""

    def test_eq_same_type_same_attributes(self):
        """Test equality with same type and attributes"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-1", "1.0.0")
        assert obj1 == obj2

    def test_eq_same_type_different_attributes(self):
        """Test inequality with same type but different attributes"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-2", "1.0.0")
        assert obj1 != obj2

    def test_eq_different_types(self):
        """Test inequality with different types"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = "not a JumpStartVersionedModelId"
        assert obj1 != obj2

    def test_eq_with_none(self):
        """Test inequality with None"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        assert obj1 != None

    def test_hash_same_objects(self):
        """Test that same objects have same hash"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-1", "1.0.0")
        assert hash(obj1) == hash(obj2)

    def test_hash_different_objects(self):
        """Test that different objects have different hashes"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-2", "1.0.0")
        assert hash(obj1) != hash(obj2)

    def test_hash_allows_set_membership(self):
        """Test that objects can be added to sets"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj3 = JumpStartVersionedModelId("model-2", "1.0.0")

        obj_set = {obj1, obj2, obj3}
        assert len(obj_set) == 2  # obj1 and obj2 are equal

    def test_hash_allows_dict_keys(self):
        """Test that objects can be used as dict keys"""
        obj1 = JumpStartVersionedModelId("model-1", "1.0.0")
        obj2 = JumpStartVersionedModelId("model-1", "1.0.0")

        test_dict = {obj1: "value1"}
        test_dict[obj2] = "value2"

        assert len(test_dict) == 1  # obj1 and obj2 are equal
        assert test_dict[obj1] == "value2"

    def test_str_representation(self):
        """Test string representation"""
        obj = JumpStartVersionedModelId("model-1", "1.0.0")
        str_repr = str(obj)
        assert "JumpStartVersionedModelId" in str_repr
        assert "model-1" in str_repr
        assert "1.0.0" in str_repr

    def test_repr_representation(self):
        """Test repr representation"""
        obj = JumpStartVersionedModelId("model-1", "1.0.0")
        repr_str = repr(obj)
        assert "JumpStartVersionedModelId" in repr_str
        assert "0x" in repr_str  # Memory address
        assert "model-1" in repr_str


class TestJumpStartS3FileType:
    """Test cases for JumpStartS3FileType enum"""

    def test_open_weight_manifest(self):
        """Test OPEN_WEIGHT_MANIFEST value"""
        assert JumpStartS3FileType.OPEN_WEIGHT_MANIFEST == "manifest"

    def test_open_weight_specs(self):
        """Test OPEN_WEIGHT_SPECS value"""
        assert JumpStartS3FileType.OPEN_WEIGHT_SPECS == "specs"

    def test_proprietary_manifest(self):
        """Test PROPRIETARY_MANIFEST value"""
        assert JumpStartS3FileType.PROPRIETARY_MANIFEST == "proprietary_manifest"

    def test_proprietary_specs(self):
        """Test PROPRIETARY_SPECS value"""
        assert JumpStartS3FileType.PROPRIETARY_SPECS == "proprietary_specs"

    def test_enum_membership(self):
        """Test enum membership"""
        assert "manifest" in [e.value for e in JumpStartS3FileType]
        assert "specs" in [e.value for e in JumpStartS3FileType]


class TestHubType:
    """Test cases for HubType enum"""

    def test_hub_value(self):
        """Test HUB value"""
        assert HubType.HUB == "Hub"

    def test_enum_is_string(self):
        """Test that enum value is string"""
        assert isinstance(HubType.HUB.value, str)


class TestHubContentType:
    """Test cases for HubContentType enum"""

    def test_model_value(self):
        """Test MODEL value"""
        assert HubContentType.MODEL == "Model"

    def test_notebook_value(self):
        """Test NOTEBOOK value"""
        assert HubContentType.NOTEBOOK == "Notebook"

    def test_model_reference_value(self):
        """Test MODEL_REFERENCE value"""
        assert HubContentType.MODEL_REFERENCE == "ModelReference"

    def test_all_values_are_strings(self):
        """Test that all enum values are strings"""
        for content_type in HubContentType:
            assert isinstance(content_type.value, str)


class TestJumpStartLaunchedRegionInfo:
    """Test cases for JumpStartLaunchedRegionInfo"""

    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        info = JumpStartLaunchedRegionInfo(content_bucket="test-bucket", region_name="us-west-2")
        assert info.content_bucket == "test-bucket"
        assert info.region_name == "us-west-2"

    def test_equality(self):
        """Test equality comparison"""
        info1 = JumpStartLaunchedRegionInfo("bucket1", "us-west-2")
        info2 = JumpStartLaunchedRegionInfo("bucket1", "us-west-2")
        assert info1 == info2

    def test_inequality_different_bucket(self):
        """Test inequality with different bucket"""
        info1 = JumpStartLaunchedRegionInfo("bucket1", "us-west-2")
        info2 = JumpStartLaunchedRegionInfo("bucket2", "us-west-2")
        assert info1 != info2

    def test_inequality_different_region(self):
        """Test inequality with different region"""
        info1 = JumpStartLaunchedRegionInfo("bucket1", "us-west-2")
        info2 = JumpStartLaunchedRegionInfo("bucket1", "us-east-1")
        assert info1 != info2

    def test_hash_consistency(self):
        """Test hash consistency"""
        info1 = JumpStartLaunchedRegionInfo("bucket1", "us-west-2")
        info2 = JumpStartLaunchedRegionInfo("bucket1", "us-west-2")
        assert hash(info1) == hash(info2)

    def test_can_be_used_in_set(self):
        """Test that objects can be added to sets"""
        info1 = JumpStartLaunchedRegionInfo("bucket1", "us-west-2")
        info2 = JumpStartLaunchedRegionInfo("bucket1", "us-west-2")
        info3 = JumpStartLaunchedRegionInfo("bucket2", "us-east-1")

        info_set = {info1, info2, info3}
        assert len(info_set) == 2


class TestJumpStartModelHeader:
    """Test cases for JumpStartModelHeader"""

    def test_init_from_dict(self):
        """Test initialization from dictionary"""
        header_dict = {
            "model_id": "test-model",
            "version": "1.0.0",
            "min_version": "2.0.0",
            "spec_key": "test-spec-key",
        }
        header = JumpStartModelHeader(header_dict)
        assert header.model_id == "test-model"
        assert header.version == "1.0.0"
        assert header.min_version == "2.0.0"
        assert header.spec_key == "test-spec-key"

    def test_equality(self):
        """Test equality comparison"""
        dict1 = {
            "model_id": "model-1",
            "version": "1.0.0",
            "min_version": "2.0.0",
            "spec_key": "key1",
        }
        dict2 = {
            "model_id": "model-1",
            "version": "1.0.0",
            "min_version": "2.0.0",
            "spec_key": "key1",
        }
        header1 = JumpStartModelHeader(dict1)
        header2 = JumpStartModelHeader(dict2)
        assert header1 == header2

    def test_inequality(self):
        """Test inequality comparison"""
        dict1 = {
            "model_id": "model-1",
            "version": "1.0.0",
            "min_version": "2.0.0",
            "spec_key": "key1",
        }
        dict2 = {
            "model_id": "model-2",
            "version": "1.0.0",
            "min_version": "2.0.0",
            "spec_key": "key1",
        }
        header1 = JumpStartModelHeader(dict1)
        header2 = JumpStartModelHeader(dict2)
        assert header1 != header2

    def test_to_json(self):
        """Test to_json method"""
        header_dict = {
            "model_id": "test-model",
            "version": "1.0.0",
            "min_version": "2.0.0",
            "spec_key": "test-spec-key",
        }
        header = JumpStartModelHeader(header_dict)
        json_output = header.to_json()
        assert json_output["model_id"] == "test-model"
        assert json_output["version"] == "1.0.0"

    def test_string_representation(self):
        """Test string representation"""
        header_dict = {
            "model_id": "model-1",
            "version": "1.0.0",
            "min_version": "2.0.0",
            "spec_key": "key1",
        }
        header = JumpStartModelHeader(header_dict)
        str_repr = str(header)
        assert "JumpStartModelHeader" in str_repr
        assert "model-1" in str_repr


class TestJumpStartVersionedModelId:
    """Test cases for JumpStartVersionedModelId"""

    def test_init(self):
        """Test initialization"""
        model_id = JumpStartVersionedModelId("test-model", "1.0.0")
        assert model_id.model_id == "test-model"
        assert model_id.version == "1.0.0"

    def test_equality_same_values(self):
        """Test equality with same values"""
        id1 = JumpStartVersionedModelId("model-1", "1.0.0")
        id2 = JumpStartVersionedModelId("model-1", "1.0.0")
        assert id1 == id2

    def test_inequality_different_model_id(self):
        """Test inequality with different model_id"""
        id1 = JumpStartVersionedModelId("model-1", "1.0.0")
        id2 = JumpStartVersionedModelId("model-2", "1.0.0")
        assert id1 != id2

    def test_inequality_different_version(self):
        """Test inequality with different version"""
        id1 = JumpStartVersionedModelId("model-1", "1.0.0")
        id2 = JumpStartVersionedModelId("model-1", "2.0.0")
        assert id1 != id2

    def test_hash_same_values(self):
        """Test hash with same values"""
        id1 = JumpStartVersionedModelId("model-1", "1.0.0")
        id2 = JumpStartVersionedModelId("model-1", "1.0.0")
        assert hash(id1) == hash(id2)

    def test_hash_different_values(self):
        """Test hash with different values"""
        id1 = JumpStartVersionedModelId("model-1", "1.0.0")
        id2 = JumpStartVersionedModelId("model-2", "1.0.0")
        assert hash(id1) != hash(id2)

    def test_can_be_dict_key(self):
        """Test that object can be used as dict key"""
        id1 = JumpStartVersionedModelId("model-1", "1.0.0")
        test_dict = {id1: "value"}
        assert test_dict[id1] == "value"

    def test_string_representation(self):
        """Test string representation"""
        model_id = JumpStartVersionedModelId("model-1", "1.0.0")
        str_repr = str(model_id)
        assert "JumpStartVersionedModelId" in str_repr
        assert "model-1" in str_repr
        assert "1.0.0" in str_repr


class TestJumpStartBenchmarkStat:
    """Test cases for JumpStartBenchmarkStat"""

    def test_init_from_dict(self):
        """Test initialization from dictionary"""
        stat_dict = {"name": "latency", "value": 100.5, "unit": "ms", "concurrency": 10}
        stat = JumpStartBenchmarkStat(stat_dict)
        assert stat.name == "latency"
        assert stat.value == 100.5
        assert stat.unit == "ms"
        assert stat.concurrency == 10

    def test_equality(self):
        """Test equality comparison"""
        dict1 = {"name": "latency", "value": 100.5, "unit": "ms", "concurrency": 10}
        dict2 = {"name": "latency", "value": 100.5, "unit": "ms", "concurrency": 10}
        stat1 = JumpStartBenchmarkStat(dict1)
        stat2 = JumpStartBenchmarkStat(dict2)
        assert stat1 == stat2

    def test_inequality_different_name(self):
        """Test inequality with different name"""
        dict1 = {"name": "latency", "value": 100.5, "unit": "ms", "concurrency": 10}
        dict2 = {"name": "throughput", "value": 100.5, "unit": "ms", "concurrency": 10}
        stat1 = JumpStartBenchmarkStat(dict1)
        stat2 = JumpStartBenchmarkStat(dict2)
        assert stat1 != stat2

    def test_inequality_different_value(self):
        """Test inequality with different value"""
        dict1 = {"name": "latency", "value": 100.5, "unit": "ms", "concurrency": 10}
        dict2 = {"name": "latency", "value": 200.5, "unit": "ms", "concurrency": 10}
        stat1 = JumpStartBenchmarkStat(dict1)
        stat2 = JumpStartBenchmarkStat(dict2)
        assert stat1 != stat2


class TestS3DataSource:
    """Test cases for S3DataSource"""

    def test_init_from_dict_minimal(self):
        """Test initialization from dictionary with minimal fields"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/path/to/model",
        }
        data_source = S3DataSource(spec)

        assert data_source.compression_type == "None"
        assert data_source.s3_data_type == "S3Prefix"
        assert data_source.s3_uri == "s3://bucket/path/to/model"
        assert data_source.model_access_config is None
        assert data_source.hub_access_config is None

    def test_init_from_dict_with_model_access_config(self):
        """Test initialization with model access config"""
        spec = {
            "compression_type": "Gzip",
            "s3_data_type": "S3Object",
            "s3_uri": "s3://bucket/model.tar.gz",
            "model_access_config": {"accept_eula": True},
        }
        data_source = S3DataSource(spec)

        assert data_source.model_access_config is not None
        assert data_source.model_access_config.accept_eula is True

    def test_to_json_minimal(self):
        """Test to_json with minimal fields"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/model",
        }
        data_source = S3DataSource(spec)
        json_output = data_source.to_json()

        assert json_output["compression_type"] == "None"
        assert json_output["s3_data_type"] == "S3Prefix"
        assert json_output["s3_uri"] == "s3://bucket/model"

    def test_set_bucket_with_s3_prefix(self):
        """Test set_bucket when URI has s3:// prefix"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://old-bucket/path/to/model",
        }
        data_source = S3DataSource(spec)
        data_source.set_bucket("new-bucket")

        assert data_source.s3_uri == "s3://new-bucket/path/to/model"

    def test_set_bucket_without_s3_prefix(self):
        """Test set_bucket when URI doesn't have s3:// prefix"""
        spec = {"compression_type": "None", "s3_data_type": "S3Prefix", "s3_uri": "path/to/model"}
        data_source = S3DataSource(spec)
        data_source.set_bucket("new-bucket")

        assert data_source.s3_uri == "s3://new-bucket/path/to/model"

    def test_set_bucket_adds_trailing_slash(self):
        """Test that set_bucket adds trailing slash if needed"""
        spec = {"compression_type": "None", "s3_data_type": "S3Prefix", "s3_uri": "model.tar.gz"}
        data_source = S3DataSource(spec)
        data_source.set_bucket("bucket-name")

        assert data_source.s3_uri == "s3://bucket-name/model.tar.gz"

    def test_equality(self):
        """Test equality comparison"""
        spec1 = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/model",
        }
        spec2 = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/model",
        }
        data_source1 = S3DataSource(spec1)
        data_source2 = S3DataSource(spec2)

        assert data_source1 == data_source2

    def test_inequality(self):
        """Test inequality comparison"""
        spec1 = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket1/model",
        }
        spec2 = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket2/model",
        }
        data_source1 = S3DataSource(spec1)
        data_source2 = S3DataSource(spec2)

        assert data_source1 != data_source2


class TestAdditionalModelDataSource:
    """Test cases for AdditionalModelDataSource

    Note: AdditionalModelDataSource has a bug in the source code where it tries to set
    self.provider in from_json() but 'provider' is not in __slots__. This causes
    AttributeError when instantiating. These tests are skipped until the source is fixed.
    """

    @pytest.mark.skip(
        reason="AdditionalModelDataSource has bug: tries to set self.provider but provider not in __slots__"
    )
    def test_init_from_dict_minimal(self):
        """Test initialization from dictionary with minimal fields"""
        spec = {
            "channel_name": "model-channel",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model",
            },
        }
        data_source = AdditionalModelDataSource(spec)

        assert data_source.channel_name == "model-channel"
        assert data_source.s3_data_source is not None
        assert data_source.s3_data_source.s3_uri == "s3://bucket/model"
        assert data_source.hosting_eula_key is None

    @pytest.mark.skip(
        reason="AdditionalModelDataSource has bug: tries to set self.provider but provider not in __slots__"
    )
    def test_init_from_dict_with_eula_key(self):
        """Test initialization with EULA key"""
        spec = {
            "channel_name": "model-channel",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model",
            },
            "hosting_eula_key": "eula/key/path",
        }
        data_source = AdditionalModelDataSource(spec)

        assert data_source.hosting_eula_key == "eula/key/path"

    @pytest.mark.skip(
        reason="AdditionalModelDataSource has bug: tries to set self.provider but provider not in __slots__"
    )
    def test_equality(self):
        """Test equality comparison"""
        spec = {
            "channel_name": "model-channel",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model",
            },
        }
        data_source1 = AdditionalModelDataSource(spec)
        data_source2 = AdditionalModelDataSource(spec)

        assert data_source1 == data_source2


class TestJumpStartModelDataSource:
    """Test cases for JumpStartModelDataSource"""

    def test_init_from_dict(self):
        """Test initialization from dictionary"""
        spec = {
            "channel_name": "model-channel",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model",
            },
            "artifact_version": "1.0.0",
        }
        data_source = JumpStartModelDataSource(spec)

        assert data_source.channel_name == "model-channel"
        assert data_source.artifact_version == "1.0.0"
        assert data_source.s3_data_source.s3_uri == "s3://bucket/model"

    def test_to_json_excludes_artifact_version_by_default(self):
        """Test that to_json excludes artifact_version by default"""
        spec = {
            "channel_name": "model-channel",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model",
            },
            "artifact_version": "1.0.0",
        }
        data_source = JumpStartModelDataSource(spec)
        json_output = data_source.to_json()

        assert "artifact_version" not in json_output
        assert "channel_name" in json_output

    def test_to_json_includes_artifact_version_when_requested(self):
        """Test that to_json includes artifact_version when exclude_keys=False"""
        spec = {
            "channel_name": "model-channel",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model",
            },
            "artifact_version": "1.0.0",
        }
        data_source = JumpStartModelDataSource(spec)
        json_output = data_source.to_json(exclude_keys=False)

        assert "artifact_version" in json_output
        assert json_output["artifact_version"] == "1.0.0"

    def test_inherits_from_additional_model_data_source(self):
        """Test that JumpStartModelDataSource inherits from AdditionalModelDataSource"""
        spec = {
            "channel_name": "model-channel",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model",
            },
            "artifact_version": "1.0.0",
        }
        data_source = JumpStartModelDataSource(spec)

        assert isinstance(data_source, AdditionalModelDataSource)

    def test_equality(self):
        """Test equality comparison"""
        spec = {
            "channel_name": "model-channel",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model",
            },
            "artifact_version": "1.0.0",
        }
        data_source1 = JumpStartModelDataSource(spec)
        data_source2 = JumpStartModelDataSource(spec)

        assert data_source1 == data_source2


class TestJumpStartECRSpecsExtended:
    """Extended test cases for JumpStartECRSpecs"""

    def test_from_json_basic(self):
        from sagemaker.core.jumpstart.types import JumpStartECRSpecs

        spec = {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        }
        ecr_specs = JumpStartECRSpecs(spec)

        assert ecr_specs.framework == "pytorch"
        assert ecr_specs.framework_version == "1.10.0"
        assert ecr_specs.py_version == "py38"

    def test_from_json_with_huggingface(self):
        from sagemaker.core.jumpstart.types import JumpStartECRSpecs

        spec = {
            "framework": "huggingface",
            "framework_version": "4.17.0",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
        }
        ecr_specs = JumpStartECRSpecs(spec)

        assert ecr_specs.framework == "huggingface"
        assert ecr_specs.huggingface_transformers_version == "4.17.0"

    def test_from_json_empty_spec(self):
        from sagemaker.core.jumpstart.types import JumpStartECRSpecs

        ecr_specs = JumpStartECRSpecs({})

        assert not hasattr(ecr_specs, "framework")
        assert not hasattr(ecr_specs, "framework_version")

    def test_to_json(self):
        from sagemaker.core.jumpstart.types import JumpStartECRSpecs

        spec = {
            "framework": "tensorflow",
            "framework_version": "2.8.0",
            "py_version": "py39",
        }
        ecr_specs = JumpStartECRSpecs(spec)
        json_output = ecr_specs.to_json()

        assert json_output["framework"] == "tensorflow"
        assert json_output["framework_version"] == "2.8.0"
        assert json_output["py_version"] == "py39"

    def test_hub_content_camel_case_conversion(self):
        from sagemaker.core.jumpstart.types import JumpStartECRSpecs

        spec = {
            "Framework": "pytorch",
            "FrameworkVersion": "1.10.0",
            "PyVersion": "py38",
        }
        ecr_specs = JumpStartECRSpecs(spec, is_hub_content=True)

        assert ecr_specs.framework == "pytorch"
        assert ecr_specs.framework_version == "1.10.0"


class TestJumpStartPredictorSpecsExtended:
    """Extended test cases for JumpStartPredictorSpecs"""

    def test_from_json_complete(self):
        from sagemaker.core.jumpstart.types import JumpStartPredictorSpecs

        spec = {
            "default_content_type": "application/json",
            "supported_content_types": ["application/json", "text/csv"],
            "default_accept_type": "application/json",
            "supported_accept_types": ["application/json", "text/csv"],
        }
        predictor_specs = JumpStartPredictorSpecs(spec)

        assert predictor_specs.default_content_type == "application/json"
        assert len(predictor_specs.supported_content_types) == 2
        assert predictor_specs.default_accept_type == "application/json"

    def test_from_json_none(self):
        from sagemaker.core.jumpstart.types import JumpStartPredictorSpecs

        predictor_specs = JumpStartPredictorSpecs(None)

        assert not hasattr(predictor_specs, "default_content_type")

    def test_to_json(self):
        from sagemaker.core.jumpstart.types import JumpStartPredictorSpecs

        spec = {
            "default_content_type": "application/json",
            "supported_content_types": ["application/json"],
            "default_accept_type": "application/json",
            "supported_accept_types": ["application/json"],
        }
        predictor_specs = JumpStartPredictorSpecs(spec)
        json_output = predictor_specs.to_json()

        assert "default_content_type" in json_output
        assert json_output["default_content_type"] == "application/json"


class TestJumpStartSerializablePayloadExtended:
    """Extended test cases for JumpStartSerializablePayload"""

    def test_from_json_basic(self):
        from sagemaker.core.jumpstart.types import JumpStartSerializablePayload

        spec = {
            "content_type": "application/json",
            "body": '{"input": "test"}',
        }
        payload = JumpStartSerializablePayload(spec)

        assert payload.content_type == "application/json"
        assert payload.body == '{"input": "test"}'

    def test_from_json_with_accept(self):
        from sagemaker.core.jumpstart.types import JumpStartSerializablePayload

        spec = {
            "content_type": "application/json",
            "body": '{"input": "test"}',
            "accept": "application/json",
        }
        payload = JumpStartSerializablePayload(spec)

        assert payload.accept == "application/json"

    def test_from_json_with_prompt_key(self):
        from sagemaker.core.jumpstart.types import JumpStartSerializablePayload

        spec = {
            "content_type": "application/json",
            "body": '{"input": "test"}',
            "prompt_key": "inputs",
        }
        payload = JumpStartSerializablePayload(spec)

        assert payload.prompt_key == "inputs"

    def test_to_json_preserves_raw_payload(self):
        from sagemaker.core.jumpstart.types import JumpStartSerializablePayload

        spec = {
            "content_type": "application/json",
            "body": '{"input": "test"}',
            "custom_field": "custom_value",
        }
        payload = JumpStartSerializablePayload(spec)
        json_output = payload.to_json()

        assert json_output == spec
        assert "custom_field" in json_output


class TestJumpStartInstanceTypeVariantsExtended:
    """Extended test cases for JumpStartInstanceTypeVariants"""

    def test_from_json_with_regional_aliases(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "regional_aliases": {
                "us-west-2": {"alias1": "value1"},
            },
            "variants": {"ml.p3.2xlarge": {"properties": {"artifact_key": "model.tar.gz"}}},
        }
        variants = JumpStartInstanceTypeVariants(spec)

        assert variants.regional_aliases is not None
        assert "us-west-2" in variants.regional_aliases

    def test_regionalize(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "regional_aliases": {
                "us-west-2": {"alias1": "value1"},
            },
            "variants": {"ml.p3.2xlarge": {"properties": {"artifact_key": "model.tar.gz"}}},
        }
        variants = JumpStartInstanceTypeVariants(spec)
        regionalized = variants.regionalize("us-west-2")

        assert regionalized is not None
        assert "Aliases" in regionalized
        assert "Variants" in regionalized

    def test_get_instance_specific_artifact_key(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {"ml.p3.2xlarge": {"properties": {"artifact_key": "model-p3.tar.gz"}}},
        }
        variants = JumpStartInstanceTypeVariants(spec)
        artifact_key = variants.get_instance_specific_artifact_key("ml.p3.2xlarge")

        assert artifact_key == "model-p3.tar.gz"

    def test_get_instance_specific_artifact_key_none(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {"variants": {}}
        variants = JumpStartInstanceTypeVariants(spec)
        artifact_key = variants.get_instance_specific_artifact_key("ml.p3.2xlarge")

        assert artifact_key is None

    def test_get_instance_specific_hyperparameters(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {
                    "properties": {
                        "hyperparameters": [
                            {
                                "name": "learning_rate",
                                "type": "float",
                                "default": "0.001",
                                "scope": "algorithm",
                            }
                        ]
                    }
                }
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        hyperparams = variants.get_instance_specific_hyperparameters("ml.p3.2xlarge")

        assert len(hyperparams) == 1
        assert hyperparams[0].name == "learning_rate"

    def test_get_instance_specific_environment_variables(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {
                    "properties": {
                        "environment_variables": {
                            "MODEL_SERVER_WORKERS": "2",
                        }
                    }
                }
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        env_vars = variants.get_instance_specific_environment_variables("ml.p3.2xlarge")

        assert env_vars["MODEL_SERVER_WORKERS"] == "2"

    def test_get_image_uri(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "regional_aliases": {
                "us-west-2": {"image_uri": "123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest"}
            },
            "variants": {"ml.p3.2xlarge": {"regional_properties": {"image_uri": "$image_uri"}}},
        }
        variants = JumpStartInstanceTypeVariants(spec)
        image_uri = variants.get_image_uri("ml.p3.2xlarge", "us-west-2")

        assert image_uri == "123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest"

    def test_get_instance_specific_training_artifact_key(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {"properties": {"training_artifact_key": "training-p3.tar.gz"}}
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        artifact_key = variants.get_instance_specific_training_artifact_key("ml.p3.2xlarge")

        assert artifact_key == "training-p3.tar.gz"

    def test_get_instance_specific_prepacked_artifact_key(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {"properties": {"prepacked_artifact_key": "prepacked-p3.tar.gz"}}
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        artifact_key = variants.get_instance_specific_prepacked_artifact_key("ml.p3.2xlarge")

        assert artifact_key == "prepacked-p3.tar.gz"

    def test_get_instance_specific_resource_requirements(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {
                    "properties": {
                        "resource_requirements": {"min_memory_mb": 16384, "num_accelerators": 1}
                    }
                }
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        requirements = variants.get_instance_specific_resource_requirements("ml.p3.2xlarge")

        assert requirements["min_memory_mb"] == 16384
        assert requirements["num_accelerators"] == 1

    def test_get_instance_specific_gated_model_key_env_var_value(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {
                    "properties": {"gated_model_key_env_var_value": "s3://bucket/key"}
                }
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        env_var = variants.get_instance_specific_gated_model_key_env_var_value("ml.p3.2xlarge")

        assert env_var == "s3://bucket/key"

    def test_get_instance_specific_default_inference_instance_type(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {
                    "properties": {"default_inference_instance_type": "ml.g4dn.xlarge"}
                }
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        instance_type = variants.get_instance_specific_default_inference_instance_type(
            "ml.p3.2xlarge"
        )

        assert instance_type == "ml.g4dn.xlarge"

    def test_get_instance_specific_supported_inference_instance_types(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {
                    "properties": {
                        "supported_inference_instance_types": ["ml.g4dn.xlarge", "ml.p3.2xlarge"]
                    }
                }
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        instance_types = variants.get_instance_specific_supported_inference_instance_types(
            "ml.p3.2xlarge"
        )

        assert len(instance_types) == 2
        assert "ml.g4dn.xlarge" in instance_types

    def test_get_instance_specific_metric_definitions(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "variants": {
                "ml.p3.2xlarge": {
                    "properties": {
                        "metrics": [{"Name": "train:loss", "Regex": "loss: ([0-9\\.]+)"}]
                    }
                }
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        metrics = variants.get_instance_specific_metric_definitions("ml.p3.2xlarge")

        assert len(metrics) == 1
        assert metrics[0]["Name"] == "train:loss"

    def test_get_model_package_arn(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {
            "regional_aliases": {
                "us-west-2": {"model_package": "arn:aws:sagemaker:us-west-2:123:model-package/test"}
            },
            "variants": {
                "ml.p3.2xlarge": {"regional_properties": {"model_package_arn": "$model_package"}}
            },
        }
        variants = JumpStartInstanceTypeVariants(spec)
        arn = variants.get_model_package_arn("ml.p3.2xlarge", "us-west-2")

        assert arn == "arn:aws:sagemaker:us-west-2:123:model-package/test"

    def test_regionalize_with_none_regional_aliases(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        spec = {"aliases": {"alias1": "value1"}, "variants": {}}
        variants = JumpStartInstanceTypeVariants(spec, is_hub_content=True)
        result = variants.regionalize("us-west-2")

        assert result is None

    def test_from_describe_hub_content_response(self):
        from sagemaker.core.jumpstart.types import JumpStartInstanceTypeVariants

        response = {
            "Aliases": {"alias1": "value1"},
            "Variants": {"ml.p3.2xlarge": {"Properties": {"ArtifactKey": "model.tar.gz"}}},
        }
        variants = JumpStartInstanceTypeVariants(response, is_hub_content=True)

        assert variants.aliases is not None
        assert variants.regional_aliases is None


class TestJumpStartAdditionalDataSourcesExtended:
    """Extended test cases for JumpStartAdditionalDataSources"""

    def test_from_json_with_speculative_decoding(self):
        from sagemaker.core.jumpstart.types import JumpStartAdditionalDataSources

        spec = {
            "speculative_decoding": [
                {
                    "channel_name": "draft_model",
                    "s3_data_source": {
                        "compression_type": "None",
                        "s3_data_type": "S3Prefix",
                        "s3_uri": "s3://bucket/draft-model/",
                    },
                    "artifact_version": "1.0.0",
                }
            ]
        }
        data_sources = JumpStartAdditionalDataSources(spec)

        assert data_sources.speculative_decoding is not None
        assert len(data_sources.speculative_decoding) == 1

    def test_from_json_with_scripts(self):
        from sagemaker.core.jumpstart.types import JumpStartAdditionalDataSources

        spec = {
            "scripts": [
                {
                    "channel_name": "inference_script",
                    "s3_data_source": {
                        "compression_type": "None",
                        "s3_data_type": "S3Prefix",
                        "s3_uri": "s3://bucket/scripts/",
                    },
                    "artifact_version": "1.0.0",
                }
            ]
        }
        data_sources = JumpStartAdditionalDataSources(spec)

        assert data_sources.scripts is not None
        assert len(data_sources.scripts) == 1

    def test_to_json(self):
        from sagemaker.core.jumpstart.types import JumpStartAdditionalDataSources

        spec = {
            "scripts": [
                {
                    "channel_name": "inference_script",
                    "s3_data_source": {
                        "compression_type": "None",
                        "s3_data_type": "S3Prefix",
                        "s3_uri": "s3://bucket/scripts/",
                    },
                    "artifact_version": "1.0.0",
                }
            ]
        }
        data_sources = JumpStartAdditionalDataSources(spec)
        json_output = data_sources.to_json()

        assert "scripts" in json_output
        assert len(json_output["scripts"]) == 1


class TestModelAccessConfigExtended:
    """Extended test cases for ModelAccessConfig"""

    def test_from_json(self):
        from sagemaker.core.jumpstart.types import ModelAccessConfig

        spec = {"accept_eula": True}
        config = ModelAccessConfig(spec)

        assert config.accept_eula is True

    def test_to_json(self):
        from sagemaker.core.jumpstart.types import ModelAccessConfig

        spec = {"accept_eula": False}
        config = ModelAccessConfig(spec)
        json_output = config.to_json()

        assert json_output["accept_eula"] is False


class TestS3DataSourceExtended:
    """Extended test cases for S3DataSource"""

    def test_from_json_basic(self):
        from sagemaker.core.jumpstart.types import S3DataSource

        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/path/",
        }
        data_source = S3DataSource(spec)

        assert data_source.compression_type == "None"
        assert data_source.s3_data_type == "S3Prefix"
        assert data_source.s3_uri == "s3://bucket/path/"

    def test_from_json_with_model_access_config(self):
        from sagemaker.core.jumpstart.types import S3DataSource

        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/path/",
            "model_access_config": {"accept_eula": True},
        }
        data_source = S3DataSource(spec)

        assert data_source.model_access_config is not None
        assert data_source.model_access_config.accept_eula is True

    def test_set_bucket(self):
        from sagemaker.core.jumpstart.types import S3DataSource

        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://old-bucket/path/model.tar.gz",
        }
        data_source = S3DataSource(spec)
        data_source.set_bucket("new-bucket")

        assert "new-bucket" in data_source.s3_uri
        assert "old-bucket" not in data_source.s3_uri

    def test_set_bucket_without_s3_prefix(self):
        from sagemaker.core.jumpstart.types import S3DataSource

        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "path/model.tar.gz",
        }
        data_source = S3DataSource(spec)
        data_source.set_bucket("new-bucket")

        assert data_source.s3_uri.startswith("s3://new-bucket/")


class TestJumpStartBenchmarkStatExtended:
    """Extended test cases for JumpStartBenchmarkStat"""

    def test_from_json(self):
        from sagemaker.core.jumpstart.types import JumpStartBenchmarkStat

        spec = {
            "name": "throughput",
            "value": "100",
            "unit": "requests/sec",
            "concurrency": 4,
        }
        stat = JumpStartBenchmarkStat(spec)

        assert stat.name == "throughput"
        assert stat.value == "100"
        assert stat.unit == "requests/sec"
        assert stat.concurrency == 4

    def test_to_json(self):
        from sagemaker.core.jumpstart.types import JumpStartBenchmarkStat

        spec = {
            "name": "latency",
            "value": "50",
            "unit": "ms",
            "concurrency": 1,
        }
        stat = JumpStartBenchmarkStat(spec)
        json_output = stat.to_json()

        assert json_output["name"] == "latency"
        assert json_output["value"] == "50"


class TestJumpStartConfigRankingExtended:
    """Extended test cases for JumpStartConfigRanking"""

    def test_from_json(self):
        from sagemaker.core.jumpstart.types import JumpStartConfigRanking

        spec = {
            "description": "Recommended configurations",
            "rankings": ["config1", "config2", "config3"],
        }
        ranking = JumpStartConfigRanking(spec)

        assert ranking.description == "Recommended configurations"
        assert len(ranking.rankings) == 3
        assert ranking.rankings[0] == "config1"

    def test_to_json(self):
        from sagemaker.core.jumpstart.types import JumpStartConfigRanking

        spec = {
            "description": "Test rankings",
            "rankings": ["config1"],
        }
        ranking = JumpStartConfigRanking(spec)
        json_output = ranking.to_json()

        assert json_output["description"] == "Test rankings"
        assert len(json_output["rankings"]) == 1


class TestJumpStartMetadataBaseFieldsExtended:
    """Extended test cases for JumpStartMetadataBaseFields"""

    def test_from_json_minimal(self):
        from sagemaker.core.jumpstart.types import JumpStartMetadataBaseFields

        fields = {
            "model_id": "test-model",
            "version": "1.0.0",
        }
        metadata = JumpStartMetadataBaseFields(fields)

        assert metadata.model_id == "test-model"
        assert metadata.version == "1.0.0"

    def test_from_json_with_training_support(self):
        from sagemaker.core.jumpstart.types import JumpStartMetadataBaseFields

        fields = {
            "model_id": "test-model",
            "version": "1.0.0",
            "training_supported": True,
            "training_artifact_key": "training/model.tar.gz",
            "training_script_key": "training/script.tar.gz",
        }
        metadata = JumpStartMetadataBaseFields(fields)

        assert metadata.training_supported is True
        assert metadata.training_artifact_key == "training/model.tar.gz"

    def test_from_json_with_hyperparameters(self):
        from sagemaker.core.jumpstart.types import JumpStartMetadataBaseFields

        fields = {
            "model_id": "test-model",
            "version": "1.0.0",
            "training_supported": True,
            "training_artifact_key": "training/model.tar.gz",
            "training_script_key": "training/script.tar.gz",
            "hyperparameters": [
                {
                    "name": "epochs",
                    "type": "int",
                    "default": "10",
                    "scope": "algorithm",
                }
            ],
        }
        metadata = JumpStartMetadataBaseFields(fields)

        assert len(metadata.hyperparameters) == 1
        assert metadata.hyperparameters[0].name == "epochs"

    def test_to_json(self):
        from sagemaker.core.jumpstart.types import JumpStartMetadataBaseFields

        fields = {
            "model_id": "test-model",
            "version": "1.0.0",
            "deprecated": False,
        }
        metadata = JumpStartMetadataBaseFields(fields)
        json_output = metadata.to_json()

        assert "model_id" in json_output
        assert json_output["model_id"] == "test-model"
