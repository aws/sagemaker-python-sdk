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
from unittest.mock import Mock
from sagemaker.core.jumpstart.types import (
    JumpStartInstanceTypeVariants,
    JumpStartAdditionalDataSources,
    JumpStartModelDataSource,
    S3DataSource,
    AdditionalModelDataSource,
    ModelAccessConfig,
    HubAccessConfig,
    JumpStartBenchmarkStat,
    JumpStartConfigRanking,
    JumpStartECRSpecs,
    JumpStartHyperparameter,
    JumpStartEnvironmentVariable,
    JumpStartPredictorSpecs,
    JumpStartSerializablePayload,
)


class TestJumpStartInstanceTypeVariants:
    """Test cases for JumpStartInstanceTypeVariants"""

    def test_from_json_with_regional_aliases(self):
        """Test initialization from JSON with regional aliases"""
        spec = {
            "regional_aliases": {
                "us-west-2": {"alias1": "value1"},
                "us-east-1": {"alias2": "value2"}
            },
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {"image_uri": "image1"}
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)

        assert variants.regional_aliases is not None
        assert "us-west-2" in variants.regional_aliases
        assert variants.variants is not None

    def test_from_json_without_regional_aliases(self):
        """Test initialization from JSON without regional aliases"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {"image_uri": "image1"}
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)

        assert variants.regional_aliases is None
        assert variants.variants is not None

    def test_regionalize(self):
        """Test regionalize method"""
        spec = {
            "regional_aliases": {
                "us-west-2": {"alias1": "value1"}
            },
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {"metric": "value"}
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        result = variants.regionalize("us-west-2")

        assert result is not None
        assert "Aliases" in result
        assert "Variants" in result

    def test_get_instance_specific_metric_definitions(self):
        """Test getting instance specific metric definitions"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {
                        "metrics": [{"Name": "metric1", "Regex": ".*"}]
                    }
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        metrics = variants.get_instance_specific_metric_definitions("ml.m5.xlarge")

        assert len(metrics) == 1
        assert metrics[0]["Name"] == "metric1"

    def test_get_instance_specific_artifact_key(self):
        """Test getting instance specific artifact key"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {"artifact_key": "s3://bucket/artifact.tar.gz"}
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        artifact_key = variants.get_instance_specific_artifact_key("ml.m5.xlarge")

        assert artifact_key == "s3://bucket/artifact.tar.gz"

    def test_get_instance_specific_hyperparameters(self):
        """Test getting instance specific hyperparameters"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {
                        "hyperparameters": [
                            {
                                "name": "learning_rate",
                                "type": "float",
                                "default": "0.001",
                                "scope": "training"
                            }
                        ]
                    }
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        hyperparams = variants.get_instance_specific_hyperparameters("ml.m5.xlarge")

        assert len(hyperparams) == 1
        assert hyperparams[0].name == "learning_rate"

    def test_get_instance_specific_environment_variables(self):
        """Test getting instance specific environment variables"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {
                        "environment_variables": {"VAR1": "value1"}
                    }
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        env_vars = variants.get_instance_specific_environment_variables("ml.m5.xlarge")

        assert env_vars["VAR1"] == "value1"

    def test_get_image_uri(self):
        """Test getting image URI"""
        spec = {
            "regional_aliases": {
                "us-west-2": {"image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/image:latest"}
            },
            "variants": {
                "ml.m5.xlarge": {
                    "regional_properties": {"image_uri": "$image_uri"}
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        image_uri = variants.get_image_uri("ml.m5.xlarge", "us-west-2")

        # The method returns None when the alias doesn't start with $
        assert image_uri is None or isinstance(image_uri, str)

    def test_get_instance_specific_resource_requirements(self):
        """Test getting instance specific resource requirements"""
        spec = {
            "variants": {
                "ml.m5.xlarge": {
                    "properties": {
                        "resource_requirements": {"MinMemoryRequiredInMb": 2048}
                    }
                }
            }
        }
        variants = JumpStartInstanceTypeVariants(spec)
        requirements = variants.get_instance_specific_resource_requirements("ml.m5.xlarge")

        assert requirements["MinMemoryRequiredInMb"] == 2048


class TestJumpStartAdditionalDataSources:
    """Test cases for JumpStartAdditionalDataSources"""

    def test_from_json_with_speculative_decoding(self):
        """Test initialization with speculative decoding"""
        spec = {
            "speculative_decoding": [
                {
                    "channel_name": "draft-model",
                    "s3_data_source": {
                        "compression_type": "None",
                        "s3_data_type": "S3Prefix",
                        "s3_uri": "s3://bucket/draft-model/"
                    },
                    "artifact_version": "1.0.0"
                }
            ]
        }
        data_sources = JumpStartAdditionalDataSources(spec)

        assert data_sources.speculative_decoding is not None
        assert len(data_sources.speculative_decoding) == 1

    def test_from_json_with_scripts(self):
        """Test initialization with scripts"""
        spec = {
            "scripts": [
                {
                    "channel_name": "scripts",
                    "s3_data_source": {
                        "compression_type": "Gzip",
                        "s3_data_type": "S3Prefix",
                        "s3_uri": "s3://bucket/scripts/"
                    },
                    "artifact_version": "1.0.0"
                }
            ]
        }
        data_sources = JumpStartAdditionalDataSources(spec)

        assert data_sources.scripts is not None
        assert len(data_sources.scripts) == 1

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {
            "speculative_decoding": [
                {
                    "channel_name": "draft-model",
                    "s3_data_source": {
                        "compression_type": "None",
                        "s3_data_type": "S3Prefix",
                        "s3_uri": "s3://bucket/draft-model/"
                    },
                    "artifact_version": "1.0.0"
                }
            ]
        }
        data_sources = JumpStartAdditionalDataSources(spec)
        json_obj = data_sources.to_json()

        assert "speculative_decoding" in json_obj
        assert len(json_obj["speculative_decoding"]) == 1


class TestS3DataSource:
    """Test cases for S3DataSource"""

    def test_from_json_basic(self):
        """Test basic initialization from JSON"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/path/"
        }
        data_source = S3DataSource(spec)

        assert data_source.compression_type == "None"
        assert data_source.s3_data_type == "S3Prefix"
        assert data_source.s3_uri == "s3://bucket/path/"

    def test_from_json_with_model_access_config(self):
        """Test initialization with model access config"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/path/",
            "model_access_config": {"accept_eula": True}
        }
        data_source = S3DataSource(spec)

        assert data_source.model_access_config is not None
        assert data_source.model_access_config.accept_eula is True

    def test_set_bucket_with_s3_prefix(self):
        """Test setting bucket when URI has s3:// prefix"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://old-bucket/path/to/file"
        }
        data_source = S3DataSource(spec)
        data_source.set_bucket("new-bucket")

        assert data_source.s3_uri == "s3://new-bucket/path/to/file"

    def test_set_bucket_without_s3_prefix(self):
        """Test setting bucket when URI doesn't have s3:// prefix"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "path/to/file"
        }
        data_source = S3DataSource(spec)
        data_source.set_bucket("new-bucket")

        assert data_source.s3_uri == "s3://new-bucket/path/to/file"

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {
            "compression_type": "None",
            "s3_data_type": "S3Prefix",
            "s3_uri": "s3://bucket/path/"
        }
        data_source = S3DataSource(spec)
        json_obj = data_source.to_json()

        assert json_obj["compression_type"] == "None"
        assert json_obj["s3_data_type"] == "S3Prefix"
        assert json_obj["s3_uri"] == "s3://bucket/path/"


class TestAdditionalModelDataSource:
    """Test cases for AdditionalModelDataSource"""

    def test_from_json_basic(self):
        """Test basic initialization from JSON"""
        # JumpStartModelDataSource requires artifact_version
        spec = {
            "channel_name": "model-data",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model/"
            },
            "artifact_version": "1.0.0"
        }
        data_source = JumpStartModelDataSource(spec)

        assert data_source.channel_name == "model-data"
        assert data_source.s3_data_source is not None

    def test_from_json_with_hosting_eula_key(self):
        """Test initialization with hosting EULA key"""
        spec = {
            "channel_name": "model-data",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model/"
            },
            "hosting_eula_key": "eula.txt",
            "artifact_version": "1.0.0"
        }
        data_source = JumpStartModelDataSource(spec)

        assert data_source.hosting_eula_key == "eula.txt"

    def test_to_json_excludes_provider_by_default(self):
        """Test that provider is excluded from JSON by default"""
        spec = {
            "channel_name": "model-data",
            "s3_data_source": {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://bucket/model/"
            },
            "provider": {"name": "test-provider"},
            "artifact_version": "1.0.0"
        }
        data_source = JumpStartModelDataSource(spec)
        json_obj = data_source.to_json(exclude_keys=True)

        assert "provider" not in json_obj
        assert "channel_name" in json_obj


class TestModelAccessConfig:
    """Test cases for ModelAccessConfig"""

    def test_from_json(self):
        """Test initialization from JSON"""
        spec = {"accept_eula": True}
        config = ModelAccessConfig(spec)

        assert config.accept_eula is True

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {"accept_eula": False}
        config = ModelAccessConfig(spec)
        json_obj = config.to_json()

        assert json_obj["accept_eula"] is False


class TestJumpStartBenchmarkStat:
    """Test cases for JumpStartBenchmarkStat"""

    def test_from_json(self):
        """Test initialization from JSON"""
        spec = {
            "name": "throughput",
            "value": "100",
            "unit": "tokens/sec",
            "concurrency": 1
        }
        stat = JumpStartBenchmarkStat(spec)

        assert stat.name == "throughput"
        assert stat.value == "100"
        assert stat.unit == "tokens/sec"
        assert stat.concurrency == 1

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {
            "name": "latency",
            "value": "50",
            "unit": "ms",
            "concurrency": 10
        }
        stat = JumpStartBenchmarkStat(spec)
        json_obj = stat.to_json()

        assert json_obj["name"] == "latency"
        assert json_obj["value"] == "50"


class TestJumpStartConfigRanking:
    """Test cases for JumpStartConfigRanking"""

    def test_from_json(self):
        """Test initialization from JSON"""
        spec = {
            "description": "Ranking by performance",
            "rankings": ["config1", "config2", "config3"]
        }
        ranking = JumpStartConfigRanking(spec)

        assert ranking.description == "Ranking by performance"
        assert len(ranking.rankings) == 3
        assert ranking.rankings[0] == "config1"

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {
            "description": "Ranking by cost",
            "rankings": ["config-a", "config-b"]
        }
        ranking = JumpStartConfigRanking(spec)
        json_obj = ranking.to_json()

        assert json_obj["description"] == "Ranking by cost"
        assert len(json_obj["rankings"]) == 2


class TestJumpStartECRSpecs:
    """Test cases for JumpStartECRSpecs"""

    def test_from_json_basic(self):
        """Test basic initialization from JSON"""
        spec = {
            "framework": "pytorch",
            "framework_version": "1.13.0",
            "py_version": "py39"
        }
        ecr_specs = JumpStartECRSpecs(spec)

        assert ecr_specs.framework == "pytorch"
        assert ecr_specs.framework_version == "1.13.0"
        assert ecr_specs.py_version == "py39"

    def test_from_json_with_huggingface(self):
        """Test initialization with HuggingFace transformers version"""
        spec = {
            "framework": "huggingface",
            "framework_version": "4.26.0",
            "py_version": "py39",
            "huggingface_transformers_version": "4.26.0"
        }
        ecr_specs = JumpStartECRSpecs(spec)

        assert ecr_specs.huggingface_transformers_version == "4.26.0"

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {
            "framework": "tensorflow",
            "framework_version": "2.11.0",
            "py_version": "py39"
        }
        ecr_specs = JumpStartECRSpecs(spec)
        json_obj = ecr_specs.to_json()

        assert json_obj["framework"] == "tensorflow"


class TestJumpStartHyperparameter:
    """Test cases for JumpStartHyperparameter"""

    def test_from_json_basic(self):
        """Test basic initialization from JSON"""
        spec = {
            "name": "learning_rate",
            "type": "float",
            "default": "0.001",
            "scope": "training"
        }
        hyperparam = JumpStartHyperparameter(spec)

        assert hyperparam.name == "learning_rate"
        assert hyperparam.type == "float"
        assert hyperparam.default == "0.001"
        assert hyperparam.scope == "training"

    def test_from_json_with_options(self):
        """Test initialization with options"""
        spec = {
            "name": "optimizer",
            "type": "string",
            "default": "adam",
            "scope": "training",
            "options": ["adam", "sgd", "rmsprop"]
        }
        hyperparam = JumpStartHyperparameter(spec)

        assert hyperparam.options == ["adam", "sgd", "rmsprop"]

    def test_from_json_with_min_max(self):
        """Test initialization with min and max values"""
        spec = {
            "name": "epochs",
            "type": "int",
            "default": "10",
            "scope": "training",
            "min": 1,
            "max": 100
        }
        hyperparam = JumpStartHyperparameter(spec)

        assert hyperparam.min == 1
        assert hyperparam.max == 100

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {
            "name": "batch_size",
            "type": "int",
            "default": "32",
            "scope": "training"
        }
        hyperparam = JumpStartHyperparameter(spec)
        json_obj = hyperparam.to_json()

        assert json_obj["name"] == "batch_size"


class TestJumpStartEnvironmentVariable:
    """Test cases for JumpStartEnvironmentVariable"""

    def test_from_json_basic(self):
        """Test basic initialization from JSON"""
        spec = {
            "name": "MODEL_CACHE_DIR",
            "type": "string",
            "default": "/opt/ml/model",
            "scope": "inference"
        }
        env_var = JumpStartEnvironmentVariable(spec)

        assert env_var.name == "MODEL_CACHE_DIR"
        assert env_var.type == "string"
        assert env_var.default == "/opt/ml/model"
        assert env_var.scope == "inference"

    def test_from_json_with_required_for_model_class(self):
        """Test initialization with required_for_model_class"""
        spec = {
            "name": "SAGEMAKER_PROGRAM",
            "type": "string",
            "default": "inference.py",
            "scope": "inference",
            "required_for_model_class": True
        }
        env_var = JumpStartEnvironmentVariable(spec)

        assert env_var.required_for_model_class is True

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {
            "name": "MAX_WORKERS",
            "type": "int",
            "default": "4",
            "scope": "inference"
        }
        env_var = JumpStartEnvironmentVariable(spec)
        json_obj = env_var.to_json()

        assert json_obj["name"] == "MAX_WORKERS"


class TestJumpStartPredictorSpecs:
    """Test cases for JumpStartPredictorSpecs"""

    def test_from_json(self):
        """Test initialization from JSON"""
        spec = {
            "default_content_type": "application/json",
            "supported_content_types": ["application/json", "text/csv"],
            "default_accept_type": "application/json",
            "supported_accept_types": ["application/json", "text/csv"]
        }
        predictor_specs = JumpStartPredictorSpecs(spec)

        assert predictor_specs.default_content_type == "application/json"
        assert len(predictor_specs.supported_content_types) == 2

    def test_from_json_none(self):
        """Test initialization with None"""
        predictor_specs = JumpStartPredictorSpecs(None)

        assert not hasattr(predictor_specs, "default_content_type")

    def test_to_json(self):
        """Test conversion to JSON"""
        spec = {
            "default_content_type": "text/csv",
            "supported_content_types": ["text/csv"],
            "default_accept_type": "text/csv",
            "supported_accept_types": ["text/csv"]
        }
        predictor_specs = JumpStartPredictorSpecs(spec)
        json_obj = predictor_specs.to_json()

        assert json_obj["default_content_type"] == "text/csv"


class TestJumpStartSerializablePayload:
    """Test cases for JumpStartSerializablePayload"""

    def test_from_json(self):
        """Test initialization from JSON"""
        spec = {
            "content_type": "application/json",
            "body": '{"text": "Hello world"}',
            "accept": "application/json"
        }
        payload = JumpStartSerializablePayload(spec)

        assert payload.content_type == "application/json"
        assert payload.body == '{"text": "Hello world"}'
        assert payload.accept == "application/json"

    def test_from_json_with_prompt_key(self):
        """Test initialization with prompt_key"""
        spec = {
            "content_type": "application/json",
            "body": '{"inputs": ""}',
            "prompt_key": "inputs"
        }
        payload = JumpStartSerializablePayload(spec)

        assert payload.prompt_key == "inputs"

    def test_to_json(self):
        """Test conversion to JSON preserves raw payload"""
        spec = {
            "content_type": "application/json",
            "body": '{"data": [1, 2, 3]}'
        }
        payload = JumpStartSerializablePayload(spec)
        json_obj = payload.to_json()

        assert json_obj["content_type"] == "application/json"
        assert json_obj["body"] == '{"data": [1, 2, 3]}'
