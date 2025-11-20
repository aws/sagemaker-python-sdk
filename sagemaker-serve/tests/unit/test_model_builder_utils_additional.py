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
"""Additional tests for _ModelBuilderUtils utility methods"""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch
from typing import Tuple

from sagemaker.serve.model_builder_utils import _ModelBuilderUtils
from sagemaker.serve.constants import Framework


class TestNormalizeFrameworkToEnum:
    """Test _normalize_framework_to_enum method"""
    
    def test_normalize_none_returns_none(self):
        """Test that None input returns None"""
        utils = _ModelBuilderUtils()
        result = utils._normalize_framework_to_enum(None)
        assert result is None
    
    def test_normalize_framework_enum_returns_same(self):
        """Test that Framework enum input returns the same enum"""
        utils = _ModelBuilderUtils()
        result = utils._normalize_framework_to_enum(Framework.PYTORCH)
        assert result == Framework.PYTORCH
        
        result = utils._normalize_framework_to_enum(Framework.TENSORFLOW)
        assert result == Framework.TENSORFLOW
    
    def test_normalize_pytorch_variants(self):
        """Test normalization of PyTorch framework variants"""
        utils = _ModelBuilderUtils()
        
        assert utils._normalize_framework_to_enum("pytorch") == Framework.PYTORCH
        assert utils._normalize_framework_to_enum("PyTorch") == Framework.PYTORCH
        assert utils._normalize_framework_to_enum("PYTORCH") == Framework.PYTORCH
        assert utils._normalize_framework_to_enum("torch") == Framework.PYTORCH
        assert utils._normalize_framework_to_enum("Torch") == Framework.PYTORCH
    
    def test_normalize_tensorflow_variants(self):
        """Test normalization of TensorFlow framework variants"""
        utils = _ModelBuilderUtils()
        
        assert utils._normalize_framework_to_enum("tensorflow") == Framework.TENSORFLOW
        assert utils._normalize_framework_to_enum("TensorFlow") == Framework.TENSORFLOW
        assert utils._normalize_framework_to_enum("TENSORFLOW") == Framework.TENSORFLOW
        assert utils._normalize_framework_to_enum("tf") == Framework.TENSORFLOW
        assert utils._normalize_framework_to_enum("TF") == Framework.TENSORFLOW
    
    def test_normalize_xgboost_variants(self):
        """Test normalization of XGBoost framework variants"""
        utils = _ModelBuilderUtils()
        
        assert utils._normalize_framework_to_enum("xgboost") == Framework.XGBOOST
        assert utils._normalize_framework_to_enum("XGBoost") == Framework.XGBOOST
        assert utils._normalize_framework_to_enum("XGBOOST") == Framework.XGBOOST
        assert utils._normalize_framework_to_enum("xgb") == Framework.XGBOOST
        assert utils._normalize_framework_to_enum("XGB") == Framework.XGBOOST
    
    def test_normalize_sklearn_variants(self):
        """Test normalization of scikit-learn framework variants"""
        utils = _ModelBuilderUtils()
        
        assert utils._normalize_framework_to_enum("sklearn") == Framework.SKLEARN
        assert utils._normalize_framework_to_enum("scikit-learn") == Framework.SKLEARN
        assert utils._normalize_framework_to_enum("scikit_learn") == Framework.SKLEARN
        assert utils._normalize_framework_to_enum("sk-learn") == Framework.SKLEARN
        assert utils._normalize_framework_to_enum("SKLEARN") == Framework.SKLEARN
    
    def test_normalize_huggingface_variants(self):
        """Test normalization of HuggingFace framework variants"""
        utils = _ModelBuilderUtils()
        
        assert utils._normalize_framework_to_enum("huggingface") == Framework.HUGGINGFACE
        assert utils._normalize_framework_to_enum("HuggingFace") == Framework.HUGGINGFACE
        assert utils._normalize_framework_to_enum("hf") == Framework.HUGGINGFACE
        assert utils._normalize_framework_to_enum("HF") == Framework.HUGGINGFACE
        assert utils._normalize_framework_to_enum("transformers") == Framework.HUGGINGFACE
        assert utils._normalize_framework_to_enum("Transformers") == Framework.HUGGINGFACE
    
    def test_normalize_other_frameworks(self):
        """Test normalization of other supported frameworks"""
        utils = _ModelBuilderUtils()
        
        assert utils._normalize_framework_to_enum("mxnet") == Framework.MXNET
        assert utils._normalize_framework_to_enum("chainer") == Framework.CHAINER
        assert utils._normalize_framework_to_enum("djl") == Framework.DJL
        assert utils._normalize_framework_to_enum("sparkml") == Framework.SPARKML
        assert utils._normalize_framework_to_enum("spark") == Framework.SPARKML
        assert utils._normalize_framework_to_enum("lda") == Framework.LDA
        assert utils._normalize_framework_to_enum("ntm") == Framework.NTM
        assert utils._normalize_framework_to_enum("smd") == Framework.SMD
        assert utils._normalize_framework_to_enum("sagemaker-distribution") == Framework.SMD
    
    def test_normalize_unsupported_framework_returns_none(self):
        """Test that unsupported framework string returns None"""
        utils = _ModelBuilderUtils()
        
        assert utils._normalize_framework_to_enum("unsupported") is None
        assert utils._normalize_framework_to_enum("random_framework") is None
        assert utils._normalize_framework_to_enum("") is None
    
    def test_normalize_non_string_non_enum_returns_none(self):
        """Test that non-string, non-enum input returns None"""
        utils = _ModelBuilderUtils()
        
        assert utils._normalize_framework_to_enum(123) is None
        assert utils._normalize_framework_to_enum([]) is None
        assert utils._normalize_framework_to_enum({}) is None
        assert utils._normalize_framework_to_enum(object()) is None
    
    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive"""
        utils = _ModelBuilderUtils()
        
        # Test various case combinations
        assert utils._normalize_framework_to_enum("PyToRcH") == Framework.PYTORCH
        assert utils._normalize_framework_to_enum("TeNsOrFlOw") == Framework.TENSORFLOW
        assert utils._normalize_framework_to_enum("XgBoOsT") == Framework.XGBOOST


class TestParseLmiVersion:
    """Test _parse_lmi_version method"""
    
    def test_parse_valid_lmi_version(self):
        """Test parsing valid LMI version from image"""
        utils = _ModelBuilderUtils()
        
        image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-deepspeed0.12.6-cu121"
        major, minor, patch = utils._parse_lmi_version(image)
        
        assert major == 0
        assert minor == 28
        assert patch == 0
    
    def test_parse_lmi_version_with_different_format(self):
        """Test parsing LMI version with different tag format"""
        utils = _ModelBuilderUtils()
        
        image = "account.dkr.ecr.region.amazonaws.com/lmi:1.2.3-gpu-py310"
        major, minor, patch = utils._parse_lmi_version(image)
        
        assert major == 1
        assert minor == 2
        assert patch == 3
    
    def test_parse_lmi_version_multiple_versions_in_tag(self):
        """Test parsing when tag has multiple version-like strings"""
        utils = _ModelBuilderUtils()
        
        # Should pick the first version-like string
        image = "account.dkr.ecr.region.amazonaws.com/lmi:2.5.1-deepspeed0.12.6"
        major, minor, patch = utils._parse_lmi_version(image)
        
        assert major == 2
        assert minor == 5
        assert patch == 1
    
    def test_parse_lmi_version_with_higher_versions(self):
        """Test parsing with higher version numbers"""
        utils = _ModelBuilderUtils()
        
        image = "account.dkr.ecr.region.amazonaws.com/lmi:10.25.99-gpu"
        major, minor, patch = utils._parse_lmi_version(image)
        
        assert major == 10
        assert minor == 25
        assert patch == 99
    
    def test_parse_lmi_version_no_version_raises_error(self):
        """Test that image without version raises ValueError"""
        utils = _ModelBuilderUtils()
        
        image = "account.dkr.ecr.region.amazonaws.com/lmi:latest"
        
        with pytest.raises(ValueError, match="Could not find version in image"):
            utils._parse_lmi_version(image)
    
    def test_parse_lmi_version_invalid_format_raises_error(self):
        """Test that invalid version format raises ValueError"""
        utils = _ModelBuilderUtils()
        
        # Version with only 2 parts (missing patch)
        image = "account.dkr.ecr.region.amazonaws.com/lmi:1.2-gpu"
        
        with pytest.raises(ValueError, match="Invalid version format"):
            utils._parse_lmi_version(image)
    
    def test_parse_lmi_version_no_colon_raises_error(self):
        """Test that image without colon raises ValueError"""
        utils = _ModelBuilderUtils()
        
        image = "account.dkr.ecr.region.amazonaws.com/lmi"
        
        with pytest.raises(ValueError):
            utils._parse_lmi_version(image)
    
    def test_parse_lmi_version_with_v_prefix(self):
        """Test parsing version that starts with 'v' (should fail as it's not a digit)"""
        utils = _ModelBuilderUtils()
        
        image = "account.dkr.ecr.region.amazonaws.com/lmi:v1.2.3-gpu"
        
        # 'v1.2.3' starts with 'v', not a digit, so should raise error
        with pytest.raises(ValueError, match="Could not find version in image"):
            utils._parse_lmi_version(image)
    
    def test_parse_lmi_version_with_build_metadata(self):
        """Test parsing version with build metadata"""
        utils = _ModelBuilderUtils()
        
        image = "account.dkr.ecr.region.amazonaws.com/lmi:3.14.159-build123-gpu"
        major, minor, patch = utils._parse_lmi_version(image)
        
        assert major == 3
        assert minor == 14
        assert patch == 159
    
    def test_parse_lmi_version_returns_tuple(self):
        """Test that return type is a tuple of three integers"""
        utils = _ModelBuilderUtils()
        
        image = "account.dkr.ecr.region.amazonaws.com/lmi:1.0.0-gpu"
        result = utils._parse_lmi_version(image)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, int) for x in result)


class TestModelBuilderUtilsHelpers:
    """Test other helper methods in _ModelBuilderUtils"""
    
    def test_normalize_framework_with_whitespace(self):
        """Test that framework normalization handles whitespace"""
        utils = _ModelBuilderUtils()
        
        # Whitespace should be handled by lower() but not stripped
        # These should return None as they don't match exactly
        assert utils._normalize_framework_to_enum(" pytorch") is None
        assert utils._normalize_framework_to_enum("pytorch ") is None
        assert utils._normalize_framework_to_enum(" pytorch ") is None
    
    def test_normalize_framework_comprehensive_mapping(self):
        """Test that all framework mappings are consistent"""
        utils = _ModelBuilderUtils()
        
        # Verify that common aliases map to the same Framework
        pytorch_aliases = ["pytorch", "torch"]
        for alias in pytorch_aliases:
            assert utils._normalize_framework_to_enum(alias) == Framework.PYTORCH
        
        tensorflow_aliases = ["tensorflow", "tf"]
        for alias in tensorflow_aliases:
            assert utils._normalize_framework_to_enum(alias) == Framework.TENSORFLOW
        
        xgboost_aliases = ["xgboost", "xgb"]
        for alias in xgboost_aliases:
            assert utils._normalize_framework_to_enum(alias) == Framework.XGBOOST
    
    def test_parse_lmi_version_edge_case_single_digit_versions(self):
        """Test parsing with single digit version numbers"""
        utils = _ModelBuilderUtils()
        
        image = "account.dkr.ecr.region.amazonaws.com/lmi:1.0.0"
        major, minor, patch = utils._parse_lmi_version(image)
        
        assert major == 1
        assert minor == 0
        assert patch == 0
    
    def test_parse_lmi_version_with_complex_tag(self):
        """Test parsing with complex tag containing multiple hyphens"""
        utils = _ModelBuilderUtils()
        
        image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-deepspeed0.12.6-cu121-ubuntu22.04"
        major, minor, patch = utils._parse_lmi_version(image)
        
        # Should find the first version-like string (0.28.0)
        assert major == 0
        assert minor == 28
        assert patch == 0


class TestFrameworkEnumConsistency:
    """Test Framework enum consistency"""
    
    def test_framework_enum_has_expected_values(self):
        """Test that Framework enum has all expected framework types"""
        expected_frameworks = [
            'PYTORCH', 'TENSORFLOW', 'XGBOOST', 'SKLEARN', 
            'HUGGINGFACE', 'MXNET', 'CHAINER', 'DJL', 
            'SPARKML', 'LDA', 'NTM', 'SMD'
        ]
        
        for fw_name in expected_frameworks:
            assert hasattr(Framework, fw_name), f"Framework.{fw_name} should exist"
    
    def test_normalize_all_enum_values_return_themselves(self):
        """Test that all Framework enum values normalize to themselves"""
        utils = _ModelBuilderUtils()
        
        # Get all Framework enum members
        for framework in Framework:
            result = utils._normalize_framework_to_enum(framework)
            assert result == framework, f"{framework} should normalize to itself"
