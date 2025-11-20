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
"""Tests for optimization validation module"""
from __future__ import absolute_import

import pytest

from sagemaker.serve.validations.optimization import (
    _validate_optimization_configuration,
    _OptimizationContainer,
    _OptimizationCombination,
    TRT_CONFIGURATION,
    VLLM_CONFIGURATION,
    NEURON_CONFIGURATION,
    TRUTHY_SET,
    FALSY_SET,
)


class TestOptimizationCombination:
    def test_validate_against_compilation_invalid(self):
        rule = _OptimizationCombination(
            compilation={None, False},
            quantization_technique={None},
            speculative_decoding={None},
            sharding={None}
        )
        
        test_combo = _OptimizationCombination(
            compilation={True},
            quantization_technique={None},
            speculative_decoding={None},
            sharding={None}
        )
        
        with pytest.raises(ValueError, match="Compilation"):
            rule.validate_against(test_combo, _OptimizationContainer.VLLM)
    
    def test_validate_against_quantization_invalid(self):
        rule = _OptimizationCombination(
            compilation={None, True},
            quantization_technique={None, "awq"},
            speculative_decoding={None},
            sharding={None}
        )
        
        test_combo = _OptimizationCombination(
            compilation={True},
            quantization_technique={"fp8"},
            speculative_decoding={None},
            sharding={None}
        )
        
        with pytest.raises(ValueError, match="Quantization"):
            rule.validate_against(test_combo, _OptimizationContainer.TRT)
    
    def test_validate_against_speculative_decoding_invalid(self):
        rule = _OptimizationCombination(
            compilation={None, False},
            quantization_technique={None},
            speculative_decoding={None, False},
            sharding={None}
        )
        
        test_combo = _OptimizationCombination(
            compilation={False},
            quantization_technique={None},
            speculative_decoding={True},
            sharding={None}
        )
        
        with pytest.raises(ValueError, match="Speculative Decoding"):
            rule.validate_against(test_combo, _OptimizationContainer.TRT)
    
    def test_validate_against_sharding_invalid(self):
        rule = _OptimizationCombination(
            compilation={None, False},
            quantization_technique={None},
            speculative_decoding={None},
            sharding={None, False}
        )
        
        test_combo = _OptimizationCombination(
            compilation={False},
            quantization_technique={None},
            speculative_decoding={None},
            sharding={True}
        )
        
        with pytest.raises(ValueError, match="Sharding"):
            rule.validate_against(test_combo, _OptimizationContainer.TRT)
    
    def test_validate_compilation_and_speculative_together(self):
        rule = _OptimizationCombination(
            compilation={None, True},
            quantization_technique={None},
            speculative_decoding={None, True},
            sharding={None}
        )
        
        test_combo = _OptimizationCombination(
            compilation={True},
            quantization_technique={None},
            speculative_decoding={True},
            sharding={None}
        )
        
        with pytest.raises(ValueError, match="Compilation and Speculative Decoding together"):
            rule.validate_against(test_combo, _OptimizationContainer.VLLM)
    
    def test_validate_trt_quantization_without_compilation(self):
        rule = TRT_CONFIGURATION["optimization_combination"]
        
        test_combo = _OptimizationCombination(
            compilation={True},  # TRT requires compilation=True
            quantization_technique={"awq"},
            speculative_decoding={False},
            sharding={False}
        )
        
        # This should pass validation for TRT
        rule.validate_against(test_combo, _OptimizationContainer.TRT)


class TestValidateOptimizationConfiguration:
    def test_invalid_instance_type(self):
        with pytest.raises(ValueError, match="not currently supported"):
            _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.t2.medium",
                quantization_config=None,
                compilation_config=None,
                sharding_config=None,
                speculative_decoding_config=None
            )
    
    def test_no_optimization_configs_non_jumpstart(self):
        with pytest.raises(ValueError, match="provide no optimization configs"):
            _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.g5.xlarge",
                quantization_config=None,
                compilation_config=None,
                sharding_config=None,
                speculative_decoding_config=None
            )
    
    def test_no_optimization_configs_jumpstart_neuron(self):
        # Should not raise for JumpStart with Neuron instances
        _validate_optimization_configuration(
            is_jumpstart=True,
            instance_type="ml.inf2.xlarge",
            quantization_config=None,
            compilation_config=None,
            sharding_config=None,
            speculative_decoding_config=None
        )
    
    def test_neuron_with_sharding_invalid(self):
        with pytest.raises(ValueError, match="not supported on Neuron"):
            _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.inf2.xlarge",
                quantization_config=None,
                compilation_config={"enabled": True},
                sharding_config={"enabled": True},
                speculative_decoding_config=None
            )
    
    def test_neuron_with_compilation_valid(self):
        # Should not raise
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.inf2.xlarge",
            quantization_config=None,
            compilation_config={"enabled": True},
            sharding_config=None,
            speculative_decoding_config=None
        )
    
    def test_gpu_with_compilation_and_quantization(self):
        # Should not raise for TRT with compilation and quantization
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.xlarge",
            quantization_config={
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}
            },
            compilation_config={"enabled": True},
            sharding_config=None,
            speculative_decoding_config=None
        )
    
    def test_gpu_with_sharding_valid(self):
        # Should not raise for VLLM with sharding
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.xlarge",
            quantization_config=None,
            compilation_config=None,
            sharding_config={"enabled": True},
            speculative_decoding_config=None
        )
    
    def test_gpu_with_speculative_decoding_valid(self):
        # Should not raise for VLLM with speculative decoding
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.g5.xlarge",
            quantization_config=None,
            compilation_config=None,
            sharding_config=None,
            speculative_decoding_config={"enabled": True}
        )
    
    def test_gpu_smoothquant_without_compilation(self):
        with pytest.raises(ValueError, match="must be provided with Compilation"):
            _validate_optimization_configuration(
                is_jumpstart=False,
                instance_type="ml.g5.xlarge",
                quantization_config={
                    "OverrideEnvironment": {"OPTION_QUANTIZE": "smoothquant"}
                },
                compilation_config=None,
                sharding_config=None,
                speculative_decoding_config=None
            )
    
    def test_p5_instance_valid(self):
        # Should not raise for p5 instance
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.p5.48xlarge",
            quantization_config=None,
            compilation_config={"enabled": True},
            sharding_config=None,
            speculative_decoding_config=None
        )
    
    def test_trn1_instance_valid(self):
        # Should not raise for trn1 instance
        _validate_optimization_configuration(
            is_jumpstart=False,
            instance_type="ml.trn1.32xlarge",
            quantization_config=None,
            compilation_config={"enabled": True},
            sharding_config=None,
            speculative_decoding_config=None
        )


class TestConstants:
    def test_truthy_set(self):
        assert None in TRUTHY_SET
        assert True in TRUTHY_SET
        assert False not in TRUTHY_SET
    
    def test_falsy_set(self):
        assert None in FALSY_SET
        assert False in FALSY_SET
        assert True not in FALSY_SET
    
    def test_trt_configuration(self):
        assert "p5" in TRT_CONFIGURATION["supported_instance_families"]
        assert "g5" in TRT_CONFIGURATION["supported_instance_families"]
        assert TRT_CONFIGURATION["optimization_combination"].optimization_container == _OptimizationContainer.TRT
    
    def test_vllm_configuration(self):
        assert "p5" in VLLM_CONFIGURATION["supported_instance_families"]
        assert "g5" in VLLM_CONFIGURATION["supported_instance_families"]
        assert VLLM_CONFIGURATION["optimization_combination"].optimization_container == _OptimizationContainer.VLLM
    
    def test_neuron_configuration(self):
        assert "inf2" in NEURON_CONFIGURATION["supported_instance_families"]
        assert "trn1" in NEURON_CONFIGURATION["supported_instance_families"]
        assert NEURON_CONFIGURATION["optimization_combination"].optimization_container == _OptimizationContainer.NEURON
