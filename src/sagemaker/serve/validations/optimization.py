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
"""Holds the validation logic used for the .optimize() function. INTERNAL only"""
from __future__ import absolute_import

import textwrap
import logging
from typing import Any, Dict, Set, Optional
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class _OptimizationContainer(Enum):
    """Optimization containers"""

    TRT = "TRT"
    VLLM = "vLLM"
    NEURON = "Neuron"


class _OptimizationCombination(BaseModel):
    """Optimization ruleset data structure for comparing input to ruleset"""

    optimization_container: _OptimizationContainer = None
    compilation: Set[Optional[bool]]
    speculative_decoding: Set[Optional[bool]]
    sharding: Set[Optional[bool]]
    quantization_technique: Set[Optional[str]]

    def validate_against(self, optimization_combination, rule_set: _OptimizationContainer):
        """Validator for optimization containers"""

        # check the validity of each individual field
        if not optimization_combination.compilation.issubset(self.compilation):
            raise ValueError("Compilation")
        if not optimization_combination.quantization_technique.issubset(
            self.quantization_technique
        ):
            copy_quantization_technique = optimization_combination.quantization_technique.copy()
            raise ValueError(f"Quantization:{copy_quantization_technique.pop()}")
        if not optimization_combination.speculative_decoding.issubset(self.speculative_decoding):
            raise ValueError("Speculative Decoding")
        if not optimization_combination.sharding.issubset(self.sharding):
            raise ValueError("Sharding")

        # optimization technique combinations that need to be validated
        if optimization_combination.compilation and optimization_combination.speculative_decoding:
            is_compiled = optimization_combination.compilation.copy().pop()
            is_speculative_decoding = optimization_combination.speculative_decoding.copy().pop()
            if is_compiled and is_speculative_decoding:
                raise ValueError("Compilation and Speculative Decoding together")

        if rule_set == _OptimizationContainer.TRT:
            is_compiled = optimization_combination.compilation.copy().pop()
            is_quantized = optimization_combination.quantization_technique.copy().pop()
            if is_quantized and not is_compiled:
                raise ValueError(f"Quantization:{is_quantized} must be provided with Compilation")


TRUTHY_SET = {None, True}
FALSY_SET = {None, False}
TRT_CONFIGURATION = {
    "supported_instance_families": {"p4d", "p4de", "p5", "g5", "g6"},
    "optimization_combination": _OptimizationCombination(
        optimization_container=_OptimizationContainer.TRT,
        compilation=TRUTHY_SET,
        quantization_technique={None, "awq", "fp8", "smoothquant"},
        speculative_decoding=FALSY_SET,
        sharding=FALSY_SET,
    ),
}
VLLM_CONFIGURATION = {
    "supported_instance_families": {"p4d", "p4de", "p5", "g5", "g6"},
    "optimization_combination": _OptimizationCombination(
        optimization_container=_OptimizationContainer.VLLM,
        compilation=FALSY_SET,
        quantization_technique={None, "awq", "fp8"},
        speculative_decoding=TRUTHY_SET,
        sharding=TRUTHY_SET,
    ),
}
NEURON_CONFIGURATION = {
    "supported_instance_families": {"inf2", "trn1", "trn1n"},
    "optimization_combination": _OptimizationCombination(
        optimization_container=_OptimizationContainer.NEURON,
        compilation=TRUTHY_SET,
        quantization_technique={None},
        speculative_decoding=FALSY_SET,
        sharding=FALSY_SET,
    ),
}


def _validate_optimization_configuration(
    is_jumpstart: bool,
    instance_type: str,
    quantization_config: Dict[str, Any],
    compilation_config: Dict[str, Any],
    sharding_config: Dict[str, Any],
    speculative_decoding_config: Dict[str, Any],
):
    """Validate .optimize() input off of standard ruleset"""

    instance_family = None
    if instance_type:
        split_instance_type = instance_type.split(".")
        if len(split_instance_type) == 3:
            instance_family = split_instance_type[1]

    if (
        instance_family not in TRT_CONFIGURATION["supported_instance_families"]
        and instance_family not in VLLM_CONFIGURATION["supported_instance_families"]
        and instance_family not in NEURON_CONFIGURATION["supported_instance_families"]
    ):
        invalid_instance_type_msg = (
            f"Optimizations that uses {instance_type} instance type are "
            "not currently supported both on GPU and Neuron instances"
        )
        raise ValueError(invalid_instance_type_msg)

    quantization_technique = None
    if (
        quantization_config
        and quantization_config.get("OverrideEnvironment")
        and quantization_config.get("OverrideEnvironment").get("OPTION_QUANTIZE")
    ):
        quantization_technique = quantization_config.get("OverrideEnvironment").get(
            "OPTION_QUANTIZE"
        )

    optimization_combination = _OptimizationCombination(
        compilation={None if compilation_config is None else True},
        speculative_decoding={None if speculative_decoding_config is None else True},
        sharding={None if sharding_config is None else True},
        quantization_technique={quantization_technique},
    )

    # Check the case where no optimization combination is provided
    if (
        optimization_combination.compilation == {None}
        and optimization_combination.quantization_technique == {None}
        and optimization_combination.speculative_decoding == {None}
        and optimization_combination.sharding == {None}
    ):
        # JumpStart has defaults for Inf/Trn instances
        if is_jumpstart and instance_family in NEURON_CONFIGURATION["supported_instance_families"]:
            return
        raise ValueError(
            (
                "Optimizations that provide no optimization configs "
                "are currently not support on both GPU and Neuron instances."
            )
        )

    # Validate based off of instance type
    if instance_family in NEURON_CONFIGURATION["supported_instance_families"]:
        try:
            (
                NEURON_CONFIGURATION["optimization_combination"].validate_against(
                    optimization_combination, rule_set=_OptimizationContainer.NEURON
                )
            )
        except ValueError as neuron_compare_error:
            raise ValueError(
                (
                    f"Optimizations that use {neuron_compare_error} "
                    "are not supported on Neuron instances."
                )
            )
    else:
        if optimization_combination.compilation.copy().pop():  # Compilation is only enabled for TRT
            try:
                TRT_CONFIGURATION["optimization_combination"].validate_against(
                    optimization_combination, rule_set=_OptimizationContainer.TRT
                )
            except ValueError as trt_compare_error:
                raise ValueError(
                    (
                        f"Optimizations that use Compilation and {trt_compare_error} "
                        "are not supported for GPU instances."
                    )
                )
        else:
            try:
                (
                    VLLM_CONFIGURATION["optimization_combination"].validate_against(
                        optimization_combination, rule_set=_OptimizationContainer.VLLM
                    )
                )
            except ValueError as vllm_compare_error:
                try:  # try both VLLM and TRT to cover both rule sets
                    (
                        TRT_CONFIGURATION["optimization_combination"].validate_against(
                            optimization_combination, rule_set=_OptimizationContainer.TRT
                        )
                    )
                except ValueError as trt_compare_error:
                    if (
                        str(trt_compare_error)
                        == "Quantization:smoothquant must be provided with Compilation"
                    ):
                        raise ValueError(
                            f"Optimizations that use {trt_compare_error} for GPU instances."
                        )
                    if str(trt_compare_error) == str(vllm_compare_error):
                        raise ValueError(
                            (
                                f"Optimizations that use {trt_compare_error} "
                                "are not supported for GPU instances."
                            )
                        )
                    joint_error_msg = f"""
                    Optimization cannot be performed for the following reasons:
                    - Optimizations that use {trt_compare_error} are not supported for GPU instances.
                    - Optimizations that use {vllm_compare_error} are not supported for GPU instances.
                    """
                    raise ValueError(textwrap.dedent(joint_error_msg))
