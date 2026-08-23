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
"""Regression test for the hub-path environment-variable name rewrite (issue #6191).

The existing coverage for get_instance_specific_environment_variables builds its
fixtures with snake_case keys, which only exercises the from_json (public catalogue)
path. This exercises from_describe_hub_content_response, where DescribeHubContent
returns UpperCamelCase keys and the environment-variable names are themselves map
keys, so a recursive key rewrite renames them.
"""
from __future__ import absolute_import

from sagemaker.jumpstart.types import JumpStartInstanceTypeVariants


HUB_INSTANCE_TYPE_VARIANTS = {
    "Variants": {
        "g6": {
            "Properties": {
                "EnvironmentVariables": {
                    "SM_VLLM_MAX_MODEL_LEN": "1024",
                    "FAMILY_LEVEL_ONLY": "yes",
                }
            }
        },
        "ml.g6.24xlarge": {
            "Properties": {
                "EnvironmentVariables": {
                    "HF_HUB_OFFLINE": "1",
                    "SM_VLLM_MAX_MODEL_LEN": "2275",
                    "SM_VLLM_TENSOR_PARALLEL_SIZE": "4",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            }
        },
    }
}


def test_hub_environment_variable_names_are_not_snake_cased():
    """Names must survive verbatim, e.g. not HF_HUB_OFFLINE -> h_f__h_u_b__o_f_f_l_i_n_e."""
    variants = JumpStartInstanceTypeVariants(HUB_INSTANCE_TYPE_VARIANTS, is_hub_content=True)

    assert variants.get_instance_specific_environment_variables(instance_type="ml.g6.24xlarge") == {
        "FAMILY_LEVEL_ONLY": "yes",
        "HF_HUB_OFFLINE": "1",
        "SM_VLLM_MAX_MODEL_LEN": "2275",
        "SM_VLLM_TENSOR_PARALLEL_SIZE": "4",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_hub_instance_specific_env_var_overrides_family_level():
    """The instance-type value must win over the family-level one for the same name."""
    variants = JumpStartInstanceTypeVariants(HUB_INSTANCE_TYPE_VARIANTS, is_hub_content=True)

    env = variants.get_instance_specific_environment_variables(instance_type="ml.g6.24xlarge")

    assert env["SM_VLLM_MAX_MODEL_LEN"] == "2275"
    assert env["FAMILY_LEVEL_ONLY"] == "yes"


def test_hub_environment_variables_absent_for_unknown_instance_type():
    variants = JumpStartInstanceTypeVariants(HUB_INSTANCE_TYPE_VARIANTS, is_hub_content=True)

    assert (
        variants.get_instance_specific_environment_variables(instance_type="ml.p4d.24xlarge") == {}
    )
