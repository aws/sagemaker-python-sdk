"""TGI ModelBuilder Utils"""

from __future__ import absolute_import

from typing import Dict
from sagemaker.serve.model_server.djl_serving.utils import (
    _get_default_tensor_parallel_degree,
    _get_default_max_tokens,
)
from sagemaker.serve.builder.schema_builder import SchemaBuilder


def _get_default_dtype():
    """Placeholder docstring"""
    return "bfloat16"


def _get_default_tgi_configurations(
    model_id: str, hf_model_config: dict, schema_builder: SchemaBuilder
) -> Dict[str, str]:
    """Get default TGI configurations"""
    default_num_shard = _get_default_tensor_parallel_degree(hf_model_config)
    _, default_max_new_tokens = _get_default_max_tokens(
        schema_builder.sample_input, schema_builder.sample_output
    )

    if default_num_shard:
        return (
            {
                "SHARDED": "true" if default_num_shard > 1 else "false",
                "NUM_SHARD": str(default_num_shard),
                "DTYPE": _get_default_dtype(),
            },
            default_max_new_tokens,
        )
    return (
        {
            "SHARDED": None,
            "NUM_SHARD": None,
            "DTYPE": _get_default_dtype(),
        },
        default_max_new_tokens,
    )


def _get_admissible_dtypes():
    """Placeholder docstring"""
    return ["bfloat16"]
