"""Placeholder docstring"""

from __future__ import absolute_import

CONFIG_TEMPLATE = """name: \"model\"
backend: \"python\"

input [
  {{
    name: \"{input_name}\"
    data_type: {input_dtype}
    dims: {input_shape}
  }}
]

output [
  {{
    name: \"{output_name}\"
    data_type: {output_dtype}
    dims: {output_shape}
  }}
]

instance_group [
    {{
    kind: {hardware_type}
    }}
]

parameters: {{
  key: \"EXECUTION_ENV_PATH\",
  value: {{string_value: \"$$TRITON_MODEL_DIRECTORY/triton_env.tar.gz\"}}
}}
"""
