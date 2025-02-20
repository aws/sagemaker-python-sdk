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
"""Placeholder docstring"""
from __future__ import absolute_import

# these imports ensure backward compatibility.
from sagemaker.deserializers import RecordDeserializer  # noqa: F401 # pylint: disable=W0611
from sagemaker.serializers import RecordSerializer  # noqa: F401 # pylint: disable=W0611
from sagemaker.serializer_utils import (  # noqa: F401 # pylint: disable=W0611
    read_recordio,
    read_records,
    write_numpy_to_dense_tensor,
    write_spmatrix_to_sparse_tensor,
    _write_recordio,
)
