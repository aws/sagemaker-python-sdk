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
"""Constants and enums for SageMaker ModelBuilder and serving functionality.

This module defines:
- Framework enum for ML framework identification
- Supported model servers and local modes
- Default serializers and deserializers by framework
- Configuration constants for model serving

Example:
    Using Framework enum::
    
        from sagemaker.serve.constants import Framework, DEFAULT_SERIALIZERS_BY_FRAMEWORK
        
        # Get serializers for PyTorch
        serializer, deserializer = DEFAULT_SERIALIZERS_BY_FRAMEWORK[Framework.PYTORCH]
"""
from __future__ import absolute_import, annotations

# Standard library imports
from enum import Enum
from typing import Dict, Set, Tuple

# SageMaker imports
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.deserializers import (
    CSVDeserializer,
    JSONDeserializer,
    NumpyDeserializer,
    RecordDeserializer,
)
from sagemaker.core.serializers import (
    JSONSerializer,
    LibSVMSerializer,
    NumpySerializer,
    RecordSerializer,
    TorchTensorSerializer,
)


# ========================================
# Mode and Server Constants
# ========================================

LOCAL_MODES: list[Mode] = [Mode.LOCAL_CONTAINER, Mode.IN_PROCESS]

SUPPORTED_MODEL_SERVERS: Set[ModelServer] = {
    ModelServer.TORCHSERVE,
    ModelServer.TRITON,
    ModelServer.DJL_SERVING,
    ModelServer.TENSORFLOW_SERVING,
    ModelServer.MMS,
    ModelServer.TGI,
    ModelServer.TEI,
    ModelServer.SMD,
}

# ========================================
# Framework Enum
# ========================================

class Framework(Enum):
    """Enumeration of supported ML frameworks for ModelBuilder.
    
    This enum provides standardized framework identifiers used throughout
    the ModelBuilder ecosystem for:
    - Framework detection from container images
    - Serializer/deserializer selection
    - Model server compatibility
    
    Example:
        Using framework enum::
        
            if detected_framework == Framework.PYTORCH:
                serializer, deserializer = DEFAULT_SERIALIZERS_BY_FRAMEWORK[Framework.PYTORCH]
    """
    XGBOOST = "XGBoost"
    LDA = "LDA"
    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"
    MXNET = "MXNet"
    CHAINER = "Chainer"
    SKLEARN = "SKLearn"
    HUGGINGFACE = "HuggingFace"
    DJL = "DJL"
    SPARKML = "SparkML"
    NTM = "NTM"
    SMD = "SMD"


# ========================================
# Framework Serialization Mapping
# ========================================

DEFAULT_SERIALIZERS_BY_FRAMEWORK: Dict[Framework, Tuple] = {
    Framework.XGBOOST: (LibSVMSerializer(), CSVDeserializer()),
    Framework.LDA: (RecordSerializer(), RecordDeserializer()),
    Framework.PYTORCH: (TorchTensorSerializer(), JSONDeserializer()),
    Framework.TENSORFLOW: (NumpySerializer(), JSONDeserializer()),
    Framework.MXNET: (RecordSerializer(), JSONDeserializer()),
    Framework.CHAINER: (NumpySerializer(), JSONDeserializer()),
    Framework.SKLEARN: (NumpySerializer(), NumpyDeserializer()),
    Framework.HUGGINGFACE: (JSONSerializer(), JSONDeserializer()),
    Framework.DJL: (JSONSerializer(), JSONDeserializer()),
    Framework.SPARKML: (NumpySerializer(), JSONDeserializer()),
    Framework.NTM: (RecordSerializer(), JSONDeserializer()),
    Framework.SMD: (JSONSerializer(), JSONDeserializer()),
}

