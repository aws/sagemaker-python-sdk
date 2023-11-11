"""Types used for SageMaker ModelBuilder"""
from __future__ import absolute_import

from enum import Enum


class ModelServer(Enum):
    """An enum for model server"""

    def __str__(self):
        """Placeholder docstring"""
        return str(self.name)

    TORCHSERVE = 1
    MMS = 2
    TENSORFLOW_SERVING = 3
    DJL_SERVING = 4
    TRITON = 5
    TGI = 6


class _DjlEngine(Enum):
    """An enum for Djl Engines"""

    def __str__(self):
        """Placeholder docstring"""
        return str(self.name)

    DEEPSPEED = 1
    FASTER_TRANSFORMER = 2
    HUGGINGFACE_ACCELERATE = 3


class HardwareType(Enum):
    """An enum for hardware type"""

    def __str__(self) -> str:
        """Placeholder docstring"""
        return str(self.name)

    CPU = 1
    GPU = 2
