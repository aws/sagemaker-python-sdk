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
    TEI = 7


class HardwareType(Enum):
    """An enum for hardware type"""

    def __str__(self) -> str:
        """Placeholder docstring"""
        return str(self.name)

    CPU = 1
    GPU = 2
    INFERENTIA_1 = 3
    INFERENTIA_2 = 4
    GRAVITON = 5


class ImageUriOption(Enum):
    """Enum type for image uri options"""

    def __str__(self) -> str:
        """Convert enum to string"""
        return str(self.name)

    CUSTOM_IMAGE = 1
    CUSTOM_1P_IMAGE = 2
    DEFAULT_IMAGE = 3
