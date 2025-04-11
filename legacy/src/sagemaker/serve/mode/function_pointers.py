"""Placeholder docstring"""

from __future__ import absolute_import
from enum import Enum

PKL_FILE_NAME = "serve.pkl"


class Mode(Enum):
    """Placeholder docstring"""

    def __str__(self):
        """Placeholder docstring"""
        return str(self.name)

    IN_PROCESS = 1
    LOCAL_CONTAINER = 2
    SAGEMAKER_ENDPOINT = 3
