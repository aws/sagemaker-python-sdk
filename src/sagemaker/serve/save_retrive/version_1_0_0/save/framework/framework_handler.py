"""Experimental"""

from __future__ import absolute_import
from abc import ABC, abstractmethod
from typing import Type
from sagemaker import Session
from sagemaker.model import Model


class FrameworkHandler(ABC):
    """Abstract class for framework handler"""

    def __init__(
        self,
        version: str,
        py_version: str,
        framework: str,
        framework_version: str,
    ) -> None:
        self.version = version
        self.py_version = py_version
        self.framework = framework
        self.framework_version = framework_version
        super().__init__()

    @abstractmethod
    def save_metadata(self) -> None:
        """Placeholder docstring"""

    @abstractmethod
    def save_model(self) -> None:
        """Placeholder docstring"""

    @abstractmethod
    def get_pysdk_model(
        self, s3_path: str, role_arn: str, sagemaker_session: Session
    ) -> Type[Model]:
        """Placeholder docstring"""
