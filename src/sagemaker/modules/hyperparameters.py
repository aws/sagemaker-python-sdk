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
"""Hyperparameters class module."""
from __future__ import absolute_import

import os
import json
import dataclasses
from typing import Any, Type, TypeVar

from sagemaker.modules import logger

T = TypeVar("T")


class DictConfig:
    """Class that supports both dict and dot notation access"""

    def __init__(self, **kwargs):
        # Store the original dict
        self._data = kwargs

        # Set all items as attributes for dot notation
        for key, value in kwargs.items():
            # Recursively convert nested dicts to DictConfig
            if isinstance(value, dict):
                value = DictConfig(**value)
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access: config['key']"""
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        """Enable dictionary-style assignment: config['key'] = value"""
        self._data[key] = value
        setattr(self, key, value)

    def __str__(self) -> str:
        """String representation"""
        return str(self._data)

    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"DictConfig({self._data})"


class Hyperparameters:
    """Class to load hyperparameters in training container."""

    @staticmethod
    def load() -> DictConfig:
        """Loads hyperparameters in training container

        Example:

        .. code:: python
            from sagemaker.modules.hyperparameters import Hyperparameters

            hps = Hyperparameters.load()
            print(hps.batch_size)

        Returns:
            DictConfig: hyperparameters as a DictConfig object
        """
        hps = json.loads(os.environ.get("SM_HPS", "{}"))
        if not hps:
            logger.warning("No hyperparameters found in SM_HPS environment variable.")
        return DictConfig(**hps)

    @staticmethod
    def load_structured(dataclass_type: Type[T]) -> T:
        """Loads hyperparameters as a structured dataclass

        Example:

        .. code:: python
            from sagemaker.modules.hyperparameters import Hyperparameters

            @dataclass
            class TrainingConfig:
                batch_size: int
                learning_rate: float

            config = Hyperparameters.load_structured(TrainingConfig)
            print(config.batch_size) # typed int

        Args:
            dataclass_type: Dataclass type to structure the config

        Returns:
            dataclass_type: Instance of provided dataclass type
        """

        if not dataclasses.is_dataclass(dataclass_type):
            raise ValueError(f"{dataclass_type} is not a dataclass type.")

        hps = json.loads(os.environ.get("SM_HPS", "{}"))
        if not hps:
            logger.warning("No hyperparameters found in SM_HPS environment variable.")

        # Convert hyperparameters to dataclass
        return dataclass_type(**hps)
