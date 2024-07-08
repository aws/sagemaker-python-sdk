"""Implements class that holds custom load and invoke function of a model"""

from __future__ import absolute_import
import abc


class InferenceSpec(abc.ABC):
    """Abstract base class for holding custom ``load``, ``invoke`` and ``prepare`` functions.

    Provides a skeleton for customization to override the methods
    ``load``, ``invoke`` and ``prepare``.
    """

    @abc.abstractmethod
    def load(self, model_dir: str):
        """Loads the model stored in model_dir and return the model object.

        Args:
            model_dir (str): Path to the directory where the model is stored.
        """

    @abc.abstractmethod
    def invoke(self, input_object: object, model: object):
        """Given model object and input, make inference and return the result.

        Args:
            input_object (object): The input to model
            model (object): The model object
        """

    def prepare(self, *args, **kwargs):
        """Custom prepare function"""

    def get_model(self):
        """Return HuggingFace model name for inference spec"""
