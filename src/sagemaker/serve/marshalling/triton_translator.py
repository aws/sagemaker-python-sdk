"""Implements class converts data from and to np.ndarray"""

from __future__ import absolute_import

import logging

import numpy as np

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument
class TorchTensorTranslator:
    """Translate torch.Tensor from and to numpy.ndarray"""

    def __init__(self) -> None:
        import torch

        self.convert_from_numpy = torch.from_numpy  # pylint: disable=E1101
        self.CONTENT_TYPE = "tensor/pt"
        self.ACCEPT = "tensor/pt"

    def serialize(self, data, content_type: str = "tensor/pt"):
        """Translate torch.Tensor to numpy ndarray"""
        try:
            return data.detach().numpy()
        except Exception as e:
            logger.error(e)
            raise ValueError("Unable to translate data %s to np.ndarray: %s" % (type(data), e))

    def deserialize(self, data, content_type: str = "application/x-npy"):
        """Translate numpy ndarray to torch.Tensor"""
        try:
            return self.convert_from_numpy(data)
        except Exception as e:
            logger.error(e)
            raise ValueError("Unable to translate data %s to torch.Tensor: %s " % (type(data), e))

    def _deserializer(self):
        """Dummy function to align with DeserializerWrapper in SchemaBuilder"""
        raise ValueError("This method is not meant to be invoked.")


class TensorflowTensorTranslator:
    """Converts tf.Tensor from and to numpy.ndarray"""

    def __init__(self) -> None:
        import tensorflow as tf

        self.convert_to_tensor = tf.convert_to_tensor
        self.CONTENT_TYPE = "tensor/tf"
        self.ACCEPT = "tensor/tf"

    def serialize(self, data, content_type: str = "tensor/tf"):
        """Translate tf.Tensor to numpy ndarray"""
        try:
            return data.numpy()
        except Exception as e:
            logger.error(e)
            raise ValueError("Unable to convert data %s to np.ndarray" % type(data)) from e

    def deserialize(self, data, content_type: str = "application/x-npy"):
        """Translate numpy ndarray to torch.Tensor"""
        try:
            return self.convert_to_tensor(data)
        except Exception as e:
            logger.error(e)
            raise ValueError("Unable to convert data %s to tf.Tensor" % type(data)) from e

    def _deserializer(self):
        """Dummy function to align with DeserializerWrapper in SchemaBuilder"""
        raise ValueError("This method is not meant to be invoked.")


class NumpyTranslator:
    """A dummy class to make sure the translator interface is aligned"""

    def __init__(self) -> None:
        self.CONTENT_TYPE = "application/x-npy"
        self.ACCEPT = "application/x-npy"

    def serialize(self, data, content_type: str = "application/x-npy"):
        """Placeholder docstring"""
        return data

    def deserialize(self, data, content_type: str = "application/x-npy"):
        """Placeholder docstring"""
        return data

    def _deserializer(self):
        """Dummy function to align with DeserializerWrapper in SchemaBuilder"""
        raise ValueError("This method is not meant to be invoked.")


class ListTranslator:
    """Translate python list from and to numpy.ndarray"""

    def __init__(self) -> None:
        self.CONTENT_TYPE = "application/list"
        self.ACCEPT = "application/list"

    def serialize(self, data, content_type: str = "application/list"):
        """Placeholder docstring"""
        try:
            return np.array(data)
        except Exception as e:
            logger.error(e)
            raise ValueError("Unable to convert data %s to np.ndarray" % type(data)) from e

    def deserialize(self, data, content_type: str = "application/x-npy"):
        """Placeholder docstring"""
        try:
            return data.tolist()
        except Exception as e:
            logger.error(e)
            raise ValueError("Unable to convert data %s to python list" % type(data)) from e

    def _deserializer(self):
        """Dummy function to align with DeserializerWrapper in SchemaBuilder"""
        raise ValueError("This method is not meant to be invoked.")
