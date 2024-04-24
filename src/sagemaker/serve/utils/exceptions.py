"""Placeholder Docstring"""

from __future__ import absolute_import


class ModelBuilderException(Exception):
    """The base exception class for ModelBuilder exceptions."""

    fmt = "An unspecified error occurred"

    def __init__(self, **kwargs):
        msg = self.fmt.format(**kwargs)
        Exception.__init__(self, msg)
        self.kwargs = kwargs


class LocalDeepPingException(ModelBuilderException):
    """Raise when local model serving does not pass the deep ping check"""

    fmt = "Error Message: {message}"
    model_builder_error_code = 1

    def __init__(self, message):
        super().__init__(message=message)


class LocalModelOutOfMemoryException(ModelBuilderException):
    """Raise when local model serving fails to load the model"""

    fmt = "Error Message: {message}"
    model_builder_error_code = 2

    def __init__(self, message):
        super().__init__(message=message)


class LocalModelLoadException(ModelBuilderException):
    """Raise when local model serving fails to load the model"""

    fmt = "Error Message: {message}"
    model_builder_error_code = 3

    def __init__(self, message):
        super().__init__(message=message)


class LocalModelInvocationException(ModelBuilderException):
    """Raise when local model serving fails to invoke the model"""

    fmt = "Error Message: {message}"
    model_builder_error_code = 4

    def __init__(self, message):
        super().__init__(message=message)


class SkipTuningComboException(ModelBuilderException):
    """Raise when tuning combination should be admissible but is not"""

    fmt = "Error Message: {message}"

    def __init__(self, message):
        super().__init__(message=message)


class TaskNotFoundException(ModelBuilderException):
    """Raise when HuggingFace task could not be found"""

    fmt = "Error Message: {message}"

    def __init__(self, message):
        super().__init__(message=message)
