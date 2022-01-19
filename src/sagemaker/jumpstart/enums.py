from enum import Enum


class ModelFramework(str, Enum):
    """Enum class for JumpStart model framework.

    The ML framework as referenced in the prefix of the model ID.
    This value does not necessarily correspond to the container name.
    """

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    MXNET = "mxnet"
    HUGGINGFACE = "huggingface"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"


class VariableScope(str, Enum):
    """Possible value of the ``scope`` attribute for a hyperparameter or environment variable.

    Used for hosting environment variables and training hyperparameters.
    """

    CONTAINER = "container"
    ALGORITHM = "algorithm"


class HyperparameterValidationMode(str, Enum):
    """Possible modes for validating hyperparameters."""

    VALIDATE_PROVIDED = "validate_provided"
    VALIDATE_ALGORITHM = "validate_algorithm"
    VALIDATE_ALL = "validate_all"


class VariableTypes(str, Enum):
    """Possible types for hyperparameters and environment variables."""

    TEXT = "text"
    INT = "int"
    FLOAT = "float"
