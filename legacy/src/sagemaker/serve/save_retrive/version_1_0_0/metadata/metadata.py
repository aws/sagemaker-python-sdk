"""Experimental"""

from __future__ import absolute_import
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import platform
import yaml

DEFAULT_VERION = "1.0.0"


@dataclass
class Metadata:
    """Placeholder docstring"""

    version: Optional[str] = field(
        default=DEFAULT_VERION,
        metadata={"help": "define the save version"},
    )
    python_version: Optional[str] = field(
        default=platform.python_version(),
        metadata={"help": "define the python version"},
    )
    model_type: Optional[str] = field(
        default="unknown",
        metadata={
            "help": "Define the type of the model. "
            "XGBoostModel, PyTorchModel, HuggingFaceModel, etc. "
            "correlates to ModelType in PySDK"
        },
    )
    framework: Optional[str] = field(
        default="xgboost",
        metadata={"help": "Define the framework of the model"},
    )
    framework_version: Optional[str] = field(
        default="unknown",
        metadata={"help": "Define the framework version of the model"},
    )


@dataclass
class XGBoostMetadata(Metadata):
    """Placeholder docstring"""

    model_file: Optional[str] = field(
        default="",
        metadata={"help": "Define the path of model directory"},
    )
    requirements_file: Optional[str] = field(
        default="",
        metadata={"help": "Defines the relative path of requirements.txt file"},
    )
    schema_file: Optional[str] = field(
        default="",
        metadata={"help": "Defines the relative path of input schema file"},
    )
    inference_spec_file: Optional[str] = field(
        default="",
        metadata={"help": "Defines the relative path of inference spec file"},
    )
    model_hmac: Optional[str] = field(
        default="unknown",
        metadata={"help": "Define the hmac version of the model"},
    )
    schema_hmac: Optional[str] = field(
        default="unknown",
        metadata={"help": "Define the hmac version of the schema"},
    )
    inference_spec_hmac: Optional[str] = field(
        default="unknown",
        metadata={"help": "Define the hmac version of the inference spec"},
    )
    task: Optional[str] = field(
        default="unknown",
        metadata={
            "help": "Define the task of the model. classification, regression, clustering, "
            "text-generation, text-classification, etc."
        },
    )
    optimizer_metadata: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Define the optimizer metadata of the model. "
            "model-size, model-size-in-memory, model-accuracy, model-latency, etc."
        },
    )

    def to_dict(self):
        """Placeholder docstring"""

        if self.optimizer_metadata is None:
            self.optimizer_metadata = {}

        base = {
            "Version": self.version,
            "PythonVersion": self.python_version,
            "Framework": self.framework,
            "FrameworkVersion": self.framework_version,
            "Model": self.model_file,
            "ModelHMAC": self.model_hmac,
            "Requirements": self.requirements_file,
            "Schema": self.schema_file,
            "SchemaHMAC": self.schema_hmac,
            "Task": self.task,
            "ModelType": self.model_type,
            "OptimizerInputs": dict(self.optimizer_metadata),
        }

        if self.inference_spec_file:
            base["InferenceSpec"] = self.inference_spec_file
            base["InferenceSpecHMAC"] = self.inference_spec_hmac

        return base


@dataclass
class PyTorchMetadata(XGBoostMetadata):
    """Placeholder docstring"""


def get_metadata(model_dir: str) -> Metadata:
    """Get metadata from the model directory and validate the integrity of the model

    Args:
        model_dir (str) : Argument

    Returns:
        ( Metadata ) :

    """
    metadata = Metadata()
    metadata_file = Path(model_dir).joinpath("metadata.yaml")
    if metadata_file.exists():
        with open(str(metadata_file), "r") as f:
            metadata = Metadata(**yaml.load(f))  # pylint: disable=E1120
        if metadata.model_type == "XGBoostModel":
            metadata = XGBoostMetadata(**yaml.load(f))  # pylint: disable=E1120
        elif metadata.model_type == "PyTorchModel":
            metadata = PyTorchMetadata(**yaml.load(f))  # pylint: disable=E1120
        return metadata
    return None
