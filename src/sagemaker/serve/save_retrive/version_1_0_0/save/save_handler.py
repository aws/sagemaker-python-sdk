"""Experimental"""

from __future__ import absolute_import
from typing import Optional, Type, Any
from dataclasses import dataclass, field
import datetime
import logging
from pathlib import Path
from sagemaker.serve.save_retrive.version_1_0_0.save.utils import (
    upload_to_s3,
    capture_dependencies,
    capture_optimization_metadata,
    detect_framework_and_its_versions,
)

from sagemaker import Session
from sagemaker.model import Model
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.save_retrive.version_1_0_0.metadata.metadata import Metadata
from sagemaker.serve.save_retrive.version_1_0_0.save.framework.pytorch_handler import PyTorchHandler
from sagemaker.serve.save_retrive.version_1_0_0.save.framework.xgboost_handler import XGBoostHandler

logger = logging.getLogger(__name__)
VERION = "1.0.0"


@dataclass
class SaveHandler:
    """Placeholder docstring"""

    save_path: Optional[str] = field(
        default="",
        metadata={"help": "Define the path of save directory"},
    )
    model_path: Optional[str] = field(
        default="",
        metadata={"help": "Define the path of model directory"},
    )
    model_name: Optional[str] = field(
        default="model",
        metadata={"help": "Define the name of the model"},
    )
    model_format: Optional[str] = field(
        default="pkl",
        metadata={"help": "Define the format of the model"},
    )
    inference_spec: Optional[InferenceSpec] = field(
        default=None,
        metadata={"help": "Define the inference spec of the model"},
    )
    inference_spec_path: Optional[str] = field(
        default="",
        metadata={"help": "Define the path of inference spec file"},
    )
    inference_spec_name: Optional[str] = field(
        default="spec",
        metadata={"help": "Define the name of the inference spec file"},
    )
    inference_spec_format: Optional[str] = field(
        default="pkl",
        metadata={"help": "Define the format of the inference spec file"},
    )
    requirements_path: Optional[str] = field(
        default="",
        metadata={"help": "Define the path of dependencies directory"},
    )
    schema_path: Optional[str] = field(
        default="",
        metadata={"help": "Define the path of input schema file"},
    )
    schema_format: Optional[str] = field(
        default="pkl",
        metadata={"help": "Define the format of input schema file"},
    )
    metadata_path: Optional[str] = field(
        default="",
    )
    metadata: Optional[Metadata] = field(
        default=None,
        metadata={"help": "define the metadata of the storage"},
    )
    framework: Optional[str] = field(
        default="unknown",
        metadata={"help": "Define the framework of the model"},
    )
    framework_version: Optional[str] = field(
        default="unknown",
        metadata={"help": "Define the framework version of the model"},
    )
    task: Optional[str] = field(
        default="unknown",
        metadata={
            "help": "Define the task of the model. "
            "classification, regression, clustering, "
            "text-generation, text-classification, etc."
        },
    )
    model_type: Optional[str] = field(
        default="unknown",
        metadata={
            "help": "Define the type of the model "
            "pytorch, xgboost, huggingface_transformers, "
            "etc. correlates to Model in PySDK"
        },
    )
    optimizer_metadata: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": "Define the optimizer metadata of the model "
            "model-size, model-size-in-memory, "
            "model-accuracy, model-latency, etc."
        },
    )
    model_hmac: Optional[str] = field(
        default="unknown",
        metadata={"help": "Define the hmac version of the model"},
    )
    schema_hmac: Optional[str] = field(
        default="unknown",
        metadata={"help": "Define the hmac version of the schema"},
    )

    def __init__(
        self,
        model: Optional[Any],
        schema_builder: SchemaBuilder,
        model_loader_path: Optional[str] = None,
        inference_spec: Optional[InferenceSpec] = None,
        save_path: Optional[str] = None,
        s3_path: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        role_arn: Optional[str] = None,
        metadata: Optional[Metadata] = None,
    ):
        self.model = model
        self.schema_builder = schema_builder
        self.model_loader_path = model_loader_path
        self.inference_spec = inference_spec
        self.s3_path = s3_path
        self._update_save_path(save_path)
        self.sagemaker_session = sagemaker_session
        self.py_version = None
        self.role_arn = role_arn
        self.metadata = metadata

    def _update_save_path(self, save_path=None):
        """Placeholder docstring"""
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = "/tmp/sagemaker/save/" + datetime.datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S"
            )

        logger.info("Save path: %s", self.save_path)
        self.model_path = f"{self.save_path}/{self.model_name}"
        self.requirements_path = f"{self.save_path}/requirements.txt"
        self.schema_path = f"{self.save_path}/schema"
        self.metadata_path = f"{self.save_path}/metadata.yaml"
        self.inference_spec_path = f"{self.save_path}/{self.inference_spec_name}"

    def save(self) -> Type[Model]:
        """Save the model and the metadata"""
        logger.info("Saving model to %s", self.save_path)

        if not Path(self.save_path).exists():
            Path(self.save_path).mkdir(parents=True, exist_ok=True)

        inferred = detect_framework_and_its_versions(
            self.model if self.model else self.inference_spec.load(self.model_loader_path)
        )
        self.framework = inferred[0][0]
        self.framework_version = inferred[0][1]
        self.py_version = inferred[1]

        capture_dependencies(self.requirements_path)
        self.optimizer_metadata = capture_optimization_metadata(self.model, self.framework)

        handler = None
        if self.framework == "pytorch":
            handler = PyTorchHandler(
                VERION,
                self.py_version,
                self.framework,
                self.framework_version,
                self.model,
                self.model_path,
                self.requirements_path,
                self.schema_builder,
                self.schema_path,
                self.schema_format,
                self.task,
                self.optimizer_metadata,
                self.inference_spec,
                self.inference_spec_path,
                self.inference_spec_format,
                self.metadata_path,
            )
        elif self.framework == "xgboost":
            handler = XGBoostHandler(
                VERION,
                self.py_version,
                self.framework,
                self.framework_version,
                self.model,
                self.model_path,
                self.requirements_path,
                self.schema_builder,
                self.schema_path,
                self.schema_format,
                self.task,
                self.optimizer_metadata,
                self.inference_spec,
                self.inference_spec_path,
                self.inference_spec_format,
                self.metadata_path,
            )
        else:
            raise ValueError("Unknown framework type {}".format(self.framework))

        # save model and the metadata
        handler.save_model()
        handler.save_metadata()

        # upload to s3
        s3_model_url = upload_to_s3(self.s3_path, self.save_path, self.sagemaker_session)

        return handler.get_pysdk_model(s3_model_url, self.role_arn, self.sagemaker_session)
