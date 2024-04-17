"""Experimental"""

from __future__ import absolute_import
import platform
import logging
from pathlib import Path
from typing import Type
from sagemaker.serve.save_retrive.version_1_0_0.save.utils import (
    generate_secret_key,
    compute_hash,
    save_pkl,
    save_yaml,
    capture_schema,
    get_image_url,
)
from sagemaker import Session
from sagemaker.predictor import Predictor
from sagemaker.serve.save_retrive.version_1_0_0.metadata.metadata import XGBoostMetadata
from sagemaker.model import Model
from sagemaker.serve.save_retrive.version_1_0_0.save.framework.framework_handler import (
    FrameworkHandler,
)
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec

logger = logging.getLogger(__name__)


# pylint: disable=attribute-defined-outside-init
class XGBoostHandler(FrameworkHandler):
    """Experimental"""

    def __init__(
        self,
        version: str,
        py_version: str,
        framework: str,
        framework_version: str,
        model: object,
        model_path: str,
        requirements_path: str,
        schema: SchemaBuilder,
        schema_path: str,
        schema_format: str,
        task: str,
        optimizer_metadata: dict,
        inference_spec: InferenceSpec,
        inference_spec_path: str,
        inference_spec_format: str,
        metadata_path: str,
    ) -> None:
        super().__init__(
            version,
            py_version,
            framework,
            framework_version,
        )
        self.model_type = "XGBoostModel"
        self.task = task
        self.model = model
        self.model_path = model_path
        self.model_format = "pkl"
        self.requirements_path = requirements_path
        self.schema = schema
        self.schema_path = schema_path
        self.schema_format = schema_format
        self.inference_spec = inference_spec
        self.inference_spec_path = inference_spec_path
        self.inference_spec_format = inference_spec_format
        self.optimizer_metadata = optimizer_metadata
        self.metadata_path = metadata_path
        self.model_hmac = None
        self.schema_hmac = None
        self.inference_spec_hmac = None

    def save_metadata(self) -> None:
        """Placeholder docstring"""
        logging.info("Getting xgboost metadata...")
        xgboost_metadata = XGBoostMetadata()
        xgboost_metadata.model_type = self.model_type
        xgboost_metadata.task = self.task
        xgboost_metadata.version = self.version
        xgboost_metadata.framework = self.framework
        xgboost_metadata.framework_version = self.framework_version
        xgboost_metadata.model_file = f"{self.model_path.split('/')[-1]}.{self.model_format}"
        xgboost_metadata.requirements_file = self.requirements_path.split("/")[-1]
        xgboost_metadata.schema_file = f"{self.schema_path.split('/')[-1]}.{self.schema_format}"
        xgboost_metadata.schema_hmac = self.schema_hmac

        if self.model:
            xgboost_metadata.model_hmac = self.model_hmac
            xgboost_metadata.inference_spec_file = None
            xgboost_metadata.inference_spec_hmac = None
        elif self.inference_spec:
            xgboost_metadata.inference_spec_file = (
                f"{self.inference_spec_path.split('/')[-1]}.{self.inference_spec_format}"
            )
            xgboost_metadata.inference_spec_hmac = self.inference_spec_hmac
            xgboost_metadata.model_file = None
            xgboost_metadata.model_hmac = None

        xgboost_metadata.optimizer_metadata = self.optimizer_metadata
        save_yaml(self.metadata_path, xgboost_metadata.to_dict())
        logging.info("XGBoost metadata captured successfully")

    def save_model(self) -> None:
        """Now we are using Pickle to save the model, we need to convert it to torchscript"""
        logging.info("Saving xgboost model...")
        if self.model:
            self.model_format = "xgb"
            path = Path(f"{self.model_path}.{self.model_format}")
            self.model.save_model(path.resolve().as_posix())

        self.schema_path, self.schema_format = capture_schema(
            self.schema_path, self.schema_format, self.schema
        )

        if self.inference_spec:
            path = Path(f"{self.inference_spec_path}.{self.inference_spec_format}")
            save_pkl(path, self.inference_spec)

        self._xgboost_generate_hmac()
        logging.info("XGBoost model saved successfully")

    def _xgboost_generate_hmac(self) -> None:
        """Placeholder docstring"""
        logger.info("Generating XGBoost model hmac...")
        self.secret_key = generate_secret_key()

        if self.model:
            with open(Path(f"{self.model_path}.{self.model_format}").absolute(), "rb") as f:
                buffer = f.read()
                self.model_hmac = compute_hash(buffer=buffer, secret_key=self.secret_key)

        with open(Path(f"{self.schema_path}.{self.schema_format}").absolute(), "rb") as f:
            buffer = f.read()
            self.schema_hmac = compute_hash(buffer=buffer, secret_key=self.secret_key)

        if self.inference_spec:
            with open(
                Path(f"{self.inference_spec_path}.{self.inference_spec_format}").absolute(), "rb"
            ) as f:
                buffer = f.read()
                self.inference_spec_hmac = compute_hash(buffer=buffer, secret_key=self.secret_key)

        logger.info("XGBoost model hmac generated successfully")

    def get_pysdk_model(
        self, s3_path: str, role_arn: str, sagemaker_session: Session
    ) -> Type[Model]:
        """Create a XGBoostModel object from the saved model"""
        self.env_vars = {
            "SAGEMAKER_REGION": sagemaker_session.boto_region_name,
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "10",
            "SAGEMAKER_SERVE_SECRET_KEY": self.secret_key,
            "LOCAL_PYTHON": platform.python_version(),
        }

        self.pysdk_model = Model(
            model_data=s3_path,
            image_uri=get_image_url("DJLServing"),
            role=role_arn,
            env=self.env_vars,
            sagemaker_session=sagemaker_session,
            predictor_cls=self._get_predictor,
        )

        return self.pysdk_model

    def _get_client_translators(self):
        """Placeholder docstring"""
        serializer = None
        if self.schema and hasattr(self.schema, "custom_input_translator"):
            serializer = self.schema.custom_input_translator
        elif self.schema:
            serializer = self.schema.input_serializer
        else:
            raise Exception("Cannot serialize")

        deserializer = None
        if self.schema and hasattr(self.schema, "custom_output_translator"):
            deserializer = self.schema.custom_output_translator
        elif self.schema:
            deserializer = self.schema.output_deserializer
        else:
            raise Exception("Cannot deserialize")

        return serializer, deserializer

    def _get_predictor(self, endpoint_name: str, sagemaker_session: Session) -> Predictor:
        """Placeholder docstring"""
        serializer, deserializer = self._get_client_translators()

        return Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )
