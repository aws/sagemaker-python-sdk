"""Module for Local TGI Serving"""

from __future__ import absolute_import

import requests
import logging
from pathlib import Path
from docker.types import DeviceRequest
from sagemaker import Session, fw_utils
from sagemaker.serve.utils.exceptions import LocalModelInvocationException
from sagemaker.base_predictor import PredictorBase
from sagemaker.s3_utils import determine_bucket_and_prefix, parse_s3_url, s3_path_join
from sagemaker.s3 import S3Uploader
from sagemaker.local.utils import get_docker_host
from sagemaker.serve.utils.optimize_utils import _is_s3_uri

MODE_DIR_BINDING = "/opt/ml/model/"
_SHM_SIZE = "2G"
_DEFAULT_ENV_VARS = {
    "TRANSFORMERS_CACHE": "/opt/ml/model/",
    "HF_HOME": "/opt/ml/model/",
    "HUGGINGFACE_HUB_CACHE": "/opt/ml/model/",
}

logger = logging.getLogger(__name__)


class LocalTgiServing:
    """Placeholder docstring"""

    def _start_tgi_serving(
        self,
        client: object,
        image: str,
        model_path: str,
        secret_key: str,
        env_vars: dict,
        jumpstart: bool,
    ):
        """Placeholder docstring"""
        if jumpstart:
            self.container = client.containers.run(
                image,
                ["--model-id", MODE_DIR_BINDING],
                shm_size=_SHM_SIZE,
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                network_mode="host",
                detach=True,
                auto_remove=True,
                volumes={
                    Path(model_path).joinpath("code"): {
                        "bind": MODE_DIR_BINDING,
                        "mode": "rw",
                    },
                },
                environment=env_vars,
            )
        else:
            self.container = client.containers.run(
                image,
                shm_size=_SHM_SIZE,
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                network_mode="host",
                detach=True,
                auto_remove=True,
                volumes={
                    Path(model_path).joinpath("code"): {
                        "bind": MODE_DIR_BINDING,
                        "mode": "rw",
                    },
                },
                environment=_update_env_vars(env_vars),
            )

    def _invoke_tgi_serving(self, request: object, content_type: str, accept: str):
        """Placeholder docstring"""
        try:
            response = requests.post(
                f"http://{get_docker_host()}:8080/generate",
                data=request,
                headers={"Content-Type": content_type, "Accept": accept},
                timeout=600,
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise Exception("Unable to send request to the local container server") from e

    def _tgi_deep_ping(self, predictor: PredictorBase):
        """Placeholder docstring"""
        response = None
        try:
            response = predictor.predict(self.schema_builder.sample_input)
            return (True, response)
            # pylint: disable=broad-except
        except Exception as e:
            if "422 Client Error: Unprocessable Entity for url" in str(e):
                raise LocalModelInvocationException(str(e))
            return (False, response)

        return (True, response)


class SageMakerTgiServing:
    """Placeholder docstring"""

    def _upload_tgi_artifacts(
        self,
        model_path: str,
        sagemaker_session: Session,
        jumpstart: bool,
        s3_model_data_url: str = None,
        image: str = None,
        env_vars: dict = None,
        should_upload_artifacts: bool = False,
    ):
        model_data_url = None
        if _is_s3_uri(model_path):
            model_data_url = model_path
        elif should_upload_artifacts:
            if s3_model_data_url:
                bucket, key_prefix = parse_s3_url(url=s3_model_data_url)
            else:
                bucket, key_prefix = None, None

            code_key_prefix = fw_utils.model_code_key_prefix(key_prefix, None, image)

            bucket, code_key_prefix = determine_bucket_and_prefix(
                bucket=bucket, key_prefix=code_key_prefix, sagemaker_session=sagemaker_session
            )

            code_dir = Path(model_path).joinpath("code")

            s3_location = s3_path_join("s3://", bucket, code_key_prefix, "code")

            logger.debug("Uploading TGI Model Resources uncompressed to: %s", s3_location)

            model_data_url = S3Uploader.upload(
                str(code_dir),
                s3_location,
                None,
                sagemaker_session,
            )

        model_data = (
            {
                "S3DataSource": {
                    "CompressionType": "None",
                    "S3DataType": "S3Prefix",
                    "S3Uri": model_data_url + "/",
                }
            }
            if model_data_url
            else None
        )
        if jumpstart:
            return (model_data, {})
        return (model_data, _update_env_vars(env_vars))


def _update_env_vars(env_vars: dict) -> dict:
    """Placeholder docstring"""
    updated_env_vars = {}
    updated_env_vars.update(_DEFAULT_ENV_VARS)
    if env_vars:
        updated_env_vars.update(env_vars)
    return updated_env_vars
