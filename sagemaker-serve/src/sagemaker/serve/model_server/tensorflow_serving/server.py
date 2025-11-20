"""Module for Local Tensorflow Server"""

from __future__ import absolute_import

import requests
import logging
import platform
from pathlib import Path
from sagemaker.core.common_utils import _is_s3_uri
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.s3.utils import determine_bucket_and_prefix, parse_s3_url
from sagemaker.core import fw_utils
from sagemaker.serve.utils.uploader import upload
from sagemaker.core.local.local_session import get_docker_host

logger = logging.getLogger(__name__)


class LocalTensorflowServing:
    """LocalTensorflowServing class."""

    def _start_tensorflow_serving(
        self, client: object, image: str, model_path: str, secret_key: str, env_vars: dict
    ):
        """Starts a local tensorflow serving container.

        Args:
            client: Docker client
            image: Image to use
            model_path: Path to the model
            secret_key: Secret key to use for authentication
            env_vars: Environment variables to set
        """
        self.container = client.containers.run(
            image,
            "serve",
            detach=True,
            auto_remove=False,  # Temporarily disabled to see crash logs
            # network_mode="host",
            ports={'8501/tcp': 8501},
            volumes={
                Path(model_path): {
                    "bind": "/opt/ml/model",
                    "mode": "rw",
                },
            },
            environment={
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SERVE_SECRET_KEY": secret_key,
                "LOCAL_PYTHON": platform.python_version(),
                **env_vars,
            },
        )

    def _invoke_tensorflow_serving(self, request: object, content_type: str, accept: str):
        """Invokes a local tensorflow serving container.

        Args:
            request: Request to send
            content_type: Content type to use
            accept: Accept to use
        """
        try:
            response = requests.post(
                f"http://{get_docker_host()}:8501/invocations",
                data=request,
                headers={"Content-Type": content_type, "Accept": accept},
                timeout=60,  # this is what SageMaker Hosting uses as timeout
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise Exception("Unable to send request to the local container server") from e


class SageMakerTensorflowServing:
    """SageMakerTensorflowServing class."""

    def _upload_tensorflow_serving_artifacts(
        self,
        model_path: str,
        sagemaker_session: Session,
        secret_key: str,
        s3_model_data_url: str = None,
        image: str = None,
        should_upload_artifacts: bool = False,
    ):
        """Uploads the model artifacts to S3.

        Args:
            model_path: Path to the model
            sagemaker_session: SageMaker session
            secret_key: Secret key to use for authentication
            s3_model_data_url: S3 model data URL
            image: Image to use
            model_data_s3_path: S3 model data URI
        """
        s3_upload_path = None
        if _is_s3_uri(model_path):
            s3_upload_path = model_path
        elif should_upload_artifacts:
            if s3_model_data_url:
                bucket, key_prefix = parse_s3_url(url=s3_model_data_url)
            else:
                bucket, key_prefix = None, None

            code_key_prefix = fw_utils.model_code_key_prefix(key_prefix, None, image)

            bucket, code_key_prefix = determine_bucket_and_prefix(
                bucket=bucket, key_prefix=code_key_prefix, sagemaker_session=sagemaker_session
            )

            logger.debug(
                "Uploading the model resources to bucket=%s, key_prefix=%s.",
                bucket,
                code_key_prefix,
            )
            s3_upload_path = upload(sagemaker_session, model_path, bucket, code_key_prefix)
            logger.debug("Model resources uploaded to: %s", s3_upload_path)

        env_vars = {
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_REGION": sagemaker_session.boto_region_name,
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "10",
            "SAGEMAKER_SERVE_SECRET_KEY": secret_key,
            "LOCAL_PYTHON": platform.python_version(),
        }
        return s3_upload_path, env_vars
