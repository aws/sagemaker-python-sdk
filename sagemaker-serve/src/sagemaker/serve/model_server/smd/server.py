"""Module for SMD Server"""

from __future__ import absolute_import

import logging
import platform
from sagemaker.core.common_utils import _is_s3_uri
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.s3.utils import determine_bucket_and_prefix, parse_s3_url
from sagemaker.core import fw_utils
from sagemaker.serve.utils.uploader import upload

logger = logging.getLogger(__name__)


class SageMakerSmdServer:
    """Placeholder docstring"""

    def _upload_smd_artifacts(
        self,
        model_path: str,
        sagemaker_session: Session,
        secret_key: str,
        s3_model_data_url: str = None,
        image: str = None,
        should_upload_artifacts: bool = False,
    ):
        """Tar the model artifact and upload to S3 bucket, then prepare for the environment variables"""
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
            "SAGEMAKER_INFERENCE_CODE_DIRECTORY": "/opt/ml/model/code",
            "SAGEMAKER_INFERENCE_CODE": "inference.handler",
            "SAGEMAKER_REGION": sagemaker_session.boto_region_name,
            "SAGEMAKER_SERVE_SECRET_KEY": secret_key,
            "LOCAL_PYTHON": platform.python_version(),
        }
        return s3_upload_path, env_vars
