"""Placeholder docerting"""

from __future__ import absolute_import
import uuid
import logging
import importlib
import platform

from sagemaker import fw_utils
from sagemaker import Session
from sagemaker.base_predictor import PredictorBase
from sagemaker.serve.utils.uploader import upload
from sagemaker.serve.utils.exceptions import LocalModelInvocationException
from sagemaker.s3_utils import determine_bucket_and_prefix, parse_s3_url
from sagemaker.local.utils import get_docker_host
import docker
from docker.types import DeviceRequest

logger = logging.getLogger(__name__)

# TODO: automatically update memory size
_SHM_SIZE = "2G"


class LocalTritonServer:
    """Placeholder docstring"""

    def __init__(self) -> None:
        self.triton_client = None

    def _start_triton_server(
        self,
        docker_client: docker.DockerClient,
        model_path: str,
        secret_key: str,
        image_uri: str,
        env_vars: dict,
    ):
        """Placeholder docstring"""
        self.container_name = "triton" + uuid.uuid1().hex
        model_repository = model_path + "/model_repository"
        env_vars.update(
            {
                "TRITON_MODEL_DIR": "/models/model",
                "SAGEMAKER_SERVE_SECRET_KEY": secret_key,
                "LOCAL_PYTHON": platform.python_version(),
            }
        )

        if "cpu" not in image_uri:
            self.container = docker_client.containers.run(
                image=image_uri,
                command=["tritonserver", "--model-repository=/models"],
                shm_size=_SHM_SIZE,
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                network_mode="host",
                detach=True,
                auto_remove=True,
                volumes={model_repository: {"bind": "/models", "mode": "rw"}},
                environment=env_vars,
            )

        else:
            self.container = docker_client.containers.run(
                image=image_uri,
                command=["tritonserver", "--model-repository=/models"],
                shm_size=_SHM_SIZE,
                network_mode="host",
                detach=True,
                auto_remove=True,
                volumes={model_repository: {"bind": "/models", "mode": "rw"}},
                environment=env_vars,
            )

    def _invoke_triton_server(self, payload, *args, **kwargs):
        """Placeholder docstring"""
        httpClient = importlib.import_module("tritonclient.http")

        if not self.triton_client:
            self.triton_client = httpClient.InferenceServerClient(url=f"{get_docker_host()}:8000")

        payload = self.schema_builder.input_serializer.serialize(payload)
        dtype = self.schema_builder._input_triton_dtype.split("_")[-1]

        input_request = httpClient.InferInput("input_1", payload.shape, datatype=dtype)
        input_request.set_data_from_numpy(payload, binary_data=True)

        response = self.triton_client.infer(model_name="model", inputs=[input_request])
        response_name = response.get_response().get("outputs")[0].get("name")

        return self.schema_builder.output_deserializer.deserialize(response.as_numpy(response_name))

    def _triton_deep_ping(self, predictor: PredictorBase) -> bool:
        # TODO: set datatype from payload
        try:
            response = predictor.predict(self.schema_builder.sample_input)
        except Exception as e:
            if "422 Client Error: Unprocessable Entity for url" in str(e):
                raise LocalModelInvocationException(str(e))
            return (False, None)

        return (True, response)


class SageMakerTritonServer:
    """Placeholder docstring"""

    def __init__(self) -> None:
        pass

    def _upload_triton_artifacts(
        self,
        model_path: str,
        sagemaker_session: Session,
        secret_key: str,
        s3_model_data_url: str = None,
        image: str = None,
    ):
        """Tar triton artifacts and upload to s3"""
        if s3_model_data_url:
            bucket, key_prefix = parse_s3_url(url=s3_model_data_url)
        else:
            bucket, key_prefix = None, None

        code_key_prefix = fw_utils.model_code_key_prefix(key_prefix, None, image)

        bucket, code_key_prefix = determine_bucket_and_prefix(
            bucket=bucket, key_prefix=code_key_prefix, sagemaker_session=sagemaker_session
        )

        logger.debug(
            "Uploading the model resources to bucket=%s, key_prefix=%s.", bucket, code_key_prefix
        )
        model_repository = model_path + "/model_repository"
        s3_upload_path = upload(sagemaker_session, model_repository, bucket, code_key_prefix)
        logger.debug("Model resources uploaded to: %s", s3_upload_path)

        env_vars = {
            "SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "model",
            "TRITON_MODEL_DIR": "/opt/ml/model/model",
            "SAGEMAKER_SERVE_SECRET_KEY": secret_key,
            "LOCAL_PYTHON": platform.python_version(),
        }
        return s3_upload_path, env_vars
