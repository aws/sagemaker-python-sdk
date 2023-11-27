"""Placeholder docstring"""
from __future__ import absolute_import
from pathlib import Path

import logging
from typing import Type

from sagemaker.session import Session
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.model_server.triton.server import SageMakerTritonServer
from sagemaker.serve.model_server.torchserve.server import SageMakerTorchServe
from sagemaker.serve.model_server.djl_serving.server import SageMakerDjlServing
from sagemaker.serve.model_server.tgi.server import SageMakerTgiServing

logger = logging.getLogger(__name__)


class SageMakerEndpointMode(
    SageMakerTorchServe, SageMakerTritonServer, SageMakerDjlServing, SageMakerTgiServing
):
    """Holds the required method to deploy a model to a SageMaker Endpoint"""

    def __init__(self, inference_spec: Type[InferenceSpec], model_server: ModelServer):
        super().__init__()
        # pylint: disable=bad-super-call
        super(SageMakerTritonServer, self).__init__()

        self.inference_spec = inference_spec
        self.model_server = model_server

    def load(self, model_path: str):
        """Placeholder docstring"""
        path = Path(model_path)
        if not path.exists():
            raise Exception("model_path does not exist")
        if not path.is_dir():
            raise Exception("model_path is not a valid directory")

        model_dir = path.joinpath("model")
        return self.inference_spec.model_fn(str(model_dir))

    def prepare(
        self,
        model_path: str,
        secret_key: str,
        s3_model_data_url: str = None,
        sagemaker_session: Session = None,
        image: str = None,
        jumpstart: bool = False,
    ):
        """Placeholder docstring"""
        try:
            sagemaker_session = sagemaker_session or Session()
        except Exception as e:
            raise Exception(
                "Failed to setup default SageMaker session. Please allow a default "
                + "session to be created or supply `sagemaker_session` into @serve.invoke."
            ) from e

        if self.model_server == ModelServer.TORCHSERVE:
            return self._upload_torchserve_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                secret_key=secret_key,
                s3_model_data_url=s3_model_data_url,
                image=image,
            )

        if self.model_server == ModelServer.TRITON:
            return self._upload_triton_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                secret_key=secret_key,
                s3_model_data_url=s3_model_data_url,
                image=image,
            )

        if self.model_server == ModelServer.DJL_SERVING:
            return self._upload_djl_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                s3_model_data_url=s3_model_data_url,
                image=image,
            )

        if self.model_server == ModelServer.TGI:
            return self._upload_tgi_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                s3_model_data_url=s3_model_data_url,
                image=image,
                jumpstart=jumpstart,
            )

        raise ValueError("%s model server is not supported" % self.model_server)
