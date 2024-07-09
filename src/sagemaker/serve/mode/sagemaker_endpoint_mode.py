"""Placeholder docstring"""

from __future__ import absolute_import
from pathlib import Path

import logging
from typing import Type

from sagemaker.serve.model_server.tei.server import SageMakerTeiServing
from sagemaker.serve.model_server.tensorflow_serving.server import SageMakerTensorflowServing
from sagemaker.session import Session
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.model_server.triton.server import SageMakerTritonServer
from sagemaker.serve.model_server.torchserve.server import SageMakerTorchServe
from sagemaker.serve.model_server.djl_serving.server import SageMakerDjlServing
from sagemaker.serve.model_server.tgi.server import SageMakerTgiServing
from sagemaker.serve.model_server.multi_model_server.server import SageMakerMultiModelServer

logger = logging.getLogger(__name__)


class SageMakerEndpointMode(
    SageMakerTorchServe,
    SageMakerTritonServer,
    SageMakerDjlServing,
    SageMakerTgiServing,
    SageMakerMultiModelServer,
    SageMakerTensorflowServing,
):
    """Holds the required method to deploy a model to a SageMaker Endpoint"""

    def __init__(self, inference_spec: Type[InferenceSpec], model_server: ModelServer):
        super().__init__()
        # pylint: disable=bad-super-call
        super(SageMakerTritonServer, self).__init__()

        self.inference_spec = inference_spec
        self.model_server = model_server

        self._tei_serving = SageMakerTeiServing()

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
        should_upload_artifacts: bool = False,
    ):
        """Placeholder docstring"""
        try:
            sagemaker_session = sagemaker_session or Session()
        except Exception as e:
            raise Exception(
                "Failed to setup default SageMaker session. Please allow a default "
                + "session to be created or supply `sagemaker_session` into @serve.invoke."
            ) from e

        upload_artifacts = None, None
        if self.model_server == ModelServer.TORCHSERVE:
            upload_artifacts = self._upload_torchserve_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                secret_key=secret_key,
                s3_model_data_url=s3_model_data_url,
                image=image,
                should_upload_artifacts=True,
            )

        if self.model_server == ModelServer.TRITON:
            upload_artifacts = self._upload_triton_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                secret_key=secret_key,
                s3_model_data_url=s3_model_data_url,
                image=image,
                should_upload_artifacts=True,
            )

        if self.model_server == ModelServer.DJL_SERVING:
            upload_artifacts = self._upload_djl_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                s3_model_data_url=s3_model_data_url,
                image=image,
                should_upload_artifacts=True,
            )

        if self.model_server == ModelServer.TENSORFLOW_SERVING:
            upload_artifacts = self._upload_tensorflow_serving_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                secret_key=secret_key,
                s3_model_data_url=s3_model_data_url,
                image=image,
                should_upload_artifacts=True,
            )

        # By default, we do not want to upload artifacts in S3 for the below server.
        # In Case of Optimization, artifacts need to be uploaded into s3.
        # In that case, `should_upload_artifacts` arg needs to come from
        # the caller of prepare.

        if self.model_server == ModelServer.TGI:
            upload_artifacts = self._upload_tgi_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                s3_model_data_url=s3_model_data_url,
                image=image,
                jumpstart=jumpstart,
                should_upload_artifacts=should_upload_artifacts,
            )

        if self.model_server == ModelServer.MMS:
            upload_artifacts = self._upload_server_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                s3_model_data_url=s3_model_data_url,
                secret_key=secret_key,
                image=image,
                should_upload_artifacts=should_upload_artifacts,
            )

        if self.model_server == ModelServer.TEI:
            upload_artifacts = self._tei_serving._upload_tei_artifacts(
                model_path=model_path,
                sagemaker_session=sagemaker_session,
                s3_model_data_url=s3_model_data_url,
                image=image,
                should_upload_artifacts=should_upload_artifacts,
            )

        if upload_artifacts or isinstance(self.model_server, ModelServer):
            return upload_artifacts

        raise ValueError("%s model server is not supported" % self.model_server)
