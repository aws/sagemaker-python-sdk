# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Placeholder docstring"""
from __future__ import absolute_import

import logging
from typing import Optional, Dict, Any

from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.session import Session

from sagemaker.djl_inference.djl_predictor import DJLPredictor

logger = logging.getLogger(__name__)


def _set_env_var_from_property(
    property_value: Optional[Any], env_key: str, env: dict, override_env_var=False
) -> dict:
    """Utility method to set an environment variable configuration"""
    if not property_value:
        return env
    if override_env_var or env_key not in env:
        env[env_key] = str(property_value)
    return env


class DJLModel(Model):
    """A DJL SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        engine: Optional[str] = None,
        djl_version: str = "0.28.0",
        djl_framework: Optional[str] = None,
        task: Optional[str] = None,
        dtype: Optional[str] = None,
        tensor_parallel_degree: Optional[int] = None,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        job_queue_size: Optional[int] = None,
        parallel_loading: bool = False,
        model_loading_timeout: Optional[int] = None,
        prediction_timeout: Optional[int] = None,
        predictor_cls: callable = DJLPredictor,
        huggingface_hub_token: Optional[str] = None,
        **kwargs,
    ):
        """Initialize a SageMaker model using one of the DJL Model Serving Containers.

        Args:
            model_id (str): This is either the HuggingFace Hub model_id, or the Amazon S3 location
                containing the uncompressed model artifacts (i.e. not a tar.gz file).
                The model artifacts are expected to be in HuggingFace pre-trained model
                format (i.e. model should be loadable from the huggingface transformers
                from_pretrained api, and should also include tokenizer configs if applicable).
                model artifact location must be specified using either the model_id parameter,
                model_data parameter, or HF_MODEL_ID environment variable in the env parameter
            engine (str): The DJL inference engine to use for your model. Defaults to None.
                If not provided, the engine is inferred based on the task. If no task is provided,
                the Python engine is used.
            djl_version (str): DJL Serving version you want to use for serving your model for
                inference. Defaults to None. If not provided, the latest available version of DJL
                Serving is used. This is not used if ``image_uri`` is provided.
            djl_framework (str): The DJL container to use. This is used along with djl_version
                to fetch the image_uri of the djl inference container. This is not used if
                ``image_uri`` is provided.
            task (str): The HuggingFace/NLP task you want to launch this model for. Defaults to
                None.
                If not provided, the task will be inferred from the model architecture by DJL.
            tensor_parallel_degree (int): The number of accelerators to partition the model across
                using tensor parallelism. Defaults to None. If not provided, the maximum number
                of available accelerators will be used.
            min_workers (int): The minimum number of worker processes. Defaults to None. If not
                provided, dJL Serving will automatically detect the minimum workers.
            max_workers (int): The maximum number of worker processes. Defaults to None. If not
                provided, DJL Serving will automatically detect the maximum workers.
            job_queue_size (int): The request job queue size. Defaults to None. If not specified,
                defaults to 1000.
            parallel_loading (bool): Whether to load model workers in parallel. Defaults to False,
                in which case DJL Serving will load the model workers sequentially to reduce the
                risk of running out of memory. Set to True if you want to reduce model loading
                time and know that peak memory usage will not cause out of memory issues.
            model_loading_timeout (int): The worker model loading timeout in seconds. Defaults to
                None. If not provided, the default is 240 seconds.
            prediction_timeout (int): The worker predict call (handler) timeout in seconds.
                Defaults to None. If not provided, the default is 120 seconds.
            predictor_cls (callable[str, sagemaker.session.Session]): A function to call to create a
                predictor with an endpoint name and SageMaker ``Session``. If specified,
                ``deploy()`` returns
                the result of invoking this function on the created endpoint name.
            huggingface_hub_token (str): The HuggingFace Hub token to use for downloading the model
                artifacts for a model stored on the huggingface hub.
                Defaults to None. If not provided, the token must be specified in the
                HF_TOKEN environment variable in the env parameter.
            **kwargs: Keyword arguments passed to the superclass
                :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
                superclass :class:`~sagemaker.model.Model`.
        """
        super(DJLModel, self).__init__(predictor_cls=predictor_cls, **kwargs)
        self.model_id = model_id
        self.djl_version = djl_version
        self.djl_framework = djl_framework
        self.engine = engine
        self.task = task
        self.dtype = dtype
        self.tensor_parallel_degree = tensor_parallel_degree
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.job_queue_size = job_queue_size
        self.parallel_loading = parallel_loading
        self.model_loading_timeout = model_loading_timeout
        self.prediction_timeout = prediction_timeout
        self.sagemaker_session = self.sagemaker_session or Session()
        self.hub_token = huggingface_hub_token
        self._initialize_model()

    def _initialize_model(self):
        """Placeholder docstring"""
        self._validate_model_artifacts()
        self.engine = self._infer_engine()
        self.env = self._configure_environment_variables()
        self.image_uri = self._infer_image_uri()

    def _validate_model_artifacts(self):
        """Placeholder docstring"""
        if self.model_id is not None and self.model_data is not None:
            raise ValueError(
                "both model_id and model_data are provided. Please only provide one of them"
            )

    def _infer_engine(self) -> Optional[str]:
        """Placeholder docstring"""
        if self.engine is not None:
            logger.info("Using provided engine %s", self.engine)
            return self.engine

        if self.task == "text-embedding":
            return "OnnxRuntime"
        return "Python"

    def _infer_image_uri(self):
        """Placeholder docstring"""
        if self.image_uri is not None:
            return self.image_uri
        if self.djl_framework is None:
            self.djl_framework = "djl-lmi"
        return image_uris.retrieve(
            framework=self.djl_framework,
            region=self.sagemaker_session.boto_region_name,
            version=self.djl_version,
        )

    def _configure_environment_variables(self) -> Dict[str, str]:
        """Placeholder docstring"""
        env = self.env.copy() if self.env else {}
        env = _set_env_var_from_property(self.model_id, "HF_MODEL_ID", env)
        env = _set_env_var_from_property(self.task, "HF_TASK", env)
        env = _set_env_var_from_property(self.dtype, "OPTION_DTYPE", env)
        env = _set_env_var_from_property(self.min_workers, "SERVING_MIN_WORKERS", env)
        env = _set_env_var_from_property(self.max_workers, "SERVING_MAX_WORKERS", env)
        env = _set_env_var_from_property(self.job_queue_size, "SERVING_JOB_QUEUE_SIZE", env)
        env = _set_env_var_from_property(self.parallel_loading, "OPTION_PARALLEL_LOADING", env)
        env = _set_env_var_from_property(
            self.model_loading_timeout, "OPTION_MODEL_LOADING_TIMEOUT", env
        )
        env = _set_env_var_from_property(self.prediction_timeout, "OPTION_PREDICT_TIMEOUT", env)
        env = _set_env_var_from_property(self.hub_token, "HF_TOKEN", env)
        env = _set_env_var_from_property(self.engine, "OPTION_ENGINE", env)
        if "TENSOR_PARALLEL_DEGREE" not in env or "OPTION_TENSOR_PARALLEL_DEGREE" not in env:
            if self.tensor_parallel_degree is not None:
                env["TENSOR_PARALLEL_DEGREE"] = str(self.tensor_parallel_degree)
        return env

    def serving_image_uri(
        self,
        region_name,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
    ):
        """Placeholder docstring"""
        if self.image_uri:
            return self.image_uri
        return image_uris.retrieve(
            framework=self.djl_framework,
            region=region_name,
            version=self.djl_version,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            image_scope="inference",
            serverless_inference_config=serverless_inference_config,
        )

    def package_for_edge(self, **_):
        """Not implemented.

        DJLModels do not support SageMaker edge.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("DJLModels do not support Sagemaker Edge")

    def compile(self, **_):
        """Not implemented.

        DJLModels do not support SageMaker Neo compilation.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "DJLModels do not currently support compilation with SageMaker Neo"
        )

    def transformer(self, **_):
        """Not implemented.

        DJLModels do not support SageMaker Batch Transform.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "DJLModels do not currently support Batch Transform inference jobs"
        )

    def right_size(self, **_):
        """Not implemented.

        DJLModels do not support SageMaker Inference Recommendation Jobs.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "DJLModels do not currently support Inference Recommendation Jobs"
        )
