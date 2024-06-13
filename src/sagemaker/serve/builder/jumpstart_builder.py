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

import copy
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Type
import logging

from sagemaker.model import Model
from sagemaker import model_uris
from sagemaker.serve.model_server.djl_serving.prepare import prepare_djl_js_resources
from sagemaker.serve.model_server.djl_serving.utils import _get_admissible_tensor_parallel_degrees
from sagemaker.serve.model_server.multi_model_server.prepare import prepare_mms_js_resources
from sagemaker.serve.model_server.tgi.prepare import prepare_tgi_js_resources, _create_dir_structure
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.exceptions import (
    LocalDeepPingException,
    LocalModelOutOfMemoryException,
    LocalModelInvocationException,
    LocalModelLoadException,
    SkipTuningComboException,
)
from sagemaker.serve.utils.predictors import (
    DjlLocalModePredictor,
    TgiLocalModePredictor,
    TransformersLocalModePredictor,
)
from sagemaker.serve.utils.local_hardware import (
    _get_nb_instance,
    _get_ram_usage_mb,
)
from sagemaker.serve.utils.telemetry_logger import _capture_telemetry
from sagemaker.serve.utils.tuning import (
    _pretty_print_results_jumpstart,
    _serial_benchmark,
    _concurrent_benchmark,
    _more_performant,
    _sharded_supported,
)
from sagemaker.serve.utils.types import ModelServer
from sagemaker.base_predictor import PredictorBase
from sagemaker.jumpstart.model import JumpStartModel

_DJL_MODEL_BUILDER_ENTRY_POINT = "inference.py"
_NO_JS_MODEL_EX = "HuggingFace JumpStart Model ID not detected. Building for HuggingFace Model ID."
_JS_SCOPE = "inference"
_CODE_FOLDER = "code"
_JS_ENABLED_MODEL_SERVERS = {
    ModelServer.DJL_SERVING,
    ModelServer.TGI,
}

logger = logging.getLogger(__name__)


class JumpStart(ABC):
    """DJL build logic for ModelBuilder()"""

    def __init__(self):
        self.model = None
        self.serve_settings = None
        self.sagemaker_session = None
        self.model_path = None
        self.dependencies = None
        self.modes = None
        self.mode = None
        self.model_server = None
        self.image_uri = None
        self._is_custom_image_uri = False
        self.vpc_config = None
        self._original_deploy = None
        self.secret_key = None
        self.js_model_config = None
        self._inferred_parallel_degree = None
        self._inferred_data_type = None
        self._inferred_max_tokens = None
        self.pysdk_model = None
        self.model_uri = None
        self.existing_properties = None
        self.prepared_for_tgi = None
        self.prepared_for_djl = None
        self.prepared_for_mms = None
        self.schema_builder = None
        self.nb_instance_type = None
        self.ram_usage_model_load = None
        self.jumpstart = None

    @abstractmethod
    def _prepare_for_mode(self):
        """Placeholder docstring"""

    @abstractmethod
    def _get_client_translators(self):
        """Placeholder docstring"""

    def _is_jumpstart_model_id(self) -> bool:
        """Placeholder docstring"""
        try:
            model_uris.retrieve(model_id=self.model, model_version="*", model_scope=_JS_SCOPE)
        except KeyError:
            logger.warning(_NO_JS_MODEL_EX)
            return False

        logger.info("JumpStart Model ID detected.")
        return True

    def _create_pre_trained_js_model(self) -> Type[Model]:
        """Placeholder docstring"""
        pysdk_model = JumpStartModel(self.model, vpc_config=self.vpc_config)
        pysdk_model.sagemaker_session = self.sagemaker_session

        self._original_deploy = pysdk_model.deploy
        pysdk_model.deploy = self._js_builder_deploy_wrapper
        return pysdk_model

    @_capture_telemetry("jumpstart.deploy")
    def _js_builder_deploy_wrapper(self, *args, **kwargs) -> Type[PredictorBase]:
        """Placeholder docstring"""
        if "mode" in kwargs and kwargs.get("mode") != self.mode:
            overwrite_mode = kwargs.get("mode")
            # mode overwritten by customer during model.deploy()
            logger.warning(
                "Deploying in %s Mode, overriding existing configurations set for %s mode",
                overwrite_mode,
                self.mode,
            )

            if overwrite_mode == Mode.SAGEMAKER_ENDPOINT:
                self.mode = self.pysdk_model.mode = Mode.SAGEMAKER_ENDPOINT
                if (
                    not hasattr(self, "prepared_for_djl")
                    or not hasattr(self, "prepared_for_tgi")
                    or not hasattr(self, "prepared_for_mms")
                ):
                    self.pysdk_model.model_data, env = self._prepare_for_mode()
            elif overwrite_mode == Mode.LOCAL_CONTAINER:
                self.mode = self.pysdk_model.mode = Mode.LOCAL_CONTAINER

                if not hasattr(self, "prepared_for_djl"):
                    (
                        self.existing_properties,
                        self.js_model_config,
                        self.prepared_for_djl,
                    ) = prepare_djl_js_resources(
                        model_path=self.model_path,
                        js_id=self.model,
                        dependencies=self.dependencies,
                        model_data=self.pysdk_model.model_data,
                    )
                elif not hasattr(self, "prepared_for_tgi"):
                    self.js_model_config, self.prepared_for_tgi = prepare_tgi_js_resources(
                        model_path=self.model_path,
                        js_id=self.model,
                        dependencies=self.dependencies,
                        model_data=self.pysdk_model.model_data,
                    )
                elif not hasattr(self, "prepared_for_mms"):
                    self.js_model_config, self.prepared_for_mms = prepare_mms_js_resources(
                        model_path=self.model_path,
                        js_id=self.model,
                        dependencies=self.dependencies,
                        model_data=self.pysdk_model.model_data,
                    )

                self._prepare_for_mode()
                env = {}
            else:
                raise ValueError("Mode %s is not supported!" % overwrite_mode)

            self.pysdk_model.env.update(env)

        serializer = self.schema_builder.input_serializer
        deserializer = self.schema_builder._output_deserializer
        if self.mode == Mode.LOCAL_CONTAINER:
            if self.model_server == ModelServer.DJL_SERVING:
                predictor = DjlLocalModePredictor(
                    self.modes[str(Mode.LOCAL_CONTAINER)], serializer, deserializer
                )
            elif self.model_server == ModelServer.TGI:
                predictor = TgiLocalModePredictor(
                    self.modes[str(Mode.LOCAL_CONTAINER)], serializer, deserializer
                )
            elif self.model_server == ModelServer.MMS:
                predictor = TransformersLocalModePredictor(
                    self.modes[str(Mode.LOCAL_CONTAINER)], serializer, deserializer
                )

            ram_usage_before = _get_ram_usage_mb()
            self.modes[str(Mode.LOCAL_CONTAINER)].create_server(
                self.image_uri,
                600,
                None,
                predictor,
                self.pysdk_model.env,
                jumpstart=True,
            )
            ram_usage_after = _get_ram_usage_mb()

            self.ram_usage_model_load = max(ram_usage_after - ram_usage_before, 0)

            return predictor

        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True
        if hasattr(self, "nb_instance_type"):
            kwargs.update({"instance_type": self.nb_instance_type})

        if "mode" in kwargs:
            del kwargs["mode"]
        if "role" in kwargs:
            self.pysdk_model.role = kwargs.get("role")
            del kwargs["role"]

        predictor = self._original_deploy(*args, **kwargs)
        predictor.serializer = serializer
        predictor.deserializer = deserializer

        return predictor

    def _build_for_djl_jumpstart(self):
        """Placeholder docstring"""

        env = {}
        _create_dir_structure(self.model_path)
        if self.mode == Mode.LOCAL_CONTAINER:
            if not hasattr(self, "prepared_for_djl"):
                (
                    self.existing_properties,
                    self.js_model_config,
                    self.prepared_for_djl,
                ) = prepare_djl_js_resources(
                    model_path=self.model_path,
                    js_id=self.model,
                    dependencies=self.dependencies,
                    model_data=self.pysdk_model.model_data,
                )
            self._prepare_for_mode()
        elif self.mode == Mode.SAGEMAKER_ENDPOINT and hasattr(self, "prepared_for_djl"):
            self.nb_instance_type = _get_nb_instance()
            self.pysdk_model.model_data, env = self._prepare_for_mode()

        self.pysdk_model.env.update(env)

    def _build_for_tgi_jumpstart(self):
        """Placeholder docstring"""

        env = {}
        if self.mode == Mode.LOCAL_CONTAINER:
            if not hasattr(self, "prepared_for_tgi"):
                self.js_model_config, self.prepared_for_tgi = prepare_tgi_js_resources(
                    model_path=self.model_path,
                    js_id=self.model,
                    dependencies=self.dependencies,
                    model_data=self.pysdk_model.model_data,
                )
            self._prepare_for_mode()
        elif self.mode == Mode.SAGEMAKER_ENDPOINT and hasattr(self, "prepared_for_tgi"):
            self.pysdk_model.model_data, env = self._prepare_for_mode()

        self.pysdk_model.env.update(env)

    def _build_for_mms_jumpstart(self):
        """Placeholder docstring"""

        env = {}
        if self.mode == Mode.LOCAL_CONTAINER:
            if not hasattr(self, "prepared_for_mms"):
                self.js_model_config, self.prepared_for_mms = prepare_mms_js_resources(
                    model_path=self.model_path,
                    js_id=self.model,
                    dependencies=self.dependencies,
                    model_data=self.pysdk_model.model_data,
                )
            self._prepare_for_mode()
        elif self.mode == Mode.SAGEMAKER_ENDPOINT and hasattr(self, "prepared_for_mms"):
            self.pysdk_model.model_data, env = self._prepare_for_mode()

        self.pysdk_model.env.update(env)

    def _tune_for_js(self, sharded_supported: bool, max_tuning_duration: int = 1800):
        """Tune for Jumpstart Models in Local Mode.

        Args:
            sharded_supported (bool): Indicates whether sharding is supported by this ``Model``
            max_tuning_duration (int): The maximum timeout to deploy this ``Model`` locally.
                Default: ``1800``
        returns:
            Tuned Model.
        """
        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            logger.warning(
                "Tuning is only a %s capability. Returning original model.", Mode.LOCAL_CONTAINER
            )
            return self.pysdk_model

        num_shard_env_var_name = "SM_NUM_GPUS"
        if "OPTION_TENSOR_PARALLEL_DEGREE" in self.pysdk_model.env.keys():
            num_shard_env_var_name = "OPTION_TENSOR_PARALLEL_DEGREE"

        initial_env_vars = copy.deepcopy(self.pysdk_model.env)
        admissible_tensor_parallel_degrees = _get_admissible_tensor_parallel_degrees(
            self.js_model_config
        )

        if len(admissible_tensor_parallel_degrees) > 1 and not sharded_supported:
            admissible_tensor_parallel_degrees = [1]
            logger.warning(
                "Sharding across multiple GPUs is not supported for this model. "
                "Model can only be sharded across [1] GPU"
            )

        benchmark_results = {}
        best_tuned_combination = None
        timeout = datetime.now() + timedelta(seconds=max_tuning_duration)
        for tensor_parallel_degree in admissible_tensor_parallel_degrees:
            if datetime.now() > timeout:
                logger.info("Max tuning duration reached. Tuning stopped.")
                break

            self.pysdk_model.env.update({num_shard_env_var_name: str(tensor_parallel_degree)})
            try:
                logger.info("Trying tensor parallel degree: %s", tensor_parallel_degree)

                predictor = self.pysdk_model.deploy(model_data_download_timeout=max_tuning_duration)

                avg_latency, p90, avg_tokens_per_second = _serial_benchmark(
                    predictor, self.schema_builder.sample_input
                )
                throughput_per_second, standard_deviation = _concurrent_benchmark(
                    predictor, self.schema_builder.sample_input
                )

                tested_env = copy.deepcopy(self.pysdk_model.env)
                logger.info(
                    "Average latency: %s, throughput/s: %s for configuration: %s",
                    avg_latency,
                    throughput_per_second,
                    tested_env,
                )
                benchmark_results[avg_latency] = [
                    tested_env,
                    p90,
                    avg_tokens_per_second,
                    throughput_per_second,
                    standard_deviation,
                ]

                if not best_tuned_combination:
                    best_tuned_combination = [
                        avg_latency,
                        tensor_parallel_degree,
                        None,
                        p90,
                        avg_tokens_per_second,
                        throughput_per_second,
                        standard_deviation,
                    ]
                else:
                    tuned_configuration = [
                        avg_latency,
                        tensor_parallel_degree,
                        None,
                        p90,
                        avg_tokens_per_second,
                        throughput_per_second,
                        standard_deviation,
                    ]
                    if _more_performant(best_tuned_combination, tuned_configuration):
                        best_tuned_combination = tuned_configuration
            except LocalDeepPingException as e:
                logger.warning(
                    "Deployment unsuccessful with %s: %s. " "Failed to invoke the model server: %s",
                    num_shard_env_var_name,
                    tensor_parallel_degree,
                    str(e),
                )
            except LocalModelOutOfMemoryException as e:
                logger.warning(
                    "Deployment unsuccessful with %s: %s. "
                    "Out of memory when loading the model: %s",
                    num_shard_env_var_name,
                    tensor_parallel_degree,
                    str(e),
                )
            except LocalModelInvocationException as e:
                logger.warning(
                    "Deployment unsuccessful with %s: %s. "
                    "Failed to invoke the model server: %s"
                    "Please check that model server configurations are as expected "
                    "(Ex. serialization, deserialization, content_type, accept).",
                    num_shard_env_var_name,
                    tensor_parallel_degree,
                    str(e),
                )
            except LocalModelLoadException as e:
                logger.warning(
                    "Deployment unsuccessful with %s: %s. " "Failed to load the model: %s.",
                    num_shard_env_var_name,
                    tensor_parallel_degree,
                    str(e),
                )
            except SkipTuningComboException as e:
                logger.warning(
                    "Deployment with %s: %s"
                    "was expected to be successful. However failed with: %s. "
                    "Trying next combination.",
                    num_shard_env_var_name,
                    tensor_parallel_degree,
                    str(e),
                )
            except Exception:  # pylint: disable=W0703
                logger.exception(
                    "Deployment unsuccessful with %s: %s. " "with uncovered exception",
                    num_shard_env_var_name,
                    tensor_parallel_degree,
                )

        if best_tuned_combination:
            self.pysdk_model.env.update({num_shard_env_var_name: str(best_tuned_combination[1])})

            _pretty_print_results_jumpstart(benchmark_results, [num_shard_env_var_name])
            logger.info(
                "Model Configuration: %s was most performant with avg latency: %s, "
                "p90 latency: %s, average tokens per second: %s, throughput/s: %s, "
                "standard deviation of request %s",
                self.pysdk_model.env,
                best_tuned_combination[0],
                best_tuned_combination[3],
                best_tuned_combination[4],
                best_tuned_combination[5],
                best_tuned_combination[6],
            )
        else:
            self.pysdk_model.env.update(initial_env_vars)
            logger.debug(
                "Failed to gather any tuning results. "
                "Please inspect the stack trace emitted from live logging for more details. "
                "Falling back to default model configurations: %s",
                self.pysdk_model.env,
            )

        return self.pysdk_model

    @_capture_telemetry("djl_jumpstart.tune")
    def tune_for_djl_jumpstart(self, max_tuning_duration: int = 1800):
        """Tune for Jumpstart Models with DJL DLC"""
        return self._tune_for_js(sharded_supported=True, max_tuning_duration=max_tuning_duration)

    @_capture_telemetry("tgi_jumpstart.tune")
    def tune_for_tgi_jumpstart(self, max_tuning_duration: int = 1800):
        """Tune for Jumpstart Models with TGI DLC"""
        sharded_supported = _sharded_supported(self.model, self.js_model_config)
        return self._tune_for_js(
            sharded_supported=sharded_supported, max_tuning_duration=max_tuning_duration
        )

    def _build_for_jumpstart(self):
        """Placeholder docstring"""
        # we do not pickle for jumpstart. set to none
        self.secret_key = None
        self.jumpstart = True

        pysdk_model = self._create_pre_trained_js_model()
        image_uri = pysdk_model.image_uri

        logger.info("JumpStart ID %s is packaged with Image URI: %s", self.model, image_uri)

        if self._is_gated_model(pysdk_model) and self.mode != Mode.SAGEMAKER_ENDPOINT:
            raise ValueError(
                "JumpStart Gated Models are only supported in SAGEMAKER_ENDPOINT mode."
            )

        if "djl-inference" in image_uri:
            logger.info("Building for DJL JumpStart Model ID...")
            self.model_server = ModelServer.DJL_SERVING
            self.pysdk_model = pysdk_model
            self.image_uri = self.pysdk_model.image_uri

            self._build_for_djl_jumpstart()

            self.pysdk_model.tune = self.tune_for_djl_jumpstart
        elif "tgi-inference" in image_uri:
            logger.info("Building for TGI JumpStart Model ID...")
            self.model_server = ModelServer.TGI
            self.pysdk_model = pysdk_model
            self.image_uri = self.pysdk_model.image_uri

            self._build_for_tgi_jumpstart()

            self.pysdk_model.tune = self.tune_for_tgi_jumpstart
        elif "huggingface-pytorch-inference:" in image_uri:
            logger.info("Building for MMS JumpStart Model ID...")
            self.model_server = ModelServer.MMS
            self.pysdk_model = pysdk_model
            self.image_uri = self.pysdk_model.image_uri

            self._build_for_mms_jumpstart()
        elif self.mode != Mode.SAGEMAKER_ENDPOINT:
            raise ValueError(
                "JumpStart Model ID was not packaged "
                "with djl-inference, tgi-inference, or mms-inference container."
            )

        return self.pysdk_model

    def _is_gated_model(self, model) -> bool:
        """Determine if ``this`` Model is Gated

        Args:
            model (Model): Jumpstart Model
        Returns:
            bool: ``True`` if ``this`` Model is Gated
        """
        s3_uri = model.model_data
        if isinstance(s3_uri, dict):
            s3_uri = s3_uri.get("S3DataSource").get("S3Uri")

        if s3_uri is None:
            return False
        return "private" in s3_uri
