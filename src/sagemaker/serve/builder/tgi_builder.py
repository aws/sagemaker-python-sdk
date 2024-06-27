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
"""Holds mixin logic to support deployment of Model ID"""
from __future__ import absolute_import
import logging
from typing import Type
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from sagemaker.model import Model
from sagemaker.serve.utils.exceptions import (
    LocalDeepPingException,
    LocalModelLoadException,
    LocalModelOutOfMemoryException,
    LocalModelInvocationException,
    SkipTuningComboException,
)
from sagemaker.serve.utils.tuning import (
    _serial_benchmark,
    _concurrent_benchmark,
    _more_performant,
    _pretty_print_results_tgi,
)
from sagemaker.serve.utils.hf_utils import _get_model_config_properties_from_hf
from sagemaker.serve.model_server.djl_serving.utils import (
    _get_admissible_tensor_parallel_degrees,
    _get_default_tensor_parallel_degree,
)
from sagemaker.serve.model_server.tgi.utils import (
    _get_default_tgi_configurations,
    _get_admissible_dtypes,
)
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from sagemaker.serve.utils.local_hardware import (
    _get_nb_instance,
    _get_ram_usage_mb,
    _get_gpu_info,
    _get_gpu_info_fallback,
)
from sagemaker.serve.model_server.tgi.prepare import _create_dir_structure
from sagemaker.serve.utils.predictors import TgiLocalModePredictor
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.telemetry_logger import _capture_telemetry
from sagemaker.base_predictor import PredictorBase

logger = logging.getLogger(__name__)

_CODE_FOLDER = "code"
_INVALID_SAMPLE_DATA_EX = (
    'For tgi, sample input must be of {"inputs": str, "parameters": dict}, '
    'sample output must be of [{"generated_text": str,}]'
)


class TGI(ABC):
    """TGI build logic for ModelBuilder()"""

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
        self.image_config = None
        self.vpc_config = None
        self._original_deploy = None
        self.hf_model_config = None
        self._default_tensor_parallel_degree = None
        self._default_data_type = None
        self._default_max_tokens = None
        self.pysdk_model = None
        self.schema_builder = None
        self.env_vars = None
        self.nb_instance_type = None
        self.ram_usage_model_load = None
        self.secret_key = None
        self.jumpstart = None
        self.role_arn = None

    @abstractmethod
    def _prepare_for_mode(self):
        """Placeholder docstring"""

    @abstractmethod
    def _get_client_translators(self):
        """Placeholder docstring"""

    def _set_to_tgi(self):
        """Placeholder docstring"""
        if self.model_server != ModelServer.TGI:
            messaging = (
                "HuggingFace Model ID support on model server: "
                f"{self.model_server} is not currently supported. "
                f"Defaulting to {ModelServer.TGI}"
            )
            logger.warning(messaging)
            self.model_server = ModelServer.TGI

    def _validate_tgi_serving_sample_data(self):
        """Placeholder docstring"""
        sample_input = self.schema_builder.sample_input
        sample_output = self.schema_builder.sample_output

        if (  # pylint: disable=R0916
            not isinstance(sample_input, dict)
            or "inputs" not in sample_input
            or "parameters" not in sample_input
            or not isinstance(sample_output, list)
            or not isinstance(sample_output[0], dict)
            or "generated_text" not in sample_output[0]
        ):
            raise ValueError(_INVALID_SAMPLE_DATA_EX)

    def _create_tgi_model(self) -> Type[Model]:
        """Placeholder docstring"""
        if not self.image_uri:
            self.image_uri = get_huggingface_llm_image_uri(
                "huggingface", session=self.sagemaker_session
            )
            logger.info("Auto detected %s. Proceeding with the the deployment.", self.image_uri)

        pysdk_model = HuggingFaceModel(
            image_uri=self.image_uri,
            image_config=self.image_config,
            vpc_config=self.vpc_config,
            env=self.env_vars,
            role=self.role_arn,
            sagemaker_session=self.sagemaker_session,
        )

        self._original_deploy = pysdk_model.deploy
        pysdk_model.deploy = self._tgi_model_builder_deploy_wrapper
        return pysdk_model

    @_capture_telemetry("tgi.deploy")
    def _tgi_model_builder_deploy_wrapper(self, *args, **kwargs) -> Type[PredictorBase]:
        """Placeholder docstring"""
        timeout = kwargs.get("model_data_download_timeout")
        if timeout:
            self.pysdk_model.env.update({"MODEL_LOADING_TIMEOUT": str(timeout)})

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
            elif overwrite_mode == Mode.LOCAL_CONTAINER:
                self._prepare_for_mode()
                self.mode = self.pysdk_model.mode = Mode.LOCAL_CONTAINER
            else:
                raise ValueError("Mode %s is not supported!" % overwrite_mode)

        serializer = self.schema_builder.input_serializer
        deserializer = self.schema_builder._output_deserializer
        if self.mode == Mode.LOCAL_CONTAINER:
            timeout = kwargs.get("model_data_download_timeout")

            predictor = TgiLocalModePredictor(
                self.modes[str(Mode.LOCAL_CONTAINER)], serializer, deserializer
            )

            ram_usage_before = _get_ram_usage_mb()
            self.modes[str(Mode.LOCAL_CONTAINER)].create_server(
                self.image_uri,
                timeout if timeout else 1800,
                None,
                predictor,
                self.pysdk_model.env,
                jumpstart=False,
            )
            ram_usage_after = _get_ram_usage_mb()

            self.ram_usage_model_load = max(ram_usage_after - ram_usage_before, 0)

            return predictor

        if "mode" in kwargs:
            del kwargs["mode"]
        if "role" in kwargs:
            self.pysdk_model.role = kwargs.get("role")
            del kwargs["role"]

        # set model_data to uncompressed s3 dict
        self.pysdk_model.model_data, env_vars = self._prepare_for_mode()
        self.env_vars.update(env_vars)
        self.pysdk_model.env.update(self.env_vars)

        # if the weights have been cached via local container mode -> set to offline
        if str(Mode.LOCAL_CONTAINER) in self.modes:
            self.pysdk_model.env.update({"TRANSFORMERS_OFFLINE": "1"})
        else:
            # if has not been built for local container we must use cache
            # that hosting has write access to.
            self.pysdk_model.env["TRANSFORMERS_CACHE"] = "/tmp"
            self.pysdk_model.env["HF_HOME"] = "/tmp"
            self.pysdk_model.env["HUGGINGFACE_HUB_CACHE"] = "/tmp"

        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True

        if self.nb_instance_type and "instance_type" not in kwargs:
            kwargs.update({"instance_type": self.nb_instance_type})
        elif not self.nb_instance_type and "instance_type" not in kwargs:
            raise ValueError(
                "Instance type must be provided when deploying " "to SageMaker Endpoint mode."
            )
        else:
            try:
                tot_gpus = _get_gpu_info(kwargs.get("instance_type"), self.sagemaker_session)
            except Exception:  # pylint: disable=W0703
                tot_gpus = _get_gpu_info_fallback(kwargs.get("instance_type"))
            default_num_shard = _get_default_tensor_parallel_degree(self.hf_model_config, tot_gpus)
            self.pysdk_model.env.update(
                {
                    "NUM_SHARD": str(default_num_shard),
                    "SHARDED": "true" if default_num_shard > 1 else "false",
                }
            )

        if "initial_instance_count" not in kwargs:
            kwargs.update({"initial_instance_count": 1})

        predictor = self._original_deploy(*args, **kwargs)

        self.pysdk_model.env.update({"TRANSFORMERS_OFFLINE": "0"})

        predictor.serializer = serializer
        predictor.deserializer = deserializer
        return predictor

    def _build_for_hf_tgi(self):
        """Placeholder docstring"""
        self.nb_instance_type = _get_nb_instance()

        _create_dir_structure(self.model_path)
        if not hasattr(self, "pysdk_model"):
            self.env_vars.update({"HF_MODEL_ID": self.model})
            self.hf_model_config = _get_model_config_properties_from_hf(
                self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
            )
            default_tgi_configurations, _default_max_new_tokens = _get_default_tgi_configurations(
                self.model, self.hf_model_config, self.schema_builder
            )
            self.env_vars.update(default_tgi_configurations)

            self.schema_builder.sample_input["parameters"][
                "max_new_tokens"
            ] = _default_max_new_tokens
        self.pysdk_model = self._create_tgi_model()

        if self.mode == Mode.LOCAL_CONTAINER:
            self._prepare_for_mode()

        return self.pysdk_model

    @_capture_telemetry("tgi.tune")
    def _tune_for_hf_tgi(self, max_tuning_duration: int = 1800):
        """Placeholder docstring"""
        if self.mode != Mode.LOCAL_CONTAINER:
            logger.warning(
                "Tuning is only a %s capability. Returning original model.", Mode.LOCAL_CONTAINER
            )
            return self.pysdk_model

        admissible_num_shard = _get_admissible_tensor_parallel_degrees(self.hf_model_config)
        admissible_dtypes = _get_admissible_dtypes()

        benchmark_results = {}
        best_tuned_combination = None
        timeout = datetime.now() + timedelta(seconds=max_tuning_duration)
        for num_shard in admissible_num_shard:
            if datetime.now() > timeout:
                logger.info("Max tuning duration reached. Tuning stopped.")
                break

            dtype_passes = 0
            for dtype in admissible_dtypes:
                logger.info("Trying num shard: %s, dtype: %s...", num_shard, dtype)

                if num_shard == 1:
                    self.env_vars.update({"SHARDED": "false"})
                else:
                    self.env_vars.update({"SHARDED": "true"})
                self.env_vars.update({"NUM_SHARD": str(num_shard), "DTYPE": dtype})
                self.pysdk_model = self._create_tgi_model()

                try:
                    predictor = self.pysdk_model.deploy(
                        model_data_download_timeout=max_tuning_duration
                    )

                    avg_latency, p90, avg_tokens_per_second = _serial_benchmark(
                        predictor, self.schema_builder.sample_input
                    )
                    throughput_per_second, standard_deviation = _concurrent_benchmark(
                        predictor, self.schema_builder.sample_input
                    )

                    tested_env = self.pysdk_model.env.copy()
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
                            num_shard,
                            dtype,
                            p90,
                            avg_tokens_per_second,
                            throughput_per_second,
                            standard_deviation,
                        ]
                    else:
                        tuned_configuration = [
                            avg_latency,
                            num_shard,
                            dtype,
                            p90,
                            avg_tokens_per_second,
                            throughput_per_second,
                            standard_deviation,
                        ]
                        if _more_performant(best_tuned_combination, tuned_configuration):
                            best_tuned_combination = tuned_configuration
                except LocalDeepPingException as e:
                    logger.warning(
                        "Deployment unsuccessful with num shard: %s. dtype: %s. "
                        "Failed to invoke the model server: %s",
                        num_shard,
                        dtype,
                        str(e),
                    )
                    break
                except LocalModelOutOfMemoryException as e:
                    logger.warning(
                        "Deployment unsuccessful with num shard: %s, dtype: %s. "
                        "Out of memory when loading the model: %s",
                        num_shard,
                        dtype,
                        str(e),
                    )
                    break
                except LocalModelInvocationException as e:
                    logger.warning(
                        "Deployment unsuccessful with num shard: %s, dtype: %s. "
                        "Failed to invoke the model server: %s"
                        "Please check that model server configurations are as expected "
                        "(Ex. serialization, deserialization, content_type, accept).",
                        num_shard,
                        dtype,
                        str(e),
                    )
                    break
                except LocalModelLoadException as e:
                    logger.warning(
                        "Deployment unsuccessful with num shard: %s, dtype: %s. "
                        "Failed to load the model: %s.",
                        num_shard,
                        dtype,
                        str(e),
                    )
                    break
                except SkipTuningComboException as e:
                    logger.warning(
                        "Deployment with num shard: %s, dtype: %s "
                        "was expected to be successful. However failed with: %s. "
                        "Trying next combination.",
                        num_shard,
                        dtype,
                        str(e),
                    )
                except Exception:  # pylint: disable=W0703
                    logger.exception(
                        "Deployment unsuccessful with num shard: %s, dtype: %s "
                        "with uncovered exception",
                        num_shard,
                        dtype,
                    )
                    break
                dtype_passes += 1
            if dtype_passes == 0:
                logger.info(
                    "Lowest admissible num shard: %s and highest dtype: "
                    "%s combination has been attempted. Tuning stopped.",
                    num_shard,
                    dtype,
                )
                break

        if best_tuned_combination:
            if best_tuned_combination[1] == 1:
                self.env_vars.update({"SHARDED": "false"})
            else:
                self.env_vars.update({"SHARDED": "true"})
            self.env_vars.update(
                {"NUM_SHARD": str(best_tuned_combination[1]), "DTYPE": best_tuned_combination[2]}
            )

            self.pysdk_model = self._create_tgi_model()

            _pretty_print_results_tgi(benchmark_results)
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
            self.hf_model_config = _get_model_config_properties_from_hf(
                self.model, self.env_vars.get("HUGGING_FACE_HUB_TOKEN")
            )
            default_tgi_configurations, _default_max_new_tokens = _get_default_tgi_configurations(
                self.model, self.hf_model_config, self.schema_builder
            )
            self.env_vars.update(default_tgi_configurations)
            self.schema_builder.sample_input["parameters"][
                "max_new_tokens"
            ] = _default_max_new_tokens
            self.pysdk_model = self._create_tgi_model()

            logger.debug(
                "Failed to gather any tuning results. "
                "Please inspect the stack trace emitted from live logging for more details. "
                "Falling back to default serving.properties: %s",
                self.pysdk_model.env,
            )

        return self.pysdk_model

    def _build_for_tgi(self):
        """Placeholder docstring"""
        self.secret_key = None

        self._set_to_tgi()
        self._validate_tgi_serving_sample_data()

        self.pysdk_model = self._build_for_hf_tgi()
        self.pysdk_model.tune = self._tune_for_hf_tgi
        return self.pysdk_model
