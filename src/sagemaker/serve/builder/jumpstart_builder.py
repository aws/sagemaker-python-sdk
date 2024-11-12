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
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Type, Any, List, Dict, Optional
import logging

from botocore.exceptions import ClientError

from sagemaker.enums import Tag
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
from sagemaker.serve.utils.optimize_utils import (
    _generate_model_source,
    _update_environment_variables,
    _extract_speculative_draft_model_provider,
    _is_image_compatible_with_optimization_job,
    _generate_channel_name,
    _extract_optimization_config_and_env,
    _is_optimized,
    _custom_speculative_decoding,
    SPECULATIVE_DRAFT_MODEL,
    _is_inferentia_or_trainium,
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
from sagemaker.utils import Tags

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
        self.instance_type = None
        self.nb_instance_type = None
        self.ram_usage_model_load = None
        self.model_hub = None
        self.model_metadata = None
        self.role_arn = None
        self.is_fine_tuned = None
        self.is_compiled = False
        self.is_quantized = False
        self.speculative_decoding_draft_model_source = None
        self.deployment_config_name = None
        self.name = None

    @abstractmethod
    def _prepare_for_mode(self, **kwargs):
        """Placeholder docstring"""

    @abstractmethod
    def _get_client_translators(self):
        """Placeholder docstring"""

    def _is_jumpstart_model_id(self) -> bool:
        """Placeholder docstring"""
        if self.model is None:
            return False

        try:
            model_uris.retrieve(model_id=self.model, model_version="*", model_scope=_JS_SCOPE)
        except KeyError:
            logger.warning(_NO_JS_MODEL_EX)
            return False

        logger.info("JumpStart Model ID detected.")
        return True

    def _create_pre_trained_js_model(self) -> Type[Model]:
        """Placeholder docstring"""
        pysdk_model = JumpStartModel(
            self.model,
            vpc_config=self.vpc_config,
            sagemaker_session=self.sagemaker_session,
            name=self.name,
        )

        self._original_deploy = pysdk_model.deploy
        pysdk_model.deploy = self._js_builder_deploy_wrapper
        return pysdk_model

    @_capture_telemetry("jumpstart.deploy")
    def _js_builder_deploy_wrapper(self, *args, **kwargs) -> Type[PredictorBase]:
        """Placeholder docstring"""
        env = {}
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
                    if not _is_optimized(self.pysdk_model):
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
            self.nb_instance_type = self.instance_type or _get_nb_instance()
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

    def set_deployment_config(self, config_name: str, instance_type: str) -> None:
        """Sets the deployment config to apply to the model.

        Args:
            config_name (str):
                The name of the deployment config to apply to the model.
                Call list_deployment_configs to see the list of config names.
            instance_type (str):
                The instance_type that the model will use after setting
                the config.
        """
        if not hasattr(self, "pysdk_model") or self.pysdk_model is None:
            raise Exception("Cannot set deployment config to an uninitialized model.")

        self.pysdk_model.set_deployment_config(config_name, instance_type)
        self.deployment_config_name = config_name

        self.instance_type = instance_type

        # JS-benchmarked models only include SageMaker-provided SD models
        if self.pysdk_model.additional_model_data_sources:
            self.speculative_decoding_draft_model_source = "sagemaker"
            self.pysdk_model.add_tags(
                {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "sagemaker"},
            )
            self.pysdk_model.remove_tag_with_key(Tag.OPTIMIZATION_JOB_NAME)
            self.pysdk_model.remove_tag_with_key(Tag.FINE_TUNING_MODEL_PATH)
            self.pysdk_model.remove_tag_with_key(Tag.FINE_TUNING_JOB_NAME)

    def get_deployment_config(self) -> Optional[Dict[str, Any]]:
        """Gets the deployment config to apply to the model.

        Returns:
            Optional[Dict[str, Any]]: Deployment config to apply to this model.
        """
        if not hasattr(self, "pysdk_model") or self.pysdk_model is None:
            self._build_for_jumpstart()

        return self.pysdk_model.deployment_config

    def display_benchmark_metrics(self, **kwargs):
        """Display Markdown Benchmark Metrics for deployment configs."""
        if not hasattr(self, "pysdk_model") or self.pysdk_model is None:
            self._build_for_jumpstart()

        self.pysdk_model.display_benchmark_metrics(**kwargs)

    def list_deployment_configs(self) -> List[Dict[str, Any]]:
        """List deployment configs for ``This`` model in the current region.

        Returns:
            List[Dict[str, Any]]: A list of deployment configs.
        """
        if not hasattr(self, "pysdk_model") or self.pysdk_model is None:
            self._build_for_jumpstart()

        return self.pysdk_model.list_deployment_configs()

    def _is_fine_tuned_model(self) -> bool:
        """Checks whether a fine-tuned model exists."""
        return self.model_metadata and (
            self.model_metadata.get("FINE_TUNING_MODEL_PATH")
            or self.model_metadata.get("FINE_TUNING_JOB_NAME")
        )

    def _update_model_data_for_fine_tuned_model(self, pysdk_model: Type[Model]) -> Type[Model]:
        """Set the model path and data and add fine-tuning tags for the model."""
        # TODO: determine precedence of FINE_TUNING_MODEL_PATH and FINE_TUNING_JOB_NAME
        if fine_tuning_model_path := self.model_metadata.get("FINE_TUNING_MODEL_PATH"):
            if not re.match("^(https|s3)://([^/]+)/?(.*)$", fine_tuning_model_path):
                raise ValueError(
                    f"Invalid path for FINE_TUNING_MODEL_PATH: {fine_tuning_model_path}."
                )
            pysdk_model.model_data["S3DataSource"]["S3Uri"] = fine_tuning_model_path
            pysdk_model.add_tags(
                {"Key": Tag.FINE_TUNING_MODEL_PATH, "Value": fine_tuning_model_path}
            )
            logger.info(
                "FINE_TUNING_MODEL_PATH detected. Using fine-tuned model found in %s.",
                fine_tuning_model_path,
            )
            return pysdk_model

        if fine_tuning_job_name := self.model_metadata.get("FINE_TUNING_JOB_NAME"):
            try:
                response = self.sagemaker_session.sagemaker_client.describe_training_job(
                    TrainingJobName=fine_tuning_job_name
                )
                fine_tuning_model_path = response["ModelArtifacts"]["S3ModelArtifacts"]
                pysdk_model.model_data["S3DataSource"]["S3Uri"] = fine_tuning_model_path
                pysdk_model.add_tags(
                    [
                        {"key": Tag.FINE_TUNING_JOB_NAME, "value": fine_tuning_job_name},
                        {"key": Tag.FINE_TUNING_MODEL_PATH, "value": fine_tuning_model_path},
                    ]
                )
                logger.info(
                    "FINE_TUNING_JOB_NAME detected. Using fine-tuned model found in %s.",
                    fine_tuning_model_path,
                )
                return pysdk_model
            except ClientError:
                raise ValueError(
                    f"Invalid job name for FINE_TUNING_JOB_NAME: {fine_tuning_job_name}."
                )

        raise ValueError(
            "Input model not found. Please provide either `model_path`, or "
            "`FINE_TUNING_MODEL_PATH` or `FINE_TUNING_JOB_NAME` under `model_metadata`."
        )

    def _build_for_jumpstart(self):
        """Placeholder docstring"""
        if hasattr(self, "pysdk_model") and self.pysdk_model is not None:
            return self.pysdk_model

        # we do not pickle for jumpstart. set to none
        self.secret_key = None

        pysdk_model = self._create_pre_trained_js_model()
        image_uri = pysdk_model.image_uri

        logger.info("JumpStart ID %s is packaged with Image URI: %s", self.model, image_uri)

        if self._is_fine_tuned_model():
            self.is_fine_tuned = True
            pysdk_model = self._update_model_data_for_fine_tuned_model(pysdk_model)

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

        if self.role_arn:
            self.pysdk_model.role = self.role_arn
        if self.sagemaker_session:
            self.pysdk_model.sagemaker_session = self.sagemaker_session
        return self.pysdk_model

    def _optimize_for_jumpstart(
        self,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        tags: Optional[Tags] = None,
        job_name: Optional[str] = None,
        accept_eula: Optional[bool] = None,
        quantization_config: Optional[Dict] = None,
        compilation_config: Optional[Dict] = None,
        speculative_decoding_config: Optional[Dict] = None,
        env_vars: Optional[Dict] = None,
        vpc_config: Optional[Dict] = None,
        kms_key: Optional[str] = None,
        max_runtime_in_sec: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Runs a model optimization job.

        Args:
            output_path (Optional[str]): Specifies where to store the compiled/quantized model.
            instance_type (str): Target deployment instance type that the model is optimized for.
            tags (Optional[Tags]): Tags for labeling a model optimization job. Defaults to ``None``.
            job_name (Optional[str]): The name of the model optimization job. Defaults to ``None``.
            accept_eula (bool): For models that require a Model Access Config, specify True or
                False to indicate whether model terms of use have been accepted.
                The `accept_eula` value must be explicitly defined as `True` in order to
                accept the end-user license agreement (EULA) that some
                models require. (Default: None).
            quantization_config (Optional[Dict]): Quantization configuration. Defaults to ``None``.
            compilation_config (Optional[Dict]): Compilation configuration. Defaults to ``None``.
            speculative_decoding_config (Optional[Dict]): Speculative decoding configuration.
                Defaults to ``None``
            env_vars (Optional[Dict]): Additional environment variables to run the optimization
                container. Defaults to ``None``.
            vpc_config (Optional[Dict]): The VpcConfig set on the model. Defaults to ``None``.
            kms_key (Optional[str]): KMS key ARN used to encrypt the model artifacts when uploading
                to S3. Defaults to ``None``.
            max_runtime_in_sec (Optional[int]): Maximum job execution time in seconds. Defaults to
                ``None``.

        Returns:
            Dict[str, Any]: Model optimization job input arguments.
        """
        if self._is_gated_model() and accept_eula is not True:
            raise ValueError(
                f"Model '{self.model}' requires accepting end-user license agreement (EULA)."
            )

        is_compilation = (not quantization_config) and (
            (compilation_config is not None) or _is_inferentia_or_trainium(instance_type)
        )

        pysdk_model_env_vars = dict()
        if is_compilation:
            pysdk_model_env_vars = self._get_neuron_model_env_vars(instance_type)

        optimization_config, override_env = _extract_optimization_config_and_env(
            quantization_config, compilation_config
        )
        if not optimization_config and is_compilation:
            override_env = override_env or pysdk_model_env_vars
            optimization_config = {
                "ModelCompilationConfig": {
                    "OverrideEnvironment": override_env,
                }
            }

        if speculative_decoding_config:
            self._set_additional_model_source(speculative_decoding_config)
        else:
            deployment_config = self._find_compatible_deployment_config(None)
            if deployment_config:
                self.pysdk_model.set_deployment_config(
                    config_name=deployment_config.get("DeploymentConfigName"),
                    instance_type=deployment_config.get("InstanceType"),
                )
                pysdk_model_env_vars = self.pysdk_model.env

        model_source = _generate_model_source(self.pysdk_model.model_data, accept_eula)
        optimization_env_vars = _update_environment_variables(pysdk_model_env_vars, env_vars)

        output_config = {"S3OutputLocation": output_path}
        if kms_key:
            output_config["KmsKeyId"] = kms_key

        deployment_config_instance_type = (
            self.pysdk_model.deployment_config.get("DeploymentArgs", {}).get("InstanceType")
            if self.pysdk_model.deployment_config
            else None
        )
        self.instance_type = instance_type or deployment_config_instance_type or _get_nb_instance()

        create_optimization_job_args = {
            "OptimizationJobName": job_name,
            "ModelSource": model_source,
            "DeploymentInstanceType": self.instance_type,
            "OptimizationConfigs": [optimization_config],
            "OutputConfig": output_config,
            "RoleArn": self.role_arn,
        }

        if optimization_env_vars:
            create_optimization_job_args["OptimizationEnvironment"] = optimization_env_vars
        if max_runtime_in_sec:
            create_optimization_job_args["StoppingCondition"] = {
                "MaxRuntimeInSeconds": max_runtime_in_sec
            }
        if tags:
            create_optimization_job_args["Tags"] = tags
        if vpc_config:
            create_optimization_job_args["VpcConfig"] = vpc_config

        if accept_eula:
            self.pysdk_model.accept_eula = accept_eula
            if isinstance(self.pysdk_model.model_data, dict):
                self.pysdk_model.model_data["S3DataSource"]["ModelAccessConfig"] = {
                    "AcceptEula": True
                }

        optimization_env_vars = _update_environment_variables(optimization_env_vars, override_env)
        if optimization_env_vars:
            self.pysdk_model.env.update(optimization_env_vars)
        if quantization_config or is_compilation:
            return create_optimization_job_args
        return None

    def _is_gated_model(self, model=None) -> bool:
        """Determine if ``this`` Model is Gated

        Args:
            model (Model): Jumpstart Model
        Returns:
            bool: ``True`` if ``this`` Model is Gated
        """
        s3_uri = model.model_data if model else self.pysdk_model.model_data
        if isinstance(s3_uri, dict):
            s3_uri = s3_uri.get("S3DataSource").get("S3Uri")

        if s3_uri is None:
            return False
        return "private" in s3_uri

    def _set_additional_model_source(
        self,
        speculative_decoding_config: Optional[Dict[str, Any]] = None,
        accept_eula: Optional[bool] = None,
    ) -> None:
        """Set Additional Model Source to ``this`` model.

        Args:
            speculative_decoding_config (Optional[Dict[str, Any]]): Speculative decoding config.
            accept_eula (Optional[bool]): For models that require a Model Access Config.
        """
        if speculative_decoding_config:
            model_provider = _extract_speculative_draft_model_provider(speculative_decoding_config)
            channel_name = _generate_channel_name(self.pysdk_model.additional_model_data_sources)

            if model_provider == "sagemaker":
                additional_model_data_sources = (
                    self.pysdk_model.deployment_config.get("DeploymentArgs", {}).get(
                        "AdditionalDataSources"
                    )
                    if self.pysdk_model.deployment_config
                    else None
                )
                if additional_model_data_sources is None:
                    deployment_config = self._find_compatible_deployment_config(
                        speculative_decoding_config
                    )
                    if deployment_config:
                        self.pysdk_model.set_deployment_config(
                            config_name=deployment_config.get("DeploymentConfigName"),
                            instance_type=deployment_config.get("InstanceType"),
                        )
                    else:
                        raise ValueError(
                            "Cannot find deployment config compatible for optimization job."
                        )

                self.pysdk_model.env.update(
                    {"OPTION_SPECULATIVE_DRAFT_MODEL": f"{SPECULATIVE_DRAFT_MODEL}/{channel_name}"}
                )
                self.pysdk_model.add_tags(
                    {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "sagemaker"},
                )
            else:
                self.pysdk_model = _custom_speculative_decoding(
                    self.pysdk_model, speculative_decoding_config, accept_eula
                )

    def _find_compatible_deployment_config(
        self, speculative_decoding_config: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Finds compatible model deployment config for optimization job.

        Args:
            speculative_decoding_config (Optional[Dict]): Speculative decoding config.

        Returns:
            Optional[Dict[str, Any]]: A compatible model deployment config for optimization job.
        """
        model_provider = _extract_speculative_draft_model_provider(speculative_decoding_config)
        for deployment_config in self.pysdk_model.list_deployment_configs():
            image_uri = deployment_config.get("deployment_config", {}).get("ImageUri")

            if _is_image_compatible_with_optimization_job(image_uri):
                if (
                    model_provider == "sagemaker"
                    and deployment_config.get("DeploymentArgs", {}).get("AdditionalDataSources")
                ) or model_provider == "custom":
                    return deployment_config

        # There's no matching config from jumpstart to add sagemaker draft model location
        if model_provider == "sagemaker":
            return None

        # fall back to the default jumpstart model deployment config for optimization job
        return self.pysdk_model.deployment_config

    def _get_neuron_model_env_vars(
        self, instance_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Gets Neuron model env vars.

        Args:
            instance_type (Optional[str]): Instance type.

        Returns:
            Optional[Dict[str, Any]]: Neuron Model environment variables.
        """
        metadata_configs = self.pysdk_model._metadata_configs
        if metadata_configs:
            metadata_config = metadata_configs.get(self.pysdk_model.config_name)
            resolve_config = metadata_config.resolved_config if metadata_config else None
            if resolve_config and instance_type not in resolve_config.get(
                "supported_inference_instance_types", []
            ):
                neuro_model_id = resolve_config.get("hosting_neuron_model_id")
                neuro_model_version = resolve_config.get("hosting_neuron_model_version", "*")
                if neuro_model_id:
                    job_model = JumpStartModel(
                        neuro_model_id,
                        model_version=neuro_model_version,
                        vpc_config=self.vpc_config,
                    )
                    return job_model.env
        return None
