"""Placeholder docstring"""

from __future__ import absolute_import
import os
import logging
import platform
from abc import abstractmethod
from pathlib import Path
import sys
from typing import Type
import shutil
import subprocess

from sagemaker import Session
from sagemaker.model import Model
from sagemaker.base_predictor import Predictor
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.detector.image_detector import _detect_framework_and_version, _get_model_base
from sagemaker.serve.utils.local_hardware import _get_available_gpus
from sagemaker.base_serializers import JSONSerializer
from sagemaker.base_deserializers import JSONDeserializer
from sagemaker.serve.detector.pickler import save_pkl
from sagemaker.serve.model_server.triton.config_template import CONFIG_TEMPLATE
from sagemaker.serve.validations.check_integrity import (
    generate_secret_key,
    compute_hash,
)

from sagemaker.remote_function.core.serialization import _MetaData


logger = logging.getLogger(__name__)

SUPPORTED_TRITON_MODE = {Mode.LOCAL_CONTAINER, Mode.SAGEMAKER_ENDPOINT}
SUPPORTED_TRITON_FRAMEWORK = {"pytorch", "tensorflow"}
INPUT_NAME = "input_1"
OUTPUT_NAME = "output_1"

TRITON_IMAGE_ACCOUNT_ID_MAP = {
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "eu-north-1": "601324751636",
    "eu-south-1": "966458181534",
    "eu-central-1": "746233611703",
    "ap-east-1": "110948597952",
    "ap-south-1": "763008648453",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "cn-northwest-1": "474822919863",
    "cn-north-1": "472730292857",
    "sa-east-1": "756306329178",
    "ca-central-1": "464438896020",
    "me-south-1": "836785723513",
    "af-south-1": "774647643957",
}

GPU_INSTANCE_FAMILIES = {
    "ml.g4dn",
    "ml.g5",
    "ml.p3",
    "ml.p3dn",
    "ml.p4",
    "ml.p4d",
    "ml.p4de",
    "local_gpu",
}

TRITON_IMAGE_BASE = "{account_id}.dkr.ecr.{region}.{base}/sagemaker-tritonserver:{version}-py3"
# As suggested by container team, we should always try to use latest version and update periodically
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md#nvidia-triton-inference-containers-sm-support-only
# However, latest version 23.07 seems to have issue loading python modules, so using 23.02 for now
LATEST_VERSION = "23.02"
# SageMaker Triton Inference Container does not support Tensorflow1 as of version 23.05 onwards
VERSION_FOR_TF1 = "23.02"


class TritonSerializer(JSONSerializer):
    """A wrapper of JSONSerializer because Triton expects input to be certain format"""

    def __init__(self, input_serializer, dtype: str, content_type="application/json"):
        """Placeholder docstring"""
        super().__init__(content_type)
        self.input_serializer = input_serializer
        self.dtype = dtype

    def serialize(self, data):
        """Placeholder docstring"""
        numpy_data = self.input_serializer.serialize(data)
        payload = {
            "inputs": [
                {
                    "name": INPUT_NAME,
                    "shape": numpy_data.shape,
                    "datatype": self.dtype,
                    "data": numpy_data.tolist(),
                }
            ]
        }

        return super().serialize(payload)


class Triton:
    """Triton build logic for model builder"""

    @abstractmethod
    def _prepare_for_mode(self):
        """Placeholder docstring"""

    def _validate_for_triton(
        self,
    ):
        """Validation for triton, expand this as we include more backend support with more framework"""
        try:
            import tritonclient.http as httpClient

            httpClient.__class__
        except ModuleNotFoundError:
            raise ImportError(
                (
                    "Launching Triton with ModelBuilder requires tritonClient[http] module. "
                    "but it was not found in your environemnt. "
                    "Checkout the instructions on the installation page of its repo: "
                    "https://github.com/triton-inference-server/client#getting-the-client-libraries-and-examples "
                    "And follow the ones that match your environment."
                    "Please note that you may need to restart your runtime after installation."
                )
            )

        if (
            self.mode == Mode.LOCAL_CONTAINER
            and not _has_nvidia_gpu()
            and self.image_uri
            and "cpu" not in self.image_uri
        ):
            # When customer does not have Nvidia GPU but tries to launch Triton in GPU mode in LOCAL_CONTAINER mode
            raise ValueError(
                (
                    "Your device does not have a Nvidia GPU. "
                    "Ubable to launch Triton container in GPU mode in your local machine. "
                    "Please provide a CPU version triton image to serve your model in LOCAL_CONTAINER mode. "
                )
            )

        if self.mode not in SUPPORTED_TRITON_MODE:
            raise ValueError("%s mode is not supported with Triton model server." % self.mode)

        # Validate model path
        model_path = Path(self.model_path)
        if not model_path.exists():
            model_path.mkdir(parents=True)
        elif not model_path.is_dir():
            raise Exception("model_path: %s is not a valid directory" % self.model_path)

        # Validate schema builder
        self.schema_builder._update_serializer_deserializer_for_triton()
        self.schema_builder._detect_dtype_for_triton()

        # Check python version - all SageMaker triton image are using python 3.8
        if not platform.python_version().startswith("3.8"):
            logger.warn(
                (
                    "SageMaker Triton image uses python 3.8, your python version: %s. "
                    "It is recommended to use the same python version to avoid incompatibility."
                )
                % platform.python_version()
            )

        if self.model:
            self._framework, self._version = _detect_framework_and_version(
                str(_get_model_base(self.model))
            )

            if self._framework not in SUPPORTED_TRITON_FRAMEWORK:
                raise ValueError("%s is not supported with Triton model server" % self._framework)

        if self.inference_spec:
            if "conda" not in sys.executable.lower():
                raise ValueError(
                    (
                        "Invalid python environment %s, please use anaconda "
                        "or miniconda to manage your python environment "
                        "as it is required by Triton to capture "
                        "and pack your python dependencies."
                    )
                    % sys.executable
                )

    def _prepare_for_triton(self):
        # Prepare directory
        model_path = Path(self.model_path)
        pkl_path = model_path.joinpath("model_repository").joinpath("model")
        if not pkl_path.exists():
            pkl_path.mkdir(parents=True)

        # Copy local model artifacts to triton model dir - excluding files under model_repository
        for root, _, files in os.walk(self.model_path):
            for f in files:
                path_file = os.path.join(root, f)
                if "model_repository" not in path_file:
                    shutil.copy2(path_file, str(pkl_path.joinpath(f)))

        export_path = model_path.joinpath("model_repository").joinpath("model").joinpath("1")
        if not export_path.exists():
            export_path.mkdir(parents=True)

        if self.model:
            self.secret_key = "dummy secret key for onnx backend"

            if self._framework == "pytorch":
                self._export_pytorch_to_onnx(
                    export_path=export_path, model=self.model, schema_builder=self.schema_builder
                )
                return

            if self._framework == "tensorflow":
                self._export_tf_to_onnx(
                    export_path=export_path, model=self.model, schema_builder=self.schema_builder
                )
                return

            raise ValueError("%s is not supported" % self._framework)

        if self.inference_spec:
            triton_model_path = Path(__file__).parent.joinpath("model.py")
            shutil.copy2(str(triton_model_path), str(export_path))

            self._generate_config_pbtxt(pkl_path=pkl_path)

            self._pack_conda_env(pkl_path=pkl_path)

            self._hmac_signing()

            return

        raise ValueError("Either model or inference_spec should be provided to ModelBuilder.")

    def _hmac_signing(self):
        """Perform HMAC signing on picke file for integrity check"""
        secret_key = generate_secret_key()
        pkl_path = Path(self.model_path).joinpath("model_repository").joinpath("model")

        with open(str(pkl_path.joinpath("serve.pkl")), "rb") as f:
            buffer = f.read()
        hash_value = compute_hash(buffer=buffer, secret_key=secret_key)

        with open(str(pkl_path.joinpath("metadata.json")), "wb") as metadata:
            metadata.write(_MetaData(hash_value).to_json())

        self.secret_key = secret_key

    def _generate_config_pbtxt(self, pkl_path: Path):
        config_path = pkl_path.joinpath("config.pbtxt")

        # get input and output shape
        input_shape = list(self.schema_builder._sample_input_ndarray.shape)
        output_shape = list(self.schema_builder._sample_output_ndarray.shape)
        input_shape[0] = -1
        output_shape[0] = -1

        config_content = CONFIG_TEMPLATE.format(
            input_name=INPUT_NAME,
            input_shape=str(input_shape),
            input_dtype=self.schema_builder._input_triton_dtype,
            output_name=OUTPUT_NAME,
            output_dtype=self.schema_builder._output_triton_dtype,
            output_shape=str(output_shape),
            hardware_type="KIND_CPU" if "-cpu" in self.image_uri else "KIND_GPU",
        )

        with open(str(config_path), "w") as f:
            f.write(config_content)

    def _pack_conda_env(self, pkl_path: Path):
        # Verify that conda-pack exists in customer's env
        # pylint: disable=no-member, attribute-defined-outside-init
        try:
            import conda_pack

            conda_pack.__version__
        except ModuleNotFoundError:
            raise ImportError(
                (
                    "Launching Triton with ModelBuilder requires conda_pack library "
                    "but it was not found in your environemnt. "
                    "Checkout the instructions on the installation page of its repo: "
                    "https://conda.github.io/conda-pack/ "
                    "And follow the ones that match your environment."
                    "Please note that you may need to restart your runtime after installation."
                )
            )

        script_path = Path(__file__).parent.joinpath("pack_conda_env.sh")
        env_tar_path = pkl_path.joinpath("triton_env.tar.gz")
        conda_env_name = os.getenv("CONDA_DEFAULT_ENV")

        # clone current env to triton_env
        subprocess.run(["bash", str(script_path), conda_env_name, str(env_tar_path)])

    def _export_tf_to_onnx(self, export_path: str, model: object, schema_builder: SchemaBuilder):
        try:
            import tensorflow as tf
            import tf2onnx

            tf2onnx.convert.from_keras(
                model=model,
                input_signature=[
                    tf.TensorSpec(shape=schema_builder.sample_input.shape, name=INPUT_NAME)
                ],
                output_path=str(export_path.joinpath("model.onnx")),
            )

        except ModuleNotFoundError:
            raise ImportError(
                (
                    "Launching Triton with ModelBuilder for a Tensorflow model requires tf2onnx module. "
                    "but it was not found in your environemnt. "
                    "Checkout the instructions on the installation page of its repo: "
                    "https://onnxruntime.ai/docs/install/ "
                    "And follow the ones that match your environment."
                    "Please note that you may need to restart your runtime after installation."
                )
            )

    def _export_pytorch_to_onnx(
        self, model: object, export_path: Path, schema_builder: SchemaBuilder
    ):
        """Export pytorch model object into onnx format"""
        logger.info("Converting pytorch model into onnx format")
        try:
            from torch.onnx import export

            export(
                model=model,
                args=schema_builder.sample_input,
                f=str(export_path.joinpath("model.onnx")),
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                verbose=False,
            )

        except ModuleNotFoundError:
            raise ImportError(
                (
                    "Launching Triton with ModelBuilder for a PyTorch model requires onnx module. "
                    "but it was not found in your environemnt. "
                    "Checkout the instructions on the installation page of its repo: "
                    "https://onnxruntime.ai/docs/install/ "
                    "And follow the ones that match your environment."
                    "Please note that you may need to restart your runtime after installation."
                )
            )

    def _auto_detect_image_for_triton(self):
        """Detect image of triton given framework, version and region.

        If InferenceSpec is provided, then default to latest version.
        """
        # This is a temporary solution.
        # TODO: migrate to image_uris.retrieve() once it starts to support Triton

        if self.image_uri:
            logger.info("Skipping auto detection as the image uri is provided %s", self.image_uri)
            return

        logger.info(
            "Auto detect container url for the provided model and on instance %s",
            self.instance_type,
        )

        region = self.sagemaker_session.boto_region_name

        if region not in TRITON_IMAGE_ACCOUNT_ID_MAP.keys():
            raise ValueError(
                "%s is not supported for triton image. Please switch to the following region: %s"
                % (region, TRITON_IMAGE_ACCOUNT_ID_MAP.keys())
            )

        base = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"

        if (
            not self.inference_spec
            and self._framework == "tensorflow"
            and self._version.startswith("1")
        ):
            self.image_uri = TRITON_IMAGE_BASE.format(
                account_id=TRITON_IMAGE_ACCOUNT_ID_MAP.get(region),
                region=region,
                base=base,
                version=VERSION_FOR_TF1,
            )
        else:
            self.image_uri = TRITON_IMAGE_BASE.format(
                account_id=TRITON_IMAGE_ACCOUNT_ID_MAP.get(region),
                region=region,
                base=base,
                version=LATEST_VERSION,
            )

        if not _is_gpu_instance(self.instance_type):
            self.image_uri += "-cpu"

        logger.info("Autodetected image: %s. Proceeding with the the deployment." % self.image_uri)
        return

    def _create_triton_model(self) -> Type[Model]:
        self.pysdk_model = Model(
            image_uri=self.image_uri,
            image_config=self.image_config,
            vpc_config=self.vpc_config,
            model_data=self.s3_upload_path,
            role=self.serve_settings.role_arn,
            env=self.env_vars,
            sagemaker_session=self.sagemaker_session,
            predictor_cls=self._get_triton_predictor,
        )

        # store the modes in the model so that we may
        # reference the configurations for local deploy() & predict()
        self.pysdk_model.mode = self.mode
        self.pysdk_model.modes = self.modes
        self.pysdk_model.serve_settings = self.serve_settings
        if hasattr(self, "role_arn") and self.role_arn:
            self.pysdk_model.role = self.role_arn
        if hasattr(self, "sagemaker_session") and self.sagemaker_session:
            self.pysdk_model.sagemaker_session = self.sagemaker_session

        # dynamically generate a method to direct model.deploy() logic based on mode
        # unique method to models created via ModelBuilder()
        self._original_deploy = self.pysdk_model.deploy
        self.pysdk_model.deploy = self._model_builder_deploy_wrapper
        self._original_register = self.pysdk_model.register
        self.pysdk_model.register = self._model_builder_register_wrapper
        self.model_package = None
        return self.pysdk_model

    def _get_triton_predictor(self, endpoint_name: str, sagemaker_session: Session) -> Predictor:
        """Placeholder docstring"""
        dtype = self.schema_builder._input_triton_dtype.split("_")[-1]
        serializer, deserializer = (
            TritonSerializer(input_serializer=self.schema_builder.input_serializer, dtype=dtype),
            JSONDeserializer(),
        )
        return Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )

    def _save_inference_spec(self) -> None:
        """Placeholder docstring"""
        if self.inference_spec:
            pkl_path = Path(self.model_path).joinpath("model_repository").joinpath("model")
            save_pkl(pkl_path, (self.inference_spec, self.schema_builder))

        return

    def _build_for_triton(self):
        """Placeholder docstring"""
        self._validate_for_triton()

        self._auto_detect_image_for_triton()

        self._save_inference_spec()

        self._prepare_for_triton()

        self._prepare_for_mode()

        return self._create_triton_model()


def _has_nvidia_gpu() -> bool:
    try:
        _get_available_gpus()
        return True
    except Exception:
        # for nvidia-smi to run, a cuda driver must be present
        logger.info("CUDA not found, launching Triton in CPU mode.")
        return False


def _is_gpu_instance(instance_type: str) -> bool:
    instance_family = instance_type.rsplit(".", 1)[0]
    return instance_family in GPU_INSTANCE_FAMILIES
