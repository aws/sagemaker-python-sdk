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
from __future__ import absolute_import

import os
from packaging import version
import pytest

from sagemaker.huggingface import HuggingFace
from sagemaker.huggingface import TrainingCompilerConfig as HFTrainingCompilerConfig
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow import TrainingCompilerConfig as TFTrainingCompilerConfig
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch import TrainingCompilerConfig as PTTrainingCompilerConfig

from tests import integ
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout


@pytest.fixture(scope="module")
def gpu_instance_type(request):
    return "ml.p3.2xlarge"


@pytest.fixture(scope="module")
def instance_count(request):
    return 1


@pytest.fixture(scope="module")
def imagenet_val_set(request, sagemaker_session, tmpdir_factory):
    """
    Copies the Imagenet dataset from the bucket it's hosted in to the local bucket in the test region.
    Due to licensing issues, access to this dataset is controlled through an allowlist
    """
    local_path = tmpdir_factory.mktemp("trcomp_imagenet_val_set")
    sagemaker_session.download_data(
        path=local_path,
        bucket="collection-of-ml-datasets",
        key_prefix="Imagenet/TFRecords/validation",
    )
    train_input = sagemaker_session.upload_data(
        path=local_path, key_prefix="integ-test-data/trcomp/tensorflow/imagenet/val"
    )
    return train_input


@pytest.fixture(scope="module")
def huggingface_dummy_dataset(request, sagemaker_session):
    """
    Copies the dataset from the local disk to the local bucket in the test region
    """
    data_path = os.path.join(DATA_DIR, "huggingface")
    train_input = sagemaker_session.upload_data(
        path=os.path.join(data_path, "train"),
        key_prefix="integ-test-data/trcomp/huggingface/dummy/train",
    )
    return train_input


@pytest.fixture(autouse=True)
def skip_if_incompatible(gpu_instance_type, request):
    """
    These tests are for training compiler enabled images/estimators only.
    """
    region = integ.test_region()
    if region not in integ.TRAINING_COMPILER_SUPPORTED_REGIONS:
        pytest.skip("SageMaker Training Compiler is not supported in this region")
    if gpu_instance_type == "ml.p3.16xlarge" and region not in integ.DATA_PARALLEL_TESTING_REGIONS:
        pytest.skip("Data parallel testing is not allowed in this region")
    if gpu_instance_type == "ml.p3.2xlarge" and region in integ.TRAINING_NO_P3_REGIONS:
        pytest.skip("no ml.p3 instances in this region")


@pytest.mark.parametrize(
    "gpu_instance_type,instance_count",
    [
        pytest.param("ml.p3.2xlarge", 1, marks=pytest.mark.release),
        pytest.param("ml.p3.16xlarge", 2),
    ],
)
@pytest.mark.skipif(
    integ.test_region() in integ.TRAINING_NO_P3_REGIONS,
    reason="No P3 instances or low capacity in this region",
)
def test_huggingface_pytorch(
    sagemaker_session,
    gpu_instance_type,
    instance_count,
    huggingface_training_compiler_latest_version,
    huggingface_training_compiler_pytorch_latest_version,
    huggingface_dummy_dataset,
):
    """
    Test the HuggingFace estimator with PyTorch
    """
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "huggingface")

        hf = HuggingFace(
            py_version="py38",
            entry_point=os.path.join(data_path, "run_glue.py"),
            role="SageMakerRole",
            transformers_version=huggingface_training_compiler_latest_version,
            pytorch_version=huggingface_training_compiler_pytorch_latest_version,
            instance_count=instance_count,
            instance_type=gpu_instance_type,
            hyperparameters={
                "model_name_or_path": "distilbert-base-cased",
                "task_name": "wnli",
                "do_train": True,
                "do_eval": True,
                "max_seq_length": 128,
                "fp16": True,
                "per_device_train_batch_size": 128,
                "output_dir": "/opt/ml/model",
            },
            sagemaker_session=sagemaker_session,
            disable_profiler=True,
            compiler_config=HFTrainingCompilerConfig(),
            distribution={"pytorchxla": {"enabled": True}} if instance_count > 1 else None,
        )

        hf.fit(huggingface_dummy_dataset)


@pytest.mark.parametrize(
    "gpu_instance_type,instance_count",
    [
        pytest.param("ml.p3.2xlarge", 1, marks=pytest.mark.release),
        pytest.param("ml.p3.16xlarge", 2),
    ],
)
@pytest.mark.skip("Temporarily skip to unblock")
def test_pytorch(
    sagemaker_session,
    gpu_instance_type,
    instance_count,
    pytorch_training_compiler_latest_version,
    huggingface_dummy_dataset,
):
    """
    Test the PyTorch estimator
    """
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        hf = PyTorch(
            py_version="py39",
            source_dir=os.path.join(DATA_DIR, "huggingface_byoc"),
            entry_point="run_glue.py",
            role="SageMakerRole",
            framework_version=pytorch_training_compiler_latest_version,
            instance_count=instance_count,
            instance_type=gpu_instance_type,
            hyperparameters={
                "model_name_or_path": "distilbert-base-cased",
                "task_name": "wnli",
                "do_train": True,
                "do_eval": True,
                "max_seq_length": 128,
                "fp16": True,
                "per_device_train_batch_size": 128,
                "output_dir": "/opt/ml/model",
            },
            sagemaker_session=sagemaker_session,
            disable_profiler=True,
            compiler_config=PTTrainingCompilerConfig(),
            distribution={"pytorchxla": {"enabled": True}} if instance_count > 1 else None,
        )

        hf.fit(huggingface_dummy_dataset)


@pytest.mark.release
def test_huggingface_tensorflow(
    sagemaker_session,
    gpu_instance_type,
    huggingface_training_compiler_latest_version,
    huggingface_training_compiler_tensorflow_latest_version,
    huggingface_dummy_dataset,
):
    """
    Test the HuggingFace estimator with TensorFlow
    """
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "huggingface")

        hf = HuggingFace(
            py_version="py38",
            entry_point=os.path.join(data_path, "run_tf.py"),
            role="SageMakerRole",
            transformers_version=huggingface_training_compiler_latest_version,
            tensorflow_version=huggingface_training_compiler_tensorflow_latest_version,
            instance_count=1,
            instance_type=gpu_instance_type,
            hyperparameters={
                "model_name_or_path": "distilbert-base-cased",
                "per_device_train_batch_size": 128,
                "per_device_eval_batch_size": 128,
                "output_dir": "/opt/ml/model",
                "overwrite_output_dir": True,
                "save_steps": 5500,
            },
            sagemaker_session=sagemaker_session,
            disable_profiler=True,
            compiler_config=HFTrainingCompilerConfig(),
        )

        hf.fit(huggingface_dummy_dataset)


@pytest.mark.release
def test_tensorflow(
    sagemaker_session,
    gpu_instance_type,
    tensorflow_training_latest_version,
    imagenet_val_set,
):
    """
    Test the TensorFlow estimator
    """
    if version.parse(tensorflow_training_latest_version) >= version.parse("2.12") or version.parse(
        tensorflow_training_latest_version
    ) < version.parse("2.9"):
        pytest.skip("Training Compiler only supports TF >= 2.9 and < 2.12")
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        epochs = 10
        batch = 256
        train_steps = int(10240 * epochs / batch)
        steps_per_loop = train_steps // 10
        overrides = (
            f"runtime.enable_xla=True,"
            f"runtime.num_gpus=1,"
            f"runtime.distribution_strategy=one_device,"
            f"runtime.mixed_precision_dtype=float16,"
            f"task.train_data.global_batch_size={batch},"
            f"task.train_data.input_path=/opt/ml/input/data/training/validation*,"
            f"task.train_data.cache=False,"
            f"trainer.train_steps={train_steps},"
            f"trainer.steps_per_loop={steps_per_loop},"
            f"trainer.summary_interval={steps_per_loop},"
            f"trainer.checkpoint_interval={train_steps},"
            f"task.model.backbone.type=resnet,"
            f"task.model.backbone.resnet.model_id=50"
        )
        tf = TensorFlow(
            py_version="py39",
            git_config={
                "repo": "https://github.com/tensorflow/models.git",
                "branch": "v" + ".".join(tensorflow_training_latest_version.split(".")[:2]) + ".0",
            },
            source_dir=".",
            entry_point="official/vision/train.py",
            model_dir=False,
            role="SageMakerRole",
            framework_version=tensorflow_training_latest_version,
            instance_count=1,
            instance_type=gpu_instance_type,
            hyperparameters={
                "experiment": "resnet_imagenet",
                "config_file": "official/vision/configs/experiments/image_classification/imagenet_resnet50_gpu.yaml",
                "mode": "train",
                "model_dir": "/opt/ml/model",
                "params_override": overrides,
            },
            sagemaker_session=sagemaker_session,
            disable_profiler=True,
            compiler_config=TFTrainingCompilerConfig(),
        )

        tf.fit(inputs=imagenet_val_set, logs=True, wait=True)
