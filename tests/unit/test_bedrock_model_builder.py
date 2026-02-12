import io
import json
import sys
import os
from types import SimpleNamespace

import pytest

# Ensure package sources are importable in tests (adds local `src` directories)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sagemaker-serve", "src"))
sys.path.insert(0, os.path.join(ROOT, "sagemaker-core", "src"))
import types
import importlib.util

# Provide minimal sagemaker core/train/telemetry stubs to avoid heavy imports
if 'sagemaker.core' not in sys.modules:
    # core package
    sagemaker_mod = types.ModuleType('sagemaker')
    core_mod = types.ModuleType('sagemaker.core')
    helper_mod = types.ModuleType('sagemaker.core.helper')
    session_helper_mod = types.ModuleType('sagemaker.core.helper.session_helper')
    # simple Session stub
    class Session:
        def __init__(self, boto_session=None):
            from types import SimpleNamespace
            self.boto_session = SimpleNamespace(client=lambda svc: None)
    session_helper_mod.Session = Session
    resources_mod = types.ModuleType('sagemaker.core.resources')
    class TrainingJob:
        pass
    class ModelPackage:
        @classmethod
        def get(cls, arn):
            return None
    resources_mod.TrainingJob = TrainingJob
    resources_mod.ModelPackage = ModelPackage
    telemetry_mod = types.ModuleType('sagemaker.core.telemetry')
    telemetry_logging_mod = types.ModuleType('sagemaker.core.telemetry.telemetry_logging')
    def _telemetry_emitter(feature=None, func_name=None):
        def deco(f):
            return f
        return deco
    telemetry_logging_mod._telemetry_emitter = _telemetry_emitter
    telemetry_constants_mod = types.ModuleType('sagemaker.core.telemetry.constants')
    class Feature:
        MODEL_CUSTOMIZATION = 'MODEL_CUSTOMIZATION'
    telemetry_constants_mod.Feature = Feature

    sys.modules['sagemaker'] = types.ModuleType('sagemaker')
    sys.modules['sagemaker.core'] = core_mod
    sys.modules['sagemaker.core.helper'] = helper_mod
    sys.modules['sagemaker.core.helper.session_helper'] = session_helper_mod
    sys.modules['sagemaker.core.resources'] = resources_mod
    sys.modules['sagemaker.core.telemetry'] = telemetry_mod
    sys.modules['sagemaker.core.telemetry.telemetry_logging'] = telemetry_logging_mod
    sys.modules['sagemaker.core.telemetry.constants'] = telemetry_constants_mod
    # train module stub
    train_mod = types.ModuleType('sagemaker.train')
    model_trainer_mod = types.ModuleType('sagemaker.train.model_trainer')
    class ModelTrainer:
        pass
    model_trainer_mod.ModelTrainer = ModelTrainer
    sys.modules['sagemaker.train'] = train_mod
    sys.modules['sagemaker.train.model_trainer'] = model_trainer_mod

# Provide minimal botocore.exceptions.ClientError for environments without botocore
if 'botocore' not in sys.modules:
    botocore_mod = types.ModuleType('botocore')
    class _ClientError(Exception):
        def __init__(self, error_response, operation_name=None):
            self.response = error_response
            super().__init__(str(error_response))
    exceptions_mod = types.SimpleNamespace(ClientError=_ClientError)
    botocore_mod.exceptions = exceptions_mod
    sys.modules['botocore'] = botocore_mod
    sys.modules['botocore.exceptions'] = exceptions_mod

# Load bedrock_model_builder module directly from source to avoid importing the
# package-level __init__ which pulls in many heavy dependencies.
module_path = os.path.join(ROOT, 'sagemaker-serve', 'src', 'sagemaker', 'serve', 'bedrock_model_builder.py')
spec = importlib.util.spec_from_file_location('bb_module', module_path)
bb_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb_module)
BedrockModelBuilder = bb_module.BedrockModelBuilder


class FakeTrainingJob:
    def __init__(self, s3_uri: str):
        self.model_artifacts = SimpleNamespace(s3_model_artifacts=s3_uri)
        self.output_model_package_arn = None


def make_s3_client_success(checkpoint_value: str):
    class S3Stub:
        def get_object(self, Bucket, Key):
            body = io.BytesIO(json.dumps({"checkpoint_s3_bucket": checkpoint_value}).encode('utf-8'))
            return {"Body": body}

    return S3Stub()


def make_s3_client_not_found():
    class S3Stub:
        def get_object(self, Bucket, Key):
            error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
            raise bb_module.ClientError(error_response, 'GetObject')

    return S3Stub()


def test_get_checkpoint_uri_success(monkeypatch):
    s3_uri = 's3://mybucket/path/output/model.tar.gz'
    fake_job = FakeTrainingJob(s3_uri)


    # Monkeypatch TrainingJob class to allow isinstance check
    bb_module.TrainingJob = FakeTrainingJob

    builder = BedrockModelBuilder(model=fake_job)
    builder.boto_session = SimpleNamespace(client=lambda service: make_s3_client_success('s3://checkpoint-bucket/checkpoint'))

    uri = builder._get_checkpoint_uri_from_manifest()
    assert uri == 's3://checkpoint-bucket/checkpoint'


def test_get_checkpoint_uri_manifest_not_found(monkeypatch):
    s3_uri = 's3://mybucket/path/output/model.tar.gz'
    fake_job = FakeTrainingJob(s3_uri)


    bb_module.TrainingJob = FakeTrainingJob

    builder = BedrockModelBuilder(model=fake_job)
    builder.boto_session = SimpleNamespace(client=lambda service: make_s3_client_not_found())

    with pytest.raises(ValueError):
        builder._get_checkpoint_uri_from_manifest()
