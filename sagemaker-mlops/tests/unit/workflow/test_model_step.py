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
"""Unit tests for workflow model_step."""

from __future__ import absolute_import

from unittest.mock import Mock, patch


def test_model_step_properties():
    """Test ModelStep has properties"""
    from sagemaker.mlops.workflow.model_step import ModelStep

    step_args = {"ModelName": "test-model"}

    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        step = ModelStep(name="model-step", step_args=step_args)
        assert step.name == "model-step"
        assert hasattr(step, "properties")


def _pipeline_session():
    from sagemaker.core.workflow.pipeline_context import PipelineSession

    ps = Mock(spec=PipelineSession)
    ps.context = Mock()
    ps.boto_region_name = "us-west-2"
    return ps


class _FakeModelStepArgs:
    """Mimics the _ModelStepArguments produced by ModelBuilder.register() under a
    PipelineSession (a register/create_model_package request that needs a repack)."""

    def __init__(self, model, need_runtime_repack):
        self.model = model
        self.need_runtime_repack = need_runtime_repack
        self.runtime_repack_output_prefix = "s3://bucket/prefix"
        self.create_model_request = None
        self.create_model_package_request = {
            "InferenceSpecification": {"Containers": [{"ModelDataUrl": "s3://orig/model.tar.gz"}]}
        }


def test_model_builder_register_appends_repack_step():
    """GH #5828/#5829: ModelBuilder.register() in a ModelStep must emit a repack step
    (v2 parity) and rewire the container ModelDataUrl to the repacked artifact."""
    from sagemaker.serve.model_builder import ModelBuilder
    from sagemaker.mlops.workflow import model_step as ms

    ps = _pipeline_session()
    builder = Mock(spec=ModelBuilder)
    builder.sagemaker_session = ps
    builder.model_name = "my-model"
    builder.role_arn = "arn:aws:iam::111122223333:role/R"
    builder.s3_model_data_url = "s3://orig/model.tar.gz"
    builder.entry_point = "inference.py"
    builder.source_dir = "/code"
    builder.source_code = Mock(requirements="requirements.txt")
    builder.vpc_config = None

    step_args = _FakeModelStepArgs(builder, {id(builder)})
    fake_repack = Mock()
    fake_repack.properties.ModelArtifacts.S3ModelArtifacts = "s3://repacked/model.tar.gz"

    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        with patch.object(ms, "_RepackModelStep", return_value=fake_repack) as mock_repack:
            step = ms.ModelStep(name="step", step_args=step_args)

    # A repack step was generated (the bug: none was on master).
    assert len(step.steps) == 1
    # ModelBuilder attributes were mapped to the repack step's parameters.
    _, kwargs = mock_repack.call_args
    assert kwargs["role"] == "arn:aws:iam::111122223333:role/R"
    assert kwargs["model_data"] == "s3://orig/model.tar.gz"
    assert kwargs["entry_point"] == "inference.py"
    assert kwargs["source_dir"] == "/code"
    assert kwargs["requirements"] == "requirements.txt"
    assert kwargs["sagemaker_session"] is ps
    # The container now points at the repacked artifact.
    container = step_args.create_model_package_request["InferenceSpecification"]["Containers"][0]
    assert container["ModelDataUrl"] == "s3://repacked/model.tar.gz"


def test_repack_step_gets_output_location_and_kms_key():
    """The repacked artifact must go to ModelBuilder's runtime_repack_output_prefix and be
    encrypted with the model's KMS key. Both reach ModelTrainer via _RepackModelStep's
    **kwargs; dropping them sends the artifact to ModelTrainer's default bucket with no CMK,
    which breaks accounts with a mandated bucket or an SSE-KMS bucket policy."""
    from sagemaker.serve.model_builder import ModelBuilder
    from sagemaker.mlops.workflow import model_step as ms

    ps = _pipeline_session()
    builder = Mock(spec=ModelBuilder)
    builder.sagemaker_session = ps
    builder.model_name = "my-model"
    builder.role_arn = "arn:aws:iam::111122223333:role/R"
    builder.s3_model_data_url = "s3://orig/model.tar.gz"
    builder.entry_point = "inference.py"
    builder.source_dir = "/code"
    builder.source_code = Mock(requirements="requirements.txt")
    builder.vpc_config = None
    builder.model_kms_key = "arn:aws:kms:us-west-2:111122223333:key/abc"

    step_args = _FakeModelStepArgs(builder, {id(builder)})
    fake_repack = Mock()
    fake_repack.properties.ModelArtifacts.S3ModelArtifacts = "s3://repacked/model.tar.gz"

    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        with patch.object(ms, "_RepackModelStep", return_value=fake_repack) as mock_repack:
            ms.ModelStep(name="step", step_args=step_args)

    _, kwargs = mock_repack.call_args
    odc = kwargs["output_data_config"]
    assert odc.s3_output_path == "s3://bucket/prefix"
    assert odc.kms_key_id == "arn:aws:kms:us-west-2:111122223333:key/abc"


def test_repack_step_output_config_respects_user_override():
    """An explicit output_data_config in repack_model_step_settings must win."""
    from sagemaker.serve.model_builder import ModelBuilder
    from sagemaker.core.shapes import OutputDataConfig
    from sagemaker.mlops.workflow import model_step as ms

    ps = _pipeline_session()
    builder = Mock(spec=ModelBuilder)
    builder.sagemaker_session = ps
    builder.model_name = "my-model"
    builder.role_arn = "arn:aws:iam::111122223333:role/R"
    builder.s3_model_data_url = "s3://orig/model.tar.gz"
    builder.entry_point = "inference.py"
    builder.source_dir = "/code"
    builder.source_code = Mock(requirements="requirements.txt")
    builder.vpc_config = None
    builder.model_kms_key = "arn:aws:kms:us-west-2:111122223333:key/abc"

    step_args = _FakeModelStepArgs(builder, {id(builder)})
    fake_repack = Mock()
    fake_repack.properties.ModelArtifacts.S3ModelArtifacts = "s3://repacked/model.tar.gz"
    mine = OutputDataConfig(s3_output_path="s3://mine/out", kms_key_id="my-key")

    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        with patch.object(ms, "_RepackModelStep", return_value=fake_repack) as mock_repack:
            ms.ModelStep(
                name="step",
                step_args=step_args,
                repack_model_step_settings={"output_data_config": mine},
            )

    _, kwargs = mock_repack.call_args
    assert kwargs["output_data_config"] is mine


def test_no_repack_step_for_unrecognized_model_type():
    """An object that is neither a core Model nor a ModelBuilder yields no repack step."""
    from sagemaker.mlops.workflow import model_step as ms

    ps = _pipeline_session()
    unknown = Mock()
    unknown.sagemaker_session = ps
    step_args = _FakeModelStepArgs(unknown, {id(unknown)})

    with patch("sagemaker.core.workflow.utilities.validate_step_args_input"):
        with patch.object(ms, "_RepackModelStep") as mock_repack:
            step = ms.ModelStep(name="step", step_args=step_args)

    assert step.steps == []
    mock_repack.assert_not_called()
