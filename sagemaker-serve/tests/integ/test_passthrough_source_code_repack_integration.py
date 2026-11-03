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
"""Integration test for ModelBuilder passthrough source_code repack.

Covers the fix where an image_uri build with a model artifact and custom
source_code repacks the code into the artifact (instead of silently dropping
it). This calls build() only (no deploy) so it runs in seconds.
"""

from __future__ import absolute_import

import io
import os
import tarfile
import tempfile
import uuid

import boto3

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.training.configs import SourceCode
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession

MODEL_NAME_PREFIX = "mb-passthrough-repack"


def _upload_raw_artifact(s3_client, bucket, prefix):
    """Upload a minimal model.tar.gz containing only a model file (no code/)."""
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "model.json"), "w") as f:
        f.write('{"weights": [1, 1, 1, 1]}')
    tar_path = os.path.join(tmp, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as t:
        t.add(os.path.join(tmp, "model.json"), arcname="model.json")
    key = f"{prefix}/model.tar.gz"
    s3_client.upload_file(tar_path, bucket, key)
    return f"s3://{bucket}/{key}", key


def _make_source_code_dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "inference.py"), "w") as f:
        f.write(
            "def model_fn(model_dir):\n    return None\n"
            "def predict_fn(data, model):\n    return data\n"
        )
    with open(os.path.join(d, "requirements.txt"), "w") as f:
        f.write("joblib\n")
    return d


def _tar_members(s3_client, s3_uri):
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return tarfile.open(fileobj=io.BytesIO(body), mode="r:gz").getnames()


def test_build_repacks_source_code_into_artifact():
    """build() with image_uri + model artifact + source_code repacks code/ into
    the model.tar.gz. No deploy - runs in seconds."""
    session = Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    role = get_execution_role(sagemaker_session=session)
    s3_client = boto3.client("s3", region_name=region)

    unique_id = uuid.uuid4().hex[:8]
    prefix = f"{MODEL_NAME_PREFIX}/{unique_id}"
    s3_keys = []
    core_model = None

    try:
        from sagemaker.core import image_uris

        image_uri = image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            instance_type="ml.m5.large",
            image_scope="inference",
        )

        artifact_uri, key = _upload_raw_artifact(s3_client, bucket, prefix)
        s3_keys.append(key)
        src_dir = _make_source_code_dir()

        model_builder = ModelBuilder(
            image_uri=image_uri,
            source_code=SourceCode(source_dir=src_dir, entry_script="inference.py"),
            s3_model_data_url=artifact_uri,
            role_arn=role,
            instance_type="ml.m5.large",
            sagemaker_session=session,
        )
        model_builder.model_path = f"/tmp/sagemaker/model-builder/{unique_id}"
        os.makedirs(model_builder.model_path, exist_ok=True)
        model_builder.dependencies = []

        core_model = model_builder.build(
            model_name=f"{MODEL_NAME_PREFIX}-{unique_id}", mode=Mode.SAGEMAKER_ENDPOINT
        )

        # A repack must have produced a new artifact (not the raw one)
        repacked = model_builder.repacked_model_data
        assert repacked is not None
        assert repacked != artifact_uri
        s3_keys.append(repacked.partition(f"s3://{bucket}/")[2])

        # The repacked artifact must contain the inference code under code/
        members = _tar_members(s3_client, repacked)
        assert any(m.endswith("code/inference.py") for m in members), members
        assert any(m.endswith("model.json") for m in members), members

        # Script-mode env vars must be wired up on the model container
        env = core_model.primary_container.environment or {}
        assert env.get("SAGEMAKER_PROGRAM") == "inference.py"
        assert env.get("SAGEMAKER_SUBMIT_DIRECTORY") == "/opt/ml/model/code"

    finally:
        if core_model is not None:
            try:
                core_model.delete()
            except Exception:  # noqa
                pass
        for k in s3_keys:
            try:
                s3_client.delete_object(Bucket=bucket, Key=k)
            except Exception:  # noqa
                pass


def test_build_preserves_source_code_with_pipeline_variable_model_uri():
    """build() under a PipelineSession with a pipeline-variable model artifact must
    keep source_code and defer the repack to pipeline execution time.

    Regression for issue #6450. The artifact is produced by an upstream step, so
    the URI is a PipelineVariable rather than a concrete ``s3://`` string; the
    passthrough path used to drop source_code for that case. No deploy, no
    pipeline execution, and no eager repack upload - runs in seconds.

    Note: generating the ``_RepackModelStep`` inside ``ModelStep`` from this
    recorded request is covered separately by PR #6338.
    """
    session = Session()
    region = session.boto_region_name
    role = get_execution_role(sagemaker_session=session)
    pipeline_session = PipelineSession(boto_session=boto3.Session(region_name=region))

    unique_id = uuid.uuid4().hex[:8]

    from sagemaker.core import image_uris

    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        instance_type="ml.m5.large",
        image_scope="inference",
    )

    src_dir = _make_source_code_dir()
    model_data = ParameterString(
        name="ModelDataUrl", default_value=f"s3://{session.default_bucket()}/model.tar.gz"
    )

    model_builder = ModelBuilder(
        image_uri=image_uri,
        source_code=SourceCode(source_dir=src_dir, entry_script="inference.py"),
        s3_model_data_url=model_data,
        role_arn=role,
        instance_type="ml.m5.large",
        sagemaker_session=pipeline_session,
    )
    model_builder.model_path = f"/tmp/sagemaker/model-builder/{unique_id}"
    os.makedirs(model_builder.model_path, exist_ok=True)
    model_builder.dependencies = []

    step_args = model_builder.build(
        model_name=f"{MODEL_NAME_PREFIX}-pv-{unique_id}", mode=Mode.SAGEMAKER_ENDPOINT
    )

    # source_code must survive the passthrough path
    assert model_builder.entry_point == "inference.py"
    assert model_builder.source_dir == src_dir

    # the repack is deferred to pipeline execution, so no new artifact is uploaded now
    assert model_builder.repacked_model_data is model_data
    assert step_args.need_runtime_repack == {id(model_builder)}
    assert step_args.runtime_repack_output_prefix is not None

    # script-mode env vars must be wired up on the recorded container definition
    env = step_args.create_model_request["PrimaryContainer"]["Environment"]
    assert env.get("SAGEMAKER_PROGRAM") == "inference.py"
    assert env.get("SAGEMAKER_SUBMIT_DIRECTORY") == "/opt/ml/model/code"
