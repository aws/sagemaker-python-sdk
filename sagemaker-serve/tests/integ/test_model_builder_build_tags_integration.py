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
"""Integration test for tagging the Model that ``ModelBuilder.build()`` creates.

Regression for issue #6501. ``build()`` took no ``tags`` argument, so every model
built without deploying -- a pipeline ``ModelStep``, batch transform, evaluation,
an inference pipeline -- was created untagged. Unit tests can only assert the
request the SDK sends; this reads the tags back off the created Model with
``ListTags`` to prove they reached the service. build() only (no deploy), so it
runs in seconds.
"""

from __future__ import absolute_import

import os
import tarfile
import tempfile
import uuid

import boto3

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes import Tag
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder import ModelBuilder

MODEL_NAME_PREFIX = "mb-build-tags"


def _upload_artifact(s3_client, bucket, prefix):
    """Upload a minimal model.tar.gz and return its URI and key."""
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.json")
    with open(model_file, "w") as handle:
        handle.write('{"weights": [1, 1, 1, 1]}')
    tar_path = os.path.join(tmp, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(model_file, arcname="model.json")
    key = "{}/model.tar.gz".format(prefix)
    s3_client.upload_file(tar_path, bucket, key)
    return "s3://{}/{}".format(bucket, key), key


def _model_builder(session, image_uri, artifact_uri, role, unique_id):
    builder = ModelBuilder(
        image_uri=image_uri,
        s3_model_data_url=artifact_uri,
        role_arn=role,
        instance_type="ml.m5.large",
        sagemaker_session=session,
    )
    builder.model_path = "/tmp/sagemaker/model-builder/{}".format(unique_id)
    os.makedirs(builder.model_path, exist_ok=True)
    builder.dependencies = []
    return builder


def _tags_on(sagemaker_client, model_arn):
    return {
        tag["Key"]: tag["Value"]
        for tag in sagemaker_client.list_tags(ResourceArn=model_arn)["Tags"]
    }


def test_build_tags_reach_the_created_model():
    """build(tags=...) must tag the Model, for both tag representations."""
    session = Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    role = get_execution_role(sagemaker_session=session)
    s3_client = boto3.client("s3", region_name=region)
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    unique_id = uuid.uuid4().hex[:8]
    prefix = "{}/{}".format(MODEL_NAME_PREFIX, unique_id)
    s3_keys = []
    created_models = []

    try:
        from sagemaker.core import image_uris

        image_uri = image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            instance_type="ml.m5.large",
            image_scope="inference",
        )
        artifact_uri, key = _upload_artifact(s3_client, bucket, prefix)
        s3_keys.append(key)

        # Boto-style tags.
        boto_model = _model_builder(session, image_uri, artifact_uri, role, unique_id).build(
            model_name="{}-boto-{}".format(MODEL_NAME_PREFIX, unique_id),
            mode=Mode.SAGEMAKER_ENDPOINT,
            tags=[{"Key": "project", "Value": "build-tags"}],
        )
        created_models.append(boto_model)
        assert _tags_on(sagemaker_client, boto_model.model_arn)["project"] == "build-tags"

        # v3 core Tag objects must work too, not just the boto-style dicts.
        core_model = _model_builder(session, image_uri, artifact_uri, role, unique_id).build(
            model_name="{}-core-{}".format(MODEL_NAME_PREFIX, unique_id),
            mode=Mode.SAGEMAKER_ENDPOINT,
            tags=[Tag(key="project", value="build-tags-core")],
        )
        created_models.append(core_model)
        assert _tags_on(sagemaker_client, core_model.model_arn)["project"] == "build-tags-core"

    finally:
        for model in created_models:
            try:
                model.delete()
            except Exception:  # noqa
                pass
        for s3_key in s3_keys:
            try:
                s3_client.delete_object(Bucket=bucket, Key=s3_key)
            except Exception:  # noqa
                pass
