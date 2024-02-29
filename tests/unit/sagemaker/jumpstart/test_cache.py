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
import copy
import datetime
import io
import json
from unittest.mock import Mock, call, mock_open
from botocore.stub import Stubber
import botocore

from mock.mock import MagicMock
import pytest
from mock import patch
from sagemaker.session_settings import SessionSettings
from sagemaker.jumpstart.cache import JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY, JumpStartModelsCache
from sagemaker.jumpstart.constants import (
    ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE,
    ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE,
)
from sagemaker.jumpstart.types import (
    JumpStartModelHeader,
    JumpStartModelSpecs,
    JumpStartVersionedModelId,
)
from tests.unit.sagemaker.jumpstart.utils import (
    get_spec_from_base_spec,
    patched_retrieval_function,
)

from tests.unit.sagemaker.jumpstart.constants import (
    BASE_MANIFEST,
    BASE_SPEC,
)
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket


REGION = "us-east-1"
REGION2 = "us-east-2"
ACCOUNT_ID = "123456789123"


@pytest.fixture()
def sagemaker_session():
    mocked_boto_session = Mock(name="boto_session")
    mocked_s3_client = Mock(name="s3_client")
    mocked_sagemaker_session = Mock(
        name="sagemaker_session",
        boto_session=mocked_boto_session,
        s3_client=mocked_s3_client,
        boto_region_name=REGION,
        config=None,
    )
    mocked_sagemaker_session.sagemaker_config = {}
    mocked_sagemaker_session._client_config.user_agent = (
        "Boto3/1.9.69 Python/3.6.5 Linux/4.14.77-70.82.amzn1.x86_64 Botocore/1.12.69 Resource"
    )
    mocked_sagemaker_session.account_id.return_value = ACCOUNT_ID
    return mocked_sagemaker_session


@patch.object(JumpStartModelsCache, "_retrieval_function", patched_retrieval_function)
@patch("sagemaker.jumpstart.utils.get_sagemaker_version", lambda: "2.68.3")
def test_jumpstart_cache_get_header():

    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic"
            "-imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
    )

    # See if we can make the same query 2 times consecutively
    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic"
            "-imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4", semantic_version_str="2.*"
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        semantic_version_str="2.0.*",
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        semantic_version_str="2.0.0",
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        semantic_version_str="1.0.0",
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4", semantic_version_str="1.*"
    )

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        semantic_version_str="1.0.*",
    )

    with pytest.raises(KeyError) as e:
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="3.*",
        )
    assert (
        "Unable to find model manifest for 'tensorflow-ic-imagenet-inception-v3-classification-4' "
        "with version '3.*' compatible with your SageMaker version ('2.68.3'). Consider upgrading "
        "your SageMaker library to at least version '4.49.0' so you can use version '3.0.0' of "
        "'tensorflow-ic-imagenet-inception-v3-classification-4'." in str(e.value)
    )

    with pytest.raises(KeyError) as e:
        cache.get_header(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="3.*"
        )
    assert (
        "Unable to find model manifest for 'pytorch-ic-imagenet-inception-v3-classification-4' with "
        "version '3.*'. Visit https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html "
        "for updated list of models. Consider using model ID 'pytorch-ic-imagenet-inception-v3-"
        "classification-4' with version '2.0.0'."
    ) in str(e.value)

    with pytest.raises(KeyError) as e:
        cache.get_header(model_id="pytorch-ic-", semantic_version_str="*")
    assert (
        "Unable to find model manifest for 'pytorch-ic-' with version '*'. "
        "Visit https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html "
        "for updated list of models. "
        "Did you mean to use model ID 'pytorch-ic-imagenet-inception-v3-classification-4'?"
    ) in str(e.value)

    with pytest.raises(KeyError) as e:
        cache.get_header(model_id="tensorflow-ic-", semantic_version_str="*")
    assert (
        "Unable to find model manifest for 'tensorflow-ic-' with version '*'. "
        "Visit https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html "
        "for updated list of models. "
        "Did you mean to use model ID 'tensorflow-ic-imagenet-inception-"
        "v3-classification-4'?"
    ) in str(e.value)

    with pytest.raises(KeyError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="BAD",
        )

    with pytest.raises(KeyError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="2.1.*",
        )

    with pytest.raises(KeyError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="3.9.*",
        )

    with pytest.raises(KeyError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="5.*",
        )

    with pytest.raises(KeyError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4-bak",
            semantic_version_str="*",
        )


@patch("boto3.client")
def test_jumpstart_cache_handles_boto3_issues(mock_boto3_client):

    mock_boto3_client.return_value.get_object.side_effect = Exception()

    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")

    with pytest.raises(Exception):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        )

    mock_boto3_client.return_value.reset_mock()

    mock_boto3_client.return_value.head_object.side_effect = Exception()

    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")

    with pytest.raises(Exception):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
        )


@patch("boto3.client")
def test_jumpstart_cache_gets_cleared_when_params_are_set(mock_boto3_client):
    cache = JumpStartModelsCache(
        s3_bucket_name="some_bucket", region=REGION, manifest_file_s3_key="some_key"
    )

    cache.clear = MagicMock()
    cache.set_s3_bucket_name("some_bucket")
    cache.clear.assert_not_called()
    cache.clear.reset_mock()
    cache.set_region(REGION)
    cache.clear.assert_not_called()
    cache.clear.reset_mock()
    cache.set_manifest_file_s3_key("some_key")
    cache.clear.assert_not_called()

    cache.clear.reset_mock()

    cache.set_s3_bucket_name("some_bucket1")
    cache.clear.assert_called_once()
    cache.clear.reset_mock()
    cache.set_region(REGION2)
    cache.clear.assert_called_once()
    cache.clear.reset_mock()
    cache.set_manifest_file_s3_key("some_key1")
    cache.clear.assert_called_once()


def test_jumpstart_cache_handles_boto3_client_errors():
    # Testing get_object
    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")
    stubbed_s3_client = Stubber(cache._s3_client)
    stubbed_s3_client.add_client_error("get_object", http_status_code=404)
    stubbed_s3_client.activate()
    with pytest.raises(botocore.exceptions.ClientError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="*",
        )

    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")
    stubbed_s3_client = Stubber(cache._s3_client)
    stubbed_s3_client.add_client_error("get_object", service_error_code="AccessDenied")
    stubbed_s3_client.activate()
    with pytest.raises(botocore.exceptions.ClientError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="*",
        )

    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")
    stubbed_s3_client = Stubber(cache._s3_client)
    stubbed_s3_client.add_client_error("get_object", service_error_code="EndpointConnectionError")
    stubbed_s3_client.activate()
    with pytest.raises(botocore.exceptions.ClientError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="*",
        )

    # Testing head_object:
    mock_now = datetime.datetime.fromtimestamp(1636730651.079551)
    with patch("datetime.datetime") as mock_datetime:
        mock_manifest_json = json.dumps(
            [
                {
                    "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
                    "version": "2.0.0",
                    "min_version": "2.49.0",
                    "spec_key": "community_models_specs/pytorch-ic-"
                    "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
                }
            ]
        )

        get_object_mocked_response = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(mock_manifest_json, "utf-8")),
                content_length=len(mock_manifest_json),
            ),
            "ETag": "etag",
        }

        mock_datetime.now.return_value = mock_now

        cache1 = JumpStartModelsCache(
            s3_bucket_name="some_bucket", s3_cache_expiration_horizon=datetime.timedelta(hours=1)
        )
        stubbed_s3_client1 = Stubber(cache1._s3_client)

        stubbed_s3_client1.add_response("get_object", copy.deepcopy(get_object_mocked_response))
        stubbed_s3_client1.activate()
        cache1.get_header(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
        )

        mock_datetime.now.return_value += datetime.timedelta(weeks=1)

        stubbed_s3_client1.add_client_error("head_object", http_status_code=404)
        with pytest.raises(botocore.exceptions.ClientError):
            cache1.get_header(
                model_id="pytorch-ic-imagenet-inception-v3-classification-4",
                semantic_version_str="*",
            )

        cache2 = JumpStartModelsCache(
            s3_bucket_name="some_bucket", s3_cache_expiration_horizon=datetime.timedelta(hours=1)
        )
        stubbed_s3_client2 = Stubber(cache2._s3_client)

        stubbed_s3_client2.add_response("get_object", copy.deepcopy(get_object_mocked_response))
        stubbed_s3_client2.activate()
        cache2.get_header(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
        )

        mock_datetime.now.return_value += datetime.timedelta(weeks=1)

        stubbed_s3_client2.add_client_error("head_object", service_error_code="AccessDenied")
        with pytest.raises(botocore.exceptions.ClientError):
            cache2.get_header(
                model_id="pytorch-ic-imagenet-inception-v3-classification-4",
                semantic_version_str="*",
            )

        cache3 = JumpStartModelsCache(
            s3_bucket_name="some_bucket", s3_cache_expiration_horizon=datetime.timedelta(hours=1)
        )
        stubbed_s3_client3 = Stubber(cache3._s3_client)

        stubbed_s3_client3.add_response("get_object", copy.deepcopy(get_object_mocked_response))
        stubbed_s3_client3.activate()
        cache3.get_header(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
        )

        mock_datetime.now.return_value += datetime.timedelta(weeks=1)

        stubbed_s3_client3.add_client_error(
            "head_object", service_error_code="EndpointConnectionError"
        )
        with pytest.raises(botocore.exceptions.ClientError):
            cache3.get_header(
                model_id="pytorch-ic-imagenet-inception-v3-classification-4",
                semantic_version_str="*",
            )


def test_jumpstart_cache_accepts_input_parameters():

    max_s3_cache_items = 1
    s3_cache_expiration_horizon = datetime.timedelta(weeks=2)
    max_semantic_version_cache_items = 3
    semantic_version_cache_expiration_horizon = datetime.timedelta(microseconds=4)
    bucket = "my-amazing-bucket"
    manifest_file_key = "some_s3_key"

    cache = JumpStartModelsCache(
        region=REGION,
        max_s3_cache_items=max_s3_cache_items,
        s3_cache_expiration_horizon=s3_cache_expiration_horizon,
        max_semantic_version_cache_items=max_semantic_version_cache_items,
        semantic_version_cache_expiration_horizon=semantic_version_cache_expiration_horizon,
        s3_bucket_name=bucket,
        manifest_file_s3_key=manifest_file_key,
    )

    assert cache.get_manifest_file_s3_key() == manifest_file_key
    assert cache.get_region() == REGION
    assert cache.get_bucket() == bucket
    assert cache._content_cache._max_cache_items == max_s3_cache_items
    assert cache._content_cache._expiration_horizon == s3_cache_expiration_horizon
    assert (
        cache._model_id_semantic_version_manifest_key_cache._max_cache_items
        == max_semantic_version_cache_items
    )
    assert (
        cache._model_id_semantic_version_manifest_key_cache._expiration_horizon
        == semantic_version_cache_expiration_horizon
    )


@patch("boto3.client")
def test_jumpstart_cache_evaluates_md5_hash(mock_boto3_client):

    mock_json = json.dumps(
        [
            {
                "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
                "version": "2.0.0",
                "min_version": "2.49.0",
                "spec_key": "community_models_specs/pytorch-ic-"
                "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
            }
        ]
    )

    bucket_name = "bucket_name"
    now = datetime.datetime.fromtimestamp(1636730651.079551)

    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = now

        cache = JumpStartModelsCache(
            s3_bucket_name=bucket_name, s3_cache_expiration_horizon=datetime.timedelta(hours=1)
        )

        mock_boto3_client.return_value.get_object.return_value = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
            ),
            "ETag": "hash1",
        }
        mock_boto3_client.return_value.head_object.return_value = {"ETag": "hash1"}

        cache.get_header(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
        )

        # first time accessing cache should just involve get_object
        mock_boto3_client.return_value.get_object.assert_called_with(
            Bucket=bucket_name, Key=JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY
        )
        mock_boto3_client.return_value.head_object.assert_not_called()

        mock_boto3_client.return_value.get_object.reset_mock()
        mock_boto3_client.return_value.head_object.reset_mock()

        # second time accessing cache should just involve head_object if hash hasn't changed
        mock_boto3_client.return_value.get_object.return_value = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
            ),
            "ETag": "hash1",
        }
        mock_boto3_client.return_value.head_object.return_value = {"ETag": "hash1"}

        # invalidate cache
        mock_datetime.now.return_value += datetime.timedelta(hours=2)

        cache.get_header(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
        )

        mock_boto3_client.return_value.head_object.assert_called_with(
            Bucket=bucket_name, Key=JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY
        )
        mock_boto3_client.return_value.get_object.assert_not_called()

        mock_boto3_client.return_value.get_object.reset_mock()
        mock_boto3_client.return_value.head_object.reset_mock()

        # third time accessing cache should involve head_object and get_object if hash has changed
        mock_boto3_client.return_value.head_object.return_value = {"ETag": "hash2"}
        mock_boto3_client.return_value.get_object.return_value = {
            "Body": botocore.response.StreamingBody(
                io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
            ),
            "ETag": "hash2",
        }

        # invalidate cache
        mock_datetime.now.return_value += datetime.timedelta(hours=2)

        cache.get_header(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
        )

        mock_boto3_client.return_value.get_object.assert_called_with(
            Bucket=bucket_name, Key=JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY
        )
        mock_boto3_client.return_value.head_object.assert_called_with(
            Bucket=bucket_name, Key=JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY
        )


@patch("sagemaker.jumpstart.cache.utils.emit_logs_based_on_model_specs")
@patch("boto3.client")
def test_jumpstart_cache_makes_correct_s3_calls(
    mock_boto3_client, mock_emit_logs_based_on_model_specs
):

    # test get_header
    mock_manifest_json = json.dumps(
        [
            {
                "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
                "version": "2.0.0",
                "min_version": "2.49.0",
                "spec_key": "community_models_specs/pytorch-ic-"
                "imagenet-inception-v3-classification-4/specs_v2.0.0.json",
            }
        ]
    )
    mock_boto3_client.return_value.get_object.return_value = {
        "Body": botocore.response.StreamingBody(
            io.BytesIO(bytes(mock_manifest_json, "utf-8")), content_length=len(mock_manifest_json)
        ),
        "ETag": "etag",
    }

    mock_boto3_client.return_value.head_object.return_value = {"ETag": "some-hash"}

    bucket_name = get_jumpstart_content_bucket("us-west-2")
    client_config = botocore.config.Config(signature_version="my_signature_version")
    cache = JumpStartModelsCache(
        s3_bucket_name=bucket_name, s3_client_config=client_config, region="us-west-2"
    )
    cache.get_header(
        model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
    )

    mock_boto3_client.return_value.get_object.assert_called_with(
        Bucket=bucket_name, Key=JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY
    )
    mock_boto3_client.return_value.head_object.assert_not_called()

    mock_boto3_client.assert_called_with("s3", region_name="us-west-2", config=client_config)

    # test get_specs. manifest already in cache, so only s3 call will be to get specs.
    mock_json = json.dumps(BASE_SPEC)

    mock_boto3_client.return_value.reset_mock()

    mock_boto3_client.return_value.get_object.return_value = {
        "Body": botocore.response.StreamingBody(
            io.BytesIO(bytes(mock_json, "utf-8")), content_length=len(mock_json)
        ),
        "ETag": "etag",
    }

    with patch("logging.Logger.warning") as mocked_warning_log:
        cache.get_specs(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
        )
        mocked_warning_log.assert_called_once_with(
            "Using model 'pytorch-ic-imagenet-inception-v3-classification-4' with wildcard "
            "version identifier '*'. You can pin to version '2.0.0' for more "
            "stable results. Note that models may have different input/output "
            "signatures after a major version upgrade."
        )
        mocked_warning_log.reset_mock()
        cache.get_specs(
            model_id="pytorch-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
        )
        mocked_warning_log.assert_not_called()

    mock_boto3_client.return_value.get_object.assert_called_with(
        Bucket=bucket_name,
        Key="community_models_specs/pytorch-ic-imagenet-"
        "inception-v3-classification-4/specs_v2.0.0.json",
    )
    mock_boto3_client.return_value.head_object.assert_not_called()


@patch.object(JumpStartModelsCache, "_retrieval_function", patched_retrieval_function)
def test_jumpstart_cache_handles_bad_semantic_version_manifest_key_cache():
    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")

    cache.clear = MagicMock()
    cache._model_id_semantic_version_manifest_key_cache = MagicMock()
    cache._model_id_semantic_version_manifest_key_cache.get.side_effect = [
        (
            JumpStartVersionedModelId(
                "tensorflow-ic-imagenet-inception-v3-classification-4", "999.0.0"
            ),
            True,
        ),
        (
            JumpStartVersionedModelId(
                "tensorflow-ic-imagenet-inception-v3-classification-4", "1.0.0"
            ),
            True,
        ),
    ]

    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-"
            "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    ) == cache.get_header(
        model_id="tensorflow-ic-imagenet-inception-v3-classification-4", semantic_version_str="*"
    )
    cache.clear.assert_called_once()
    cache.clear.reset_mock()

    cache._model_id_semantic_version_manifest_key_cache.get.side_effect = [
        (
            JumpStartVersionedModelId(
                "tensorflow-ic-imagenet-inception-v3-classification-4", "999.0.0"
            ),
            True,
        ),
        (
            JumpStartVersionedModelId(
                "tensorflow-ic-imagenet-inception-v3-classification-4", "987.0.0"
            ),
            True,
        ),
    ]
    with pytest.raises(KeyError):
        cache.get_header(
            model_id="tensorflow-ic-imagenet-inception-v3-classification-4",
            semantic_version_str="*",
        )
    cache.clear.assert_called_once()


@patch.object(JumpStartModelsCache, "_retrieval_function", patched_retrieval_function)
@patch("sagemaker.jumpstart.utils.get_sagemaker_version", lambda: "2.68.3")
def test_jumpstart_get_full_manifest():
    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")
    raw_manifest = [header.to_json() for header in cache.get_manifest()]

    raw_manifest == BASE_MANIFEST


@patch.object(JumpStartModelsCache, "_retrieval_function", patched_retrieval_function)
@patch("sagemaker.jumpstart.utils.get_sagemaker_version", lambda: "2.68.3")
def test_jumpstart_cache_get_specs():
    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")

    model_id, version = "tensorflow-ic-imagenet-inception-v3-classification-4", "2.0.0"
    assert get_spec_from_base_spec(model_id=model_id, version=version) == cache.get_specs(
        model_id=model_id, semantic_version_str=version
    )

    model_id = "tensorflow-ic-imagenet-inception-v3-classification-4"
    assert get_spec_from_base_spec(model_id=model_id, version="2.0.0") == cache.get_specs(
        model_id=model_id, semantic_version_str="2.0.*"
    )

    model_id, version = "tensorflow-ic-imagenet-inception-v3-classification-4", "1.0.0"
    assert get_spec_from_base_spec(model_id=model_id, version=version) == cache.get_specs(
        model_id=model_id, semantic_version_str=version
    )

    model_id = "pytorch-ic-imagenet-inception-v3-classification-4"
    assert get_spec_from_base_spec(model_id=model_id, version="1.0.0") == cache.get_specs(
        model_id=model_id, semantic_version_str="1.*"
    )

    model_id = "pytorch-ic-imagenet-inception-v3-classification-4"
    assert get_spec_from_base_spec(model_id=model_id, version="1.0.0") == cache.get_specs(
        model_id=model_id, semantic_version_str="1.0.*"
    )

    with pytest.raises(KeyError):
        cache.get_specs(model_id=model_id + "bak", semantic_version_str="*")

    with pytest.raises(KeyError):
        cache.get_specs(model_id=model_id, semantic_version_str="9.*")

    with pytest.raises(KeyError):
        cache.get_specs(model_id=model_id, semantic_version_str="BAD")

    with pytest.raises(KeyError):
        cache.get_specs(
            model_id=model_id,
            semantic_version_str="2.1.*",
        )

    with pytest.raises(KeyError):
        cache.get_specs(
            model_id=model_id,
            semantic_version_str="3.9.*",
        )

    with pytest.raises(KeyError):
        cache.get_specs(
            model_id=model_id,
            semantic_version_str="5.*",
        )


@patch.object(JumpStartModelsCache, "_get_json_file_and_etag_from_s3")
@patch("sagemaker.jumpstart.utils.get_sagemaker_version", lambda: "2.68.3")
@patch.dict(
    "sagemaker.jumpstart.cache.os.environ",
    {
        ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE: "/some/directory/metadata/manifest/root",
        ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE: "/some/directory/metadata/specs/root",
    },
)
@patch("sagemaker.jumpstart.cache.os.path.isdir")
@patch("builtins.open")
def test_jumpstart_local_metadata_override_header(
    mocked_open: Mock,
    mocked_is_dir: Mock,
    mocked_get_json_file_and_etag_from_s3: Mock,
    sagemaker_session: Mock,
):
    mocked_open.side_effect = mock_open(read_data=json.dumps(BASE_MANIFEST))
    mocked_is_dir.return_value = True
    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")

    model_id, version = "tensorflow-ic-imagenet-inception-v3-classification-4", "2.0.0"
    assert JumpStartModelHeader(
        {
            "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
            "version": "2.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-imagenet-inception-v3-classification-4/specs_v2.0.0.json",
        }
    ) == cache.get_header(model_id=model_id, semantic_version_str=version)

    mocked_is_dir.assert_any_call("/some/directory/metadata/manifest/root")
    mocked_is_dir.assert_any_call("/some/directory/metadata/specs/root")
    assert mocked_is_dir.call_count == 2
    mocked_open.assert_called_with(
        "/some/directory/metadata/manifest/root/models_manifest.json", "r"
    )
    mocked_get_json_file_and_etag_from_s3.assert_not_called()


@patch("sagemaker.jumpstart.cache.utils.emit_logs_based_on_model_specs")
@patch.object(JumpStartModelsCache, "_get_json_file_and_etag_from_s3")
@patch("sagemaker.jumpstart.utils.get_sagemaker_version", lambda: "2.68.3")
@patch.dict(
    "sagemaker.jumpstart.cache.os.environ",
    {
        ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE: "/some/directory/metadata/manifest/root",
        ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE: "/some/directory/metadata/specs/root",
    },
)
@patch("sagemaker.jumpstart.cache.os.path.isdir")
@patch("builtins.open")
def test_jumpstart_local_metadata_override_specs(
    mocked_open: Mock,
    mocked_is_dir: Mock,
    mocked_get_json_file_and_etag_from_s3: Mock,
    mock_emit_logs_based_on_model_specs,
    sagemaker_session,
):

    mocked_open.side_effect = [
        mock_open(read_data=json.dumps(BASE_MANIFEST)).return_value,
        mock_open(read_data=json.dumps(BASE_SPEC)).return_value,
    ]

    mocked_is_dir.return_value = True
    cache = JumpStartModelsCache(
        s3_bucket_name="some_bucket", s3_client=Mock(), sagemaker_session=sagemaker_session
    )

    model_id, version = "tensorflow-ic-imagenet-inception-v3-classification-4", "2.0.0"
    assert JumpStartModelSpecs(BASE_SPEC) == cache.get_specs(
        model_id=model_id, semantic_version_str=version
    )

    mocked_is_dir.assert_any_call("/some/directory/metadata/specs/root")
    mocked_is_dir.assert_any_call("/some/directory/metadata/manifest/root")
    assert mocked_is_dir.call_count == 4
    mocked_open.assert_any_call("/some/directory/metadata/manifest/root/models_manifest.json", "r")
    mocked_open.assert_any_call(
        "/some/directory/metadata/specs/root/community_models_specs/tensorflow-ic-imagenet-"
        "inception-v3-classification-4/specs_v2.0.0.json",
        "r",
    )
    assert mocked_open.call_count == 2
    mocked_get_json_file_and_etag_from_s3.assert_not_called()


@patch("sagemaker.jumpstart.cache.utils.emit_logs_based_on_model_specs")
@patch.object(JumpStartModelsCache, "_get_json_file_and_etag_from_s3")
@patch("sagemaker.jumpstart.utils.get_sagemaker_version", lambda: "2.68.3")
@patch.dict(
    "sagemaker.jumpstart.cache.os.environ",
    {
        ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE: "/some/directory/metadata/manifest/root",
        ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE: "/some/directory/metadata/specs/root",
    },
)
@patch("sagemaker.jumpstart.cache.os.path.isdir")
@patch("builtins.open")
def test_jumpstart_local_metadata_override_specs_not_exist_both_directories(
    mocked_open: Mock,
    mocked_is_dir: Mock,
    mocked_get_json_file_and_etag_from_s3: Mock,
    mocked_emit_logs_based_on_model_specs,
):
    model_id, version = "tensorflow-ic-imagenet-inception-v3-classification-4", "2.0.0"

    mocked_get_json_file_and_etag_from_s3.side_effect = [
        (BASE_MANIFEST, "blah1"),
        (get_spec_from_base_spec(model_id=model_id, version=version).to_json(), "blah2"),
    ]

    mocked_is_dir.side_effect = [False, False]
    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")

    assert get_spec_from_base_spec(model_id=model_id, version=version) == cache.get_specs(
        model_id=model_id, semantic_version_str=version
    )

    mocked_is_dir.assert_any_call("/some/directory/metadata/manifest/root")
    assert mocked_is_dir.call_count == 2
    assert mocked_open.call_count == 2
    mocked_get_json_file_and_etag_from_s3.assert_has_calls(
        calls=[
            call("models_manifest.json"),
            call(
                "community_models_specs/tensorflow-ic-imagenet-inception-v3-classification-4/specs_v2.0.0.json"
            ),
        ]
    )


@patch.object(JumpStartModelsCache, "_retrieval_function", patched_retrieval_function)
def test_jumpstart_cache_get_hub_model():
    cache = JumpStartModelsCache(s3_bucket_name="some_bucket")

    model_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub-content/HubName/Model/huggingface-mock-model-123/1.2.3"
    assert get_spec_from_base_spec(
        model_id="huggingface-mock-model-123", version="1.2.3"
    ) == cache.get_hub_model(hub_model_arn=model_arn)

    model_arn = "arn:aws:sagemaker:us-west-2:000000000000:hub-content/HubName/Model/pytorch-mock-model-123/*"
    assert get_spec_from_base_spec(
        model_id="pytorch-mock-model-123", version="*"
    ) == cache.get_hub_model(hub_model_arn=model_arn)
