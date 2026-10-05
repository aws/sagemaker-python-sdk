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
"""Tests for the ExpectedBucketOwner spot-check across S3 operations.

One test per file that was modified for the bucket-sniping fix, verifying that
ExpectedBucketOwner is passed when the bucket is the SDK-generated default and
omitted for cross-account buckets.
"""
from __future__ import absolute_import

import io
import json
import os
import tempfile

import pytest
from mock import MagicMock, Mock, patch
from botocore.exceptions import ClientError

import sagemaker
from sagemaker import s3 as s3_module
from sagemaker.session import Session
from sagemaker.lambda_helper import _upload_to_s3

REGION = "us-west-2"
ACCOUNT_ID = "111111111111"
DEFAULT_BUCKET = "sagemaker-{}-{}".format(REGION, ACCOUNT_ID)
CROSS_ACCOUNT_BUCKET = "partner-cross-account-bucket"
STS_ENDPOINT = "sts.us-west-2.amazonaws.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(default_bucket=DEFAULT_BUCKET, sdk_generated=True):
    """Create a minimal real Session with mocked boto internals."""
    boto_mock = MagicMock(name="boto_session", region_name=REGION)
    boto_mock.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Account": ACCOUNT_ID,
    }
    sess = Session(boto_session=boto_mock, sagemaker_client=MagicMock())
    sess._default_bucket = default_bucket
    sess._default_bucket_set_by_sdk = sdk_generated
    return sess


# ===========================================================================
# 1. session.py  –  _get_account_id_if_default_bucket helper
# ===========================================================================

class TestGetAccountIdIfDefaultBucket:

    def test_returns_account_id_for_sdk_generated_default(self):
        sess = _make_session()
        assert sess._get_account_id_if_default_bucket(DEFAULT_BUCKET) == ACCOUNT_ID

    def test_returns_none_for_cross_account_bucket(self):
        sess = _make_session()
        assert sess._get_account_id_if_default_bucket(CROSS_ACCOUNT_BUCKET) is None

    def test_returns_none_when_user_overrode_bucket(self):
        sess = _make_session(sdk_generated=False)
        assert sess._get_account_id_if_default_bucket(DEFAULT_BUCKET) is None

    def test_returns_none_for_empty_bucket(self):
        sess = _make_session()
        assert sess._get_account_id_if_default_bucket(None) is None
        assert sess._get_account_id_if_default_bucket("") is None


# ===========================================================================
# 2. session.py  –  general_bucket_check_if_user_has_permission
# ===========================================================================

class TestGeneralBucketCheckExpectedOwner:

    def test_head_bucket_includes_expected_owner_when_sdk_selected(self):
        sess = _make_session()
        mock_s3 = Mock()
        mock_s3.meta.client.head_bucket.return_value = None

        sess.general_bucket_check_if_user_has_permission(
            DEFAULT_BUCKET, mock_s3, Mock(), REGION, False
        )

        mock_s3.meta.client.head_bucket.assert_called_once_with(
            Bucket=DEFAULT_BUCKET, ExpectedBucketOwner=ACCOUNT_ID
        )

    def test_head_bucket_omits_expected_owner_when_user_selected(self):
        sess = _make_session(sdk_generated=False)
        mock_s3 = Mock()
        mock_s3.meta.client.head_bucket.return_value = None

        sess.general_bucket_check_if_user_has_permission(
            CROSS_ACCOUNT_BUCKET, mock_s3, Mock(), REGION, False
        )

        mock_s3.meta.client.head_bucket.assert_called_once_with(
            Bucket=CROSS_ACCOUNT_BUCKET
        )


# ===========================================================================
# 3. session.py  –  upload_data
# ===========================================================================

class TestUploadDataSpotCheck:

    def test_upload_to_default_bucket_includes_expected_owner(self, tmp_path):
        sess = _make_session()
        test_file = tmp_path / "test.txt"
        test_file.write_text("x")

        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        sess.s3_resource = mock_s3_resource

        sess.upload_data(
            path=str(test_file), bucket=DEFAULT_BUCKET, key_prefix="data"
        )

        call_args = mock_s3_object.upload_file.call_args
        assert call_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}

    def test_upload_to_non_default_bucket_omits_expected_owner(self, tmp_path):
        sess = _make_session()
        test_file = tmp_path / "test.txt"
        test_file.write_text("x")

        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        sess.s3_resource = mock_s3_resource

        sess.upload_data(
            path=str(test_file), bucket=CROSS_ACCOUNT_BUCKET, key_prefix="data"
        )

        call_args = mock_s3_object.upload_file.call_args
        assert call_args[1]["ExtraArgs"] is None


# ===========================================================================
# 4. session.py  –  upload_string_as_file_body
# ===========================================================================

class TestUploadStringSpotCheck:

    def test_to_default_bucket_includes_expected_owner(self):
        sess = _make_session()
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        sess.s3_resource = mock_s3_resource

        sess.upload_string_as_file_body(body="data", bucket=DEFAULT_BUCKET, key="k")

        mock_s3_object.put.assert_called_once_with(
            Body="data", ExpectedBucketOwner=ACCOUNT_ID
        )

    def test_to_non_default_bucket_omits_expected_owner(self):
        sess = _make_session()
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        sess.s3_resource = mock_s3_resource

        sess.upload_string_as_file_body(body="data", bucket=CROSS_ACCOUNT_BUCKET, key="k")

        mock_s3_object.put.assert_called_once_with(Body="data")


# ===========================================================================
# 5. session.py  –  read_s3_file
# ===========================================================================

class TestReadS3FileSpotCheck:

    def test_read_from_default_bucket_includes_expected_owner(self):
        sess = _make_session()
        mock_s3_client = Mock()
        mock_body = Mock()
        mock_body.read.return_value = b"content"
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        sess.s3_client = mock_s3_client

        sess.read_s3_file(DEFAULT_BUCKET, "k")

        mock_s3_client.get_object.assert_called_once_with(
            Bucket=DEFAULT_BUCKET, Key="k", ExpectedBucketOwner=ACCOUNT_ID
        )

    def test_read_from_non_default_bucket_omits_expected_owner(self):
        sess = _make_session()
        mock_s3_client = Mock()
        mock_body = Mock()
        mock_body.read.return_value = b"content"
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        sess.s3_client = mock_s3_client

        sess.read_s3_file(CROSS_ACCOUNT_BUCKET, "k")

        mock_s3_client.get_object.assert_called_once_with(
            Bucket=CROSS_ACCOUNT_BUCKET, Key="k"
        )


# ===========================================================================
# 6. session.py  –  download_data
# ===========================================================================

class TestDownloadDataSpotCheck:

    def test_download_from_default_bucket_includes_expected_owner(self, tmp_path):
        sess = _make_session()
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "p/f.txt", "Size": 1}]
        }
        sess.s3_client = mock_s3_client

        sess.download_data(path=str(tmp_path), bucket=DEFAULT_BUCKET, key_prefix="p/f.txt")

        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket=DEFAULT_BUCKET, Prefix="p/f.txt", ExpectedBucketOwner=ACCOUNT_ID
        )
        assert (
            mock_s3_client.download_file.call_args[1]["ExtraArgs"]
            == {"ExpectedBucketOwner": ACCOUNT_ID}
        )

    def test_download_from_non_default_bucket_omits_expected_owner(self, tmp_path):
        sess = _make_session()
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "p/f.txt", "Size": 1}]
        }
        sess.s3_client = mock_s3_client

        sess.download_data(path=str(tmp_path), bucket=CROSS_ACCOUNT_BUCKET, key_prefix="p/f.txt")

        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket=CROSS_ACCOUNT_BUCKET, Prefix="p/f.txt"
        )
        assert mock_s3_client.download_file.call_args[1]["ExtraArgs"] is None


# ===========================================================================
# 7. s3.py  –  S3Uploader.upload_bytes
# ===========================================================================

class TestS3UploaderUploadBytesSpotCheck:

    def test_upload_bytes_to_default_bucket_includes_expected_owner(self):
        mock_session = Mock(name="sagemaker_session")
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
        mock_bucket = Mock()
        mock_session.s3_resource.Bucket.return_value = mock_bucket

        s3_module.S3Uploader.upload_bytes(
            b"hello", "s3://{}/key".format(DEFAULT_BUCKET), sagemaker_session=mock_session
        )

        call_args = mock_bucket.upload_fileobj.call_args
        assert call_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}

    def test_upload_bytes_to_non_default_bucket_omits_expected_owner(self):
        mock_session = Mock(name="sagemaker_session")
        mock_session._get_account_id_if_default_bucket.return_value = None
        mock_bucket = Mock()
        mock_session.s3_resource.Bucket.return_value = mock_bucket

        s3_module.S3Uploader.upload_bytes(
            b"hello", "s3://{}/key".format(CROSS_ACCOUNT_BUCKET), sagemaker_session=mock_session
        )

        call_args = mock_bucket.upload_fileobj.call_args
        assert call_args[1]["ExtraArgs"] is None


# ===========================================================================
# 8. s3.py  –  S3Downloader.read_bytes
# ===========================================================================

class TestS3DownloaderReadBytesSpotCheck:

    def test_read_bytes_from_default_bucket_includes_expected_owner(self):
        mock_session = Mock(name="sagemaker_session")
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
        mock_bucket = Mock()
        mock_session.s3_resource.Bucket.return_value = mock_bucket

        s3_module.S3Downloader.read_bytes(
            "s3://{}/key".format(DEFAULT_BUCKET), sagemaker_session=mock_session
        )

        call_args = mock_bucket.download_fileobj.call_args
        assert call_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}

    def test_read_bytes_from_non_default_bucket_omits_expected_owner(self):
        mock_session = Mock(name="sagemaker_session")
        mock_session._get_account_id_if_default_bucket.return_value = None
        mock_bucket = Mock()
        mock_session.s3_resource.Bucket.return_value = mock_bucket

        s3_module.S3Downloader.read_bytes(
            "s3://{}/key".format(CROSS_ACCOUNT_BUCKET), sagemaker_session=mock_session
        )

        call_args = mock_bucket.download_fileobj.call_args
        assert call_args[1]["ExtraArgs"] is None


# ===========================================================================
# 9. utils.py  –  download_folder
# ===========================================================================

class TestDownloadFolderSpotCheck:

    @patch("os.makedirs")
    def test_download_folder_default_bucket_includes_expected_owner(self, makedirs):
        sess = _make_session()
        mock_s3 = Mock()
        mock_obj = Mock()
        mock_s3.Object.return_value = mock_obj
        sess.s3_resource = mock_s3

        # Make the single-file download succeed so we hit the Object().download_file path
        sagemaker.utils.download_folder(DEFAULT_BUCKET, "prefix/file.tar.gz", "/tmp", sess)

        call_args = mock_s3.Object.return_value.download_file.call_args
        assert call_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}

    @patch("os.makedirs")
    def test_download_folder_cross_account_omits_expected_owner(self, makedirs):
        sess = _make_session()
        mock_s3 = Mock()
        mock_obj = Mock()
        mock_s3.Object.return_value = mock_obj
        sess.s3_resource = mock_s3

        sagemaker.utils.download_folder(CROSS_ACCOUNT_BUCKET, "prefix/file.tar.gz", "/tmp", sess)

        call_args = mock_s3.Object.return_value.download_file.call_args
        assert call_args[1]["ExtraArgs"] is None


# ===========================================================================
# 10. utils.py  –  download_file
# ===========================================================================

class TestDownloadFileSpotCheck:

    def test_download_file_default_bucket_includes_expected_owner(self):
        sess = _make_session()
        mock_bucket = Mock()
        sess.boto_session.resource("s3").Bucket.return_value = mock_bucket

        sagemaker.utils.download_file(DEFAULT_BUCKET, "k", "/tmp/f", sess)

        mock_bucket.download_file.assert_called_once_with(
            "k", "/tmp/f", ExtraArgs={"ExpectedBucketOwner": ACCOUNT_ID}
        )

    def test_download_file_cross_account_omits_expected_owner(self):
        sess = _make_session()
        mock_bucket = Mock()
        sess.boto_session.resource("s3").Bucket.return_value = mock_bucket

        sagemaker.utils.download_file(CROSS_ACCOUNT_BUCKET, "k", "/tmp/f", sess)

        mock_bucket.download_file.assert_called_once_with(
            "k", "/tmp/f", ExtraArgs=None
        )


# ===========================================================================
# 11. utils.py  –  _save_model
# ===========================================================================

class TestSaveModelSpotCheck:

    def test_save_to_default_bucket_includes_expected_owner(self, tmp_path):
        model_file = tmp_path / "m.tar.gz"
        model_file.write_text("x")

        sess = _make_session()
        mock_obj = Mock()
        sess.boto_session.resource("s3").Object.return_value = mock_obj
        sess.settings = sagemaker.session_settings.SessionSettings(
            encrypt_repacked_artifacts=False
        )

        sagemaker.utils._save_model(
            "s3://{}/m.tar.gz".format(DEFAULT_BUCKET), str(model_file), sess, kms_key=None
        )

        call_args = mock_obj.upload_file.call_args
        assert call_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}

    def test_save_to_non_default_bucket_omits_expected_owner(self, tmp_path):
        model_file = tmp_path / "m.tar.gz"
        model_file.write_text("x")

        sess = _make_session()
        mock_obj = Mock()
        sess.boto_session.resource("s3").Object.return_value = mock_obj
        sess.settings = sagemaker.session_settings.SessionSettings(
            encrypt_repacked_artifacts=False
        )

        sagemaker.utils._save_model(
            "s3://{}/m.tar.gz".format(CROSS_ACCOUNT_BUCKET), str(model_file), sess, kms_key=None
        )

        call_args = mock_obj.upload_file.call_args
        assert call_args[1]["ExtraArgs"] is None


# ===========================================================================
# 12. fw_utils.py  –  tar_and_upload_dir
# ===========================================================================

class TestTarAndUploadDirSpotCheck:

    def test_expected_bucket_owner_injected_into_extra_args(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("print('hi')")

        mock_session = Mock()
        mock_session.region_name = REGION
        mock_s3_resource = Mock()
        mock_obj = Mock()
        mock_s3_resource.Object.return_value = mock_obj

        sagemaker.fw_utils.tar_and_upload_dir(
            session=mock_session,
            bucket=DEFAULT_BUCKET,
            s3_key_prefix="prefix",
            script=str(script),
            s3_resource=mock_s3_resource,
            expected_bucket_owner=ACCOUNT_ID,
        )

        call_args = mock_obj.upload_file.call_args
        assert call_args[1]["ExtraArgs"]["ExpectedBucketOwner"] == ACCOUNT_ID

    def test_no_expected_bucket_owner_when_none(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("print('hi')")

        mock_session = Mock()
        mock_session.region_name = REGION
        mock_s3_resource = Mock()
        mock_obj = Mock()
        mock_s3_resource.Object.return_value = mock_obj

        sagemaker.fw_utils.tar_and_upload_dir(
            session=mock_session,
            bucket=CROSS_ACCOUNT_BUCKET,
            s3_key_prefix="prefix",
            script=str(script),
            s3_resource=mock_s3_resource,
            expected_bucket_owner=None,
        )

        call_args = mock_obj.upload_file.call_args
        extra = call_args[1]["ExtraArgs"]
        # Should be None or not contain ExpectedBucketOwner
        assert extra is None or "ExpectedBucketOwner" not in extra


# ===========================================================================
# 13. lambda_helper.py  –  _upload_to_s3
# ===========================================================================

class TestLambdaUploadToS3SpotCheck:

    def test_upload_with_expected_bucket_owner(self):
        mock_s3_client = Mock()

        _upload_to_s3(
            mock_s3_client, "my-func", "/path/code.zip", DEFAULT_BUCKET,
            s3_key_prefix="prefix", expected_bucket_owner=ACCOUNT_ID,
        )

        mock_s3_client.upload_file.assert_called_once()
        call_args = mock_s3_client.upload_file.call_args
        assert call_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}

    def test_upload_without_expected_bucket_owner(self):
        mock_s3_client = Mock()

        _upload_to_s3(
            mock_s3_client, "my-func", "/path/code.zip", CROSS_ACCOUNT_BUCKET,
            s3_key_prefix="prefix", expected_bucket_owner=None,
        )

        call_args = mock_s3_client.upload_file.call_args
        assert call_args[1]["ExtraArgs"] is None


# ===========================================================================
# 14. experiments/_helper.py  –  _ArtifactUploader.upload_artifact
# ===========================================================================

class TestExperimentsHelperSpotCheck:

    def test_upload_artifact_to_default_bucket_includes_expected_owner(self, tmp_path):
        from sagemaker.experiments._helper import _ArtifactUploader

        test_file = tmp_path / "artifact.txt"
        test_file.write_text("data")

        mock_session = Mock()
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
        mock_session.default_bucket_prefix = None
        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {"ETag": '"abc"'}

        uploader = _ArtifactUploader(
            trial_component_name="tc-1",
            artifact_bucket=DEFAULT_BUCKET,
            artifact_prefix="artifacts",
            sagemaker_session=mock_session,
        )
        uploader._s3_client = mock_s3_client

        uploader.upload_artifact(str(test_file))

        call_args = mock_s3_client.upload_file.call_args
        assert call_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}

    def test_upload_object_artifact_to_default_bucket_includes_expected_owner(self):
        from sagemaker.experiments._helper import _ArtifactUploader

        mock_session = Mock()
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
        mock_session.default_bucket_prefix = None
        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {"ETag": '"abc"'}

        uploader = _ArtifactUploader(
            trial_component_name="tc-1",
            artifact_bucket=DEFAULT_BUCKET,
            artifact_prefix="artifacts",
            sagemaker_session=mock_session,
        )
        uploader._s3_client = mock_s3_client

        uploader.upload_object_artifact("metric", {"accuracy": 0.95})

        call_args = mock_s3_client.put_object.call_args
        assert call_args[1]["ExpectedBucketOwner"] == ACCOUNT_ID


# ===========================================================================
# 15. predictor_async.py  –  AsyncPredictor._upload_data_to_s3
# ===========================================================================

class TestAsyncPredictorSpotCheck:

    def test_upload_data_to_default_bucket_includes_expected_owner(self):
        mock_session = Mock()
        mock_session.default_bucket.return_value = DEFAULT_BUCKET
        mock_session.default_bucket_prefix = None
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID

        mock_predictor = Mock()
        mock_predictor.endpoint_name = "ep"
        mock_predictor.sagemaker_session = mock_session

        from sagemaker.predictor_async import AsyncPredictor
        async_pred = AsyncPredictor(predictor=mock_predictor, name="ep")
        async_pred.sagemaker_session = mock_session
        async_pred.s3_client = Mock()
        async_pred.serializer = Mock()
        async_pred.serializer.serialize.return_value = b"data"
        async_pred.serializer.CONTENT_TYPE = "application/octet-stream"

        async_pred._upload_data_to_s3(b"data")

        call_args = async_pred.s3_client.put_object.call_args
        assert call_args[1]["ExpectedBucketOwner"] == ACCOUNT_ID


# ===========================================================================
# 16. async_inference/async_inference_response.py
# ===========================================================================

class TestAsyncInferenceResponseSpotCheck:

    def test_get_result_from_default_bucket_includes_expected_owner(self):
        from sagemaker.async_inference import AsyncInferenceResponse

        mock_predictor_async = Mock()
        mock_predictor_async.sagemaker_session._get_account_id_if_default_bucket.return_value = (
            ACCOUNT_ID
        )
        mock_predictor_async.s3_client.get_object.return_value = {"Body": b"result"}
        mock_predictor_async.predictor._handle_response.return_value = "ok"

        response = AsyncInferenceResponse(
            predictor_async=mock_predictor_async,
            output_path="s3://{}/output/result.json".format(DEFAULT_BUCKET),
            failure_path=None,
        )

        response._get_result_from_s3_output_path(
            "s3://{}/output/result.json".format(DEFAULT_BUCKET)
        )

        call_args = mock_predictor_async.s3_client.get_object.call_args
        assert call_args[1]["ExpectedBucketOwner"] == ACCOUNT_ID


# ===========================================================================
# 17. multidatamodel.py  –  MultiDataModel.add_model (local upload path)
# ===========================================================================

class TestMultiDataModelSpotCheck:

    def test_add_model_local_to_default_bucket_includes_expected_owner(self, tmp_path):
        model_file = tmp_path / "model.tar.gz"
        model_file.write_text("x")

        mock_session = Mock()
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
        mock_session.s3_client = Mock()

        from sagemaker.multidatamodel import MultiDataModel
        mdm = MultiDataModel.__new__(MultiDataModel)
        mdm.model_data_prefix = "s3://{}/models/".format(DEFAULT_BUCKET)
        mdm.sagemaker_session = mock_session
        mdm.s3_client = mock_session.s3_client

        mdm.add_model(str(model_file))

        call_args = mock_session.s3_client.upload_file.call_args
        assert call_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}


# ===========================================================================
# 18. amazon/amazon_estimator.py  –  upload_numpy_to_s3_shards
# ===========================================================================

class TestUploadNumpySpotCheck:

    def test_upload_shards_to_default_bucket_includes_expected_owner(self):
        from sagemaker.amazon.amazon_estimator import upload_numpy_to_s3_shards

        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")

        mock_session = Mock()
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
        mock_s3 = Mock()
        mock_obj = Mock()
        mock_s3.Object.return_value = mock_obj

        array = np.array([[1, 2], [3, 4]])

        upload_numpy_to_s3_shards(
            1, mock_s3, DEFAULT_BUCKET, "prefix/", array,
            sagemaker_session=mock_session,
        )

        # Check that put() calls include ExpectedBucketOwner
        for c in mock_obj.put.call_args_list:
            assert c[1].get("ExpectedBucketOwner") == ACCOUNT_ID


# ===========================================================================
# 19. pytorch/estimator.py  –  _create_recipe_copy
# ===========================================================================

class TestPyTorchRecipeCopySpotCheck:

    def test_copy_object_to_default_bucket_includes_expected_owner(self):
        mock_session = Mock()
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
        mock_s3_client = Mock()
        mock_session.boto_session.client.return_value = mock_s3_client

        from sagemaker.pytorch.estimator import PyTorch
        estimator = PyTorch.__new__(PyTorch)
        estimator.sagemaker_session = mock_session

        estimator._create_recipe_copy("s3://{}/recipes/my_recipe.yaml".format(DEFAULT_BUCKET))

        call_args = mock_s3_client.copy_object.call_args
        assert call_args[1]["ExpectedBucketOwner"] == ACCOUNT_ID


# ===========================================================================
# 20. serve/model_format/mlflow/utils.py  –  _download_s3_artifacts
# ===========================================================================

class TestMlflowDownloadSpotCheck:

    def test_download_from_default_bucket_includes_expected_owner(self, tmp_path):
        from sagemaker.serve.model_format.mlflow.utils import _download_s3_artifacts

        mock_session = Mock()
        mock_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
        mock_s3_client = Mock()
        mock_session.boto_session.client.return_value = mock_s3_client

        mock_paginator = Mock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/file.txt"}]}
        ]

        _download_s3_artifacts(
            "s3://{}/prefix".format(DEFAULT_BUCKET), str(tmp_path), mock_session
        )

        # Verify list_objects_v2 pagination included ExpectedBucketOwner
        paginate_kwargs = mock_paginator.paginate.call_args[1]
        assert paginate_kwargs["ExpectedBucketOwner"] == ACCOUNT_ID

        # Verify download_file included ExpectedBucketOwner
        dl_args = mock_s3_client.download_file.call_args
        assert dl_args[1]["ExtraArgs"] == {"ExpectedBucketOwner": ACCOUNT_ID}
