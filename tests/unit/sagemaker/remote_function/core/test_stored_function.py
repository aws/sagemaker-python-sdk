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

import pytest
import random
import string
from mock import MagicMock, Mock, patch
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments.run import Run
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent

from sagemaker.remote_function.core.stored_function import StoredFunction
from sagemaker.remote_function.core.serialization import deserialize_obj_from_s3
from sagemaker.remote_function.errors import SerializationError
from tests.unit.sagemaker.experiments.helpers import (
    TEST_EXP_DISPLAY_NAME,
    TEST_EXP_NAME,
    TEST_RUN_DISPLAY_NAME,
    TEST_RUN_NAME,
    mock_tc_load_or_create_func,
    mock_trial_load_or_create_func,
)

KMS_KEY = "kms-key"
HMAC_KEY = "some-hmac-key"

mock_s3 = {}


def random_s3_uri():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def upload_bytes(b, s3_uri, kms_key=None, sagemaker_session=None):
    assert kms_key == KMS_KEY
    mock_s3[s3_uri] = b


def read_bytes(s3_uri, sagemaker_session=None):
    return mock_s3[s3_uri]


def quadratic(x=2, *, a=1, b=0, c=0):
    return a * x * x + b * x + c


def log_bigger(a, b, run: Run):
    if a >= b:
        run.log_metric("bigger", a)
    else:
        run.log_metric("bigger", b)


@pytest.mark.parametrize(
    "args, kwargs",
    [([], {}), ([3], {}), ([], {"a": 2, "b": 1, "c": 1})],
)
@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload_bytes)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read_bytes)
@patch("sagemaker.s3.S3Uploader.upload")
@patch("sagemaker.s3.S3Downloader.download")
def test_save_and_load(s3_source_dir_download, s3_source_dir_upload, args, kwargs):
    session = Mock()
    s3_base_uri = random_s3_uri()

    stored_function = StoredFunction(
        sagemaker_session=session, s3_base_uri=s3_base_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )
    stored_function.save(quadratic, *args, **kwargs)
    stored_function.load_and_invoke()

    assert deserialize_obj_from_s3(
        session, s3_uri=f"{s3_base_uri}/results", hmac_key=HMAC_KEY
    ) == quadratic(*args, **kwargs)


@patch(
    "sagemaker.experiments.run.Experiment._load_or_create",
    MagicMock(return_value=Experiment(experiment_name=TEST_EXP_NAME)),
)
@patch(
    "sagemaker.experiments.run._Trial._load_or_create",
    MagicMock(side_effect=mock_trial_load_or_create_func),
)
@patch.object(_Trial, "add_trial_component", MagicMock(return_value=None))
@patch(
    "sagemaker.experiments.run._TrialComponent._load_or_create",
    MagicMock(side_effect=mock_tc_load_or_create_func),
)
@patch("sagemaker.s3.S3Uploader.upload_bytes", new=upload_bytes)
@patch("sagemaker.s3.S3Downloader.read_bytes", new=read_bytes)
@patch.object(_TrialComponent, "save")
@patch("sagemaker.s3.S3Uploader.upload")
@patch("sagemaker.s3.S3Downloader.download")
def test_save_with_parameter_of_run_type(
    s3_source_dir_download, s3_source_dir_upload, mock_tc_save
):
    session = Mock()
    s3_base_uri = random_s3_uri()
    session.sagemaker_client.search.return_value = {"Results": []}

    run = Run(
        experiment_name=TEST_EXP_NAME,
        run_name=TEST_RUN_NAME,
        experiment_display_name=TEST_EXP_DISPLAY_NAME,
        run_display_name=TEST_RUN_DISPLAY_NAME,
        sagemaker_session=session,
    )
    stored_function = StoredFunction(
        sagemaker_session=session, s3_base_uri=s3_base_uri, s3_kms_key=KMS_KEY, hmac_key=HMAC_KEY
    )
    with pytest.raises(SerializationError) as e:
        stored_function.save(log_bigger, 1, 2, run)
        assert isinstance(e.__cause__, NotImplementedError)
