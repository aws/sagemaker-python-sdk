from __future__ import absolute_import

import boto3
import pytest

from sagemaker.core.utils.utils import SageMakerClient

PASSED_KEY = "AKIAPASSEDSESSION"


@pytest.fixture
def client_holder():
    SageMakerClient.reset()
    session = boto3.Session(
        aws_access_key_id=PASSED_KEY,
        aws_secret_access_key="test-secret",
        region_name="us-west-2",
    )
    yield SageMakerClient(session=session, region_name="us-west-2")
    SageMakerClient.reset()


@pytest.mark.parametrize(
    "service",
    ["sagemaker", "sagemaker-runtime", "sagemaker-featurestore-runtime", "sagemaker-metrics"],
)
def test_clients_sign_with_passed_session_credentials(client_holder, service):
    client = client_holder.get_client(service)
    signing_key = client._request_signer._credentials.get_frozen_credentials().access_key
    assert signing_key == PASSED_KEY
