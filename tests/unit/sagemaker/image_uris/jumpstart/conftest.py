from mock.mock import Mock
import pytest

REGION_NAME = "us-west-2"
BUCKET_NAME = "some-bucket-name"


@pytest.fixture(scope="module")
def session():
    boto_mock = Mock(region_name=REGION_NAME)
    sms = Mock(
        boto_session=boto_mock,
        boto_region_name=REGION_NAME,
        config=None,
    )
    sms.default_bucket = Mock(return_value=BUCKET_NAME)
    return sms
