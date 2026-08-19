import sagemaker
from sagemaker import base_serializers


def test_import():
    base_serializers.BaseSerializer()
    sagemaker.Session()
