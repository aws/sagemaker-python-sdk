import os
import pytest
from unittest.mock import Mock


@pytest.fixture
def base_config_with_schema():
    return {"SchemaVersion": "1.0"}


@pytest.fixture
def base_local_mode_config():
    return {"local": {"local_code": True, "region_name": "us-west-2"}}


@pytest.fixture
def valid_iam_role_arn():
    return "arn:aws:iam::012345678901:role/SageMakerRole"


@pytest.fixture
def valid_vpc_config():
    return {"SecurityGroupIds": ["sg-12345"], "Subnets": ["subnet-12345"]}


@pytest.fixture
def valid_config_with_all_the_scopes(get_data_dir):
    import yaml
    config_file_path = os.path.join(get_data_dir, "config.yaml")
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("SageMaker", {})


@pytest.fixture
def valid_feature_group_config():
    return {
        "OnlineStoreConfig": {"SecurityConfig": {"KmsKeyId": "somekmskeyid"}},
        "OfflineStoreConfig": {"S3StorageConfig": {"KmsKeyId": "someotherkmskeyid"}},
    }


@pytest.fixture
def valid_edge_packaging_config(valid_iam_role_arn):
    return {"OutputConfig": {"KmsKeyId": "somekeyid"}, "RoleArn": valid_iam_role_arn}


@pytest.fixture
def valid_training_job_config(valid_iam_role_arn, valid_vpc_config):
    return {
        "EnableNetworkIsolation": True,
        "OutputDataConfig": {"KmsKeyId": "somekmskey"},
        "ResourceConfig": {"VolumeKmsKeyId": "somevolumekmskey"},
        "RoleArn": valid_iam_role_arn,
        "VpcConfig": valid_vpc_config,
    }


@pytest.fixture
def valid_processing_job_config(valid_iam_role_arn):
    return {"RoleArn": valid_iam_role_arn}


@pytest.fixture
def valid_model_package_config():
    return {}


@pytest.fixture
def valid_model_config(valid_iam_role_arn):
    return {"ExecutionRoleArn": valid_iam_role_arn}


@pytest.fixture
def valid_pipeline_config(valid_iam_role_arn):
    return {"RoleArn": valid_iam_role_arn}


@pytest.fixture
def valid_compilation_job_config(valid_iam_role_arn):
    return {"RoleArn": valid_iam_role_arn}


@pytest.fixture
def valid_transform_job_config():
    return {}


@pytest.fixture
def valid_automl_config(valid_iam_role_arn):
    return {"RoleArn": valid_iam_role_arn}


@pytest.fixture
def valid_endpointconfig_config():
    return {}


@pytest.fixture
def valid_monitoring_schedule_config(valid_iam_role_arn):
    return {
        "MonitoringScheduleConfig": {
            "MonitoringJobDefinition": {"RoleArn": valid_iam_role_arn}
        }
    }


@pytest.fixture
def valid_remote_function_config():
    return {"S3RootUri": "s3://my-bucket/my-prefix"}


@pytest.fixture
def get_data_dir():
    return os.path.join(os.path.dirname(__file__), "../../data/config")


@pytest.fixture
def s3_resource_mock():
    return Mock()
