"""
Test fixtures and mock helpers for ModelBuilder unit tests.
Based on patterns from legacy PySDK tests.
"""

from unittest.mock import Mock, MagicMock


# Mock constants
MOCK_IMAGE_CONFIG = {"RepositoryAccessMode": "Vpc"}
MOCK_VPC_CONFIG = {"Subnets": ["subnet-1234"], "SecurityGroupIds": ["sg123"]}
MOCK_REGION = "us-west-2"
MOCK_ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerRole"
MOCK_S3_URI = "s3://test-bucket/model.tar.gz"
MOCK_INSTANCE_TYPE = "ml.m5.xlarge"
MOCK_IMAGE_URI = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.8.0-gpu-py3"


def mock_sagemaker_session():
    """Create a properly mocked SageMaker session for testing."""
    session = Mock()
    
    # Basic session attributes
    session.settings = Mock()
    session.settings.include_jumpstart_tags = False
    session.settings._local_download_dir = None
    session.boto_region_name = MOCK_REGION
    session.sagemaker_config = {}
    session.config = {}
    session._append_sagemaker_config_tags = Mock(return_value=[])
    session.default_bucket_prefix = "test-prefix"
    session.default_bucket = Mock(return_value="test-bucket")
    session.local_mode = False
    
    # Boto session mock
    session.boto_session = Mock()
    session.boto_session.region_name = MOCK_REGION
    
    # Credentials mock
    mock_credentials = Mock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None
    session.boto_session.get_credentials = Mock(return_value=mock_credentials)
    
    # Client mocks
    def mock_client(service_name, **kwargs):
        client = Mock()
        
        if service_name == "sagemaker":
            # SageMaker client methods
            client.describe_endpoint = Mock(return_value={
                'EndpointName': 'test-endpoint',
                'EndpointArn': 'arn:aws:sagemaker:us-west-2:123456789012:endpoint/test',
                'EndpointStatus': 'InService',
                'CreationTime': '2024-01-01T00:00:00Z',
                'LastModifiedTime': '2024-01-01T00:00:00Z',
                'ProductionVariants': []
            })
            
            client.describe_model = Mock(return_value={
                'ModelName': 'test-model',
                'ModelArn': 'arn:aws:sagemaker:us-west-2:123456789012:model/test',
                'CreationTime': '2024-01-01T00:00:00Z',
                'ExecutionRoleArn': MOCK_ROLE_ARN,
                'PrimaryContainer': {
                    'Image': MOCK_IMAGE_URI,
                    'ModelDataUrl': MOCK_S3_URI
                }
            })
            
            client.create_model = Mock(return_value={
                'ModelArn': 'arn:aws:sagemaker:us-west-2:123456789012:model/test'
            })
            
            client.create_endpoint_config = Mock(return_value={
                'EndpointConfigArn': 'arn:aws:sagemaker:us-west-2:123456789012:endpoint-config/test'
            })
            
            client.create_endpoint = Mock(return_value={
                'EndpointArn': 'arn:aws:sagemaker:us-west-2:123456789012:endpoint/test'
            })
            
            client.describe_inference_component = Mock(return_value={
                'InferenceComponentName': 'test-ic',
                'InferenceComponentArn': 'arn:aws:sagemaker:us-west-2:123456789012:inference-component/test',
                'InferenceComponentStatus': 'InService'
            })
            
        elif service_name == "sts":
            # STS client methods
            client.get_caller_identity = Mock(return_value={
                'UserId': 'AIDACKCEVSQ6C2EXAMPLE',
                'Account': '123456789012',
                'Arn': MOCK_ROLE_ARN
            })
        
        return client
    
    session.boto_session.client = mock_client
    session.sagemaker_client = mock_client("sagemaker")
    
    # Session helper methods
    session.endpoint_in_service_or_not = Mock(return_value=False)
    session.endpoint_from_production_variants = Mock()
    session.create_endpoint_config = Mock(return_value="test-endpoint-config")
    session.update_endpoint = Mock()
    session.create_inference_component = Mock()
    session.describe_inference_component = Mock(return_value={
        'InferenceComponentName': 'test-ic',
        'InferenceComponentStatus': 'InService'
    })
    session.update_inference_component = Mock()
    session.get_caller_identity_arn = Mock(return_value=MOCK_ROLE_ARN)
    
    return session


def mock_model_object():
    """Create a mock model object for testing."""
    model = Mock()
    model.__class__.__module__ = "torch.nn"
    model.__class__.__name__ = "Module"
    return model


def mock_inference_spec():
    """Create a mock InferenceSpec for testing."""
    spec = Mock()
    spec.load = Mock()
    spec.invoke = Mock(return_value={"predictions": [1, 2, 3]})
    return spec


def mock_schema_builder():
    """Create a mock SchemaBuilder for testing."""
    schema = Mock()
    schema.sample_input = {"inputs": "test input"}
    schema.sample_output = [{"generated_text": "test output"}]
    schema.input_serializer = Mock()
    schema.output_deserializer = Mock()
    return schema


def mock_core_model():
    """Create a mock sagemaker.core.resources.Model for testing."""
    from sagemaker.core.utils.utils import Unassigned
    
    model = Mock()
    model.model_name = "test-model"
    model.model_arn = "arn:aws:sagemaker:us-west-2:123456789012:model/test"
    model.execution_role_arn = MOCK_ROLE_ARN
    model.containers = []
    
    # Primary container
    container = Mock()
    container.image = MOCK_IMAGE_URI
    container.model_data_url = MOCK_S3_URI
    container.environment = {"KEY": "value"}
    container.image_config = Unassigned()
    model.primary_container = container
    
    return model


def mock_endpoint():
    """Create a mock sagemaker.core.resources.Endpoint for testing."""
    endpoint = Mock()
    endpoint.endpoint_name = "test-endpoint"
    endpoint.endpoint_arn = "arn:aws:sagemaker:us-west-2:123456789012:endpoint/test"
    endpoint.endpoint_status = "InService"
    endpoint.invoke = Mock(return_value={"predictions": [1, 2, 3]})
    return endpoint


def mock_uploaded_code():
    """Create a mock UploadedCode object."""
    from sagemaker.core import fw_utils
    return fw_utils.UploadedCode(
        s3_prefix="s3://test-bucket/code",
        script_name="inference.py"
    )
