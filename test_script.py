import sagemaker, boto3, json
from sagemaker import get_execution_role
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.session import Session
from sagemaker.session_settings import SessionSettings
from sagemaker.utils import name_from_base


aws_role = get_execution_role()
aws_region = boto3.Session().region_name
sess = sagemaker.Session()

model_id, model_version, = (
    "pytorch-ic-mobilenet-v2",
    "*",
)

inference_instance_type = "ml.m5.xlarge"

# Retrieve the inference docker container uri.
deploy_image_uri = image_uris.retrieve(
    region=None,
    framework=None,
    image_scope="inference",
    model_id=model_id,
    model_version=model_version,
    instance_type=inference_instance_type,
)
# Retrieve the inference script uri.
deploy_source_uri = script_uris.retrieve(
    model_id=model_id, model_version=model_version, script_scope="inference"
)
# Retrieve the base model uri.
base_model_uri = model_uris.retrieve(
    model_id=model_id, model_version=model_version, model_scope="inference"
)

sm_session = Session()
sm_session.settings._local_download_dir = "/home/evakravi/workspace/github/sdk_change_download_directory/sagemaker-python-sdk/temp"
# Create the SageMaker model instance. Note that we need to pass Predictor class when we deploy model through Model class,
# for being able to run inference through the sagemaker API.
model = Model(
    image_uri=deploy_image_uri,
    source_dir=deploy_source_uri,
    model_data=base_model_uri,
    entry_point="inference.py",
    role=aws_role,
    sagemaker_session=sm_session,

)
# deploy the Model.
base_model_predictor = model.deploy(
    initial_instance_count=1,
    instance_type=inference_instance_type,
)
