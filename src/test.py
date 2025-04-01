from sagemaker.jumpstart.hub.hub import Hub
from sagemaker import hyperparameters
from sagemaker.session import Session
from sagemaker.jumpstart.estimator import JumpStartEstimator


hub = Hub(hub_name="temp-bencrab-hub", sagemaker_session=Session())

# hub.create(description="hello haha")

model_id = "meta-vlm-llama-3-2-11b-vision"
model_version = "*"
hub_arn = hub.hub_name

my_hyperparameters = hyperparameters.retrieve_default(
    model_id=model_id, model_version=model_version, hub_arn=hub_arn
)
print(my_hyperparameters)
hyperparameters.validate(
    model_id=model_id,
    model_version=model_version,
    hyperparameters=my_hyperparameters,
    hub_arn=hub_arn,
)
estimator = JumpStartEstimator(
    model_id=model_id,
    hub_name=hub_arn,
    model_version=model_version,
    environment={"accept_eula": "true"},  # Please change {"accept_eula": "true"}
    disable_output_compression=True,
    instance_type="ml.p4d.24xlarge",
    hyperparameters=my_hyperparameters,
)
estimator.fit(
    {"training": "s3://jumpstart-cache-prod-us-west-2/training-datasets/docVQA-small-3000ex/"}
)
