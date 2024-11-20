from aws_cdk import (
    Stack,
    aws_sagemaker as sagemaker,
    CfnOutput,
    App,
)
from constructs import Construct
from aws_cdk.cloudformation_include import CfnInclude
import json

class SageMakerEndpointConfigStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create SageMaker EndpointConfig
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "MyEndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    initial_instance_count=1,
                    instance_type="ml.t2.medium",
                    model_name="model-name-1348706a967911ef97190653ff4a3db9",
                    variant_name="AllTraffic",
                    initial_variant_weight=1.0
                )
            ],
            endpoint_config_name="test-endpoint-config-11-20"
        )

        # Output the EndpointConfig name
        CfnOutput(self, "EndpointConfigName", value=endpoint_config.endpoint_config_name)

app = App()
stack = SageMakerEndpointConfigStack(app, "SageMakerEndpointConfigStack")

# Synthesize the stack to a CloudFormation template
cfn_template = app.synth().get_stack_by_name(stack.stack_name).template

# Save the CloudFormation template as a JSON file
with open("sagemaker_endpoint_config_template.json", "w") as f:
    json.dump(cfn_template, f, indent=2)
print("cccccc")
app.synth()
