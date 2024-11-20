from aws_cdk import (
    Stack,
    aws_sagemaker as sagemaker,
    CfnOutput,
    App,
)
from constructs import Construct
import json

class SageMakerEndpointStack(Stack):

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
            endpoint_config_name="test-endpoint-config-11-20-001"
        )

        # Create SageMaker Endpoint
        endpoint = sagemaker.CfnEndpoint(
            self, "MyEndpoint",
            endpoint_config_name=endpoint_config.endpoint_config_name,
            endpoint_name="test-endpoint-11-20"
        )

        # Ensure the Endpoint is created after the EndpointConfig
        endpoint.add_dependency(endpoint_config)

        # Output the EndpointConfig name
        CfnOutput(self, "EndpointConfigName", value=endpoint_config.endpoint_config_name)

        # Output the Endpoint name
        CfnOutput(self, "EndpointName", value=endpoint.endpoint_name)

def generate_cfn_template():
    app = App()
    stack = SageMakerEndpointStack(app, "SageMakerEndpointStack")

    # Synthesize the stack to a CloudFormation template
    cfn_template = app.synth().get_stack_by_name(stack.stack_name).template

    # Save the CloudFormation template as a JSON file
    with open("sagemaker_endpoint_template.json", "w") as f:
        json.dump(cfn_template, f, indent=2)
    
    print("CloudFormation template saved as sagemaker_endpoint_template.json")
    
    return cfn_template

if __name__ == "__main__":
    generate_cfn_template()
    print("CDK synthesis complete")
