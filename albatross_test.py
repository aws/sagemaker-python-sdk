import sys
print(sys.path)
sys.path.append("/home/upravali/telemetry/sagemaker-python-sdk/src/sagemaker")
sys.path.append('/home/upravali/langchain/langchain-aws/libs/aws/')
print("Updated sys.path: ", sys.path)

import json
import os
import time

from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
import langchain_aws
import langchain_core
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

INPUTS = {
    'CPU': {
        'INFERENCE_IMAGE': '763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.4.0-cpu-py311-ubuntu22.04-sagemaker',
        'INSTANCE_TYPE': 'ml.m5.xlarge'
    },
    'GPU': {
        'INFERENCE_IMAGE': '763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker',
        'INSTANCE_TYPE': 'ml.g5.xlarge' 
    },
    'SERVICE': { 
        'ROLE': 'arn:aws:iam::971812153697:role/upravali-test-role'
    }
}

def deploy(device):

    class CustomerInferenceSpec(InferenceSpec):

        def load(self, model_dir): 
            from langchain_aws import ChatBedrockConverse
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            return \
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a verbose assistant that gives long-winded responses at least 500 words long for every comment/question.",
                        ),
                        ("human", "{input}"),
                    ]
                ) | \
                ChatBedrockConverse(
                    model = 'anthropic.claude-3-sonnet-20240229-v1:0',
                    temperature = 0,
                    region_name = 'us-west-2'
                ) | \
                StrOutputParser()

        def invoke(self, x, model):
            return model.invoke({'input': x['input']}) if x['stream'].lower() != 'true' \
                else model.stream({'input': x['input']})
        


    model = ModelBuilder(
        ##################################################################
        # can be service or customer who defines these
        ##################################################################
        name = f'model-{int(time.time())}',

        ##################################################################
        # service should define these
        ##################################################################
        image_uri = INPUTS[device]['INFERENCE_IMAGE'],
        env_vars = {
            'TS_DISABLE_TOKEN_AUTHORIZATION' : 'true' # ABSOLUTELY NECESSARY
        },

        ##################################################################
        # customer should define these
        ##################################################################
        schema_builder = SchemaBuilder(
            json.dumps({
                'stream': 'true',
                'input': 'hello'
            }),
            "<EOF>"
        ),
        inference_spec = CustomerInferenceSpec(), # Won't be pickled correctly if Python version locally and DLC don't match
        dependencies = {
            "auto": True,
            # 'requirements' : './inference/code/requirements2.txt'
        },
        role_arn = INPUTS['SERVICE']['ROLE']
    ).build()
    endpoint = model.deploy(
        initial_instance_count = 1,
        instance_type = INPUTS[device]['INSTANCE_TYPE'],
    )
    return (model, endpoint)


###################################################################################################
#
#
# PoC DEMO CODE ONLY
#
# Note: invoke vs invoke_stream matters  
###################################################################################################
def invoke(endpoint, x):
    res = endpoint.predict(x)
    return res

def invoke_stream(endpoint, x):
    res = endpoint.predict_stream(x)
    print(str(res)) # Generator
    return res

def clean(model, endpoint):
    try:
        endpoint.delete_endpoint()
    except Exception as e:
        print(e)
        pass

    try:
        model.delete_model()
    except Exception as e:
        print(e)
        pass

def main(device):
    print("before deploying")
    model, endpoint = deploy(device)
    print("after deploying")

    while True:
        x = input(f">>> ")
        if x == 'exit':
            break
        try:
            if json.loads(x)['stream'].lower() == 'true':
                for chunk in invoke_stream(endpoint, x):
                    print(
                        str(chunk, encoding = 'utf-8'), 
                        end = "", 
                        flush = True
                    )
                print()
            else:
                print(invoke(endpoint, x))
        except Exception as e:
            print(e)

    clean(model, endpoint)

if __name__ == '__main__':
    os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
    main('CPU')