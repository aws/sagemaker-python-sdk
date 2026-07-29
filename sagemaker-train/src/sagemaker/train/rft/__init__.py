"""SageMaker RFT SDK - Integration library for multi-turn RL training platform.

Strands + AgentCore (simplest)::

    from sagemaker.train.rft import sagemaker_rft_handler, RolloutFeedbackClient
    from sagemaker.train.rft.adapters.strands import wrap_model

    model = wrap_model(OpenAIModel(...))

    @app.entrypoint
    @sagemaker_rft_handler
    async def invoke_agent(payload):
        result = await agent.invoke_async(payload["instance"])
        return result

Strands Standalone::

    from sagemaker.train.rft import set_rollout_context, RolloutFeedbackClient
    from sagemaker.train.rft.adapters.strands import wrap_model

    model = wrap_model(model)

    @app.post("/rollout")
    def rollout(request):
        set_rollout_context(request.metadata, request.inference_params)
        result = agent(request.instance)
        RolloutFeedbackClient(request.metadata).report_complete(reward)

Custom Integration::

    from sagemaker.train.rft import make_inference_headers, RolloutFeedbackClient

    @app.post("/rollout")
    def handle(request):
        headers = make_inference_headers(request.metadata)
        client = OpenAI(base_url=endpoint, default_headers=headers)
        result = my_agent.run(request.instance, client)
        RolloutFeedbackClient(request.metadata).report_complete(reward)
"""

from sagemaker.train.rft.headers import make_inference_headers, get_inference_headers
from sagemaker.train.rft.feedback import RolloutFeedbackClient
from sagemaker.train.rft.models import RolloutMetadata, RolloutRequest, InferenceParams
from sagemaker.train.rft.decorators import sagemaker_rft_handler
from sagemaker.train.rft.context import set_rollout_context, clear_rollout_context, get_inference_params

__all__ = [
    "make_inference_headers",
    "get_inference_headers",
    "RolloutFeedbackClient",
    "RolloutMetadata",
    "RolloutRequest",
    "InferenceParams",
    "sagemaker_rft_handler",
    "set_rollout_context",
    "clear_rollout_context",
    "get_inference_params",
]

__version__ = "0.1.0"
