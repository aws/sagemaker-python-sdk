"""SageMaker RFT SDK - Integration library for multi-turn RL training platform.

Strands + AgentCore (simplest)::

    from sagemaker.rft import rft_handler, RolloutFeedbackClient
    from sagemaker.rft.adapters.strands import wrap_model

    model = wrap_model(OpenAIModel(...))

    @app.entrypoint
    @rft_handler
    async def invoke_agent(payload):
        result = await agent.invoke_async(payload["instance"])
        return result

Strands Standalone::

    from sagemaker.rft import set_rollout_context, RolloutFeedbackClient
    from sagemaker.rft.adapters.strands import wrap_model

    model = wrap_model(model)

    @app.post("/rollout")
    def rollout(request):
        set_rollout_context(request.metadata, request.inference_params)
        result = agent(request.instance)
        RolloutFeedbackClient(request.metadata).report_complete(reward)

Custom Integration::

    from sagemaker.rft import make_inference_headers, RolloutFeedbackClient

    @app.post("/rollout")
    def handle(request):
        headers = make_inference_headers(request.metadata)
        client = OpenAI(base_url=endpoint, default_headers=headers)
        result = my_agent.run(request.instance, client)
        RolloutFeedbackClient(request.metadata).report_complete(reward)
"""

from sagemaker.rft.headers import make_inference_headers, get_inference_headers
from sagemaker.rft.feedback import RolloutFeedbackClient
from sagemaker.rft.models import RolloutMetadata, RolloutRequest, InferenceParams
from sagemaker.rft.decorators import rft_handler
from sagemaker.rft.context import set_rollout_context, clear_rollout_context, get_inference_params

__all__ = [
    "make_inference_headers",
    "get_inference_headers",
    "RolloutFeedbackClient",
    "RolloutMetadata",
    "RolloutRequest",
    "InferenceParams",
    "rft_handler",
    "set_rollout_context",
    "clear_rollout_context",
    "get_inference_params",
]

__version__ = "0.1.0"
