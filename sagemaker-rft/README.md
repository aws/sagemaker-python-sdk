# SageMaker RFT SDK

Integration SDK for the multi-turn reinforcement fine-tuning (RFT) platform on Amazon SageMaker.

## Installation

```bash
pip install sagemaker-rft
```

With framework adapters:
```bash
pip install sagemaker-rft[strands]    # Strands framework support
pip install sagemaker-rft[langchain]  # LangChain framework support
```

## Integration Patterns

### Strands + AgentCore (simplest)

```python
from sagemaker.rft import rft_handler, RolloutFeedbackClient
from sagemaker.rft.adapters.strands import wrap_model

model = OpenAIModel(client_args={"base_url": "$TRAINING_INFERENCE_ENDPOINT"}, model_id="$TRAINING_MODEL_NAME")
model = wrap_model(model)
agent = Agent(model=model, tools=[...])

@app.entrypoint
@rft_handler
async def invoke_agent(payload):
    result = await agent.invoke_async(payload["instance"])
    return result.message["content"][0]["text"]
```

### Strands Standalone

```python
from sagemaker.rft import set_rollout_context, RolloutFeedbackClient
from sagemaker.rft.adapters.strands import wrap_model

model = wrap_model(OpenAIModel(...))
agent = Agent(model=model, tools=[...])

@app.post("/rollout")
def rollout(request: RolloutRequest):
    set_rollout_context(request.metadata, request.inference_params)
    try:
        result = agent(request.instance["instruction"])
        reward = compute_reward(result)
        RolloutFeedbackClient(request.metadata).report_complete(reward)
    except Exception:
        RolloutFeedbackClient(request.metadata).report_error()
        raise
    return {"status": "ok"}
```

### Custom Integration

```python
from sagemaker.rft import make_inference_headers, RolloutFeedbackClient, RolloutRequest

@app.post("/rollout")
def rollout(request: RolloutRequest):
    headers = make_inference_headers(request.metadata)
    client = OpenAI(base_url=endpoint, default_headers=headers)
    result = my_agent.run(request.instance, client)
    RolloutFeedbackClient(request.metadata).report_complete(compute_reward(result))
    return {"status": "ok"}
```
