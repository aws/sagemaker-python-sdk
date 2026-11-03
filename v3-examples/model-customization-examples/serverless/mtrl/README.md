# Multi-turn RL with SageMaker AI: an end-to-end SOP-Bench example

A complete, working multi-turn reinforcement learning (MTRL) example. It fine-tunes
`openai-reasoning-gpt-oss-20b` to follow a multi-step Standard Operating Procedure with tool calls
on the [SOP-Bench](https://github.com/amazon-science/SOP-Bench) `aircraft_inspection` task. The
rollout agent is hosted on Amazon Bedrock AgentCore. Unlike the template-style
`mtrl_finetuning_example_notebook_v3_prod.ipynb`, every piece here (environment, tools, reward,
agent, data) is real code that you can run as-is and then adapt to your own task.

| # | Notebook | What you do |
|---|---|---|
| 1 | [`01_build_and_deploy_agent.ipynb`](01_build_and_deploy_agent.ipynb) | Explore the task, tools, and reward (including the reward hack it prevents), then deploy the rollout agent to AgentCore |
| 2 | [`02_train.ipynb`](02_train.ipynb) | Build the prompt dataset, launch a GRPO job with `MultiTurnRLTrainer`, and read the metrics and trajectories in MLflow |
| 3 | [`03_evaluate_and_deploy.ipynb`](03_evaluate_and_deploy.ipynb) | Compare base and fine-tuned models with `MultiTurnRLEvaluator`, diagnose failures, deploy, and clean up |

## Quick start

1. **Use a sandbox/dev account** in a region with SageMaker AI multi-turn RL and Amazon Bedrock
   AgentCore (tested in `us-east-1`; set `AWS_REGION`). Deploy the prerequisites stack once. It creates
   the job role, the agent execution role, an S3 bucket, an MLflow app, and two managed policies:
   ```bash
   aws cloudformation deploy --stack-name mtrl-prerequisites \
     --template-file prerequisites.yaml --capabilities CAPABILITY_NAMED_IAM
   ```
   Accounts allow 3 MLflow apps by default. To reuse an existing app, add
   `--parameter-overrides ExistingMlflowAppArn=<mlflow-app ARN>`.
2. **Grant permissions to the identity that runs the notebooks.** Attach `AmazonSageMakerFullAccess`,
   `BedrockAgentCoreFullAccess`, and the stack outputs `NotebookPolicyArn` and
   `AgentCoreToolkitPolicyArn`.
3. **Run the notebooks in order** from this folder, using Python 3.10+ (3.11 recommended).
   Notebook 1 installs the dependencies. Docker is not needed, because the agent image builds on
   AWS CodeBuild.

The default run is 10 training steps, which takes well under an hour (about 3-4 minutes per step).
Lower `max_steps` in notebook 2 for a quicker pass. The longer 30-step recipe in
`sopbench_agent/scripts/run_rft_training.py` takes about 2 h 15 min.

## What's in this folder

```
prerequisites.yaml        CloudFormation: IAM roles, S3 bucket, MLflow app, notebook policies
sopbench_agent/           the rollout agent (an installable package + AgentCore Dockerfile)
  src/sopbench_agentcore/   main.py (entrypoint), openai_agent.py (agent loop), tools.py,
                            prompts.py, reward.py, config.py, samples.py
  scripts/                  deploy.sh, preprocess.py, inspect_reward.py, test_local.py,
                            run_rft_training.py (non-interactive training), run_manual_eval.py
images/                   figures used by the notebooks
```

## Adapting it to your task

Keep the trainer, the evaluator, and the agent structure. Replace these task-specific parts:
* `tools.py`: the environment and its tools;
* `prompts.py`: the output contract;
* `reward.py`: return `{"reward": <float>}` (or `{"status": "error"}` to fail the rollout);
* `scripts/preprocess.py`: one `prompt` column, which the service forwards verbatim to the agent.

This example is built for SOP-Bench's multi-field `aircraft_inspection` benchmark. Moving to another
benchmark means updating `SUPPORTED_BENCHMARKS` and `BENCHMARK_NAME` (`config.py`, `Dockerfile`),
`EXCLUDED_FIELDS` (`samples.py`), the example answer in `prompts.py`, and the dataset paths in
`scripts/preprocess.py`, the notebooks, and `scripts/run_rft_training.py`.

Agents hosted on Amazon EKS, Amazon EC2, AWS Fargate, or your own infrastructure use the same
contract through a Lambda forwarder. See
[Preparing your agent](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl-agent.html).

## Cost and cleanup

You pay for the following until you delete them:
* the training and evaluation jobs;
* AgentCore invocations;
* the CodeBuild image build;
* S3 storage;
* the MLflow app;
* any endpoint you deploy.

Section 9 of notebook 3 tears everything down.

## Third-party data

The agent installs `amazon-sop-bench` from
[github.com/amazon-science/SOP-Bench](https://github.com/amazon-science/SOP-Bench), which is
licensed under CC-BY-NC-4.0, and uses its benchmark data at runtime. No SOP-Bench data is stored in
this folder.
