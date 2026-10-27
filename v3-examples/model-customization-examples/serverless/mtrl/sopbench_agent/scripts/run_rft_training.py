#!/usr/bin/env python3
"""Launch a full SOP-Bench aircraft_inspection GPT-OSS run against your AgentCore-hosted agent.

Non-interactive equivalent of ../02_train.ipynb with a longer recipe. Account-specific values are
resolved at runtime:

  * prompts bucket, job role, and MLflow app ARN from the `mtrl-prerequisites` stack outputs
    (override with --stack-name or pass the values explicitly);
  * the AgentCore runtime ARN from the READY runtime named AGENT_NAME (default
    `sopbench_agentcore`), or --agent-runtime-arn.

Exits non-zero if the job does not complete successfully.

Run from sopbench_agent/ with PYTHONPATH=src. Prereqs: the prerequisites stack, the deployed
agent (./scripts/deploy.sh), the uploaded prompts
(scripts/preprocess.py --s3-prefix ...), and AWS credentials for the region (AWS_REGION).
"""

import argparse
import os
import sys

import boto3
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

from sopbench_agentcore.reward import FINAL_REWARD_SCALE

DEFAULT_MODEL = "openai-reasoning-gpt-oss-20b"
DEFAULT_STACK_NAME = "mtrl-prerequisites"
AGENT_NAME = os.environ.get("AGENT_NAME", "sopbench_agentcore")  # `agentcore configure --name`
PREFIX = "sopbench-mtrl/aircraft"  # must match scripts/preprocess.py --s3-prefix

HYPERPARAMETERS = {
    "global_batch_size": 64,
    "group_size": 4,
    "advantage_method": "group_based",  # GRPO
    "loss_fn": "ppo",
    "clip_low_threshold": 0.8,
    "clip_high_threshold": 1.2,
    "temperature": 1.0,
    "sampling_top_p": 1.0,
    "sampling_max_tokens": 8192,
    "max_epochs": 15,
    "max_steps": 30,
    "val_every": 10,
    # pass@k counts a rollout as successful when reward >= success_threshold. With rewards in
    # [-1, 5], only a fully correct answer (5.0) is a task success.
    "success_threshold": FINAL_REWARD_SCALE,
    "lora_rank": 32,
    "lora_alpha": 64,
    "learning_rate": 4e-5,
    # AgentCore has no server-side concurrency cap, so this is the only throttle on concurrent
    # inference connections. Lower it toward 32 if you see httpx ConnectTimeout errors.
    "rollout_max_concurrency": 64,
    "rollout_timeout": 600,
    "rollout_max_retries": 3,
}


def _stack_outputs(session, stack_name):
    try:
        stack = session.client("cloudformation").describe_stacks(StackName=stack_name)["Stacks"][0]
    except Exception as e:  # noqa: BLE001
        print(f"(stack '{stack_name}' not found: {e})")
        return {}
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def _find_agent_runtime_arn(session):
    """ARN of the READY runtime named AGENT_NAME (exact match; the SDK call paginates)."""
    for r in MultiTurnRLTrainer.list_bedrock_agentcore_runtimes(session=session):
        if r["name"] == AGENT_NAME and r["status"] == "READY":
            return r["arn"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stack-name", default=DEFAULT_STACK_NAME)
    ap.add_argument("--agent-runtime-arn", default=None, help="default: auto-discover")
    ap.add_argument("--bucket", default=None, help="default: stack output PromptsBucketName")
    ap.add_argument("--role-arn", default=None, help="default: stack output JobRoleArn")
    ap.add_argument("--mlflow-app-arn", default=None, help="default: stack output MlflowAppArn")
    ap.add_argument("--attach", metavar="JOB_NAME", default=None, help="monitor an existing job")
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="wait timeout in seconds")
    args = ap.parse_args()

    session = boto3.session.Session()
    if not session.region_name:
        raise SystemExit("Set AWS_REGION (or a default region) to the region of your agent.")

    if args.attach:
        job = MultiTurnRLTrainer.attach(job_name=args.attach)
    else:
        if args.model not in MultiTurnRLTrainer.list_supported_models():
            raise SystemExit(f"{args.model} is not a supported MTRL model")
        outs = _stack_outputs(session, args.stack_name)
        bucket = args.bucket or outs.get("PromptsBucketName")
        cfg = {
            "agent_env": args.agent_runtime_arn or _find_agent_runtime_arn(session),
            "role": args.role_arn or outs.get("JobRoleArn"),
            "mlflow_app_arn": args.mlflow_app_arn or outs.get("MlflowAppArn"),
            "bucket": bucket,
        }
        missing = [k for k, v in cfg.items() if not v]
        if missing:
            raise SystemExit(f"Could not resolve {missing}; pass them explicitly (see --help).")
        trainer = MultiTurnRLTrainer(
            model=args.model,
            agent_env=cfg["agent_env"],
            training_dataset=f"s3://{bucket}/{PREFIX}/train/aircraft_inspection_train.parquet",
            validation_dataset=f"s3://{bucket}/{PREFIX}/validation/aircraft_inspection_val.parquet",
            mlflow_app_arn=cfg["mlflow_app_arn"],
            s3_output_path=f"s3://{bucket}/output/",
            role=cfg["role"],
            accept_eula=True,
        )
        hp = trainer.hyperparameters
        for key, value in HYPERPARAMETERS.items():
            setattr(hp, key, value)
        job = trainer.train(wait=False)
        print(f"Submitted {job.job_name}")

    job.wait(poll=60, timeout=args.timeout)
    job.refresh()
    print(f"Final status: {job.job_status}")
    print(f"Output model package: {job.output_model_package_arn}")
    return 0 if job.job_status in ("Completed", "Succeeded") else 1


if __name__ == "__main__":
    sys.exit(main())
