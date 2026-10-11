#!/usr/bin/env bash
# Deploy the SOP-Bench rollout agent to Amazon Bedrock AgentCore.
#
# Usage (from sopbench_agent/, after `agentcore configure`):
#   ./scripts/deploy.sh
#   DEBUG=1 ./scripts/deploy.sh        # verbose per-turn logs (debugging only: large log volume)
#   AGENT_NAME=my_agent ./scripts/deploy.sh
#
# `agentcore deploy` builds the arm64 image from the Dockerfile on AWS CodeBuild (no local Docker
# needed), pushes it to ECR, and creates or updates the runtime. It uses the execution role you
# passed to `agentcore configure --execution-role`.
set -euo pipefail

AGENT_NAME="${AGENT_NAME:-sopbench_agentcore}"   # must match `agentcore configure --name`
DEBUG="${DEBUG:-0}"

cd "$(dirname "$0")/.."

echo "Deploying agent '$AGENT_NAME' (DEBUG=$DEBUG) ..."
# --auto-update-on-conflict updates an existing runtime of the same name instead of failing.
AGENTCORE_SUPPRESS_RECOMMENDATION=1 agentcore deploy \
  -a "$AGENT_NAME" \
  --auto-update-on-conflict \
  --env "DEBUG=$DEBUG"

echo
echo "Next: 'agentcore status' to confirm the runtime is READY."
echo "Tear down: 'agentcore destroy -a $AGENT_NAME --force'."
