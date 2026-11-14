# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""Fixtures for AI Registry integration tests."""

import json
import os
import tempfile
import uuid
import zipfile

import boto3
import pytest

from sagemaker.ai_registry.air_utils import _get_default_bucket
from sagemaker.train.defaults import TrainDefaults


@pytest.fixture
def unique_name():
    """Generate unique name for testing."""
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_bucket():
    """Get test S3 bucket name."""
    return _get_default_bucket()


@pytest.fixture
def test_role():
    """Get test IAM role ARN."""
    return TrainDefaults.get_role()


@pytest.fixture
def sample_jsonl_file():
    """Create sample JSONL dataset file."""
    content = """{"prompt": "What is AI?", "completion": "AI is artificial intelligence."}
{"prompt": "What is ML?", "completion": "ML is machine learning."}
{"prompt": "What is DL?", "completion": "DL is deep learning."}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(content)
        f.flush()  # Ensure content is written to disk
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_lambda_py_file():
    """Create a raw Python Lambda file with a non-default filename to test handler derivation."""
    code = """import json
def lambda_handler(event, context):
    return {"statusCode": 200, "body": json.dumps({"score": 0.9})}
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="my_custom_evaluator_", delete=False
    ) as f:
        f.write(code)
        f.flush()
        os.fsync(f.fileno())
        fname = f.name
    yield fname
    os.unlink(fname)


@pytest.fixture
def sample_lambda_code():
    """Create sample Lambda function code as zip."""
    code = """import json
def lambda_handler(event, context):
    return {"statusCode": 200, "body": json.dumps({"score": 0.8})}
"""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as zip_f:
        with zipfile.ZipFile(zip_f.name, "w") as zf:
            zf.writestr("lambda_function.py", code)
        yield zip_f.name
    os.unlink(zip_f.name)


@pytest.fixture
def sample_prompt_file():
    """Create sample prompt file."""
    content = "Evaluate the response: {response}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_hub_content_document():
    """Create sample hub content document."""
    from sagemaker.ai_registry.dataset_utils import DataSetHubContentDocument
    from sagemaker.ai_registry.air_constants import (
        DATASET_DEFAULT_TYPE,
        DATASET_DEFAULT_CONVERSATION_ID,
        DATASET_DEFAULT_CHECKPOINT_ID,
    )

    document = DataSetHubContentDocument(
        dataset_s3_bucket=_get_default_bucket(),
        dataset_s3_prefix="test",
        dataset_context_s3_uri='""',
        dataset_type=DATASET_DEFAULT_TYPE,
        dataset_role_arn=TrainDefaults.get_role(),
        conversation_id=DATASET_DEFAULT_CONVERSATION_ID,
        conversation_checkpoint_id=DATASET_DEFAULT_CHECKPOINT_ID,
        dependencies=[],
    )
    # Pinned to the version this fixture's callers declare to ImportHubContent.
    # to_json() defaults to the current schema, which would emit the v5 shape
    # under a 2.0.0 declaration and fail validation on five required legacy
    # fields. Moving the fixture to v5 is a larger change than it looks: v5's
    # DatasetType enum is ["SyntheticDataGeneration", "BYOD"], so
    # DATASET_DEFAULT_TYPE ("CUSTOMER_PROVIDED") is rejected there, and
    # DatasetFileFormat becomes required. These tests exercise hub operations,
    # not document schemas, so the legacy document is sufficient.
    return document.to_json(schema_version="2.0.0")


@pytest.fixture
def cleanup_list():
    """Track resources for cleanup."""
    resources = []
    yield resources
    for resource in resources:
        try:
            resource.delete()
        except Exception:
            pass


@pytest.fixture(scope="session", autouse=True)
def _stage_s3_test_fixtures():
    """Upload per-technique fixture JSONL files before S3-sourced dataset tests
    and delete them afterwards.

    The seven tests in test_dataset.py hard-code S3 keys under test_datasets/.
    That prefix is never persisted to the bucket, so the tests 404 unless it is
    provisioned at run time.  autouse=True with session scope means no test body
    changes are needed.

    Schema choices:
      oss_sft   -> hf_prompt_completion (prompt + completion; OSS SFT convention)
      oss_rlvr  -> verl               (prompt-only reward-training; VERL format)
      oss_dpo   -> hf_preference      (chosen + rejected; standard OSS DPO)
      nova_sft  -> converse           (Bedrock Converse is Nova's native format)
      nova_dpo  -> dpo                (converse-style with preference candidates)
      nova_rft  -> verl               (RFT = reinforcement fine-tuning = RLVR)
      nova_eval -> genqa              (query[+response]; purpose-built eval schema)
    """
    bucket = _get_default_bucket()
    s3 = boto3.client("s3")

    # OSS SFT: hf_prompt_completion requires ``prompt`` + ``completion``.
    _oss_sft = [
        {
            "prompt": "What is supervised learning?",
            "completion": "Supervised learning trains a model on labelled examples to predict outputs for new inputs.",
        },
        {
            "prompt": "What is a neural network?",
            "completion": "A neural network is a stack of weighted layers that learns data representations.",
        },
        {
            "prompt": "What is gradient descent?",
            "completion": "Gradient descent updates model weights iteratively to minimise a loss function.",
        },
    ]

    # OSS RLVR: verl requires only ``prompt``; reward_model is optional metadata.
    _oss_rlvr = [
        {
            "prompt": "Solve: 12 times 7.",
            "reward_model": {"style": "rule", "ground_truth": "84"},
            "data_source": "math",
        },
        {
            "prompt": "What is the square root of 144?",
            "reward_model": {"style": "rule", "ground_truth": "12"},
            "data_source": "math",
        },
        {
            "prompt": "Simplify 18 divided by 6.",
            "reward_model": {"style": "rule", "ground_truth": "3"},
            "data_source": "math",
        },
    ]

    # OSS DPO: hf_preference requires ``chosen`` + ``rejected``.
    _oss_dpo = [
        {
            "prompt": "How should I prepare for a coding interview?",
            "chosen": "Practice algorithms and system design, then review your solutions.",
            "rejected": "Just show up and answer whatever comes to mind.",
        },
        {
            "prompt": "What is the best way to learn a new programming language?",
            "chosen": "Build small projects while reading the official documentation.",
            "rejected": "Copy code from the internet without understanding it.",
        },
        {
            "prompt": "How do I improve model accuracy?",
            "chosen": "Tune hyperparameters, improve data quality, and apply regularisation.",
            "rejected": "Always add more data regardless of quality.",
        },
    ]

    # Nova SFT: converse — Bedrock Converse is Nova's native conversational format.
    # Requires messages array >= 2 items; role user|assistant; content array.
    _nova_sft = [
        {
            "messages": [
                {"role": "user", "content": [{"text": "What is Amazon Nova?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"text": "Amazon Nova is a family of frontier AI models built on AWS."}
                    ],
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": [{"text": "How do I fine-tune a foundation model?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Fine-tuning adapts a pre-trained model to a target task using labelled examples."
                        }
                    ],
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": [{"text": "What is transfer learning?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Transfer learning applies knowledge from one task to speed up learning on another."
                        }
                    ],
                },
            ]
        },
    ]

    # Nova DPO: dpo — native preference format for Nova fine-tuning.
    # The final turn is an assistant candidates message with preferenceLabel.
    _nova_dpo = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Explain reinforcement learning from human feedback."}],
                },
                {
                    "role": "assistant",
                    "candidates": [
                        {
                            "content": [
                                {
                                    "text": "RLHF aligns model outputs using ranked human preferences."
                                }
                            ],
                            "preferenceLabel": "preferred",
                        },
                        {
                            "content": [{"text": "RLHF is some ML technique."}],
                            "preferenceLabel": "non-preferred",
                        },
                    ],
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": [{"text": "What makes a good system prompt?"}]},
                {
                    "role": "assistant",
                    "candidates": [
                        {
                            "content": [
                                {
                                    "text": "A good system prompt is concise, unambiguous, and task-scoped."
                                }
                            ],
                            "preferenceLabel": "preferred",
                        },
                        {
                            "content": [{"text": "Any text works."}],
                            "preferenceLabel": "non-preferred",
                        },
                    ],
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": [{"text": "Describe a transformer model."}]},
                {
                    "role": "assistant",
                    "candidates": [
                        {
                            "content": [
                                {
                                    "text": "A transformer uses self-attention to process token sequences in parallel."
                                }
                            ],
                            "preferenceLabel": "preferred",
                        },
                        {
                            "content": [{"text": "Transformers transform data."}],
                            "preferenceLabel": "non-preferred",
                        },
                    ],
                },
            ]
        },
    ]

    # Nova RFT: verl — RFT (reinforcement fine-tuning) maps to CustomizationTechnique.RLVR.
    # Prompt uses the conversation-style array variant natural for Nova.
    _nova_rft = [
        {
            "prompt": [{"role": "user", "content": "Solve: 15 multiplied by 6."}],
            "reward_model": {"style": "rule", "ground_truth": "90"},
        },
        {
            "prompt": [{"role": "user", "content": "What is the derivative of x squared?"}],
            "reward_model": {"style": "rule", "ground_truth": "2x"},
        },
        {
            "prompt": [{"role": "user", "content": "Find the sum of integers from 1 to 10."}],
            "reward_model": {"style": "rule", "ground_truth": "55"},
        },
    ]

    # Nova eval: genqa — purpose-built evaluation schema; requires only ``query``.
    _nova_eval = [
        {
            "query": "What are the capabilities of Amazon Nova Pro?",
            "response": "Nova Pro supports multimodal understanding across text, images, and video.",
        },
        {
            "query": "How does chain-of-thought prompting improve model performance?",
            "response": "It guides the model to reason step by step, reducing errors on complex problems.",
        },
        {
            "query": "What is a foundation model?",
            "response": "A foundation model is pre-trained on broad data and adapted to many downstream tasks.",
        },
    ]

    fixtures = {
        "test_datasets/OSS/oss_sft_train.jsonl": _oss_sft,
        "test_datasets/OSS/oss_rlvr_train.jsonl": _oss_rlvr,
        "test_datasets/OSS/oss_dpo_train.jsonl": _oss_dpo,
        "test_datasets/Nova/nova_sft_train.jsonl": _nova_sft,
        "test_datasets/Nova/nova_dpo_train.jsonl": _nova_dpo,
        "test_datasets/Nova/nova_rft_train.jsonl": _nova_rft,
        "test_datasets/Nova/nova_eval.jsonl": _nova_eval,
    }

    for key, records in fixtures.items():
        body = "\n".join(json.dumps(r) for r in records) + "\n"
        s3.put_object(Bucket=bucket, Key=key, Body=body.encode())

    yield

    # Every xdist worker runs its own session, so all of them stage and would all
    # tear down these fixed keys — the first to finish deletes the objects the
    # others are still reading. Under xdist the staged fixtures are left in place:
    # the content is deterministic and the keys are hard-coded, so leaving them
    # costs nothing and keeps the suite runnable in parallel.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    for key in fixtures:
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass
