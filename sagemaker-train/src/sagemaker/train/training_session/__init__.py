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
"""SageMaker training session SDK: client surface for interactive training.

Create a ``ServiceClient``, derive a ``TrainingClient``, run the training loop,
then derive a ``SamplingClient`` for inference::

    from sagemaker.train.training_session import (
        AdamParams,
        Datum,
        ModelInput,
        SamplingParams,
        ServiceClient,
        TensorData,
        TensorDtype,
    )

    svc = ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/SageMakerTrainingSessionsRole",
        model_package_group_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-group",
        s3_output_path="s3://my-bucket/output",
    )
    tc = svc.create_lora_training_client(base_model=svc.list_base_models()[0])
    for tokens in your_token_id_batches:
        # Every datum carries the loss inputs its loss function needs, one value
        # per token of the datum.
        batch = [
            Datum(
                model_input=ModelInput.from_ints(tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData(
                        data=tokens[1:] + [0], dtype=TensorDtype.INT64
                    ),
                    "weights": TensorData(data=[1.0] * len(tokens)),
                },
            )
        ]
        # Each call returns an APIFuture right away, so the optim_step is queued
        # behind the forward_backward without waiting for it to land.
        fwd_bwd = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim = tc.optim_step(AdamParams(learning_rate=1e-4))
        print(fwd_bwd.result().metrics, optim.result().metrics)
    sc = tc.save_weights_and_get_sampling_client()
    sc.sample(
        "What is the capital of France?",
        num_samples=8,
        sampling_params=SamplingParams(max_tokens=512),
    ).result()

From async code, ``await`` the same futures instead of calling ``.result()``::

    fwd_bwd = tc.forward_backward(batch, loss_fn="cross_entropy")
    optim = tc.optim_step(AdamParams(learning_rate=1e-4))
    fwd_bwd_result, optim_result = await asyncio.gather(fwd_bwd, optim)
"""

from __future__ import annotations

from sagemaker.train.training_session.futures import APIFuture, RequestStatus
from sagemaker.train.training_session.service_client import (
    DEFAULT_MODEL_PACKAGE_GROUP_NAME,
    DEFAULT_SESSION_NAME_PREFIX,
    ServiceClient,
)
from sagemaker.train.training_session.training_client import TrainingClient
from sagemaker.train.training_session.sampling_client import SamplingClient
from sagemaker.train.training_session.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LoadStateResponse,
    LossFunction,
    ModelInput,
    ModelInputChunk,
    ModelPackageDeletion,
    ModelPackageInfo,
    ModelPackageKind,
    OptimStepResponse,
    SampledSequence,
    SampleResponse,
    SamplingParams,
    SaveStateResponse,
    SaveWeightsResponse,
    SessionStatus,
    TrainingSessionSummary,
    StopReason,
    TensorData,
    TensorDtype,
    TokenLogprob,
    VpcConfig,
)
from sagemaker.train.training_session.exceptions import (
    ArtifactDeletionError,
    OperationError,
    ResultMismatchError,
    TrainingSessionError,
    is_retryable,
)

__all__ = [
    "ServiceClient",
    "TrainingClient",
    "SamplingClient",
    "APIFuture",
    "RequestStatus",
    "DEFAULT_MODEL_PACKAGE_GROUP_NAME",
    "DEFAULT_SESSION_NAME_PREFIX",
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "LoadStateResponse",
    "LossFunction",
    "ModelInput",
    "ModelInputChunk",
    "ModelPackageDeletion",
    "ModelPackageInfo",
    "ModelPackageKind",
    "OptimStepResponse",
    "SampledSequence",
    "SampleResponse",
    "SamplingParams",
    "SaveStateResponse",
    "SaveWeightsResponse",
    "SessionStatus",
    "TrainingSessionSummary",
    "StopReason",
    "TensorData",
    "TensorDtype",
    "TokenLogprob",
    "VpcConfig",
    "ArtifactDeletionError",
    "OperationError",
    "ResultMismatchError",
    "TrainingSessionError",
    "is_retryable",
]
