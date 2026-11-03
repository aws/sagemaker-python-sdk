# sagemaker.train.training_session

Client-driven interactive training sessions on Amazon SageMaker. A
`ServiceClient` creates a session and hands back a `TrainingClient` that exposes
low-level training primitives (`forward_backward`, `optim_step`, `forward`,
`save_state`, `load_state`, `save_weights_for_sampler`) you orchestrate in a
custom training loop, with automatic operation pipelining via sequence numbers.

The client surface:

| Client            | Role                                                |
|-------------------|-----------------------------------------------------|
| `ServiceClient`   | Entry point / factory                               |
| `TrainingClient`  | Drives the training loop                            |
| `SamplingClient`  | Sample from a saved checkpoint                      |

## Requirements

- **Python >= 3.10**
- **AWS credentials** configured (e.g., via `aws configure` or an IAM role)

## Installation

```bash
pip install sagemaker-train
```

What a training session needs is a botocore whose service description carries
the training-session operations, and that is checked directly: a botocore
without them makes creating a client raise, naming what is missing.

The built-in tokenizer (`get_tokenizer()` and string prompts/batches) needs
`transformers`, which ships as an optional extra rather than a core dependency.
Install it only if you pass text:

```bash
pip install sagemaker-train[tokenizer]
```

Callers who tokenize client-side can pass token IDs directly and skip the extra.

## Example Notebook

A runnable, end-to-end walkthrough (create, train, save/load state, sample,
delete) is in
[`v3-examples/training-examples/training_session_example.ipynb`](../../../../../v3-examples/training-examples/training_session_example.ipynb).

## Quick Start

`role_arn` is the only required argument. Everything else has a default, listed
under [ServiceClient](#serviceclient).

```python
from sagemaker.train.training_session import (
    AdamParams,
    Datum,
    ModelInput,
    ServiceClient,
    TensorData,
    TensorDtype,
)

svc = ServiceClient(
    region="us-west-2",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
)
print(svc.list_base_models())   # base models you can fine-tune in this region
tc = svc.create_lora_training_client(
    base_model="<id from list_base_models()>",   # resolved to a Hub ARN in your region
)

for tokens in your_token_id_batches:
    # Every datum carries the loss inputs its loss function needs, one value per
    # token; see Loss function inputs below.
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

    # Each call returns an APIFuture right away, so optim_step is queued behind
    # the forward_backward without waiting for it to land. Resolve both once
    # they are submitted.
    fwd_bwd_op = tc.forward_backward(batch, loss_fn="cross_entropy")
    optim_op = tc.optim_step(AdamParams(learning_rate=1e-4))

    print(fwd_bwd_op.result().metrics, optim_op.result().metrics)

print(tc.save_state().result().model_package_arn)
```

See [Loss function inputs](#loss-function-inputs) for which keys each loss
function needs and how to build them for reinforcement learning.

Set the account-level plumbing (model package group, S3 output, session name
prefix, and the optional `kms_key_arn` / `vpc_config`) once on the
`ServiceClient` and every client it creates inherits it. Each is also accepted
per session on `create_lora_training_client` when one run needs something
different.

```python
svc = ServiceClient(
    region="us-west-2",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    model_package_group_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-models",
    s3_output_path="s3://my-bucket/output/",
    session_name_prefix="my-experiment",   # sessions get named "my-experiment-<8 hex>"
)
```

## Prerequisites

### 1. Create a Model Package Group

```bash
aws sagemaker create-model-package-group \
    --model-package-group-name "<MODEL_PACKAGE_GROUP_NAME>" \
    --region us-west-2
```

### 2. Create an S3 bucket for outputs

```bash
aws s3 mb s3://<BUCKET_NAME> --region us-west-2
```

### 3. Create an execution role

SageMaker assumes this role to write checkpoints to your bucket and register them
in your model package group. You pass its ARN to `ServiceClient`.

Attach the `AmazonSageMakerTrainingSessionFullAccess` managed policy:

```bash
aws iam attach-role-policy \
    --role-name "<EXECUTION_ROLE_NAME>" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerTrainingSessionFullAccess
```

Trust policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": { "Service": "training-session.sagemaker.amazonaws.com" },
            "Action": ["sts:AssumeRole", "sts:TagSession"]
        }
    ]
}
```

### 4. Grant your own identity permission to call the SDK

Attach the `AmazonSageMakerTrainingSessionAccess` managed policy to whichever
identity runs your training script. It covers the session and operation calls the
SDK makes, and the `iam:PassRole` that SageMaker checks when you create a
session with the role from step 3.

```bash
aws iam attach-role-policy \
    --role-name "<YOUR_ROLE_NAME>" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerTrainingSessionAccess
```

Its S3 permissions cover buckets whose names begin with `sagemaker-`, so name the
bucket from step 2 accordingly if you call
`delete_model_package(delete_artifacts=True)`, which removes a checkpoint's
objects from your bucket.

## API Reference

### ServiceClient

The entry point. Holds the control-plane and runtime clients.

`role_arn` is required: it is the execution role the session assumes. Session
defaults set here (`model_package_group_arn`, `s3_output_path`,
`session_name_prefix`, `kms_key_arn`, `vpc_config`) apply to every client this
`ServiceClient` creates, and each can be overridden per session on
`create_lora_training_client`.

| argument | required | default when omitted |
|---|---|---|
| `role_arn` | yes | n/a |
| `region` | no | `AWS_REGION`, then the standard boto3 chain; an error if neither names one |
| `model_package_group_arn` | no | the account's `sagemaker-default` group, created on first use |
| `s3_output_path` | no | `s3://{SageMaker default bucket}/{session name}/output` |
| `session_name_prefix` | no | `sagemaker-training-session` |
| `kms_key_arn` | no | Amazon S3 managed encryption |
| `vpc_config` | no | checkpoint transfers do not traverse your VPC |

```python
svc = ServiceClient(
    region="us-west-2",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    model_package_group_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-models",
    s3_output_path="s3://my-bucket/output/",
    session_name_prefix="my-experiment",
)
```

#### create_lora_training_client

Create a `TrainingSession` resource and return a `TrainingClient`. Blocks until
the session reaches `Running`.

`base_model` is the only required argument and takes either a model id from
`list_base_models()` or a full Hub model ARN. An id resolves to the latest Hub
ARN in your region; pass the ARN to pin a specific model version.

```python
tc = svc.create_lora_training_client(
    base_model="<id from list_base_models()>",
)
```

| argument | required | default when omitted |
|---|---|---|
| `base_model` | yes | n/a |
| `training_session_name` | no | `{session_name_prefix}-{8 hex}` |
| `rank` | no | `32` |
| `alpha` | no | `64` |
| `model_package_group_arn`, `s3_output_path`, `kms_key_arn`, `vpc_config` | no | the `ServiceClient`'s |

```python
tc = svc.create_lora_training_client(
    base_model="<id from list_base_models()>",
    training_session_name="my-session",           # optional; names it yourself instead
    rank=8,
    alpha=16,
    s3_output_path="s3://another-bucket/run-7/",  # optional; overrides the ServiceClient's
)
```

The `*_async` twin is a coroutine, for creating the session from async code:

```python
tc = await svc.create_lora_training_client_async(
    base_model="<id from list_base_models()>", rank=8, alpha=16
)
```

#### Encrypting outputs with your own KMS key

Pass `kms_key_arn` to encrypt the session's outputs (saved weights and
checkpoints) in Amazon S3 with your key instead of Amazon S3 managed
encryption. Set it on `ServiceClient` as a default for every session, or on
`create_lora_training_client` per session. The execution role needs
`kms:DescribeKey`, `kms:GenerateDataKey`, and `kms:Decrypt` on the key.

```python
svc = ServiceClient(
    region="us-west-2",
    role_arn="...",
    kms_key_arn="arn:aws:kms:us-west-2:123456789012:key/1234abcd-...",
)
```

#### Routing checkpoint transfers through your VPC

Pass `vpc_config` (a `VpcConfig` or a dict with `subnets` and
`security_group_ids`) to have checkpoint transfers between the service and your
account go through your VPC, so that traffic reaches your S3 bucket over your
own network path rather than the public internet. It applies to that transfer
only; the training and sampling operations themselves are unaffected.

Make sure the subnets can reach Amazon S3 (for example via an S3 gateway
endpoint on the subnets' route tables) and the security groups allow outbound
HTTPS, or checkpoint saves fail. The execution role needs the standard Amazon
EC2 networking permissions (`ec2:DescribeVpcs`, `ec2:DescribeSubnets`,
`ec2:DescribeSecurityGroups`, `ec2:DescribeDhcpOptions`, and the
`ec2:*NetworkInterface*` actions).

```python
from sagemaker.train.training_session import VpcConfig

svc = ServiceClient(
    region="us-west-2",
    role_arn="...",
    vpc_config=VpcConfig(
        subnets=["subnet-0123456789abcdef0", "subnet-0fedcba9876543210"],
        security_group_ids=["sg-0123456789abcdef0"],
    ),
)
```

These settings also apply when `create_sampling_client` creates a training
session on your behalf.

#### create_training_client_from_state / create_training_client_from_state_with_optimizer

Resume from a `save_state` checkpoint in a **new** session: the one-call form of
`create_lora_training_client(...)` followed by `load_state(...)`. As with the
`load_state` pair, the method name decides what comes back from the checkpoint:

| factory | restores | optimizer |
|---|---|---|
| `create_training_client_from_state(...)` | weights only | resets |
| `create_training_client_from_state_with_optimizer(...)` | weights + optimizer | continues where it left off |

Both read the same checkpoint. Any other keyword argument is forwarded to
`create_lora_training_client`, so `rank`, `s3_output_path`, `vpc_config` and the
rest work here too. Omit `base_model` and it is read from the checkpoint's model
package metadata, pinned to the version the weights were trained on.

Both return only once the restore has landed, so the client you get back is ready
to train. If the restore fails, its session is stopped and the error propagates.

```python
from sagemaker.train.training_session import ModelPackageKind

ckpt = svc.get_latest_model_package(
    ModelPackageKind.CHECKPOINT,
    model_package_group_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-models",
)

# Exact resume, optimizer momentum included
tc = svc.create_training_client_from_state_with_optimizer(ckpt.model_package_arn)

# Weights only, e.g. to start a new schedule from trained weights
tc = svc.create_training_client_from_state(ckpt.model_package_arn, rank=8)
```

Both have `*_async` coroutine twins with the same contract, awaitable without
holding up the event loop:

```python
tc = await svc.create_training_client_from_state_with_optimizer_async(
    ckpt.model_package_arn
)
```

#### list_base_models

Query the SageMaker Public Hub and return the base models available for
fine-tuning in the client's region (models are region-specific). Each is named by
its own model id, spelled as its provider publishes it wherever the Hub carries
one, so an id normally reads the same here as it does anywhere else the model is
named. Pass one as `base_model=`.

```python
svc.list_base_models()
```

The set changes as models are added, so call it rather than hard-coding an id.

#### create_sampling_client

Get a `SamplingClient` without running a training loop, which is mainly
evaluation work: scoring a saved checkpoint after training finishes, comparing
two checkpoints or a checkpoint against the base model, or reranking with
`compute_logprobs`. The SDK creates a training session on your behalf to serve
the requests. Specify exactly one of `base_model` (sample the base model) or
`model_package_arn` (sample saved weights; the base model is resolved from the
package metadata).

Because the SDK created that session for this client, **the client owns it**:
use it as a context manager, or call `sc.stop()`, so the session is released at
a point you choose.

```python
with svc.create_sampling_client(model_package_arn=ckpt_arn) as sc:
    out = sc.sample([1, 2, 3]).result()
# session stopped here

with svc.create_sampling_client(base_model="<id from list_base_models()>") as base_sc:
    ...
```

If you neither scope nor stop it, the SDK stops the session for you once the
last client referencing it becomes unreachable. See
[Session lifecycle](#session-lifecycle).

> **Note:** To sample from the client that is actively training, use
> `TrainingClient.create_sampling_client(...)` /
> `TrainingClient.save_weights_and_get_sampling_client(...)` instead. Those
> share the `TrainingClient`'s session, so they do **not** own it: leaving a
> `with` block on one of those clients drains its queued submits but leaves the
> training session running.

#### stop_session

Stop a running session, transitioning it toward `Stopped`.

```python
svc.stop_session("my-session")
```

#### delete_session

Delete a session. The service only deletes sessions in a terminal state
(`Stopped`, `Failed`, or `DeletionFailed`) and removal is asynchronous (the
session flips to `Deleting` and a background worker finishes the cleanup). Pass
`stop_if_active=True` to stop a still-running session and wait for a terminal
state before deleting.

```python
svc.delete_session("my-session")                       # session already terminal
svc.delete_session("my-session", stop_if_active=True)  # stop, wait, then delete
```

#### delete_model_package

Delete a saved checkpoint or set of weights. The model package carries the only
reference to the S3 location its weights were written to, so by default this
deletes the package and leaves the objects, which keep costing storage with
nothing pointing at them. Keep the returned `s3_uri` if you plan to clean up
later.

```python
result = svc.delete_model_package(ckpt.model_package_arn)
print("weights still at", result.s3_uri)
```

Pass `delete_artifacts=True` to delete the objects too. They go first and the
package second, so a failure part-way leaves the package still pointing at
whatever survived and the call can be retried. That case raises
`ArtifactDeletionError` with the surviving keys on `.failed_keys`.

```python
result = svc.delete_model_package(ckpt.model_package_arn, delete_artifacts=True)
print("removed", result.objects_deleted, "objects")
```

> **Note:** `create_sampling_client(model_package_arn=...)` and
> `load_state(model_package_arn=...)` read a package's artifacts, so once they are
> deleted do not rely on this package for either.

Deleting artifacts signs with your own credentials, not the session's `role_arn`,
so you need `s3:ListBucket` on the bucket and `s3:DeleteObject` on the objects
under it. Where SageMaker stores a package's weights on your behalf rather than
in your own bucket there is nothing for you to delete: the call reports them
retained, and deleting the package releases them.

The call is idempotent. A package that is already gone comes back with
`model_package_deleted=False` rather than raising. To sweep a group:

```python
for mp in svc.list_model_packages(max_results=None):
    svc.delete_model_package(mp.model_package_arn, delete_artifacts=True)
```

Deleting every version leaves the (empty) model package group in place; removing
the group is a separate SageMaker call.

#### list_sessions

List training sessions, auto-paginating. Optionally filter by name substring or
status, and cap the total with `max_results`.

```python
svc.list_sessions()                              # all sessions
svc.list_sessions(status_equals="Running")       # only running
svc.list_sessions(name_contains="exp", max_results=10)
```

Each entry is a `TrainingSessionSummary` with `training_session_arn`, `status`,
`creation_time` and `last_modified_time`, plus a `training_session_name` read off
the ARN. `status` is a `SessionStatus`, so it compares against the enum rather
than a raw string. `stop_session` and `delete_session` accept either a name or an
ARN, so either attribute can be passed straight through.

Sweeping up every session still running, e.g. after an interrupted experiment:

```python
from sagemaker.train.training_session import SessionStatus

for summary in svc.list_sessions(status_equals=SessionStatus.RUNNING):
    print("stopping", summary.training_session_name, summary.creation_time)
    svc.stop_session(summary.training_session_arn)
```

Pass `name_contains=` to narrow the sweep to one experiment's sessions.

### TrainingClient

Drives the training loop over a session.

Every operation has two forms, and both return an `APIFuture`:

| form | returns | resolve it with |
|---|---|---|
| `tc.forward_backward(...)` | `APIFuture` | `.result()`, or `await` the future |
| `await tc.forward_backward_async(...)` | `APIFuture` | `await` the future |

The base method queues its submit on a background thread and hands back the
future immediately; a submit error surfaces when the future resolves, not at the
call site. The `*_async` twin is a coroutine, so the enqueue yields to the event
loop instead of running on the calling thread, and resolves to the same future.

Training operations carry sequence numbers and must stay ordered, so the enqueue
and the result are separate steps: the future is your handle on a submit that
already holds its place in the sequence.

```python
result = tc.forward_backward(batch).result()        # sync code
result = await tc.forward_backward(batch)           # async code, base method

future = await tc.forward_backward_async(batch)     # async code, coroutine twin
result = await future
```

The examples below mostly use the base form.

#### Bounding how long you wait

`result()` waits indefinitely by default, because a training operation that
succeeds should not fail just because it ran past an arbitrary clock. Pass
`timeout` (seconds) to bound a specific wait. When set, it bounds the *whole*
wait, including the submit round trip and every chunk of a batch large enough to
be split across several operations, so `result(timeout=60)` is a ~60-second
call, not 60 seconds per stage. (It bounds when a new wait may *start*, so a
status check already in flight can overrun it by that request's latency.)

`await future` takes no argument, so it uses the default; `result_async` is the
await form that accepts one.

```python
result = tc.forward_backward(batch).result(timeout=300)

future = tc.forward_backward(batch)
result = await future                                  # waits indefinitely
result = await future.result_async(timeout=300)        # bounded
```

A `TimeoutError` means the wait ran out, not that the operation was cancelled.
It keeps running server-side, and resolving the same future again picks it back
up.

#### Loss function inputs

Every datum needs loss inputs, and which keys are required depends on the loss:

| `loss_fn` | Required keys |
|---|---|
| `cross_entropy` | `target_tokens`, `weights` |
| `ppo` / `importance_sampling` / `cispo` | `target_tokens`, `weights`, `logprobs`, `advantages` |

Each is **full-sequence**: one value per token of the datum, prompt included.
`weights` is what selects the tokens the loss applies to: `0.0` over the prompt,
`1.0` over the response. Pass them as `TensorData` (which also accepts a torch
tensor or numpy array) so they carry a dtype.

For reinforcement learning, build them per datum: `advantages` differ from one
datum to the next, so each `Datum` carries its own. This is the form a training
loop usually yields.

```python
from sagemaker.train.training_session import Datum, ModelInput, TensorData

def make_datum(tokens, prompt_len, logprobs, advantages):
    """One datum whose loss inputs are full-sequence."""
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(data=tokens[1:] + [0], dtype="int64"),
            "weights": TensorData(data=[0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)),
            "logprobs": TensorData(data=logprobs),      # from sampling, one per token
            "advantages": TensorData(data=advantages),  # one per token
        },
    )

result = tc.forward_backward(batch=[make_datum(...)], loss_fn="ppo").result()
```

The `loss_fn_inputs` argument on `forward_backward` / `forward` is the shortcut
for the other case: one set of values shared by the whole batch. It is filled in
only for datums that do not already carry their own, so the two forms mix freely.

```python
tokens, prompt_len = [1, 2, 3, 4, 5, 6], 3

result = tc.forward_backward(
    batch=[tokens],
    loss_fn="cross_entropy",
    loss_fn_inputs={
        "target_tokens": TensorData(data=tokens[1:] + [0], dtype="int64"),
        "weights": TensorData(data=[0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)),
    },
).result()
```

Every datum must end up with the inputs its loss function requires, and two
mistakes are caught before the request leaves. A datum missing any of them
raises, naming the datum, the loss function and the keys it needs. Giving
response-length values instead of full-sequence ones raises too, naming the
datum and the key.

#### forward_backward

Run a forward-backward pass. Accepts token ID lists, strings (with a tokenizer),
or raw datum dicts.

```python
from sagemaker.train.training_session import TensorData, TensorDtype

tokens, prompt_len = [1, 2, 3, 4, 5, 6], 3
response_len = len(tokens) - prompt_len

result = tc.forward_backward(
    batch=[tokens],
    loss_fn="importance_sampling",   # or "ppo", "cispo", "cross_entropy"
    loss_fn_inputs={
        "target_tokens": TensorData(data=tokens[1:] + [0], dtype=TensorDtype.INT64),
        "weights": TensorData(data=[0.0] * prompt_len + [1.0] * response_len),
        "logprobs": TensorData(data=sampled_logprobs),    # from the sampling step
        "advantages": TensorData(data=[reward] * len(tokens)),
    },
).result()
print(result.metrics)          # {"actor/pg_loss": ..., "global_token_num": ...}

# Loss-function tensor outputs, one entry per datum:
for datum_outputs in result.loss_fn_outputs:
    print(datum_outputs)       # {"actor/pg_loss": ..., "actor/grad_norm": ...}
```

#### forward

Run a forward-only pass (no gradient accumulation), useful for computing
loss-function outputs or logprobs without updating the model. Takes the same
arguments as `forward_backward`.

```python
result = tc.forward(
    batch=[tokens],
    loss_fn="cross_entropy",
    loss_fn_inputs={
        "target_tokens": TensorData(data=tokens[1:] + [0], dtype="int64"),
        "weights": TensorData(data=[0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)),
    },
).result()
for datum_outputs in result.loss_fn_outputs:
    print(datum_outputs)
```

#### optim_step

Run an optimizer step.

```python
result = tc.optim_step(AdamParams(learning_rate=1e-4)).result()
```

#### save_weights_for_sampler

Save model weights for inference, registering a **model** package. Pass
`merge_weights=True` to merge the adapter into the base weights, so the result
can be deployed without the adapter.

```python
print(tc.save_weights_for_sampler().result().model_package_arn)

# or, in async code
future = await tc.save_weights_for_sampler_async(merge_weights=True)
print((await future).model_package_arn)
```

To save and immediately get a client bound to the result, use
`save_weights_and_get_sampling_client()` instead. It returns a `SamplingClient`,
not a future.

#### save_state

Save a resumable checkpoint (weights + optimizer state). Always creates a model
package that can be passed to `load_state` to resume training.

```python
model_package_arn = tc.save_state().result().model_package_arn
print(model_package_arn)
```

#### load_state / load_state_with_optimizer

Load a previously saved `save_state` checkpoint to resume training. Two peer
methods, no flag: the method name decides what is restored from the checkpoint.

| method | restores | optimizer |
|---|---|---|
| `load_state(arn)` | weights only | resets |
| `load_state_with_optimizer(arn)` | weights + optimizer | continues where it left off |

Both read the same checkpoint (which always contains both).

```python
# Weights only, optimizer resets
tc.load_state(model_package_arn=model_package_arn).result()

# Weights + optimizer, exact resume
tc.load_state_with_optimizer(model_package_arn=model_package_arn).result()
```

#### Model package discovery

Every `save_state` / `save_weights_for_sampler` registers a **model package**
version in the session's group. A `save_state` one is a **checkpoint** (weights +
optimizer state, for `load_state`); a `save_weights_for_sampler` one is a
deployable **model** (weights only, for inference or a new run).
`ModelPackageKind` names that split, and one accessor takes it as an argument.

`tc.get_latest_model_package(kind)` answers from an in-memory cache of what
**this client** saved in this process, so there is no API call and no exposure to
the brief lag before a freshly-saved package is describable. It returns `None`
when this client hasn't saved that kind (e.g. a fresh or resumed session): a
client deliberately does not imply a package from a session it did not drive. To
discover packages a *different* client saved, use the `ServiceClient` twin, which
queries the group: `svc.get_latest_model_package(kind, model_package_group_arn=...)`.
`tc.list_model_packages(...)` forwards to the `ServiceClient` with this session's
group and returns full metadata.

```python
from sagemaker.train.training_session import ModelPackageKind

tc.save_state().result()
ckpt = tc.get_latest_model_package(ModelPackageKind.CHECKPOINT)   # cache; resume with load_state*
tc.load_state_with_optimizer(model_package_arn=ckpt.model_package_arn).result()

tc.save_weights_for_sampler().result()
model = tc.get_latest_model_package(ModelPackageKind.MODEL)       # cache; for inference / sampling

tc.list_model_packages(max_results=20)                           # full metadata, newest first

# Across processes, query the group instead of a client's cache
ckpt = svc.get_latest_model_package(
    ModelPackageKind.CHECKPOINT,
    model_package_group_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-models",
)
```

To remove one, see
[`ServiceClient.delete_model_package`](#delete_model_package). A model package
outlives the session that saved it, so nothing cleans it up when the session
goes.

#### stop / delete

Stop or delete this client's session without repeating its name.

```python
tc.stop()                      # stop this session
tc.delete(stop_if_active=True) # stop, wait for terminal, then delete
```

### Session lifecycle

A training session is live compute, so the SDK does not leave one `Running`
just because you forgot to stop it. Once the last client referencing a session
becomes unreachable, the SDK stops that session and emits a `ResourceWarning`:

```python
tc = svc.create_lora_training_client(base_model="<id from list_base_models()>")
...
tc = None          # or the script ends, or the notebook cell is re-run
                   # -> session stopped
```

The automatic teardown **stops** the session. A stopped session is still
describable, so its history stays available to audit and a run that went wrong
can still be inspected; deleting one is an explicit `tc.delete(...)` /
`svc.delete_session(...)` call.

A session can be shared by more than one client, a `TrainingClient` and any
`SamplingClient` derived from it, and it is the **last** of them to go away that
triggers the stop, not any particular one. So dropping one client while another
is still in use leaves the session running:

```python
tc = svc.create_lora_training_client(base_model="<id from list_base_models()>")
sp = tc.create_sampling_client()   # shares tc's session; 2 clients now hold it

del tc                             # 1 left (sp) -> session stays Running
sp.sample([1, 2, 3]).result()      # still works

del sp                             # 0 left -> session stops here
```

Call `stop()` to release the session at a moment you choose. It flushes queued
operations, then stops without waiting for them to finish, so call it once you
have the results you need.

Save a checkpoint for anything you want to keep: model packages are durable on
their own, so trained weights outlive the process and a fresh session can be
created from one whenever you need it.

```python
tc = svc.create_lora_training_client(base_model="<id from list_base_models()>")
...  # train
mp = tc.save_weights_for_sampler().result().model_package_arn
# later, in any other process:
sp = svc.create_sampling_client(model_package_arn=mp)
```

### SamplingClient

Sample from a saved checkpoint (or the base model). Get one from a
`TrainingClient` via `save_weights_and_get_sampling_client(...)` (saves weights,
then binds to the resulting checkpoint) or `create_sampling_client(...)`, or
without a `TrainingClient` from `ServiceClient.create_sampling_client(...)`.

Only the `ServiceClient` factory creates the backing session, so only its client
stops that session on `with`-block exit or `stop()`. Clients handed out by a
`TrainingClient` share the training session and leave it running, though they do
keep it alive; see [Session lifecycle](#session-lifecycle).

Like every method on the training client, `sample` and `compute_logprobs` queue
their submit on a background thread and return an `APIFuture` immediately. Call
`.result()` on it, or `await` it, to get the parsed response. Sample submits are
unordered, so many requests can be queued concurrently and resolved in any
order.

Because there is no sequence to hold a place in, the sampling coroutines return
the response itself rather than a future:

| form | returns |
|---|---|
| `sc.sample(...)` | `APIFuture` resolving to `SampleResponse` |
| `await sc.sample_async(...)` | `SampleResponse` |
| `sc.compute_logprobs(...)` | `APIFuture` resolving to `List[Optional[float]]` |
| `await sc.compute_logprobs_async(...)` | `List[Optional[float]]` |

```python
from sagemaker.train.training_session import SamplingParams

sc = tc.save_weights_and_get_sampling_client()
future = sc.sample(
    [1, 2, 3],                       # token IDs, a string (with a tokenizer), or a ModelInput
    num_samples=4,
    sampling_params=SamplingParams(max_tokens=1024),
)
result = future.result()
for seq in result.sequences:
    print(seq.tokens, seq.stop_reason)
```

Queue a batch of prompts and gather them as they land; hundreds can be in flight
at once:

```python
futures = [sc.sample(p, num_samples=4) for p in prompts]   # all queued up front
for f in futures:
    print(f.result().sequences[0].tokens)

# async: await the futures, or call the coroutine twin
results = await asyncio.gather(*(sc.sample(p, num_samples=4) for p in prompts))
one = await sc.sample_async(prompts[0], num_samples=4)     # -> SampleResponse
```

#### compute_logprobs

Score a prompt without generating: returns one log probability per prompt token
(useful for evaluation, reranking, or importance weighting).

```python
logprobs = sc.compute_logprobs([1, 2, 3]).result()   # or a string, with a tokenizer
print(logprobs)   # one logprob per prompt token (first entry is None)

logprobs = await sc.compute_logprobs_async([1, 2, 3])   # -> List[Optional[float]]
```

#### get_tokenizer

The runtime accepts only token IDs. To pass text prompts (or string training
batches), call `get_tokenizer()` to load the base model's HuggingFace tokenizer;
the SDK then tokenizes strings for you. It is loaded once and cached. Available
on both `TrainingClient` and `SamplingClient`.

```python
tokenizer = sc.get_tokenizer()   # raises if the model has no published tokenizer
result = sc.sample("Hello, world", num_samples=1).result()
```

Sampled sequences come back as token IDs, so use the same tokenizer to read them.
It is an ordinary HuggingFace tokenizer, so `encode` and `decode` behave as they
do anywhere else:

```python
tokenizer = sc.get_tokenizer()

prompt_ids = tokenizer.encode("What is 12 x 7?")
result = sc.sample(prompt_ids, num_samples=1).result()

print(tokenizer.decode(result.sequences[0].tokens))
```

Passing the string directly, as above, tokenizes it the same way. Encode yourself
when you need the IDs for something else too, such as measuring prompt length or
building a `ModelInput` with several chunks.

## Errors

Service errors reach you as raised by botocore, so `botocore.exceptions.ClientError`
and its subclasses carry the error code, message, and response metadata you would
see from any other AWS client. Catch them the way you already do, and read
`error.response["Error"]["Code"]` to tell them apart:

| error code | meaning |
|---|---|
| `ValidationException` | the request was rejected; resending it gets the same answer |
| `ResourceNotFoundException` | the session, operation, or checkpoint does not exist |
| `ConflictException` | the session is not in a state that accepts the call |
| `ThrottlingException` | too many requests; retryable |
| `InternalServerException` | the service could not serve the request; retryable |

Session lifecycle calls go to the SageMaker control plane, which reports
`ResourceNotFound`, `ResourceInUse`, and `ResourceLimitExceeded` under those
shorter names rather than the `*Exception` spellings above.

On top of those, three errors describe conditions no service error covers:

| error | raised when |
|---|---|
| `OperationError` | the service reports a submitted operation as failed; carries `operation_id` |
| `ResultMismatchError` | a result does not correspond to the request it was awaited for |
| `ArtifactDeletionError` | `delete_model_package(delete_artifacts=True)` partially completed; carries `failed_keys` |

All three subclass `TrainingSessionError`, so a single
`except TrainingSessionError` catches every error this package defines. A wait
that runs out of time is the one exception: `result(timeout=...)` raises the
built-in `TimeoutError`, the same as any other future, so catch that alongside
`TrainingSessionError` to cover both.

`is_retryable(error)` classifies these alongside botocore's own errors, so a
retry policy does not have to match on error types:

```python
from sagemaker.train.training_session import is_retryable

try:
    result = tc.forward_backward(batch, loss_fn="ppo").result()
except Exception as error:
    if not is_retryable(error):
        raise
    result = tc.forward_backward(batch, loss_fn="ppo").result()
```

## Pipelining

Every base method already pipelines: it returns an `APIFuture` immediately
and the SDK stamps sequence numbers at call time, so you can submit
`optim_step` before the preceding `forward_backward` resolves and both execute
in the same clock cycle. Submits ride a background sender, so queueing a
large batch of operations takes seconds, not one wire round trip per op:

```python
for batch in data_loader:
    fwd_bwd_op = tc.forward_backward(batch, loss_fn="ppo")
    optim_op = tc.optim_step()

    fwd_bwd_result = fwd_bwd_op.result()
    optim_result = optim_op.result()
```

The same loop in async code: `await` the futures instead of calling `.result()`,
so the event loop stays free while the ops run.

```python
for batch in data_loader:
    fwd_bwd_op = tc.forward_backward(batch, loss_fn="ppo")
    optim_op = tc.optim_step()

    fwd_bwd_result = await fwd_bwd_op
    optim_result = await optim_op
```

Use the `*_async` coroutine twins when the enqueue should yield to the event loop
as well; each resolves to the same `APIFuture`:

```python
fwd_bwd_op = await tc.forward_backward_async(batch, loss_fn="ppo")
optim_op = await tc.optim_step_async()
fwd_bwd_result, optim_result = await asyncio.gather(fwd_bwd_op, optim_op)
```

## Concurrent Access

A training session must only be accessed by one `TrainingClient` at a time. If
two clients reference the same session simultaneously, they issue conflicting
sequence numbers and the service rejects the later one with a
`ConflictException`.

## Running the Example

See
[`v3-examples/training-examples/training_session_example.ipynb`](../../../../../v3-examples/training-examples/training_session_example.ipynb)
for a runnable end-to-end walkthrough. Update the configuration cell with your
own model package group, execution role, and S3 output path before running; the
final cell deletes the session.
