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
"""Shallow submission tests for ``ModelTrainer``.

Each test submits a real ``CreateTrainingJob``, asserts the service returned a
TrainingJobArn, then stops the job. A returned ARN proves the SDK-shaped payload
cleared every synchronous server-side gate (model validation, IAM authorization,
PassRole, the backend's request validators, S3/ECR resolution, routing) -- see
``harness`` for the full reasoning.

These tests assert acceptance, never training behaviour. Anything that requires
a job to actually run belongs in the deep suites.
"""

from __future__ import absolute_import

import os

import pytest
from sagemaker.core import shapes
from sagemaker.core.training.configs import Compute, InputData, Networking, SourceCode
from sagemaker.train.distributed import MPI, DistributedConfig, Torchrun
from sagemaker.train.model_trainer import ModelTrainer

from .harness import (
    CPU_IMAGE,
    DEFAULT_INSTANCE_COUNT,
    DEFAULT_INSTANCE_TYPE,
    MAX_RUNTIME_IN_SECONDS,
    assert_rejected,
    assert_submitted,
    stop_quietly,
    submitted,
    unique_name,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
PARAM_SCRIPT_SOURCE_DIR = os.path.join(DATA_DIR, "params_script")

# Mirrors the hyperparameter contract asserted by the existing deep suite, so a
# serialization regression is caught here (cheaply, on every PR) rather than only
# in the slow tests.
CONTRACT_HYPERPARAMETERS = {
    "integer": 1,
    "boolean": True,
    "float": 3.14,
    "string": "Hello World",
    "list": [1, 2, 3],
    "dict": {
        "string": "value",
        "integer": 3,
        "float": 3.14,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "boolean": True,
    },
}


def _source_code():
    """Source code bundle used by most tests here.

    A real local source_dir is used (rather than a stub) because the SDK tars and
    uploads it to S3 during submission, and the backend then validates that S3
    location. Skipping it would skip a real part of the path.
    """
    return SourceCode(
        source_dir=PARAM_SCRIPT_SOURCE_DIR,
        requirements="requirements.txt",
        entry_script="train.py",
    )


def _compute(instance_type=DEFAULT_INSTANCE_TYPE, instance_count=DEFAULT_INSTANCE_COUNT):
    """Small CPU compute config. Never sets keep_alive_period_in_seconds -- a warm
    pool would outlive the stop and keep instances provisioned."""
    return Compute(instance_type=instance_type, instance_count=instance_count)


def _stopping_condition():
    return shapes.StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS)


def _trainer(sagemaker_session, name, **overrides):
    """Build a ModelTrainer with the minimum viable accepted configuration.

    Centralised so that a change to what "minimally valid" means is a one-line
    edit rather than a sweep across every test.
    """
    kwargs = dict(
        sagemaker_session=sagemaker_session,
        training_image=CPU_IMAGE,
        source_code=_source_code(),
        compute=_compute(),
        stopping_condition=_stopping_condition(),
        base_job_name=name,
    )
    kwargs.update(overrides)
    return ModelTrainer(**kwargs)


class TestMinimalSubmission:
    """The baseline: does the simplest well-formed request get accepted?

    If these fail, everything else in the suite is noise -- they isolate "can we
    talk to the service at all with a valid payload" from the feature-specific
    tests below.
    """

    def test_minimal_request_is_accepted(self, sagemaker_session):
        name = unique_name("shallow-minimal")
        trainer = _trainer(sagemaker_session, name)

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_explicit_job_name_is_honoured(self, sagemaker_session):
        """The name we ask for is the name that gets created.

        Guards against the SDK silently rewriting or regenerating job names,
        which would break every user script that reconstructs an ARN from a name.
        """
        name = unique_name("shallow-named")
        trainer = _trainer(sagemaker_session, name)

        with submitted(trainer) as job:
            arn = assert_submitted(job)
            # base_job_name is a prefix; the SDK appends a timestamp suffix.
            assert (
                name in job.training_job_name
            ), f"requested base name {name!r} absent from {job.training_job_name!r}"
            assert job.training_job_name in arn

    def test_explicit_role_is_accepted(self, sagemaker_session, execution_role):
        """An explicitly passed role must pass PassRole server-side.

        The default path resolves the role implicitly; this proves the explicit
        path produces a payload the service also accepts.
        """
        name = unique_name("shallow-explicit-role")
        trainer = _trainer(sagemaker_session, name, role=execution_role)

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_command_instead_of_entry_script(self, sagemaker_session):
        """SourceCode.command is an alternative to entry_script; both must submit."""
        name = unique_name("shallow-command")
        source_code = SourceCode(
            source_dir=PARAM_SCRIPT_SOURCE_DIR,
            requirements="requirements.txt",
            command="python train.py",
        )
        trainer = _trainer(sagemaker_session, name, source_code=source_code)

        with submitted(trainer) as job:
            assert_submitted(job)


class TestSourceCodePackaging:
    """How ``source_code`` is packaged and uploaded before submission.

    Each variant produces a different S3 artifact, and the backend's
    role-assuming validators resolve that artifact -- so a packaging regression
    surfaces as a rejected request rather than a silent difference.

    Mirrors the source-code cases in the existing ``test_model_trainer.py`` deep
    suite (local tar file, shell entry script, custom distributed driver) so
    replacing it on the PR gate does not drop them.
    """

    def test_local_tar_file_source_dir(self, sagemaker_session):
        """A pre-built local ``.tar.gz`` is uploaded as-is rather than re-tarred."""
        name = unique_name("shallow-tar-source")
        source_code = SourceCode(
            source_dir=os.path.join(DATA_DIR, "script_mode", "code.tar.gz"),
            requirements="requirements.txt",
            entry_script="custom_script.py",
        )
        trainer = _trainer(sagemaker_session, name, source_code=source_code)

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_shell_entry_script(self, sagemaker_session):
        """A ``.sh`` entry script takes a different container-entrypoint path
        from a ``.py`` one."""
        name = unique_name("shallow-sh-entry")
        source_code = SourceCode(
            source_dir=PARAM_SCRIPT_SOURCE_DIR,
            requirements="requirements.txt",
            entry_script="train.sh",
        )
        trainer = _trainer(
            sagemaker_session,
            name,
            source_code=source_code,
            hyperparameters=CONTRACT_HYPERPARAMETERS,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_custom_distributed_driver(self, sagemaker_session):
        """A user-supplied distributed driver is uploaded alongside the source
        and changes the container entrypoint.

        Ported from ``test_model_trainer.py::test_custom_distributed_driver``:
        the driver directory is packaged separately from ``source_dir``, so this
        exercises a second upload the other tests never trigger.
        """

        class CustomDriver(DistributedConfig):
            process_count_per_node: int = None

            @property
            def driver_dir(self) -> str:
                return os.path.join(DATA_DIR, "custom_drivers")

            @property
            def driver_script(self) -> str:
                return "driver.py"

        name = unique_name("shallow-custom-driver")
        source_code = SourceCode(
            source_dir=os.path.join(DATA_DIR, "scripts"),
            entry_script="entry_script.py",
        )
        trainer = _trainer(
            sagemaker_session,
            name,
            source_code=source_code,
            hyperparameters={"epochs": 1},
            distributed=CustomDriver(process_count_per_node=2),
        )

        with submitted(trainer) as job:
            assert_submitted(job)


class TestPayloadShaping:
    """Fields the SDK must serialize into a form the service accepts.

    These are the highest-value tests in the suite: they are exactly the
    regressions that unit tests miss (because a mock accepts anything) and that
    deep integ tests catch far too slowly and expensively.
    """

    def test_hyperparameters_contract(self, sagemaker_session):
        """Nested/typed hyperparameters must survive serialization.

        The service requires a flat string->string map, so the SDK has to encode
        ints, floats, bools, lists and nested dicts. A regression here is a
        ValidationException at submit time, which is precisely what this catches.
        """
        name = unique_name("shallow-hp-contract")
        trainer = _trainer(sagemaker_session, name, hyperparameters=CONTRACT_HYPERPARAMETERS)

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_hyperparameters_from_json_file(self, sagemaker_session):
        """Hyperparameters given as a path to JSON must load and serialize."""
        name = unique_name("shallow-hp-json")
        trainer = _trainer(
            sagemaker_session,
            name,
            hyperparameters=os.path.join(PARAM_SCRIPT_SOURCE_DIR, "hyperparameters.json"),
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_hyperparameters_from_yaml_file(self, sagemaker_session):
        """Hyperparameters given as a path to YAML must load and serialize."""
        name = unique_name("shallow-hp-yaml")
        trainer = _trainer(
            sagemaker_session,
            name,
            hyperparameters=os.path.join(PARAM_SCRIPT_SOURCE_DIR, "hyperparameters.yaml"),
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_environment_variables(self, sagemaker_session):
        """Environment map must be accepted (the backend validates key syntax)."""
        name = unique_name("shallow-env")
        trainer = _trainer(
            sagemaker_session,
            name,
            environment={"MY_SETTING": "value", "ANOTHER_SETTING": "42"},
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_tags_are_accepted(self, sagemaker_session):
        """Tags travel a distinct authorization path.

        Tag-on-create is enforced by an interceptor at the public front end and
        by tag-governance checks, so a tagged request exercises gates an untagged
        one never reaches.
        """
        name = unique_name("shallow-tags")
        trainer = _trainer(
            sagemaker_session,
            name,
            tags=[shapes.Tag(key="Purpose", value="shallow-integ-test")],
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_explicit_output_data_config(self, sagemaker_session, output_path):
        """A caller-specified output location must validate server-side."""
        name = unique_name("shallow-output")
        trainer = _trainer(
            sagemaker_session,
            name,
            output_data_config=shapes.OutputDataConfig(s3_output_path=output_path),
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    @pytest.mark.parametrize("input_mode", ["File", "FastFile", "Pipe"])
    def test_training_input_modes(self, sagemaker_session, input_mode):
        """Every advertised input mode must be accepted.

        Cheap to cover here and easy to break: the mode is validated server-side
        against the channel configuration.
        """
        name = unique_name(f"shallow-mode-{input_mode.lower()}")
        trainer = _trainer(sagemaker_session, name, training_input_mode=input_mode)

        with submitted(trainer) as job:
            assert_submitted(job)


class TestInputDataConfiguration:
    """Input channels are resolved against S3 by the backend's role-assuming
    validators, so these tests prove both serialization and real S3 reachability
    under the execution role."""

    def test_single_s3_channel(self, sagemaker_session, train_data_uri):
        name = unique_name("shallow-one-channel")
        trainer = _trainer(sagemaker_session, name)

        with submitted(
            trainer,
            input_data_config=[InputData(channel_name="train", data_source=train_data_uri)],
        ) as job:
            assert_submitted(job)

    def test_multiple_s3_channels(self, sagemaker_session, train_data_uri, validation_data_uri):
        """Multiple channels must each resolve; channel-name rules are enforced
        server-side."""
        name = unique_name("shallow-two-channels")
        trainer = _trainer(sagemaker_session, name)

        with submitted(
            trainer,
            input_data_config=[
                InputData(channel_name="train", data_source=train_data_uri),
                InputData(channel_name="validation", data_source=validation_data_uri),
            ],
        ) as job:
            assert_submitted(job)

    def test_channel_with_content_type(self, sagemaker_session, train_data_uri):
        name = unique_name("shallow-content-type")
        trainer = _trainer(sagemaker_session, name)

        with submitted(
            trainer,
            input_data_config=[
                InputData(
                    channel_name="train",
                    data_source=train_data_uri,
                    content_type="application/jsonlines",
                )
            ],
        ) as job:
            assert_submitted(job)

    def test_s3_data_source_object(self, sagemaker_session, train_data_uri):
        """An explicit S3DataSource shape (rather than a bare URI) must serialize
        into a payload the service accepts."""
        name = unique_name("shallow-s3-datasource")
        trainer = _trainer(sagemaker_session, name)
        data_source = shapes.S3DataSource(
            s3_data_type="S3Prefix",
            s3_uri=train_data_uri,
            s3_data_distribution_type="FullyReplicated",
        )

        with submitted(
            trainer,
            input_data_config=[InputData(channel_name="train", data_source=data_source)],
        ) as job:
            assert_submitted(job)


class TestCheckpointingAndSpot:
    """Checkpointing and managed spot each add fields with their own backend
    validators, and spot additionally requires MaxWaitTimeInSeconds >=
    MaxRuntimeInSeconds -- a cross-field rule only the service enforces."""

    def test_checkpoint_config(self, sagemaker_session, output_path):
        """CheckpointConfig has a dedicated validator and an S3 location the
        backend resolves."""
        name = unique_name("shallow-checkpoint")
        trainer = _trainer(
            sagemaker_session,
            name,
            checkpoint_config=shapes.CheckpointConfig(
                s3_uri=f"{output_path}checkpoints/",
                local_path="/opt/ml/checkpoints/",
            ),
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_managed_spot_training(self, sagemaker_session):
        """Managed spot requires a max wait time at least as large as the max
        runtime; the service rejects the combination otherwise.

        Note this deliberately does not set ``keep_alive_period_in_seconds``:
        spot and warm pools are mutually exclusive, and a warm pool would outlive
        the stop.
        """
        name = unique_name("shallow-spot")
        compute = Compute(
            instance_type=DEFAULT_INSTANCE_TYPE,
            instance_count=DEFAULT_INSTANCE_COUNT,
            enable_managed_spot_training=True,
        )
        trainer = _trainer(
            sagemaker_session,
            name,
            compute=compute,
            stopping_condition=shapes.StoppingCondition(
                max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
                max_wait_time_in_seconds=MAX_RUNTIME_IN_SECONDS,
            ),
        )

        with submitted(trainer) as job:
            assert_submitted(job)


class TestComputeConfiguration:
    """Compute shapes are validated by several distinct backend validators
    (instance type, instance count, volume size, distribution)."""

    def test_multi_instance_request(self, sagemaker_session):
        """instance_count > 1 changes the accepted shape of the request."""
        name = unique_name("shallow-multi-instance")
        trainer = _trainer(sagemaker_session, name, compute=_compute(instance_count=2))

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_explicit_volume_size(self, sagemaker_session):
        """Volume size has its own validator with min/max bounds."""
        name = unique_name("shallow-volume")
        compute = Compute(
            instance_type=DEFAULT_INSTANCE_TYPE,
            instance_count=DEFAULT_INSTANCE_COUNT,
            volume_size_in_gb=50,
        )
        trainer = _trainer(sagemaker_session, name, compute=compute)

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_torchrun_distributed(self, sagemaker_session):
        """Distributed configs inject env/entrypoint changes; the resulting
        payload must still be accepted."""
        name = unique_name("shallow-torchrun")
        trainer = _trainer(
            sagemaker_session,
            name,
            compute=_compute(instance_count=2),
            distributed=Torchrun(),
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_mpi_distributed(self, sagemaker_session):
        name = unique_name("shallow-mpi")
        trainer = _trainer(
            sagemaker_session,
            name,
            compute=_compute(instance_count=2),
            distributed=MPI(),
        )

        with submitted(trainer) as job:
            assert_submitted(job)


class TestNetworkingAndSecurity:
    """Isolation and encryption flags are surfaced as IAM condition keys, so
    these requests are authorized differently from the baseline."""

    def test_network_isolation(self, sagemaker_session):
        name = unique_name("shallow-net-isolation")
        trainer = _trainer(
            sagemaker_session, name, networking=Networking(enable_network_isolation=True)
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_inter_container_traffic_encryption(self, sagemaker_session):
        """Encryption between nodes only applies to multi-instance jobs."""
        name = unique_name("shallow-icte")
        trainer = _trainer(
            sagemaker_session,
            name,
            compute=_compute(instance_count=2),
            networking=Networking(enable_inter_container_traffic_encryption=True),
        )

        with submitted(trainer) as job:
            assert_submitted(job)


class TestRejectedRequests:
    """Negative cases.

    Without these the suite would pass as long as *something* was accepted,
    which would hide a bug that made the SDK send a permissive-but-wrong
    payload. Each case asserts a specific rejection, and the harness stops the
    job if one is unexpectedly accepted.
    """

    def test_nonexistent_input_data_is_rejected(self, sagemaker_session, nonexistent_data_uri):
        """Proves input validation genuinely reaches S3.

        The single most valuable negative test here: it is the assertion that the
        expensive role-assuming validators actually ran, rather than being
        skipped or silently swallowed.
        """
        trainer = _trainer(sagemaker_session, unique_name("shallow-bad-input"))

        assert_rejected(
            trainer,
            ("does not exist", "ValidationException", "ValidationError", "S3", "not found"),
            input_data_config=[InputData(channel_name="train", data_source=nonexistent_data_uri)],
        )

    def test_invalid_instance_type_is_rejected(self, sagemaker_session):
        """A syntactically-valid but nonexistent instance type must be refused."""
        trainer = _trainer(
            sagemaker_session,
            unique_name("shallow-bad-instance"),
            compute=_compute(instance_type="ml.nonexistent.xlarge"),
        )

        assert_rejected(
            trainer,
            ("instance", "Instance", "ValidationException", "ValidationError", "not supported"),
        )

    def test_nonexistent_training_image_is_rejected(self, sagemaker_session, account_id, region):
        """The backend resolves the training image against ECR under the
        customer's role, so an image that does not exist must be refused.

        Uses the caller's own account so the failure is "repository absent"
        rather than "cross-account access denied".
        """
        bogus_image = (
            f"{account_id}.dkr.ecr.{region}.amazonaws.com/" "shallow-integ-test-no-such-repo:latest"
        )
        trainer = _trainer(
            sagemaker_session, unique_name("shallow-bad-image"), training_image=bogus_image
        )

        assert_rejected(
            trainer,
            (
                "image",
                "Image",
                "ECR",
                "repository",
                "RepositoryNotFound",
                "ValidationException",
                "ValidationError",
            ),
        )

    def test_unassumable_role_is_rejected(self, sagemaker_session, account_id):
        """A role that cannot be used for training must be refused.

        Covers the "does the caller hold the required permissions" half of what
        this suite exists to assert.

        Note where this is caught: ``ModelTrainer.__init__`` resolves and
        validates the role via ``iam:SimulatePrincipalPolicy``, so a bad role is
        rejected at *construction* -- the request never reaches
        CreateTrainingJob. That is strictly better than a server-side rejection
        (faster, clearer message), so this asserts around the constructor rather
        than around ``train()``. Verified against AWS: the SDK raises
        ``RoleValidationError`` naming the role and the permissions it lacks.
        """
        bogus_role = f"arn:aws:iam::{account_id}:role/shallow-integ-test-no-such-role"

        with pytest.raises(Exception) as excinfo:
            _trainer(sagemaker_session, unique_name("shallow-bad-role"), role=bogus_role)

        message = str(excinfo.value)
        assert any(
            token in message
            for token in (
                "cannot be used",
                "RoleValidationError",
                "AccessDenied",
                "not authorized",
                "cannot be assumed",
                "does not exist",
            )
        ), f"unexpected rejection reason: {message}"

    def test_duplicate_job_name_is_rejected(self, sagemaker_session, execution_role, output_path):
        """The final gate before the ARN is a conditional write that rejects
        duplicate job names with ResourceInUse.

        Asserting it proves a submission reached the very *end* of the create
        path -- the durable write -- and not merely the validators in front of
        it. ``ModelTrainer`` appends a timestamp to ``base_job_name``, so it can
        never produce a collision by design; this drives the underlying resource
        API directly in order to re-use one exact name twice.
        """
        from sagemaker.core.resources import TrainingJob

        job_name = unique_name("shallow-duplicate")

        def create():
            return TrainingJob.create(
                session=sagemaker_session.boto_session,
                training_job_name=job_name,
                role_arn=execution_role,
                algorithm_specification=shapes.AlgorithmSpecification(
                    training_image=CPU_IMAGE, training_input_mode="File"
                ),
                output_data_config=shapes.OutputDataConfig(s3_output_path=output_path),
                resource_config=shapes.ResourceConfig(
                    instance_type=DEFAULT_INSTANCE_TYPE,
                    instance_count=DEFAULT_INSTANCE_COUNT,
                    volume_size_in_gb=30,
                ),
                stopping_condition=_stopping_condition(),
            )

        first = None
        try:
            first = create()
            assert_submitted(first, expected_name=job_name)

            with pytest.raises(Exception) as excinfo:
                create()

            message = str(excinfo.value)
            assert any(
                token in message
                for token in ("already exists", "ResourceInUse", "ResourceInUseException")
            ), f"unexpected rejection reason: {message}"
        finally:
            stop_quietly(first)
