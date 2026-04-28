"""End-to-end integration tests for session propagation in wait flows.

Tests that wait=True works correctly for various resource types when using
sagemaker_session. These tests create real AWS resources and wait for them
to complete, verifying the session fix from GitHub issue #5765.

Usage:
    python -m pytest tests/integ/test_session_wait_e2e.py -v -s
    # Or run individual tests:
    python -m pytest tests/integ/test_session_wait_e2e.py::test_processing_job_wait -v -s
    python -m pytest tests/integ/test_session_wait_e2e.py::test_training_job_wait -v -s

Prerequisites:
    - Valid AWS credentials configured
    - IAM role with SageMaker permissions
    - pip install sagemaker-core (from this repo)
    - pip install sagemaker-train (for test_training_job_wait)

Note: These tests create real SageMaker jobs and incur AWS costs.
      Each test takes 3-10 minutes to complete.
"""

import os
import time
import tempfile
import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core import image_uris


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sm_session():
    """Create a SageMaker session for all tests."""
    return Session()


@pytest.fixture(scope="module")
def role(sm_session):
    """Get the execution role."""
    return get_execution_role()


@pytest.fixture(scope="module")
def region(sm_session):
    """Get the region."""
    return sm_session.boto_region_name


@pytest.fixture(scope="module")
def training_image(region):
    """Get a PyTorch training image URI."""
    return image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.2.0",
        py_version="py310",
        instance_type="ml.m5.large",
        image_scope="training",
    )


@pytest.fixture(scope="module")
def processing_image():
    """Get a processing image URI."""
    return "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.2.0-cpu-py310"


# ── Test 1: Processing job wait via ScriptProcessor ─────────────────────────


def test_processing_job_wait(sm_session, role, processing_image):
    """Test that ScriptProcessor.run(wait=True) works with sagemaker_session.

    This is the primary flow reported in GitHub issue #5765.
    The ScriptProcessor passes sagemaker_session, and wait=True triggers
    ProcessingJob.refresh() which must use the session's credentials.
    """
    from sagemaker.core.processing import ScriptProcessor

    script_dir = tempfile.mkdtemp()
    script_path = os.path.join(script_dir, "hello.py")
    with open(script_path, "w") as f:
        f.write('print("Hello from processing job!")\n')

    processor = ScriptProcessor(
        image_uri=processing_image,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=sm_session,
        base_job_name="integ-session-proc",
    )

    start = time.time()
    processor.run(code=script_path, wait=True, logs=False)
    elapsed = time.time() - start

    assert processor.latest_job is not None
    print(f"\nProcessing job completed in {elapsed:.0f}s")
    print(f"Job: {processor.latest_job.processing_job_name}")
    print(f"Status: {processor.latest_job.processing_job_status}")

    assert processor.latest_job.processing_job_status == "Completed"


# ── Test 2: Training job wait via ModelTrainer ──────────────────────────────


def test_training_job_wait(sm_session, role, region, training_image):
    """Test that ModelTrainer.train(wait=True) works with sagemaker_session.

    This is the other primary flow reported in GitHub issue #5765.
    ModelTrainer passes sagemaker_session, creates a TrainingJob, and
    wait=True triggers TrainingJob.refresh() which must use the session.

    Requires sagemaker-train to be installed.
    """
    pytest.importorskip("sagemaker.train", reason="sagemaker-train not installed")
    from sagemaker.train.model_trainer import ModelTrainer
    from sagemaker.core.training.configs import Compute, SourceCode

    script_dir = tempfile.mkdtemp()
    script_path = os.path.join(script_dir, "train.py")
    with open(script_path, "w") as f:
        f.write(
            'import os\n'
            'print("Hello from training job!")\n'
            'model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")\n'
            'os.makedirs(model_dir, exist_ok=True)\n'
            'with open(os.path.join(model_dir, "dummy.txt"), "w") as f:\n'
            '    f.write("done")\n'
            'print("Training complete!")\n'
        )

    trainer = ModelTrainer(
        training_image=training_image,
        source_code=SourceCode(source_dir=script_dir, entry_script="train.py"),
        compute=Compute(instance_type="ml.m5.large", instance_count=1),
        role=role,
        sagemaker_session=sm_session,
        base_job_name="integ-session-train",
    )

    start = time.time()
    trainer.train(wait=True, logs=False)
    elapsed = time.time() - start

    job = trainer._latest_training_job
    assert job is not None
    print(f"\nTraining job completed in {elapsed:.0f}s")
    print(f"Job: {job.training_job_name}")
    print(f"Status: {job.training_job_status}")

    assert job.training_job_status == "Completed"


# ── Test 3: Processing job wait via resource class directly ─────────────────


def test_processing_job_wait_via_resource(sm_session, role, region):
    """Test ProcessingJob.create() + wait() using the resource class directly.

    Verifies that create() stores _session on the instance and wait()
    uses it for refresh() calls.
    """
    from sagemaker.core.resources import ProcessingJob
    from sagemaker.core.shapes import (
        AppSpecification,
        ProcessingResources,
        ProcessingClusterConfig,
    )

    job_name = f"integ-session-proc-direct-{int(time.time())}"

    processing_job = ProcessingJob.create(
        processing_job_name=job_name,
        role_arn=role,
        app_specification=AppSpecification(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.2.0-cpu-py310",
            container_entrypoint=["python3", "-c", 'print("Hello from direct processing!")'],
        ),
        processing_resources=ProcessingResources(
            cluster_config=ProcessingClusterConfig(
                instance_count=1,
                instance_type="ml.m5.large",
                volume_size_in_gb=10,
            ),
        ),
        session=sm_session.boto_session,
    )

    assert processing_job is not None
    assert hasattr(processing_job, "_session"), "ProcessingJob should have _session attribute"

    print(f"\nCreated processing job: {job_name}")
    print(f"Waiting for completion...")

    start = time.time()
    processing_job.wait()
    elapsed = time.time() - start

    print(f"Processing job completed in {elapsed:.0f}s")
    print(f"Status: {processing_job.processing_job_status}")

    assert processing_job.processing_job_status == "Completed"


# ── Test 4: Verify _session survives refresh cycle ──────────────────────────


def test_session_survives_refresh(sm_session, role):
    """Test that _session persists through multiple refresh() calls.

    Creates a processing job, then manually calls refresh() multiple times
    to verify the session attribute isn't lost during deserialization.
    """
    from sagemaker.core.resources import ProcessingJob
    from sagemaker.core.shapes import (
        AppSpecification,
        ProcessingResources,
        ProcessingClusterConfig,
    )

    job_name = f"integ-session-refresh-{int(time.time())}"

    job = ProcessingJob.create(
        processing_job_name=job_name,
        role_arn=role,
        app_specification=AppSpecification(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.2.0-cpu-py310",
            container_entrypoint=["python3", "-c", 'print("Refresh test!")'],
        ),
        processing_resources=ProcessingResources(
            cluster_config=ProcessingClusterConfig(
                instance_count=1,
                instance_type="ml.m5.large",
                volume_size_in_gb=10,
            ),
        ),
        session=sm_session.boto_session,
    )

    original_session = job._session

    # Refresh multiple times and verify _session persists
    for i in range(3):
        job.refresh()
        assert job._session is original_session, f"_session lost after refresh #{i+1}"
        print(f"Refresh #{i+1}: status={job.processing_job_status}, _session intact ✓")
        time.sleep(2)

    # Now wait for completion
    job.wait()
    assert job._session is original_session, "_session lost after wait()"
    assert job.processing_job_status == "Completed"
    print(f"Final status: {job.processing_job_status}, _session intact ✓")
