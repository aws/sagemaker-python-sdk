"""End-to-end integration tests for session propagation in wait flows.

Tests that wait=True works correctly for various resource types when using
sagemaker_session. These tests create real AWS resources and wait for them
to complete, verifying the session fix from GitHub issue #5765.

Usage:
    python -m pytest tests/integ/test_session_wait_e2e.py -v -s
    # Or run individual tests:
    python -m pytest tests/integ/test_session_wait_e2e.py::test_processing_job_wait -v -s
    python -m pytest tests/integ/test_session_wait_e2e.py::test_training_job_wait -v -s
    python -m pytest tests/integ/test_session_wait_e2e.py::test_training_job_wait_via_resource -v -s
    python -m pytest tests/integ/test_session_wait_e2e.py::test_transform_job_wait -v -s

Prerequisites:
    - Valid AWS credentials configured
    - IAM role with SageMaker permissions
    - pip install sagemaker-core (from this repo)

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
    assert processor.latest_job.processing_job_status in ("Completed", "Failed", "Stopped")
    print(f"\nProcessing job completed in {elapsed:.0f}s")
    print(f"Job: {processor.latest_job.processing_job_name}")
    print(f"Status: {processor.latest_job.processing_job_status}")

    # Verify the job actually completed (not just that wait returned)
    assert processor.latest_job.processing_job_status == "Completed"


# ── Test 2: Training job wait via ModelTrainer ──────────────────────────────

def test_training_job_wait(sm_session, role, region, training_image):
    """Test that ModelTrainer.train(wait=True) works with sagemaker_session.

    This is the other primary flow reported in GitHub issue #5765.
    ModelTrainer passes sagemaker_session, creates a TrainingJob, and
    wait=True triggers TrainingJob.refresh() which must use the session.
    """
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


# ── Test 3: Training job wait via resource class directly ───────────────────

def test_training_job_wait_via_resource(sm_session, role, region, training_image):
    """Test TrainingJob.create() + wait() using the resource class directly.

    This bypasses ModelTrainer and tests the resource class session
    propagation directly: create() stores _session, wait() calls refresh()
    which uses the stored _session.

    Uses a simple container command instead of a training script to avoid
    script packaging complexity.
    """
    from sagemaker.core.resources import TrainingJob
    from sagemaker.core.shapes import (
        AlgorithmSpecification,
        OutputDataConfig,
        ResourceConfig,
        StoppingCondition,
    )

    bucket = sm_session.default_bucket()
    prefix = f"integ-session-direct-{int(time.time())}"
    job_name = f"integ-direct-{int(time.time())}"

    # Use container_entrypoint + container_arguments to run inline code
    # This avoids the script packaging issue entirely
    training_job = TrainingJob.create(
        training_job_name=job_name,
        role_arn=role,
        algorithm_specification=AlgorithmSpecification(
            training_image=training_image,
            training_input_mode="File",
            container_entrypoint=["python3", "-c"],
            container_arguments=[
                "import os; "
                "print('Direct resource class training!'); "
                "model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model'); "
                "os.makedirs(model_dir, exist_ok=True); "
                "open(os.path.join(model_dir, 'dummy.txt'), 'w').write('done'); "
                "print('Training complete!')"
            ],
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{bucket}/{prefix}/output",
        ),
        resource_config=ResourceConfig(
            instance_type="ml.m5.large",
            instance_count=1,
            volume_size_in_gb=10,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=600),
        session=sm_session.boto_session,
    )

    assert training_job is not None
    assert hasattr(training_job, "_session"), "TrainingJob should have _session attribute"
    assert training_job._session is sm_session.boto_session, "_session should be the boto session"

    print(f"\nCreated training job: {job_name}")
    print(f"Waiting for completion...")

    start = time.time()
    training_job.wait()
    elapsed = time.time() - start

    print(f"Training job completed in {elapsed:.0f}s")
    print(f"Status: {training_job.training_job_status}")

    assert training_job.training_job_status == "Completed"


# ── Test 4: Processing job wait via resource class directly ─────────────────

def test_processing_job_wait_via_resource(sm_session, role, region):
    """Test ProcessingJob.create() + wait() using the resource class directly."""
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


# ── Test 5: Verify _session survives refresh cycle ──────────────────────────

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
        assert job._session is original_session, (
            f"_session lost after refresh #{i+1}"
        )
        print(f"Refresh #{i+1}: status={job.processing_job_status}, _session intact ✓")
        time.sleep(2)

    # Now wait for completion
    job.wait()
    assert job._session is original_session, "_session lost after wait()"
    assert job.processing_job_status == "Completed"
    print(f"Final status: {job.processing_job_status}, _session intact ✓")


# ── Test 6: TrainingJob.get() with session, then wait ───────────────────────

def test_get_then_wait(sm_session, role, training_image):
    """Test TrainingJob.get() with session, then call wait().

    This tests the get() → _session storage → wait() → refresh() flow
    without going through create().
    """
    from sagemaker.core.resources import TrainingJob
    from sagemaker.core.shapes import (
        AlgorithmSpecification,
        OutputDataConfig,
        ResourceConfig,
        StoppingCondition,
    )

    bucket = sm_session.default_bucket()
    job_name = f"integ-get-wait-{int(time.time())}"

    # Create the job using container_entrypoint to avoid script packaging
    TrainingJob.create(
        training_job_name=job_name,
        role_arn=role,
        algorithm_specification=AlgorithmSpecification(
            training_image=training_image,
            training_input_mode="File",
            container_entrypoint=["python3", "-c"],
            container_arguments=[
                "import os; "
                "print('Get-then-wait test!'); "
                "model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model'); "
                "os.makedirs(model_dir, exist_ok=True); "
                "open(os.path.join(model_dir, 'dummy.txt'), 'w').write('done')"
            ],
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{bucket}/integ-get-wait/output",
        ),
        resource_config=ResourceConfig(
            instance_type="ml.m5.large",
            instance_count=1,
            volume_size_in_gb=10,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=600),
        session=sm_session.boto_session,
    )

    # Now get() the job with a session and wait
    job = TrainingJob.get(
        training_job_name=job_name,
        session=sm_session.boto_session,
    )

    assert job._session is sm_session.boto_session, "get() should store _session"
    print(f"\nGot training job: {job_name}")
    print(f"_session stored: ✓")

    start = time.time()
    job.wait()
    elapsed = time.time() - start

    print(f"Wait completed in {elapsed:.0f}s")
    print(f"Status: {job.training_job_status}")
    assert job.training_job_status == "Completed"


# ── Test 7: Transform job wait ──────────────────────────────────────────────

def test_transform_job_wait(sm_session, role, region, training_image):
    """Test TransformJob.create() + wait() with session.

    Creates a model from a training job output, then runs a batch transform
    job and waits for it. Tests TransformJob.refresh() session propagation.
    """
    from sagemaker.core.resources import Model, TransformJob
    from sagemaker.core.shapes import (
        ContainerDefinition,
        TransformInput,
        TransformDataSource,
        TransformS3DataSource,
        TransformOutput,
        TransformResources,
    )

    bucket = sm_session.default_bucket()
    ts = int(time.time())

    # Create a dummy model artifact (empty tar.gz)
    model_dir = tempfile.mkdtemp()
    model_tar = os.path.join(model_dir, "model.tar.gz")
    import tarfile
    with tarfile.open(model_tar, "w:gz") as tar:
        # Add a dummy file
        dummy_path = os.path.join(model_dir, "dummy.txt")
        with open(dummy_path, "w") as f:
            f.write("dummy model")
        tar.add(dummy_path, arcname="dummy.txt")

    # Upload model artifact
    model_s3_uri = sm_session.upload_data(
        path=model_tar,
        bucket=bucket,
        key_prefix=f"integ-transform-{ts}/model",
    )

    # Create dummy input data
    input_dir = tempfile.mkdtemp()
    input_path = os.path.join(input_dir, "input.csv")
    with open(input_path, "w") as f:
        f.write("1,2,3\n4,5,6\n")
    input_s3_uri = sm_session.upload_data(
        path=input_dir,
        bucket=bucket,
        key_prefix=f"integ-transform-{ts}/input",
    )

    # Use a simple inference image that just echoes input
    # sklearn image is lightweight and handles CSV
    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        instance_type="ml.m5.large",
        image_scope="inference",
    )

    # Create model
    model_name = f"integ-transform-model-{ts}"
    model = Model.create(
        model_name=model_name,
        primary_container=ContainerDefinition(
            image=sklearn_image,
            model_data_url=model_s3_uri,
        ),
        execution_role_arn=role,
        session=sm_session.boto_session,
    )
    assert model is not None
    assert model._session is sm_session.boto_session
    print(f"\nCreated model: {model_name}")

    # Create transform job
    transform_job_name = f"integ-transform-{ts}"
    transform_job = TransformJob.create(
        transform_job_name=transform_job_name,
        model_name=model_name,
        transform_input=TransformInput(
            data_source=TransformDataSource(
                s3_data_source=TransformS3DataSource(
                    s3_data_type="S3Prefix",
                    s3_uri=f"s3://{bucket}/integ-transform-{ts}/input/",
                ),
            ),
            content_type="text/csv",
        ),
        transform_output=TransformOutput(
            s3_output_path=f"s3://{bucket}/integ-transform-{ts}/output/",
        ),
        transform_resources=TransformResources(
            instance_type="ml.m5.large",
            instance_count=1,
        ),
        session=sm_session.boto_session,
    )

    assert transform_job is not None
    assert hasattr(transform_job, "_session")
    assert transform_job._session is sm_session.boto_session
    print(f"Created transform job: {transform_job_name}")

    start = time.time()
    try:
        transform_job.wait()
    except Exception as e:
        # Transform may fail because the model isn't a real inference model,
        # but the wait mechanism itself should work (session propagation)
        print(f"Transform job ended with: {type(e).__name__}: {e}")
        # Verify the job reached a terminal state (not a credentials error)
        transform_job.refresh()
        assert transform_job.transform_job_status in ("Completed", "Failed", "Stopped"), (
            f"Unexpected status: {transform_job.transform_job_status}"
        )

    elapsed = time.time() - start
    print(f"Transform job finished in {elapsed:.0f}s")
    print(f"Status: {transform_job.transform_job_status}")

    # Clean up model
    try:
        model.delete()
        print(f"Deleted model: {model_name}")
    except Exception:
        pass


# ── Test 8: Endpoint wait_for_status and wait_for_delete ────────────────────

def test_endpoint_wait_for_status_and_delete(sm_session, role, region):
    """Test Endpoint.wait_for_status() and wait_for_delete() with session.

    Creates a model + endpoint config + endpoint, waits for InService,
    then deletes and waits for deletion. Tests two different wait patterns
    that both use refresh() internally.
    """
    from sagemaker.core.resources import Model, EndpointConfig, Endpoint
    from sagemaker.core.shapes import (
        ContainerDefinition,
        ProductionVariant,
    )

    bucket = sm_session.default_bucket()
    ts = int(time.time())

    # Create a dummy model artifact
    model_dir = tempfile.mkdtemp()
    model_tar = os.path.join(model_dir, "model.tar.gz")
    import tarfile
    with tarfile.open(model_tar, "w:gz") as tar:
        dummy_path = os.path.join(model_dir, "dummy.txt")
        with open(dummy_path, "w") as f:
            f.write("dummy model")
        tar.add(dummy_path, arcname="dummy.txt")

    model_s3_uri = sm_session.upload_data(
        path=model_tar,
        bucket=bucket,
        key_prefix=f"integ-endpoint-{ts}/model",
    )

    # Use a lightweight inference image
    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        instance_type="ml.m5.large",
        image_scope="inference",
    )

    # Create model
    model_name = f"integ-ep-model-{ts}"
    model = Model.create(
        model_name=model_name,
        primary_container=ContainerDefinition(
            image=sklearn_image,
            model_data_url=model_s3_uri,
        ),
        execution_role_arn=role,
        session=sm_session.boto_session,
    )
    print(f"\nCreated model: {model_name}")

    # Create endpoint config
    ep_config_name = f"integ-ep-config-{ts}"
    ep_config = EndpointConfig.create(
        endpoint_config_name=ep_config_name,
        production_variants=[
            ProductionVariant(
                variant_name="AllTraffic",
                model_name=model_name,
                instance_type="ml.m5.large",
                initial_instance_count=1,
            ),
        ],
        session=sm_session.boto_session,
    )
    print(f"Created endpoint config: {ep_config_name}")

    # Create endpoint
    ep_name = f"integ-ep-{ts}"
    endpoint = Endpoint.create(
        endpoint_name=ep_name,
        endpoint_config_name=ep_config_name,
        session=sm_session.boto_session,
    )

    assert endpoint is not None
    assert hasattr(endpoint, "_session")
    assert endpoint._session is sm_session.boto_session
    print(f"Created endpoint: {ep_name}")

    # Wait for InService
    print("Waiting for endpoint to reach InService...")
    start = time.time()
    endpoint.wait_for_status(target_status="InService")
    elapsed = time.time() - start
    print(f"Endpoint InService in {elapsed:.0f}s")
    assert endpoint.endpoint_status == "InService"

    # Delete endpoint and wait for deletion
    print("Deleting endpoint...")
    endpoint.delete()
    start = time.time()
    endpoint.wait_for_delete()
    elapsed = time.time() - start
    print(f"Endpoint deleted in {elapsed:.0f}s")

    # Clean up endpoint config and model
    try:
        ep_config.delete()
        print(f"Deleted endpoint config: {ep_config_name}")
    except Exception:
        pass
    try:
        model.delete()
        print(f"Deleted model: {model_name}")
    except Exception:
        pass


# ── Test 9: CompilationJob wait ─────────────────────────────────────────────

def test_compilation_job_wait(sm_session, role, region):
    """Test CompilationJob.create() + wait() with session.

    Compiles a model for a target device. Tests CompilationJob.refresh()
    session propagation.
    """
    from sagemaker.core.resources import CompilationJob
    from sagemaker.core.shapes import (
        InputConfig,
        OutputConfig,
        StoppingCondition,
    )

    bucket = sm_session.default_bucket()
    ts = int(time.time())

    # Create a dummy model artifact (Neo expects a tar.gz with model files)
    model_dir = tempfile.mkdtemp()
    model_tar = os.path.join(model_dir, "model.tar.gz")
    import tarfile
    with tarfile.open(model_tar, "w:gz") as tar:
        dummy_path = os.path.join(model_dir, "model.pth")
        with open(dummy_path, "w") as f:
            f.write("dummy pytorch model")
        tar.add(dummy_path, arcname="model.pth")

    model_s3_uri = sm_session.upload_data(
        path=model_tar,
        bucket=bucket,
        key_prefix=f"integ-compile-{ts}/model",
    )

    job_name = f"integ-compile-{ts}"

    compilation_job = CompilationJob.create(
        compilation_job_name=job_name,
        role_arn=role,
        input_config=InputConfig(
            s3_uri=model_s3_uri,
            data_input_config='{"input0": [1, 3, 224, 224]}',
            framework="PYTORCH",
        ),
        output_config=OutputConfig(
            s3_output_location=f"s3://{bucket}/integ-compile-{ts}/output/",
            target_device="ml_m5",
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=900),
        session=sm_session.boto_session,
    )

    assert compilation_job is not None
    assert hasattr(compilation_job, "_session")
    assert compilation_job._session is sm_session.boto_session
    print(f"\nCreated compilation job: {job_name}")

    start = time.time()
    try:
        compilation_job.wait()
    except Exception as e:
        # Compilation may fail because the model isn't a real PyTorch model,
        # but the wait mechanism should work (session propagation)
        print(f"Compilation job ended with: {type(e).__name__}: {e}")
        compilation_job.refresh()
        assert compilation_job.compilation_job_status in ("COMPLETED", "FAILED", "STOPPED"), (
            f"Unexpected status: {compilation_job.compilation_job_status}"
        )

    elapsed = time.time() - start
    print(f"Compilation job finished in {elapsed:.0f}s")
    print(f"Status: {compilation_job.compilation_job_status}")
