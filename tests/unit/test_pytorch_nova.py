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
from __future__ import absolute_import
import pytest
import tempfile
from mock import Mock, patch
from omegaconf import OmegaConf

from sagemaker.estimator import EstimatorBase

from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.estimator import (
    _is_nova_recipe,
    _device_get_distribution,
)
from sagemaker.inputs import TrainingInput
from sagemaker.session_settings import SessionSettings

# Constants for testing
ROLE = "Dummy"
REGION = "us-west-2"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
INSTANCE_TYPE_GPU = "ml.p4d.24xlarge"
IMAGE_URI = "sagemaker-pytorch"


@pytest.fixture(name="sagemaker_session")
def fixture_sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_resource=None,
        s3_client=None,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    session.upload_data = Mock(return_value="s3://mybucket/recipes/nova-recipe.yaml")
    session.sagemaker_config = {}
    return session


def test_is_nova_recipe():
    """Test that _is_nova_recipe correctly identifies Nova recipes."""
    # Valid Nova recipe
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foo-bar",
                "model_name_or_path": "foo-bar/foo-bar123",
            }
        }
    )
    assert _is_nova_recipe(recipe) is True

    # Not a Nova recipe - missing run section
    recipe = OmegaConf.create(
        {
            "trainer": {
                "model_type": "amazon.nova.foo-bar",
                "model_name_or_path": "foo-bar/foo-bar123",
            }
        }
    )
    assert _is_nova_recipe(recipe) is False

    # Not a Nova recipe - wrong model_type
    recipe = OmegaConf.create(
        {"run": {"model_type": "foo-bar3", "model_name_or_path": "foo-bar/foo-bar123"}}
    )
    assert _is_nova_recipe(recipe) is False

    # Not a Nova recipe - missing model_name_or_path
    recipe = OmegaConf.create({"run": {"model_type": "amazon.nova.foo-bar"}})
    assert _is_nova_recipe(recipe) is False


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_resolve_and_save")
def test_setup_for_nova_recipe_with_model_name(mock_resolve_save, sagemaker_session):
    """Test that _setup_for_nova_recipe correctly sets up hyperparameters for Nova recipes with model name."""
    # Create a mock recipe
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foobar3",
                "model_name_or_path": "foobar/foobar-3-8b",
                "replicas": 4,
            }
        }
    )

    # Setup the expected return value
    expected_args = {
        "hyperparameters": {"base_model": "foobar/foobar-3-8b"},
        "entry_point": None,
        "source_dir": None,
        "distribution": {},
        "default_image_uri": IMAGE_URI,
    }

    # Mock the _setup_for_nova_recipe method
    with patch(
        "sagemaker.pytorch.estimator.PyTorch._setup_for_nova_recipe", return_value=expected_args
    ) as mock_nova_setup:
        # Create the PyTorch estimator with mocked _recipe_load
        with patch(
            "sagemaker.pytorch.estimator.PyTorch._recipe_load", return_value=("nova_recipe", recipe)
        ):
            # Mock _recipe_resolve_and_save to return our recipe
            mock_resolve_save.return_value = recipe

            pytorch = PyTorch(
                training_recipe="nova_recipe",
                role=ROLE,
                sagemaker_session=sagemaker_session,
                instance_count=INSTANCE_COUNT,
                instance_type=INSTANCE_TYPE_GPU,
                image_uri=IMAGE_URI,
                framework_version="1.13.1",
                py_version="py3",
            )

            # Check that the Nova recipe was correctly identified
            assert pytorch.is_nova_or_eval_recipe is True

            # Verify _setup_for_nova_recipe was called
            mock_nova_setup.assert_called_once()
            call_args = mock_nova_setup.call_args
            assert len(call_args[0]) >= 2  # Check that at least recipe and recipe_name were passed
            assert call_args[0][0] == recipe  # first arg should be recipe
            assert call_args[0][1] == "nova_recipe"  # second arg should be recipe_name


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_resolve_and_save")
def test_setup_for_nova_recipe_with_s3_path(mock_resolve_save, sagemaker_session):
    """Test that _setup_for_nova_recipe correctly sets up hyperparameters for Nova recipes with S3 path."""
    # Create a mock recipe with S3 path
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foobar3",
                "model_name_or_path": "s3://mybucket/models/foobar3",
                "replicas": 4,
            }
        }
    )

    # Setup the expected return value
    expected_args = {
        "hyperparameters": {"base_model_location": "s3://mybucket/models/foobar3"},
        "entry_point": None,
        "source_dir": None,
        "distribution": {},
        "default_image_uri": IMAGE_URI,
    }

    # Mock the _setup_for_nova_recipe method
    with patch(
        "sagemaker.pytorch.estimator.PyTorch._setup_for_nova_recipe", return_value=expected_args
    ) as mock_nova_setup:
        # Create the PyTorch estimator with mocked _recipe_load
        with patch(
            "sagemaker.pytorch.estimator.PyTorch._recipe_load", return_value=("nova_recipe", recipe)
        ):
            # Mock _recipe_resolve_and_save to return our recipe
            mock_resolve_save.return_value = recipe

            pytorch = PyTorch(
                training_recipe="nova_recipe",
                role=ROLE,
                sagemaker_session=sagemaker_session,
                instance_count=INSTANCE_COUNT,
                instance_type=INSTANCE_TYPE_GPU,
                image_uri=IMAGE_URI,
                framework_version="1.13.1",
                py_version="py3",
            )

            # Check that the Nova recipe was correctly identified
            assert pytorch.is_nova_or_eval_recipe is True

            # Verify _setup_for_nova_recipe was called
            mock_nova_setup.assert_called_once()

            # Verify that hyperparameters were set correctly
            assert (
                pytorch._hyperparameters.get("base_model_location")
                == "s3://mybucket/models/foobar3"
            )


def test_device_handle_instance_count_with_nova_replicas():
    """Test that _device_handle_instance_count correctly gets instance_count from Nova recipe replicas."""
    # Create mock recipe with replicas
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foobar3",
                "model_name_or_path": "foobar/foobar-3-8b",
                "replicas": 4,
            }
        }
    )

    # Test with no instance_count in kwargs
    kwargs = {}
    PyTorch._device_handle_instance_count(kwargs, recipe)
    assert kwargs["instance_count"] == 4


def test_device_handle_instance_count_with_nova_no_replicas():
    """Test that _device_handle_instance_count raises an error when no instance_count or replicas are provided."""
    # Create mock recipe without replicas
    recipe = OmegaConf.create(
        {"run": {"model_type": "amazon.nova.foobar3", "model_name_or_path": "foobar/foobar-3-8b"}}
    )

    # Test with no instance_count in kwargs
    kwargs = {}
    with pytest.raises(ValueError) as error:
        PyTorch._device_handle_instance_count(kwargs, recipe)

    assert "Must set either instance_count argument for estimator or" in str(error)


@patch("sagemaker.pytorch.estimator.logger.warning")
def test_device_handle_instance_count_with_nova_both_provided(mock_warning):
    """Test that _device_handle_instance_count warns when both instance_count and replicas are provided."""
    # Create mock recipe with replicas
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foobar3",
                "model_name_or_path": "foobar/foobar-3-8b",
                "replicas": 4,
            }
        }
    )

    # Test with instance_count in kwargs
    kwargs = {"instance_count": 2}
    PyTorch._device_handle_instance_count(kwargs, recipe)

    # Verify warning was logged
    mock_warning.assert_called_with(
        "Using instance_count argument to estimator to set number "
        "of nodes. Ignoring run -> replicas in recipe."
    )

    # Verify instance_count wasn't changed
    assert kwargs["instance_count"] == 2


def test_device_validate_and_get_type_with_nova():
    """Test that _device_validate_and_get_type works correctly with Nova recipes."""
    # Create mock recipe
    recipe = OmegaConf.create(
        {"run": {"model_type": "amazon.nova.foobar3", "model_name_or_path": "foobar/foobar-3-8b"}}
    )

    # Test with GPU instance type
    kwargs = {"instance_type": INSTANCE_TYPE_GPU}
    device_type = PyTorch._device_validate_and_get_type(kwargs, recipe)
    assert device_type == "gpu"

    # Test with CPU instance type
    kwargs = {"instance_type": INSTANCE_TYPE}
    device_type = PyTorch._device_validate_and_get_type(kwargs, recipe)
    assert device_type == "cpu"


def test_device_validate_and_get_type_no_instance_type():
    """Test that _device_validate_and_get_type raises an error when no instance_type is provided."""
    # Create mock recipe
    recipe = OmegaConf.create(
        {"run": {"model_type": "amazon.nova.foobar3", "model_name_or_path": "foobar/foobar-3-8b"}}
    )

    # Test with no instance_type
    kwargs = {}
    with pytest.raises(ValueError) as error:
        PyTorch._device_validate_and_get_type(kwargs, recipe)

    assert "Must pass instance type to estimator" in str(error)


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("time.time", return_value=1714500000)  # May 1, 2024
def test_upload_recipe_to_s3(mock_time, mock_recipe_load, sagemaker_session):
    """Test that _upload_recipe_to_s3 correctly uploads the recipe file to S3."""
    # Create a mock recipe that will be identified as a Nova recipe
    mock_recipe = OmegaConf.create(
        {"run": {"model_type": "amazon.nova.foobar3", "model_name_or_path": "foobar/foobar-3-8b"}}
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("nova_recipe", mock_recipe)

    # Setup
    pytorch = PyTorch(
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE_GPU,
        image_uri=IMAGE_URI,
        framework_version="1.13.1",
        py_version="py3",
        training_recipe="nova_recipe",
    )

    # Set Nova recipe attributes
    pytorch.is_nova_or_eval_recipe = True

    # Create a temporary file to use as the recipe file
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        # Test uploading the recipe file to S3
        s3_uri = pytorch._upload_recipe_to_s3(sagemaker_session, temp_file.name)

        # Verify the upload_data method was called with the correct parameters
        sagemaker_session.upload_data.assert_called_once()

        # Check that the S3 URI is returned correctly
        assert s3_uri == sagemaker_session.upload_data.return_value


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("tempfile.NamedTemporaryFile")
@patch("omegaconf.OmegaConf.save")
@patch("sagemaker.pytorch.estimator._try_resolve_recipe")
def test_recipe_resolve_and_save(
    mock_try_resolve, mock_save, mock_temp_file, mock_recipe_load, sagemaker_session
):
    """Test that _recipe_resolve_and_save correctly resolves an`d saves the recipe."""
    # Create a mock recipe that will be identified as a Nova recipe
    mock_recipe = OmegaConf.create(
        {"run": {"model_type": "amazon.nova.foobar3", "model_name_or_path": "foobar/foobar-3-8b"}}
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("nova_recipe", mock_recipe)

    # Setup
    pytorch = PyTorch(
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE_GPU,
        image_uri=IMAGE_URI,
        framework_version="1.13.1",
        py_version="py3",
        training_recipe="nova_recipe",
    )

    # Set Nova recipe attributes
    pytorch.is_nova_or_eval_recipe = True

    # Mock the temporary file
    mock_temp_file_instance = Mock()
    mock_temp_file_instance.name = "/tmp/nova-recipe_12345.yaml"
    mock_temp_file.return_value = mock_temp_file_instance

    # Create mock recipe
    recipe = OmegaConf.create(
        {"run": {"model_type": "amazon.nova.foobar3", "model_name_or_path": "foobar/foobar-3-8b"}}
    )

    # Mock the recipe resolution
    mock_try_resolve.side_effect = [recipe, None, None]

    # Call the _recipe_resolve_and_save method
    result = pytorch._recipe_resolve_and_save(recipe, "nova-recipe", ".")

    # Verify the recipe was resolved and saved
    mock_try_resolve.assert_called_with(recipe)
    mock_save.assert_called_with(config=recipe, f=mock_temp_file_instance.name)

    # Verify the result is the resolved recipe
    assert result == recipe


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("sagemaker.pytorch.estimator.Framework.fit")
def test_fit_with_nova_recipe_s3_upload(mock_framework_fit, mock_recipe_load, sagemaker_session):
    """Test that fit correctly uploads the recipe to S3 and adds it to the inputs."""
    # Create a mock recipe that will be identified as a Nova recipe
    mock_recipe = OmegaConf.create(
        {"run": {"model_type": "amazon.nova.foobar", "model_name_or_path": "foobar/foobar123"}}
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("nova_recipe", mock_recipe)

    # Create a PyTorch estimator with a Nova recipe
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        pytorch = PyTorch(
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
            training_recipe="nova_recipe",
        )

        # Set Nova recipe attributes
        pytorch.is_nova_or_eval_recipe = True
        pytorch.training_recipe_file = temp_file

        # Mock the _upload_recipe_to_s3 method
        with patch.object(pytorch, "_upload_recipe_to_s3") as mock_upload_recipe:
            mock_upload_recipe.return_value = "s3://mybucket/recipes/nova-recipe.yaml"

            # Call the fit method
            pytorch.fit()

            # Verify the upload_recipe_to_s3 method was called
            mock_upload_recipe.assert_called_once_with(sagemaker_session, temp_file.name)

            # Verify the fit method was called with the recipe channel
            call_args = mock_framework_fit.call_args[1]
            assert "inputs" in call_args
            assert "recipe" in call_args["inputs"]

            # Verify the hyperparameters were updated with the recipe path
            assert "sagemaker_recipe_local_path" in pytorch._hyperparameters


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("sagemaker.pytorch.estimator.PyTorch._upload_recipe_to_s3")
@patch("sagemaker.pytorch.estimator.Framework.fit")
def test_fit_with_nova_recipe_and_inputs(
    mock_framework_fit, mock_upload_recipe, mock_recipe_load, sagemaker_session
):
    """Test that fit correctly handles Nova recipes with additional inputs."""
    # Create a mock recipe that will be identified as a Nova recipe
    mock_recipe = OmegaConf.create(
        {"run": {"model_type": "amazon.nova.foobar3", "model_name_or_path": "foobar/foobar-3-8b"}}
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("nova_recipe", mock_recipe)
    mock_upload_recipe.return_value = "s3://mybucket/recipes/nova-recipe.yaml"

    # Create a PyTorch estimator with a Nova recipe
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        pytorch = PyTorch(
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
            training_recipe="nova_recipe",
        )

        # Set Nova recipe attributes
        pytorch.is_nova_or_eval_recipe = True
        pytorch.training_recipe_file = temp_file

        # Create training inputs
        train_input = TrainingInput(s3_data="s3://mybucket/train")
        val_input = TrainingInput(s3_data="s3://mybucket/validation")
        inputs = {"train": train_input, "validation": val_input}

        # Call the fit method with inputs
        pytorch.fit(inputs=inputs)

        # Verify the fit method was called with both the recipe channel and the provided inputs
        call_args = mock_framework_fit.call_args[1]
        assert "inputs" in call_args
        assert "recipe" in call_args["inputs"]
        assert "train" in call_args["inputs"]
        assert "validation" in call_args["inputs"]

        # Verify the hyperparameters were updated with the recipe path
        assert "sagemaker_recipe_local_path" in pytorch._hyperparameters


def test_device_get_distribution():
    """Test that _device_get_distribution returns the correct distribution configuration."""
    # Test with GPU device type
    gpu_distribution = _device_get_distribution("gpu")
    expected_gpu_distribution = {
        "torch_distributed": {"enabled": True},
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "placement_strategy": "cluster",
                },
            },
        },
    }
    assert gpu_distribution == expected_gpu_distribution

    # Test with Trainium device type
    trainium_distribution = _device_get_distribution("trainium")
    expected_trainium_distribution = {
        "torch_distributed": {"enabled": True},
    }
    assert trainium_distribution == expected_trainium_distribution

    # Test with CPU device type
    cpu_distribution = _device_get_distribution("cpu")
    assert cpu_distribution == {}


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("sagemaker.pytorch.estimator.PyTorch._upload_recipe_to_s3")
@patch("sagemaker.pytorch.estimator.Framework.fit")
def test_fit_with_nova_recipe(
    mock_framework_fit, mock_upload_recipe, mock_recipe_load, sagemaker_session
):
    """Test that fit correctly handles Nova recipes."""

    # Create a mock recipe that will be identified as a Nova recipe
    mock_recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foo-bar",
                "model_name_or_path": "foo-bar123",
            }
        }
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("nova_recipe", mock_recipe)

    # Create a PyTorch estimator with a Nova recipe
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        pytorch = PyTorch(
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
            training_recipe="nova_recipe",
        )

        # Set Nova recipe attributes
        pytorch.is_nova_or_eval_recipe = True
        pytorch.training_recipe_file = temp_file

        # Mock the upload_recipe_to_s3 method
        mock_upload_recipe.return_value = "s3://mybucket/recipes/nova-recipe.yaml"

        # Call the fit method
        pytorch.fit()

        # Verify the upload_recipe_to_s3 method was called
        mock_upload_recipe.assert_called_once_with(sagemaker_session, temp_file.name)

        # Verify the fit method was called with the recipe channel
        call_args = mock_framework_fit.call_args[1]
        assert "inputs" in call_args
        assert "recipe" in call_args["inputs"]

        # Verify the hyperparameters were updated with the recipe path
        assert "sagemaker_recipe_local_path" in pytorch._hyperparameters


def test_nova_encode_hyperparameters():
    """Test that _nova_encode_hyperparameters correctly preserves string values and encodes non-string values."""
    # Setup test hyperparameters
    hyperparameters = {
        "string_param": "string_value",
        "int_param": 42,
        "float_param": 3.14,
        "bool_param": True,
        "list_param": [1, 2, 3],
        "dict_param": {"key": "value"},
    }

    # Call the method
    encoded = EstimatorBase._nova_encode_hyperparameters(hyperparameters)

    # Verify string values are preserved
    assert encoded["string_param"] == "string_value"

    # Verify non-string values are JSON-encoded
    assert encoded["int_param"] == "42"
    assert encoded["float_param"] == "3.14"
    assert encoded["bool_param"] == "true"
    assert encoded["list_param"] == "[1, 2, 3]"
    assert encoded["dict_param"] == '{"key": "value"}'


def test_framework_set_hyperparameters_nova():
    """Test that Framework.set_hyperparameters uses _nova_encode_hyperparameters for Nova jobs."""
    # Setup
    framework = PyTorch(
        entry_point="dummy.py",
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        framework_version="1.13.1",
        py_version="py3",
        image_uri=IMAGE_URI,
    )

    framework.is_nova_job = True

    # Add hyperparameters
    framework.set_hyperparameters(string_param="string_value", int_param=42, bool_param=True)

    # Verify string values are preserved and non-string values are encoded
    assert framework._hyperparameters["string_param"] == "string_value"
    assert framework._hyperparameters["int_param"] == "42"
    assert framework._hyperparameters["bool_param"] == "true"


def test_framework_set_hyperparameters_non_nova():
    """Test that Framework.set_hyperparameters uses _json_encode_hyperparameters for non-Nova jobs."""
    # Setup
    framework = PyTorch(
        entry_point="dummy.py",
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        framework_version="1.13.1",
        py_version="py3",
        image_uri=IMAGE_URI,
    )
    framework.is_nova_or_eval_recipe = False

    # Add hyperparameters
    framework.set_hyperparameters(string_param="string_value", int_param=42, bool_param=True)

    # Verify all values are JSON-encoded
    assert framework._hyperparameters["string_param"] == '"string_value"'
    assert framework._hyperparameters["int_param"] == "42"
    assert framework._hyperparameters["bool_param"] == "true"


def test_framework_hyperparameters_nova():
    """Test that Framework.hyperparameters uses _nova_encode_hyperparameters for Nova jobs."""
    # Setup
    framework = PyTorch(
        entry_point="dummy.py",
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        framework_version="1.13.1",
        py_version="py3",
        image_uri=IMAGE_URI,
    )

    framework.is_nova_job = True

    # Add hyperparameters directly to _hyperparameters
    framework._hyperparameters = {
        "string_param": "string_value",
        "int_param": 42,
        "bool_param": True,
    }

    # Get hyperparameters
    hyperparams = framework.hyperparameters()

    # Verify string values are preserved and non-string values are encoded
    assert hyperparams["string_param"] == "string_value"
    assert hyperparams["int_param"] == "42"
    assert hyperparams["bool_param"] == "true"


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_resolve_and_save")
def test_setup_for_nova_recipe_with_evaluation_lambda(mock_resolve_save, sagemaker_session):
    """Test that _setup_for_nova_recipe correctly handles evaluation lambda configuration."""
    # Create a mock recipe with evaluation and processor config
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foobar3",
                "model_name_or_path": "foobar/foobar-3-8b",
                "replicas": 1,
            },
            "evaluation": {"task:": "gen_qa", "strategy": "gen_qa", "metric": "all"},
            "processor": {
                "lambda_arn": "arn:aws:lambda:us-west-2:123456789012:function:eval-function"
            },
        }
    )

    with patch(
        "sagemaker.pytorch.estimator.PyTorch._recipe_load", return_value=("nova_recipe", recipe)
    ):
        mock_resolve_save.return_value = recipe

        pytorch = PyTorch(
            training_recipe="nova_recipe",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
        )

        # Check that the Nova recipe was correctly identified
        assert pytorch.is_nova_or_eval_recipe is True

        # Verify that eval_lambda_arn hyperparameter was set correctly
        assert (
            pytorch._hyperparameters.get("eval_lambda_arn")
            == "arn:aws:lambda:us-west-2:123456789012:function:eval-function"
        )


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_resolve_and_save")
def test_setup_for_nova_recipe_with_distillation(mock_resolve_save, sagemaker_session):
    """Test that _setup_for_nova_recipe correctly handles distillation configurations."""
    # Create a mock recipe with distillation config
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foobar3",
                "model_name_or_path": "foobar/foobar-3-8b",
                "replicas": 4,
            },
            "training_config": {
                "distillation_data": "s3://mybucket/distillation-data",
                "kms_key": "alias/my-kms-key",
            },
        }
    )

    # Setup the expected return value
    expected_args = {
        "hyperparameters": {
            "base_model": "foobar/foobar-3-8b",
            "distillation_data": "s3://mybucket/distillation-data",
            "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
            "kms_key": "alias/my-kms-key",
        },
        "entry_point": None,
        "source_dir": None,
        "distribution": {},
        "default_image_uri": IMAGE_URI,
    }

    with patch(
        "sagemaker.pytorch.estimator.PyTorch._setup_for_nova_recipe", return_value=expected_args
    ) as mock_nova_setup:
        with patch(
            "sagemaker.pytorch.estimator.PyTorch._recipe_load", return_value=("nova_recipe", recipe)
        ):
            mock_resolve_save.return_value = recipe

            pytorch = PyTorch(
                training_recipe="nova_recipe",
                role="arn:aws:iam::123456789012:role/SageMakerRole",
                sagemaker_session=sagemaker_session,
                instance_count=INSTANCE_COUNT,
                instance_type=INSTANCE_TYPE_GPU,
                image_uri=IMAGE_URI,
                framework_version="1.13.1",
                py_version="py3",
            )

            # Check that the Nova recipe was correctly identified
            assert pytorch.is_nova_or_eval_recipe is True

            # Verify _setup_for_nova_recipe was called
            mock_nova_setup.assert_called_once()

            # Verify that hyperparameters were set correctly for distillation
            assert (
                pytorch._hyperparameters.get("distillation_data")
                == "s3://mybucket/distillation-data"
            )
            assert pytorch._hyperparameters.get("kms_key") == "alias/my-kms-key"
            assert (
                pytorch._hyperparameters.get("role_arn")
                == "arn:aws:iam::123456789012:role/SageMakerRole"
            )


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_resolve_and_save")
def test_setup_for_nova_recipe_sets_model_type(mock_resolve_save, sagemaker_session):
    """Test that _setup_for_nova_recipe correctly sets model_type hyperparameter."""
    # Create a mock nova recipe with model_type
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.llama-2-7b",
                "model_name_or_path": "llama/llama-2-7b",
                "replicas": 1,
            }
        }
    )

    with patch(
        "sagemaker.pytorch.estimator.PyTorch._recipe_load", return_value=("nova_recipe", recipe)
    ):
        mock_resolve_save.return_value = recipe

        pytorch = PyTorch(
            training_recipe="nova_recipe",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
        )

        # Check that the Nova recipe was correctly identified
        assert pytorch.is_nova_or_eval_recipe is True

        # Verify that model_type hyperparameter was set correctly
        assert pytorch._hyperparameters.get("model_type") == "amazon.nova.llama-2-7b"


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_resolve_and_save")
def test_setup_for_nova_recipe_with_reward_lambda(mock_resolve_save, sagemaker_session):
    """Test that _setup_for_nova_recipe correctly handles reward lambda configuration."""
    # Create a mock recipe with reward lambda config
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foobar3",
                "model_name_or_path": "foobar/foobar-3-8b",
                "reward_lambda_arn": "arn:aws:lambda:us-west-2:123456789012:function:reward-function",
                "replicas": 1,
            },
        }
    )

    with patch(
        "sagemaker.pytorch.estimator.PyTorch._recipe_load", return_value=("nova_recipe", recipe)
    ):
        mock_resolve_save.return_value = recipe

        pytorch = PyTorch(
            training_recipe="nova_recipe",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
        )

        # Check that the Nova recipe was correctly identified
        assert pytorch.is_nova_or_eval_recipe is True

        # Verify that reward_lambda_arn hyperparameter was set correctly
        assert (
            pytorch._hyperparameters.get("reward_lambda_arn")
            == "arn:aws:lambda:us-west-2:123456789012:function:reward-function"
        )


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_resolve_and_save")
def test_setup_for_nova_recipe_without_reward_lambda(mock_resolve_save, sagemaker_session):
    """Test that _setup_for_nova_recipe does not set reward_lambda_arn when not present."""
    # Create a mock recipe without reward lambda config
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.foobar3",
                "model_name_or_path": "foobar/foobar-3-8b",
                "replicas": 1,
            },
        }
    )

    with patch(
        "sagemaker.pytorch.estimator.PyTorch._recipe_load", return_value=("nova_recipe", recipe)
    ):
        mock_resolve_save.return_value = recipe

        pytorch = PyTorch(
            training_recipe="nova_recipe",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
        )

        # Check that the Nova recipe was correctly identified
        assert pytorch.is_nova_or_eval_recipe is True

        # Verify that reward_lambda_arn hyperparameter was not set
        assert "reward_lambda_arn" not in pytorch._hyperparameters
