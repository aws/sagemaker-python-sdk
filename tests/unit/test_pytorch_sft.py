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

from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.estimator import _is_llmft_recipe
from sagemaker.inputs import TrainingInput
from sagemaker.session_settings import SessionSettings

# Constants for testing
ROLE = "Dummy"
REGION = "us-west-2"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
INSTANCE_TYPE_GPU = "ml.p4d.24xlarge"
INSTANCE_TYPE_TRN = "ml.trn1.32xlarge"
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
    session.upload_data = Mock(return_value="s3://mybucket/recipes/llmft-recipe.yaml")
    session.sagemaker_config = {}
    return session


def test_is_llmft_recipe():
    """Test that _is_llmft_recipe correctly identifies LLMFT recipes."""
    # Valid LLMFT recipe
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )
    assert _is_llmft_recipe(recipe) is True

    # Not an LLMFT recipe - missing run section
    recipe = OmegaConf.create(
        {
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )
    assert _is_llmft_recipe(recipe) is False

    # Not an LLMFT recipe - wrong model_type
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "dpo",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )
    assert _is_llmft_recipe(recipe) is False

    # Not an LLMFT recipe - missing training_config section
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
        }
    )
    assert _is_llmft_recipe(recipe) is False


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_resolve_and_save")
def test_setup_for_llmft_recipe_basic(mock_resolve_save, sagemaker_session):
    """Test that _setup_for_llmft_recipe correctly sets up hyperparameters for LLMFT recipes."""
    # Create a mock LLMFT recipe
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 2,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                    "peft_config": {
                        "peft_type": "lora",
                        "target_modules": "all-linear",
                        "r": 16,
                        "lora_alpha": 32,
                    },
                },
                "training_args": {
                    "trainer_type": "sft",
                    "learning_rate": 0.0001,
                    "max_epochs": 1,
                },
            },
        }
    )

    # Setup the expected return value
    expected_args = {
        "hyperparameters": {},
        "entry_point": None,
        "source_dir": None,
        "distribution": {},
        "default_image_uri": IMAGE_URI,
    }

    # Mock the _setup_for_llmft_recipe method
    with patch(
        "sagemaker.pytorch.estimator.PyTorch._setup_for_llmft_recipe", return_value=expected_args
    ) as mock_llmft_setup:
        # Create the PyTorch estimator with mocked _recipe_load
        with patch(
            "sagemaker.pytorch.estimator.PyTorch._recipe_load",
            return_value=("llmft_recipe", recipe),
        ):
            # Mock _recipe_resolve_and_save to return our recipe
            mock_resolve_save.return_value = recipe

            pytorch = PyTorch(
                training_recipe="llmft_recipe",
                role=ROLE,
                sagemaker_session=sagemaker_session,
                instance_count=INSTANCE_COUNT,
                instance_type=INSTANCE_TYPE_GPU,
                image_uri=IMAGE_URI,
                framework_version="1.13.1",
                py_version="py3",
            )

            # Check that the LLMFT recipe was correctly identified
            assert pytorch.is_llmft_recipe is True

            # Verify _setup_for_llmft_recipe was called
            mock_llmft_setup.assert_called_once()
            call_args = mock_llmft_setup.call_args
            assert len(call_args[0]) >= 2  # Check that at least recipe and recipe_name were passed
            assert call_args[0][0] == recipe  # first arg should be recipe
            assert call_args[0][1] == "llmft_recipe"  # second arg should be recipe_name


def test_device_handle_instance_count_with_llmft_num_nodes():
    """Test that _device_handle_instance_count correctly gets instance_count from LLMFT recipe num_nodes."""
    # Create mock LLMFT recipe with num_nodes
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 4,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Test with no instance_count in kwargs
    kwargs = {}
    PyTorch._device_handle_instance_count(kwargs, recipe)
    assert kwargs["instance_count"] == 4


def test_device_handle_instance_count_with_llmft_no_num_nodes():
    """Test that _device_handle_instance_count raises an error when no instance_count or num_nodes are provided."""
    # Create mock LLMFT recipe without num_nodes
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Test with no instance_count in kwargs
    kwargs = {}
    with pytest.raises(ValueError) as error:
        PyTorch._device_handle_instance_count(kwargs, recipe)

    assert "Must set either instance_count argument for estimator or" in str(error)


@patch("sagemaker.pytorch.estimator.logger.warning")
def test_device_handle_instance_count_with_llmft_both_provided(mock_warning):
    """Test that _device_handle_instance_count warns when both instance_count and num_nodes are provided."""
    # Create mock LLMFT recipe with num_nodes
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 4,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Test with instance_count in kwargs
    kwargs = {"instance_count": 2}
    PyTorch._device_handle_instance_count(kwargs, recipe)

    # Verify warning was logged
    mock_warning.assert_called_with(
        "Using instance_count argument to estimator to set number "
        "of nodes. Ignoring trainer -> num_nodes in recipe."
    )

    # Verify instance_count wasn't changed
    assert kwargs["instance_count"] == 2


def test_device_validate_and_get_type_with_llmft():
    """Test that _device_validate_and_get_type works correctly with LLMFT recipes."""
    # Create mock LLMFT recipe
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Test with GPU instance type
    kwargs = {"instance_type": INSTANCE_TYPE_GPU}
    device_type = PyTorch._device_validate_and_get_type(kwargs, recipe)
    assert device_type == "gpu"

    # Test with CPU instance type
    kwargs = {"instance_type": INSTANCE_TYPE}
    device_type = PyTorch._device_validate_and_get_type(kwargs, recipe)
    assert device_type == "cpu"

    # Test with TRN instance type
    kwargs = {"instance_type": INSTANCE_TYPE_TRN}
    device_type = PyTorch._device_validate_and_get_type(kwargs, recipe)
    assert device_type == "trainium"


def test_device_validate_and_get_type_no_instance_type_llmft():
    """Test that _device_validate_and_get_type raises an error when no instance_type is provided for LLMFT."""
    # Create mock LLMFT recipe
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Test with no instance_type
    kwargs = {}
    with pytest.raises(ValueError) as error:
        PyTorch._device_validate_and_get_type(kwargs, recipe)

    assert "Must pass instance type to estimator" in str(error)


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("time.time", return_value=1714500000)  # May 1, 2024
def test_upload_recipe_to_s3_llmft(mock_time, mock_recipe_load, sagemaker_session):
    """Test that _upload_recipe_to_s3 correctly uploads the LLMFT recipe file to S3."""
    # Create a mock LLMFT recipe
    mock_recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("llmft_recipe", mock_recipe)

    # Setup
    pytorch = PyTorch(
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE_GPU,
        image_uri=IMAGE_URI,
        framework_version="1.13.1",
        py_version="py3",
        training_recipe="llmft_recipe",
    )

    # Set llmft recipe attributes
    pytorch.is_llmft_recipe = True

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
def test_recipe_resolve_and_save_llmft(
    mock_try_resolve, mock_save, mock_temp_file, mock_recipe_load, sagemaker_session
):
    """Test that _recipe_resolve_and_save correctly resolves and saves the llmft recipe."""
    # Create a mock llmft recipe
    mock_recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("llmft_recipe", mock_recipe)

    # Setup
    pytorch = PyTorch(
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE_GPU,
        image_uri=IMAGE_URI,
        framework_version="1.13.1",
        py_version="py3",
        training_recipe="llmft_recipe",
    )

    # Set llmft recipe attributes
    pytorch.is_llmft_recipe = True

    # Mock the temporary file
    mock_temp_file_instance = Mock()
    mock_temp_file_instance.name = "/tmp/llmft-recipe_12345.yaml"
    mock_temp_file.return_value = mock_temp_file_instance

    # Create mock recipe
    recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llmft",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Mock the recipe resolution
    mock_try_resolve.side_effect = [recipe, None, None]

    # Call the _recipe_resolve_and_save method
    result = pytorch._recipe_resolve_and_save(recipe, "llmft-recipe", ".")

    # Verify the recipe was resolved and saved
    mock_try_resolve.assert_called_with(recipe)
    mock_save.assert_called_with(config=recipe, f=mock_temp_file_instance.name)

    # Verify the result is the resolved recipe
    assert result == recipe


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("sagemaker.pytorch.estimator.Framework.fit")
def test_fit_with_llmft_recipe_s3_upload(mock_framework_fit, mock_recipe_load, sagemaker_session):
    """Test that fit correctly uploads the llmft recipe to S3 and adds it to the inputs."""
    # Create a mock llmft recipe
    mock_recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("llmft_recipe", mock_recipe)

    # Create a PyTorch estimator with an llmft recipe
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        pytorch = PyTorch(
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
            training_recipe="llmft_recipe",
        )

        # Set llmft recipe attributes
        pytorch.is_llmft_recipe = True
        pytorch.training_recipe_file = temp_file

        # Mock the _upload_recipe_to_s3 and _create_recipe_copy methods
        with (
            patch.object(pytorch, "_upload_recipe_to_s3") as mock_upload_recipe,
            patch.object(pytorch, "_create_recipe_copy") as mock_create_copy,
        ):
            mock_upload_recipe.return_value = "s3://mybucket/recipes/llmft-recipe.yaml"
            mock_create_copy.return_value = "s3://mybucket/recipes/recipe.yaml"

            # Call the fit method
            pytorch.fit()

            # Verify the upload_recipe_to_s3 method was called
            mock_upload_recipe.assert_called_once_with(sagemaker_session, temp_file.name)

            # Verify the create_recipe_copy method was called
            mock_create_copy.assert_called_once_with("s3://mybucket/recipes/llmft-recipe.yaml")

            # Verify the fit method was called with the recipe channel
            call_args = mock_framework_fit.call_args[1]
            assert "inputs" in call_args
            assert "recipe" in call_args["inputs"]

            # Verify the hyperparameters were updated with the recipe path
            assert "sagemaker_recipe_local_path" in pytorch._hyperparameters


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("sagemaker.pytorch.estimator.PyTorch._upload_recipe_to_s3")
@patch("sagemaker.pytorch.estimator.PyTorch._create_recipe_copy")
@patch("sagemaker.pytorch.estimator.Framework.fit")
def test_fit_with_llmft_recipe_and_inputs(
    mock_framework_fit, mock_create_copy, mock_upload_recipe, mock_recipe_load, sagemaker_session
):
    """Test that fit correctly handles llmft recipes with additional inputs."""
    # Create a mock llmft recipe
    mock_recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("llmft_recipe", mock_recipe)
    mock_upload_recipe.return_value = "s3://mybucket/recipes/llmft-recipe.yaml"
    mock_create_copy.return_value = "s3://mybucket/recipes/recipe.yaml"

    # Create a PyTorch estimator with an llmft recipe
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        pytorch = PyTorch(
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
            training_recipe="llmft_recipe",
        )

        # Set llmft recipe attributes
        pytorch.is_llmft_recipe = True
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


@patch("sagemaker.pytorch.estimator.PyTorch._recipe_load")
@patch("sagemaker.pytorch.estimator.PyTorch._upload_recipe_to_s3")
@patch("sagemaker.pytorch.estimator.PyTorch._create_recipe_copy")
@patch("sagemaker.pytorch.estimator.Framework.fit")
def test_fit_with_llmft_recipe(
    mock_framework_fit, mock_create_copy, mock_upload_recipe, mock_recipe_load, sagemaker_session
):
    """Test that fit correctly handles llmft recipes."""

    # Create a mock llmft recipe
    mock_recipe = OmegaConf.create(
        {
            "run": {
                "model_type": "llm_finetuning_aws",
                "name": "foo-bar123",
            },
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
            },
            "training_config": {
                "model_config": {
                    "model_name_or_path": "foo-bar/foo-bar123",
                }
            },
        }
    )

    # Set up the mock to return a recipe name and the mock recipe
    mock_recipe_load.return_value = ("llmft_recipe", mock_recipe)

    # Create a PyTorch estimator with an llmft recipe
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        pytorch = PyTorch(
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE_GPU,
            image_uri=IMAGE_URI,
            framework_version="1.13.1",
            py_version="py3",
            training_recipe="llmft_recipe",
        )

        # Set llmft recipe attributes
        pytorch.is_llmft_recipe = True
        pytorch.training_recipe_file = temp_file

        # Mock the upload_recipe_to_s3 and create_recipe_copy methods
        mock_upload_recipe.return_value = "s3://mybucket/recipes/llmft-recipe.yaml"
        mock_create_copy.return_value = "s3://mybucket/recipes/recipe.yaml"

        # Call the fit method
        pytorch.fit()

        # Verify the upload_recipe_to_s3 method was called
        mock_upload_recipe.assert_called_once_with(sagemaker_session, temp_file.name)

        # Verify the create_recipe_copy method was called
        mock_create_copy.assert_called_once_with("s3://mybucket/recipes/llmft-recipe.yaml")

        # Verify the fit method was called with the recipe channel
        call_args = mock_framework_fit.call_args[1]
        assert "inputs" in call_args
        assert "recipe" in call_args["inputs"]

        # Verify the hyperparameters were updated with the recipe path
        assert "sagemaker_recipe_local_path" in pytorch._hyperparameters
