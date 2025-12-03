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

import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf
from sagemaker.core.modules.train.sm_recipes.utils import (
    _try_resolve_recipe,
    _determine_device_type,
    _load_recipes_cfg,
    _load_base_recipe,
    _register_custom_resolvers,
    _get_trainining_recipe_gpu_model_name_and_script,
    _configure_gpu_args,
    _configure_trainium_args,
    _get_args_from_recipe,
)
from sagemaker.core.modules.configs import Compute


class TestSMRecipesUtils:
    """Test cases for SM recipes utility functions"""

    def test_try_resolve_recipe_success(self):
        """Test _try_resolve_recipe with resolvable recipe"""
        recipe = OmegaConf.create({"value": 10, "doubled": "${value}"})

        result = _try_resolve_recipe(recipe)

        assert result is not None
        assert result["doubled"] == 10

    def test_try_resolve_recipe_with_key(self):
        """Test _try_resolve_recipe with key parameter"""
        recipe = 10

        result = _try_resolve_recipe(recipe, key="test")

        assert result is not None
        assert result == 10

    def test_try_resolve_recipe_unresolvable(self):
        """Test _try_resolve_recipe with unresolvable recipe"""
        recipe = OmegaConf.create({"value": "${missing_var}"})

        result = _try_resolve_recipe(recipe)

        assert result is None

    def test_determine_device_type_gpu_p_instance(self):
        """Test _determine_device_type with P instance (GPU)"""
        result = _determine_device_type("ml.p3.2xlarge")
        assert result == "gpu"

    def test_determine_device_type_gpu_g_instance(self):
        """Test _determine_device_type with G instance (GPU)"""
        result = _determine_device_type("ml.g4dn.xlarge")
        assert result == "gpu"

    def test_determine_device_type_trainium(self):
        """Test _determine_device_type with Trainium instance"""
        result = _determine_device_type("ml.trn1.2xlarge")
        assert result == "trainium"

    def test_determine_device_type_cpu(self):
        """Test _determine_device_type with CPU instance"""
        result = _determine_device_type("ml.m5.xlarge")
        assert result == "cpu"

    @patch("sagemaker.core.modules.train.sm_recipes.utils.open")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.json.load")
    def test_load_recipes_cfg(self, mock_json_load, mock_open):
        """Test _load_recipes_cfg loads configuration"""
        mock_json_load.return_value = {"launcher_repo": "test_repo", "adapter_repo": "test_adapter"}

        result = _load_recipes_cfg()

        assert isinstance(result, dict)
        assert "launcher_repo" in result or "adapter_repo" in result or "neuron_dist_repo" in result

    @patch("sagemaker.core.modules.train.sm_recipes.utils.os.path.isfile")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.shutil.copy")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.OmegaConf.load")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.OmegaConf.merge")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.os.unlink")
    def test_load_base_recipe_from_file(
        self, mock_unlink, mock_merge, mock_load, mock_copy, mock_isfile
    ):
        """Test _load_base_recipe from local file"""
        mock_isfile.return_value = True
        mock_recipe = OmegaConf.create({"model": {"model_type": "llama_v3"}})
        mock_load.return_value = mock_recipe
        mock_merge.return_value = mock_recipe

        result = _load_base_recipe("recipe.yaml")

        assert result is not None
        mock_copy.assert_called_once()

    @patch("sagemaker.core.modules.train.sm_recipes.utils.os.path.isfile")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.urlretrieve")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.OmegaConf.load")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.OmegaConf.merge")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.os.unlink")
    def test_load_base_recipe_from_url(
        self, mock_unlink, mock_merge, mock_load, mock_urlretrieve, mock_isfile
    ):
        """Test _load_base_recipe from URL"""
        mock_isfile.return_value = False
        mock_recipe = OmegaConf.create({"model": {"model_type": "llama_v3"}})
        mock_load.return_value = mock_recipe
        mock_merge.return_value = mock_recipe

        result = _load_base_recipe("https://example.com/recipe.yaml")

        assert result is not None
        mock_urlretrieve.assert_called_once()

    @patch("sagemaker.core.modules.train.sm_recipes.utils.os.path.isfile")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.urlretrieve")
    def test_load_base_recipe_url_error(self, mock_urlretrieve, mock_isfile):
        """Test _load_base_recipe raises error on URL fetch failure"""
        mock_isfile.return_value = False
        mock_urlretrieve.side_effect = Exception("Network error")

        with pytest.raises(ValueError, match="Could not fetch the provided recipe"):
            _load_base_recipe("https://example.com/recipe.yaml")

    def test_register_custom_resolvers(self):
        """Test _register_custom_resolvers registers OmegaConf resolvers"""
        _register_custom_resolvers()

        # Test multiply resolver
        recipe = OmegaConf.create({"a": 5, "b": "${multiply:${a},2}"})
        OmegaConf.resolve(recipe)
        assert recipe["b"] == 10

        # Test divide_ceil resolver
        recipe = OmegaConf.create({"a": 10, "b": "${divide_ceil:${a},3}"})
        OmegaConf.resolve(recipe)
        assert recipe["b"] == 4

        # Test divide_floor resolver
        recipe = OmegaConf.create({"a": 10, "b": "${divide_floor:${a},3}"})
        OmegaConf.resolve(recipe)
        assert recipe["b"] == 3

        # Test add resolver
        recipe = OmegaConf.create({"a": "${add:1,2,3}"})
        OmegaConf.resolve(recipe)
        assert recipe["a"] == 6

    def test_get_trainining_recipe_gpu_model_name_and_script_llama(self):
        """Test _get_trainining_recipe_gpu_model_name_and_script for Llama"""
        model_name, script = _get_trainining_recipe_gpu_model_name_and_script("llama_v3_8b")

        assert model_name == "llama"
        assert script == "llama_pretrain.py"

    def test_get_trainining_recipe_gpu_model_name_and_script_mistral(self):
        """Test _get_trainining_recipe_gpu_model_name_and_script for Mistral"""
        model_name, script = _get_trainining_recipe_gpu_model_name_and_script("mistral_7b")

        assert model_name == "mistral"
        assert script == "mistral_pretrain.py"

    def test_get_trainining_recipe_gpu_model_name_and_script_mixtral(self):
        """Test _get_trainining_recipe_gpu_model_name_and_script for Mixtral"""
        model_name, script = _get_trainining_recipe_gpu_model_name_and_script("mixtral_8x7b")

        assert model_name == "mixtral"
        assert script == "mixtral_pretrain.py"

    def test_get_trainining_recipe_gpu_model_name_and_script_deepseek(self):
        """Test _get_trainining_recipe_gpu_model_name_and_script for DeepSeek"""
        model_name, script = _get_trainining_recipe_gpu_model_name_and_script("deepseek_v2")

        assert model_name == "deepseek"
        assert script == "deepseek_pretrain.py"

    def test_get_trainining_recipe_gpu_model_name_and_script_unsupported(self):
        """Test _get_trainining_recipe_gpu_model_name_and_script with unsupported model"""
        with pytest.raises(ValueError, match="Model type .* not supported"):
            _get_trainining_recipe_gpu_model_name_and_script("unsupported_model")

    @patch("sagemaker.core.modules.train.sm_recipes.utils._run_clone_command_silent")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.retrieve")
    def test_configure_gpu_args(self, mock_retrieve, mock_clone):
        """Test _configure_gpu_args"""
        training_recipes_cfg = {
            "adapter_repo": "https://github.com/test/adapter",
            "gpu_image": {"framework": "pytorch", "version": "2.0", "additional_args": {}},
        }

        recipe = OmegaConf.create({"model": {"model_type": "llama_v3"}})

        recipe_train_dir = tempfile.TemporaryDirectory()
        mock_retrieve.return_value = "test-image:latest"

        result = _configure_gpu_args(training_recipes_cfg, "us-west-2", recipe, recipe_train_dir)

        assert "source_code" in result
        assert "training_image" in result
        assert "distributed" in result
        assert result["training_image"] == "test-image:latest"

    @patch("sagemaker.core.modules.train.sm_recipes.utils._run_clone_command_silent")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.retrieve")
    def test_configure_gpu_args_string_image(self, mock_retrieve, mock_clone):
        """Test _configure_gpu_args with string image config"""
        training_recipes_cfg = {
            "adapter_repo": "https://github.com/test/adapter",
            "gpu_image": "custom-image:latest",
        }

        recipe = OmegaConf.create({"model": {"model_type": "mistral"}})

        recipe_train_dir = tempfile.TemporaryDirectory()

        result = _configure_gpu_args(training_recipes_cfg, "us-west-2", recipe, recipe_train_dir)

        assert result["training_image"] == "custom-image:latest"

    @patch("sagemaker.core.modules.train.sm_recipes.utils._run_clone_command_silent")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.retrieve")
    def test_configure_gpu_args_missing_model(self, mock_retrieve, mock_clone):
        """Test _configure_gpu_args raises error when model field is missing"""
        training_recipes_cfg = {
            "adapter_repo": "https://github.com/test/adapter",
            "gpu_image": "test-image:latest",
        }

        recipe = OmegaConf.create({})
        recipe_train_dir = tempfile.TemporaryDirectory()

        with pytest.raises(ValueError, match="does not contain required field model"):
            _configure_gpu_args(training_recipes_cfg, "us-west-2", recipe, recipe_train_dir)

    @patch("sagemaker.core.modules.train.sm_recipes.utils._run_clone_command_silent")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.retrieve")
    def test_configure_trainium_args(self, mock_retrieve, mock_clone):
        """Test _configure_trainium_args"""
        training_recipes_cfg = {
            "neuron_dist_repo": "https://github.com/test/neuron",
            "neuron_image": {"framework": "pytorch", "version": "1.13", "additional_args": {}},
        }

        recipe_train_dir = tempfile.TemporaryDirectory()
        mock_retrieve.return_value = "neuron-image:latest"

        result = _configure_trainium_args(training_recipes_cfg, "us-west-2", recipe_train_dir)

        assert "source_code" in result
        assert "training_image" in result
        assert "distributed" in result
        assert result["training_image"] == "neuron-image:latest"

    @patch("sagemaker.core.modules.train.sm_recipes.utils._load_recipes_cfg")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._load_base_recipe")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._configure_gpu_args")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._register_custom_resolvers")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._try_resolve_recipe")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.OmegaConf.save")
    def test_get_args_from_recipe_gpu(
        self,
        mock_save,
        mock_resolve,
        mock_register,
        mock_configure_gpu,
        mock_load_recipe,
        mock_load_cfg,
    ):
        """Test _get_args_from_recipe for GPU instance"""
        compute = Compute(instance_type="ml.p3.2xlarge", instance_count=2)

        mock_load_cfg.return_value = {}
        mock_recipe = OmegaConf.create(
            {"trainer": {"num_nodes": 1}, "model": {"model_type": "llama_v3"}}
        )
        mock_load_recipe.return_value = mock_recipe
        mock_resolve.return_value = mock_recipe

        mock_configure_gpu.return_value = {
            "source_code": Mock(source_dir="/tmp/source"),
            "training_image": "test-image:latest",
            "distributed": Mock(),
        }

        result, temp_dir = _get_args_from_recipe(
            training_recipe="llama_recipe",
            compute=compute,
            region_name="us-west-2",
            recipe_overrides=None,
            requirements=None,
        )

        assert "source_code" in result
        assert "training_image" in result
        assert "compute" in result
        assert "hyperparameters" in result
        assert result["compute"].instance_count == 2

    @patch("sagemaker.core.modules.train.sm_recipes.utils._load_recipes_cfg")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._load_base_recipe")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._configure_trainium_args")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._register_custom_resolvers")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._try_resolve_recipe")
    @patch("sagemaker.core.modules.train.sm_recipes.utils.OmegaConf.save")
    def test_get_args_from_recipe_trainium(
        self,
        mock_save,
        mock_resolve,
        mock_register,
        mock_configure_trainium,
        mock_load_recipe,
        mock_load_cfg,
    ):
        """Test _get_args_from_recipe for Trainium instance"""
        compute = Compute(instance_type="ml.trn1.2xlarge", instance_count=1)

        mock_load_cfg.return_value = {}
        mock_recipe = OmegaConf.create({"trainer": {"num_nodes": 1}})
        mock_load_recipe.return_value = mock_recipe
        mock_resolve.return_value = mock_recipe

        mock_configure_trainium.return_value = {
            "source_code": Mock(source_dir="/tmp/source"),
            "training_image": "neuron-image:latest",
            "distributed": Mock(),
        }

        result, temp_dir = _get_args_from_recipe(
            training_recipe="neuron_recipe",
            compute=compute,
            region_name="us-west-2",
            recipe_overrides=None,
            requirements=None,
        )

        assert "source_code" in result
        assert "training_image" in result

    def test_get_args_from_recipe_no_instance_type(self):
        """Test _get_args_from_recipe raises error without instance_type"""
        compute = Compute(instance_count=1)

        with pytest.raises(ValueError, match="Must set `instance_type`"):
            _get_args_from_recipe(
                training_recipe="test_recipe",
                compute=compute,
                region_name="us-west-2",
                recipe_overrides=None,
                requirements=None,
            )

    @patch("sagemaker.core.modules.train.sm_recipes.utils._load_recipes_cfg")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._load_base_recipe")
    def test_get_args_from_recipe_missing_trainer(self, mock_load_recipe, mock_load_cfg):
        """Test _get_args_from_recipe raises error when trainer field is missing"""
        compute = Compute(instance_type="ml.p3.2xlarge", instance_count=1)

        mock_load_cfg.return_value = {}
        mock_recipe = OmegaConf.create({})
        mock_load_recipe.return_value = mock_recipe

        with pytest.raises(ValueError, match="does not contain required field trainer"):
            _get_args_from_recipe(
                training_recipe="test_recipe",
                compute=compute,
                region_name="us-west-2",
                recipe_overrides=None,
                requirements=None,
            )

    @patch("sagemaker.core.modules.train.sm_recipes.utils._load_recipes_cfg")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._load_base_recipe")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._configure_gpu_args")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._register_custom_resolvers")
    @patch("sagemaker.core.modules.train.sm_recipes.utils._try_resolve_recipe")
    def test_get_args_from_recipe_unresolvable(
        self, mock_resolve, mock_register, mock_configure_gpu, mock_load_recipe, mock_load_cfg
    ):
        """Test _get_args_from_recipe raises error when recipe cannot be resolved"""
        compute = Compute(instance_type="ml.p3.2xlarge", instance_count=1)

        mock_load_cfg.return_value = {}
        mock_recipe = OmegaConf.create(
            {"trainer": {"num_nodes": 1}, "model": {"model_type": "llama_v3"}}
        )
        mock_load_recipe.return_value = mock_recipe
        mock_resolve.return_value = None  # Cannot resolve

        mock_configure_gpu.return_value = {
            "source_code": Mock(source_dir="/tmp/source"),
            "training_image": "test-image:latest",
            "distributed": Mock(),
        }

        with pytest.raises(RuntimeError, match="Could not resolve provided recipe"):
            _get_args_from_recipe(
                training_recipe="test_recipe",
                compute=compute,
                region_name="us-west-2",
                recipe_overrides=None,
                requirements=None,
            )

    def test_get_args_from_recipe_cpu_not_supported(self):
        """Test _get_args_from_recipe raises error for CPU instances"""
        compute = Compute(instance_type="ml.m5.xlarge", instance_count=1)

        with patch("sagemaker.core.modules.train.sm_recipes.utils._load_recipes_cfg"):
            with patch(
                "sagemaker.core.modules.train.sm_recipes.utils._load_base_recipe"
            ) as mock_load:
                mock_load.return_value = OmegaConf.create({"trainer": {"num_nodes": 1}})

                with pytest.raises(ValueError, match="Devices of type cpu are not supported"):
                    _get_args_from_recipe(
                        training_recipe="test_recipe",
                        compute=compute,
                        region_name="us-west-2",
                        recipe_overrides=None,
                        requirements=None,
                    )
