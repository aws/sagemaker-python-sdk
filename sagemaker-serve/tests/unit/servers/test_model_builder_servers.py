"""Unit tests for ModelBuilderServers class."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import unittest

# Prevent JumpStart from loading region config during import
os.environ["SAGEMAKER_INTERNAL_SKIP_REGION_CONFIG"] = "1"

from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder_servers import _ModelBuilderServers


class MockModelBuilderServers(_ModelBuilderServers):
    """Mock class that inherits _ModelBuilderServers behavior."""

    def __init__(self):
        self.model_server = ModelServer.TORCHSERVE
        self.model = None
        self.model_metadata = {}
        self.inference_spec = None
        self.mode = Mode.SAGEMAKER_ENDPOINT
        self.model_path = tempfile.mkdtemp()
        self.shared_libs = []
        self.dependencies = {}
        self.sagemaker_session = Mock()
        self.image_uri = "test-image-uri"
        self.secret_key = ""
        self.env_vars = {}
        self.schema_builder = Mock()
        self.schema_builder.sample_input = {"inputs": "test"}
        self.hf_model_config = {}
        self.s3_upload_path = None
        self.s3_model_data_url = None
        self.instance_type = "ml.m5.large"
        self._user_provided_instance_type = False
        self._optimizing = False
        self.model_data_download_timeout = None
        self.role_arn = "arn:aws:iam::123456789012:role/test"
        self.region = "us-east-1"
        self.model_version = None
        self.framework = None
        self.framework_version = None
        self._is_mlflow_model = False
        self.config_name = None
        self._enable_network_isolation = False

    def _deploy_local_endpoint(self, **kwargs):
        return Mock()

    def _deploy_core_endpoint(self, *args, **kwargs):
        return Mock()

    def _save_model_inference_spec(self):
        pass

    def _is_jumpstart_model_id(self):
        return False

    def _auto_detect_image_uri(self):
        pass

    def _prepare_for_mode(self, should_upload_artifacts=False):
        return ("s3://bucket/model.tar.gz", None)

    def _create_model(self):
        return Mock()

    def _validate_tgi_serving_sample_data(self):
        pass

    def _validate_djl_serving_sample_data(self):
        pass

    def _validate_for_triton(self):
        pass

    def _auto_detect_image_for_triton(self):
        pass

    def _save_inference_spec(self):
        pass

    def _prepare_for_triton(self):
        pass

    def get_huggingface_model_metadata(self, model_id, token=None):
        return {}

    def _normalize_framework_to_enum(self, framework):
        return framework

    def _get_processing_unit(self):
        return "cpu"

    def _get_smd_image_uri(self, processing_unit):
        return "smd-image-uri"

    def _create_conda_env(self):
        pass


class TestBuildForModelServer(unittest.TestCase):
    """Test _build_for_model_server method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()

    def test_unsupported_model_server(self):
        """Test error for unsupported model server."""
        self.builder.model_server = "INVALID_SERVER"
        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_model_server()
        self.assertIn("not supported", str(ctx.exception))

    def test_missing_required_parameters(self):
        """Test error when model, MLflow path, and inference_spec are all missing."""
        self.builder.model = None
        self.builder.model_metadata = {}
        self.builder.inference_spec = None
        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_model_server()
        self.assertIn("Missing required parameter", str(ctx.exception))

    @patch.object(MockModelBuilderServers, "_build_for_torchserve")
    def test_route_to_torchserve(self, mock_build):
        """Test routing to TorchServe builder."""
        self.builder.model_server = ModelServer.TORCHSERVE
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_triton")
    def test_route_to_triton(self, mock_build):
        """Test routing to Triton builder."""
        self.builder.model_server = ModelServer.TRITON
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_tensorflow_serving")
    def test_route_to_tensorflow_serving(self, mock_build):
        """Test routing to TensorFlow Serving builder."""
        self.builder.model_server = ModelServer.TENSORFLOW_SERVING
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_djl")
    def test_route_to_djl(self, mock_build):
        """Test routing to DJL builder."""
        self.builder.model_server = ModelServer.DJL_SERVING
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_tei")
    def test_route_to_tei(self, mock_build):
        """Test routing to TEI builder."""
        self.builder.model_server = ModelServer.TEI
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_tgi")
    def test_route_to_tgi(self, mock_build):
        """Test routing to TGI builder."""
        self.builder.model_server = ModelServer.TGI
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_transformers")
    def test_route_to_mms(self, mock_build):
        """Test routing to MMS builder."""
        self.builder.model_server = ModelServer.MMS
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_smd")
    def test_route_to_smd(self, mock_build):
        """Test routing to SMD builder."""
        self.builder.model_server = ModelServer.SMD
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()


class TestBuildForTorchServe(unittest.TestCase):
    """Test _build_for_torchserve method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TORCHSERVE

    @patch.object(MockModelBuilderServers, "_save_model_inference_spec")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model_id(
        self, mock_create, mock_prepare, mock_detect, mock_js, mock_save
    ):
        """Test building with HuggingFace model ID."""
        mock_js.return_value = False
        mock_create.return_value = Mock()
        self.builder.mode = Mode.IN_PROCESS
        self.builder.model = "bert-base-uncased"
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "test-token"}

        result = self.builder._build_for_torchserve()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "bert-base-uncased")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "test-token")
        self.assertIsNone(self.builder.s3_upload_path)
        mock_save.assert_called_once()
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers.prepare_for_torchserve")
    @patch.object(MockModelBuilderServers, "_save_model_inference_spec")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_local_container_mode(
        self, mock_create, mock_prepare, mock_detect, mock_save, mock_ts_prepare
    ):
        """Test building for LOCAL_CONTAINER mode."""
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model = Mock()
        mock_ts_prepare.return_value = ""
        mock_create.return_value = Mock()

        result = self.builder._build_for_torchserve()

        mock_ts_prepare.assert_called_once()
        self.assertEqual(self.builder.secret_key, "")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers.prepare_for_torchserve")
    @patch.object(MockModelBuilderServers, "_save_model_inference_spec")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_mode(
        self, mock_create, mock_prepare, mock_detect, mock_save, mock_ts_prepare
    ):
        """Test building for SAGEMAKER_ENDPOINT mode."""
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = Mock()
        mock_ts_prepare.return_value = ""
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        result = self.builder._build_for_torchserve()

        mock_ts_prepare.assert_called_once()
        self.assertEqual(self.builder.secret_key, "")
        mock_prepare.assert_called_with(should_upload_artifacts=True)


class TestBuildForTGI(unittest.TestCase):
    """Test _build_for_tgi method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TGI

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_notebook_instance(
        self, mock_create, mock_prepare, mock_detect, mock_validate, mock_dir, mock_nb
    ):
        """Test building with notebook instance detection."""
        mock_nb.return_value = "ml.g4dn.xlarge"
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = Mock()

        result = self.builder._build_for_tgi()

        self.assertEqual(self.builder.instance_type, "ml.g4dn.xlarge")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tgi_config,
        mock_hf_config,
    ):
        """Test building with HuggingFace model."""
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_tgi_config.return_value = ({"MAX_INPUT_LENGTH": "1024"}, 512)
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "gpt2"
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_tgi()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "gpt2")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "token")
        self.assertEqual(self.builder.env_vars["SHARDED"], "false")
        self.assertEqual(self.builder.env_vars["NUM_SHARD"], "1")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_with_gpu(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tp,
        mock_gpu,
    ):
        """Test building for SAGEMAKER_ENDPOINT with GPU sharding."""
        mock_nb.return_value = None
        mock_gpu.return_value = 4
        mock_tp.return_value = 2
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = Mock()
        self.builder.hf_model_config = {"model_type": "gpt2"}

        result = self.builder._build_for_tgi()

        self.assertEqual(self.builder.env_vars["NUM_SHARD"], "2")
        self.assertEqual(self.builder.env_vars["SHARDED"], "true")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info_fallback")
    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_gpu_fallback(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tp,
        mock_gpu,
        mock_fallback,
    ):
        """Test GPU info fallback when primary method fails."""
        mock_nb.return_value = None
        mock_gpu.side_effect = Exception("GPU info failed")
        mock_fallback.return_value = 2
        mock_tp.return_value = 1
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = Mock()

        result = self.builder._build_for_tgi()

        mock_fallback.assert_called_once()
        mock_create.assert_called_once()

    # ------------------------------------------------------------------
    # Bug condition tests (tgi-s3-model-loading bugfix spec)
    #
    # Properties 1-3: Expected Behavior - TGI honors S3 model inputs.
    #
    # These were the bug-condition exploration tests written in task 1 to
    # assert the buggy behavior on the UNFIXED code. Now that the additive
    # S3 fix from task 3.1 is in place, the SAME tests have had their
    # assertions inverted to encode the expected (fixed) behavior:
    #   * an S3 model_path no longer reaches the local mkdir (Property 1),
    #   * the container env points HF_MODEL_ID at /opt/ml/model with
    #     HF_HUB_OFFLINE=1 and the S3 source is routed through
    #     _prepare_for_mode(model_path=...) so the uncompressed
    #     ModelDataSource is built (Property 2),
    #   * a user-supplied HF_MODEL_ID is preserved (Property 3).
    # See .kiro/specs/tgi-s3-model-loading/{bugfix,design,tasks}.md.
    # ------------------------------------------------------------------

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_fixed_s3_model_path_skips_local_mkdir(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tgi_config,
        mock_hf_config,
        mock_tp,
        mock_gpu,
    ):
        """Property 1 (clause 2.1): an S3 model_path skips the local mkdir.

        On the FIXED code the S3 weight source is detected before any local
        directory is created, so ``_create_dir_structure`` is NOT invoked for
        an ``s3://...`` model_path and no bogus ``s3:/bucket/weights`` dir is
        created on disk. EXPECTED OUTCOME: this test PASSES on fixed code.

        Validates: Requirements 2.1
        """
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_tgi_config.return_value = ({"MAX_INPUT_LENGTH": "1024"}, 512)
        mock_gpu.return_value = 1
        mock_tp.return_value = 1
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = "mistralai/Mistral-7B-v0.1"
        self.builder.model_path = "s3://bucket/weights/"

        self.builder._build_for_tgi()

        # Property 1: the S3 URI is never fed into the local mkdir helper.
        mock_dir.assert_not_called()

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_fixed_s3_model_path_points_at_mounted_weights(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tgi_config,
        mock_hf_config,
        mock_tp,
        mock_gpu,
    ):
        """Property 2 (clauses 2.2, 2.3, 2.4): an S3 model_path points the
        container at the mounted weights.

        On the FIXED code the container env (captured at ``_create_model`` time)
        has ``HF_MODEL_ID=/opt/ml/model`` and ``HF_HUB_OFFLINE=1``, and the S3
        source is routed through ``_prepare_for_mode(model_path=...)`` so the
        ``_upload_tgi_artifacts`` S3 branch builds the uncompressed
        ``ModelDataSource``. EXPECTED OUTCOME: this test PASSES on fixed code.

        Validates: Requirements 2.2, 2.3, 2.4
        """
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_tgi_config.return_value = ({"MAX_INPUT_LENGTH": "1024"}, 512)
        mock_gpu.return_value = 1
        mock_tp.return_value = 1
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = "mistralai/Mistral-7B-v0.1"
        self.builder.model_path = "s3://bucket/weights/"

        # The container env is baked at _create_model() time (the post-build
        # HF_HUB_OFFLINE reset only mutates the in-memory env_vars afterwards),
        # so capture a snapshot of env_vars when _create_model is invoked.
        captured_env = {}

        def _capture_container_env():
            captured_env.update(self.builder.env_vars)
            return Mock()

        mock_create.side_effect = _capture_container_env

        self.builder._build_for_tgi()

        # Property 2: container points at the mounted S3 weights, offline.
        self.assertEqual(captured_env["HF_MODEL_ID"], "/opt/ml/model")
        self.assertEqual(captured_env["HF_HUB_OFFLINE"], "1")
        # Property 2 (2.4): the S3 source is routed through _prepare_for_mode so
        # the _upload_tgi_artifacts S3 branch builds the ModelDataSource.
        mock_prepare.assert_called_once_with(model_path="s3://bucket/weights/")

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_fixed_s3_hf_hub_offline_survives_post_build_reset(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tgi_config,
        mock_hf_config,
        mock_tp,
        mock_gpu,
    ):
        """Property 2 (clause 2.3): HF_HUB_OFFLINE stays "1" through end of build.

        The end-of-method reset (``if "HF_HUB_OFFLINE" in self.env_vars: ... "0"``)
        runs after ``_create_model`` and previously clobbered the S3 offline flag
        back to "0". For an S3-mounted weight source it must remain "1" so TGI
        loads from /opt/ml/model and does not phone home. Regression test for the
        HF_HUB_OFFLINE-reset defect surfaced during real deployment.

        Validates: Requirements 2.3
        """
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_tgi_config.return_value = ({"MAX_INPUT_LENGTH": "1024"}, 512)
        mock_gpu.return_value = 1
        mock_tp.return_value = 1
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = "mistralai/Mistral-7B-v0.1"
        self.builder.model_path = "s3://bucket/weights/"

        self.builder._build_for_tgi()

        # After the full build (including the post-build reset), the S3 path must
        # still be offline so TGI serves from the mounted weights.
        self.assertEqual(self.builder.env_vars["HF_HUB_OFFLINE"], "1")

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_fixed_s3_model_data_url_routed_to_s3_branch(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tgi_config,
        mock_hf_config,
        mock_tp,
        mock_gpu,
    ):
        """Property 2 (clause 2.4): an s3_model_data_url weight source is honored.

        On the FIXED code the non-local branch detects the S3 weight source
        supplied via ``s3_model_data_url`` and calls
        ``_prepare_for_mode(model_path=<s3 url>)`` so it reaches the
        ``_upload_tgi_artifacts`` S3 upload branch, and the container env
        (captured at ``_create_model`` time) has ``HF_MODEL_ID=/opt/ml/model``.
        EXPECTED OUTCOME: this test PASSES on fixed code.

        Validates: Requirements 2.4
        """
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_tgi_config.return_value = ({"MAX_INPUT_LENGTH": "1024"}, 512)
        mock_gpu.return_value = 1
        mock_tp.return_value = 1
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = "org/model"
        # model_path stays the genuine local default; the S3 weight source is
        # supplied via s3_model_data_url.
        self.builder.s3_model_data_url = "s3://bucket/weights/"

        captured_env = {}

        def _capture_container_env():
            captured_env.update(self.builder.env_vars)
            return Mock()

        mock_create.side_effect = _capture_container_env

        self.builder._build_for_tgi()

        # Property 2 (2.4): the s3_model_data_url is routed through the S3
        # upload branch via _prepare_for_mode(model_path=...).
        mock_prepare.assert_called_once_with(model_path="s3://bucket/weights/")
        self.assertEqual(captured_env["HF_MODEL_ID"], "/opt/ml/model")
        self.assertEqual(captured_env["HF_HUB_OFFLINE"], "1")

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_fixed_s3_preserves_user_supplied_hf_model_id(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tgi_config,
        mock_hf_config,
        mock_tp,
        mock_gpu,
    ):
        """Property 3 (clause 2.5): a user-supplied HF_MODEL_ID is preserved.

        With an S3 weight source and ``env_vars={"HF_MODEL_ID":
        "/opt/ml/model/custom"}`` the fixed code uses ``setdefault`` so the
        user value is preserved and NOT overwritten with ``/opt/ml/model``.
        ``HF_HUB_OFFLINE`` is still defaulted to ``"1"``.
        EXPECTED OUTCOME: this test PASSES on fixed code.

        Validates: Requirements 2.5
        """
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_tgi_config.return_value = ({"MAX_INPUT_LENGTH": "1024"}, 512)
        mock_gpu.return_value = 1
        mock_tp.return_value = 1
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = "mistralai/Mistral-7B-v0.1"
        self.builder.model_path = "s3://bucket/weights/"
        self.builder.env_vars = {"HF_MODEL_ID": "/opt/ml/model/custom"}

        captured_env = {}

        def _capture_container_env():
            captured_env.update(self.builder.env_vars)
            return Mock()

        mock_create.side_effect = _capture_container_env

        self.builder._build_for_tgi()

        # Property 3: the user-supplied HF_MODEL_ID is preserved.
        self.assertEqual(captured_env["HF_MODEL_ID"], "/opt/ml/model/custom")
        self.assertEqual(captured_env["HF_HUB_OFFLINE"], "1")

    # ------------------------------------------------------------------
    # Preservation property tests (tgi-s3-model-loading bugfix spec)
    #
    # Property 4: Preservation - non-bug TGI and non-TGI builds are unchanged.
    #
    # Observation-first methodology: the outputs asserted below were recorded
    # by running the UNFIXED code for inputs where isBugCondition is false
    # (no S3 weight source). They assert the baseline behavior to preserve, so
    # they PASS on the unfixed code and must keep passing after the additive
    # S3 fix. Each test parametrizes across multiple non-bug inputs (varied
    # local paths, repo ids, JumpStart ids) to assert the universal
    # preservation property.
    # See .kiro/specs/tgi-s3-model-loading/{bugfix,design,tasks}.md.
    # ------------------------------------------------------------------

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_preserve_local_path_and_hf_hub(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tgi_config,
        mock_hf_config,
        mock_tp,
        mock_gpu,
    ):
        """Property 4 (clauses 3.1, 3.2): genuine local path + HF repo id unchanged.

        Property-based style: across varied genuine local ``model_path`` and HF
        repo id inputs (where ``isBugCondition`` is false because there is no S3
        weight source), the TGI build always:
          * calls ``_create_dir_structure`` with the genuine local path (3.1),
          * sets ``HF_MODEL_ID`` to the HF repo id and never sets
            ``HF_HUB_OFFLINE`` in SAGEMAKER_ENDPOINT mode (3.2),
          * routes ``_prepare_for_mode`` with no ``model_path`` so no S3
            ``ModelDataSource`` is attached (3.2),
          * leaves ``s3_upload_path`` as ``None``.

        Observed on UNFIXED code. EXPECTED OUTCOME: PASSES (baseline to preserve).

        Validates: Requirements 3.1, 3.2
        """
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_tgi_config.return_value = ({"MAX_INPUT_LENGTH": "1024"}, 512)
        mock_gpu.return_value = 1
        mock_tp.return_value = 1
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        non_bug_cases = [
            ("/tmp/local", "gpt2"),
            ("/tmp/foo", "mistralai/Mistral-7B-v0.1"),
            ("/home/user/models", "org/custom-model"),
            (tempfile.mkdtemp(), "bert-base-uncased"),
        ]
        for model_path, repo_id in non_bug_cases:
            with self.subTest(model_path=model_path, repo_id=repo_id):
                mock_dir.reset_mock()
                mock_prepare.reset_mock()
                builder = MockModelBuilderServers()
                builder.model_server = ModelServer.TGI
                builder.mode = Mode.SAGEMAKER_ENDPOINT
                builder.model = repo_id
                builder.model_path = model_path

                builder._build_for_tgi()

                # 3.1: the genuine local dir is created (no S3 short-circuit).
                mock_dir.assert_called_once_with(model_path)
                # 3.2: HF-Hub download preserved, no offline flag in endpoint mode.
                self.assertEqual(builder.env_vars["HF_MODEL_ID"], repo_id)
                self.assertNotIn("HF_HUB_OFFLINE", builder.env_vars)
                # 3.2: no S3 ModelDataSource routing (prepare called with no model_path).
                mock_prepare.assert_called_once_with()
                self.assertIsNone(builder.s3_upload_path)

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_preserve_jumpstart_build(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tp,
        mock_gpu,
    ):
        """Property 4 (clause 3.4): JumpStart TGI builds unchanged.

        When ``_is_jumpstart_model_id()`` is true the HF configuration block in
        ``_build_for_tgi`` is skipped, so ``HF_MODEL_ID`` is never set from the
        model id and the JumpStart artifact path runs as before. The genuine
        local dir is still created. Across varied JumpStart ids this baseline
        holds on the UNFIXED code and must be preserved (the bug condition
        explicitly excludes JumpStart ids).

        EXPECTED OUTCOME: PASSES on unfixed code.

        Validates: Requirements 3.4
        """
        mock_js.return_value = True
        mock_nb.return_value = None
        mock_gpu.return_value = 1
        mock_tp.return_value = 1
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        jumpstart_ids = [
            "huggingface-llm-falcon-7b",
            "meta-textgeneration-llama-2-7b",
            "huggingface-textgeneration1-gpt-j-6b",
        ]
        for js_id in jumpstart_ids:
            with self.subTest(js_id=js_id):
                mock_dir.reset_mock()
                builder = MockModelBuilderServers()
                builder.model_server = ModelServer.TGI
                builder.mode = Mode.SAGEMAKER_ENDPOINT
                builder.model = js_id
                builder.model_path = "/tmp/local"

                builder._build_for_tgi()

                # JumpStart path: local dir created, HF block skipped entirely.
                mock_dir.assert_called_once_with("/tmp/local")
                self.assertNotIn("HF_MODEL_ID", builder.env_vars)


class TestBuildForDJL(unittest.TestCase):
    """Test _build_for_djl method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.DJL_SERVING

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_djl_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_timeout(
        self, mock_create, mock_prepare, mock_detect, mock_validate, mock_dir, mock_nb
    ):
        """Test building with model_data_download_timeout."""
        mock_nb.return_value = None
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model_data_download_timeout = 600

        result = self.builder._build_for_djl()

        self.assertEqual(self.builder.env_vars["MODEL_LOADING_TIMEOUT"], "600")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_djl_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_djl_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_djl_config,
        mock_hf_config,
    ):
        """Test building with HuggingFace model."""
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_djl_config.return_value = ({"OPTION_ENGINE": "Python"}, 512)
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "gpt2"
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_djl()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "gpt2")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "token")
        self.assertEqual(self.builder.env_vars["OPTION_ENGINE"], "Python")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_djl_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_tensor_parallel(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tp,
        mock_gpu,
    ):
        """Test building for SAGEMAKER_ENDPOINT with tensor parallelism."""
        mock_nb.return_value = None
        mock_gpu.return_value = 4
        mock_tp.return_value = 4
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = Mock()
        self.builder.hf_model_config = {"model_type": "gpt2"}

        result = self.builder._build_for_djl()

        self.assertEqual(self.builder.env_vars["TENSOR_PARALLEL_DEGREE"], "4")
        mock_create.assert_called_once()

    # ------------------------------------------------------------------
    # Preservation property test (tgi-s3-model-loading bugfix spec)
    #
    # Property 4 (clause 3.3): a non-TGI build (DJL) is not touched by the
    # TGI-only S3 fix. Extends the existing DJL HF-model coverage with a
    # property-based parametrization across repo ids. Observed on UNFIXED code.
    # ------------------------------------------------------------------

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_djl_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_djl_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_preserve_djl_build_unchanged(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_djl_config,
        mock_hf_config,
    ):
        """Property 4 (clause 3.3): non-TGI DJL build is unaffected by the TGI fix.

        Across varied HF repo ids the DJL build keeps ``HF_MODEL_ID=<repo id>``,
        applies its own DJL configuration (#5529), and creates its own local dir
        structure. The TGI-only S3 fix must not change any of this.

        EXPECTED OUTCOME: PASSES on unfixed code.

        Validates: Requirements 3.3
        """
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_djl_config.return_value = ({"OPTION_ENGINE": "Python"}, 512)
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        for repo_id in ["gpt2", "mistralai/Mistral-7B-v0.1", "org/custom-model"]:
            with self.subTest(repo_id=repo_id):
                mock_dir.reset_mock()
                builder = MockModelBuilderServers()
                builder.model_server = ModelServer.DJL_SERVING
                builder.mode = Mode.LOCAL_CONTAINER
                builder.model = repo_id

                builder._build_for_djl()

                self.assertEqual(builder.env_vars["HF_MODEL_ID"], repo_id)
                self.assertEqual(builder.env_vars["OPTION_ENGINE"], "Python")
                mock_dir.assert_called_once_with(builder.model_path)


class TestBuildForTriton(unittest.TestCase):
    """Test _build_for_triton method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TRITON

    @patch.object(MockModelBuilderServers, "get_huggingface_model_metadata")
    @patch.object(MockModelBuilderServers, "_validate_for_triton")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_save_inference_spec")
    @patch.object(MockModelBuilderServers, "_prepare_for_triton")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model_string(
        self,
        mock_create,
        mock_prepare_mode,
        mock_prepare_triton,
        mock_save,
        mock_js,
        mock_validate,
        mock_hf_meta,
    ):
        """Test building with HuggingFace model string."""
        mock_js.return_value = False
        mock_hf_meta.return_value = {"pipeline_tag": "text-generation"}
        mock_create.return_value = Mock()
        mock_prepare_mode.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "gpt2"
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_triton()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "gpt2")
        self.assertEqual(self.builder.env_vars["HF_TASK"], "text-generation")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "token")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._detect_framework_and_version")
    @patch("sagemaker.serve.model_builder_servers._get_model_base")
    @patch.object(MockModelBuilderServers, "_normalize_framework_to_enum")
    @patch.object(MockModelBuilderServers, "_validate_for_triton")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_for_triton")
    @patch.object(MockModelBuilderServers, "_save_inference_spec")
    @patch.object(MockModelBuilderServers, "_prepare_for_triton")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_model_object(
        self,
        mock_create,
        mock_prepare_mode,
        mock_prepare_triton,
        mock_save,
        mock_detect_img,
        mock_validate,
        mock_normalize,
        mock_base,
        mock_detect_fw,
    ):
        """Test building with model object."""
        mock_base.return_value = "pytorch_model"
        mock_detect_fw.return_value = ("pytorch", "1.8.0")
        mock_normalize.return_value = "PYTORCH"
        mock_create.return_value = Mock()
        mock_prepare_mode.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = Mock()
        self.builder.image_uri = None

        result = self.builder._build_for_triton()

        self.assertEqual(self.builder.framework_version, "1.8.0")
        mock_detect_img.assert_called_once()
        mock_create.assert_called_once()


class TestBuildForTensorFlowServing(unittest.TestCase):
    """Test _build_for_tensorflow_serving method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TENSORFLOW_SERVING
        self.builder._is_mlflow_model = True

    @patch("sagemaker.serve.model_builder_servers.save_pkl")
    @patch("sagemaker.serve.model_builder_servers.prepare_for_tf_serving")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_mlflow_model(self, mock_create, mock_prepare_mode, mock_tf_prepare, mock_save):
        """Test building MLflow model for TensorFlow Serving."""
        mock_tf_prepare.return_value = ""
        mock_create.return_value = Mock()
        mock_prepare_mode.return_value = ("s3://bucket/model.tar.gz", None)

        result = self.builder._build_for_tensorflow_serving()

        self.assertEqual(self.builder.secret_key, "")
        mock_save.assert_called_once()
        mock_create.assert_called_once()

    def test_build_non_mlflow_model_error(self):
        """Test error when building non-MLflow model."""
        self.builder._is_mlflow_model = False

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_tensorflow_serving()
        self.assertIn("mlflow", str(ctx.exception).lower())

    def test_build_missing_image_uri_error(self):
        """Test error when image_uri is missing."""
        self.builder.image_uri = None

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_tensorflow_serving()
        self.assertIn("image_uri", str(ctx.exception))


class TestBuildForTEI(unittest.TestCase):
    """Test _build_for_tei method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TEI

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model(
        self, mock_create, mock_prepare, mock_detect, mock_js, mock_dir, mock_nb, mock_hf_config
    ):
        """Test building with HuggingFace model."""
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "bert"}
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "bert-base-uncased"
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_tei()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "bert-base-uncased")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "token")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_missing_instance_type(
        self, mock_create, mock_prepare, mock_detect, mock_dir, mock_nb
    ):
        """Test error when instance_type is missing for SAGEMAKER_ENDPOINT."""
        mock_nb.return_value = None
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.instance_type = None
        self.builder.model = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_tei()
        self.assertIn("Instance type", str(ctx.exception))

    # ------------------------------------------------------------------
    # Preservation property test (tgi-s3-model-loading bugfix spec)
    #
    # Property 4 (clause 3.3): a non-TGI build (TEI) is not touched by the
    # TGI-only S3 fix. Extends the existing TEI HF-model coverage with a
    # property-based parametrization across repo ids. Observed on UNFIXED code.
    # ------------------------------------------------------------------

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_preserve_tei_build_unchanged(
        self, mock_create, mock_prepare, mock_detect, mock_js, mock_dir, mock_nb, mock_hf_config
    ):
        """Property 4 (clause 3.3): non-TGI TEI build is unaffected by the TGI fix.

        Across varied HF repo ids the TEI build keeps ``HF_MODEL_ID=<repo id>``
        and creates its local dir structure. The TGI-only S3 fix must not change
        this behavior.

        EXPECTED OUTCOME: PASSES on unfixed code.

        Validates: Requirements 3.3
        """
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "bert"}
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        for repo_id in [
            "bert-base-uncased",
            "sentence-transformers/all-MiniLM-L6-v2",
            "intfloat/e5-large-v2",
        ]:
            with self.subTest(repo_id=repo_id):
                mock_dir.reset_mock()
                builder = MockModelBuilderServers()
                builder.model_server = ModelServer.TEI
                builder.model = repo_id

                builder._build_for_tei()

                self.assertEqual(builder.env_vars["HF_MODEL_ID"], repo_id)
                mock_dir.assert_called_once_with(builder.model_path)


class TestBuildForSMD(unittest.TestCase):
    """Test _build_for_smd method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.SMD

    @patch("sagemaker.serve.model_builder_servers.prepare_for_smd")
    @patch.object(MockModelBuilderServers, "_save_model_inference_spec")
    @patch.object(MockModelBuilderServers, "_get_processing_unit")
    @patch.object(MockModelBuilderServers, "_get_smd_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_auto_image(
        self,
        mock_create,
        mock_prepare_mode,
        mock_get_img,
        mock_get_unit,
        mock_save,
        mock_smd_prepare,
    ):
        """Test building with auto-detected image."""
        mock_get_unit.return_value = "gpu"
        mock_get_img.return_value = "smd-image-uri"
        mock_smd_prepare.return_value = ""
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None
        self.builder.model = Mock()

        result = self.builder._build_for_smd()

        self.assertEqual(self.builder.image_uri, "smd-image-uri")
        self.assertEqual(self.builder.secret_key, "")
        mock_create.assert_called_once()


class TestBuildForTransformers(unittest.TestCase):
    """Test _build_for_transformers method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.MMS

    @patch("sagemaker.serve.model_builder_servers.save_pkl")
    @patch("sagemaker.serve.model_builder_servers.prepare_for_mms")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_create_conda_env")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_inference_spec_local_container(
        self,
        mock_create,
        mock_prepare_mode,
        mock_conda,
        mock_detect,
        mock_dir,
        mock_nb,
        mock_mms_prepare,
        mock_save,
    ):
        """Test building with inference_spec for LOCAL_CONTAINER."""
        mock_nb.return_value = None
        mock_mms_prepare.return_value = ""
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.inference_spec = Mock()

        result = self.builder._build_for_transformers()

        mock_save.assert_called_once()
        mock_mms_prepare.assert_called_once()
        self.assertEqual(self.builder.secret_key, "")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model_string(
        self, mock_create, mock_prepare, mock_detect, mock_js, mock_dir, mock_nb, mock_hf_config
    ):
        """Test building with HuggingFace model string."""
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "gpt2"
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_transformers()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "gpt2")
        mock_hf_config.assert_called_once_with(
            "gpt2",
            "token",
        )
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_missing_instance_type(
        self, mock_create, mock_prepare, mock_detect, mock_dir, mock_nb
    ):
        """Test error when instance_type is missing for SAGEMAKER_ENDPOINT."""
        mock_nb.return_value = None
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.instance_type = None
        self.builder.model = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_transformers()
        self.assertIn("Instance type", str(ctx.exception))

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_clean_empty_secret_key(
        self, mock_create, mock_prepare, mock_detect, mock_dir, mock_nb
    ):
        """Test cleaning empty secret key from env_vars."""
        mock_nb.return_value = None
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = Mock()
        self.builder.env_vars["SAGEMAKER_SERVE_SECRET_KEY"] = ""

        result = self.builder._build_for_transformers()

        self.assertNotIn("SAGEMAKER_SERVE_SECRET_KEY", self.builder.env_vars)
        mock_create.assert_called_once()


class TestBuildForJumpStart(unittest.TestCase):
    """Test _build_for_jumpstart and related methods."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model = "huggingface-llm-falcon-7b"

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_djl_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_djl_local_container(
        self, mock_create, mock_prepare_mode, mock_djl_res, mock_init
    ):
        """Test building DJL JumpStart model for LOCAL_CONTAINER."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "djl-inference:0.21.0"
        mock_init_kwargs.env = {"TEST": "value"}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_djl_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None

        result = self.builder._build_for_jumpstart()

        self.assertEqual(self.builder.model_server, ModelServer.DJL_SERVING)
        self.assertTrue(self.builder.prepared_for_djl)
        mock_create.assert_called_once()

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_tgi_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_tgi_local_container(
        self, mock_create, mock_prepare_mode, mock_tgi_res, mock_init
    ):
        """Test building TGI JumpStart model for LOCAL_CONTAINER."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "tgi-inference:1.0.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_tgi_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None

        result = self.builder._build_for_jumpstart()

        self.assertEqual(self.builder.model_server, ModelServer.TGI)
        self.assertTrue(self.builder.prepared_for_tgi)
        mock_create.assert_called_once()

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_mms_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_mms_local_container(
        self, mock_create, mock_prepare_mode, mock_mms_res, mock_init
    ):
        """Test building MMS JumpStart model for LOCAL_CONTAINER."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "huggingface-pytorch-inference:1.10.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_mms_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None

        result = self.builder._build_for_jumpstart()

        self.assertEqual(self.builder.model_server, ModelServer.MMS)
        self.assertTrue(self.builder.prepared_for_mms)
        mock_create.assert_called_once()

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    def test_build_unsupported_image_uri(self, mock_init):
        """Test error for unsupported JumpStart image URI."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "unsupported-image:1.0.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = None
        mock_init.return_value = mock_init_kwargs
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_jumpstart()
        self.assertIn("Local container mode is not yet supported", str(ctx.exception))

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_djl_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_passes_config_name_to_get_init_kwargs(
        self, mock_create, mock_prepare_mode, mock_djl_res, mock_init
    ):
        """Test that config_name is forwarded to get_init_kwargs."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "djl-inference:0.21.0"
        mock_init_kwargs.env = {"TEST": "value"}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_djl_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None
        self.builder.config_name = "lmi-optimized"

        self.builder._build_for_jumpstart()

        mock_init.assert_called_once_with(
            model_id=self.builder.model,
            model_version="*",
            region=self.builder.region,
            instance_type=self.builder.instance_type,
            sagemaker_session=self.builder.sagemaker_session,
            tolerate_vulnerable_model=None,
            tolerate_deprecated_model=None,
            config_name="lmi-optimized",
        )

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_djl_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_passes_none_config_name_when_not_set(
        self, mock_create, mock_prepare_mode, mock_djl_res, mock_init
    ):
        """Test that config_name defaults to None when not set."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "djl-inference:0.21.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_djl_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None
        self.builder.config_name = None

        self.builder._build_for_jumpstart()

        mock_init.assert_called_once_with(
            model_id=self.builder.model,
            model_version="*",
            region=self.builder.region,
            instance_type=self.builder.instance_type,
            sagemaker_session=self.builder.sagemaker_session,
            tolerate_vulnerable_model=None,
            tolerate_deprecated_model=None,
            config_name=None,
        )

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_djl(self, mock_create, mock_prepare, mock_init):
        """Test building DJL JumpStart for SAGEMAKER_ENDPOINT."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "djl-inference:0.21.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_create.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.image_uri = None

        result = self.builder._build_for_jumpstart()

        mock_create.assert_called_once()


class TestDeployWrappers(unittest.TestCase):
    """Test deploy wrapper methods."""

    def setUp(self):
        self.builder = MockModelBuilderServers()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_djl_deploy_in_process(self, mock_deploy):
        """Test DJL deploy wrapper for IN_PROCESS mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.IN_PROCESS

        result = self.builder._djl_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_djl_deploy_local_container(self, mock_deploy):
        """Test DJL deploy wrapper for LOCAL_CONTAINER mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER

        result = self.builder._djl_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_djl_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test DJL deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._djl_model_builder_deploy_wrapper(model_data_download_timeout=600)

        self.assertEqual(self.builder.env_vars["MODEL_LOADING_TIMEOUT"], "600")
        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_djl_deploy_with_defaults(self, mock_deploy):
        """Test DJL deploy wrapper sets default values."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._djl_model_builder_deploy_wrapper()

        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs["endpoint_logging"], True)
        self.assertEqual(call_kwargs["initial_instance_count"], 1)

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_tgi_deploy_local_container(self, mock_deploy):
        """Test TGI deploy wrapper for LOCAL_CONTAINER mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER

        result = self.builder._tgi_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_tgi_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test TGI deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._tgi_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_tei_deploy_in_process(self, mock_deploy):
        """Test TEI deploy wrapper for IN_PROCESS mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.IN_PROCESS

        result = self.builder._tei_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_tei_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test TEI deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._tei_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_js_deploy_local_container(self, mock_deploy):
        """Test JumpStart deploy wrapper for LOCAL_CONTAINER mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER

        result = self.builder._js_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_js_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test JumpStart deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.instance_type = "ml.g5.xlarge"

        result = self.builder._js_builder_deploy_wrapper()

        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs["instance_type"], "ml.g5.xlarge")
        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_transformers_deploy_local_container(self, mock_deploy):
        """Test Transformers deploy wrapper for LOCAL_CONTAINER mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER

        result = self.builder._transformers_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_transformers_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test Transformers deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._transformers_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_deploy_wrapper_removes_mode_and_role(self, mock_deploy):
        """Test deploy wrapper removes mode and role from kwargs."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._djl_model_builder_deploy_wrapper(
            mode=Mode.LOCAL_CONTAINER, role="arn:aws:iam::123456789012:role/test"
        )

        call_kwargs = mock_deploy.call_args[1]
        self.assertNotIn("mode", call_kwargs)
        self.assertNotIn("role", call_kwargs)
        self.assertEqual(self.builder.role_arn, "arn:aws:iam::123456789012:role/test")


class TestJumpStartBuilders(unittest.TestCase):
    """Test JumpStart-specific builder methods."""

    def setUp(self):
        self.builder = MockModelBuilderServers()

    @patch("sagemaker.serve.model_builder_servers.prepare_djl_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_for_djl_jumpstart_local(self, mock_create, mock_prepare, mock_djl_res):
        """Test _build_for_djl_jumpstart for local mode."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_djl_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model = "jumpstart-model-id"
        self.builder.s3_model_data_url = "s3://bucket/model.tar.gz"

        result = self.builder._build_for_djl_jumpstart(mock_init_kwargs)

        self.assertEqual(self.builder.model_server, ModelServer.DJL_SERVING)
        self.assertTrue(self.builder.prepared_for_djl)
        mock_djl_res.assert_called_once()
        mock_create.assert_called_once()

    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_for_djl_jumpstart_sagemaker(self, mock_create):
        """Test _build_for_djl_jumpstart for SAGEMAKER_ENDPOINT mode."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_create.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = "jumpstart-model-id"

        result = self.builder._build_for_djl_jumpstart(mock_init_kwargs)

        self.assertEqual(self.builder.s3_upload_path, "s3://bucket/model.tar.gz")
        self.assertTrue(self.builder.prepared_for_djl)
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers.prepare_tgi_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_for_tgi_jumpstart_local(self, mock_create, mock_prepare, mock_tgi_res):
        """Test _build_for_tgi_jumpstart for local mode."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_tgi_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model = "jumpstart-model-id"
        self.builder.s3_model_data_url = "s3://bucket/model.tar.gz"

        result = self.builder._build_for_tgi_jumpstart(mock_init_kwargs)

        self.assertEqual(self.builder.model_server, ModelServer.TGI)
        self.assertTrue(self.builder.prepared_for_tgi)
        mock_tgi_res.assert_called_once()
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers.prepare_mms_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_for_mms_jumpstart_local(self, mock_create, mock_prepare, mock_mms_res):
        """Test _build_for_mms_jumpstart for local mode."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_mms_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model = "jumpstart-model-id"
        self.builder.s3_model_data_url = "s3://bucket/model.tar.gz"

        result = self.builder._build_for_mms_jumpstart(mock_init_kwargs)

        self.assertEqual(self.builder.model_server, ModelServer.MMS)
        self.assertTrue(self.builder.prepared_for_mms)
        mock_mms_res.assert_called_once()
        mock_create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
