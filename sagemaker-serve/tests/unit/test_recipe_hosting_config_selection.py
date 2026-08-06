"""Unit tests for ModelBuilder._select_recipe_hosting_config.

A recipe publishes several pre-benchmarked hosting configurations. When the caller supplies an
instance type that matches one of them, ModelBuilder must select THAT config (its image,
environment, and compute requirements) rather than pinning the caller's instance onto the
Default config's environment. These tests pin that selection behavior.
"""

import unittest
from unittest.mock import Mock, patch

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode

# Mirrors a real recipe's HostingConfigs (e.g. llmft_qwen3_0_dot_6b_seq4k_gpu_sft_lora):
# a 1-GPU Default plus a same-topology alternative and two 8-GPU / TP=8 alternatives.
HOSTING_CONFIGS = [
    {
        "Profile": "Default",
        "InstanceType": "ml.g6.4xlarge",
        "EcrAddress": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16",
        "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
        "ComputeResourceRequirements": {
            "MinMemoryRequiredInMb": 32768,
            "NumberOfAcceleratorDevicesRequired": 1,
            "NumberOfCpuCoresRequired": 12,
        },
    },
    {
        "InstanceType": "ml.g5.4xlarge",
        "EcrAddress": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16",
        "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
    },
    {
        "InstanceType": "ml.g6e.48xlarge",
        "EcrAddress": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16",
        "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "8"},
    },
    {
        "InstanceType": "ml.p5.48xlarge",
        "EcrAddress": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16",
        "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "8"},
    },
]


def _make_mock_session():
    """A minimally-populated SageMaker session mock sufficient to construct a ModelBuilder."""
    mock_session = Mock()
    mock_session.boto_region_name = "us-west-2"
    mock_session.default_bucket.return_value = "test-bucket"
    mock_session.default_bucket_prefix = "test-prefix"
    mock_session.config = {}
    mock_session.sagemaker_config = {}
    mock_session.settings = Mock()
    mock_session.settings.include_jumpstart_tags = False
    creds = Mock()
    creds.access_key = "k"
    creds.secret_key = "s"
    creds.token = None
    mock_session.boto_session = Mock()
    mock_session.boto_session.get_credentials.return_value = creds
    mock_session.boto_session.region_name = "us-west-2"
    return mock_session


class TestSelectRecipeHostingConfig(unittest.TestCase):
    def setUp(self):
        self.mock_session = _make_mock_session()

    def _builder(self, instance_type):
        return ModelBuilder(
            model="huggingface-reasoning-qwen3-06b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-reasoning-qwen3-06b",
                "CUSTOM_MODEL_VERSION": "3.9.0",
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type=instance_type,
        )

    def test_matching_alternative_instance_selects_that_config(self):
        # A different-topology alternative (8 GPU / TP=8) must be selected in full, not Default.
        b = self._builder("ml.g6e.48xlarge")
        cfg = b._select_recipe_hosting_config(HOSTING_CONFIGS)
        self.assertEqual(cfg["InstanceType"], "ml.g6e.48xlarge")
        self.assertEqual(cfg["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "8")

    def test_matching_same_topology_alternative_selects_that_config(self):
        b = self._builder("ml.g5.4xlarge")
        cfg = b._select_recipe_hosting_config(HOSTING_CONFIGS)
        self.assertEqual(cfg["InstanceType"], "ml.g5.4xlarge")

    def test_default_instance_selects_default_config(self):
        b = self._builder("ml.g6.4xlarge")
        cfg = b._select_recipe_hosting_config(HOSTING_CONFIGS)
        self.assertEqual(cfg["Profile"], "Default")
        self.assertEqual(cfg["InstanceType"], "ml.g6.4xlarge")

    def test_no_user_instance_type_falls_back_to_default(self):
        b = self._builder("ml.g6e.48xlarge")
        # Simulate "user did not pin an instance type".
        b._user_provided_instance_type = False
        cfg = b._select_recipe_hosting_config(HOSTING_CONFIGS)
        self.assertEqual(cfg["Profile"], "Default")

    def test_unmatched_instance_type_falls_back_to_default(self):
        # A constructor-supplied instance not published for the recipe keeps prior behavior: WARN
        # and fall back to Default (contrast with an explicit selection, which raises).
        b = self._builder("ml.g6.xlarge")
        with self.assertLogs(level="WARNING") as logs:
            cfg = b._select_recipe_hosting_config(HOSTING_CONFIGS)
        self.assertEqual(cfg["Profile"], "Default")
        self.assertTrue(
            any("does not match any published hosting config" in m for m in logs.output)
        )

    def test_no_default_profile_falls_back_to_first_entry(self):
        b = self._builder("ml.g6.xlarge")  # unmatched -> fallback path
        configs = [c for c in HOSTING_CONFIGS if c.get("Profile") != "Default"]
        cfg = b._select_recipe_hosting_config(configs)
        self.assertEqual(cfg, configs[0])

    def test_stale_explicit_selection_raises_not_silent_fallback(self):
        # An EXPLICIT selection (_selected_hosting_config) whose instance type is no longer present
        # at build time must RAISE — never silently fall back to Default. (A constructor-provided
        # instance type, by contrast, warns and falls back; see the unmatched test above.)
        b = self._builder("ml.g6.4xlarge")
        b._selected_hosting_config = {"InstanceType": "ml.not-published.xlarge", "Environment": {}}
        with self.assertRaises(ValueError) as ctx:
            b._select_recipe_hosting_config(HOSTING_CONFIGS)
        msg = str(ctx.exception)
        self.assertIn("no longer published", msg)
        self.assertIn("ml.not-published.xlarge", msg)

    def test_stale_selection_still_published_via_supported_instance(self):
        # The stored selection offers [g5.2xlarge, g5.12xlarge]; the fresh list has a config that
        # offers g5.12xlarge (as its primary). They share an offered instance -> NOT stale.
        b = self._builder("ml.g5.12xlarge")
        b._selected_hosting_config = {
            "DefaultInstanceType": "ml.g5.2xlarge",
            "SupportedInstanceTypes": ["ml.g5.2xlarge", "ml.g5.12xlarge"],
            "Environment": {},
        }
        fresh = [{"InstanceType": "ml.g5.12xlarge", "Environment": {}}]
        # Must not raise; returns the stored selection exactly.
        self.assertIs(b._select_recipe_hosting_config(fresh), b._selected_hosting_config)

    def test_stale_selection_raises_when_no_offered_instance_shared(self):
        # Stored selection offers only g5.2xlarge/g5.12xlarge; fresh list shares none -> stale.
        b = self._builder("ml.g5.12xlarge")
        b._selected_hosting_config = {
            "DefaultInstanceType": "ml.g5.2xlarge",
            "SupportedInstanceTypes": ["ml.g5.2xlarge", "ml.g5.12xlarge"],
            "Environment": {},
        }
        fresh = [{"InstanceType": "ml.p5.48xlarge", "Environment": {}}]
        with self.assertRaises(ValueError) as ctx:
            b._select_recipe_hosting_config(fresh)
        self.assertIn("no longer published", str(ctx.exception))

    def test_explicit_selection_applied_exactly(self):
        # An explicit selection is applied EXACTLY (the stored raw config), not a re-derived one —
        # even if the freshly-resolved list has a same-instance entry with different contents.
        b = self._builder("ml.g6e.48xlarge")
        b._selected_hosting_config = {
            "InstanceType": "ml.g6e.48xlarge",
            "EcrAddress": "stored-image",
            "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "8"},
        }
        # Fresh list has the same instance type but different (drifted) contents.
        fresh = [
            {"Profile": "Default", "InstanceType": "ml.g6.4xlarge", "Environment": {}},
            {"InstanceType": "ml.g6e.48xlarge", "EcrAddress": "drifted-image", "Environment": {}},
        ]
        cfg = b._select_recipe_hosting_config(fresh)
        self.assertEqual(cfg["EcrAddress"], "stored-image")
        self.assertEqual(cfg["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "8")


class TestDeploymentConfigForFineTunedModels(unittest.TestCase):
    """The unified deployment-config API (list/get/set) applied to fine-tuned models.

    Same public methods as base/JumpStart models — the base "deployment config" vs recipe
    "hosting config" split is internal and never surfaces to the caller.
    """

    def setUp(self):
        self.mock_session = _make_mock_session()

    def _builder(self, instance_type="ml.g6.4xlarge"):
        return ModelBuilder(
            model="huggingface-reasoning-qwen3-06b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-reasoning-qwen3-06b",
                "CUSTOM_MODEL_VERSION": "3.9.0",
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type=instance_type,
        )

    def _customization_builder(self, instance_type="ml.g6.4xlarge"):
        """A builder patched to look like a fine-tuned model with the fixture configs."""
        b = self._builder(instance_type)
        patcher_cust = patch.object(ModelBuilder, "_is_model_customization", return_value=True)
        patcher_cfgs = patch.object(
            ModelBuilder, "_resolve_recipe_hosting_configs", return_value=HOSTING_CONFIGS
        )
        patcher_cust.start()
        patcher_cfgs.start()
        self.addCleanup(patcher_cust.stop)
        self.addCleanup(patcher_cfgs.stop)
        return b

    def test_list_deployment_configs_returns_base_shape(self):
        # Fine-tuned configs are normalized into the SAME shape as the base/JumpStart response:
        # DeploymentConfigName + a nested DeploymentArgs block + BenchmarkMetrics, plus the
        # additive IsDefault flag. A unified consumer reads DeploymentArgs["InstanceType"] etc.
        # identically for base and fine-tuned.
        b = self._customization_builder()
        configs = b.list_deployment_configs()
        self.assertEqual(len(configs), 4)
        default = next(c for c in configs if c["IsDefault"])
        self.assertEqual(default["DeploymentConfigName"], "Default")
        self.assertEqual(default["DeploymentArgs"]["InstanceType"], "ml.g6.4xlarge")
        self.assertEqual(default["DeploymentArgs"]["ImageUri"].split(":")[-1], "0.34.0-lmi16")
        self.assertEqual(
            default["DeploymentArgs"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "1"
        )
        # BenchmarkMetrics present but None (recipes publish none today; base emits None when
        # absent, so None matches the base sentinel).
        self.assertIsNone(default["BenchmarkMetrics"])
        # Unnamed configs get their instance type as a stable identifier.
        g6e = next(c for c in configs if c["DeploymentArgs"]["InstanceType"] == "ml.g6e.48xlarge")
        self.assertEqual(g6e["DeploymentConfigName"], "ml.g6e.48xlarge")
        self.assertFalse(g6e["IsDefault"])
        self.assertEqual(g6e["DeploymentArgs"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "8")

    def test_list_deployment_configs_filters_by_instance_type(self):
        b = self._customization_builder()
        configs = b.list_deployment_configs(instance_type="ml.g6e.48xlarge")
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["DeploymentArgs"]["InstanceType"], "ml.g6e.48xlarge")
        # A non-published instance filters to empty.
        self.assertEqual(b.list_deployment_configs(instance_type="ml.g6.xlarge"), [])

    def test_set_deployment_config_selects_and_pins_instance(self):
        b = self._customization_builder()
        b.set_deployment_config(instance_type="ml.g6e.48xlarge")
        self.assertEqual(b.instance_type, "ml.g6e.48xlarge")
        self.assertTrue(b._user_provided_instance_type)
        # _selected_hosting_config stores the RAW recipe config (not the normalized shape).
        self.assertEqual(b._raw_config_instance_type(b._selected_hosting_config), "ml.g6e.48xlarge")
        # The selection flows into the internal build-time selector (which reads raw configs).
        chosen = b._select_recipe_hosting_config(HOSTING_CONFIGS)
        self.assertEqual(chosen["InstanceType"], "ml.g6e.48xlarge")
        self.assertEqual(chosen["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "8")

    def test_set_deployment_config_requires_instance_type(self):
        b = self._customization_builder()
        with self.assertRaises(ValueError) as ctx:
            b.set_deployment_config()
        self.assertIn("instance_type", str(ctx.exception))

    def test_set_deployment_config_unknown_instance_raises_with_available(self):
        b = self._customization_builder()
        with self.assertRaises(ValueError) as ctx:
            b.set_deployment_config(instance_type="ml.g6.xlarge")
        msg = str(ctx.exception)
        self.assertIn("ml.g6.4xlarge", msg)  # lists available instance types
        self.assertIn("ml.p5.48xlarge", msg)

    def test_set_deployment_config_ambiguous_raises(self):
        b = self._builder()
        dup = [
            {"InstanceType": "ml.g6.4xlarge", "Environment": {}},
            {"InstanceType": "ml.g6.4xlarge", "Environment": {}},
        ]
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True), patch.object(
            ModelBuilder, "_resolve_recipe_hosting_configs", return_value=dup
        ):
            with self.assertRaises(ValueError) as ctx:
                b.set_deployment_config(instance_type="ml.g6.4xlarge")
        self.assertIn("ambiguous", str(ctx.exception))

    def test_get_deployment_config_returns_selected_then_default(self):
        b = self._customization_builder()
        # Before any selection -> the Default config.
        self.assertEqual(b.get_deployment_config()["DeploymentConfigName"], "Default")
        # After selecting an alternative -> that config.
        b.set_deployment_config(instance_type="ml.p5.48xlarge")
        got = b.get_deployment_config()
        self.assertEqual(got["DeploymentArgs"]["InstanceType"], "ml.p5.48xlarge")
        self.assertFalse(got["IsDefault"])

    def test_set_deployment_config_base_requires_config_name(self):
        # A base/JumpStart model going through the SAME setter must still require config_name — the
        # instance_type-only form is fine-tuned-only. Guards the config_name-None base branch.
        b = self._builder()
        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                b.set_deployment_config(instance_type="ml.g5.2xlarge")
        self.assertIn("config_name is required", str(ctx.exception))

    def test_get_deployment_config_base_returns_none_until_set(self):
        # Base/JumpStart contract (distinct from fine-tuned): get returns None until a config is
        # explicitly set. Pins the documented semantic difference from the fine-tuned path.
        b = self._builder()
        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True):
            b.config_name = None
            self.assertIsNone(b.get_deployment_config())

    def test_set_deployment_config_base_success_sets_name_and_instance(self):
        # Base happy path: config_name + instance_type are recorded (contrast with the fine-tuned
        # instance-type-only selection). Metadata confirms the config publishes the instance.
        b = self._builder()
        b.additional_model_data_sources = None  # normally populated by the build path
        meta = {
            "lmi": Mock(
                resolved_config={
                    "default_inference_instance_type": "ml.g5.2xlarge",
                    "supported_inference_instance_types": ["ml.g5.2xlarge"],
                }
            )
        }
        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True), patch.object(
            ModelBuilder, "_ensure_metadata_configs"
        ), patch.object(ModelBuilder, "get_deployment_config", return_value=None):
            b._metadata_configs = meta
            b.set_deployment_config(config_name="lmi", instance_type="ml.g5.2xlarge")
        self.assertEqual(b.config_name, "lmi")
        self.assertEqual(b.instance_type, "ml.g5.2xlarge")

    def test_set_deployment_config_base_rejects_unknown_config_name(self):
        # Base setter must fail fast on an unpublished config name (mirrors the fine-tuned branch),
        # not silently record a bogus selection that applies nothing at build.
        b = self._builder()
        meta = {"lmi": Mock(resolved_config={"default_inference_instance_type": "ml.g5.2xlarge"})}
        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True), patch.object(
            ModelBuilder, "_ensure_metadata_configs"
        ):
            b._metadata_configs = meta
            with self.assertRaises(ValueError) as ctx:
                b.set_deployment_config(config_name="bogus", instance_type="ml.g5.2xlarge")
        self.assertIn("bogus", str(ctx.exception))

    def test_set_deployment_config_base_rejects_unsupported_instance(self):
        # Base setter must fail fast when the named config does not support the requested instance.
        b = self._builder()
        meta = {
            "lmi": Mock(
                resolved_config={
                    "default_inference_instance_type": "ml.g5.2xlarge",
                    "supported_inference_instance_types": ["ml.g5.2xlarge"],
                }
            )
        }
        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True), patch.object(
            ModelBuilder, "_ensure_metadata_configs"
        ):
            b._metadata_configs = meta
            with self.assertRaises(ValueError) as ctx:
                b.set_deployment_config(config_name="lmi", instance_type="ml.bogus.xlarge")
        self.assertIn("does not support", str(ctx.exception))

    def test_repeated_selection_applies_latest(self):
        # Selecting A then B must fully replace A — no stale env/compute from the first selection.
        b = self._customization_builder()
        b.set_deployment_config(instance_type="ml.g5.4xlarge")  # TP=1 alternative
        b.set_deployment_config(instance_type="ml.g6e.48xlarge")  # TP=8 alternative
        self.assertEqual(b._raw_config_instance_type(b._selected_hosting_config), "ml.g6e.48xlarge")
        got = b.get_deployment_config()
        self.assertEqual(got["DeploymentArgs"]["InstanceType"], "ml.g6e.48xlarge")
        self.assertEqual(got["DeploymentArgs"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "8")

    def test_returned_config_mutation_does_not_corrupt_selection(self):
        # Mutating a dict returned by get_deployment_config() must not alter internal state.
        b = self._customization_builder()
        b.set_deployment_config(instance_type="ml.g6e.48xlarge")
        got = b.get_deployment_config()
        got["DeploymentArgs"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"] = "999"
        got["DeploymentConfigName"] = "mutated"
        again = b.get_deployment_config()
        self.assertEqual(
            again["DeploymentArgs"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "8"
        )
        self.assertEqual(again["DeploymentConfigName"], "ml.g6e.48xlarge")

    def test_list_config_mutation_does_not_corrupt_next_list(self):
        b = self._customization_builder()
        first = b.list_deployment_configs()
        first[0]["DeploymentArgs"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"] = "999"
        second = b.list_deployment_configs()
        self.assertNotEqual(
            second[0]["DeploymentArgs"]["Environment"].get("OPTION_TENSOR_PARALLEL_DEGREE"), "999"
        )

    # ---- SupportedInstanceTypes: a config offered under a multi-instance list ----

    def _supported_types_builder(self):
        # A recipe config whose DEFAULT is ml.g5.2xlarge but which also OFFERS ml.g5.12xlarge via
        # SupportedInstanceTypes. Recipe configs are per-instance bundles by contract, but the API
        # honors SupportedInstanceTypes for matching so such a config is still findable/selectable.
        configs = [
            {
                "Profile": "Default",
                "InstanceType": "ml.g6.4xlarge",
                "EcrAddress": "img:default",
                "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
            },
            {
                "DefaultInstanceType": "ml.g5.2xlarge",
                "SupportedInstanceTypes": ["ml.g5.2xlarge", "ml.g5.12xlarge"],
                "EcrAddress": "img:multi",
                "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "4"},
            },
        ]
        b = self._builder()
        p1 = patch.object(ModelBuilder, "_is_model_customization", return_value=True)
        p2 = patch.object(
            ModelBuilder, "_resolve_recipe_hosting_configs", return_value=configs
        )
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        return b

    def test_list_filter_finds_config_by_supported_non_default_instance(self):
        # Filtering for a supported non-default instance returns the config, materialized FOR it.
        b = self._supported_types_builder()
        configs = b.list_deployment_configs(instance_type="ml.g5.12xlarge")
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["DeploymentArgs"]["InstanceType"], "ml.g5.12xlarge")
        # Unnamed config's identifier tracks the materialized instance.
        self.assertEqual(configs[0]["DeploymentConfigName"], "ml.g5.12xlarge")
        # The config's own env travels with it.
        self.assertEqual(
            configs[0]["DeploymentArgs"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "4"
        )

    def test_set_selects_config_by_supported_non_default_instance(self):
        # Selecting a supported non-default instance matches the config and pins that instance.
        b = self._supported_types_builder()
        b.set_deployment_config(instance_type="ml.g5.12xlarge")
        self.assertEqual(b.instance_type, "ml.g5.12xlarge")
        self.assertEqual(b._selected_hosting_config["EcrAddress"], "img:multi")
        # Available-instances error message includes the supported (non-default) instance too.
        with self.assertRaises(ValueError) as ctx:
            b.set_deployment_config(instance_type="ml.nonexistent.xlarge")
        self.assertIn("ml.g5.12xlarge", str(ctx.exception))

    def test_get_after_set_agrees_with_list_for_supported_non_default_instance(self):
        # Round-trip consistency: for a config selected on a SupportedInstanceTypes entry that
        # DIFFERS from its default, list / set / get must all report the SAME instance and config
        # identity — get_deployment_config() must materialize the pinned instance, not the default.
        b = self._supported_types_builder()
        listed = b.list_deployment_configs(instance_type="ml.g5.12xlarge")[0]
        b.set_deployment_config(instance_type="ml.g5.12xlarge")
        got = b.get_deployment_config()
        self.assertEqual(got["DeploymentArgs"]["InstanceType"], "ml.g5.12xlarge")
        self.assertEqual(got["DeploymentConfigName"], listed["DeploymentConfigName"])
        self.assertEqual(
            got["DeploymentArgs"]["InstanceType"], listed["DeploymentArgs"]["InstanceType"]
        )

    def test_get_default_before_selection_reports_offered_instance(self):
        # With no explicit selection but a constructor instance that a config offers via
        # SupportedInstanceTypes, get_deployment_config() reports that pinned instance.
        b = self._builder(instance_type="ml.g5.12xlarge")
        configs = [
            {
                "DefaultInstanceType": "ml.g5.2xlarge",
                "SupportedInstanceTypes": ["ml.g5.2xlarge", "ml.g5.12xlarge"],
                "EcrAddress": "img:multi",
                "Environment": {},
            },
        ]
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True), patch.object(
            ModelBuilder, "_resolve_recipe_hosting_configs", return_value=configs
        ):
            got = b.get_deployment_config()
        self.assertEqual(got["DeploymentArgs"]["InstanceType"], "ml.g5.12xlarge")

    def test_list_returns_both_configs_that_offer_same_instance(self):
        # Listing does not dedup: two distinct configs both offering the requested instance are
        # both returned (materialized for it). Selection, by contrast, rejects this as ambiguous.
        configs = [
            {"InstanceType": "ml.g5.2xlarge", "SupportedInstanceTypes": ["ml.g5.12xlarge"]},
            {"InstanceType": "ml.g6.4xlarge", "SupportedInstanceTypes": ["ml.g5.12xlarge"]},
        ]
        b = self._builder()
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True), patch.object(
            ModelBuilder, "_resolve_recipe_hosting_configs", return_value=configs
        ):
            listed = b.list_deployment_configs(instance_type="ml.g5.12xlarge")
            with self.assertRaises(ValueError) as ctx:
                b.set_deployment_config(instance_type="ml.g5.12xlarge")
        self.assertEqual(len(listed), 2)
        self.assertIn("ambiguous", str(ctx.exception))


class TestBuildAppliesSelectedConfig(unittest.TestCase):
    """The point of the feature: the SELECTED hosting config's container image, environment, and
    compute requirements must actually reach the built artifact via _fetch_and_cache_recipe_config
    — not just be returned by the selector. Also pins the DefaultInstanceType round-trip so a
    config published with only DefaultInstanceType is not silently dropped at build time.
    """

    def setUp(self):
        self.mock_session = _make_mock_session()

    def _builder(self, instance_type):
        return ModelBuilder(
            model="huggingface-reasoning-qwen3-06b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-reasoning-qwen3-06b",
                "CUSTOM_MODEL_VERSION": "3.9.0",
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type=instance_type,
        )

    def _patch_recipe(self, b, hosting_configs, recipe_name="recipe-1"):
        hub = {"RecipeCollection": [{"Name": recipe_name, "HostingConfigs": hosting_configs}]}
        mp = Mock()
        container = Mock()
        container.base_model.recipe_name = recipe_name
        # Non-Nova names so _is_nova_model() (now consulted first at build) returns False.
        container.base_model.hub_content_name = "test-hub-content"
        mp.inference_specification.containers = [container]
        b._fetch_hub_document_for_custom_model = Mock(return_value=hub)
        b._fetch_model_package = Mock(return_value=mp)
        b.s3_upload_path = "s3://bucket/model"  # skip the s3-uri resolution block
        b._resolve_compute_requirements_from_config = Mock(return_value={})

    def test_build_applies_selected_alternative_not_default(self):
        # User picked the 8-GPU / TP=8 alternative; build must apply THAT config's env/image,
        # not the 1-GPU Default's.
        b = self._builder("ml.g6e.48xlarge")
        self._patch_recipe(b, HOSTING_CONFIGS)
        b.image_uri = None
        b.env_vars = None
        b._fetch_and_cache_recipe_config()
        self.assertEqual(b.image_uri.split(":")[-1], "0.34.0-lmi16")
        # The decisive assertion: TP degree came from the selected alternative (8), not Default (1).
        self.assertEqual(b.env_vars["OPTION_TENSOR_PARALLEL_DEGREE"], "8")
        self.assertEqual(b.instance_type, "ml.g6e.48xlarge")

    def test_build_merges_selected_env_over_user_env(self):
        b = self._builder("ml.g6e.48xlarge")
        self._patch_recipe(b, HOSTING_CONFIGS)
        b.image_uri = None
        b.env_vars = {"MY_FLAG": "1"}
        b._fetch_and_cache_recipe_config()
        self.assertEqual(b.env_vars["MY_FLAG"], "1")  # user env preserved
        self.assertEqual(b.env_vars["OPTION_TENSOR_PARALLEL_DEGREE"], "8")  # recipe env applied

    def test_defaultinstancetype_only_config_round_trips(self):
        # A config published with only DefaultInstanceType (no InstanceType) must be selectable AND
        # re-match at build time — otherwise it silently falls back to Default (the M1 bug).
        img = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16"
        raw = [
            {
                "Profile": "Default",
                "InstanceType": "ml.g6.4xlarge",
                "EcrAddress": img,
                "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
            },
            {
                "DefaultInstanceType": "ml.g6e.48xlarge",
                "EcrAddress": img,
                "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "8"},
            },
        ]
        b = self._builder("ml.g6.4xlarge")
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True), patch.object(
            ModelBuilder, "_resolve_recipe_hosting_configs", return_value=raw
        ):
            b.set_deployment_config(instance_type="ml.g6e.48xlarge")
            # Stored selection is the RAW config (only DefaultInstanceType, no InstanceType key).
            self.assertEqual(
                b._raw_config_instance_type(b._selected_hosting_config), "ml.g6e.48xlarge"
            )
            # Build-time selector must re-match the DefaultInstanceType-only entry, not fall back.
            chosen = b._select_recipe_hosting_config(raw)
        self.assertEqual(chosen.get("DefaultInstanceType"), "ml.g6e.48xlarge")
        self.assertEqual(chosen["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "8")

    def test_supported_instance_selection_preserved_at_build(self):
        # A config offered via SupportedInstanceTypes (default differs) selected on a NON-default
        # supported instance must, at build, apply that config's image/env AND keep the pinned
        # instance — not snap back to the config's default instance.
        img = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16"
        raw = [
            {
                "Profile": "Default",
                "InstanceType": "ml.g6.4xlarge",
                "EcrAddress": img,
                "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
            },
            {
                "DefaultInstanceType": "ml.g5.2xlarge",
                "SupportedInstanceTypes": ["ml.g5.2xlarge", "ml.g5.12xlarge"],
                "EcrAddress": img,
                "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "4"},
            },
        ]
        b = self._builder("ml.g6.4xlarge")
        self._patch_recipe(b, raw)
        b.image_uri = None
        b.env_vars = None
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True):
            b.set_deployment_config(instance_type="ml.g5.12xlarge")
            b._fetch_and_cache_recipe_config()
        # The multi-instance config's env is applied...
        self.assertEqual(b.env_vars["OPTION_TENSOR_PARALLEL_DEGREE"], "4")
        # ...and the pinned non-default supported instance is preserved (not snapped to g5.2xlarge).
        self.assertEqual(b.instance_type, "ml.g5.12xlarge")

    def test_build_tolerates_explicit_null_environment(self):
        # A config that publishes an explicit "Environment": null (key present, value None) must
        # not blow up the build's env merge (dict(None)/update(None) would raise TypeError).
        img = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16"
        raw = [
            {
                "Profile": "Default",
                "InstanceType": "ml.g6.4xlarge",
                "EcrAddress": img,
                "Environment": None,
            },
        ]
        b = self._builder("ml.g6.4xlarge")
        self._patch_recipe(b, raw)
        b.image_uri = None
        b.env_vars = None
        b._fetch_and_cache_recipe_config()
        self.assertEqual(b.env_vars, {})
        self.assertEqual(b.image_uri, img)

    def test_top_level_config_selected_and_applied_at_build(self):
        # P1 (blocking): a config that lives ONLY at the top level (recipe entry publishes no
        # HostingConfigs) must be both selectable AND applied at build. Previously the build path
        # walked only recipe-level configs, so a selected top-level config was silently dropped.
        img = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16"
        top = [
            {
                "Profile": "Default",
                "InstanceType": "ml.g6.4xlarge",
                "EcrAddress": img,
                "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "1"},
            },
            {
                "InstanceType": "ml.g6e.48xlarge",
                "EcrAddress": img,
                "Environment": {"OPTION_TENSOR_PARALLEL_DEGREE": "8"},
            },
        ]
        # Recipe entry exists but carries NO HostingConfigs -> top-level fallback.
        hub = {"RecipeCollection": [{"Name": "recipe-1"}], "HostingConfigs": top}
        b = self._builder("ml.g6e.48xlarge")
        mp = Mock()
        container = Mock()
        container.base_model.recipe_name = "recipe-1"
        container.base_model.hub_content_name = "test-hub-content"  # non-Nova
        mp.inference_specification.containers = [container]
        b._fetch_hub_document_for_custom_model = Mock(return_value=hub)
        b._fetch_model_package = Mock(return_value=mp)
        b.s3_upload_path = "s3://bucket/model"
        b._resolve_compute_requirements_from_config = Mock(return_value={})
        b.image_uri = None
        b.env_vars = None
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True):
            # Select via the public API (resolves through the top-level fallback)...
            b.set_deployment_config(instance_type="ml.g6e.48xlarge")
            # ...and the build path must apply that same top-level config.
            b._fetch_and_cache_recipe_config()
        self.assertEqual(b.image_uri.split(":")[-1], "0.34.0-lmi16")
        self.assertEqual(b.env_vars["OPTION_TENSOR_PARALLEL_DEGREE"], "8")

    def test_user_image_uri_not_overridden_by_selected_config(self):
        # A caller-provided image_uri wins; the recipe env is still applied on top.
        b = self._builder("ml.g6e.48xlarge")
        self._patch_recipe(b, HOSTING_CONFIGS)
        b.image_uri = "user-supplied-image"
        b.env_vars = None
        b._fetch_and_cache_recipe_config()
        self.assertEqual(b.image_uri, "user-supplied-image")
        self.assertEqual(b.env_vars["OPTION_TENSOR_PARALLEL_DEGREE"], "8")

    def test_constructor_instance_duplicate_matches_first(self):
        # Constructor-supplied instance type (no explicit selection) with duplicate matches: the
        # first published config wins. (set_deployment_config, by contrast, rejects ambiguity.)
        b = self._builder("ml.dup.xlarge")
        dup = [
            {"InstanceType": "ml.dup.xlarge", "EcrAddress": "first", "Environment": {}},
            {"InstanceType": "ml.dup.xlarge", "EcrAddress": "second", "Environment": {}},
        ]
        cfg = b._select_recipe_hosting_config(dup)
        self.assertEqual(cfg["EcrAddress"], "first")

    def test_build_nova_model_uses_nova_path_not_generic(self):
        # A Nova model that publishes top-level HostingConfigs must still resolve via the Nova path
        # (Nova env precedence + SMI validation), NOT the generic recipe branch. Guards the
        # regression where the top-level fallback would divert Nova to the generic path.
        b = self._builder("ml.p5.48xlarge")
        hub = {
            "HostingConfigs": [
                {
                    "InstanceType": "ml.p5.48xlarge",
                    "EcrAddress": "generic-image",
                    "Environment": {"GENERIC": "1"},
                }
            ]
        }
        mp = Mock()
        container = Mock()
        container.base_model.recipe_name = "nova-lite-recipe"  # matches _is_nova_model
        container.base_model.hub_content_name = "nova-textgeneration-lite"
        mp.inference_specification.containers = [container]
        b._fetch_hub_document_for_custom_model = Mock(return_value=hub)
        b._fetch_model_package = Mock(return_value=mp)
        b.s3_upload_path = "s3://bucket/model"
        b.image_uri = None
        b.env_vars = {"USER_FLAG": "keep"}
        b._get_nova_hosting_config = Mock(
            return_value={
                "image_uri": "nova-image",
                "env_vars": {"CONTEXT_LENGTH": "8000", "USER_FLAG": "recipe-default"},
                "instance_type": "ml.p5.48xlarge",
            }
        )
        validate = Mock()
        b._validate_nova_smi_config = validate
        b._fetch_and_cache_recipe_config()
        # Nova image applied (not the generic top-level EcrAddress).
        self.assertEqual(b.image_uri, "nova-image")
        # Nova env precedence: user override wins over the recipe default.
        self.assertEqual(b.env_vars["USER_FLAG"], "keep")
        self.assertEqual(b.env_vars["CONTEXT_LENGTH"], "8000")
        validate.assert_called_once()

    def test_build_resyncs_instance_type_to_explicit_selection(self):
        # If a caller reassigns instance_type AFTER an explicit selection, build re-syncs to the
        # selected bundle's instance type so the endpoint instance matches the applied image/env.
        b = self._builder("ml.g6.4xlarge")
        self._patch_recipe(b, HOSTING_CONFIGS)
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True):
            b.set_deployment_config(instance_type="ml.g6e.48xlarge")
        b.instance_type = "ml.g5.4xlarge"  # contradictory direct reassignment
        b.image_uri = None
        b.env_vars = None
        b._fetch_and_cache_recipe_config()
        self.assertEqual(b.instance_type, "ml.g6e.48xlarge")  # re-synced to the selection
        self.assertEqual(b.env_vars["OPTION_TENSOR_PARALLEL_DEGREE"], "8")

    def test_build_passes_selected_config_to_compute_requirements(self):
        # The SELECTED config (not Default) must reach _resolve_compute_requirements_from_config,
        # and its result is cached — proving compute requirements travel with the selection.
        b = self._builder("ml.g6e.48xlarge")
        self._patch_recipe(b, HOSTING_CONFIGS)
        calls = {}

        def recorder(instance_type, config, user_resource_requirements):
            calls["instance_type"] = instance_type
            calls["config"] = config
            return {"NumberOfAcceleratorDevicesRequired": 8}

        b._resolve_compute_requirements_from_config = recorder
        b.image_uri = None
        b.env_vars = None
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True):
            b.set_deployment_config(instance_type="ml.g6e.48xlarge")
        b._fetch_and_cache_recipe_config()
        self.assertEqual(calls["instance_type"], "ml.g6e.48xlarge")
        self.assertEqual(calls["config"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"], "8")
        self.assertEqual(b._cached_compute_requirements["NumberOfAcceleratorDevicesRequired"], 8)

    def test_build_no_configs_non_nova_raises(self):
        # No hosting configs anywhere and not a Nova model -> clear unsupported error.
        b = self._builder("ml.g6.4xlarge")
        hub = {"RecipeCollection": [{"Name": "r1"}]}  # no recipe-level or top-level configs
        mp = Mock()
        container = Mock()
        container.base_model.recipe_name = "r1"
        container.base_model.hub_content_name = "not-nova"
        mp.inference_specification.containers = [container]
        b._fetch_hub_document_for_custom_model = Mock(return_value=hub)
        b._fetch_model_package = Mock(return_value=mp)
        b.s3_upload_path = "s3://bucket/model"
        with self.assertRaises(ValueError) as ctx:
            b._fetch_and_cache_recipe_config()
        self.assertIn("not supported for deployment", str(ctx.exception))

    def test_returned_config_mutation_does_not_corrupt_build(self):
        # Cross-path isolation: mutating a config returned by get/list must not change what build
        # applies.
        b = self._builder("ml.g6e.48xlarge")
        self._patch_recipe(b, HOSTING_CONFIGS)
        with patch.object(ModelBuilder, "_is_model_customization", return_value=True):
            b.set_deployment_config(instance_type="ml.g6e.48xlarge")
            got = b.get_deployment_config()
            got["DeploymentArgs"]["Environment"]["OPTION_TENSOR_PARALLEL_DEGREE"] = "999"
        b.image_uri = None
        b.env_vars = None
        b._fetch_and_cache_recipe_config()
        self.assertEqual(b.env_vars["OPTION_TENSOR_PARALLEL_DEGREE"], "8")


class TestRecipeHostingConfigHelpers(unittest.TestCase):
    """Direct coverage of the internal helpers behind the unified deployment-config API, plus a
    guard that a base/JumpStart model still routes through the same public entry point — the
    base-vs-fine-tuned split is chosen only by ``_is_model_customization()``, never surfaced.
    """

    def setUp(self):
        self.mock_session = _make_mock_session()

    def _builder(self, model="huggingface-reasoning-qwen3-06b"):
        return ModelBuilder(
            model=model,
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-reasoning-qwen3-06b",
                "CUSTOM_MODEL_VERSION": "3.9.0",
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            # Pin an instance type so construction stays hermetic — without it the constructor
            # auto-detects the instance type via a live JumpStart S3 lookup.
            instance_type="ml.g6.4xlarge",
        )

    # ---- _normalize_hosting_config: stable list/get-friendly shape ----

    def test_normalize_named_default_base_shape(self):
        norm = ModelBuilder._normalize_hosting_config(HOSTING_CONFIGS[0])
        self.assertEqual(norm["DeploymentConfigName"], "Default")
        self.assertTrue(norm["IsDefault"])
        args = norm["DeploymentArgs"]
        self.assertEqual(args["InstanceType"], "ml.g6.4xlarge")
        self.assertEqual(args["ImageUri"].split(":")[-1], "0.34.0-lmi16")
        self.assertEqual(
            args["ComputeResourceRequirements"]["NumberOfAcceleratorDevicesRequired"], 1
        )
        # BenchmarkMetrics/AccelerationConfigs present but None, matching the base sentinel.
        self.assertIsNone(norm["BenchmarkMetrics"])
        self.assertIsNone(norm["AccelerationConfigs"])

    def test_normalize_unnamed_uses_instance_type_identifier(self):
        # Unnamed configs (no Profile) get their instance type as a stable identifier.
        norm = ModelBuilder._normalize_hosting_config(HOSTING_CONFIGS[2])
        self.assertEqual(norm["DeploymentConfigName"], "ml.g6e.48xlarge")
        self.assertFalse(norm["IsDefault"])
        self.assertEqual(norm["DeploymentArgs"]["InstanceType"], "ml.g6e.48xlarge")

    def test_normalize_defaults_missing_fields(self):
        norm = ModelBuilder._normalize_hosting_config({"InstanceType": "ml.g6.4xlarge"})
        args = norm["DeploymentArgs"]
        self.assertEqual(args["Environment"], {})
        self.assertEqual(args["ComputeResourceRequirements"], {})
        self.assertIsNone(args["ImageUri"])

    # ---- _resolve_recipe_hosting_configs: recipe match, top-level fallback, no-configs raise ----

    def _patch_recipe(self, b, hub_document, recipe_name):
        mp = Mock()
        container = Mock()
        container.base_model.recipe_name = recipe_name
        mp.inference_specification.containers = [container]
        b._fetch_hub_document_for_custom_model = Mock(return_value=hub_document)
        b._fetch_model_package = Mock(return_value=mp)

    def test_resolve_from_recipe_collection(self):
        b = self._builder()
        hub = {"RecipeCollection": [{"Name": "r1", "HostingConfigs": HOSTING_CONFIGS}]}
        self._patch_recipe(b, hub, "r1")
        self.assertEqual(b._resolve_recipe_hosting_configs(), HOSTING_CONFIGS)

    def test_resolve_falls_back_to_top_level(self):
        # Recipe entry matches but publishes no HostingConfigs (missing key) -> fall back to top.
        b = self._builder()
        hub = {"RecipeCollection": [{"Name": "r1"}], "HostingConfigs": HOSTING_CONFIGS}
        self._patch_recipe(b, hub, "r1")
        self.assertEqual(b._resolve_recipe_hosting_configs(), HOSTING_CONFIGS)

    def test_resolve_recipe_empty_list_falls_back_to_top_level(self):
        # Recipe entry matches with an EMPTY HostingConfigs list (not just a missing key) -> top.
        b = self._builder()
        hub = {
            "RecipeCollection": [{"Name": "r1", "HostingConfigs": []}],
            "HostingConfigs": HOSTING_CONFIGS,
        }
        self._patch_recipe(b, hub, "r1")
        self.assertEqual(b._resolve_recipe_hosting_configs(), HOSTING_CONFIGS)

    def test_resolve_unmatched_recipe_name_falls_back_to_top_level(self):
        # The model's recipe_name matches no RecipeCollection entry -> top-level fallback (does not
        # accidentally pick another recipe's configs).
        b = self._builder()
        other = [{"InstanceType": "ml.other.xlarge", "Environment": {}}]
        hub = {
            "RecipeCollection": [{"Name": "some-other-recipe", "HostingConfigs": other}],
            "HostingConfigs": HOSTING_CONFIGS,
        }
        self._patch_recipe(b, hub, "r1")  # model's recipe_name is r1, unmatched
        self.assertEqual(b._resolve_recipe_hosting_configs(), HOSTING_CONFIGS)

    def test_extract_hosting_configs_from_hub_precedence(self):
        # Direct coverage of the single-source-of-truth discovery helper used by BOTH the selection
        # API and the build path.
        rc = [{"InstanceType": "ml.recipe.xlarge"}]
        top = [{"InstanceType": "ml.top.xlarge"}]
        # recipe-level present -> recipe-level wins
        self.assertEqual(
            ModelBuilder._extract_hosting_configs_from_hub(
                {"RecipeCollection": [{"Name": "r1", "HostingConfigs": rc}], "HostingConfigs": top},
                "r1",
            ),
            rc,
        )
        # recipe matched but empty -> top-level
        self.assertEqual(
            ModelBuilder._extract_hosting_configs_from_hub(
                {"RecipeCollection": [{"Name": "r1", "HostingConfigs": []}], "HostingConfigs": top},
                "r1",
            ),
            top,
        )
        # recipe unmatched -> top-level
        self.assertEqual(
            ModelBuilder._extract_hosting_configs_from_hub(
                {
                    "RecipeCollection": [{"Name": "other", "HostingConfigs": rc}],
                    "HostingConfigs": top,
                },
                "r1",
            ),
            top,
        )
        # neither -> empty
        self.assertEqual(
            ModelBuilder._extract_hosting_configs_from_hub({"RecipeCollection": []}, "r1"), []
        )

    def test_resolve_no_configs_raises(self):
        b = self._builder()
        self._patch_recipe(b, {"RecipeCollection": []}, "r1")
        with self.assertRaises(ValueError) as ctx:
            b._resolve_recipe_hosting_configs()
        self.assertIn("does not publish any hosting configurations", str(ctx.exception))

    def test_resolve_no_container_raises_clean_error(self):
        # A model package with no inference container must raise a clear ValueError, not a raw
        # AttributeError/IndexError from containers[0].
        b = self._builder()
        b._fetch_hub_document_for_custom_model = Mock(return_value={"RecipeCollection": []})
        mp = Mock()
        mp.inference_specification.containers = []
        b._fetch_model_package = Mock(return_value=mp)
        with self.assertRaises(ValueError) as ctx:
            b._resolve_recipe_hosting_configs()
        self.assertIn("no inference container", str(ctx.exception))

    def test_normalized_shape_covers_base_deployment_config_keys(self):
        # SHAPE PARITY (the whole point of normalization): the normalized fine-tuned config must
        # expose at least every key the real base/JumpStart response emits, so a caller can index
        # the same keys on either pathway. Tie the assertion to the actual base dataclasses so it
        # fails loudly if base ever adds a field we don't mirror.
        from sagemaker.core.jumpstart.types import DeploymentArgs, DeploymentConfigMetadata

        def pascal(s):
            return s.replace("_", " ").title().replace(" ", "")

        base_top_keys = {pascal(s) for s in DeploymentConfigMetadata.__slots__}
        base_args_keys = {pascal(s) for s in DeploymentArgs.__slots__}

        norm = ModelBuilder._normalize_hosting_config(HOSTING_CONFIGS[0])
        self.assertTrue(
            base_top_keys <= set(norm),
            f"normalized top-level missing base keys: {base_top_keys - set(norm)}",
        )
        self.assertTrue(
            base_args_keys <= set(norm["DeploymentArgs"]),
            f"normalized DeploymentArgs missing base keys: "
            f"{base_args_keys - set(norm['DeploymentArgs'])}",
        )

    # ---- base-model routing through the SAME unified entry point ----

    def test_list_deployment_configs_routes_to_base_for_string_model(self):
        # For a base/JumpStart string model, list_deployment_configs must take the base branch
        # (never the recipe resolver). With no instance_type it returns every config materialized
        # at its default (the original base API).
        b = self._builder()
        base_configs = [
            {"DeploymentConfigName": "lmi", "DeploymentArgs": {"InstanceType": "ml.g5.2xlarge"}},
            {"DeploymentConfigName": "tgi", "DeploymentArgs": {"InstanceType": "ml.g6.4xlarge"}},
        ]
        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True), patch.object(
            ModelBuilder, "_get_deployment_configs", return_value=["sentinel"]
        ), patch.object(
            ModelBuilder, "deployment_config_response_data", return_value=base_configs
        ), patch.object(
            ModelBuilder, "_resolve_recipe_hosting_configs"
        ) as resolve_recipe:
            self.assertEqual(len(b.list_deployment_configs()), 2)
            # The fine-tuned resolver is never touched on the base path.
            resolve_recipe.assert_not_called()

    def test_list_deployment_configs_base_filter_keeps_supported_non_default_instance(self):
        # Regression for the base-filter bug (reviewer P1): a base config is a MULTI-instance
        # bundle. list_deployment_configs(instance_type=X) must return a config that SUPPORTS X even
        # when X is not its default — materialized FOR X — instead of discarding it. Exercised via
        # the supported-instance metadata + the real per-config materialization loop.
        b = self._builder()
        # "big" supports the requested ml.g5.12xlarge but DEFAULTS to ml.g5.2xlarge; "small" does
        # not support it at all and must be filtered out.
        meta_big = Mock(
            resolved_config={
                "default_inference_instance_type": "ml.g5.2xlarge",
                "supported_inference_instance_types": ["ml.g5.2xlarge", "ml.g5.12xlarge"],
            }
        )
        meta_small = Mock(
            resolved_config={
                "default_inference_instance_type": "ml.m5.large",
                "supported_inference_instance_types": ["ml.m5.large"],
            }
        )
        b._metadata_configs = {"big": meta_big, "small": meta_small}

        def _fake_get_configs(selected_config_name, selected_instance_type):
            # Mimic _get_deployment_configs: the SELECTED config is materialized at the requested
            # instance; every config is returned.
            return [
                {
                    "DeploymentConfigName": name,
                    "DeploymentArgs": {
                        "InstanceType": (
                            selected_instance_type
                            if name == selected_config_name
                            else "DEFAULT"
                        )
                    },
                }
                for name in ("big", "small")
            ]

        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True), patch.object(
            ModelBuilder, "_ensure_metadata_configs"
        ), patch.object(
            ModelBuilder, "_get_deployment_configs", side_effect=_fake_get_configs
        ), patch.object(
            ModelBuilder,
            "deployment_config_response_data",
            side_effect=lambda configs: configs,
        ):
            result = b.list_deployment_configs(instance_type="ml.g5.12xlarge")

        # Only the supporting config is returned, materialized FOR the requested instance.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["DeploymentConfigName"], "big")
        self.assertEqual(result[0]["DeploymentArgs"]["InstanceType"], "ml.g5.12xlarge")

    def test_get_deployment_configs_honors_requested_instance_for_selected(self):
        # Exercises the REAL _get_deployment_configs path (not a mocked deployment_config_response_
        # data). The SELECTED config must be materialized with the REQUESTED instance type — even
        # when that differs from its default — while other configs keep their defaults, none dropped.
        from sagemaker.serve import model_builder_utils as mbu

        b = self._builder()
        b.image_uri = "img"
        meta_lmi = Mock(
            benchmark_metrics=None,
            config_components={},
            resolved_config={"default_inference_instance_type": "ml.g5.2xlarge"},
        )
        meta_tgi = Mock(
            benchmark_metrics=None,
            config_components={},
            resolved_config={"default_inference_instance_type": "ml.g5.2xlarge"},
        )
        b._metadata_configs = {"lmi": meta_lmi, "tgi": meta_tgi}

        captured = {}

        def _capture_init_kwargs(**kwargs):
            captured[kwargs["config_name"]] = kwargs["instance_type"]
            return Mock()

        with patch.object(mbu, "get_init_kwargs", side_effect=_capture_init_kwargs), patch.object(
            mbu, "get_deploy_kwargs", return_value=Mock()
        ), patch.object(mbu, "DeploymentConfigMetadata", side_effect=lambda *a, **k: Mock()):
            configs = b._get_deployment_configs("lmi", "ml.g5.12xlarge")

        # Both configs materialized (none dropped).
        self.assertEqual(len(configs), 2)
        # Selected config uses the REQUESTED instance; the other keeps its default.
        self.assertEqual(captured["lmi"], "ml.g5.12xlarge")
        self.assertEqual(captured["tgi"], "ml.g5.2xlarge")

    def test_set_deployment_config_base_requires_instance_type(self):
        # Base/JumpStart setter: config_name alone is not enough — instance_type is still required.
        # Guards against the optional-signature (added for the fine-tuned path) silently forwarding
        # None into the base selection machinery instead of failing fast.
        b = self._builder()
        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                b.set_deployment_config(config_name="lmi")
        self.assertIn("instance_type is required", str(ctx.exception))

    def test_list_deployment_configs_base_non_jumpstart_raises(self):
        # A non-customization, non-JumpStart string model still hits the base guard.
        b = self._builder()
        with patch.object(
            ModelBuilder, "_is_model_customization", return_value=False
        ), patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=False), patch.object(
            ModelBuilder, "_use_jumpstart_equivalent", return_value=False
        ):
            with self.assertRaises(ValueError) as ctx:
                b.list_deployment_configs()
        self.assertIn("JumpStart", str(ctx.exception))

    def test_is_nova_model_tolerates_non_string_attrs(self):
        # Regression: _is_nova_model (consulted first on the model-customization build path) must
        # not raise when a partially-populated model package leaves recipe_name/hub_content_name as
        # a non-string (e.g. an unset Mock attribute). "nova" in <non-str> would raise TypeError.
        b = self._builder()
        pkg = Mock()
        container = Mock()
        container.base_model.recipe_name = "test-recipe"  # non-Nova string
        # hub_content_name intentionally left as an auto Mock (not a string).
        pkg.inference_specification.containers = [container]
        with patch.object(ModelBuilder, "_fetch_model_package", return_value=pkg):
            self.assertFalse(b._is_nova_model())  # returns False instead of raising


# A matrix of ADVERSARIAL raw HostingConfigs sets that deliberately covers the shapes the earlier
# hand-picked fixture never did — most importantly configs whose default instance DIFFERS from an
# instance they offer via SupportedInstanceTypes (the exact case that hid the get() bug), plus
# DefaultInstanceType-only entries, ambiguous overlaps, and a Default profile with a supported
# superset. The invariant tests below run EVERY scenario against ALL of list/set/get so a
# disagreement between the methods surfaces — coverage that can't be tuned to the implementation.
_ADVERSARIAL_CONFIG_SETS = {
    "single_instance_bundles": [
        {"Profile": "Default", "InstanceType": "ml.g6.4xlarge", "Environment": {"T": "d"}},
        {"InstanceType": "ml.g6e.48xlarge", "Environment": {"T": "a"}},
    ],
    "default_instance_type_only": [
        {"DefaultInstanceType": "ml.g6.4xlarge", "Environment": {"T": "d"}},
        {"DefaultInstanceType": "ml.g6e.48xlarge", "Environment": {"T": "a"}},
    ],
    "supported_superset_differs_from_default": [
        {"Profile": "Default", "InstanceType": "ml.g6.4xlarge", "Environment": {"T": "d"}},
        {
            "DefaultInstanceType": "ml.g5.2xlarge",
            "SupportedInstanceTypes": ["ml.g5.2xlarge", "ml.g5.12xlarge"],
            "Environment": {"T": "m"},
        },
    ],
    "supported_not_including_primary": [
        {
            "InstanceType": "ml.g6.4xlarge",
            "SupportedInstanceTypes": ["ml.p5.48xlarge"],
            "Environment": {"T": "s"},
        },
    ],
    "default_profile_with_supported_superset": [
        {
            "Profile": "Default",
            "InstanceType": "ml.g6.4xlarge",
            "SupportedInstanceTypes": ["ml.g6.4xlarge", "ml.g5.12xlarge"],
            "Environment": {"T": "d"},
        },
        {"InstanceType": "ml.p5.48xlarge", "Environment": {"T": "a"}},
    ],
    "ambiguous_overlap": [
        {"InstanceType": "ml.g6.4xlarge", "SupportedInstanceTypes": ["ml.g5.12xlarge"]},
        {"InstanceType": "ml.p5.48xlarge", "SupportedInstanceTypes": ["ml.g5.12xlarge"]},
    ],
    "many_mixed": [
        {"Profile": "Default", "InstanceType": "ml.g6.4xlarge", "Environment": {"T": "d"}},
        {"DefaultInstanceType": "ml.g5.2xlarge", "Environment": {"T": "a"}},
        {
            "InstanceType": "ml.g6e.48xlarge",
            "SupportedInstanceTypes": ["ml.g6e.48xlarge", "ml.p5.48xlarge"],
            "Environment": {"T": "b"},
        },
    ],
}


class TestDeploymentConfigInvariants(unittest.TestCase):
    """Cross-method invariants for the unified deployment-config API, run over an adversarial matrix.

    These exist because the earlier suite was superficial in two ways that let a real bug ship:
    (1) every method was tested in ISOLATION, so a DISAGREEMENT between list/set/get could not be
    seen; and (2) the single hand-picked fixture had one instance per config (default == the
    requested instance), so the materialization step was always a no-op and a missing
    materialization in get_deployment_config() was invisible.

    Each test asserts an invariant that must hold for EVERY scenario in `_ADVERSARIAL_CONFIG_SETS`
    — including `supported_superset_differs_from_default`, the exact shape the shipped get() bug
    got wrong. The round-trip test fails against that bug; the earlier per-method tests could not.
    (No external test dependency — deterministic table-driven checks; hypothesis is not a declared
    dependency of this package.)
    """

    def setUp(self):
        self.mock_session = _make_mock_session()

    def _builder(self, configs):
        # A fine-tuned-looking builder over `configs`. Patches stay active for the caller (started
        # here, stopped via addCleanup) so list/get/set all see the same configs.
        b = ModelBuilder(
            model="huggingface-reasoning-qwen3-06b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-reasoning-qwen3-06b",
                "CUSTOM_MODEL_VERSION": "3.9.0",
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.g6.4xlarge",
        )
        p1 = patch.object(ModelBuilder, "_is_model_customization", return_value=True)
        p2 = patch.object(ModelBuilder, "_resolve_recipe_hosting_configs", return_value=configs)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        return b

    @staticmethod
    def _offered_counts(configs):
        counts = {}
        for cfg in configs:
            for inst in ModelBuilder._raw_config_offered_instances(cfg):
                counts[inst] = counts.get(inst, 0) + 1
        return counts

    _DEPLOYMENT_ARGS_KEYS = {
        "ImageUri",
        "InstanceType",
        "Environment",
        "ComputeResourceRequirements",
        "ModelData",
        "ModelPackageArn",
        "ModelDataDownloadTimeout",
        "ContainerStartupHealthCheckTimeout",
        "AdditionalDataSources",
    }

    def test_list_shape_invariant(self):
        # INVARIANT: every listed config exposes the full normalized shape, and exactly one config
        # is flagged IsDefault iff a "Default" profile exists.
        for name, configs in _ADVERSARIAL_CONFIG_SETS.items():
            with self.subTest(scenario=name):
                b = self._builder(configs)
                listed = b.list_deployment_configs()
                self.assertEqual(len(listed), len(configs))
                for c in listed:
                    self.assertIn("DeploymentConfigName", c)
                    self.assertIn("BenchmarkMetrics", c)
                    self.assertIn("AccelerationConfigs", c)
                    self.assertIn("IsDefault", c)
                    self.assertEqual(set(c["DeploymentArgs"]), self._DEPLOYMENT_ARGS_KEYS)
                num_default = sum(1 for c in listed if c["IsDefault"])
                expected_default = sum(1 for cfg in configs if cfg.get("Profile") == "Default")
                self.assertEqual(num_default, expected_default)

    def test_list_filter_sound_and_complete(self):
        # INVARIANT: for every offered instance X, list(instance_type=X) returns EXACTLY the configs
        # that offer X (completeness), each materialized FOR X (soundness). This is the invariant the
        # base-filter bug violated (a supported non-default config was dropped).
        for name, configs in _ADVERSARIAL_CONFIG_SETS.items():
            with self.subTest(scenario=name):
                b = self._builder(configs)
                for x, count in self._offered_counts(configs).items():
                    filtered = b.list_deployment_configs(instance_type=x)
                    self.assertEqual(len(filtered), count, f"{name}/{x}")
                    for c in filtered:
                        self.assertEqual(c["DeploymentArgs"]["InstanceType"], x)
                self.assertEqual(
                    b.list_deployment_configs(instance_type="ml.not-offered.xlarge"), []
                )

    def test_list_set_get_round_trip(self):
        # INVARIANT (the one that would have caught the shipped bug): for any UNAMBIGUOUS offered
        # instance X, set(instance_type=X) then get() must agree with list(instance_type=X) on BOTH
        # DeploymentArgs.InstanceType (== X) and DeploymentConfigName. Fresh builder per X.
        for name, configs in _ADVERSARIAL_CONFIG_SETS.items():
            counts = self._offered_counts(configs)
            for x, count in counts.items():
                if count != 1:
                    continue  # ambiguous instances covered by the ambiguity invariant
                with self.subTest(scenario=name, instance=x):
                    b = self._builder(configs)
                    listed = b.list_deployment_configs(instance_type=x)
                    self.assertEqual(len(listed), 1)
                    b.set_deployment_config(instance_type=x)
                    got = b.get_deployment_config()
                    self.assertEqual(got["DeploymentArgs"]["InstanceType"], x)
                    self.assertEqual(
                        got["DeploymentConfigName"], listed[0]["DeploymentConfigName"]
                    )

    def test_ambiguous_instance_rejected_by_set(self):
        # INVARIANT: an instance offered by MORE THAN ONE config must be rejected by set() rather
        # than silently picking one.
        for name, configs in _ADVERSARIAL_CONFIG_SETS.items():
            counts = self._offered_counts(configs)
            for x, count in counts.items():
                if count < 2:
                    continue
                with self.subTest(scenario=name, instance=x):
                    b = self._builder(configs)
                    with self.assertRaises(ValueError) as ctx:
                        b.set_deployment_config(instance_type=x)
                    self.assertIn("ambiguous", str(ctx.exception))

    def test_returned_configs_are_isolated_from_internal_state(self):
        # INVARIANT: mutating any dict returned by list()/get() must not affect a later call —
        # returned data is always freshly built / copied.
        for name, configs in _ADVERSARIAL_CONFIG_SETS.items():
            with self.subTest(scenario=name):
                b = self._builder(configs)
                first = b.list_deployment_configs()
                for c in first:
                    c["DeploymentConfigName"] = "MUTATED"
                    c["DeploymentArgs"]["Environment"]["T"] = "MUTATED"
                second = b.list_deployment_configs()
                self.assertTrue(all(c["DeploymentConfigName"] != "MUTATED" for c in second))
                self.assertTrue(
                    all(c["DeploymentArgs"]["Environment"].get("T") != "MUTATED" for c in second)
                )


if __name__ == "__main__":
    unittest.main()
