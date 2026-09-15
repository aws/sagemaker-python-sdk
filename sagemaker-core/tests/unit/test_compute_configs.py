"""Unit tests for Compute and HyperPodCompute config classes."""

import pytest
from sagemaker.core.training.configs import Compute, HyperPodCompute


class TestComputeClass:
    """Test the extended Compute class."""

    def test_basic_construction(self):
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=4)
        assert compute.instance_type == "ml.p5.48xlarge"
        assert compute.instance_count == 4
        assert compute.volume_size_in_gb == 30

    def test_defaults(self):
        compute = Compute(instance_type="ml.m5.xlarge")
        assert compute.volume_size_in_gb == 30
        assert compute.enable_managed_spot_training is None

    def test_to_resource_config_basic(self):
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=4)
        rc = compute._to_resource_config()
        assert rc.instance_type == "ml.p5.48xlarge"
        assert rc.instance_count == 4

    def test_to_resource_config_with_volume(self):
        compute = Compute(
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            volume_size_in_gb=100,
        )
        rc = compute._to_resource_config()
        assert rc.volume_size_in_gb == 100

    def test_to_resource_config_with_warm_pool(self):
        compute = Compute(
            instance_type="ml.p5.48xlarge",
            keep_alive_period_in_seconds=3600,
        )
        rc = compute._to_resource_config()
        assert rc.keep_alive_period_in_seconds == 3600


class TestHyperPodComputeClass:
    """Test the HyperPodCompute config class."""

    def test_defaults(self):
        config = HyperPodCompute()
        assert config.cluster_name == ""
        assert config.namespace == "kubeflow"
        assert config.instance_type is None
        assert config.node_count == 1

    def test_full_construction(self):
        config = HyperPodCompute(
            cluster_name="prod-cluster",
            namespace="training",
            instance_type="ml.p5.48xlarge",
            node_count=8,
        )
        assert config.cluster_name == "prod-cluster"
        assert config.namespace == "training"
        assert config.instance_type == "ml.p5.48xlarge"
        assert config.node_count == 8

    def test_is_not_compute_instance(self):
        """HyperPodCompute is not an instance of Compute (separate class hierarchies)."""
        config = HyperPodCompute(cluster_name="cluster")
        assert not isinstance(config, Compute)

    def test_compute_is_not_hyperpod_instance(self):
        """Compute is not an instance of HyperPodCompute."""
        compute = Compute(instance_type="ml.p5.48xlarge")
        assert not isinstance(compute, HyperPodCompute)


class TestComputeInstancePreferences:
    """Instance Preferences (multi-instance-type) support on the Compute config."""

    def test_training_compute_instance_preferences_round_trip(self):
        from sagemaker.core.shapes.shapes import InstancePreference

        prefs = [
            InstancePreference(instance_type="ml.p5.48xlarge"),
            InstancePreference(instance_type="ml.p4d.24xlarge"),
        ]
        compute = Compute(instance_preferences=prefs, instance_count=2)
        rc = compute._to_resource_config()
        assert [p.instance_type for p in rc.instance_preferences] == [
            "ml.p5.48xlarge",
            "ml.p4d.24xlarge",
        ]
        assert rc.instance_count == 2

    def test_training_compute_per_preference_count(self):
        """A per-preference (unset uniform) count must round-trip without error."""
        from sagemaker.core.shapes.shapes import InstancePreference
        from sagemaker.core.utils.utils import Unassigned

        prefs = [
            InstancePreference(instance_type="ml.p5.48xlarge", instance_count=2),
            InstancePreference(instance_type="ml.p4d.24xlarge", instance_count=4),
        ]
        compute = Compute(instance_preferences=prefs)
        rc = compute._to_resource_config()
        assert rc.instance_preferences[0].instance_count == 2
        assert rc.instance_preferences[1].instance_count == 4

    def test_training_compute_per_preference_training_plan(self):
        from sagemaker.core.shapes.shapes import InstancePreference

        prefs = [
            InstancePreference(
                instance_type="ml.p5.48xlarge",
                training_plan_arns=[
                    "arn:aws:sagemaker:us-west-2:111122223333:training-plan/p5-plan"
                ],
            ),
            InstancePreference(instance_type="ml.p4d.24xlarge"),
        ]
        rc = Compute(instance_preferences=prefs, instance_count=1)._to_resource_config()
        assert rc.instance_preferences[0].training_plan_arns == [
            "arn:aws:sagemaker:us-west-2:111122223333:training-plan/p5-plan"
        ]

    def test_selected_fields_not_sent_on_create(self):
        """selected_instance_type/count are output-only and must not be populated on create."""
        from sagemaker.core.shapes.shapes import InstancePreference
        from sagemaker.core.utils.utils import Unassigned

        rc = Compute(
            instance_preferences=[InstancePreference(instance_type="ml.p5.48xlarge")],
            instance_count=1,
        )._to_resource_config()
        # training Compute filters out None/Unassigned values -> stays Unassigned (not sent)
        assert isinstance(rc.selected_instance_type, Unassigned)
        assert isinstance(rc.selected_instance_count, Unassigned)

    def test_single_type_still_works(self):
        """Classic single-type path is unchanged when instance_preferences is not set."""
        from sagemaker.core.utils.utils import Unassigned

        rc = Compute(instance_type="ml.m5.xlarge", instance_count=1)._to_resource_config()
        assert rc.instance_type == "ml.m5.xlarge"
        assert rc.instance_count == 1
        # instance_preferences is filtered out on create -> stays Unassigned (not sent)
        assert isinstance(rc.instance_preferences, Unassigned)


class TestComputeInstancePreferencesClientValidation:
    """Client-side V1/V4 validation on both Compute classes (server remains
    the source of truth)."""

    @pytest.fixture(params=["training", "modules"])
    def compute_cls(self, request):
        if request.param == "training":
            from sagemaker.core.training.configs import Compute as ComputeCls
        else:
            from sagemaker.core.modules.configs import Compute as ComputeCls
        return ComputeCls

    def test_instance_type_rejected_with_preferences(self, compute_cls):
        from sagemaker.core.shapes.shapes import InstancePreference

        with pytest.raises(ValueError, match="mutually exclusive with instance_type"):
            compute_cls(
                instance_type="ml.m5.xlarge",
                instance_count=1,
                instance_preferences=[InstancePreference(instance_type="ml.m5.xlarge")],
            )

    def test_managed_spot_rejected_with_preferences(self, compute_cls):
        from sagemaker.core.shapes.shapes import InstancePreference

        with pytest.raises(ValueError, match="mutually exclusive with managed spot training"):
            compute_cls(
                instance_count=1,
                enable_managed_spot_training=True,
                instance_preferences=[InstancePreference(instance_type="ml.m5.xlarge")],
            )

    def test_managed_spot_false_allowed_with_preferences(self, compute_cls):
        from sagemaker.core.shapes.shapes import InstancePreference

        compute = compute_cls(
            instance_count=1,
            enable_managed_spot_training=False,
            instance_preferences=[InstancePreference(instance_type="ml.m5.xlarge")],
        )
        assert compute.enable_managed_spot_training is False

    def test_uniform_count_rejected_with_per_preference_counts(self, compute_cls):
        from sagemaker.core.shapes.shapes import InstancePreference

        with pytest.raises(ValueError, match="top-level instance_count and per-preference"):
            compute_cls(
                instance_count=1,
                instance_preferences=[
                    InstancePreference(instance_type="ml.m5.xlarge", instance_count=2),
                    InstancePreference(instance_type="ml.m4.xlarge"),
                ],
            )

    def test_partial_per_preference_counts_rejected(self, compute_cls):
        from sagemaker.core.shapes.shapes import InstancePreference

        with pytest.raises(ValueError, match="every element"):
            compute_cls(
                instance_preferences=[
                    InstancePreference(instance_type="ml.m5.xlarge", instance_count=2),
                    InstancePreference(instance_type="ml.m4.xlarge"),
                ],
            )

    def test_no_count_at_all_rejected(self, compute_cls):
        from sagemaker.core.shapes.shapes import InstancePreference

        with pytest.raises(ValueError, match="every element"):
            compute_cls(
                instance_preferences=[InstancePreference(instance_type="ml.m5.xlarge")],
            )

    def test_duplicate_instance_types_rejected(self, compute_cls):
        from sagemaker.core.shapes.shapes import InstancePreference

        with pytest.raises(ValueError, match="duplicate instance types"):
            compute_cls(
                instance_count=1,
                instance_preferences=[
                    InstancePreference(instance_type="ml.m5.xlarge"),
                    InstancePreference(instance_type="ml.m5.xlarge"),
                ],
            )

    def test_whole_job_plan_rejected_with_per_preference_plans(self, compute_cls):
        """V8: whole-job training_plan_arn XOR per-preference training_plan_arns."""
        from sagemaker.core.shapes.shapes import InstancePreference

        with pytest.raises(ValueError, match="training_plan_arn and per-preference"):
            compute_cls(
                instance_count=1,
                training_plan_arn=(
                    "arn:aws:sagemaker:us-west-2:111122223333:training-plan/whole-job"
                ),
                instance_preferences=[
                    InstancePreference(
                        instance_type="ml.p5.48xlarge",
                        training_plan_arns=[
                            "arn:aws:sagemaker:us-west-2:111122223333:training-plan/p5"
                        ],
                    ),
                    InstancePreference(instance_type="ml.p4d.24xlarge"),
                ],
            )

    def test_whole_job_plan_allowed_without_per_preference_plans(self, compute_cls):
        """A whole-job training_plan_arn with NO per-preference plans is valid."""
        from sagemaker.core.shapes.shapes import InstancePreference

        compute_cls(
            instance_count=1,
            training_plan_arn=("arn:aws:sagemaker:us-west-2:111122223333:training-plan/whole-job"),
            instance_preferences=[
                InstancePreference(instance_type="ml.p5.48xlarge"),
                InstancePreference(instance_type="ml.p4d.24xlarge"),
            ],
        )


class TestModulesComputeInstancePreferences:
    """Instance Preferences support on the modules.configs Compute class."""

    def test_modules_compute_instance_preferences_round_trip(self):
        from sagemaker.core.modules.configs import Compute as ModulesCompute
        from sagemaker.core.shapes.shapes import InstancePreference

        prefs = [
            InstancePreference(instance_type="ml.p5.48xlarge"),
            InstancePreference(instance_type="ml.p4d.24xlarge"),
        ]
        rc = ModulesCompute(instance_preferences=prefs, instance_count=2)._to_resource_config()
        assert [p.instance_type for p in rc.instance_preferences] == [
            "ml.p5.48xlarge",
            "ml.p4d.24xlarge",
        ]

    def test_modules_compute_single_type_still_works(self):
        from sagemaker.core.modules.configs import Compute as ModulesCompute

        rc = ModulesCompute(instance_type="ml.m5.xlarge", instance_count=1)._to_resource_config()
        assert rc.instance_type == "ml.m5.xlarge"


class TestProcessingClusterConfigInstancePreferences:
    """InstancePreferences on the ProcessingClusterConfig shape."""

    def test_processing_cluster_config_accepts_instance_preferences(self):
        from sagemaker.core.shapes.shapes import (
            ProcessingClusterConfig,
            ProcessingInstancePreference,
        )
        from sagemaker.core.utils.utils import Unassigned

        pcc = ProcessingClusterConfig(
            instance_preferences=[
                ProcessingInstancePreference(instance_type="ml.m5.4xlarge"),
                ProcessingInstancePreference(instance_type="ml.m5.2xlarge"),
            ],
            volume_size_in_gb=100,
        )
        assert [p.instance_type for p in pcc.instance_preferences] == [
            "ml.m5.4xlarge",
            "ml.m5.2xlarge",
        ]
        # instance_type / instance_count are now optional (mutually exclusive with prefs)
        assert isinstance(pcc.instance_type, Unassigned)
        assert isinstance(pcc.instance_count, Unassigned)

    def test_processing_instance_preference_has_no_training_plan_arns(self):
        """Processing preferences are a separate shape without training plans (training-only)."""
        import pydantic
        import pytest
        from sagemaker.core.shapes.shapes import ProcessingInstancePreference

        with pytest.raises(pydantic.ValidationError):
            ProcessingInstancePreference(
                instance_type="ml.m5.4xlarge",
                training_plan_arns=["arn:aws:sagemaker:us-west-2:111122223333:training-plan/p"],
            )

    def test_processing_cluster_config_single_type_still_works(self):
        from sagemaker.core.shapes.shapes import ProcessingClusterConfig

        pcc = ProcessingClusterConfig(
            instance_type="ml.m5.xlarge", instance_count=1, volume_size_in_gb=30
        )
        assert pcc.instance_type == "ml.m5.xlarge"
        assert pcc.instance_count == 1
