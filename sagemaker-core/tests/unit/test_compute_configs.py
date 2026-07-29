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
