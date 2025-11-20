"""Unit tests for sagemaker.serve.configs module."""
import unittest
from sagemaker.serve.configs import Network, Compute


class TestNetwork(unittest.TestCase):
    """Test cases for Network dataclass."""

    def test_network_default_values(self):
        """Test Network with default values."""
        network = Network()
        self.assertIsNone(network.subnets)
        self.assertIsNone(network.security_group_ids)
        self.assertFalse(network.enable_network_isolation)
        self.assertIsNone(network.vpc_config)

    def test_network_with_subnets(self):
        """Test Network with subnets."""
        subnets = ["subnet-123", "subnet-456"]
        network = Network(subnets=subnets)
        self.assertEqual(network.subnets, subnets)
        self.assertIsNone(network.security_group_ids)

    def test_network_with_security_groups(self):
        """Test Network with security groups."""
        sg_ids = ["sg-123", "sg-456"]
        network = Network(security_group_ids=sg_ids)
        self.assertEqual(network.security_group_ids, sg_ids)

    def test_network_with_isolation_enabled(self):
        """Test Network with network isolation enabled."""
        network = Network(enable_network_isolation=True)
        self.assertTrue(network.enable_network_isolation)

    def test_network_with_vpc_config(self):
        """Test Network with VPC config."""
        vpc_config = {
            "Subnets": ["subnet-123"],
            "SecurityGroupIds": ["sg-123"]
        }
        network = Network(vpc_config=vpc_config)
        self.assertEqual(network.vpc_config, vpc_config)

    def test_network_with_all_parameters(self):
        """Test Network with all parameters set."""
        subnets = ["subnet-123"]
        sg_ids = ["sg-456"]
        vpc_config = {"Subnets": subnets, "SecurityGroupIds": sg_ids}
        
        network = Network(
            subnets=subnets,
            security_group_ids=sg_ids,
            enable_network_isolation=True,
            vpc_config=vpc_config
        )
        
        self.assertEqual(network.subnets, subnets)
        self.assertEqual(network.security_group_ids, sg_ids)
        self.assertTrue(network.enable_network_isolation)
        self.assertEqual(network.vpc_config, vpc_config)


class TestCompute(unittest.TestCase):
    """Test cases for Compute dataclass."""

    def test_compute_with_instance_type(self):
        """Test Compute with instance type."""
        compute = Compute(instance_type="ml.m5.xlarge")
        self.assertEqual(compute.instance_type, "ml.m5.xlarge")
        self.assertEqual(compute.instance_count, 1)

    def test_compute_with_custom_instance_count(self):
        """Test Compute with custom instance count."""
        compute = Compute(instance_type="ml.m5.xlarge", instance_count=3)
        self.assertEqual(compute.instance_type, "ml.m5.xlarge")
        self.assertEqual(compute.instance_count, 3)

    def test_compute_with_none_instance_type(self):
        """Test Compute with None instance type."""
        compute = Compute(instance_type=None)
        self.assertIsNone(compute.instance_type)
        self.assertEqual(compute.instance_count, 1)

    def test_compute_default_instance_count(self):
        """Test Compute default instance count is 1."""
        compute = Compute(instance_type="ml.g4dn.xlarge")
        self.assertEqual(compute.instance_count, 1)


if __name__ == "__main__":
    unittest.main()
