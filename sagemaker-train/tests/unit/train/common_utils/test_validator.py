import pytest
from unittest.mock import Mock, patch

from sagemaker.train.common_utils.validator import validate_hyperpod_compute


class TestValidateHyperpodCompute:
    """Test cases for HyperPod compute validation."""

    def _make_compute(self, cluster_name="test-cluster", instance_type="ml.p5.48xlarge", node_count=4):
        """Helper to create a mock HyperPodCompute object."""
        compute = Mock()
        compute.cluster_name = cluster_name
        compute.instance_type = instance_type
        compute.node_count = node_count
        return compute

    def _make_session(self, region_name="us-east-1"):
        """Helper to create a mock sagemaker_session."""
        session = Mock()
        session.boto_session.region_name = region_name
        return session

    # --- Cluster name validation ---

    def test_raises_value_error_when_cluster_name_is_none(self):
        compute = self._make_compute(cluster_name=None)
        session = self._make_session()

        with pytest.raises(ValueError, match="HyperPod cluster name is required"):
            validate_hyperpod_compute(compute, session)

    def test_raises_value_error_when_cluster_name_is_empty(self):
        compute = self._make_compute(cluster_name="")
        session = self._make_session()

        with pytest.raises(ValueError, match="HyperPod cluster name is required"):
            validate_hyperpod_compute(compute, session)

    # --- describe_cluster permission errors ---

    def test_raises_permission_error_on_access_denied(self):
        compute = self._make_compute()
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.side_effect = Exception("AccessDenied: User is not authorized")
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(PermissionError, match="sagemaker:DescribeCluster required"):
            validate_hyperpod_compute(compute, session)

    def test_raises_permission_error_on_unauthorized_operation(self):
        compute = self._make_compute()
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.side_effect = Exception("UnauthorizedOperation")
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(PermissionError, match="sagemaker:DescribeCluster required"):
            validate_hyperpod_compute(compute, session)

    def test_raises_runtime_error_on_other_describe_failure(self):
        compute = self._make_compute()
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.side_effect = Exception("InternalServerError")
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(RuntimeError, match="Failed to describe cluster 'test-cluster'"):
            validate_hyperpod_compute(compute, session)

    # --- Instance type not available (non-Nova / normal instance groups) ---

    def test_raises_value_error_when_instance_type_not_in_normal_groups(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker-group-1",
                    "InstanceType": "ml.g5.12xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
                {
                    "InstanceGroupName": "worker-group-2",
                    "InstanceType": "ml.g5.24xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 2,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError, match="Instance type 'ml.p5.48xlarge' not available") as exc_info:
            validate_hyperpod_compute(compute, session, is_nova=False)

        assert "ml.g5.12xlarge" in str(exc_info.value)
        assert "ml.g5.24xlarge" in str(exc_info.value)

    # --- Instance type not available (Nova / restricted instance groups) ---

    def test_raises_value_error_when_instance_type_not_in_restricted_groups(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "restricted-group-1",
                    "InstanceType": "ml.g5.12xlarge",
                    "CurrentCount": 8,
                    "TargetCount": 8,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError, match="Instance type 'ml.p5.48xlarge' not available") as exc_info:
            validate_hyperpod_compute(compute, session, is_nova=True)

        assert "restricted instance groups" in str(exc_info.value)
        assert "ml.g5.12xlarge" in str(exc_info.value)

    # --- Insufficient capacity ---

    def test_raises_value_error_when_insufficient_capacity_in_normal_groups(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=8)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 8,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError, match="Insufficient capacity") as exc_info:
            validate_hyperpod_compute(compute, session, is_nova=False)

        assert "Required: 8" in str(exc_info.value)
        assert "Maximum available: 4" in str(exc_info.value)

    def test_raises_value_error_when_insufficient_capacity_in_restricted_groups(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "restricted-group",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 1,
                    "TargetCount": 4,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError, match="Insufficient capacity") as exc_info:
            validate_hyperpod_compute(compute, session, is_nova=True)

        assert "Required: 4" in str(exc_info.value)
        assert "Maximum available: 1" in str(exc_info.value)

    # --- Successful validation ---

    def test_succeeds_with_matching_instance_type_and_sufficient_capacity(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Should not raise
        validate_hyperpod_compute(compute, session, is_nova=False)

    def test_succeeds_nova_with_restricted_groups(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "restricted-worker-group",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 8,
                    "TargetCount": 8,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Should not raise
        validate_hyperpod_compute(compute, session, is_nova=True)

    def test_succeeds_with_excess_capacity(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=2)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 16,
                    "TargetCount": 16,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Should not raise
        validate_hyperpod_compute(compute, session, is_nova=False)

    # --- node_count defaults to 1 when None ---

    def test_succeeds_with_node_count_none_and_single_instance_available(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=None)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 1,
                    "TargetCount": 1,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Should not raise - node_count defaults to 1
        validate_hyperpod_compute(compute, session, is_nova=False)

    # --- Multiple instance groups ---

    def test_succeeds_when_matching_group_among_multiple_groups(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "group-1",
                    "InstanceType": "ml.g5.12xlarge",
                    "CurrentCount": 8,
                    "TargetCount": 8,
                    "Status": "InService",
                },
                {
                    "InstanceGroupName": "group-2",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
                {
                    "InstanceGroupName": "group-3",
                    "InstanceType": "ml.g5.24xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 2,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Should not raise
        validate_hyperpod_compute(compute, session, is_nova=False)

    def test_picks_max_capacity_across_multiple_compatible_groups(self):
        """When multiple groups have the same instance type, capacity check uses any group."""
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=6)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "group-a",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 4,
                    "Status": "InService",
                },
                {
                    "InstanceGroupName": "group-b",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 8,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Max available across compatible groups is 4, required is 6
        with pytest.raises(ValueError, match="Insufficient capacity") as exc_info:
            validate_hyperpod_compute(compute, session, is_nova=False)

        assert "Required: 6" in str(exc_info.value)
        assert "Maximum available: 4" in str(exc_info.value)

    def test_succeeds_when_one_of_multiple_compatible_groups_has_capacity(self):
        """When multiple groups have the same instance type, succeeds if any has enough."""
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "group-a",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 4,
                    "Status": "InService",
                },
                {
                    "InstanceGroupName": "group-b",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 8,
                    "TargetCount": 8,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # group-b has enough capacity (8 >= 4)
        validate_hyperpod_compute(compute, session, is_nova=False)

    # --- Empty instance groups ---

    def test_raises_value_error_when_no_instance_groups_exist(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError, match="Instance type 'ml.p5.48xlarge' not available"):
            validate_hyperpod_compute(compute, session, is_nova=False)

    def test_raises_value_error_when_no_restricted_groups_exist(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=4)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError, match="Instance type 'ml.p5.48xlarge' not available"):
            validate_hyperpod_compute(compute, session, is_nova=True)

    # --- Region propagation ---

    def test_uses_session_region_for_sagemaker_client(self):
        compute = self._make_compute()
        session = self._make_session(region_name="eu-west-1")
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "group",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        validate_hyperpod_compute(compute, session, is_nova=False)

        session.boto_session.client.assert_called_once_with("sagemaker", region_name="eu-west-1")

    # --- is_nova flag behavior ---

    def test_is_nova_false_reads_instance_groups_key(self):
        """Non-Nova validation reads 'InstanceGroups' from describe_cluster response."""
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=2)
        session = self._make_session()
        mock_sm_client = Mock()
        # Only InstanceGroups present, no RestrictedInstanceGroups
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Should succeed using InstanceGroups
        validate_hyperpod_compute(compute, session, is_nova=False)

    def test_is_nova_true_reads_restricted_instance_groups_key(self):
        """Nova validation reads 'RestrictedInstanceGroups' from describe_cluster response."""
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=2)
        session = self._make_session()
        mock_sm_client = Mock()
        # Only RestrictedInstanceGroups present, no InstanceGroups
        mock_sm_client.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [
                {
                    "InstanceGroupName": "restricted-worker",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Should succeed using RestrictedInstanceGroups
        validate_hyperpod_compute(compute, session, is_nova=True)

    def test_is_nova_default_is_false(self):
        """Default is_nova=False behavior uses normal InstanceGroups."""
        compute = self._make_compute(instance_type="ml.p5.48xlarge", node_count=2)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        # Should succeed with default is_nova (False)
        validate_hyperpod_compute(compute, session)

    # --- Error message contains cluster name ---

    def test_error_message_includes_cluster_name_for_missing_type(self):
        compute = self._make_compute(cluster_name="my-training-cluster", instance_type="ml.p5e.48xlarge")
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "group",
                    "InstanceType": "ml.g5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError, match="my-training-cluster"):
            validate_hyperpod_compute(compute, session)

    def test_error_message_includes_cluster_name_for_insufficient_capacity(self):
        compute = self._make_compute(cluster_name="my-training-cluster", instance_type="ml.p5.48xlarge", node_count=16)
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "group",
                    "InstanceType": "ml.p5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError, match="my-training-cluster"):
            validate_hyperpod_compute(compute, session)

    # --- Available types are sorted in error message ---

    def test_available_types_are_sorted_in_error(self):
        compute = self._make_compute(instance_type="ml.p5.48xlarge")
        session = self._make_session()
        mock_sm_client = Mock()
        mock_sm_client.describe_cluster.return_value = {
            "InstanceGroups": [
                {
                    "InstanceGroupName": "z-group",
                    "InstanceType": "ml.g5.48xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
                {
                    "InstanceGroupName": "a-group",
                    "InstanceType": "ml.g5.12xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 2,
                    "Status": "InService",
                },
            ],
        }
        session.boto_session.client.return_value = mock_sm_client

        with pytest.raises(ValueError) as exc_info:
            validate_hyperpod_compute(compute, session)

        error_msg = str(exc_info.value)
        # Types should appear in sorted order
        idx_12 = error_msg.index("ml.g5.12xlarge")
        idx_48 = error_msg.index("ml.g5.48xlarge")
        assert idx_12 < idx_48
