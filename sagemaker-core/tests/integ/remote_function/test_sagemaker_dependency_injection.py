"""Integration tests for sagemaker dependency injection in remote functions.

These tests verify that the sagemaker>=3.2.0 dependency is properly injected
into remote function jobs, preventing version mismatch issues.
"""

import os
import sys
import tempfile
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from sagemaker.core.remote_function import remote


class TestRemoteFunctionDependencyInjection:
    """Integration tests for dependency injection in remote functions."""

    @pytest.mark.integ
    def test_remote_function_without_dependencies(
        self, dev_sdk_pre_execution_commands, role, image_uri, sagemaker_session
    ):
        """Test remote function execution without explicit dependencies."""
        @remote(
            instance_type="ml.m5.large",
            role=role,
            image_uri=image_uri,
            sagemaker_session=sagemaker_session,
            pre_execution_commands=dev_sdk_pre_execution_commands,
        )
        def simple_add(x, y):
            return x + y

        result = simple_add(5, 3)
        assert result == 8, f"Expected 8, got {result}"

    @pytest.mark.integ
    def test_remote_function_with_user_dependencies_no_sagemaker(
        self, dev_sdk_pre_execution_commands, role, image_uri, sagemaker_session
    ):
        """Test remote function with user dependencies but no sagemaker."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("numpy>=1.20.0\npandas>=1.3.0\n")
            req_file = f.name

        try:
            @remote(
                instance_type="ml.m5.large",
                role=role,
                image_uri=image_uri,
                sagemaker_session=sagemaker_session,
                dependencies=req_file,
                pre_execution_commands=dev_sdk_pre_execution_commands,
            )
            def compute_with_numpy(x):
                import numpy as np
                return np.array([x, x*2, x*3]).sum()

            result = compute_with_numpy(5)
            assert result == 30, f"Expected 30, got {result}"
        finally:
            os.remove(req_file)


class TestRemoteFunctionVersionCompatibility:
    """Tests for version compatibility between local and remote environments."""

    @pytest.mark.integ
    def test_deserialization_with_injected_sagemaker(
        self, dev_sdk_pre_execution_commands, role, image_uri, sagemaker_session
    ):
        """Test that deserialization works with injected sagemaker dependency."""
        @remote(
            instance_type="ml.m5.large",
            role=role,
            image_uri=image_uri,
            sagemaker_session=sagemaker_session,
            pre_execution_commands=dev_sdk_pre_execution_commands,
        )
        def complex_computation(data):
            result = sum(data) * len(data)
            return result

        test_data = [1, 2, 3, 4, 5]
        result = complex_computation(test_data)
        assert result == 75, f"Expected 75, got {result}"

    @pytest.mark.integ
    def test_multiple_remote_functions_with_dependencies(
        self, dev_sdk_pre_execution_commands, role, image_uri, sagemaker_session
    ):
        """Test multiple remote functions with different dependency configurations."""
        @remote(
            instance_type="ml.m5.large",
            role=role,
            image_uri=image_uri,
            sagemaker_session=sagemaker_session,
            pre_execution_commands=dev_sdk_pre_execution_commands,
        )
        def func1(x):
            return x + 1

        @remote(
            instance_type="ml.m5.large",
            role=role,
            image_uri=image_uri,
            sagemaker_session=sagemaker_session,
            pre_execution_commands=dev_sdk_pre_execution_commands,
        )
        def func2(x):
            return x * 2

        result1 = func1(5)
        result2 = func2(5)

        assert result1 == 6, f"func1: Expected 6, got {result1}"
        assert result2 == 10, f"func2: Expected 10, got {result2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integ"])
