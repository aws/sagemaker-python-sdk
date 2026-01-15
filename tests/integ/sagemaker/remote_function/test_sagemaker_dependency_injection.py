"""Integration tests for sagemaker dependency injection in remote functions.

These tests verify that the sagemaker>=2.256.0 dependency is properly injected
into remote function jobs, preventing version mismatch issues.
"""

import os
import sys
import tempfile
import pytest

# Skip decorator for AWS configuration
skip_if_no_aws_region = pytest.mark.skipif(
    not os.environ.get('AWS_DEFAULT_REGION'), reason="AWS credentials not configured"
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

from sagemaker.remote_function import remote


class TestRemoteFunctionDependencyInjection:
    """Integration tests for dependency injection in remote functions."""

    @pytest.mark.integ
    @skip_if_no_aws_region
    def test_remote_function_without_dependencies(self):
        """Test remote function execution without explicit dependencies.

        This test verifies that when no dependencies are provided, the remote
        function still executes successfully because sagemaker>=2.256.0 is
        automatically injected.
        """

        @remote(
            instance_type="ml.m5.large",
            # No dependencies specified - sagemaker should be injected automatically
        )
        def simple_add(x, y):
            """Simple function that adds two numbers."""
            return x + y

        # Execute the function
        result = simple_add(5, 3)

        # Verify result
        assert result == 8, f"Expected 8, got {result}"
        print("✓ Remote function without dependencies executed successfully")

    @pytest.mark.integ
    @skip_if_no_aws_region
    def test_remote_function_with_user_dependencies_no_sagemaker(self):
        """Test remote function with user dependencies but no sagemaker.

        This test verifies that when user provides dependencies without sagemaker,
        sagemaker>=2.256.0 is automatically appended.
        """
        # Create a temporary requirements.txt without sagemaker
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("numpy>=1.20.0\npandas>=1.3.0\n")
            req_file = f.name

        try:
            @remote(
                instance_type="ml.m5.large",
                dependencies=req_file,
            )
            def compute_with_numpy(x):
                """Function that uses numpy."""
                import numpy as np

                return np.array([x, x*2, x*3]).sum()

            # Execute the function
            result = compute_with_numpy(5)

            # Verify result (5 + 10 + 15 = 30)
            assert result == 30, f"Expected 30, got {result}"
            print("✓ Remote function with user dependencies executed successfully")
        finally:
            os.remove(req_file)


class TestRemoteFunctionVersionCompatibility:
    """Tests for version compatibility between local and remote environments."""

    @pytest.mark.integ
    @skip_if_no_aws_region
    def test_deserialization_with_injected_sagemaker(self):
        """Test that deserialization works with injected sagemaker dependency.

        This test verifies that the remote environment can properly deserialize
        functions when sagemaker>=2.256.0 is available.
        """

        @remote(
            instance_type="ml.m5.large",
        )
        def complex_computation(data):
            """Function that performs complex computation."""
            result = sum(data) * len(data)
            return result

        # Execute with various data types
        test_data = [1, 2, 3, 4, 5]
        result = complex_computation(test_data)

        # Verify result (sum=15, len=5, 15*5=75)
        assert result == 75, f"Expected 75, got {result}"
        print("✓ Deserialization with injected sagemaker works correctly")

    @pytest.mark.integ
    @skip_if_no_aws_region
    def test_multiple_remote_functions_with_dependencies(self):
        """Test multiple remote functions with different dependency configurations.

        This test verifies that the dependency injection works correctly
        when multiple remote functions are defined and executed.
        """

        @remote(instance_type="ml.m5.large")
        def func1(x):
            return x + 1

        @remote(instance_type="ml.m5.large")
        def func2(x):
            return x * 2

        # Execute both functions
        result1 = func1(5)
        result2 = func2(5)

        assert result1 == 6, f"func1: Expected 6, got {result1}"
        assert result2 == 10, f"func2: Expected 10, got {result2}"
        print("✓ Multiple remote functions with dependencies executed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integ"])
