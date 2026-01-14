"""Unit tests for _ensure_sagemaker_dependency function.

Tests the logic that ensures sagemaker>=2.256.0 is included in remote function dependencies
to prevent version mismatch issues with HMAC key integrity checks.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from sagemaker.remote_function.job import _ensure_sagemaker_dependency, _check_sagemaker_version_compatibility


class TestEnsureSagemakerDependency(unittest.TestCase):
    """Test cases for _ensure_sagemaker_dependency function."""

    def test_no_dependencies_creates_temp_requirements_file(self):
        """Test that a temp requirements.txt is created when no dependencies provided."""
        result = _ensure_sagemaker_dependency(None)
        
        # Verify file was created
        self.assertTrue(os.path.exists(result), f"Requirements file not created at {result}")
        
        # Verify it's in temp directory
        self.assertIn(tempfile.gettempdir(), result)
        
        # Verify content
        with open(result, "r") as f:
            content = f.read()
        self.assertIn("sagemaker>=2.256.0,<3.0.0", content)
        
        # Cleanup
        os.remove(result)

    def test_no_dependencies_file_has_correct_format(self):
        """Test that created requirements.txt has correct format."""
        result = _ensure_sagemaker_dependency(None)
        
        with open(result, "r") as f:
            lines = f.readlines()
        
        # Should have exactly one line with sagemaker dependency
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].strip(), "sagemaker>=2.256.0,<3.0.0")
        
        # Cleanup
        os.remove(result)

    def test_appends_sagemaker_to_existing_requirements(self):
        """Test that sagemaker is appended to existing requirements.txt."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("numpy>=1.20.0\npandas>=1.3.0\n")
            temp_file = f.name
        
        try:
            result = _ensure_sagemaker_dependency(temp_file)
            
            # Should return the same file
            self.assertEqual(result, temp_file)
            
            # Verify content
            with open(result, "r") as f:
                content = f.read()
            
            self.assertIn("numpy>=1.20.0", content)
            self.assertIn("pandas>=1.3.0", content)
            self.assertIn("sagemaker>=2.256.0,<3.0.0", content)
        finally:
            os.remove(temp_file)

    def test_does_not_duplicate_sagemaker_if_already_present(self):
        """Test that sagemaker is not duplicated if already in requirements."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("numpy>=1.20.0\nsagemaker>=2.256.0,<3.0.0\npandas>=1.3.0\n")
            temp_file = f.name
        
        try:
            result = _ensure_sagemaker_dependency(temp_file)
            
            with open(result, "r") as f:
                content = f.read()
            
            # Count occurrences of sagemaker
            sagemaker_count = content.lower().count("sagemaker")
            self.assertEqual(sagemaker_count, 1, "sagemaker should appear exactly once")
            
            # Verify user's version is preserved
            self.assertIn("sagemaker>=2.256.0,<3.0.0", content)
        finally:
            os.remove(temp_file)

    def test_preserves_user_dependencies(self):
        """Test that user's existing dependencies are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("torch>=1.9.0\ntorchvision>=0.10.0\nscikit-learn>=0.24.0\n")
            temp_file = f.name
        
        try:
            result = _ensure_sagemaker_dependency(temp_file)
            
            with open(result, "r") as f:
                content = f.read()
            
            # All user dependencies should be present
            self.assertIn("torch>=1.9.0", content)
            self.assertIn("torchvision>=0.10.0", content)
            self.assertIn("scikit-learn>=0.24.0", content)
            self.assertIn("sagemaker>=2.256.0,<3.0.0", content)
        finally:
            os.remove(temp_file)

    def test_handles_yml_files_gracefully(self):
        """Test that yml files are returned unchanged."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("name: test-env\nchannels:\n  - conda-forge\ndependencies:\n  - numpy\n")
            temp_file = f.name
        
        try:
            result = _ensure_sagemaker_dependency(temp_file)
            
            # Should return the same file
            self.assertEqual(result, temp_file)
            
            # Content should be unchanged (yml files are not modified)
            with open(result, "r") as f:
                content = f.read()
            
            self.assertNotIn("sagemaker", content.lower())
        finally:
            os.remove(temp_file)

    def test_handles_yaml_files_gracefully(self):
        """Test that yaml files are returned unchanged."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("name: test-env\nchannels:\n  - conda-forge\n")
            temp_file = f.name
        
        try:
            result = _ensure_sagemaker_dependency(temp_file)
            
            # Should return the same file
            self.assertEqual(result, temp_file)
            
            # Content should be unchanged
            with open(result, "r") as f:
                content = f.read()
            
            self.assertNotIn("sagemaker", content.lower())
        finally:
            os.remove(temp_file)

    def test_case_insensitive_sagemaker_detection(self):
        """Test that sagemaker detection is case-insensitive."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("numpy>=1.20.0\nSAGEMAKER>=2.256.0,<3.0.0\n")
            temp_file = f.name
        
        try:
            result = _ensure_sagemaker_dependency(temp_file)
            
            with open(result, "r") as f:
                content = f.read()
            
            # Should not duplicate even with different case
            sagemaker_count = content.lower().count("sagemaker")
            self.assertEqual(sagemaker_count, 1)
        finally:
            os.remove(temp_file)

    def test_temp_file_location(self):
        """Test that temp file is created in system temp directory."""
        result = _ensure_sagemaker_dependency(None)
        
        # Should be in system temp directory
        temp_dir = tempfile.gettempdir()
        self.assertTrue(result.startswith(temp_dir))
        
        # Should have correct prefix
        self.assertIn("sagemaker_requirements_", result)
        
        # Cleanup
        os.remove(result)

    def test_version_constraint_format(self):
        """Test that version constraint has correct format."""
        result = _ensure_sagemaker_dependency(None)
        
        with open(result, "r") as f:
            content = f.read().strip()
        
        # Should have both lower and upper bounds
        self.assertIn(">=2.256.0", content)
        self.assertIn("<3.0.0", content)
        
        # Cleanup
        os.remove(result)


class TestCheckSagemakerVersionCompatibility(unittest.TestCase):
    """Test cases for _check_sagemaker_version_compatibility function."""

    # ===== GOOD CASES (should NOT raise ValueError) =====

    def test_v2_good_exact_version_256(self):
        """Test V2 exact version 2.256.0 (good - SHA256)."""
        # Should not raise
        _check_sagemaker_version_compatibility("sagemaker==2.256.0")

    def test_v2_good_exact_version_300(self):
        """Test V2 exact version 2.300.0 (good - SHA256)."""
        # Should not raise
        _check_sagemaker_version_compatibility("sagemaker==2.300.0")

    def test_v2_good_range_256_to_300(self):
        """Test V2 range 2.256.0 to 2.300.0 (good - SHA256)."""
        # Should not raise
        _check_sagemaker_version_compatibility("sagemaker>=2.256.0,<2.300.0")

    def test_v3_good_exact_version_32(self):
        """Test V3 exact version 3.2.0 (good - SHA256)."""
        # Should not raise
        _check_sagemaker_version_compatibility("sagemaker==3.2.0")

    def test_v3_good_greater_equal_32(self):
        """Test V3 greater or equal 3.2.0 (good - SHA256)."""
        # Should not raise
        _check_sagemaker_version_compatibility("sagemaker>=3.2.0")

    def test_v3_good_range_32_to_40(self):
        """Test V3 range 3.2.0 to 4.0.0 (good - SHA256)."""
        # Should not raise
        _check_sagemaker_version_compatibility("sagemaker>=3.2.0,<4.0.0")

    def test_unparseable_requirement_no_error(self):
        """Test that unparseable requirements don't raise (let pip handle it)."""
        # Should not raise - let pip handle invalid syntax
        _check_sagemaker_version_compatibility("sagemaker")
        _check_sagemaker_version_compatibility("invalid-requirement")

    def test_v2_bad_exact_version_255(self):
        """Test V2 exact version 2.255.0 (bad - HMAC)."""
        with self.assertRaises(ValueError) as context:
            _check_sagemaker_version_compatibility("sagemaker==2.255.0")
        self.assertIn("HMAC-based integrity checks", str(context.exception))

    def test_v2_bad_exact_version_200(self):
        """Test V2 exact version 2.200.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker==2.200.0")

    def test_v2_bad_less_than_256(self):
        """Test V2 less than 2.256.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker<2.256.0")

    def test_v2_bad_less_equal_255(self):
        """Test V2 less or equal 2.255.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker<=2.255.0")

    def test_v2_bad_greater_than_255_0(self):
        """Test V2 greater than 2.255.0 (not checked - treat as lower bound only)."""
        # Should not raise - > is treated as a lower bound, we don't check those
        _check_sagemaker_version_compatibility("sagemaker>2.255.0")

    def test_v2_bad_range_200_to_255(self):
        """Test V2 range 2.200.0 to 2.255.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker>=2.200.0,<2.256.0")

    def test_v3_bad_exact_version_31(self):
        """Test V3 exact version 3.1.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker==3.1.0")

    def test_v3_bad_exact_version_300(self):
        """Test V3 exact version 3.0.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker==3.0.0")

    def test_v3_bad_less_than_32(self):
        """Test V3 less than 3.2.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker<3.2.0")

    def test_v3_bad_less_equal_31(self):
        """Test V3 less or equal 3.1.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker<=3.1.0")

    def test_v3_bad_range_300_to_31(self):
        """Test V3 range 3.0.0 to 3.1.0 (bad - HMAC)."""
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker>=3.0.0,<3.2.0")

    # ===== EDGE CASES =====

    def test_multiple_version_specifiers_good(self):
        """Test multiple version specifiers that are good."""
        # Should not raise
        _check_sagemaker_version_compatibility("sagemaker>=2.256.0,<3.0.0")

    def test_multiple_version_specifiers_good_with_lower_bound(self):
        """Test multiple version specifiers that are good (upper bound resolves to good version)."""
        # Should not raise - <2.300.0 decrements to 2.299.0 which is >= 2.256.0
        _check_sagemaker_version_compatibility("sagemaker>=2.200.0,<2.300.0")

    def test_multiple_version_specifiers_bad(self):
        """Test multiple version specifiers that are bad."""
        # Should raise - <2.256.0 decrements to 2.255.0 which is < 2.256.0 (HMAC)
        with self.assertRaises(ValueError):
            _check_sagemaker_version_compatibility("sagemaker>=2.200.0,<2.256.0")


if __name__ == "__main__":
    unittest.main()
