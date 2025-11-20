#!/usr/bin/env python3
"""
Comprehensive test runner for all sagemaker-serve unit tests.

This script discovers and runs all unit tests in the tests/unit directory.
"""

import sys
import unittest
import os

def run_all_tests():
    """Discover and run all unit tests."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover all tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
