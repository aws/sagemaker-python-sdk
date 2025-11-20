# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import pytest
import warnings


def test_base_deserializers_deprecation_warning():
    """Test that importing from base_deserializers raises DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Import the module which should trigger the warning
        import sagemaker.core.base_deserializers  # noqa: F401
        
        # Check that a warning was raised
        assert len(w) >= 1
        
        # Find the deprecation warning
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        
        # Check the warning message
        assert "base_deserializers is deprecated" in str(deprecation_warnings[0].message)
        assert "sagemaker.core.deserializers" in str(deprecation_warnings[0].message)


def test_base_deserializers_imports_from_deserializers():
    """Test that base_deserializers re-exports from deserializers module."""
    import sagemaker.core.base_deserializers as base_deser
    import sagemaker.core.deserializers as deser
    
    # Check that the modules have the same attributes
    # (excluding private attributes and module-specific ones)
    base_attrs = {attr for attr in dir(base_deser) if not attr.startswith('_')}
    deser_attrs = {attr for attr in dir(deser) if not attr.startswith('_')}
    
    # base_deserializers should have at least the public attributes from deserializers
    assert base_attrs.intersection(deser_attrs) == deser_attrs
