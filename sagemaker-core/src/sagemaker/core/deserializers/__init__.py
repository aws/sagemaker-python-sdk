"""Deserializers for SageMaker inference."""

from __future__ import absolute_import

# Re-export from base
from sagemaker.core.deserializers.base import *  # noqa: F401, F403

# Note: implementations is not imported here to avoid circular imports
# Import explicitly if needed:
#   from sagemaker.core.deserializers.implementations import ...
