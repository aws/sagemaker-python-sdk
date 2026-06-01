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
"""Tests for Model.sagemaker_session property used by ModelStep.

These tests verify that the sagemaker_session property works correctly
when applied to Model-like classes, as required by ModelStep during
pipeline composition (issue #5829).

Note: Tests use a FakeV3Model class rather than instantiating the real
V3 Model class directly, because the real Model class may require
arguments or have side effects (like API calls) that are inappropriate
for unit tests.
"""
from __future__ import absolute_import

import pytest
from unittest.mock import MagicMock, patch

from sagemaker.core.session_mixin import apply_sagemaker_session_property


def _make_mock_session() -> MagicMock:
    """Create a mock that simulates a SageMaker Session."""
    mock_session = MagicMock()
    mock_session.boto_session = MagicMock()
    mock_session.boto_region_name = "us-east-1"
    return mock_session


def _make_mock_pipeline_session() -> MagicMock:
    """Create a mock that simulates a PipelineSession.

    Includes PipelineSession-specific attributes to differentiate from
    a regular Session mock.
    """
    mock_session = MagicMock()
    mock_session.boto_session = MagicMock()
    mock_session.boto_region_name = "us-west-2"
    # PipelineSession-specific attributes
    mock_session.context = MagicMock()
    mock_session._context = "pipeline"
    mock_session.default_bucket_prefix = "pipeline-prefix"
    return mock_session


def _make_fake_v3_model_class():
    """Create a fake V3 Model class that simulates the real Model's structure.

    This avoids depending on the real V3 Model constructor signature,
    which may require arguments or trigger side effects.
    """

    class FakeV3Model:
        """A fake V3 Model class for testing."""

        def __init__(self, model_name="test-model", containers=None, **kwargs):
            self.model_name = model_name
            self.containers = containers or []
            # Simulate V3 Model having various attributes
            for key, value in kwargs.items():
                setattr(self, key, value)

        def create(self):
            return {"ModelArn": f"arn:aws:sagemaker:us-east-1:123456789012:model/{self.model_name}"}

        def describe(self):
            return {"ModelName": self.model_name}

        def delete(self):
            pass

    apply_sagemaker_session_property(FakeV3Model)
    return FakeV3Model


class TestApplySagemakerSessionProperty:
    """Tests for the apply_sagemaker_session_property function.

    These tests verify that monkey-patching works correctly on arbitrary
    classes, simulating how it would work with the real V3Model.
    """

    def test_apply_to_plain_class(self):
        """Test that apply_sagemaker_session_property adds the property to a plain class."""

        class FakeV3Model:
            def __init__(self, name="test"):
                self.name = name

        apply_sagemaker_session_property(FakeV3Model)

        instance = FakeV3Model(name="my-model")
        assert instance.sagemaker_session is None
        assert instance.name == "my-model"

    def test_apply_to_class_with_existing_attributes(self):
        """Test that patching does not interfere with existing class attributes."""

        class FakeV3Model:
            def __init__(self, model_name, containers=None):
                self.model_name = model_name
                self.containers = containers or []

            def describe(self):
                return {"ModelName": self.model_name}

        apply_sagemaker_session_property(FakeV3Model)

        instance = FakeV3Model(model_name="test-model", containers=["c1"])
        assert instance.sagemaker_session is None
        assert instance.model_name == "test-model"
        assert instance.containers == ["c1"]
        assert instance.describe() == {"ModelName": "test-model"}

        # Set session after construction
        mock_session = _make_mock_session()
        instance.sagemaker_session = mock_session
        assert instance.sagemaker_session is mock_session
        # Original attributes still work
        assert instance.model_name == "test-model"

    def test_apply_does_not_override_existing_property(self):
        """Test that apply does not override if sagemaker_session property already exists."""

        class AlreadyPatched:
            @property
            def sagemaker_session(self):
                return "already-set"

        apply_sagemaker_session_property(AlreadyPatched)

        instance = AlreadyPatched()
        # Should keep the existing property
        assert instance.sagemaker_session == "already-set"

    def test_apply_does_not_override_inherited_property(self):
        """Test that apply does not override a property inherited from a parent class."""

        class Base:
            @property
            def sagemaker_session(self):
                return "from-base"

        class Child(Base):
            pass

        apply_sagemaker_session_property(Child)

        instance = Child()
        # Should keep the inherited property from Base
        assert instance.sagemaker_session == "from-base"

    def test_apply_overrides_non_property_attribute(self):
        """Test that apply overrides a regular (non-property) class attribute.

        If a class defines sagemaker_session as a plain attribute (not a
        property descriptor), the patch should replace it with the property.
        """

        class HasPlainAttribute:
            sagemaker_session = None

        apply_sagemaker_session_property(HasPlainAttribute)

        instance = HasPlainAttribute()
        # Should now be the property (returns None via getter)
        assert instance.sagemaker_session is None

        # Should be settable via property setter
        mock_session = _make_mock_session()
        instance.sagemaker_session = mock_session
        assert instance.sagemaker_session is mock_session

    def test_apply_to_class_without_cooperative_init(self):
        """Test that patching works even if __init__ does not call super().

        This verifies the fix for the MRO fragility concern: since we use
        getattr with a default in the getter, the property works even if
        _sagemaker_session was never set in __init__.
        """

        class NonCooperativeModel:
            def __init__(self, name):
                # Does NOT call super().__init__()
                self.name = name

        apply_sagemaker_session_property(NonCooperativeModel)

        instance = NonCooperativeModel(name="test")
        # Should return None (via getattr default), not raise AttributeError
        assert instance.sagemaker_session is None

        # Setting should work
        mock_session = _make_mock_session()
        instance.sagemaker_session = mock_session
        assert instance.sagemaker_session is mock_session

    def test_same_class_object_is_returned(self):
        """Test that apply_sagemaker_session_property patches in place and returns same class."""

        class FakeV3Model:
            pass

        result = apply_sagemaker_session_property(FakeV3Model)
        assert result is FakeV3Model

    def test_isinstance_checks_still_work(self):
        """Test that isinstance checks work after patching (no new subclass created)."""

        class FakeV3Model:
            pass

        apply_sagemaker_session_property(FakeV3Model)

        instance = FakeV3Model()
        assert isinstance(instance, FakeV3Model)


class TestModelSagemakerSessionProperty:
    """Tests for the sagemaker_session property on a Model-like class.

    These tests use a FakeV3Model class that simulates the V3 Model's
    structure without depending on the real V3 Model constructor signature
    or triggering any side effects.
    """

    def setup_method(self):
        """Set up a fresh FakeV3Model class for each test."""
        self.ModelClass = _make_fake_v3_model_class()

    def test_model_class_has_sagemaker_session_property(self):
        """Test that the Model class has a sagemaker_session property."""
        assert hasattr(self.ModelClass, "sagemaker_session")
        # Verify it's actually a property descriptor
        assert isinstance(
            self.ModelClass.__dict__.get("sagemaker_session", None), property
        )

    def test_sagemaker_session_property_when_not_set_returns_none(self):
        """Test that sagemaker_session returns None when not explicitly set."""
        model = self.ModelClass(model_name="test-model")
        assert model.sagemaker_session is None

    def test_sagemaker_session_property_when_set_returns_value(self):
        """Test that sagemaker_session returns the value after being set via setter."""
        model = self.ModelClass(model_name="test-model")
        mock_session = _make_mock_session()
        model.sagemaker_session = mock_session
        assert model.sagemaker_session is mock_session

    def test_sagemaker_session_property_when_set_to_pipeline_session(self):
        """Test that sagemaker_session works with a PipelineSession-like object."""
        model = self.ModelClass(model_name="test-model")
        mock_session = _make_mock_pipeline_session()
        model.sagemaker_session = mock_session
        assert model.sagemaker_session is mock_session
        # Verify PipelineSession-specific attributes are accessible
        assert model.sagemaker_session._context == "pipeline"

    def test_sagemaker_session_property_when_overwritten_returns_new_value(self):
        """Test that sagemaker_session returns the latest value after being overwritten."""
        model = self.ModelClass(model_name="test-model")
        mock_session_1 = _make_mock_session()
        mock_session_2 = _make_mock_pipeline_session()

        model.sagemaker_session = mock_session_1
        assert model.sagemaker_session is mock_session_1

        model.sagemaker_session = mock_session_2
        assert model.sagemaker_session is mock_session_2

    def test_sagemaker_session_setter_accepts_none(self):
        """Test that setting sagemaker_session to None is allowed."""
        model = self.ModelClass(model_name="test-model")
        mock_session = _make_mock_session()
        model.sagemaker_session = mock_session
        model.sagemaker_session = None
        assert model.sagemaker_session is None

    def test_sagemaker_session_setter_accepts_any_object(self):
        """Test that the setter accepts any non-None object.

        The setter is permissive to support mocks, custom session
        implementations, and all session types (Session, PipelineSession,
        LocalSession).
        """
        model = self.ModelClass(model_name="test-model")
        custom_session = object()
        model.sagemaker_session = custom_session
        assert model.sagemaker_session is custom_session

    def test_sagemaker_session_via_property_setter(self):
        """Test that sagemaker_session can be set via the property setter after construction.

        This approach is compatible with both the patched V3 Model and the
        fallback class, since the V3 Model constructor may not accept a
        sagemaker_session keyword argument.
        """
        mock_session = _make_mock_session()
        model = self.ModelClass(model_name="test-model")
        model.sagemaker_session = mock_session
        assert model.sagemaker_session is mock_session

    def test_existing_model_methods_still_work_after_patch(self):
        """Test that patching does not break existing Model methods."""
        model = self.ModelClass(model_name="my-model", containers=["container-1"])
        mock_session = _make_mock_session()
        model.sagemaker_session = mock_session

        # Existing methods should still work
        assert model.describe() == {"ModelName": "my-model"}
        assert model.model_name == "my-model"
        assert model.containers == ["container-1"]


class TestModelStepIntegration:
    """Tests verifying ModelStep can access model.sagemaker_session.

    These tests simulate the pattern used by ModelStep during pipeline
    composition, particularly for repack steps (issue #5829).
    """

    def setup_method(self):
        """Set up a fresh FakeV3Model class for each test."""
        self.ModelClass = _make_fake_v3_model_class()

    def test_model_step_can_access_sagemaker_session(self):
        """Test that ModelStep-like code can access model.sagemaker_session.

        This simulates the access pattern in ModelStep where it reads
        model.sagemaker_session to determine the session type and
        configure repack steps.
        """
        mock_pipeline_session = _make_mock_pipeline_session()

        model = self.ModelClass(model_name="my-model")
        model.sagemaker_session = mock_pipeline_session

        # Simulate ModelStep accessing sagemaker_session
        session = model.sagemaker_session
        assert session is not None
        assert session is mock_pipeline_session
        assert session.boto_region_name == "us-west-2"

    def test_model_step_repack_pattern(self):
        """Test the pattern ModelStep uses when determining if repack is needed.

        ModelStep checks model.sagemaker_session to get the session for
        creating repack steps. This test verifies that pattern works.
        """
        mock_pipeline_session = _make_mock_pipeline_session()
        mock_pipeline_session.default_bucket.return_value = "my-pipeline-bucket"

        model = self.ModelClass(model_name="repack-model")
        model.sagemaker_session = mock_pipeline_session

        # Simulate what ModelStep does internally:
        # 1. Access the session from the model
        session = model.sagemaker_session
        assert session is not None

        # 2. Use the session to get bucket info (for repack step)
        bucket = session.default_bucket()
        assert bucket == "my-pipeline-bucket"

        # 3. Check session type attributes (ModelStep checks for PipelineSession)
        assert hasattr(session, "context")

    def test_model_step_with_none_session_raises_appropriately(self):
        """Test that ModelStep-like code can detect when session is not set.

        ModelStep should be able to check if sagemaker_session is None
        and handle it appropriately.
        """
        model = self.ModelClass(model_name="no-session-model")

        # Simulate ModelStep checking for session
        session = model.sagemaker_session
        assert session is None

        # ModelStep would typically raise an error in this case
        # This verifies the property returns None cleanly (no AttributeError)

    def test_model_step_session_assignment_during_step_creation(self):
        """Test that ModelStep can assign a session to the model.

        In some flows, ModelStep assigns its own pipeline session to the
        model if one isn't already set.
        """
        model = self.ModelClass(model_name="step-model")
        assert model.sagemaker_session is None

        # Simulate ModelStep assigning its pipeline session to the model
        pipeline_session = _make_mock_pipeline_session()
        model.sagemaker_session = pipeline_session

        assert model.sagemaker_session is pipeline_session
        assert model.sagemaker_session._context == "pipeline"

    def test_model_step_session_not_overwritten_if_already_set(self):
        """Test pattern where ModelStep checks before overwriting session.

        ModelStep should respect an existing session on the model.
        """
        original_session = _make_mock_pipeline_session()
        model = self.ModelClass(model_name="preset-model")
        model.sagemaker_session = original_session

        # Simulate ModelStep's check-before-assign pattern
        if model.sagemaker_session is None:
            model.sagemaker_session = _make_mock_pipeline_session()

        # Original session should be preserved
        assert model.sagemaker_session is original_session
