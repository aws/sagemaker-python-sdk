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
"""Additional tests for spec.inference_base module"""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch, MagicMock
from abc import ABC

from sagemaker.serve.spec.inference_base import CustomOrchestrator, AsyncCustomOrchestrator


class TestCustomOrchestrator:
    """Test CustomOrchestrator base class"""
    
    def test_custom_orchestrator_is_abstract(self):
        """Test that CustomOrchestrator is an abstract base class"""
        assert issubclass(CustomOrchestrator, ABC)
    
    def test_custom_orchestrator_cannot_be_instantiated_directly(self):
        """Test that CustomOrchestrator cannot be instantiated without implementing handle"""
        with pytest.raises(TypeError):
            CustomOrchestrator()
    
    def test_custom_orchestrator_requires_handle_implementation(self):
        """Test that subclass must implement handle method"""
        class IncompleteOrchestrator(CustomOrchestrator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteOrchestrator()
    
    def test_custom_orchestrator_with_handle_implementation(self):
        """Test that CustomOrchestrator can be instantiated with handle implementation"""
        class CompleteOrchestrator(CustomOrchestrator):
            def handle(self, data, context=None):
                return "processed"
        
        orchestrator = CompleteOrchestrator()
        assert orchestrator is not None
        assert orchestrator.handle("data") == "processed"
    
    def test_custom_orchestrator_client_property_lazy_initialization(self):
        """Test that client property is lazily initialized"""
        class TestOrchestrator(CustomOrchestrator):
            def handle(self, data, context=None):
                return data
        
        orchestrator = TestOrchestrator()
        
        # Client should not be set initially
        assert orchestrator._client is None
        
        # Access client property
        with patch('boto3.Session') as mock_session:
            mock_client = Mock()
            mock_session.return_value.client.return_value = mock_client
            
            client = orchestrator.client
            
            assert client == mock_client
            mock_session.return_value.client.assert_called_once_with("sagemaker-runtime")
    
    def test_custom_orchestrator_client_property_caching(self):
        """Test that client property is cached after first access"""
        class TestOrchestrator(CustomOrchestrator):
            def handle(self, data, context=None):
                return data
        
        orchestrator = TestOrchestrator()
        
        with patch('boto3.Session') as mock_session:
            mock_client = Mock()
            mock_session.return_value.client.return_value = mock_client
            
            # First access
            client1 = orchestrator.client
            # Second access
            client2 = orchestrator.client
            
            # Should be the same client instance
            assert client1 is client2
            # Session.client should only be called once
            assert mock_session.return_value.client.call_count == 1
    
    def test_custom_orchestrator_client_property_returns_sagemaker_runtime(self):
        """Test that client property returns sagemaker-runtime client"""
        class TestOrchestrator(CustomOrchestrator):
            def handle(self, data, context=None):
                return data
        
        orchestrator = TestOrchestrator()
        
        with patch('boto3.Session') as mock_session:
            mock_client = Mock()
            mock_session.return_value.client.return_value = mock_client
            
            client = orchestrator.client
            
            # Verify it's requesting sagemaker-runtime client
            mock_session.return_value.client.assert_called_with("sagemaker-runtime")
    
    def test_custom_orchestrator_handle_with_context(self):
        """Test that handle method can accept context parameter"""
        class TestOrchestrator(CustomOrchestrator):
            def handle(self, data, context=None):
                if context:
                    return f"{data}-{context}"
                return data
        
        orchestrator = TestOrchestrator()
        
        assert orchestrator.handle("data") == "data"
        assert orchestrator.handle("data", "context") == "data-context"
    
    def test_custom_orchestrator_init_sets_client_to_none(self):
        """Test that __init__ sets _client to None"""
        class TestOrchestrator(CustomOrchestrator):
            def handle(self, data, context=None):
                return data
        
        orchestrator = TestOrchestrator()
        assert hasattr(orchestrator, '_client')
        assert orchestrator._client is None


class TestAsyncCustomOrchestrator:
    """Test AsyncCustomOrchestrator base class"""
    
    def test_async_custom_orchestrator_is_abstract(self):
        """Test that AsyncCustomOrchestrator is an abstract base class"""
        assert issubclass(AsyncCustomOrchestrator, ABC)
    
    def test_async_custom_orchestrator_cannot_be_instantiated_directly(self):
        """Test that AsyncCustomOrchestrator cannot be instantiated without implementing handle"""
        with pytest.raises(TypeError):
            AsyncCustomOrchestrator()
    
    def test_async_custom_orchestrator_requires_handle_implementation(self):
        """Test that subclass must implement async handle method"""
        class IncompleteAsyncOrchestrator(AsyncCustomOrchestrator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteAsyncOrchestrator()
    
    def test_async_custom_orchestrator_with_handle_implementation(self):
        """Test that AsyncCustomOrchestrator can be instantiated with handle implementation"""
        class CompleteAsyncOrchestrator(AsyncCustomOrchestrator):
            async def handle(self, data, context=None):
                return "processed"
        
        orchestrator = CompleteAsyncOrchestrator()
        assert orchestrator is not None
    
    def test_async_custom_orchestrator_handle_method_exists(self):
        """Test that AsyncCustomOrchestrator subclass has handle method"""
        class TestAsyncOrchestrator(AsyncCustomOrchestrator):
            async def handle(self, data, context=None):
                return f"async-{data}"
        
        orchestrator = TestAsyncOrchestrator()
        assert hasattr(orchestrator, 'handle')
        assert callable(orchestrator.handle)
    
    def test_async_custom_orchestrator_no_client_property(self):
        """Test that AsyncCustomOrchestrator doesn't have client property like CustomOrchestrator"""
        class TestAsyncOrchestrator(AsyncCustomOrchestrator):
            async def handle(self, data, context=None):
                return data
        
        orchestrator = TestAsyncOrchestrator()
        
        # AsyncCustomOrchestrator doesn't have client property
        assert not hasattr(orchestrator, 'client')


class TestOrchestratorComparison:
    """Test differences between CustomOrchestrator and AsyncCustomOrchestrator"""
    
    def test_custom_orchestrator_has_client_property(self):
        """Test that CustomOrchestrator has client property"""
        class TestOrchestrator(CustomOrchestrator):
            def handle(self, data, context=None):
                return data
        
        orchestrator = TestOrchestrator()
        assert hasattr(orchestrator, '_client')
    
    def test_async_orchestrator_no_init(self):
        """Test that AsyncCustomOrchestrator doesn't define __init__"""
        class TestAsyncOrchestrator(AsyncCustomOrchestrator):
            async def handle(self, data, context=None):
                return data
        
        orchestrator = TestAsyncOrchestrator()
        # Should not have _client attribute
        assert not hasattr(orchestrator, '_client')
    
    def test_both_orchestrators_require_handle_method(self):
        """Test that both orchestrator types require handle method"""
        # Sync version
        with pytest.raises(TypeError):
            class BadSync(CustomOrchestrator):
                pass
            BadSync()
        
        # Async version
        with pytest.raises(TypeError):
            class BadAsync(AsyncCustomOrchestrator):
                pass
            BadAsync()
    
    def test_handle_signatures_match(self):
        """Test that both handle methods have same signature (data, context=None)"""
        class SyncOrch(CustomOrchestrator):
            def handle(self, data, context=None):
                return (data, context)
        
        class AsyncOrch(AsyncCustomOrchestrator):
            async def handle(self, data, context=None):
                return (data, context)
        
        sync_orch = SyncOrch()
        async_orch = AsyncOrch()
        
        # Both should accept same parameters
        sync_result = sync_orch.handle("data", "context")
        assert sync_result == ("data", "context")
