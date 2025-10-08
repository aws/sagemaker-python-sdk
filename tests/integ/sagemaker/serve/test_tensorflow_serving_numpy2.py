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
"""Simple integration test for TensorFlow Serving builder with numpy 2.0 compatibility."""

from __future__ import absolute_import

import pytest
import io
import os
import numpy as np
import logging
from tests.integ import DATA_DIR

from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
from sagemaker.serve.builder.schema_builder import SchemaBuilder, CustomPayloadTranslator
from sagemaker.serve.utils.types import ModelServer

logger = logging.getLogger(__name__)


class TestTensorFlowServingNumpy2:
    """Simple integration tests for TensorFlow Serving with numpy 2.0."""

    def test_tensorflow_serving_validation_with_numpy2(self, sagemaker_session):
        """Test TensorFlow Serving validation works with numpy 2.0."""
        logger.info(f"Testing TensorFlow Serving validation with numpy {np.__version__}")

        # Create a simple schema builder with numpy 2.0 arrays
        input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        output_data = np.array([4.0], dtype=np.float32)

        schema_builder = SchemaBuilder(sample_input=input_data, sample_output=output_data)

        # Test without MLflow model - should raise validation error
        model_builder = ModelBuilder(
            mode=Mode.SAGEMAKER_ENDPOINT,
            model_server=ModelServer.TENSORFLOW_SERVING,
            schema_builder=schema_builder,
            sagemaker_session=sagemaker_session,
        )

        with pytest.raises(
            ValueError, match="Tensorflow Serving is currently only supported for mlflow models"
        ):
            model_builder._validate_for_tensorflow_serving()

        logger.info("TensorFlow Serving validation test passed")

    def test_tensorflow_serving_with_sample_mlflow_model(self, sagemaker_session):
        """Test TensorFlow Serving builder initialization with sample MLflow model."""
        logger.info("Testing TensorFlow Serving with sample MLflow model")

        # Use constant MLflow model structure from test data
        mlflow_model_dir = os.path.join(DATA_DIR, "serve_resources", "mlflow", "tensorflow_numpy2")

        # Create schema builder with numpy 2.0 arrays
        input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        output_data = np.array([5.0], dtype=np.float32)

        schema_builder = SchemaBuilder(sample_input=input_data, sample_output=output_data)

        # Create ModelBuilder - this should not raise validation errors
        model_builder = ModelBuilder(
            mode=Mode.SAGEMAKER_ENDPOINT,
            model_server=ModelServer.TENSORFLOW_SERVING,
            schema_builder=schema_builder,
            sagemaker_session=sagemaker_session,
            model_metadata={"MLFLOW_MODEL_PATH": mlflow_model_dir},
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        )

        # Initialize MLflow handling to set _is_mlflow_model flag
        model_builder._handle_mlflow_input()

        # Test validation passes
        model_builder._validate_for_tensorflow_serving()
        logger.info("TensorFlow Serving with sample MLflow model test passed")

    def test_numpy2_custom_payload_translators(self):
        """Test custom payload translators work with numpy 2.0."""
        logger.info(f"Testing custom payload translators with numpy {np.__version__}")

        class Numpy2RequestTranslator(CustomPayloadTranslator):
            def serialize_payload_to_bytes(self, payload: object) -> bytes:
                buffer = io.BytesIO()
                np.save(buffer, payload, allow_pickle=False)
                return buffer.getvalue()

            def deserialize_payload_from_stream(self, stream) -> object:
                return np.load(io.BytesIO(stream.read()), allow_pickle=False)

        class Numpy2ResponseTranslator(CustomPayloadTranslator):
            def serialize_payload_to_bytes(self, payload: object) -> bytes:
                buffer = io.BytesIO()
                np.save(buffer, np.array(payload), allow_pickle=False)
                return buffer.getvalue()

            def deserialize_payload_from_stream(self, stream) -> object:
                return np.load(io.BytesIO(stream.read()), allow_pickle=False)

        # Test data
        test_input = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        test_output = np.array([4.0], dtype=np.float32)

        # Create translators
        request_translator = Numpy2RequestTranslator()
        response_translator = Numpy2ResponseTranslator()

        # Test request translator
        serialized_input = request_translator.serialize_payload_to_bytes(test_input)
        assert isinstance(serialized_input, bytes)

        deserialized_input = request_translator.deserialize_payload_from_stream(
            io.BytesIO(serialized_input)
        )
        np.testing.assert_array_equal(test_input, deserialized_input)

        # Test response translator
        serialized_output = response_translator.serialize_payload_to_bytes(test_output)
        assert isinstance(serialized_output, bytes)

        deserialized_output = response_translator.deserialize_payload_from_stream(
            io.BytesIO(serialized_output)
        )
        np.testing.assert_array_equal(test_output, deserialized_output)

        logger.info("Custom payload translators test passed")

    def test_numpy2_schema_builder_creation(self):
        """Test SchemaBuilder creation with numpy 2.0 arrays."""
        logger.info(f"Testing SchemaBuilder with numpy {np.__version__}")

        # Create test data with numpy 2.0
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        output_data = np.array([10.0], dtype=np.float32)

        # Create SchemaBuilder
        schema_builder = SchemaBuilder(sample_input=input_data, sample_output=output_data)

        # Verify schema builder properties
        assert schema_builder.sample_input is not None
        assert schema_builder.sample_output is not None

        # Test with custom translators
        class TestTranslator(CustomPayloadTranslator):
            def serialize_payload_to_bytes(self, payload: object) -> bytes:
                buffer = io.BytesIO()
                np.save(buffer, payload, allow_pickle=False)
                return buffer.getvalue()

            def deserialize_payload_from_stream(self, stream) -> object:
                return np.load(io.BytesIO(stream.read()), allow_pickle=False)

        translator = TestTranslator()
        schema_builder_with_translator = SchemaBuilder(
            sample_input=input_data,
            sample_output=output_data,
            input_translator=translator,
            output_translator=translator,
        )

        assert schema_builder_with_translator.custom_input_translator is not None
        assert schema_builder_with_translator.custom_output_translator is not None

        logger.info("SchemaBuilder creation test passed")

    def test_numpy2_basic_operations(self):
        """Test basic numpy 2.0 operations used in TensorFlow Serving."""
        logger.info(f"Testing basic numpy 2.0 operations. Version: {np.__version__}")

        # Test array creation
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert arr.dtype == np.float32
        assert arr.shape == (4,)

        # Test array operations
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert arr_2d.shape == (2, 2)

        # Test serialization without pickle (numpy 2.0 safe)
        buffer = io.BytesIO()
        np.save(buffer, arr_2d, allow_pickle=False)
        buffer.seek(0)
        loaded_arr = np.load(buffer, allow_pickle=False)

        np.testing.assert_array_equal(arr_2d, loaded_arr)

        # Test dtype preservation
        assert loaded_arr.dtype == np.float32

        logger.info("Basic numpy 2.0 operations test passed")
