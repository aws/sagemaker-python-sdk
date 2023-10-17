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
import base64
from unittest import TestCase
from mock.mock import patch
import pytest

from sagemaker.jumpstart.payload_utils import (
    PayloadSerializer,
    _extract_generated_text_from_response,
)
from sagemaker.jumpstart.types import JumpStartSerializablePayload


from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


class TestResponseExtraction(TestCase):
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_extract_generated_text(self, patched_get_model_specs):
        patched_get_model_specs.side_effect = get_special_model_spec

        model_id = "response-keys"
        region = "us-west-2"
        generated_text = _extract_generated_text_from_response(
            response={"key1": {"key2": {"generated_text": "top secret"}}},
            model_id=model_id,
            model_version="*",
            region=region,
        )

        self.assertEqual(
            _extract_generated_text_from_response(
                response={"key1": {"key2": {"generated_text": "top secret"}}},
                model_id=model_id,
                model_version="*",
                region=region,
                accept_type="application/json",
            ),
            generated_text,
        )

        self.assertEqual(
            generated_text,
            "top secret",
        )

        with pytest.raises(ValueError):
            _extract_generated_text_from_response(
                response={"key1": {"key2": {"generated_texts": "top secret"}}},
                model_id=model_id,
                model_version="*",
                region=region,
            )

        with pytest.raises(ValueError):
            _extract_generated_text_from_response(
                response={"key1": {"key2": {"generated_text": "top secret"}}},
                model_id=model_id,
                model_version="*",
                region=region,
                accept_type="blah/blah",
            )

        with pytest.raises(ValueError):
            _extract_generated_text_from_response(
                response={"key1": {"key2": {"generated_text": "top secret"}}},
                model_id="env-var-variant-model",  # some model without the required metadata
                model_version="*",
                region=region,
            )
        with pytest.raises(ValueError):
            _extract_generated_text_from_response(
                response={"key1": {"generated_texts": "top secret"}},
                model_id=model_id,
                model_version="*",
                region=region,
            )

        with pytest.raises(ValueError):
            _extract_generated_text_from_response(
                response="blah",
                model_id=model_id,
                model_version="*",
                region=region,
            )


class TestPayloadSerializer(TestCase):

    payload_serializer = PayloadSerializer()

    @patch("sagemaker.jumpstart.payload_utils.JumpStartS3PayloadAccessor.get_object_cached")
    def test_serialize_bytes_payload(self, mock_get_object_cached):

        mock_get_object_cached.return_value = "7897"
        payload = JumpStartSerializablePayload(
            {
                "content_type": "audio/wav",
                "body": "$s3<inference-notebook-assets/speaker_1_angry.wav>",
            }
        )
        serialized_payload = self.payload_serializer.serialize(payload)
        self.assertEqual(serialized_payload, "7897")

    @patch("sagemaker.jumpstart.payload_utils.JumpStartS3PayloadAccessor.get_object_cached")
    def test_serialize_json_payload(self, mock_get_object_cached):

        mock_get_object_cached.return_value = base64.b64decode("encodedimage")
        payload = JumpStartSerializablePayload(
            {
                "content_type": "application/json",
                "body": {
                    "prompt": "a dog",
                    "num_images_per_prompt": 2,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "seed": 43,
                    "eta": 0.7,
                    "image": "$s3_b64<inference-notebook-assets/inpainting_cow.jpg>",
                },
            }
        )
        serialized_payload = self.payload_serializer.serialize(payload)
        self.assertEqual(
            serialized_payload,
            '{"prompt": "a dog", "num_images_per_prompt": 2, '
            '"num_inference_steps": 20, "guidance_scale": 7.5, "seed": '
            '43, "eta": 0.7, "image": "encodedimage"}',
        )
