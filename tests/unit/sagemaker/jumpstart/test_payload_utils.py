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

from sagemaker.jumpstart.payload_utils import (
    PayloadSerializer,
    _construct_payload,
)
from sagemaker.jumpstart.types import JumpStartSerializablePayload
from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


class TestConstructPayload(TestCase):
    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_construct_payload(self, patched_get_model_specs):
        patched_get_model_specs.side_effect = get_special_model_spec

        model_id = "prompt-key"
        region = "us-west-2"

        constructed_payload_body = _construct_payload(
            prompt="kobebryant", model_id=model_id, model_version="*", region=region
        ).body

        self.assertEqual(
            {
                "hello": {"prompt": "kobebryant"},
                "seed": 43,
            },
            constructed_payload_body,
        )

        # Unsupported model
        self.assertIsNone(
            _construct_payload(
                prompt="blah",
                model_id="default_payloads",
                model_version="*",
                region=region,
            )
        )

    @patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
    def test_construct_payload_with_specific_alias(self, patched_get_model_specs):
        patched_get_model_specs.side_effect = get_special_model_spec

        model_id = "prompt-key"
        region = "us-west-2"

        constructed_payload_body = _construct_payload(
            prompt="kobebryant", model_id=model_id, model_version="*", region=region, alias="Dog"
        ).body

        self.assertEqual(
            {
                "hello": {"prompt": "kobebryant"},
                "seed": 43,
            },
            constructed_payload_body,
        )

        # Unsupported model
        self.assertIsNone(
            _construct_payload(
                prompt="blah",
                model_id="default_payloads",
                model_version="*",
                region=region,
            )
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
