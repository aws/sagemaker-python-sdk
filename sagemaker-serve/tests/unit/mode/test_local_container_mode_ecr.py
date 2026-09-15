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
"""Tests for ECR host classification / login-target extraction in LocalContainerMode.

These guard against the parser-confusion vulnerability where the "is this an ECR image?"
classifier and the "which host do I docker login to?" extractor disagreed, allowing a crafted
image URI to leak the ECR authorization token to an attacker-controlled host.
"""
from __future__ import absolute_import

import unittest
from unittest.mock import MagicMock, patch

from sagemaker.serve.mode.local_container_mode import LocalContainerMode


def _bare_mode():
    """Build a LocalContainerMode without running the heavy constructor.

    ``_is_ecr_image`` / ``_ecr_registry_host`` are pure w.r.t. their ``image`` argument, so we
    only need a bound method, not a fully initialized instance.
    """
    return LocalContainerMode.__new__(LocalContainerMode)


class TestEcrHostClassification(unittest.TestCase):
    def setUp(self):
        self.mode = _bare_mode()

    def test_valid_ecr_uri_is_recognized(self):
        image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest"
        self.assertTrue(self.mode._is_ecr_image(image))
        self.assertEqual(
            self.mode._ecr_registry_host(image),
            "123456789012.dkr.ecr.us-east-1.amazonaws.com",
        )

    def test_valid_ecr_uri_china_partition(self):
        image = "123456789012.dkr.ecr.cn-north-1.amazonaws.com.cn/my-repo:latest"
        self.assertTrue(self.mode._is_ecr_image(image))
        self.assertEqual(
            self.mode._ecr_registry_host(image),
            "123456789012.dkr.ecr.cn-north-1.amazonaws.com.cn",
        )

    def test_attacker_crafted_uri_is_not_ecr(self):
        # The malicious host is first; the ECR-looking substrings appear only in a later segment.
        image = "attacker.com/x.dkr.ecr.us-east-1.amazonaws.com/repo:tag"
        self.assertFalse(self.mode._is_ecr_image(image))
        self.assertIsNone(self.mode._ecr_registry_host(image))

    def test_ecr_substrings_in_repo_path_do_not_classify_as_ecr(self):
        image = "evil.example.com/a.dkr.ecr.b.amazonaws.com:latest"
        self.assertFalse(self.mode._is_ecr_image(image))
        self.assertIsNone(self.mode._ecr_registry_host(image))

    def test_public_image_is_not_ecr(self):
        for image in (
            "docker.io/library/nginx:latest",
            "nginx:latest",
            "ubuntu",
        ):
            self.assertFalse(self.mode._is_ecr_image(image))
            self.assertIsNone(self.mode._ecr_registry_host(image))

    def test_bad_account_id_length_is_not_ecr(self):
        # 11 digits instead of 12 must not match.
        image = "12345678901.dkr.ecr.us-east-1.amazonaws.com/repo:tag"
        self.assertFalse(self.mode._is_ecr_image(image))


class TestPullImageLoginTarget(unittest.TestCase):
    """Ensure docker login is only ever pointed at the validated ECR host."""

    @patch("sagemaker.serve.mode.local_container_mode.subprocess.run")
    @patch("sagemaker.serve.mode.local_container_mode._get_docker_client")
    def test_login_uses_validated_host_for_valid_ecr(self, mock_client, mock_run):
        import base64

        mode = _bare_mode()
        mode.client = MagicMock()
        mock_client.return_value = mode.client
        mode.ecr = MagicMock()
        token = base64.b64encode(b"AWS:secret-token").decode("utf-8")
        mode.ecr.get_authorization_token.return_value = {
            "authorizationData": [{"authorizationToken": token}]
        }

        image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest"
        mode._pull_image(image)

        # docker login must target the validated ECR host, never any other segment.
        self.assertTrue(mock_run.called)
        login_args = mock_run.call_args[0][0]
        self.assertEqual(login_args[0:2], ["docker", "login"])
        self.assertEqual(login_args[-1], "123456789012.dkr.ecr.us-east-1.amazonaws.com")

    @patch("sagemaker.serve.mode.local_container_mode.subprocess.run")
    @patch("sagemaker.serve.mode.local_container_mode._get_docker_client")
    def test_no_login_for_attacker_crafted_uri(self, mock_client, mock_run):
        mode = _bare_mode()
        mode.client = MagicMock()
        mock_client.return_value = mode.client
        mode.ecr = MagicMock()

        image = "attacker.com/x.dkr.ecr.us-east-1.amazonaws.com/repo:tag"
        mode._pull_image(image)

        # The crafted URI is not ECR: no token fetched, no docker login, no credential leak.
        mode.ecr.get_authorization_token.assert_not_called()
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
