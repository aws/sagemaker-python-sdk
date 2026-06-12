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
"""Helper for creating AWS Secrets Manager secrets."""
from __future__ import absolute_import

import uuid
from dataclasses import dataclass
from typing import Optional

import boto3


@dataclass
class Secret:
    """A handle to a Secrets Manager secret.

    Supports context-manager use; the secret is deleted on context exit.
    """

    arn: str

    @classmethod
    def from_string(
        cls,
        value: str,
        *,
        name: Optional[str] = None,
        session: Optional[boto3.session.Session] = None,
    ) -> "Secret":
        """Create a new Secrets Manager secret from a plaintext value.

        Note: this helper creates the secret with default permissions only.
        If the consuming workload needs to read the secret at job runtime,
        you may need to attach a resource policy granting
        ``secretsmanager:GetSecretValue`` to the appropriate principal.

        Args:
            value: The plaintext secret value to store.
            name: Optional secret name. Defaults to ``sagemaker-workload-<uuid>``.
            session: Optional boto3 session. Defaults to the ambient session.

        Returns:
            A ``Secret`` referencing the newly created secret.
        """
        secret_name = name or f"sagemaker-workload-{uuid.uuid4()}"
        client = (session or boto3).client("secretsmanager")
        response = client.create_secret(Name=secret_name, SecretString=value)
        return cls(arn=response["ARN"])

    def delete(
        self,
        *,
        force_delete_without_recovery: bool = True,
        session: Optional[boto3.session.Session] = None,
    ) -> None:
        """Delete the underlying Secrets Manager secret.

        Args:
            force_delete_without_recovery: If True (default), delete
                immediately. Pass False to use the 30-day recovery window.
            session: Optional boto3 session. Defaults to the ambient session.
        """
        client = (session or boto3).client("secretsmanager")
        client.delete_secret(
            SecretId=self.arn,
            ForceDeleteWithoutRecovery=force_delete_without_recovery,
        )

    def __enter__(self) -> "Secret":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.delete()
