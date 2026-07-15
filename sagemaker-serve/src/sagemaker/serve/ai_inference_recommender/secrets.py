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
from dataclasses import dataclass, field
from typing import Any, Optional

import boto3


@dataclass
class Secret:
    """A handle to a Secrets Manager secret.

    Supports context-manager use. On context exit, only a secret this object
    created (via :meth:`from_string`) is deleted; a secret you merely wrapped
    by ARN is left untouched.
    """

    arn: str
    # Set only by from_string(): whether this handle owns the underlying secret
    # (safe to auto-delete) and the session used to create it (reused on delete).
    _created: bool = field(default=False, repr=False)
    _session: Optional[Any] = field(default=None, repr=False)

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
        return cls(arn=response["ARN"], _created=True, _session=session)

    def delete(
        self,
        *,
        force_delete_without_recovery: bool = False,
        session: Optional[boto3.session.Session] = None,
    ) -> None:
        """Delete the underlying Secrets Manager secret.

        Args:
            force_delete_without_recovery: If True, delete immediately with no
                recovery window. Defaults to False, which uses Secrets Manager's
                recovery window so an accidental delete can be restored.
            session: Optional boto3 session. Defaults to the session that
                created this secret, then to the ambient session.
        """
        client = (session or self._session or boto3).client("secretsmanager")
        client.delete_secret(
            SecretId=self.arn,
            ForceDeleteWithoutRecovery=force_delete_without_recovery,
        )

    def __enter__(self) -> "Secret":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Only auto-delete a secret this object created; never delete a
        # pre-existing secret that was merely wrapped by ARN.
        if self._created:
            self.delete()
