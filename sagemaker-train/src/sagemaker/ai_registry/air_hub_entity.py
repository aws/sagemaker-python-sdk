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
"""Base entity class for AI Registry Hub content."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional

from rich.console import Group
from rich.live import Live  
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.status import Status
from rich.style import Style
from sagemaker.core.utils.code_injection.constants import Color
from sagemaker.core.utils.exceptions import FailedStatusError, TimeoutExceededError

from sagemaker.ai_registry.air_constants import (
    HubContentStatus,
    RESPONSE_KEY_HUB_CONTENT_ARN,
    RESPONSE_KEY_HUB_CONTENT_NAME,
    RESPONSE_KEY_HUB_CONTENT_STATUS,
    RESPONSE_KEY_HUB_CONTENT_VERSION,
    RESPONSE_KEY_CREATION_TIME,
)
from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.core.helper.session_helper import Session

class AIRHubEntity(ABC):
    """Base entity for AI Registry Hub content."""

    def __init__(
        self,
        name: str,
        version: str,
        arn: str,
        status: Optional[HubContentStatus] = None,
        created_time: Optional[str] = None,
        updated_time: Optional[str] = None,
        description: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> None:
        """Initialize AIR Hub Entity.
        
        Args:
            name: Name of the hub content
            version: Version of the hub content
            arn: ARN of the hub content
            status: Status of the hub content
            created_time: Creation timestamp
            updated_time: Last update timestamp
            description: Description of the hub content
            role: Optional IAM role ARN
            sagemaker_session: Optional SageMaker session
        """
        self.name = name
        self.version = version
        self.arn = arn
        self.hub_name = AIRHub.get_hub_name()
        self.status = status
        self.created = created_time
        self.updated = updated_time
        self.description = description
        self.sagemaker_session = sagemaker_session

    @property
    @abstractmethod
    def hub_content_type(self) -> str:
        """Return the hub content type for this entity."""
        pass

    @classmethod
    @abstractmethod
    def _get_hub_content_type_for_list(cls) -> str:
        """Return the hub content type for list operation."""
        pass

    @classmethod
    def list(cls, max_results: Optional[int] = None, next_token: Optional[str] = None) -> List:
        """List all entities of this type.
        
        Args:
            max_results: Maximum number of results to return
            next_token: Token for pagination
            
        Returns:
            List of hub content entities
        """
        return AIRHub.list_hub_content(cls._get_hub_content_type_for_list(), max_results, next_token)

    def get_versions(self) -> List:
        """List all versions of this entity.
        
        Returns:
            List of version information dictionaries
        """
        versions = AIRHub.list_hub_content_versions(self.hub_content_type, self.name)
        return [{
            "version": v.get(RESPONSE_KEY_HUB_CONTENT_VERSION),
            "name": v.get(RESPONSE_KEY_HUB_CONTENT_NAME),
            "arn": v.get(RESPONSE_KEY_HUB_CONTENT_ARN),
            "status": v.get(RESPONSE_KEY_HUB_CONTENT_STATUS),
            "created_time": v.get(RESPONSE_KEY_CREATION_TIME)
        } for v in versions]

    def delete(self, version: Optional[str] = None) -> bool:
        """Delete this entity instance.
        
        Args:
            version: Specific version to delete. If None, deletes all versions.
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if version is None:
                # If a version is not provided, delete all versions
                versions = AIRHub.list_hub_content_versions(self.hub_content_type, self.name)
                for v in versions:
                    AIRHub.delete_hub_content(self.hub_content_type, self.name, v[RESPONSE_KEY_HUB_CONTENT_VERSION])
            else:
                AIRHub.delete_hub_content(self.hub_content_type, self.name, version)
            return True
        except Exception:
            return False

    @classmethod
    def delete_by_name(cls, name: str, version: Optional[str] = None) -> bool:
        """Delete entity by name and version.
        
        Args:
            name: Name of the entity to delete
            version: Specific version to delete. If None, deletes all versions.
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if version is None:
                # If a version is not provided, delete all versions
                versions = AIRHub.list_hub_content_versions(cls._get_hub_content_type_for_list(), name)
                for v in versions:
                    AIRHub.delete_hub_content(cls._get_hub_content_type_for_list(), name, v[RESPONSE_KEY_HUB_CONTENT_VERSION])
            else:
                AIRHub.delete_hub_content(cls._get_hub_content_type_for_list(), name, version)
            return True
        except Exception:
            return False

    def wait(
        self,
        poll: int = 5,
        timeout: Optional[int] = None,
    ) -> None:
        """Wait for AIR Hub Entity to reach a terminal state.

        Args:
            poll: The number of seconds to wait between each poll.
            timeout: The maximum number of seconds to wait before timing out.

        Raises:
            TimeoutExceededError: If the resource does not reach a terminal state before timeout.
            FailedStatusError: If the resource reaches a failed state.
        """
        terminal_states = ["Available", "ImportFailed"]
        start_time = time.time()

        progress = Progress(
            SpinnerColumn("bouncingBar"),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
        )
        progress.add_task("Waiting for AIRegistry object creation...")
        status = Status("Current status:")

        with Live(
            Panel(
                Group(progress, status),
                title="Wait Log Panel",
                border_style=Style(color=Color.BLUE.value),
            ),
            transient=True,
        ):
            while True:
                self.refresh()
                current_status = self.status
                status.update(f"Current status: [bold]{current_status}")

                if current_status in terminal_states:
                    print(f"Final Resource Status: {current_status}")

                    if "failed" in str(current_status).lower():
                        raise FailedStatusError(
                            resource_type="AIRHubEntity",
                            status=str(current_status),
                            reason=f"AI Registry hub entity '{self.name}' (version {self.version}) failed to import. "
                                   f"Check CloudWatch logs or contact AWS support for assistance."
                        )
                    return

                if timeout is not None and time.time() - start_time >= timeout:
                    raise TimeoutExceededError(
                        resource_type="AIRHubEntity", status=current_status
                    )
                time.sleep(poll)

    def refresh(self) -> None:
        """Refresh entity status from hub content."""
        response = AIRHub.describe_hub_content(self.hub_content_type, self.name, self.version)
        self.status = response.get(RESPONSE_KEY_HUB_CONTENT_STATUS)
