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
"""A base class for starting/accessing apps hosted on Amazon SageMaker Studio"""

from __future__ import absolute_import

import abc
import base64
import json
import logging
import os
import re
import webbrowser

from typing import Optional
import boto3
from sagemaker.session import Session, NOTEBOOK_METADATA_FILE

logger = logging.getLogger(__name__)


class BaseInteractiveApp(abc.ABC):
    """BaseInteractiveApp is a base class for creating/accessing apps hosted on SageMaker."""

    def __init__(
        self,
        region: Optional[str] = None,
    ):
        """Initialize a BaseInteractiveApp object.

        Args:
            region (str): Optional. The AWS Region, e.g. us-east-1. If not specified,
                one is created using the default AWS configuration chain.
                Default: ``None``
        """
        if isinstance(region, str):
            self.region = region
        else:
            try:
                self.region = Session().boto_region_name
            except ValueError:
                raise ValueError(
                    "Failed to get the Region information from the default config. Please either "
                    "pass your Region manually as an input argument or set up the local AWS"
                    " configuration."
                )

        self._sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        # Used to store domain and user profile info retrieved from Studio environment.
        self._domain_id = None
        self._user_profile_name = None
        self._in_studio_env = False
        self._get_domain_and_user()

    def __str__(self):
        """Return str(self)."""
        return f"{type(self).__name__}(region={self.region})"

    def __repr__(self):
        """Return repr(self)."""
        return self.__str__()

    def _get_domain_and_user(self):
        """Get domain id and user profile from Studio environment.

        To verify Studio environment, we check if NOTEBOOK_METADATA_FILE exists
        and domain id and user profile name are present in the file.
        """
        if not os.path.isfile(NOTEBOOK_METADATA_FILE):
            return

        try:
            with open(NOTEBOOK_METADATA_FILE, "rb") as metadata_file:
                metadata = json.loads(metadata_file.read())
        except OSError as err:
            logger.warning("Could not load metadata due to unexpected error. %s", err)
            return

        if "DomainId" in metadata and "UserProfileName" in metadata:
            self._in_studio_env = True
            self._domain_id = metadata.get("DomainId")
            self._user_profile_name = metadata.get("UserProfileName")

    def _get_presigned_url(
        self,
        create_presigned_url_kwargs: dict,
        redirect: Optional[str] = None,
        state: Optional[str] = None,
    ):
        """Generate a presigned URL to access a user's domain / user profile.

        Optional state and redirect parameters can be used to to have presigned URL automatically
        redirect to a specific app and provide modifying data.

        Args:
            create_presigned_url_kwargs (dict): Required. This dictionary should include the
                parameters that will be used when calling create_presigned_domain_url via the boto3
                client. At a minimum, this should include the "DomainId" and "UserProfileName"
                parameters as defined by create_presigned_domain_url's documentation.
                Default: ``None``
            redirect (str): Optional. This value will be appended to the resulting presigned URL
                in the format "&redirect=<redirect parameter>". This is used to automatically
                redirect the user into a specific Studio app.
                Default: ``None``
            state (str): Optional. This value will be appended to the resulting presigned URL
                in the format "&state=<state parameter base64 encoded>". This is used to
                automatically apply a state to the given app. Should be used in conjuction with
                the redirect parameter.
                Default: ``None``

        Returns:
            str: A presigned URL.
        """
        response = self._sagemaker_client.create_presigned_domain_url(**create_presigned_url_kwargs)
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            url = response["AuthorizedUrl"]
        else:
            raise ValueError(
                "An invalid status code was returned when creating a presigned URL."
                f" See response for more: {response}"
            )

        if redirect:
            url += f"&redirect={redirect}"

        if state:
            url += f"&state={base64.b64encode(bytes(state, 'utf-8')).decode('utf-8')}"

        logger.warning(
            "A presigned domain URL was generated. This is sensitive and should not be shared with"
            " others."
        )

        return url

    def _open_url_in_web_browser(self, url: str):
        """Open a URL in the default web browser.

        Args:
            url (str): The URL to open.
        """
        webbrowser.open(url)

    def _validate_job_name(self, job_name: str):
        """Validate training job name format.

        Args:
            job_name (str): The job name to validate.

        Returns:
            bool: Whether the supplied job name is valid.
        """
        job_name_regex = "^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}"
        if not re.fullmatch(job_name_regex, job_name):
            raise ValueError(
                f"Invalid job name. Job name must match regular expression {job_name_regex}"
            )

    def _validate_domain_id(self, domain_id: str):
        """Validate domain id format.

        Args:
            domain_id (str): Required. The domain ID to validate.

        Returns:
            bool: Whether the supplied domain ID is valid.
        """
        if domain_id is None or len(domain_id) > 63:
            return False
        return True

    def _validate_user_profile_name(self, user_profile_name: str):
        """Validate user profile name format.

        Args:
            user_profile_name (str): Required. The user profile name to validate.

        Returns:
            bool: Whether the supplied user profile name is valid.
        """
        user_profile_name_regex = "^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}"
        if user_profile_name is None or not re.fullmatch(
            user_profile_name_regex, user_profile_name
        ):
            return False
        return True

    @abc.abstractmethod
    def get_app_url(self):
        """Abstract method to generate a URL to help access the application in Studio.

        Classes that inherit from BaseInteractiveApp should implement and override with what
        parameters are needed for its specific use case.
        """
