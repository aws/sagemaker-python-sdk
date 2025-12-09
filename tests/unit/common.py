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



from botocore.exceptions import ClientError


def _raise_unexpected_client_error(**kwargs):
    response = {
        "Error": {"Code": "ValidationException", "Message": "Name does not satisfy expression."}
    }
    raise ClientError(error_response=response, operation_name="foo")


def _raise_does_not_exist_client_error(**kwargs):
    response = {"Error": {"Code": "ValidationException", "Message": "Could not find entity."}}
    raise ClientError(error_response=response, operation_name="foo")


def _raise_does_already_exists_client_error(**kwargs):
    response = {"Error": {"Code": "ValidationException", "Message": "Resource already exists."}}
    raise ClientError(error_response=response, operation_name="foo")


def _raise_access_denied_client_error(**kwargs):
    response = {"Error": {"Code": "AccessDeniedException", "Message": "Could not access entity."}}
    raise ClientError(error_response=response, operation_name="foo")
