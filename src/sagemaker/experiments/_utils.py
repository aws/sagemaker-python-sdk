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
"""Contains the SageMaker Experiment utility methods."""
from __future__ import absolute_import

import os

import mimetypes
import urllib
from functools import wraps

from sagemaker.apiutils import _utils


def resolve_artifact_name(file_path):
    """Resolve artifact name from given file path.

    If not specified, will auto create one.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: The resolved artifact name.
    """
    _, filename = os.path.split(file_path)
    if filename:
        return filename

    return _utils.name("artifact")


def guess_media_type(file_path):
    """Infer the media type of a file based on its file name.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: The guessed media type.
    """
    file_url = urllib.parse.urljoin("file:", urllib.request.pathname2url(file_path))
    guessed_media_type, _ = mimetypes.guess_type(file_url, strict=False)
    return guessed_media_type


def verify_length_of_true_and_predicted(true_labels, predicted_attrs, predicted_attrs_name):
    """Verify if lengths match between lists of true labels and predicted attributes.

    Args:
        true_labels (list or array): The list of the true labels.
        predicted_attrs (list or array): The list of the predicted labels/probabilities/scores.
        predicted_attrs_name (str): The name of the predicted attributes.

    Raises:
        ValueError: If lengths mismatch between true labels and predicted attributes.
    """
    if len(true_labels) != len(predicted_attrs):
        raise ValueError(
            "Lengths mismatch between true labels and {}: "
            "({} vs {}).".format(predicted_attrs_name, len(true_labels), len(predicted_attrs))
        )


def validate_invoked_inside_run_context(func):
    """A Decorator to force the decorated method called under Run context."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        self_instance = args[0]
        if not self_instance._inside_load_context and not self_instance._inside_init_context:
            raise RuntimeError("This method should be called inside context of 'with' statement.")
        return func(*args, **kwargs)

    return wrapper


def is_already_exist_error(error):
    """Check if the error indicates resource already exists

    Args:
        error (dict): The "Error" field in the response of the
            `botocore.exceptions.ClientError`
    """
    return error["Code"] == "ValidationException" and "already exists" in error["Message"]
