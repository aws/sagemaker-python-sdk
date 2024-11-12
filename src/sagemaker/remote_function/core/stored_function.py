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
"""SageMaker job function serializer/deserializer."""
from __future__ import absolute_import

import os
from dataclasses import dataclass
from typing import Any


from sagemaker.s3 import s3_path_join
from sagemaker.remote_function import logging_config
from sagemaker.remote_function.core.pipeline_variables import Context, resolve_pipeline_variables

import sagemaker.remote_function.core.serialization as serialization
from sagemaker.session import Session


logger = logging_config.get_logger()


FUNCTION_FOLDER = "function"
ARGUMENTS_FOLDER = "arguments"
RESULTS_FOLDER = "results"
EXCEPTION_FOLDER = "exception"
JSON_SERIALIZED_RESULT_KEY = "Result"
JSON_RESULTS_FILE = "results.json"


@dataclass
class _SerializedData:
    """Data class to store serialized function and arguments"""

    func: bytes
    args: bytes


class StoredFunction:
    """Class representing a remote function stored in S3."""

    def __init__(
        self,
        sagemaker_session: Session,
        s3_base_uri: str,
        hmac_key: str,
        s3_kms_key: str = None,
        context: Context = Context(),
        use_torchrun: bool = False,
        nproc_per_node: int = 1,
    ):
        """Construct a StoredFunction object.

        Args:
            sagemaker_session: (sagemaker.session.Session): The underlying sagemaker session which
                AWS service calls are delegated to.
            s3_base_uri: the base uri to which serialized artifacts will be uploaded.
            s3_kms_key: KMS key used to encrypt artifacts uploaded to S3.
            hmac_key: Key used to encrypt serialized and deserialized function and arguments.
            context: Build or run context of a pipeline step.
            use_torchrun: Whether to use torchrun for distributed training.
            nproc_per_node: Number of processes per node for distributed training.
        """
        self.sagemaker_session = sagemaker_session
        self.s3_base_uri = s3_base_uri
        self.s3_kms_key = s3_kms_key
        self.hmac_key = hmac_key
        self.context = context
        self.use_torchrun = use_torchrun
        self.nproc_per_node = nproc_per_node

        self.func_upload_path = s3_path_join(
            s3_base_uri, context.step_name, context.func_step_s3_dir
        )
        self.results_upload_path = s3_path_join(
            s3_base_uri, context.execution_id, context.step_name
        )

    def save(self, func, *args, **kwargs):
        """Serialize and persist the function and arguments.

        Args:
            func: the python function.
            args: the positional arguments to func.
            kwargs: the keyword arguments to func.
        Returns:
            None
        """

        logger.info(
            "Serializing function code to %s", s3_path_join(self.func_upload_path, FUNCTION_FOLDER)
        )
        serialization.serialize_func_to_s3(
            func=func,
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.func_upload_path, FUNCTION_FOLDER),
            s3_kms_key=self.s3_kms_key,
            hmac_key=self.hmac_key,
        )

        logger.info(
            "Serializing function arguments to %s",
            s3_path_join(self.func_upload_path, ARGUMENTS_FOLDER),
        )

        serialization.serialize_obj_to_s3(
            obj=(args, kwargs),
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.func_upload_path, ARGUMENTS_FOLDER),
            hmac_key=self.hmac_key,
            s3_kms_key=self.s3_kms_key,
        )

    def save_pipeline_step_function(self, serialized_data):
        """Upload serialized function and arguments to s3.

        Args:
            serialized_data (_SerializedData): The serialized function
                and function arguments of a function step.
        """

        logger.info(
            "Uploading serialized function code to %s",
            s3_path_join(self.func_upload_path, FUNCTION_FOLDER),
        )
        serialization._upload_payload_and_metadata_to_s3(
            bytes_to_upload=serialized_data.func,
            hmac_key=self.hmac_key,
            s3_uri=s3_path_join(self.func_upload_path, FUNCTION_FOLDER),
            sagemaker_session=self.sagemaker_session,
            s3_kms_key=self.s3_kms_key,
        )

        logger.info(
            "Uploading serialized function arguments to %s",
            s3_path_join(self.func_upload_path, ARGUMENTS_FOLDER),
        )
        serialization._upload_payload_and_metadata_to_s3(
            bytes_to_upload=serialized_data.args,
            hmac_key=self.hmac_key,
            s3_uri=s3_path_join(self.func_upload_path, ARGUMENTS_FOLDER),
            sagemaker_session=self.sagemaker_session,
            s3_kms_key=self.s3_kms_key,
        )

    def load_and_invoke(self) -> Any:
        """Load and deserialize the function and the arguments and then execute it."""

        logger.info(
            "Deserializing function code from %s",
            s3_path_join(self.func_upload_path, FUNCTION_FOLDER),
        )
        func = serialization.deserialize_func_from_s3(
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.func_upload_path, FUNCTION_FOLDER),
            hmac_key=self.hmac_key,
        )

        logger.info(
            "Deserializing function arguments from %s",
            s3_path_join(self.func_upload_path, ARGUMENTS_FOLDER),
        )
        args, kwargs = serialization.deserialize_obj_from_s3(
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.func_upload_path, ARGUMENTS_FOLDER),
            hmac_key=self.hmac_key,
        )

        logger.info("Resolving pipeline variables")
        resolved_args, resolved_kwargs = resolve_pipeline_variables(
            self.context,
            args,
            kwargs,
            hmac_key=self.hmac_key,
            s3_base_uri=self.s3_base_uri,
            sagemaker_session=self.sagemaker_session,
        )

        logger.info("Invoking the function")
        result = func(*resolved_args, **resolved_kwargs)

        logger.info(
            "Serializing the function return and uploading to %s",
            s3_path_join(self.results_upload_path, RESULTS_FOLDER),
        )
        serialization.serialize_obj_to_s3(
            obj=result,
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.results_upload_path, RESULTS_FOLDER),
            hmac_key=self.hmac_key,
            s3_kms_key=self.s3_kms_key,
        )

        if self.context and self.context.serialize_output_to_json:
            logger.info(
                "JSON Serializing the function return and uploading to %s",
                s3_path_join(self.results_upload_path, RESULTS_FOLDER),
            )
            serialization.json_serialize_obj_to_s3(
                obj=result,
                json_key=JSON_SERIALIZED_RESULT_KEY,
                sagemaker_session=self.sagemaker_session,
                s3_uri=s3_path_join(
                    os.path.join(self.results_upload_path, RESULTS_FOLDER, JSON_RESULTS_FILE)
                ),
                s3_kms_key=self.s3_kms_key,
            )
