# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This module contains code related to SparkML Processors, which are used
for Analytics jobs. These jobs let customers perform data pre-processing,
post-processing, feature engineering, data validation, and model evaluation
and interpretation on SageMaker.
"""
from __future__ import absolute_import

import os
from six.moves.urllib.parse import urlparse

from sagemaker import Session
from sagemaker.processor import Processor, FileInput
from sagemaker.fw_utils import create_image_uri
from sagemaker.s3 import S3Uploader


class SparkMLJavaProcessor(Processor):
    """Handles Amazon SageMaker processing tasks for jobs using SparkML with Java."""

    def __init__(
        self,
        framework_version,
        role,
        processing_instance_count,
        processing_instance_type,
        submit_app_class,
        image_uri=None,
        arguments=None,
        processing_volume_size_in_gb=30,
        processing_volume_kms_key=None,
        processing_max_runtime_in_seconds=24 * 60 * 60,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        session = sagemaker_session or Session()
        region = session.boto_region_name
        image = image_uri or create_image_uri(
            region=region,
            framework="sparkml",
            instance_type=processing_instance_type,
            framework_version=framework_version,
        )
        env_vars = env or {}
        env_vars["SUBMIT_APP_CLASS"] = submit_app_class

        super(SparkMLJavaProcessor, self).__init__(
            role=role,
            image_uri=image,
            processing_instance_count=processing_instance_count,
            processing_instance_type=processing_instance_type,
            arguments=arguments,
            processing_volume_size_in_gb=processing_volume_size_in_gb,
            processing_volume_kms_key=processing_volume_kms_key,
            processing_max_runtime_in_seconds=processing_max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=session,
            env=env_vars,
            tags=tags,
            network_config=network_config,
        )

    def _submit_app_input(self, submit_app):
        """Turn a submit_app string into a FileInput object.

        Args:
            submit_app (str): path to a local file or an S3 uri file

        Returns:
            sagemaker.processor.FileInput:
        """
        parse_result = urlparse(submit_app)
        if parse_result.scheme != "s3":
            desired_s3_uri = os.path.join(
                "s3://", self.sagemaker_session.default_bucket(), self._current_job_name, "code"
            )
            s3_uri = S3Uploader.upload(
                local_path=submit_app, desired_s3_uri=desired_s3_uri, session=self.sagemaker_session
            )
        else:
            s3_uri = submit_app

        file_input = FileInput(
            source=s3_uri, destination="/code/submit_app", input_name="submit_app"
        )

        return file_input

    def _submit_app_jars_input(self, submit_app_jars):
        """Turns a submit_app_jars string into a FileInput object.

        Args:
            submit_app_jars (str): local directory path or an S3 uri directory

        Returns:
            sagemaker.processor.FileInput
        """
        parse_result = urlparse(submit_app_jars)
        if parse_result.scheme != "s3":
            desired_s3_uri = os.path.join(
                "s3://", self.sagemaker_session.default_bucket(), self._current_job_name, "code"
            )
            s3_uri = S3Uploader.upload(
                local_path=submit_app_jars,
                desired_s3_uri=desired_s3_uri,
                session=self.sagemaker_session,
            )
        else:
            s3_uri = submit_app_jars

        file_input = FileInput(
            source=s3_uri, destination="/code/submit_app_jars", input_name="submit_app_jars"
        )

        return file_input

    def run(
        self,
        submit_app,
        submit_app_jars,
        inputs=None,
        outputs=None,
        wait=True,
        logs=True,
        job_name=None,
    ):
        """Run a processing job using Spark Java.

        Args:
            submit_app (str or sagemaker.processor.FileInput):
            submit_app_jars (str): A local path or S3 uri to a directory.
            inputs ([sagemaker.processor.FileInput]): Input files for the processing
                job. These must be provided as FileInput objects.
            outputs ([str or sagemaker.processor.FileOutput]): Outputs for the processing
                job. These can be specified as either a path string or a FileOutput
                object.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        submit_app_input = self._submit_app_input(submit_app)
        submit_app_jars_input = self._submit_app_jars_input(submit_app_jars)

        submit_inputs = (inputs or []) + [submit_app_input, submit_app_jars_input]

        super(SparkMLJavaProcessor, self).run(
            inputs=submit_inputs, outputs=outputs, wait=wait, logs=logs, job_name=job_name
        )


class SparkMLPythonProcessor(Processor):
    """Handles Amazon SageMaker processing tasks for jobs using SparkML with Python."""

    def __init__(
        self,
        framework_version,
        role,
        processing_instance_count,
        processing_instance_type,
        py_version="py3",
        image_uri=None,
        arguments=None,
        processing_volume_size_in_gb=30,
        processing_volume_kms_key=None,
        processing_max_runtime_in_seconds=24 * 60 * 60,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        session = sagemaker_session or Session()
        region = session.boto_region_name
        image = image_uri or create_image_uri(
            region=region,
            framework="sparkml",
            instance_type=processing_instance_type,
            framework_version=framework_version,
            py_version=py_version,
        )

        super(SparkMLPythonProcessor, self).__init__(
            role=role,
            image_uri=image,
            processing_instance_count=processing_instance_count,
            processing_instance_type=processing_instance_type,
            arguments=arguments,
            processing_volume_size_in_gb=processing_volume_size_in_gb,
            processing_volume_kms_key=processing_volume_kms_key,
            processing_max_runtime_in_seconds=processing_max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=session,
            env=env,
            tags=tags,
            network_config=network_config,
        )

    def _submit_app_input(self, submit_app):
        """Turn a submit_app string into a FileInput object.

        Args:
            submit_app (str): path to a local file or an S3 uri file

        Returns:
            sagemaker.processor.FileInput:
        """
        parse_result = urlparse(submit_app)
        if parse_result.scheme != "s3":
            desired_s3_uri = os.path.join(
                "s3://", self.sagemaker_session.default_bucket(), self._current_job_name, "code"
            )
            s3_uri = S3Uploader.upload(
                local_path=submit_app, desired_s3_uri=desired_s3_uri, session=self.sagemaker_session
            )
        else:
            s3_uri = submit_app

        file_input = FileInput(
            source=s3_uri, destination="/code/submit_app", input_name="submit_app"
        )

        return file_input

    def _py_files_input(self, py_files):
        """Turns a py_files string into a FileInput object.

        Args:
            py_files (str): local directory path or an S3 uri directory

        Returns:
            sagemaker.processor.FileInput
        """
        parse_result = urlparse(py_files)
        if parse_result.scheme != "s3":
            desired_s3_uri = os.path.join(
                "s3://", self.sagemaker_session.default_bucket(), self._current_job_name, "code"
            )
            s3_uri = S3Uploader.upload(
                local_path=py_files, desired_s3_uri=desired_s3_uri, session=self.sagemaker_session
            )
        else:
            s3_uri = py_files

        file_input = FileInput(source=s3_uri, destination="/code/py_files", input_name="py_files")

        return file_input

    def run(
        self, submit_app, py_files, inputs=None, outputs=None, wait=True, logs=True, job_name=None
    ):
        """Run a processing job using Spark Python.

        Args:
            submit_app (str or sagemaker.processor.FileInput):
            py_files (str): A local path or S3 uri to a directory.
            inputs ([sagemaker.processor.FileInput]): Input files for the processing
                job. These must be provided as FileInput objects.
            outputs ([str or sagemaker.processor.FileOutput]): Outputs for the processing
                job. These can be specified as either a path string or a FileOutput
                object.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        submit_app_input = self._submit_app_input(submit_app)
        py_files_input = self._py_files_input(py_files)

        submit_inputs = (inputs or []) + [submit_app_input, py_files_input]

        super(SparkMLPythonProcessor, self).run(
            inputs=submit_inputs, outputs=outputs, wait=wait, logs=logs, job_name=job_name
        )
