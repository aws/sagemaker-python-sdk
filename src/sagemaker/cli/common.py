# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

import json
import logging
import os
import shutil
import tarfile
import tempfile

import sagemaker

logger = logging.getLogger(__name__)


class HostCommand(object):
    """Placeholder docstring"""

    def __init__(self, args):
        """
        Args:
            args:
        """
        self.endpoint_name = args.job_name
        self.bucket = args.bucket_name  # may be None
        self.role_name = args.role_name
        self.python = args.python
        self.data = args.data
        self.script = args.script
        self.instance_type = args.instance_type
        self.instance_count = args.instance_count
        self.environment = {k: v for k, v in (kv.split("=") for kv in args.env)}

        self.session = sagemaker.Session()

    def upload_model(self):
        """Placeholder docstring"""
        prefix = "{}/model".format(self.endpoint_name)

        archive = self.create_model_archive(self.data)
        model_uri = self.session.upload_data(path=archive, bucket=self.bucket, key_prefix=prefix)
        shutil.rmtree(os.path.dirname(archive))

        return model_uri

    @staticmethod
    def create_model_archive(src):
        """
        Args:
            src:
        """
        if os.path.isdir(src):
            arcname = "."
        else:
            arcname = os.path.basename(src)

        tmp = tempfile.mkdtemp()
        archive = os.path.join(tmp, "model.tar.gz")

        with tarfile.open(archive, mode="w:gz") as t:
            t.add(src, arcname=arcname)
        return archive

    def create_model(self, model_url):
        """
        Args:
            model_url:
        """
        raise NotImplementedError  # subclasses must override

    def start(self):
        """Placeholder docstring"""
        model_url = self.upload_model()
        model = self.create_model(model_url)
        predictor = model.deploy(
            initial_instance_count=self.instance_count, instance_type=self.instance_type
        )

        return predictor


class TrainCommand(object):
    """Placeholder docstring"""

    def __init__(self, args):
        """
        Args:
            args:
        """
        self.job_name = args.job_name
        self.bucket = args.bucket_name  # may be None
        self.role_name = args.role_name
        self.python = args.python
        self.data = args.data
        self.script = args.script
        self.instance_type = args.instance_type
        self.instance_count = args.instance_count
        self.hyperparameters = self.load_hyperparameters(args.hyperparameters)

        self.session = sagemaker.Session()

    @staticmethod
    def load_hyperparameters(src):
        """
        Args:
            src:
        """
        hp = {}
        if src and os.path.exists(src):
            with open(src, "r") as f:
                hp = json.load(f)
        return hp

    def upload_training_data(self):
        """Placeholder docstring"""
        prefix = "{}/data".format(self.job_name)
        data_url = self.session.upload_data(path=self.data, bucket=self.bucket, key_prefix=prefix)
        return data_url

    def create_estimator(self):
        """Placeholder docstring"""
        raise NotImplementedError  # subclasses must override

    def start(self):
        """Placeholder docstring"""
        data_url = self.upload_training_data()
        estimator = self.create_estimator()
        estimator.fit(data_url)
        logger.debug("code location: %s", estimator.uploaded_code.s3_prefix)
        logger.debug(
            "model location: %s%s/output/model.tar.gz",
            estimator.output_path,
            estimator._current_job_name,
        )
