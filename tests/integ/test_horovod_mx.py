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
from __future__ import absolute_import

import json
import os
import tarfile
from six.moves.urllib.parse import urlparse

import boto3
import pytest

import sagemaker.utils
import tests.integ as integ
from sagemaker.mxnet import MXNet
from tests.integ import timeout

horovod_dir = os.path.join(os.path.dirname(__file__), "..", "data", "horovod")


@pytest.mark.local_mode
@pytest.mark.parametrize("instances, processes", [[1, 1]])
def test_horovod_local_mode(sagemaker_local_session, instances, processes, tmpdir):
    output_path = "file://%s" % tmpdir
    job_name = sagemaker.utils.unique_name_from_base("mx-local-horovod")
    estimator = MXNet(
        entry_point=os.path.join(horovod_dir, "hvd_mnist_mxnet.py"),
        role="SageMakerRole",
        image_name="preprod-mxnet:1.6.0-gpu-py3",
        train_instance_count=1,
        train_instance_type="local_gpu",
        sagemaker_session=sagemaker_local_session,
        py_version=integ.PYTHON_VERSION,
        output_path=output_path,
        framework_version="1.6.0",
        distributions={"mpi": {"enabled": True, "processes_per_host": processes}},
    )

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(job_name=job_name)

        tmp = str(tmpdir)
        extract_files(output_path.replace("file://", ""), tmp)

        size = instances * processes

        for rank in range(size):
            assert read_json("rank-%s" % rank, tmp)["rank"] == rank


def extract_files(output_path, tmpdir):
    with tarfile.open(os.path.join(output_path, "model.tar.gz")) as tar:
        tar.extractall(tmpdir)


def read_json(file, tmp):
    with open(os.path.join(tmp, file)) as f:
        return json.load(f)
