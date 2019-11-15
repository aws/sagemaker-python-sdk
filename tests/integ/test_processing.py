# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

from sagemaker.processing import ProcessingInput
from sagemaker.sklearn.processing import SKLearnProcessor
from tests.integ import DATA_DIR

# TODO-reinvent-2019: Replace this role ARN
ROLE = "arn:aws:iam::142577830533:role/SageMakerRole"


def test_sklearn(sagemaker_beta_session, sklearn_full_version, cpu_instance_type):
    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")

    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_full_version,
        role=ROLE,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_beta_session,
    )

    sklearn_processor.run(
        command=["python3"],
        code=script_path,
        inputs=[ProcessingInput(source=input_file_path, destination="/inputs/")],
        wait=False,
        logs=False,
    )

    job_description = sklearn_processor.latest_job.describe()

    assert len(job_description["ProcessingInputs"]) == 2
    assert job_description["ProcessingResources"] == {
        "ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m4.xlarge", "VolumeSizeInGB": 30}
    }
    assert job_description["StoppingCondition"] == {"MaxRuntimeInSeconds": 86400}
    assert job_description["AppSpecification"]["ContainerEntrypoint"] == [
        "python3",
        "/input/code/dummy_script.py",
    ]
    assert job_description["RoleArn"] == ROLE
