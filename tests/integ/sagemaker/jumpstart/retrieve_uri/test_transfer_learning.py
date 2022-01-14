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

import pandas as pd


from tests.integ.sagemaker.jumpstart.retrieve_uri.utils import (
    get_model_tarball_full_uri_from_base_uri,
    get_training_dataset_for_model_and_version,
)
from tests.integ.sagemaker.jumpstart.retrieve_uri.inference import (
    EndpointInvoker,
    InferenceJobLauncher,
)
from tests.integ.sagemaker.jumpstart.retrieve_uri.training import TrainingJobLauncher
from sagemaker import environment_variables, image_uris
from sagemaker import script_uris
from sagemaker import model_uris
from sagemaker import hyperparameters


def test_jumpstart_transfer_learning_retrieve_functions(setup):

    model_id, model_version = "huggingface-spc-bert-base-cased", "1.0.0"
    training_instance_type = "ml.p3.2xlarge"
    inference_instance_type = "ml.p2.xlarge"

    # training
    print("Starting training...")
    image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        image_scope="training",
        model_id=model_id,
        model_version=model_version,
        instance_type=training_instance_type,
    )

    script_uri = script_uris.retrieve(
        model_id=model_id, model_version=model_version, script_scope="training"
    )

    model_uri = model_uris.retrieve(
        model_id=model_id, model_version=model_version, model_scope="training"
    )

    default_hyperparameters = hyperparameters.retrieve_default(
        model_id=model_id, model_version=model_version, include_container_hyperparameters=True
    )

    training_job = TrainingJobLauncher(
        image_uri=image_uri,
        script_uri=script_uri,
        model_uri=model_uri,
        hyperparameters=default_hyperparameters,
        instance_type=training_instance_type,
        training_dataset_s3_key=get_training_dataset_for_model_and_version(model_id, model_version),
        base_name="huggingface",
    )

    training_job.create_training_job()
    training_job.wait_until_training_job_complete()

    # inference
    print("Starting inference...")
    image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        image_scope="inference",
        model_id=model_id,
        model_version=model_version,
        instance_type=inference_instance_type,
    )

    script_uri = script_uris.retrieve(
        model_id=model_id, model_version=model_version, script_scope="inference"
    )

    environment_vars = environment_variables.retrieve_default(
        model_id=model_id, model_version=model_version
    )

    inference_job = InferenceJobLauncher(
        image_uri=image_uri,
        script_uri=script_uri,
        model_uri=get_model_tarball_full_uri_from_base_uri(
            training_job.output_tarball_base_path, training_job.training_job_name
        ),
        instance_type=inference_instance_type,
        base_name="huggingface",
        environment_variables=environment_vars,
    )

    inference_job.launch_inference_job()
    inference_job.wait_until_endpoint_in_service()

    endpoint_invoker = EndpointInvoker(
        endpoint_name=inference_job.endpoint_name,
    )

    response = endpoint_invoker.invoke_spc_endpoint(["hello", "world"])
    entail, no_entail = response[0][0], response[0][1]

    assert entail is not None
    assert no_entail is not None

    assert pd.isna(entail) is False
    assert pd.isna(no_entail) is False
