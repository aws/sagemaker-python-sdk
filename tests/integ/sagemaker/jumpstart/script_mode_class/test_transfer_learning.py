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
import os

from sagemaker import hyperparameters, image_uris, model_uris, script_uris
from sagemaker.estimator import Estimator
from sagemaker.jumpstart.constants import (
    INFERENCE_ENTRY_POINT_SCRIPT_NAME,
    JUMPSTART_DEFAULT_REGION_NAME,
    TRAINING_ENTRY_POINT_SCRIPT_NAME,
)
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.predictor import Predictor
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    JUMPSTART_TAG,
)
from tests.integ.sagemaker.jumpstart.utils import (
    EndpointInvoker,
    get_sm_session,
    get_training_dataset_for_model_and_version,
)


def test_jumpstart_transfer_learning_estimator_class(setup):

    model_id, model_version = "huggingface-spc-bert-base-cased", "1.0.0"
    training_instance_type = "ml.p3.2xlarge"
    inference_instance_type = "ml.p2.xlarge"
    instance_count = 1

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
        model_id=model_id,
        model_version=model_version,
    )

    default_hyperparameters["epochs"] = "1"

    estimator = Estimator(
        image_uri=image_uri,
        source_dir=script_uri,
        model_uri=model_uri,
        entry_point=TRAINING_ENTRY_POINT_SCRIPT_NAME,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        enable_network_isolation=True,
        hyperparameters=default_hyperparameters,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        instance_count=instance_count,
        instance_type=training_instance_type,
    )

    estimator.fit(
        {
            "training": f"s3://{get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }
    )

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

    model_uri = model_uris.retrieve(
        model_id=model_id, model_version=model_version, model_scope="inference"
    )

    predictor: Predictor = estimator.deploy(
        initial_instance_count=instance_count,
        instance_type=inference_instance_type,
        entry_point=INFERENCE_ENTRY_POINT_SCRIPT_NAME,
        image_uri=image_uri,
        source_dir=script_uri,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
    )

    endpoint_invoker = EndpointInvoker(
        endpoint_name=predictor.endpoint_name,
    )

    response = endpoint_invoker.invoke_spc_endpoint(["hello", "world"])

    assert response is not None
