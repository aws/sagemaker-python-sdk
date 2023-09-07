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

from sagemaker import (
    environment_variables,
    hyperparameters,
    instance_types,
    metric_definitions,
    image_uris,
    model_uris,
    script_uris,
)
from sagemaker.estimator import Estimator
from sagemaker.jumpstart.artifacts import (
    _retrieve_estimator_init_kwargs,
    _retrieve_estimator_fit_kwargs,
)
from sagemaker.jumpstart.artifacts.kwargs import (
    _retrieve_model_deploy_kwargs,
    _retrieve_model_init_kwargs,
)
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

    model_id, model_version = "huggingface-spc-bert-base-cased", "1.2.3"

    inference_instance_type = instance_types.retrieve_default(
        model_id=model_id, model_version=model_version, scope="inference"
    )
    training_instance_type = instance_types.retrieve_default(
        model_id=model_id, model_version=model_version, scope="training"
    )
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

    default_metric_definitions = metric_definitions.retrieve_default(
        model_id=model_id,
        model_version=model_version,
    )

    estimator_kwargs = _retrieve_estimator_init_kwargs(
        model_id=model_id,
        model_version=model_version,
        instance_type=training_instance_type,
    )

    # Avoid exceeding resource limits
    if "max_run" in estimator_kwargs:
        del estimator_kwargs["max_run"]

    estimator = Estimator(
        image_uri=image_uri,
        source_dir=script_uri,
        model_uri=model_uri,
        entry_point=TRAINING_ENTRY_POINT_SCRIPT_NAME,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        hyperparameters=default_hyperparameters,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        instance_count=instance_count,
        instance_type=training_instance_type,
        metric_definitions=default_metric_definitions,
        **estimator_kwargs,
    )

    fit_kwargs = _retrieve_estimator_fit_kwargs(
        model_id=model_id,
        model_version=model_version,
    )

    estimator.fit(
        {
            "training": f"s3://{get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        },
        **fit_kwargs,
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

    env = environment_variables.retrieve_default(
        model_id=model_id,
        model_version=model_version,
        include_aws_sdk_env_vars=False,
    )
    model_kwargs = _retrieve_model_init_kwargs(
        model_id=model_id,
        model_version=model_version,
    )

    deploy_kwargs = _retrieve_model_deploy_kwargs(
        model_id=model_id,
        model_version=model_version,
        instance_type=inference_instance_type,
    )

    predictor: Predictor = estimator.deploy(
        initial_instance_count=instance_count,
        instance_type=inference_instance_type,
        entry_point=INFERENCE_ENTRY_POINT_SCRIPT_NAME,
        image_uri=image_uri,
        source_dir=script_uri,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        env=env,
        **model_kwargs,
        **deploy_kwargs,
    )

    endpoint_invoker = EndpointInvoker(
        endpoint_name=predictor.endpoint_name,
    )

    response = endpoint_invoker.invoke_spc_endpoint(["hello", "world"])

    assert response is not None
