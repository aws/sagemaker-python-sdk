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

from sagemaker import image_uris, instance_types, model_uris, script_uris, environment_variables
from sagemaker.jumpstart.constants import INFERENCE_ENTRY_POINT_SCRIPT_NAME
from sagemaker.jumpstart.artifacts import _retrieve_kwargs
from sagemaker.jumpstart.enums import EnvVariableUseCase, KwargUseCase
from sagemaker.model import Model
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    JUMPSTART_TAG,
    InferenceTabularDataname,
)
from tests.integ.sagemaker.jumpstart.utils import (
    EndpointInvoker,
    download_inference_assets,
    get_sm_session,
    get_tabular_data,
)


def test_jumpstart_inference_model_class(setup):

    model_id, model_version = "catboost-classification-model", "1.2.7"

    instance_type = instance_types.retrieve_default(
        model_id=model_id, model_version=model_version, scope="inference"
    )
    instance_count = 1

    print("Starting inference...")

    image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        image_scope="inference",
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
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
        use_case=EnvVariableUseCase.SAGEMAKER_PYTHON_SDK,
    )
    model_kwargs = _retrieve_kwargs(
        model_id=model_id, model_version=model_version, use_case=KwargUseCase.MODEL
    )

    model = Model(
        image_uri=image_uri,
        model_data=model_uri,
        source_dir=script_uri,
        entry_point=INFERENCE_ENTRY_POINT_SCRIPT_NAME,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        enable_network_isolation=True,
        env=env,
        **model_kwargs,
    )

    deploy_kwargs = _retrieve_kwargs(
        model_id=model_id, model_version=model_version, use_case=KwargUseCase.MODEL_DEPLOY
    )

    model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        **deploy_kwargs,
    )

    endpoint_invoker = EndpointInvoker(
        endpoint_name=model.endpoint_name,
    )

    download_inference_assets()
    ground_truth_label, features = get_tabular_data(InferenceTabularDataname.MULTICLASS)

    response = endpoint_invoker.invoke_tabular_endpoint(features)

    assert response is not None
