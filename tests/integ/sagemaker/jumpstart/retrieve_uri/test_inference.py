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


from tests.integ.sagemaker.jumpstart.retrieve_uri.inference import (
    InferenceJobLauncher,
)
from sagemaker import environment_variables, image_uris
from sagemaker import script_uris
from sagemaker import model_uris

from tests.integ.sagemaker.jumpstart.constants import InferenceTabularDataname

from tests.integ.sagemaker.jumpstart.utils import (
    download_inference_assets,
    get_tabular_data,
    EndpointInvoker,
)


def test_jumpstart_inference_retrieve_functions(setup):

    model_id, model_version = "catboost-classification-model", "2.1.6"
    instance_type = "ml.m5.4xlarge"

    print("Starting inference...")

    image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        image_scope="inference",
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
        tolerate_vulnerable_model=True,
    )

    model_uri = model_uris.retrieve(
        model_id=model_id,
        model_version=model_version,
        model_scope="inference",
        tolerate_vulnerable_model=True,
    )

    environment_vars = environment_variables.retrieve_default(
        model_id=model_id,
        model_version=model_version,
        tolerate_vulnerable_model=True,
    )

    inference_job = InferenceJobLauncher(
        image_uri=image_uri,
        script_uri=None,
        model_uri=model_uri,
        instance_type=instance_type,
        base_name="catboost",
        environment_variables=environment_vars,
    )

    inference_job.launch_inference_job()
    inference_job.wait_until_endpoint_in_service()

    endpoint_invoker = EndpointInvoker(
        endpoint_name=inference_job.endpoint_name,
    )

    download_inference_assets()
    ground_truth_label, features = get_tabular_data(InferenceTabularDataname.MULTICLASS)

    response = endpoint_invoker.invoke_tabular_endpoint(features)

    assert response is not None
