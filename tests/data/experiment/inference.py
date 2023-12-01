#  Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os

sdk_name = "sagemaker-dev-1.0.tar.gz"
code_dir = "/opt/ml/code"

sdk_file = f"{code_dir}/{sdk_name}"
os.system(f"pip install {sdk_file}")

import json
import logging
import pickle as pkl
import boto3
import numpy as np
import sagemaker_xgboost_container.encoder as xgb_encoders


def _get_client_config_in_dict(cfg_in_str) -> dict:
    return json.loads(cfg_in_str) if cfg_in_str else None


from sagemaker.session import Session
from sagemaker.experiments import load_run

boto_session = boto3.Session(region_name=os.environ["AWS_REGION"])

sagemaker_client_config = _get_client_config_in_dict(os.environ.get("SM_CLIENT_CONFIG", None))
sagemaker_metrics_config = _get_client_config_in_dict(os.environ.get("SM_METRICS_CONFIG", None))
sagemaker_client = (
    boto_session.client("sagemaker", **sagemaker_client_config) if sagemaker_client_config else None
)
metrics_client = (
    boto_session.client("sagemaker-metrics", **sagemaker_metrics_config)
    if sagemaker_metrics_config
    else None
)

sagemaker_session = Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_metrics_client=metrics_client,
)


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    with load_run(
        experiment_name=os.environ["EXPERIMENT_NAME"],
        run_name=os.environ["RUN_NAME"],
        sagemaker_session=sagemaker_session,
    ) as run:
        logging.info(f"Run name: {run.run_name}")
        logging.info(f"Experiment name: {run.experiment_name}")
        logging.info(f"Trial component name: {run._trial_component.trial_component_name}")
        run.log_parameters({"p3": 3.0, "p4": 4.0})
        run.log_metric("test-job-load-log-metric", 0.1)

    with load_run(sagemaker_session=sagemaker_session) as run:
        run.log_parameters({"p5": 5.0, "p6": 6})

    model_file = "xgboost-model"
    booster = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return booster


def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.
    Return a DMatrix (an object that can be passed to predict_fn).
    """
    if request_content_type == "text/libsvm":
        return xgb_encoders.libsvm_to_dmatrix(request_body)
    else:
        raise ValueError("Content type {} is not supported.".format(request_content_type))


def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.
    Return a two-dimensional NumPy array where the first columns are predictions
    and the remaining columns are the feature contributions (SHAP values) for that prediction.
    """
    prediction = model.predict(input_data)
    feature_contribs = model.predict(input_data, pred_contribs=True, validate_features=False)
    output = np.hstack((prediction[:, np.newaxis], feature_contribs))
    return output


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    if content_type == "text/csv" or content_type == "application/json":
        return ",".join(str(x) for x in predictions[0])
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
