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
import importlib
import logging
import os
import pickle as pkl

import boto3
import numpy as np
import sagemaker_xgboost_container.encoder as xgb_encoders


sm_json = "sagemaker-2017-07-24.normal.json"
metrics_json = "sagemaker-metrics-2022-09-30.normal.json"
beta_sdk = "sagemaker-beta-1.0.tar.gz"
resource_dir = "/opt/ml/code/resources"

os.system("pip install awscli")
os.system(
    f"aws configure add-model --service-model file://{resource_dir}/{sm_json} --service-name sagemaker"
)
os.system(
    f"aws configure add-model --service-model file://{resource_dir}/{metrics_json} --service-name sagemaker-metrics"
)
importlib.reload(boto3)  # Reload boto3 to let the added API models take effect

sdk_file = f"{resource_dir}/{beta_sdk}"
os.system(f"pip install {sdk_file}")

from sagemaker.session import Session
from sagemaker.experiments.run import Run


boto_session = boto3.Session(region_name=os.environ["AWS_REGION"])
sagemaker_session = Session(boto_session=boto_session)


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    with Run.load(
        experiment_name=os.environ["EXPERIMENT_NAME"],
        run_name=os.environ["RUN_NAME"],
        sagemaker_session=sagemaker_session,
    ) as run:
        logging.info(f"Run name: {run.run_name}")
        logging.info(f"Experiment name: {run.experiment_name}")
        logging.info(f"Trial component name: {run._trial_component.trial_component_name}")
        run.log_parameters({"p3": 3.0, "p4": 4.0})
        run.log_metric("test-job-load-log-metric", 0.1)

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
