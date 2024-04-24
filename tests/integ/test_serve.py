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
import shutil
import pytest
import pathlib
import collections
import platform
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sagemaker import serve, image_uris
from tests.integ import DATA_DIR
from tests.integ.timeout import timeout

SKIP_COND_MET = platform.python_version_tuple()[1] != "8"

# TODO: switch Role to SageMakerRole
ROLE = "arn:aws:iam::079808353510:role/service-role/AmazonSageMaker-ExecutionRole-20220210T102952"
SERVE_DIR = os.path.join(DATA_DIR, "serve")
XGB_RESOURCE_DIR = os.path.join(SERVE_DIR, "xgboost")
DESTINATION_BUCKET = "s3:/bucket/serve/"

# Fixtures
XgbHappyFns = collections.namedtuple("XgbHappyFns", "prepare_fn load_fn predict_fn")
XgbTrainTestSplit = collections.namedtuple("XgbTrainTestSplit", "x_train y_train x_test y_test")


@pytest.fixture
def base_user_settings(sagemaker_session, cpu_instance_type):
    return {
        "role_arn": ROLE,
        "s3_model_data_url": DESTINATION_BUCKET,
        "sagemaker_session": sagemaker_session,
        "instance_type": cpu_instance_type,
        "port": 8080,
    }


@pytest.fixture
def xgb_user_settings(sagemaker_session, base_user_settings):
    base_user_settings["image"] = image_uris.retrieve(
        "xgboost", sagemaker_session._region_name, "1.7-1", "py38"
    )
    base_user_settings["model_path"] = os.path.join(XGB_RESOURCE_DIR, "tmp")
    base_user_settings["content_type"] = "application/x-npy"
    base_user_settings["accept_type"] = "application/json"
    return base_user_settings


@pytest.fixture
def xgb_train_test_data():
    dataset = loadtxt(
        os.path.join(XGB_RESOURCE_DIR, "classification_training_data.data.csv"), delimiter=","
    )

    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # split data into train and test sets
    seed = 7
    test_size = 0.33

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )

    return XgbTrainTestSplit(x_train, y_train, x_test, y_test)


# Helper Fns
def xgb_happy_fns(x_train, y_train, serve_settings):
    @serve.prepare
    def prepare_fn(model_dir):
        model_dir_path = pathlib.Path(model_dir)
        model = XGBClassifier()
        model.fit(x_train, y_train)
        model.save_model(model_dir + "/model.xgb")

        whl_path = model_dir_path.joinpath("whl")
        whl_path.mkdir(exist_ok=True)

        # TODO: remove when we launch
        shutil.copy(
            os.path.join(SERVE_DIR, "sagemaker-2.181.1.dev0-py2.py3-none-any.whl"), whl_path
        )

    @serve.load
    def load_fn(model_dir):
        model = XGBClassifier()
        model.load_model(model_dir + "/model.xgb")
        return model

    @serve.invoke(**serve_settings)
    def predict_fn(model, input):
        y_pred = model.predict(input)
        predictions = [round(value) for value in y_pred]
        return predictions

    return XgbHappyFns(prepare_fn, load_fn, predict_fn)


def xgb_happy_infer(fns, model_path, x_test):
    # prepare
    fns.prepare_fn(model_path)

    # load
    model = fns.load_fn(model_path)

    # invoke
    response = fns.predict_fn(model, x_test)

    assert response
    assert isinstance(response, list)
    assert isinstance(response[0], int)


# TODO: introduce cleanup option in serve settings so that we can clean up after ourselves


# XGB Integ Tests
@pytest.mark.skipif(
    SKIP_COND_MET, reason="The goal of these test are to test the serving components of our feature"
)
def test_happy_xgb_in_process(xgb_train_test_data, xgb_user_settings):
    xgb_user_settings["mode"] = serve.Mode.IN_PROCESS

    with timeout(minutes=2):
        try:
            xgb_happy_infer(
                xgb_happy_fns(
                    xgb_train_test_data.x_train, xgb_train_test_data.y_train, xgb_user_settings
                ),
                xgb_user_settings["model_path"],
                xgb_train_test_data.x_test,
            )
        except Exception as e:
            assert False, f"{e} was thrown when running xgb in process test"


@pytest.mark.skipif(
    SKIP_COND_MET, reason="The goal of these test are to test the serving components of our feature"
)
def test_happy_xgb_local_container(xgb_train_test_data, xgb_user_settings):
    xgb_user_settings["mode"] = serve.Mode.LOCAL_CONTAINER

    with timeout(minutes=10):
        try:
            xgb_happy_infer(
                xgb_happy_fns(
                    xgb_train_test_data.x_train, xgb_train_test_data.y_train, xgb_user_settings
                ),
                xgb_user_settings["model_path"],
                xgb_train_test_data.x_test,
            )
        except Exception as e:
            assert False, f"{e} was thrown when running xgb local container test"


@pytest.mark.skipif(
    SKIP_COND_MET, reason="The goal of these test are to test the serving components of our feature"
)
def test_happy_xgb_sagemaker_endpoint(xgb_train_test_data, xgb_user_settings):
    xgb_user_settings["mode"] = serve.Mode.SAGEMAKER_ENDPOINT

    with timeout(minutes=10):
        try:
            xgb_happy_infer(
                xgb_happy_fns(
                    xgb_train_test_data.x_train, xgb_train_test_data.y_train, xgb_user_settings
                ),
                xgb_user_settings["model_path"],
                xgb_train_test_data.x_test,
            )
        except Exception as e:
            assert False, f"{e} was thrown when running xgb sagemaker endpoint test"
