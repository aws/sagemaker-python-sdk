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
# from __future__ import absolute_import

# import os
# import pytest
# import platform
# import collections
# from numpy import loadtxt
# from sagemaker.serve.spec.inference_spec import InferenceSpec

# if platform.python_version_tuple()[1] == "8":
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split

# from tests.integ.sagemaker.serve.constants import XGB_RESOURCE_DIR


# XgbTestSplit = collections.namedtuple("XgbTrainTestSplit", "x_test y_test")


# @pytest.fixture(scope="session")
# def loaded_xgb_model():
#     model = XGBClassifier()
#     model.load_model(XGB_RESOURCE_DIR + "/model.xgb")
#     return model


# @pytest.fixture(scope="session")
# def xgb_inference_spec():
#     class MyXGBoostModel(InferenceSpec):
#         def load(self, model_dir: str):
#             model = XGBClassifier()
#             model.load_model(model_dir + "/model.xgb")
#             return model

#         def invoke(
#             self,
#             input: object,
#             model: object,
#         ):
#             y_pred = model.predict(input)
#             predictions = [round(value) for value in y_pred]
#             return predictions

#     return MyXGBoostModel()


# @pytest.fixture(scope="session")
# def xgb_test_sets():
#     dataset = loadtxt(
#         os.path.join(XGB_RESOURCE_DIR, "classification_training_data.data.csv"), delimiter=","
#     )

#     X = dataset[:, 0:8]
#     Y = dataset[:, 8]

#     seed = 7
#     test_size = 0.33

#     _, x_test, _, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#     return XgbTestSplit(x_test, y_test)
