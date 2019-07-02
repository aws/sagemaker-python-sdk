# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import gzip
import os
import pickle
import sys
import time

from sagemaker import KNN, KNNModel
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


def test_knn_regressor(sagemaker_session):
    job_name = unique_name_from_base("knn")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        knn = KNN(
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            k=10,
            predictor_type="regressor",
            sample_size=500,
            sagemaker_session=sagemaker_session,
        )

        # training labels must be 'float32'
        knn.fit(
            knn.record_set(train_set[0][:200], train_set[1][:200].astype("float32")),
            job_name=job_name,
        )

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = KNNModel(knn.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, "ml.c4.xlarge", endpoint_name=job_name)
        result = predictor.predict(train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None


def test_async_knn_classifier(sagemaker_session):
    job_name = unique_name_from_base("knn")

    with timeout(minutes=5):
        data_path = os.path.join(DATA_DIR, "one_p_mnist", "mnist.pkl.gz")
        pickle_args = {} if sys.version_info.major == 2 else {"encoding": "latin1"}

        # Load the data into memory as numpy arrays
        with gzip.open(data_path, "rb") as f:
            train_set, _, _ = pickle.load(f, **pickle_args)

        knn = KNN(
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            k=10,
            predictor_type="classifier",
            sample_size=500,
            index_type="faiss.IVFFlat",
            index_metric="L2",
            sagemaker_session=sagemaker_session,
        )

        # training labels must be 'float32'
        knn.fit(
            knn.record_set(train_set[0][:200], train_set[1][:200].astype("float32")),
            wait=False,
            job_name=job_name,
        )

        print("Detached from training job. Will re-attach in 20 seconds")
        time.sleep(20)
        print("attaching now...")

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        estimator = KNN.attach(training_job_name=job_name, sagemaker_session=sagemaker_session)
        model = KNNModel(
            estimator.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, "ml.c4.xlarge", endpoint_name=job_name)
        result = predictor.predict(train_set[0][:10])

        assert len(result) == 10
        for record in result:
            assert record.label["score"] is not None
