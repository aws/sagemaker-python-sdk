# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase, registry
from sagemaker.amazon.common import numpy_to_record_serializer, record_deserializer
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import ge, le
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session


class RandomCutForest(AmazonAlgorithmEstimatorBase):

    repo_name = 'randomcutforest'
    repo_version = 1
    MINI_BATCH_SIZE = 1000

    eval_metrics = hp(name='eval_metrics',
                      validation_message='A comma separated list of "accuracy" or "precision_recall_fscore"',
                      data_type=list)

    num_trees = hp('num_trees', (ge(50), le(1000)), 'An integer in [50, 1000]', int)
    num_samples_per_tree = hp('num_samples_per_tree', (ge(1), le(2048)), 'An integer in [1, 2048]', int)
    feature_dim = hp("feature_dim", (ge(1), le(10000)), 'An integer in [1, 10000]', int)

    def __init__(self, role, train_instance_count, train_instance_type,
                 num_samples_per_tree=None, num_trees=None, eval_metrics=None, **kwargs):
        """RandomCutForest is :class:`Estimator` used for anomaly detection.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`. It requires Amazon
        :class:`~sagemaker.amazon.record_pb2.Record` protobuf serialized data to be stored in S3.
        There is an utility :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.record_set` that
        can be used to upload data to S3 and creates :class:`~sagemaker.amazon.amazon_estimator.RecordSet` to be passed
        to the `fit` call.

        To learn more about the Amazon protobuf Record class and how to prepare bulk data in this format, please
        consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        After this Estimator is fit, model data is stored in S3. The model may be deployed to an Amazon SageMaker
        Endpoint by invoking :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as deploying an
        Endpoint, deploy returns a :class:`~sagemaker.amazon.ntm.RandomCutForestPredictor` object that can be used
        for inference calls using the trained model hosted in the SageMaker Endpoint.

        RandomCutForest Estimators can be configured by setting hyperparameters. The available hyperparameters for
        RandomCutForest are documented below.

        For further information on the AWS Random Cut Forest algorithm,
        please consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            num_samples_per_tree (int): Optional. The number of samples used to build each tree in the forest.
                The total number of samples drawn from the train dataset is num_trees * num_samples_per_tree.
            num_trees (int): Optional. The number of trees used in the forest.
            eval_metrics(list): Optional. JSON list of metrics types to be used for reporting the score for the model.
                Allowed values are "accuracy", "precision_recall_fscore": positive and negative precision, recall,
                and f1 scores. If test data is provided, the score shall be reported in terms of all requested metrics.
            **kwargs: base class keyword argument values.
        """

        super(RandomCutForest, self).__init__(role, train_instance_count, train_instance_type, **kwargs)
        self.num_samples_per_tree = num_samples_per_tree
        self.num_trees = num_trees
        self.eval_metrics = eval_metrics

    def create_model(self):
        """Return a :class:`~sagemaker.amazon.RandomCutForestModel` referencing the latest
        s3 model data produced by this Estimator."""

        return RandomCutForestModel(self.model_data, self.role, sagemaker_session=self.sagemaker_session)

    def fit(self, records, mini_batch_size=None, **kwargs):
        if mini_batch_size is None:
            mini_batch_size = RandomCutForest.MINI_BATCH_SIZE
        elif mini_batch_size != RandomCutForest.MINI_BATCH_SIZE:
            raise ValueError("Random Cut Forest uses a fixed mini_batch_size of {}"
                             .format(RandomCutForest.MINI_BATCH_SIZE))
        super(RandomCutForest, self).fit(records, mini_batch_size, **kwargs)


class RandomCutForestPredictor(RealTimePredictor):
    """Assigns an anomaly score to each of the datapoints provided.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain the
    same number of columns as the feature-dimension of the data used to fit the model this
    Predictor performs inference on.

    :meth:`predict()` returns a list of :class:`~sagemaker.amazon.record_pb2.Record` objects,
    one for each row in the input. Each row's score is stored in the key ``score`` of the
    ``Record.label`` field."""

    def __init__(self, endpoint, sagemaker_session=None):
        super(RandomCutForestPredictor, self).__init__(endpoint, sagemaker_session,
                                                       serializer=numpy_to_record_serializer(),
                                                       deserializer=record_deserializer())


class RandomCutForestModel(Model):
    """Reference RandomCutForest s3 model data. Calling :meth:`~sagemaker.model.Model.deploy` creates an
    Endpoint and returns a Predictor that calculates anomaly scores for datapoints."""

    def __init__(self, model_data, role, sagemaker_session=None):
        sagemaker_session = sagemaker_session or Session()
        repo = '{}:{}'.format(RandomCutForest.repo_name, RandomCutForest.repo_version)
        image = '{}/{}'.format(registry(sagemaker_session.boto_session.region_name,
                                        RandomCutForest.repo_name), repo)
        super(RandomCutForestModel, self).__init__(model_data, image, role,
                                                   predictor_cls=RandomCutForestPredictor,
                                                   sagemaker_session=sagemaker_session)
