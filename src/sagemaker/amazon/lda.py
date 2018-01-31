# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from sagemaker.amazon.validation import gt
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session


class LDA(AmazonAlgorithmEstimatorBase):

    repo_name = 'lda'
    repo_version = 1

    num_topics = hp('num_topics', gt(0), 'An integer greater than zero', int)
    alpha0 = hp('alpha0', gt(0), 'A positive float', float)
    max_restarts = hp('max_restarts', gt(0), 'An integer greater than zero', int)
    max_iterations = hp('max_iterations', gt(0), 'An integer greater than zero', int)
    tol = hp('tol', gt(0), 'A positive float', float)

    def __init__(self, role, train_instance_type, num_topics,
                 alpha0=None, max_restarts=None, max_iterations=None, tol=None, **kwargs):
        """Latent Dirichlet Allocation (LDA) is :class:`Estimator` used for unsupervised learning.

        Amazon SageMaker Latent Dirichlet Allocation is an unsupervised learning algorithm that attempts to describe
        a set of observations as a mixture of distinct categories. LDA is most commonly used to discover
        a user-specified number of topics shared by documents within a text corpus.
        Here each observation is a document, the features are the presence (or occurrence count) of each word, and
        the categories are the topics.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`. It requires Amazon
        :class:`~sagemaker.amazon.record_pb2.Record` protobuf serialized data to be stored in S3.
        There is an utility :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.record_set` that
        can be used to upload data to S3 and creates :class:`~sagemaker.amazon.amazon_estimator.RecordSet` to be passed
        to the `fit` call.

        To learn more about the Amazon protobuf Record class and how to prepare bulk data in this format, please
        consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        After this Estimator is fit, model data is stored in S3. The model may be deployed to an Amazon SageMaker
        Endpoint by invoking :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as deploying an Endpoint,
        deploy returns a :class:`~sagemaker.amazon.lda.LDAPredictor` object that can be used
        for inference calls using the trained model hosted in the SageMaker Endpoint.

        LDA Estimators can be configured by setting hyperparameters. The available hyperparameters for
        LDA are documented below.

        For further information on the AWS LDA algorithm,
        please consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/lda.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            num_topics (int): The number of topics for LDA to find within the data.
            alpha0 (float): Optional. Initial guess for the concentration parameter
            max_restarts (int): Optional. The number of restarts to perform during the Alternating Least Squares (ALS)
                spectral decomposition phase of the algorithm.
            max_iterations (int): Optional. The maximum number of iterations to perform during the
                ALS phase of the algorithm.
            tol (float): Optional. Target error tolerance for the ALS phase of the algorithm.
            **kwargs: base class keyword argument values.
        """

        # this algorithm only supports single instance training
        super(LDA, self).__init__(role, 1, train_instance_type, **kwargs)
        self.num_topics = num_topics
        self.alpha0 = alpha0
        self.max_restarts = max_restarts
        self.max_iterations = max_iterations
        self.tol = tol

    def create_model(self):
        """Return a :class:`~sagemaker.amazon.LDAModel` referencing the latest
        s3 model data produced by this Estimator."""

        return LDAModel(self.model_data, self.role, sagemaker_session=self.sagemaker_session)

    def fit(self, records, mini_batch_size, **kwargs):
        # mini_batch_size is required, prevent explicit calls with None
        if mini_batch_size is None:
            raise ValueError("mini_batch_size must be set")
        super(LDA, self).fit(records, mini_batch_size, **kwargs)


class LDAPredictor(RealTimePredictor):
    """Transforms input vectors to lower-dimesional representations.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain the
    same number of columns as the feature-dimension of the data used to fit the model this
    Predictor performs inference on.

    :meth:`predict()` returns a list of :class:`~sagemaker.amazon.record_pb2.Record` objects, one
    for each row in the input ``ndarray``. The lower dimension vector result is stored in the ``projection``
    key of the ``Record.label`` field."""

    def __init__(self, endpoint, sagemaker_session=None):
        super(LDAPredictor, self).__init__(endpoint, sagemaker_session, serializer=numpy_to_record_serializer(),
                                           deserializer=record_deserializer())


class LDAModel(Model):
    """Reference LDA s3 model data. Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return
    a Predictor that transforms vectors to a lower-dimensional representation."""

    def __init__(self, model_data, role, sagemaker_session=None):
        sagemaker_session = sagemaker_session or Session()
        repo = '{}:{}'.format(LDA.repo_name, LDA.repo_version)
        image = '{}/{}'.format(registry(sagemaker_session.boto_session.region_name, LDA.repo_name), repo)
        super(LDAModel, self).__init__(model_data, image, role, predictor_cls=LDAPredictor,
                                       sagemaker_session=sagemaker_session)
