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
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session


class PCA(AmazonAlgorithmEstimatorBase):

    repo_name = 'pca'
    repo_version = 1

    DEFAULT_MINI_BATCH_SIZE = 500

    num_components = hp(name='num_components', validate=lambda x: x > 0,
                        validation_message='Value must be an integer greater than zero', data_type=int)
    algorithm_mode = hp(name='algorithm_mode', validate=lambda x: x in ['regular', 'stable', 'randomized'],
                        validation_message='Value must be one of "regular", "stable", "randomized"', data_type=str)
    subtract_mean = hp(name='subtract_mean', validation_message='Value must be a boolean', data_type=bool)
    extra_components = hp(name='extra_components', validate=lambda x: x >= 0,
                          validation_message="Value must be an integer greater than or equal to 0", data_type=int)

    def __init__(self, role, train_instance_count, train_instance_type, num_components,
                 algorithm_mode=None, subtract_mean=None, extra_components=None, **kwargs):
        """A Principal Components Analysis (PCA) :class:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase`.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit_ndarray`
        or :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`. The former allows a PCA model
        to be fit on a 2-dimensional numpy array. The latter requires Amazon
        :class:`~sagemaker.amazon.record_pb2.Record` protobuf serialized data to be stored in S3.

        To learn more about the Amazon protobuf Record class and how to prepare bulk data in this format, please
        consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        After this Estimator is fit, model data is stored in S3. The model may be deployed to an Amazon SageMaker
        Endpoint by invoking :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as deploying an Endpoint,
        deploy returns a :class:`~sagemaker.amazon.pca.PCAPredictor` object that can be used to project
        input vectors to the learned lower-dimensional representation, using the trained PCA model hosted in the
        SageMaker Endpoint.

        PCA Estimators can be configured by setting hyperparameters. The available hyperparameters for PCA
        are documented below. For further information on the AWS PCA algorithm, please consult AWS technical
        documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html

        This Estimator uses Amazon SageMaker PCA to perform training and host deployed models. To
        learn more about Amazon SageMaker PCA, please read:
        https://docs.aws.amazon.com/sagemaker/latest/dg/how-pca-works.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            num_components(int): The number of principal components. Must be greater than zero.
            algorithm_mode (str): Mode for computing the principal components. One of 'regular', 'stable' or
                'randomized'.
            subtract_mean (bool): Whether the data should be unbiased both during train and at inference.
            extra_components (int): As the value grows larger, the solution becomes more accurate but the
                runtime and memory consumption increase linearly. If this value is unset, then a default value equal
                to the maximum of 10 and num_components will be used. Valid for randomized mode only.
            **kwargs: base class keyword argument values.
        """
        super(PCA, self).__init__(role, train_instance_count, train_instance_type, **kwargs)
        self.num_components = num_components
        self.algorithm_mode = algorithm_mode
        self.subtract_mean = subtract_mean
        self.extra_components = extra_components

    def create_model(self):
        """Return a :class:`~sagemaker.amazon.pca.PCAModel` referencing the latest
        s3 model data produced by this Estimator."""

        return PCAModel(self.model_data, self.role, sagemaker_session=self.sagemaker_session)

    def fit(self, records, mini_batch_size=None, **kwargs):
        # mini_batch_size is a required parameter
        default_mini_batch_size = min(self.DEFAULT_MINI_BATCH_SIZE,
                                      max(1, int(records.num_records / self.train_instance_count)))
        use_mini_batch_size = mini_batch_size or default_mini_batch_size
        super(PCA, self).fit(records, use_mini_batch_size, **kwargs)


class PCAPredictor(RealTimePredictor):
    """Transforms input vectors to lower-dimesional representations.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain the
    same number of columns as the feature-dimension of the data used to fit the model this
    Predictor performs inference on.

    :meth:`predict()` returns a list of :class:`~sagemaker.amazon.record_pb2.Record` objects, one
    for each row in the input ``ndarray``. The lower dimension vector result is stored in the ``projection``
    key of the ``Record.label`` field."""

    def __init__(self, endpoint, sagemaker_session=None):
        super(PCAPredictor, self).__init__(endpoint, sagemaker_session, serializer=numpy_to_record_serializer(),
                                           deserializer=record_deserializer())


class PCAModel(Model):
    """Reference PCA s3 model data. Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return
    a Predictor that transforms vectors to a lower-dimensional representation."""

    def __init__(self, model_data, role, sagemaker_session=None):
        sagemaker_session = sagemaker_session or Session()
        repo = '{}:{}'.format(PCA.repo_name, PCA.repo_version)
        image = '{}/{}'.format(registry(sagemaker_session.boto_session.region_name), repo)
        super(PCAModel, self).__init__(model_data, image, role, predictor_cls=PCAPredictor,
                                       sagemaker_session=sagemaker_session)
