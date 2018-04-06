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
from sagemaker.amazon.validation import ge, le, isin
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session


class NTM(AmazonAlgorithmEstimatorBase):

    repo_name = 'ntm'
    repo_version = 1

    num_topics = hp('num_topics', (ge(2), le(1000)), 'An integer in [2, 1000]', int)
    encoder_layers = hp(name='encoder_layers', validation_message='A comma separated list of '
                                                                  'positive integers', data_type=list)
    epochs = hp('epochs', (ge(1), le(100)), 'An integer in [1, 100]', int)
    encoder_layers_activation = hp('encoder_layers_activation', isin('sigmoid', 'tanh', 'relu'),
                                   'One of "sigmoid", "tanh" or "relu"', str)
    optimizer = hp('optimizer', isin('adagrad', 'adam', 'rmsprop', 'sgd', 'adadelta'),
                   'One of "adagrad", "adam", "rmsprop", "sgd" and "adadelta"', str)
    tolerance = hp('tolerance', (ge(1e-6), le(0.1)), 'A float in [1e-6, 0.1]', float)
    num_patience_epochs = hp('num_patience_epochs', (ge(1), le(10)), 'An integer in [1, 10]', int)
    batch_norm = hp(name='batch_norm', validation_message='Value must be a boolean', data_type=bool)
    rescale_gradient = hp('rescale_gradient', (ge(1e-3), le(1.0)), 'A float in [1e-3, 1.0]', float)
    clip_gradient = hp('clip_gradient', ge(1e-3), 'A float greater equal to 1e-3', float)
    weight_decay = hp('weight_decay', (ge(0.0), le(1.0)), 'A float in [0.0, 1.0]', float)
    learning_rate = hp('learning_rate', (ge(1e-6), le(1.0)), 'A float in [1e-6, 1.0]', float)

    def __init__(self, role, train_instance_count, train_instance_type, num_topics,
                 encoder_layers=None, epochs=None, encoder_layers_activation=None, optimizer=None, tolerance=None,
                 num_patience_epochs=None, batch_norm=None, rescale_gradient=None, clip_gradient=None,
                 weight_decay=None, learning_rate=None, **kwargs):
        """Neural Topic Model (NTM) is :class:`Estimator` used for unsupervised learning.

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
        deploy returns a :class:`~sagemaker.amazon.ntm.NTMPredictor` object that can be used
        for inference calls using the trained model hosted in the SageMaker Endpoint.

        NTM Estimators can be configured by setting hyperparameters. The available hyperparameters for
        NTM are documented below.

        For further information on the AWS NTM algorithm,
        please consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/ntm.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            num_topics (int): Required. The number of topics for NTM to find within the data.
            encoder_layers (list): Optional. Represents number of layers in the encoder and the output size of
                each layer.
            epochs (int): Optional. Maximum number of passes over the training data.
            encoder_layers_activation (str): Optional. Activation function to use in the encoder layers.
            optimizer (str): Optional. Optimizer to use for training.
            tolerance (float): Optional. Maximum relative change in the loss function within the last
                num_patience_epochs number of epochs below which early stopping is triggered.
            num_patience_epochs (int): Optional. Number of successive epochs over which early stopping criterion
                is evaluated.
            batch_norm (bool): Optional. Whether to use batch normalization during training.
            rescale_gradient (float): Optional. Rescale factor for gradient.
            clip_gradient (float): Optional. Maximum magnitude for each gradient component.
            weight_decay (float): Optional. Weight decay coefficient. Adds L2 regularization.
            learning_rate (float): Optional. Learning rate for the optimizer.
            **kwargs: base class keyword argument values.
        """

        super(NTM, self).__init__(role, train_instance_count, train_instance_type, **kwargs)
        self.num_topics = num_topics
        self.encoder_layers = encoder_layers
        self.epochs = epochs
        self.encoder_layers_activation = encoder_layers_activation
        self.optimizer = optimizer
        self.tolerance = tolerance
        self.num_patience_epochs = num_patience_epochs
        self.batch_norm = batch_norm
        self.rescale_gradient = rescale_gradient
        self.clip_gradient = clip_gradient
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

    def create_model(self):
        """Return a :class:`~sagemaker.amazon.NTMModel` referencing the latest
        s3 model data produced by this Estimator."""

        return NTMModel(self.model_data, self.role, sagemaker_session=self.sagemaker_session)

    def fit(self, records, mini_batch_size=None, **kwargs):
        if mini_batch_size is not None and (mini_batch_size < 1 or mini_batch_size > 10000):
            raise ValueError("mini_batch_size must be in [1, 10000]")
        super(NTM, self).fit(records, mini_batch_size, **kwargs)


class NTMPredictor(RealTimePredictor):
    """Transforms input vectors to lower-dimesional representations.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain the
    same number of columns as the feature-dimension of the data used to fit the model this
    Predictor performs inference on.

    :meth:`predict()` returns a list of :class:`~sagemaker.amazon.record_pb2.Record` objects, one
    for each row in the input ``ndarray``. The lower dimension vector result is stored in the ``projection``
    key of the ``Record.label`` field."""

    def __init__(self, endpoint, sagemaker_session=None):
        super(NTMPredictor, self).__init__(endpoint, sagemaker_session, serializer=numpy_to_record_serializer(),
                                           deserializer=record_deserializer())


class NTMModel(Model):
    """Reference NTM s3 model data. Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return
    a Predictor that transforms vectors to a lower-dimensional representation."""

    def __init__(self, model_data, role, sagemaker_session=None):
        sagemaker_session = sagemaker_session or Session()
        repo = '{}:{}'.format(NTM.repo_name, NTM.repo_version)
        image = '{}/{}'.format(registry(sagemaker_session.boto_session.region_name, NTM.repo_name), repo)
        super(NTMModel, self).__init__(model_data, image, role, predictor_cls=NTMPredictor,
                                       sagemaker_session=sagemaker_session)
