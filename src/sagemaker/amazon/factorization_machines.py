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
from sagemaker.amazon.validation import gt, isin, isint, ge, isfloat
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session


class FactorizationMachines(AmazonAlgorithmEstimatorBase):

    repo = 'factorization-machines:1'

    num_factors = hp('num_factors', (gt(0), isint), 'An integer greater than zero')
    predictor_type = hp('predictor_type', isin('binary_classifier', 'regressor'),
                        'Value "binary_classifier" or "regressor"')
    epochs = hp('epochs', (gt(0), isint), "An integer greater than 0")
    clip_gradient = hp('clip_gradient', isfloat, "A float value")
    eps = hp('eps', isfloat, "A float value")
    rescale_grad = hp('rescale_grad', isfloat, "A float value")
    bias_lr = hp('bias_lr', (ge(0), isfloat), "A non-negative float")
    linear_lr = hp('linear_lr', (ge(0), isfloat), "A non-negative float")
    factors_lr = hp('factors_lr', (ge(0), isfloat), "A non-negative float")
    bias_wd = hp('bias_wd', (ge(0), isfloat), "A non-negative float")
    linear_wd = hp('linear_wd', (ge(0), isfloat), "A non-negative float")
    factors_wd = hp('factors_wd', (ge(0), isfloat), "A non-negative float")
    bias_init_method = hp('bias_init_method', isin('normal', 'uniform', 'constant'),
                          'Value "normal", "uniform" or "constant"')
    bias_init_scale = hp('bias_init_scale', (ge(0), isfloat), "A non-negative float")
    bias_init_sigma = hp('bias_init_sigma', (ge(0), isfloat), "A non-negative float")
    bias_init_value = hp('bias_init_value', isfloat, "A float value")
    linear_init_method = hp('linear_init_method', isin('normal', 'uniform', 'constant'),
                            'Value "normal", "uniform" or "constant"')
    linear_init_scale = hp('linear_init_scale', (ge(0), isfloat), "A non-negative float")
    linear_init_sigma = hp('linear_init_sigma', (ge(0), isfloat), "A non-negative float")
    linear_init_value = hp('linear_init_value', isfloat, "A float value")
    factors_init_method = hp('factors_init_method', isin('normal', 'uniform', 'constant'),
                             'Value "normal", "uniform" or "constant"')
    factors_init_scale = hp('factors_init_scale', (ge(0), isfloat), "A non-negative float")
    factors_init_sigma = hp('factors_init_sigma', (ge(0), isfloat), "A non-negative float")
    factors_init_value = hp('factors_init_value', isfloat, "A float value")

    def __init__(self, role, train_instance_count, train_instance_type,
                 num_factors, predictor_type,
                 epochs=None, clip_gradient=None, eps=None, rescale_grad=None,
                 bias_lr=None, linear_lr=None, factors_lr=None,
                 bias_wd=None, linear_wd=None, factors_wd=None,
                 bias_init_method=None, bias_init_scale=None, bias_init_sigma=None, bias_init_value=None,
                 linear_init_method=None, linear_init_scale=None, linear_init_sigma=None, linear_init_value=None,
                 factors_init_method=None, factors_init_scale=None, factors_init_sigma=None, factors_init_value=None,
                 **kwargs):
        """Factorization Machines (FM) :class:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase`.

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
        https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines-howitworks.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            num_factors(int): Dimensionality of factorization.
            predictor_type (str): Type of predictor 'binary_classifier' or 'regressor'.

            subtract_mean (bool): Whether the data should be unbiased both during train and at inference.
            extra_components (int): As the value grows larger, the solution becomes more accurate but the
                runtime and memory consumption increase linearly. If this value is unset, then a default value equal
                to the maximum of 10 and num_components will be used. Valid for randomized mode only.
            **kwargs: base class keyword argument values.
        """
        super(FactorizationMachines, self).__init__(role, train_instance_count, train_instance_type, **kwargs)

        self.num_factors = num_factors
        self.predictor_type = predictor_type
        self.epochs = epochs
        self.clip_gradient = clip_gradient
        self.eps = eps
        self.rescale_grad = rescale_grad
        self.bias_lr = bias_lr
        self.linear_lr = linear_lr
        self.factors_lr = factors_lr
        self.bias_wd = bias_wd
        self.linear_wd = linear_wd
        self.factors_wd = factors_wd
        self.bias_init_method = bias_init_method
        self.bias_init_scale = bias_init_scale
        self.bias_init_sigma = bias_init_sigma
        self.bias_init_value = bias_init_value
        self.linear_init_method = linear_init_method
        self.linear_init_scale = linear_init_scale
        self.linear_init_sigma = linear_init_sigma
        self.linear_init_value = linear_init_value
        self.factors_init_method = factors_init_method
        self.factors_init_scale = factors_init_scale
        self.factors_init_sigma = factors_init_sigma
        self.factors_init_value = factors_init_value

    def create_model(self):
        """Return a :class:`~sagemaker.amazon.fm.FMModel` referencing the latest
        s3 model data produced by this Estimator."""

        return FMModel(self.model_data, self.role, sagemaker_session=self.sagemaker_session)


class FMPredictor(RealTimePredictor):
    """Transforms input vectors to lower-dimesional representations.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain the
    same number of columns as the feature-dimension of the data used to fit the model this
    Predictor performs inference on.

    :meth:`predict()` returns a list of :class:`~sagemaker.amazon.record_pb2.Record` objects, one
    for each row in the input ``ndarray``. The lower dimension vector result is stored in the ``projection``
    key of the ``Record.label`` field."""

    def __init__(self, endpoint, sagemaker_session=None):
        super(FMPredictor, self).__init__(endpoint, sagemaker_session, serializer=numpy_to_record_serializer(),
                                          deserializer=record_deserializer())


class FMModel(Model):
    """Reference FM s3 model data. Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return
    a Predictor that transforms vectors to a lower-dimensional representation."""

    def __init__(self, model_data, role, sagemaker_session=None):
        sagemaker_session = sagemaker_session or Session()
        image = registry(sagemaker_session.boto_session.region_name) + "/" + FactorizationMachines.repo
        super(FMModel, self).__init__(model_data, image, role, predictor_cls=FMPredictor,
                                      sagemaker_session=sagemaker_session)
