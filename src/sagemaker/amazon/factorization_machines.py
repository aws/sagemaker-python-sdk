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
from sagemaker.amazon.validation import gt, isin, ge
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session


class FactorizationMachines(AmazonAlgorithmEstimatorBase):

    repo_name = 'factorization-machines'
    repo_version = 1

    num_factors = hp('num_factors', gt(0), 'An integer greater than zero', int)
    predictor_type = hp('predictor_type', isin('binary_classifier', 'regressor'),
                        'Value "binary_classifier" or "regressor"', str)
    epochs = hp('epochs', gt(0), "An integer greater than 0", int)
    clip_gradient = hp('clip_gradient', (), "A float value", float)
    eps = hp('eps', (), "A float value", float)
    rescale_grad = hp('rescale_grad', (), "A float value", float)
    bias_lr = hp('bias_lr', ge(0), "A non-negative float", float)
    linear_lr = hp('linear_lr', ge(0), "A non-negative float", float)
    factors_lr = hp('factors_lr', ge(0), "A non-negative float", float)
    bias_wd = hp('bias_wd', ge(0), "A non-negative float", float)
    linear_wd = hp('linear_wd', ge(0), "A non-negative float", float)
    factors_wd = hp('factors_wd', ge(0), "A non-negative float", float)
    bias_init_method = hp('bias_init_method', isin('normal', 'uniform', 'constant'),
                          'Value "normal", "uniform" or "constant"', str)
    bias_init_scale = hp('bias_init_scale', ge(0), "A non-negative float", float)
    bias_init_sigma = hp('bias_init_sigma', ge(0), "A non-negative float", float)
    bias_init_value = hp('bias_init_value', (), "A float value", float)
    linear_init_method = hp('linear_init_method', isin('normal', 'uniform', 'constant'),
                            'Value "normal", "uniform" or "constant"', str)
    linear_init_scale = hp('linear_init_scale', ge(0), "A non-negative float", float)
    linear_init_sigma = hp('linear_init_sigma', ge(0), "A non-negative float", float)
    linear_init_value = hp('linear_init_value', (), "A float value", float)
    factors_init_method = hp('factors_init_method', isin('normal', 'uniform', 'constant'),
                             'Value "normal", "uniform" or "constant"', str)
    factors_init_scale = hp('factors_init_scale', ge(0), "A non-negative float", float)
    factors_init_sigma = hp('factors_init_sigma', ge(0), "A non-negative float", float)
    factors_init_value = hp('factors_init_value', (), "A float value", float)

    def __init__(self, role, train_instance_count, train_instance_type,
                 num_factors, predictor_type,
                 epochs=None, clip_gradient=None, eps=None, rescale_grad=None,
                 bias_lr=None, linear_lr=None, factors_lr=None,
                 bias_wd=None, linear_wd=None, factors_wd=None,
                 bias_init_method=None, bias_init_scale=None, bias_init_sigma=None, bias_init_value=None,
                 linear_init_method=None, linear_init_scale=None, linear_init_sigma=None, linear_init_value=None,
                 factors_init_method=None, factors_init_scale=None, factors_init_sigma=None, factors_init_value=None,
                 **kwargs):
        """Factorization Machines is :class:`Estimator` for general-purpose supervised learning.

        Amazon SageMaker Factorization Machines is a general-purpose supervised learning algorithm that you can use
        for both classification and regression tasks. It is an extension of a linear model that is designed
        to parsimoniously capture interactions between features within high dimensional sparse datasets.

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
        deploy returns a :class:`~sagemaker.amazon.pca.FactorizationMachinesPredictor` object that can be used
        for inference calls using the trained model hosted in the SageMaker Endpoint.

        FactorizationMachines Estimators can be configured by setting hyperparameters. The available hyperparameters for
        FactorizationMachines are documented below.

        For further information on the AWS FactorizationMachines algorithm,
        please consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            num_factors (int): Dimensionality of factorization.
            predictor_type (str): Type of predictor 'binary_classifier' or 'regressor'.
            epochs (int): Number of training epochs to run.
            clip_gradient (float): Optimizer parameter. Clip the gradient by projecting onto
                the box [-clip_gradient, +clip_gradient]
            eps (float): Optimizer parameter. Small value to avoid division by 0.
            rescale_grad (float): Optimizer parameter. If set, multiplies the gradient with rescale_grad
                before updating. Often choose to be 1.0/batch_size.
            bias_lr (float): Non-negative learning rate for the bias term.
            linear_lr (float): Non-negative learning rate for linear terms.
            factors_lr (float): Noon-negative learning rate for factorization terms.
            bias_wd (float): Non-negative weight decay for the bias term.
            linear_wd (float): Non-negative weight decay for linear terms.
            factors_wd (float): Non-negative weight decay for factorization terms.
            bias_init_method (string): Initialization method for the bias term: 'normal', 'uniform' or 'constant'.
            bias_init_scale (float): Non-negative range for initialization of the bias term that takes
                effect when bias_init_method parameter is 'uniform'
            bias_init_sigma (float): Non-negative standard deviation for initialization of the bias term that takes
                effect when bias_init_method parameter is 'normal'.
            bias_init_value (float): Initial value of the bias term  that takes effect
                when bias_init_method parameter is 'constant'.
            linear_init_method (string): Initialization method for linear term: 'normal', 'uniform' or 'constant'.
            linear_init_scale (float): Non-negative range for initialization of linear terms that takes
                effect when linear_init_method parameter is 'uniform'.
            linear_init_sigma (float): Non-negative standard deviation for initialization of linear terms that takes
                effect when linear_init_method parameter is 'normal'.
            linear_init_value (float): Initial value of linear terms that takes effect
                when linear_init_method parameter is 'constant'.
            factors_init_method (string): Initialization method for factorization term: 'normal',
                'uniform' or 'constant'.
            factors_init_scale (float): Non-negative range for initialization of factorization terms that takes
                effect when factors_init_method parameter is 'uniform'.
            factors_init_sigma (float): Non-negative standard deviation for initialization of factorization terms that
                takes effect when factors_init_method parameter is 'normal'.
            factors_init_value (float): Initial value of factorization terms that takes
                effect when factors_init_method parameter is 'constant'.
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
        """Return a :class:`~sagemaker.amazon.FactorizationMachinesModel` referencing the latest
        s3 model data produced by this Estimator."""

        return FactorizationMachinesModel(self.model_data, self.role, sagemaker_session=self.sagemaker_session)


class FactorizationMachinesPredictor(RealTimePredictor):
    """Performs binary-classification or regression prediction from input vectors.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain the
    same number of columns as the feature-dimension of the data used to fit the model this
    Predictor performs inference on.

    :meth:`predict()` returns a list of :class:`~sagemaker.amazon.record_pb2.Record` objects, one
    for each row in the input ``ndarray``. The prediction is stored in the ``"score"``
    key of the ``Record.label`` field.
    Please refer to the formats details described: https://docs.aws.amazon.com/sagemaker/latest/dg/fm-in-formats.html
    """

    def __init__(self, endpoint, sagemaker_session=None):
        super(FactorizationMachinesPredictor, self).__init__(endpoint,
                                                             sagemaker_session,
                                                             serializer=numpy_to_record_serializer(),
                                                             deserializer=record_deserializer())


class FactorizationMachinesModel(Model):
    """Reference S3 model data created by FactorizationMachines estimator. Calling :meth:`~sagemaker.model.Model.deploy`
    creates an Endpoint and returns :class:`FactorizationMachinesPredictor`."""

    def __init__(self, model_data, role, sagemaker_session=None):
        sagemaker_session = sagemaker_session or Session()
        repo = '{}:{}'.format(FactorizationMachines.repo_name, FactorizationMachines.repo_version)
        image = '{}/{}'.format(registry(sagemaker_session.boto_session.region_name), repo)
        super(FactorizationMachinesModel, self).__init__(model_data,
                                                         image,
                                                         role,
                                                         predictor_cls=FactorizationMachinesPredictor,
                                                         sagemaker_session=sagemaker_session)
