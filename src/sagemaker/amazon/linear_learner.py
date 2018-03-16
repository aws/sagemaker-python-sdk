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
from sagemaker.amazon.validation import isin, gt, lt
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session


class LinearLearner(AmazonAlgorithmEstimatorBase):

    repo_name = 'linear-learner'
    repo_version = 1

    DEFAULT_MINI_BATCH_SIZE = 1000

    binary_classifier_model_selection_criteria = hp('binary_classifier_model_selection_criteria',
                                                    isin('accuracy', 'f1', 'precision_at_target_recall',
                                                         'recall_at_target_precision', 'cross_entropy_loss'),
                                                    data_type=str)
    target_recall = hp('target_recall', (gt(0), lt(1)), "A float in (0,1)", float)
    target_precision = hp('target_precision', (gt(0), lt(1)), "A float in (0,1)", float)
    positive_example_weight_mult = hp('positive_example_weight_mult', gt(0), "A float greater than 0", float)
    epochs = hp('epochs', gt(0), "An integer greater-than 0", int)
    predictor_type = hp('predictor_type', isin('binary_classifier', 'regressor'),
                        'One of "binary_classifier" or "regressor"', str)
    use_bias = hp('use_bias', (), "Either True or False", bool)
    num_models = hp('num_models', gt(0), "An integer greater-than 0", int)
    num_calibration_samples = hp('num_calibration_samples', gt(0), "An integer greater-than 0", int)
    init_method = hp('init_method', isin('uniform', 'normal'), 'One of "uniform" or "normal"', str)
    init_scale = hp('init_scale', (gt(-1), lt(1)), 'A float in (-1, 1)', float)
    init_sigma = hp('init_sigma', (gt(0), lt(1)), 'A float in (0, 1)', float)
    init_bias = hp('init_bias', (), 'A number', float)
    optimizer = hp('optimizer', isin('sgd', 'adam', 'auto'), 'One of "sgd", "adam" or "auto', str)
    loss = hp('loss', isin('logistic', 'squared_loss', 'absolute_loss', 'auto'),
              '"logistic", "squared_loss", "absolute_loss" or"auto"', str)
    wd = hp('wd', (gt(0), lt(1)), 'A float in (0,1)', float)
    l1 = hp('l1', (gt(0), lt(1)), 'A float in (0,1)', float)
    momentum = hp('momentum', (gt(0), lt(1)), 'A float in (0,1)', float)
    learning_rate = hp('learning_rate', (gt(0), lt(1)), 'A float in (0,1)', float)
    beta_1 = hp('beta_1', (gt(0), lt(1)), 'A float in (0,1)', float)
    beta_2 = hp('beta_2', (gt(0), lt(1)), 'A float in (0,1)', float)
    bias_lr_mult = hp('bias_lr_mult', gt(0), 'A float greater-than 0', float)
    bias_wd_mult = hp('bias_wd_mult', gt(0), 'A float greater-than 0', float)
    use_lr_scheduler = hp('use_lr_scheduler', (), 'A boolean', bool)
    lr_scheduler_step = hp('lr_scheduler_step', gt(0), 'An integer greater-than 0', int)
    lr_scheduler_factor = hp('lr_scheduler_factor', (gt(0), lt(1)), 'A float in (0,1)', float)
    lr_scheduler_minimum_lr = hp('lr_scheduler_minimum_lr', gt(0), 'A float greater-than 0', float)
    normalize_data = hp('normalize_data', (), 'A boolean', bool)
    normalize_label = hp('normalize_label', (), 'A boolean', bool)
    unbias_data = hp('unbias_data', (), 'A boolean', bool)
    unbias_label = hp('unbias_label', (), 'A boolean', bool)
    num_point_for_scaler = hp('num_point_for_scaler', gt(0), 'An integer greater-than 0', int)

    def __init__(self, role, train_instance_count, train_instance_type, predictor_type='binary_classifier',
                 binary_classifier_model_selection_criteria=None, target_recall=None, target_precision=None,
                 positive_example_weight_mult=None, epochs=None, use_bias=None, num_models=None,
                 num_calibration_samples=None, init_method=None, init_scale=None, init_sigma=None, init_bias=None,
                 optimizer=None, loss=None, wd=None, l1=None, momentum=None, learning_rate=None, beta_1=None,
                 beta_2=None, bias_lr_mult=None, bias_wd_mult=None, use_lr_scheduler=None, lr_scheduler_step=None,
                 lr_scheduler_factor=None, lr_scheduler_minimum_lr=None, normalize_data=None,
                 normalize_label=None, unbias_data=None, unbias_label=None, num_point_for_scaler=None, **kwargs):
        """An :class:`Estimator` for binary classification and regression.

        Amazon SageMaker Linear Learner provides a solution for both classification and regression problems, allowing
        for exploring different training objectives simultaneously and choosing the best solution from a validation set.
        It allows the user to explore a large number of models and choose the best, which optimizes either continuous
        objectives such as mean square error, cross entropy loss, absolute error, etc., or discrete objectives suited
        for classification such as F1 measure, precision@recall, accuracy. The implementation provides a significant
        speedup over naive hyperparameter optimization techniques and an added convenience, when compared with
        solutions providing a solution only to continuous objectives.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit_ndarray`
        or :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`. The former allows a
        LinearLearner model to be fit on a 2-dimensional numpy array. The latter requires Amazon
        :class:`~sagemaker.amazon.record_pb2.Record` protobuf serialized data to be stored in S3.

        To learn more about the Amazon protobuf Record class and how to prepare bulk data in this format, please
        consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        After this Estimator is fit, model data is stored in S3. The model may be deployed to an Amazon SageMaker
        Endpoint by invoking :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as deploying an Endpoint,
        ``deploy`` returns a :class:`~sagemaker.amazon.linear_learner.LinearLearnerPredictor` object that can be used
        to make class or regression predictions, using the trained model.

        LinearLearner Estimators can be configured by setting hyperparameters. The available hyperparameters for
        LinearLearner are documented below. For further information on the AWS LinearLearner algorithm, please consult
        AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            predictor_type (str): The type of predictor to learn. Either "binary_classifier" or "regressor".
            binary_classifier_model_selection_criteria (str): One of 'accuracy', 'f1', 'precision_at_target_recall',
            'recall_at_target_precision', 'cross_entropy_loss'
            target_recall (float): Target recall. Only applicable if binary_classifier_model_selection_criteria is
                precision_at_target_recall.
            target_precision (float): Target precision. Only applicable if binary_classifier_model_selection_criteria
                is recall_at_target_precision.
            positive_example_weight_mult (float): The importance weight of positive examples is multiplied by this
                constant. Useful for skewed datasets. Only applies for classification tasks.
            epochs (int): The maximum number of passes to make over the training data.
            use_bias (bool): Whether to include a bias field
            num_models (int): Number of models to train in parallel. If not set, the number of parallel models to
                train will be decided by the algorithm itself. One model will be trained according to the given training
            parameter (regularization, optimizer, loss) and the rest by close by parameters.
            num_calibration_samples (int): Number of observations to use from validation dataset for doing model
            calibration (finding the best threshold).
            init_method (str): Function to use to set the initial model weights. One of "uniform" or "normal"
            init_scale (float): For "uniform" init, the range of values.
            init_sigma (float): For "normal" init, the standard-deviation.
            init_bias (float):  Initial weight for bias term
            optimizer (str): One of 'sgd', 'adam' or 'auto'
            loss (str): One of  'logistic', 'squared_loss', 'absolute_loss' or 'auto'
            wd (float): L2 regularization parameter i.e. the weight decay parameter. Use 0 for no L2 regularization.
            l1 (float): L1 regularization parameter. Use 0 for no L1 regularization.
            momentum (float): Momentum parameter of sgd optimizer.
            learning_rate (float): The SGD learning rate
            beta_1 (float): Exponential decay rate for first moment estimates. Only applies for adam optimizer.
            beta_2 (float): Exponential decay rate for second moment estimates. Only applies for adam optimizer.
            bias_lr_mult (float): Allows different learning rate for the bias term. The actual learning rate for the
            bias is learning rate times bias_lr_mult.
            bias_wd_mult (float): Allows different regularization for the bias term. The actual L2 regularization weight
            for the bias is wd times bias_wd_mult. By default there is no regularization on the bias term.
            use_lr_scheduler (bool): If true, we use a scheduler for the learning rate.
            lr_scheduler_step (int): The number of steps between decreases of the learning rate. Only applies to
                learning rate scheduler.
            lr_scheduler_factor (float): Every lr_scheduler_step the learning rate will decrease by this quantity.
                Only applies for learning rate scheduler.
            lr_scheduler_minimum_lr (float): The learning rate will never decrease to a value lower than this.
            lr_scheduler_minimum_lr (float): Only applies for learning rate scheduler.
            normalize_data (bool): Normalizes the features before training to have standard deviation of 1.0.
            normalize_label (bool): Normalizes the regression label to have a standard deviation of 1.0.
                If set for classification, it will be ignored.
            unbias_data (bool): If true, features are modified to have mean 0.0.
            ubias_label (bool): If true, labels are modified to have mean 0.0.
            num_point_for_scaler (int): The number of data points to use for calculating the normalizing  and
                unbiasing terms.
            **kwargs: base class keyword argument values.
        """
        super(LinearLearner, self).__init__(role, train_instance_count, train_instance_type, **kwargs)
        self.predictor_type = predictor_type
        self.binary_classifier_model_selection_criteria = binary_classifier_model_selection_criteria
        self.target_recall = target_recall
        self.target_precision = target_precision
        self.positive_example_weight_mult = positive_example_weight_mult
        self.epochs = epochs
        self.use_bias = use_bias
        self.num_models = num_models
        self.num_calibration_samples = num_calibration_samples
        self.init_method = init_method
        self.init_scale = init_scale
        self.init_sigma = init_sigma
        self.init_bias = init_bias
        self.optimizer = optimizer
        self.loss = loss
        self.wd = wd
        self.l1 = l1
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.bias_lr_mult = bias_lr_mult
        self.bias_wd_mult = bias_wd_mult
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_minimum_lr = lr_scheduler_minimum_lr
        self.normalize_data = normalize_data
        self.normalize_label = normalize_label
        self.unbias_data = unbias_data
        self.unbias_label = unbias_label
        self.num_point_for_scaler = num_point_for_scaler

    def create_model(self):
        """Return a :class:`~sagemaker.amazon.kmeans.LinearLearnerModel` referencing the latest
        s3 model data produced by this Estimator."""

        return LinearLearnerModel(self.model_data, self.role, self.sagemaker_session)

    def fit(self, records, mini_batch_size=None, **kwargs):
        # mini_batch_size can't be greater than number of records or training job fails
        default_mini_batch_size = min(self.DEFAULT_MINI_BATCH_SIZE,
                                      max(1, int(records.num_records / self.train_instance_count)))
        use_mini_batch_size = mini_batch_size or default_mini_batch_size
        super(LinearLearner, self).fit(records, use_mini_batch_size, **kwargs)


class LinearLearnerPredictor(RealTimePredictor):
    """Performs binary-classification or regression prediction from input vectors.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain the
    same number of columns as the feature-dimension of the data used to fit the model this
    Predictor performs inference on.

    :func:`predict` returns a list of :class:`~sagemaker.amazon.record_pb2.Record` objects, one
    for each row in the input ``ndarray``. The prediction is stored in the ``"predicted_label"``
    key of the ``Record.label`` field."""

    def __init__(self, endpoint, sagemaker_session=None):
        super(LinearLearnerPredictor, self).__init__(endpoint, sagemaker_session,
                                                     serializer=numpy_to_record_serializer(),
                                                     deserializer=record_deserializer())


class LinearLearnerModel(Model):
    """Reference LinearLearner s3 model data. Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint
    and returns a :class:`LinearLearnerPredictor`"""

    def __init__(self, model_data, role, sagemaker_session=None):
        sagemaker_session = sagemaker_session or Session()
        repo = '{}:{}'.format(LinearLearner.repo_name, LinearLearner.repo_version)
        image = '{}/{}'.format(registry(sagemaker_session.boto_session.region_name), repo)
        super(LinearLearnerModel, self).__init__(model_data, image, role,
                                                 predictor_cls=LinearLearnerPredictor,
                                                 sagemaker_session=sagemaker_session)
