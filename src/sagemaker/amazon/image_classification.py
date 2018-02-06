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
from sagemaker.amazon.amazon_estimator import AmazonS3AlgorithmEstimatorBase, registry
from sagemaker.amazon.common import file_to_image_serializer, response_deserializer
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import gt, isin, isint, ge, isstr, le
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session


class ImageClassification(AmazonS3AlgorithmEstimatorBase):

    repo='image-classification:latest'

    num_classes = hp('num_classes', (gt(1), isint), 'num_classes should be an integer greater-than 1')
    num_training_samples = hp('num_training_samples', (gt(1), isint),
                              'num_training_samples should be an integer greater-than 1')
    use_pretrained_model = hp('use_pretrained_model', (isin(0, 1), isint),
                              'use_pretrained_model should be in the set, [0,1]')
    checkpoint_frequency = hp('checkpoint_frequency', (ge(1), isint),
                              'checkpoint_frequency should be an integer greater-than 1')
    num_layers = hp('num_layers', (isin(18, 34, 50, 101, 152, 200, 20, 32, 44, 56, 110), isint),
                    'num_layers should be in the set [18, 34, 50, 101, 152, 200, 20, 32, 44, 56, 110]' )
    resize = hp('resize', (gt(1), isint), 'resize should be an integer greater-than 1')
    epochs = hp('epochs', (ge(1), isint), 'epochs should be an integer greater-than 1')
    learning_rate = hp('learning_rate', (gt(0)), 'learning_rate shoudl be a floating point greater than 0' )
    lr_scheduler_factor = hp('lr_scheduler_factor', (gt(0)),
                            'lr_schedule_factor should be a floating point greater than 0')
    lr_scheduler_step = hp('lr_scheduler_step',(isstr), 'lr_scheduler_step should be a string input.')
    optimizer = hp('optimizer', (isin('sgd', 'adam', 'rmsprop', 'nag')),
                   'Should be one optimizer among the list sgd, adam, rmsprop, or nag.')
    momentum = hp('momentum', (ge(0), le(1)), 'momentum is expected in the range 0, 1')
    weight_decay = hp('weight_decay', (ge(0), le(1)), 'weight_decay in range 0 , 1 ')
    beta_1 = hp('beta_1',  (ge(0), le(1)), 'beta_1 should be in range 0, 1')
    beta_2 = hp('beta_2',  (ge(0), le(1)), 'beta_2 should be in the range 0, 1')
    eps = hp('eps',  (gt(0), le(1)), 'eps should be in the range 0, 1')
    gamma = hp('gamma',  (ge(0), le(1)), 'gamma should be in the range 0, 1')
    mini_batch_size = hp('mini_batch_size',  (gt(0)), 'mini_batch_size should be an integer greater than 0')
    image_shape = hp('image_shape',  (isstr), 'image_shape is expected to be a string')
    augmentation_type = hp('beta_1',  (isin ('crop', 'crop_color', 'crop_color_transform')),
                           'beta_1 must be from one option offered')
    top_k = hp('top_k', (ge(1), isint), 'top_k should be greater than or equal to 1')
    kv_store=hp ('kv_store',  (isin ('dist_sync', 'dist_async' )), 'Can be dist_sync or dist_async')

    def __init__(self, role, train_instance_count, train_instance_type, num_classes, num_training_samples, resize=None,
                 lr_scheduler_step=None, use_pretrained_model=0, checkpoint_frequency=1 , num_layers=18,
                 epochs=30, learning_rate=0.1,
                 lr_schedule_factor=0.1, optimizer='sgd', momentum=0., weight_decay=0.0001, beta_1=0.9,
                 beta_2=0.999, eps=1e-8, gamma=0.9 , mini_batch_size=32 , image_shape='3,224,224', 
                 augmentation_type=None, top_k=None, kv_store=None, **kwargs):
        """
        An Image classification algorithm :class:`~sagemaker.amazon.AmazonAlgorithmEstimatorBase`. Learns a classifier model that 

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonS3AlgorithmEstimatorBase.fit`

        After this Estimator is fit, model data is stored in S3. The model may be deployed to an Amazon SageMaker
        Endpoint by invoking :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as deploying an Endpoint,
        ``deploy`` returns a :class:`~sagemaker.amazon.kmeans.ImageClassificationPredictor` object that can be used to label
        assignment, using the trained model hosted in the SageMaker Endpoint.

        ImageClassification Estimators can be configured by setting hyperparameters. The available hyperparameters for
        ImageClassification are documented below. For further information on the AWS ImageClassification algorithm, please consult AWS technical
        documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/IC-Hyperparameter.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
                For more information, see <link>???.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            num_classes (int): Number of output classes. This parameter defines the dimensions of the network output
                         and is typically set to the number of classes in the dataset.
            num_training_samples (int): Number of training examples in the input dataset. If there is a 
                                mismatch between this value and the number of samples in the training 
                                set, then the behavior of the lr_scheduler_step parameter is undefined 
                                and distributed training accuracy might be affected.
            use_pretrained_model (int): Flag to indicate whether to use pre-trained model for training. 
                                If set to `1`, then the pretrained model with the corresponding number 
                                of layers is loaded and used for training. Only the top FC layer are 
                                reinitialized with random weights. Otherwise, the network is trained from scratch.
                                Default value: 0
            checkpoint_frequency (int): Period to store model parameters (in number of epochs). Default value: 1
            num_layers (int): Number of layers for the network. For data with large image size (for example, 224x224 -
                              like ImageNet), we suggest selecting the number of layers from the set [18, 34, 50, 101,
                              152, 200]. For data with small image size (for example, 28x28 - like CFAR), we suggest
                              selecting the number of layers from the set [20, 32, 44, 56, 110]. The number of layers
                              in each set is based on the ResNet paper. For transfer learning, the number of layers
                              defines the architecture of base network and hence can only be selected from the set
                              [18, 34, 50, 101, 152, 200]. Default value: 152
            resize (int): Resize the image before using it for training. The images are resized so that the shortest
                          side is of this parameter. If the parameter is not set, then the training data is used as such
                          without resizing.
                          Note: This option is available only for inputs specified as application/x-image content-type in
                          training and validation channels.
            epochs (int): Number of training epochs. Default value: 30
            learning_rate (float): Initial learning rate. Float. Range in [0, 1]. Default value: 0.1
            lr_scheduler_factor (flaot): The ratio to reduce learning rate used in conjunction with the `lr_scheduler_step` parameter, 
                                defined as `lr_new=lr_old * lr_scheduler_factor`. Valid values: Float. Range in [0, 1]. Default value: 0.1
            lr_scheduler_step (str): The epochs at which to reduce the learning rate. As explained in the ``lr_scheduler_factor`` parameter, the 
                                learning rate is reduced by ``lr_scheduler_factor`` at these epochs. For example, if the value is set 
                                to "10, 20", then the learning rate is reduced by ``lr_scheduler_factor`` after 10th epoch and again by 
                                ``lr_scheduler_factor`` after 20th epoch. The epochs are delimited by ",".
            optimizer (str): The optimizer types. For more details of the parameters for the optimizers, please refer to MXNet's API. 
                             Valid values: One of sgd, adam, rmsprop, or nag. Default value: `sgd`.
            momentum (float): The momentum for sgd and nag, ignored for other optimizers. Valid values: Float. Range in [0, 1]. Default value: 0
            weight_decay (float): The coefficient weight decay for sgd and nag, ignored for other optimizers. Range in [0, 1]. Default value: 0.0001
            beta_1 (float): The beta1 for adam, in other words, exponential decay rate for the first moment estimates. Range in [0, 1]. Default value: 0.9
            beta_2 (float): The beta2 for adam, in other words, exponential decay rate for the second moment estimates. Range in [0, 1]. Default value: 0.999
            eps	(float): The epsilon for adam and rmsprop. It is usually set to a small value to avoid division by 0. Range in [0, 1]. Default value: 1e-8 
            gamma (float): The gamma for rmsprop. A decay factor of moving average of the squared gradient. Range in [0, 1]. Default value: 0.9
            mini_batch_size	(int): The batch size for training. In a single-machine multi-GPU setting, each GPU handles mini_batch_size/num_gpu 
                                training samples. For the multi-machine training in dist_sync mode, the actual batch size is mini_batch_size*number 
                                of machines. See MXNet docs for more details. Default value: 32
            image_shape	(str): The input image dimensions, which is the same size as the input layer of the network. 
                                The format is defined as 'num_channels, height, width'. The image dimension can take on any value as the 
                                network can handle varied dimensions of the input. However, there may be memory constraints if a larger image 
                                dimension is used. Typical image dimensions for image classification are '3, 224, 224'. This is similar to the ImageNet dataset.
                                Default value: ‘3, 224, 224’
            augmentation_type: (str): Data augmentation type. The input images can be augmented in multiple ways as specified below.
                                'crop' - Randomly crop the image and flip the image horizontally
                                'crop_color' - In addition to ‘crop’, three random values in the range [-36, 36], [-50, 50], and [-50, 50] 
                                            are added to the corresponding Hue-Saturation-Lightness channels respectively
                                'crop_color_transform': In addition to crop_color, random transformations, including rotation, 
                                            shear, and aspect ratio variations are applied to the image. The maximum angle of rotation 
                                            is 10 degrees, the maximum shear ratio is 0.1, and the maximum aspect changing ratio is 0.25.
            top_k (int): Report the top-k accuracy during training. This parameter has to be greater than 1, 
                            since the top-1 training accuracy is the same as the regular training accuracy that has already been reported.
            kv_store (str): Weight update synchronization mode during distributed training. The weight updates can be updated either synchronously 
                                or asynchronously across machines. Synchronous updates typically provide better accuracy than asynchronous 
                                updates but can be slower. See distributed training in MXNet for more details. This parameter is not applicable 
                                to single machine training.
                                'dist_sync' -  The gradients are synchronized after every batch with all the workers. With dist_sync,
                                         batch-size now means the batch size used on each machine. So if there are n machines and we use 
                                         batch size b, then dist_sync behaves like local with batch size n*b
                                'dist_async'- Performs asynchronous updates. The weights are updated whenever gradients are received from any
                                         machine and the weight updates are atomic. However, the order is not guaranteed.                    
            **kwargs: base class keyword argument values.
        """
        super(ImageClassification, self).__init__(role, train_instance_count, train_instance_type,
                                                  algorithm='image_classification', **kwargs)
        self.num_classes = num_classes
        self.num_training_samples = num_training_samples
        self.resize = resize
        self.lr_scheduler_step = lr_scheduler_step
        self.use_pretrained_model = use_pretrained_model
        self.checkpoint_frequency = checkpoint_frequency
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_schedule_factor = lr_schedule_factor
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.gamma = gamma
        self.mini_batch_size = mini_batch_size
        self.image_shape = image_shape
        self.augmentation_type = augmentation_type
        self.top_k = top_k
        self.kv_store = kv_store

    def create_model(self):
        """Return a :class:`~sagemaker.amazon.image_classification.ImageClassification` referencing the latest
        s3 model data produced by this Estimator."""
        return ImageClassificationModel(self.model_data, self.role, self.sagemaker_session)

    def hyperparameters(self):
        """Return the SageMaker hyperparameters for training this ImageClassification Estimator"""
        hp = dict()
        hp.update(super(ImageClassification, self).hyperparameters())
        return hp


class ImageClassificationPredictor(RealTimePredictor):
    """Assigns input vectors to their closest cluster in a ImageClassification model.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a `x-image` as input.

    ``predict()`` returns """

    def __init__(self, endpoint, sagemaker_session=None):
        super(ImageClassificationPredictor, self).__init__(endpoint, sagemaker_session,
                                                           serializer=file_to_image_serializer(),
                                                           deserializer=response_deserializer(),
                                                           content_type='application/x-image')


class ImageClassificationModel(Model):
    """Reference KMeans s3 model data. Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return
    a Predictor to performs classification assignment."""

    def __init__(self, model_data, role, sagemaker_session=None):
        sagemaker_session = sagemaker_session or Session()
        image = registry(sagemaker_session.boto_session.region_name, algorithm='image_classification') + \
                "/" + ImageClassification.repo
        super(ImageClassificationModel, self).__init__(model_data, image, role,
                                                       predictor_cls=ImageClassificationPredictor,
                                                       sagemaker_session=sagemaker_session)
