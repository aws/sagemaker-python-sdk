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

from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase, registry
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import ge, le
from sagemaker.predictor import RealTimePredictor, csv_serializer, json_deserializer
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT


class IPInsights(AmazonAlgorithmEstimatorBase):
    repo_name = 'ipinsights'
    repo_version = 1
    MINI_BATCH_SIZE = 10000

    num_entity_vectors = hp('num_entity_vectors', (ge(1), le(250000000)), 'An integer in [1, 250000000]', int)
    vector_dim = hp('vector_dim', (ge(4), le(4096)), 'An integer in [4, 4096]', int)

    batch_metrics_publish_interval = hp('batch_metrics_publish_interval', (ge(1)), 'An integer greater than 0', int)
    epochs = hp('epochs', (ge(1)), 'An integer greater than 0', int)
    learning_rate = hp('learning_rate', (ge(1e-6), le(10.0)), 'A float in [1e-6, 10.0]', float)
    num_ip_encoder_layers = hp('num_ip_encoder_layers', (ge(0), le(100)), 'An integer in [0, 100]', int)
    random_negative_sampling_rate = hp('random_negative_sampling_rate', (ge(0), le(500)), 'An integer in [0, 500]', int)
    shuffled_negative_sampling_rate = hp('shuffled_negative_sampling_rate', (ge(0), le(500)), 'An integer in [0, 500]',
                                         int)
    weight_decay = hp('weight_decay', (ge(0.0), le(10.0)), 'A float in [0.0, 10.0]', float)

    def __init__(self, role, train_instance_count, train_instance_type, num_entity_vectors, vector_dim,
                 batch_metrics_publish_interval=None, epochs=None, learning_rate=None,
                 num_ip_encoder_layers=None, random_negative_sampling_rate=None,
                 shuffled_negative_sampling_rate=None, weight_decay=None, **kwargs):
        """This estimator is for IP Insights, an unsupervised algorithm that learns usage patterns of IP addresses.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`. It requires
        CSV data to be stored in S3.

        After this Estimator is fit, model data is stored in S3. The model may be deployed to an Amazon SageMaker
        Endpoint by invoking :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as deploying an Endpoint,
        deploy returns a :class:`~sagemaker.amazon.IPInsightPredictor` object that can be used
        for inference calls using the trained model hosted in the SageMaker Endpoint.

        IPInsights Estimators can be configured by setting hyperparamters.
        The available hyperparamters are documented below.

        For further information on the AWS IPInsights algorithm, please consult AWS technical documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights-hyperparameters.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.m5.xlarge'.
            num_entity_vectors (int): Required. The number of embeddings to train for entities accessing online
                resources. We recommend 2x the total number of unique entity IDs.
            vector_dim (int): Required. The size of the embedding vectors for both entity and IP addresses.
            batch_metrics_publish_interval (int): Optional. The period at which to publish metrics (batches).
            epochs (int): Optional. Maximum number of passes over the training data.
            learning_rate (float): Optional. Learning rate for the optimizer.
            num_ip_encoder_layers (int): Optional. The number of fully-connected layers to encode IP address embedding.
            random_negative_sampling_rate (int): Optional. The ratio of random negative samples to draw during training.
                Random negative samples are randomly drawn IPv4 addresses.
            shuffled_negative_sampling_rate (int): Optional. The ratio of shuffled negative samples to draw during
                training. Shuffled negative samples are IP addresses picked from within a batch.
            weight_decay (float): Optional. Weight decay coefficient. Adds L2 regularization.
            **kwargs: base class keyword argument values.
        """
        super(IPInsights, self).__init__(role, train_instance_count, train_instance_type, **kwargs)
        self.num_entity_vectors = num_entity_vectors
        self.vector_dim = vector_dim
        self.batch_metrics_publish_interval = batch_metrics_publish_interval
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_ip_encoder_layers = num_ip_encoder_layers
        self.random_negative_sampling_rate = random_negative_sampling_rate
        self.shuffled_negative_sampling_rate = shuffled_negative_sampling_rate
        self.weight_decay = weight_decay

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT):
        """Create a model for the latest s3 model produced by this estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
        Returns:
            :class:`~sagemaker.amazon.IPInsightsModel`: references the latest s3 model data produced by this estimator.
        """
        return IPInsightsModel(self.model_data, self.role, sagemaker_session=self.sagemaker_session,
                               vpc_config=self.get_vpc_config(vpc_config_override))

    def _prepare_for_training(self, records, mini_batch_size=None, job_name=None):
        if mini_batch_size is not None and (mini_batch_size < 1 or mini_batch_size > 500000):
            raise ValueError("mini_batch_size must be in [1, 500000]")
        super(IPInsights, self)._prepare_for_training(records, mini_batch_size=mini_batch_size, job_name=job_name)


class IPInsightsPredictor(RealTimePredictor):
    """Returns dot product of entity and IP address embeddings as a score for compatibility.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain
    two columns. The first column should contain the entity ID. The second column should
    contain the IPv4 address in dot notation.
    """

    def __init__(self, endpoint, sagemaker_session=None):
        super(IPInsightsPredictor, self).__init__(endpoint, sagemaker_session,
                                                  serializer=csv_serializer,
                                                  deserializer=json_deserializer)


class IPInsightsModel(Model):
    """Reference IPInsights s3 model data. Calling :meth:`~sagemaker.model.Model.deploy` creates an
    Endpoint and returns a Predictor that calculates anomaly scores for data points."""

    def __init__(self, model_data, role, sagemaker_session=None, **kwargs):
        sagemaker_session = sagemaker_session or Session()
        repo = '{}:{}'.format(IPInsights.repo_name, IPInsights.repo_version)
        image = '{}/{}'.format(registry(sagemaker_session.boto_session.region_name,
                                        IPInsights.repo_name), repo)

        super(IPInsightsModel, self).__init__(
            model_data, image, role,
            predictor_cls=IPInsightsPredictor,
            sagemaker_session=sagemaker_session,
            **kwargs)
