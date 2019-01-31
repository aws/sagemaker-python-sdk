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
from sagemaker.amazon.common import numpy_to_record_serializer, record_deserializer
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import ge, isin
from sagemaker.predictor import RealTimePredictor
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT


class KNN(AmazonAlgorithmEstimatorBase):
    repo_name = 'knn'
    repo_version = 1

    k = hp('k', (ge(1)), 'An integer greater than 0', int)
    sample_size = hp('sample_size', (ge(1)), 'An integer greater than 0', int)
    predictor_type = hp('predictor_type', isin('classifier', 'regressor'),
                        'One of "classifier" or "regressor"', str)
    dimension_reduction_target = hp('dimension_reduction_target', (ge(1)),
                                    'An integer greater than 0 and less than feature_dim', int)
    dimension_reduction_type = hp('dimension_reduction_type', isin('sign', 'fjlt'), 'One of "sign" or "fjlt"', str)
    index_metric = hp('index_metric', isin('COSINE', 'INNER_PRODUCT', 'L2'),
                      'One of "COSINE", "INNER_PRODUCT", "L2"', str)
    index_type = hp('index_type', isin('faiss.Flat', 'faiss.IVFFlat', 'faiss.IVFPQ'),
                    'One of "faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ"', str)
    faiss_index_ivf_nlists = hp('faiss_index_ivf_nlists', (), '"auto" or an integer greater than 0', str)
    faiss_index_pq_m = hp('faiss_index_pq_m', (ge(1)), 'An integer greater than 0', int)

    def __init__(self, role, train_instance_count, train_instance_type, k, sample_size, predictor_type,
                 dimension_reduction_type=None, dimension_reduction_target=None, index_type=None,
                 index_metric=None, faiss_index_ivf_nlists=None, faiss_index_pq_m=None, **kwargs):
        """k-nearest neighbors (KNN) is :class:`Estimator` used for classification and regression.
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
        deploy returns a :class:`~sagemaker.amazon.knn.KNNPredictor` object that can be used
        for inference calls using the trained model hosted in the SageMaker Endpoint.
        KNN Estimators can be configured by setting hyperparameters. The available hyperparameters for
        KNN are documented below.
        For further information on the AWS KNN algorithm,
        please consult AWS technical documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/knn.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and
                APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if accessing AWS resource.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            k (int): Required. Number of nearest neighbors.
            sample_size(int): Required. Number of data points to be sampled from the training data set.
            predictor_type (str): Required. Type of inference to use on the data's labels,
                allowed values are 'classifier' and 'regressor'.
            dimension_reduction_type (str): Optional. Type of dimension reduction technique to use.
                Valid values: "sign", "fjlt"
            dimension_reduction_target (int): Optional. Target dimension to reduce to. Required when
                dimension_reduction_type is specified.
            index_type (str): Optional. Type of index to use. Valid values are
                "faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ".
            index_metric(str): Optional. Distance metric to measure between points when finding nearest neighbors.
                Valid values are "COSINE", "INNER_PRODUCT", "L2"
            faiss_index_ivf_nlists(str): Optional. Number of centroids to construct in the index if
                index_type is "faiss.IVFFlat" or "faiss.IVFPQ".
            faiss_index_pq_m(int): Optional. Number of vector sub-components to construct in the index,
                if index_type is "faiss.IVFPQ".
            **kwargs: base class keyword argument values.
        """

        super(KNN, self).__init__(role, train_instance_count, train_instance_type, **kwargs)
        self.k = k
        self.sample_size = sample_size
        self.predictor_type = predictor_type
        self.dimension_reduction_type = dimension_reduction_type
        self.dimension_reduction_target = dimension_reduction_target
        self.index_type = index_type
        self.index_metric = index_metric
        self.faiss_index_ivf_nlists = faiss_index_ivf_nlists
        self.faiss_index_pq_m = faiss_index_pq_m
        if dimension_reduction_type and not dimension_reduction_target:
            raise ValueError('"dimension_reduction_target" is required when "dimension_reduction_type" is set.')

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT):
        """Return a :class:`~sagemaker.amazon.KNNModel` referencing the latest
        s3 model data produced by this Estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
        """
        return KNNModel(self.model_data, self.role, sagemaker_session=self.sagemaker_session,
                        vpc_config=self.get_vpc_config(vpc_config_override))

    def _prepare_for_training(self, records, mini_batch_size=None, job_name=None):
        super(KNN, self)._prepare_for_training(records, mini_batch_size=mini_batch_size, job_name=job_name)


class KNNPredictor(RealTimePredictor):
    """Performs classification or regression prediction from input vectors.

    The implementation of :meth:`~sagemaker.predictor.RealTimePredictor.predict` in this
    `RealTimePredictor` requires a numpy ``ndarray`` as input. The array should contain the
    same number of columns as the feature-dimension of the data used to fit the model this
    Predictor performs inference on.

    :func:`predict` returns a list of :class:`~sagemaker.amazon.record_pb2.Record` objects, one
    for each row in the input ``ndarray``. The prediction is stored in the ``"predicted_label"``
    key of the ``Record.label`` field."""

    def __init__(self, endpoint, sagemaker_session=None):
        super(KNNPredictor, self).__init__(endpoint, sagemaker_session, serializer=numpy_to_record_serializer(),
                                           deserializer=record_deserializer())


class KNNModel(Model):
    """Reference S3 model data created by KNN estimator. Calling :meth:`~sagemaker.model.Model.deploy`
    creates an Endpoint and returns :class:`KNNPredictor`."""

    def __init__(self, model_data, role, sagemaker_session=None, **kwargs):
        sagemaker_session = sagemaker_session or Session()
        repo = '{}:{}'.format(KNN.repo_name, KNN.repo_version)
        image = '{}/{}'.format(registry(sagemaker_session.boto_session.region_name, KNN.repo_name), repo)
        super(KNNModel, self).__init__(model_data, image, role, predictor_cls=KNNPredictor,
                                       sagemaker_session=sagemaker_session, **kwargs)
