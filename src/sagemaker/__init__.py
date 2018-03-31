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
from __future__ import absolute_import

from sagemaker import estimator
from sagemaker.amazon.kmeans import KMeans, KMeansModel, KMeansPredictor
from sagemaker.amazon.pca import PCA, PCAModel, PCAPredictor
from sagemaker.amazon.lda import LDA, LDAModel, LDAPredictor
from sagemaker.amazon.linear_learner import LinearLearner, LinearLearnerModel, LinearLearnerPredictor
from sagemaker.amazon.factorization_machines import FactorizationMachines, FactorizationMachinesModel
from sagemaker.amazon.factorization_machines import FactorizationMachinesPredictor
from sagemaker.amazon.ntm import NTM, NTMModel, NTMPredictor

from sagemaker.model import Model
from sagemaker.predictor import RealTimePredictor
from sagemaker.session import Session
from sagemaker.session import container_def
from sagemaker.session import production_variant
from sagemaker.session import s3_input
from sagemaker.session import get_execution_role


__all__ = [estimator, KMeans, KMeansModel, KMeansPredictor, PCA, PCAModel, PCAPredictor, LinearLearner,
           LinearLearnerModel, LinearLearnerPredictor,
           LDA, LDAModel, LDAPredictor,
           FactorizationMachines, FactorizationMachinesModel, FactorizationMachinesPredictor,
           Model, NTM, NTMModel, NTMPredictor, RealTimePredictor, Session,
           container_def, s3_input, production_variant, get_execution_role]
