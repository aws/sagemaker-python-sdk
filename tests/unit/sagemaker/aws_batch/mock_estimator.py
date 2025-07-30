from __future__ import absolute_import
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorch


class Estimator(Estimator):
    def __init__(self):
        self.sagemaker_session = Session()
        self.tags = [
            {"Key": "batch-non-prod", "Value": "true"},
            {"Key": "batch-training-job-name", "Value": "training-job"},
        ]

    def prepare_workflow_for_training(self, job_name):
        pass


class PyTorch(PyTorch):
    def __init__(self):
        self.sagemaker_session = Session()
        self.tags = [
            {"Key": "batch-non-prod", "Value": "true"},
            {"Key": "batch-training-job-name", "Value": "training-job"},
        ]

    def prepare_workflow_for_training(self, job_name):
        pass


class Session:
    def __init__(self):
        pass

    def get_train_request(self, **kwargs):
        return kwargs
