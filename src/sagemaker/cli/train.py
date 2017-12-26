from __future__ import absolute_import

import json
import logging
import os

import sagemaker

logger = logging.getLogger(__name__)


def train(args):
    return TrainingCommand(args).start()


class TrainingCommand(object):
    def __init__(self, args):
        self.job_name = args.job_name
        self.bucket = args.bucket_name  # may be None
        self.role_name = args.role_name
        self.data = args.data
        self.script = args.script
        self.python = args.python
        self.instance_type = args.instance_type
        self.instance_count = args.instance_count
        self.framework = 'tensorflow' if args.tf else 'mxnet' if args.mx else 'undefined'

        self.hyperparameters = self.load_hyperparameters(args.hyperparameters)

        # tensorflow only
        self.training_steps = args.training_steps
        self.evaluation_steps = args.evaluation_steps

        self.session = sagemaker.Session()

    @staticmethod
    def load_hyperparameters(src):
        hp = {}
        if src and os.path.exists(src):
            with open(src, 'r') as f:
                hp = json.load(f)
        return hp

    def upload_training_data(self):
        prefix = '{}/data'.format(self.job_name)
        data_url = self.session.upload_data(path=self.data, bucket=self.bucket, key_prefix=prefix)
        return data_url

    def create_estimator(self):
        if self.framework == 'tensorflow':
            from sagemaker.tensorflow import TensorFlow
            return TensorFlow(self.script,
                              role=self.role_name,
                              base_job_name=self.job_name,
                              train_instance_count=self.instance_count,
                              train_instance_type=self.instance_type,
                              training_steps=self.training_steps,
                              evaluation_steps=self.evaluation_steps,
                              hyperparameters=self.hyperparameters, py_version=self.python)
        elif self.framework == 'mxnet':
            from sagemaker.mxnet.estimator import MXNet
            return MXNet(self.script,
                         role=self.role_name,
                         base_job_name=self.job_name,
                         train_instance_count=self.instance_count,
                         train_instance_type=self.instance_type,
                         hyperparameters=self.hyperparameters, py_version=self.python)
        else:
            raise ValueError('unsupported framework value: {}'.format(self.framework))

    def start(self):
        data_url = self.upload_training_data()
        estimator = self.create_estimator()
        estimator.fit(data_url)
        logger.info(' code location: {}'.format(estimator.uploaded_code.s3_prefix))
        logger.info('model location: {}{}/output/model.tar.gz'.format(estimator.output_path,
                                                                      estimator._current_job_name))
