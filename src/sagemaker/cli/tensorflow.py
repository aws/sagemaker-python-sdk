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

from sagemaker.cli.common import HostCommand, TrainCommand


def train(args):
    TensorFlowTrainCommand(args).start()


def host(args):
    TensorFlowHostCommand(args).start()


class TensorFlowTrainCommand(TrainCommand):
    def __init__(self, args):
        super(TensorFlowTrainCommand, self).__init__(args)
        self.training_steps = args.training_steps
        self.evaluation_steps = args.evaluation_steps

    def create_estimator(self):
        from sagemaker.tensorflow import TensorFlow
        return TensorFlow(training_steps=self.training_steps,
                          evaluation_steps=self.evaluation_steps,
                          py_version=self.python,
                          entry_point=self.script,
                          role=self.role_name,
                          base_job_name=self.job_name,
                          train_instance_count=self.instance_count,
                          train_instance_type=self.instance_type,
                          hyperparameters=self.hyperparameters)


class TensorFlowHostCommand(HostCommand):
    def __init__(self, args):
        super(TensorFlowHostCommand, self).__init__(args)

    def create_model(self, model_url):
        from sagemaker.tensorflow.model import TensorFlowModel
        return TensorFlowModel(model_data=model_url, role=self.role_name, entry_point=self.script,
                               py_version=self.python, name=self.endpoint_name, env=self.environment)
