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
from sagemaker.estimator import Framework
from sagemaker.fw_utils import framework_name_from_image, create_image_uri
from sagemaker.keras.model import KerasModel

LATEST_VERSION = '2.1.4'


class Keras(Framework):
    __framework_name__ = "keras"

    def __init__(self, entry_point, source_dir=None, hyperparameters=None, py_version='py2', **kwargs):
        super(Keras, self).__init__(entry_point, source_dir, hyperparameters, **kwargs)
        self.py_version = py_version

    def train_image(self):
        instance_type = 'gpu' if self.train_instance_type[3] in ['p', 'g'] else 'cpu'
        tag = '{}-{}-{}'.format(LATEST_VERSION, instance_type, self.py_version)

        region_name = self.sagemaker_session.boto_session.region_name
        self.image =  create_image_uri(region_name, self.sagemaker_session.aws_account(), self.__framework_name__, tag)
        return self.image

    def create_model(self, model_server_workers=None):
        return KerasModel(self.model_data, self.role, self.entry_point, source_dir=self.source_dir,
                          enable_cloudwatch_metrics=self.enable_cloudwatch_metrics, name=self._current_job_name,
                          container_log_level=self.container_log_level, code_location=self.code_location,
                          py_version=self.py_version, model_server_workers=model_server_workers,
                          sagemaker_session=self.sagemaker_session, image=self.image)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        init_params = super(Keras, cls)._prepare_init_params_from_job_description(job_details)
        framework, py_version = framework_name_from_image(init_params.pop('image'))

        init_params['py_version'] = py_version
        training_job_name = init_params['base_job_name']

        if framework != cls.__framework_name__:
            raise ValueError("Training job: {} didn't use image for requested framework".format(training_job_name))

        return init_params
