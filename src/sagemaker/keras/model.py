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

import sagemaker
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer
from sagemaker.utils import name_from_image


class KerasPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session=None):
        super(KerasPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)


class KerasModel(FrameworkModel):
    __framework_name__ = 'keras'

    def __init__(self, model_data, role, entry_point, image=None, py_version='py2',
                 predictor_cls=KerasPredictor, model_server_workers=None, **kwargs):
        super(KerasModel, self).__init__(model_data, image, role, entry_point, predictor_cls=predictor_cls,
                                         **kwargs)
        self.py_version = py_version
        self.model_server_workers = model_server_workers

    def prepare_container_def(self, instance_type):
        """Return a container definition with framework configuration set in model environment variables.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.

        Returns:
            dict[str, str]: A container definition object usable with the CreateModel API.
        """
        deploy_image = self.image
        deploy_key_prefix = self.key_prefix or self.name or name_from_image(deploy_image)
        self._upload_code(deploy_key_prefix)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(self.model_server_workers)

        return sagemaker.container_def(deploy_image, self.model_data, deploy_env)