# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Helper classes that handles intelligent default values for serve function"""
from __future__ import absolute_import
from typing import Dict

from sagemaker.session import Session
from sagemaker.utils import resolve_value_from_config

from sagemaker.config.config_schema import (
    SERVE_ENVIRONMENT_VARIABLES,
    SERVE_INSTANCE_TYPE,
    SERVE_ROLE_ARN,
    SERVE_S3_MODEL_DATA_URI,
    TELEMETRY_OPT_OUT_PATH,
)


class _ServeSettings(object):
    """Helper class that processes the job settings.

    It validates the job settings and provides default values if necessary.
    """

    def __init__(
        self,
        role_arn: str = None,
        s3_model_data_url: str = None,
        instance_type: str = None,
        env_vars: Dict[str, str] = None,
        sagemaker_session: Session = None,
    ):
        self.sagemaker_session = sagemaker_session or Session()

        self.role_arn = resolve_value_from_config(
            direct_input=role_arn,
            config_path=SERVE_ROLE_ARN,
            sagemaker_session=self.sagemaker_session,
        )
        self.s3_model_data_url = resolve_value_from_config(
            direct_input=s3_model_data_url,
            config_path=SERVE_S3_MODEL_DATA_URI,
            sagemaker_session=self.sagemaker_session,
        )
        self.instance_type = resolve_value_from_config(
            direct_input=instance_type,
            config_path=SERVE_INSTANCE_TYPE,
            sagemaker_session=self.sagemaker_session,
        )
        self.env_vars = resolve_value_from_config(
            direct_input=env_vars,
            config_path=SERVE_ENVIRONMENT_VARIABLES,
            default_value={},
            sagemaker_session=self.sagemaker_session,
        )
        self.telemetry_opt_out = resolve_value_from_config(
            direct_input=None,
            config_path=TELEMETRY_OPT_OUT_PATH,
            default_value=False,
            sagemaker_session=self.sagemaker_session,
        )
