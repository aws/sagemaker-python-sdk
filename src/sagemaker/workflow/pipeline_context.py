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
"""The pipeline context for workflow"""
from __future__ import absolute_import

import warnings
import inspect
from typing import Dict

from sagemaker.session import Session, SessionSettings


class PipelineSession(Session):
    """Managing interactions with SageMaker APIs and AWS services needed under Pipeline Context

    This class inherits the SageMaker session, it provides convenient methods
    for manipulating entities and resources that Amazon SageMaker uses,
    such as training jobs, endpoints, and input datasets in S3. When composing
    SageMaker Model-Building Pipeline, PipelineSession is recommended over
    regular SageMaker Session
    """

    def __init__(
        self,
        boto_session=None,
        sagemaker_client=None,
        default_bucket=None,
        settings=SessionSettings(),
    ):
        """Initialize a ``PipelineSession``.

        Args:
            boto_session (boto3.session.Session): The underlying Boto3 session which AWS service
                calls are delegated to (default: None). If not provided, one is created with
                default AWS configuration chain.
            sagemaker_client (boto3.SageMaker.Client): Client which makes Amazon SageMaker service
                calls other than ``InvokeEndpoint`` (default: None). Estimators created using this
                ``Session`` use this client. If not provided, one will be created using this
                instance's ``boto_session``.
            default_bucket (str): The default Amazon S3 bucket to be used by this session.
                This will be created the next time an Amazon S3 bucket is needed (by calling
                :func:`default_bucket`).
                If not provided, a default bucket will be created based on the following format:
                "sagemaker-{region}-{aws-account-id}".
                Example: "sagemaker-my-custom-bucket".
            settings (sagemaker.session_settings.SessionSettings): Optional. Set of optional
                parameters to apply to the session.
        """
        super().__init__(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            default_bucket=default_bucket,
            settings=settings,
        )
        self._context = None

    @property
    def context(self):
        """Hold contextual information useful to the session"""
        return self._context

    @context.setter
    def context(self, args: Dict):
        self._context = args

    def _intercept_create_request(self, request: Dict, create):
        self.context = request


def runnable_by_pipeline(run_func):
    """A convenient Decorator

    This is a decorator designed to annotate, during pipeline session,
    the methods that downstream managed to
        1. preprocess user inputs, outputs, and configurations
        2. generate the create request
        3. start the job.
    For instance, `Processor.run`, `Estimator.fit`, or `Transformer.transform`.
    This decorator will essentially run 1, and capture the request shape from 2,
    then instead of starting a new job in 3, it will return request shape from 2
    to `sagemaker.workflow.steps.Step`. The request shape will be used to construct
    the arguments needed to compose that particular step as part of the pipeline.
    The job will be started during pipeline execution.
    """

    def wrapper(*args, **kwargs):
        if isinstance(args[0].sagemaker_session, PipelineSession):
            run_func_sig = inspect.signature(run_func)
            arg_list = list(args)

            override_wait, override_logs = False, False
            for i, (arg_name, _) in enumerate(run_func_sig.parameters.items()):
                if i >= len(arg_list):
                    break
                if arg_name == "wait":
                    override_wait = True
                    arg_list[i] = False
                elif arg_name == "logs":
                    override_logs = True
                    arg_list[i] = False

            args = tuple(arg_list)

            if not override_wait:
                kwargs["wait"] = False
            if not override_logs:
                kwargs["logs"] = False

            warnings.warn(
                "Running within a PipelineSession, there will be No Wait, "
                "No Logs, and No Job being started.",
                UserWarning,
            )
            run_func(*args, **kwargs)
            return args[0].sagemaker_session.context

        run_func(*args, **kwargs)
        return None

    return wrapper
