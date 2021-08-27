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
"""A class for SageMaker AutoML Jobs."""
from __future__ import absolute_import

import logging
from six import string_types

from sagemaker import Model, PipelineModel
from sagemaker.automl.candidate_estimator import CandidateEstimator
from sagemaker.job import _Job
from sagemaker.session import Session
from sagemaker.utils import name_from_base

logger = logging.getLogger("sagemaker")


class AutoML(object):
    """A class for creating and interacting with SageMaker AutoML jobs."""

    def __init__(
        self,
        role,
        target_attribute_name,
        output_kms_key=None,
        output_path=None,
        base_job_name=None,
        compression_type=None,
        sagemaker_session=None,
        volume_kms_key=None,
        encrypt_inter_container_traffic=False,
        vpc_config=None,
        problem_type=None,
        max_candidates=None,
        max_runtime_per_training_job_in_seconds=None,
        total_job_runtime_in_seconds=None,
        job_objective=None,
        generate_candidate_definitions_only=False,
        tags=None,
    ):
        self.role = role
        self.output_kms_key = output_kms_key
        self.output_path = output_path
        self.base_job_name = base_job_name
        self.compression_type = compression_type
        self.volume_kms_key = volume_kms_key
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        self.vpc_config = vpc_config
        self.problem_type = problem_type
        self.max_candidate = max_candidates
        self.max_runtime_per_training_job_in_seconds = max_runtime_per_training_job_in_seconds
        self.total_job_runtime_in_seconds = total_job_runtime_in_seconds
        self.target_attribute_name = target_attribute_name
        self.job_objective = job_objective
        self.generate_candidate_definitions_only = generate_candidate_definitions_only
        self.tags = tags

        self.current_job_name = None
        self._auto_ml_job_desc = None
        self._best_candidate = None
        self.sagemaker_session = sagemaker_session or Session()

        self._check_problem_type_and_job_objective(self.problem_type, self.job_objective)

    def fit(self, inputs=None, wait=True, logs=True, job_name=None):
        """Create an AutoML Job with the input dataset.

        Args:
            inputs (str or list[str] or AutoMLInput): Local path or S3 Uri where the training data
                is stored. Or an AutoMLInput object. If a local path is provided, the dataset will
                be uploaded to an S3 location.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job. Only meaningful when wait
                is True (default: True). if ``wait`` is False, ``logs`` will be set to False as
                well.
            job_name (str): Training job name. If not specified, the estimator generates
                a default job name, based on the training image name and current timestamp.
        """
        if not wait and logs:
            logs = False
            logger.warning("Setting logs to False. logs is only meaningful when wait is True.")

        # upload data for users if provided local path
        # validations are done in _Job._format_inputs_to_input_config
        if isinstance(inputs, string_types):
            if not inputs.startswith("s3://"):
                inputs = self.sagemaker_session.upload_data(inputs, key_prefix="auto-ml-input-data")
        self._prepare_for_auto_ml_job(job_name=job_name)

        self.latest_auto_ml_job = AutoMLJob.start_new(self, inputs)  # pylint: disable=W0201
        if wait:
            self.latest_auto_ml_job.wait(logs=logs)

    @classmethod
    def attach(cls, auto_ml_job_name, sagemaker_session=None):
        """Attach to an existing AutoML job.

        Creates and returns a AutoML bound to an existing automl job.

        Args:
            auto_ml_job_name (str): AutoML job name
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, the one originally associated with the ``AutoML`` instance is used.

        Returns:
            sagemaker.automl.AutoML: A ``AutoML`` instance with the attached automl job.

        """
        sagemaker_session = sagemaker_session or Session()

        auto_ml_job_desc = sagemaker_session.describe_auto_ml_job(auto_ml_job_name)
        automl_job_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=auto_ml_job_desc["AutoMLJobArn"]
        )["Tags"]

        amlj = AutoML(
            role=auto_ml_job_desc["RoleArn"],
            target_attribute_name=auto_ml_job_desc["InputDataConfig"][0]["TargetAttributeName"],
            output_kms_key=auto_ml_job_desc["OutputDataConfig"].get("KmsKeyId"),
            output_path=auto_ml_job_desc["OutputDataConfig"]["S3OutputPath"],
            base_job_name=auto_ml_job_name,
            compression_type=auto_ml_job_desc["InputDataConfig"][0].get("CompressionType"),
            sagemaker_session=sagemaker_session,
            volume_kms_key=auto_ml_job_desc.get("AutoMLJobConfig", {})
            .get("SecurityConfig", {})
            .get("VolumeKmsKeyId"),
            encrypt_inter_container_traffic=auto_ml_job_desc.get("AutoMLJobConfig", {})
            .get("SecurityConfig", {})
            .get("EnableInterContainerTrafficEncryption", False),
            vpc_config=auto_ml_job_desc.get("AutoMLJobConfig", {})
            .get("SecurityConfig", {})
            .get("VpcConfig"),
            problem_type=auto_ml_job_desc.get("ProblemType"),
            max_candidates=auto_ml_job_desc.get("AutoMLJobConfig", {})
            .get("CompletionCriteria", {})
            .get("MaxCandidates"),
            max_runtime_per_training_job_in_seconds=auto_ml_job_desc.get("AutoMLJobConfig", {})
            .get("CompletionCriteria", {})
            .get("MaxRuntimePerTrainingJobInSeconds"),
            total_job_runtime_in_seconds=auto_ml_job_desc.get("AutoMLJobConfig", {})
            .get("CompletionCriteria", {})
            .get("MaxAutoMLJobRuntimeInSeconds"),
            job_objective=auto_ml_job_desc.get("AutoMLJobObjective", {}).get("MetricName"),
            generate_candidate_definitions_only=auto_ml_job_desc.get(
                "GenerateCandidateDefinitionsOnly", False
            ),
            tags=automl_job_tags,
        )
        amlj.current_job_name = auto_ml_job_name
        amlj.latest_auto_ml_job = auto_ml_job_name  # pylint: disable=W0201
        amlj._auto_ml_job_desc = auto_ml_job_desc
        return amlj

    def describe_auto_ml_job(self, job_name=None):
        """Returns the job description of an AutoML job for the given job name.

        Args:
            job_name (str): The name of the AutoML job to describe.
                If None, will use object's latest_auto_ml_job name.

        Returns:
            dict: A dictionary response with the AutoML Job description.
        """
        if job_name is None:
            job_name = self.current_job_name
        self._auto_ml_job_desc = self.sagemaker_session.describe_auto_ml_job(job_name)
        return self._auto_ml_job_desc

    def best_candidate(self, job_name=None):
        """Returns the best candidate of an AutoML job for a given name.

        Args:
            job_name (str): The name of the AutoML job. If None, will use object's
                _current_auto_ml_job_name.

        Returns:
            dict: A dictionary with information of the best candidate.
        """
        if self._best_candidate:
            return self._best_candidate

        if job_name is None:
            job_name = self.current_job_name
        if self._auto_ml_job_desc is None:
            self._auto_ml_job_desc = self.sagemaker_session.describe_auto_ml_job(job_name)
        elif self._auto_ml_job_desc["AutoMLJobName"] != job_name:
            self._auto_ml_job_desc = self.sagemaker_session.describe_auto_ml_job(job_name)

        self._best_candidate = self._auto_ml_job_desc["BestCandidate"]
        return self._best_candidate

    def list_candidates(
        self,
        job_name=None,
        status_equals=None,
        candidate_name=None,
        candidate_arn=None,
        sort_order=None,
        sort_by=None,
        max_results=None,
    ):
        """Returns the list of candidates of an AutoML job for a given name.

        Args:
            job_name (str): The name of the AutoML job. If None, will use object's
                _current_job name.
            status_equals (str): Filter the result with candidate status, values could be
                "Completed", "InProgress", "Failed", "Stopped", "Stopping"
            candidate_name (str): The name of a specified candidate to list.
                Default to None.
            candidate_arn (str): The Arn of a specified candidate to list.
                Default to None.
            sort_order (str): The order that the candidates will be listed in result.
                Default to None.
            sort_by (str): The value that the candidates will be sorted by.
                Default to None.
            max_results (int): The number of candidates will be listed in results,
                between 1 to 100. Default to None. If None, will return all the candidates.

        Returns:
            list: A list of dictionaries with candidates information.
        """
        if job_name is None:
            job_name = self.current_job_name

        list_candidates_args = {"job_name": job_name}

        if status_equals:
            list_candidates_args["status_equals"] = status_equals
        if candidate_name:
            list_candidates_args["candidate_name"] = candidate_name
        if candidate_arn:
            list_candidates_args["candidate_arn"] = candidate_arn
        if sort_order:
            list_candidates_args["sort_order"] = sort_order
        if sort_by:
            list_candidates_args["sort_by"] = sort_by
        if max_results:
            list_candidates_args["max_results"] = max_results

        return self.sagemaker_session.list_candidates(**list_candidates_args)["Candidates"]

    def create_model(
        self,
        name,
        sagemaker_session=None,
        candidate=None,
        vpc_config=None,
        enable_network_isolation=False,
        model_kms_key=None,
        predictor_cls=None,
        inference_response_keys=None,
    ):
        """Creates a model from a given candidate or the best candidate from the job.

        Args:
            name (str): The pipeline model name.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, the one originally associated with the ``AutoML`` instance is used.:
            candidate (CandidateEstimator or dict): a CandidateEstimator used for deploying
                to a SageMaker Inference Pipeline. If None, the best candidate will
                be used. If the candidate input is a dict, a CandidateEstimator will be
                created from it.
            vpc_config (dict): Specifies a VPC that your training jobs and hosted models have
                access to. Contents include "SecurityGroupIds" and "Subnets".
            enable_network_isolation (bool): Isolates the training container. No inbound or
                outbound network calls can be made, except for calls between peers within a
                training cluster for distributed training. Default: False
            model_kms_key (str): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If
                specified, ``deploy()``  returns the result of invoking this
                function on the created endpoint name.
            inference_response_keys (list): List of keys for response content. The order of the
                keys will dictate the content order in the response.

        Returns:
            PipelineModel object.
        """
        sagemaker_session = sagemaker_session or self.sagemaker_session

        if candidate is None:
            candidate_dict = self.best_candidate()
            candidate = CandidateEstimator(candidate_dict, sagemaker_session=sagemaker_session)
        elif isinstance(candidate, dict):
            candidate = CandidateEstimator(candidate, sagemaker_session=sagemaker_session)

        inference_containers = candidate.containers

        self.validate_and_update_inference_response(inference_containers, inference_response_keys)

        # construct Model objects
        models = []

        for container in inference_containers:
            image_uri = container["Image"]
            model_data = container["ModelDataUrl"]
            env = container["Environment"]

            model = Model(
                image_uri=image_uri,
                model_data=model_data,
                role=self.role,
                env=env,
                vpc_config=vpc_config,
                sagemaker_session=sagemaker_session or self.sagemaker_session,
                enable_network_isolation=enable_network_isolation,
                model_kms_key=model_kms_key,
            )
            models.append(model)

        pipeline = PipelineModel(
            models=models,
            role=self.role,
            predictor_cls=predictor_cls,
            name=name,
            vpc_config=vpc_config,
            enable_network_isolation=enable_network_isolation,
            sagemaker_session=sagemaker_session or self.sagemaker_session,
        )
        return pipeline

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        serializer=None,
        deserializer=None,
        candidate=None,
        sagemaker_session=None,
        name=None,
        endpoint_name=None,
        tags=None,
        wait=True,
        vpc_config=None,
        enable_network_isolation=False,
        model_kms_key=None,
        predictor_cls=None,
        inference_response_keys=None,
    ):
        """Deploy a candidate to a SageMaker Inference Pipeline.

        Args:
            initial_instance_count (int): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``.
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
                serializer object, used to encode data for an inference endpoint
                (default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``.
            deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
                deserializer object, used to decode data from an inference
                endpoint (default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``.
            candidate (CandidateEstimator or dict): a CandidateEstimator used for deploying
                to a SageMaker Inference Pipeline. If None, the best candidate will
                be used. If the candidate input is a dict, a CandidateEstimator will be
                created from it.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, the one originally associated with the ``AutoML`` instance is used.
            name (str): The pipeline model name. If None, a default model name will
                be selected on each ``deploy``.
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
            tags (List[dict[str, str]]): The list of tags to attach to this
                specific endpoint.
            wait (bool): Whether the call should wait until the deployment of
                model completes (default: True).
            vpc_config (dict): Specifies a VPC that your training jobs and hosted models have
                access to. Contents include "SecurityGroupIds" and "Subnets".
            enable_network_isolation (bool): Isolates the training container. No inbound or
                outbound network calls can be made, except for calls between peers within a
                training cluster for distributed training. Default: False
            model_kms_key (str): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If
                specified, ``deploy()``  returns the result of invoking this
                function on the created endpoint name.
            inference_response_keys (list): List of keys for response content. The order of the
                keys will dictate the content order in the response.

        Returns:
            callable[string, sagemaker.session.Session] or ``None``:
                If ``predictor_cls`` is specified, the invocation of ``self.predictor_cls`` on
                the created endpoint name. Otherwise, ``None``.
        """
        sagemaker_session = sagemaker_session or self.sagemaker_session
        model = self.create_model(
            name=name,
            sagemaker_session=sagemaker_session,
            candidate=candidate,
            inference_response_keys=inference_response_keys,
            vpc_config=vpc_config,
            enable_network_isolation=enable_network_isolation,
            model_kms_key=model_kms_key,
            predictor_cls=predictor_cls,
        )

        return model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            serializer=serializer,
            deserializer=deserializer,
            endpoint_name=endpoint_name,
            kms_key=model_kms_key,
            tags=tags,
            wait=wait,
        )

    def _check_problem_type_and_job_objective(self, problem_type, job_objective):
        """Validate if problem_type and job_objective are both None or are both provided.

        Args:
            problem_type (str): The type of problem of this AutoMLJob. Valid values are
                "Regression", "BinaryClassification", "MultiClassClassification".
            job_objective (dict): AutoMLJob objective, contains "AutoMLJobObjectiveType" (optional),
                "MetricName" and "Value".

        Raises (ValueError): raises ValueError if one of problem_type and job_objective is provided
            while the other is None.
        """
        if not (problem_type and job_objective) and (problem_type or job_objective):
            raise ValueError(
                "One of problem type and objective metric provided. "
                "Either both of them should be provided or none of them should be provided."
            )

    def _prepare_for_auto_ml_job(self, job_name=None):
        """Set any values in the AutoMLJob that need to be set before creating request.

        Args:
            job_name (str): The name of the AutoML job. If None, a job name will be
                created from base_job_name or "sagemaker-auto-ml".
        """
        if job_name is not None:
            self.current_job_name = job_name
        else:
            if self.base_job_name:
                base_name = self.base_job_name
            else:
                base_name = "automl"
            # CreateAutoMLJob API validates that member length less than or equal to 32
            self.current_job_name = name_from_base(base_name, max_length=32)

        if self.output_path is None:
            self.output_path = "s3://{}/".format(self.sagemaker_session.default_bucket())

    @classmethod
    def _get_supported_inference_keys(cls, container, default=None):
        """Returns the inference keys supported by the container.

        Args:
            container (dict): Dictionary representing container
            default (object): The value to be returned if the container definition
                              has no marker environment variable

        Returns:
            List of keys the container support or default

        Raises:
            KeyError if the default is None and the container definition has
            no marker environment variable SAGEMAKER_INFERENCE_SUPPORTED.
        """
        try:
            return [
                x.strip()
                for x in container["Environment"]["SAGEMAKER_INFERENCE_SUPPORTED"].split(",")
            ]
        except KeyError:
            if default is None:
                raise
        return default

    @classmethod
    def _check_inference_keys(cls, inference_response_keys, containers):
        """Checks if the pipeline supports the inference keys for the containers.

        Given inference response keys and list of containers, determines whether
        the keys are supported.

        Args:
            inference_response_keys (list): List of keys for inference response content.
            containers (list): list of inference container.

        Raises:
            ValueError, if one or more keys in inference_response_keys are not supported
            the inference pipeline.
        """
        if not inference_response_keys:
            return
        try:
            supported_inference_keys = cls._get_supported_inference_keys(container=containers[-1])
        except KeyError:
            raise ValueError(
                "The inference model does not support selection of inference content beyond "
                "it's default content. Please retry without setting "
                "inference_response_keys key word argument."
            )
        bad_keys = []
        for key in inference_response_keys:
            if key not in supported_inference_keys:
                bad_keys.append(key)

        if bad_keys:
            raise ValueError(
                "Requested inference output keys [{bad_keys_str}] are unsupported. "
                "The supported inference keys are [{allowed_keys_str}]".format(
                    bad_keys_str=", ".join(bad_keys),
                    allowed_keys_str=", ".join(supported_inference_keys),
                )
            )

    @classmethod
    def validate_and_update_inference_response(cls, inference_containers, inference_response_keys):
        """Validates the requested inference keys and updates response content.

        On validation, also updates the inference containers to emit appropriate response
        content in the inference response.

        Args:
            inference_containers (list): list of inference containers
            inference_response_keys (list): list of inference response keys

        Raises:
            ValueError: if one or more of inference_response_keys are unsupported by the model
        """
        if not inference_response_keys:
            return

        cls._check_inference_keys(inference_response_keys, inference_containers)

        previous_container_output = None

        for container in inference_containers:
            supported_inference_keys_container = cls._get_supported_inference_keys(
                container, default=[]
            )
            if not supported_inference_keys_container:
                previous_container_output = None
                continue
            current_container_output = None
            for key in inference_response_keys:
                if key in supported_inference_keys_container:
                    current_container_output = (
                        current_container_output + "," + key if current_container_output else key
                    )

            if previous_container_output:
                container["Environment"].update(
                    {"SAGEMAKER_INFERENCE_INPUT": previous_container_output}
                )
            if current_container_output:
                container["Environment"].update(
                    {"SAGEMAKER_INFERENCE_OUTPUT": current_container_output}
                )
            previous_container_output = current_container_output


class AutoMLInput(object):
    """Accepts parameters that specify an S3 input for an auto ml job

    Provides a method to turn those parameters into a dictionary.
    """

    def __init__(self, inputs, target_attribute_name, compression=None):
        """Convert an S3 Uri or a list of S3 Uri to an AutoMLInput object.

        :param inputs (str, list[str]): a string or a list of string that points to (a)
            S3 location(s) where input data is stored.
        :param target_attribute_name (str): the target attribute name for regression
            or classification.
        :param compression (str): if training data is compressed, the compression type.
            The default value is None.
        """
        self.inputs = inputs
        self.target_attribute_name = target_attribute_name
        self.compression = compression

    def to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        # Create the request dictionary.
        auto_ml_input = []
        if isinstance(self.inputs, string_types):
            self.inputs = [self.inputs]
        for entry in self.inputs:
            input_entry = {
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": entry}},
                "TargetAttributeName": self.target_attribute_name,
            }
            if self.compression is not None:
                input_entry["CompressionType"] = self.compression
            auto_ml_input.append(input_entry)
        return auto_ml_input


class AutoMLJob(_Job):
    """A class for interacting with CreateAutoMLJob API."""

    def __init__(self, sagemaker_session, job_name, inputs):
        self.inputs = inputs
        self.job_name = job_name
        super(AutoMLJob, self).__init__(sagemaker_session=sagemaker_session, job_name=job_name)

    @classmethod
    def start_new(cls, auto_ml, inputs):
        """Create a new Amazon SageMaker AutoML job from auto_ml.

        Args:
            auto_ml (sagemaker.automl.AutoML): AutoML object
                created by the user.
            inputs (str, list[str]): Parameters used when called
                :meth:`~sagemaker.automl.AutoML.fit`.

        Returns:
            sagemaker.automl.AutoMLJob: Constructed object that captures
            all information about the started AutoML job.
        """
        config = cls._load_config(inputs, auto_ml)
        auto_ml_args = config.copy()
        auto_ml_args["job_name"] = auto_ml.current_job_name
        auto_ml_args["problem_type"] = auto_ml.problem_type
        auto_ml_args["job_objective"] = auto_ml.job_objective
        auto_ml_args["tags"] = auto_ml.tags

        auto_ml.sagemaker_session.auto_ml(**auto_ml_args)
        return cls(auto_ml.sagemaker_session, auto_ml.current_job_name, inputs)

    @classmethod
    def _load_config(cls, inputs, auto_ml, expand_role=True, validate_uri=True):
        """Load job_config, input_config and output config from auto_ml and inputs.

        Args:
            inputs (str): S3 Uri where the training data is stored, must start
                with "s3://".
            auto_ml (AutoML): an AutoML object that user initiated.
            expand_role (str): The expanded role arn that allows for Sagemaker
                executionts.
            validate_uri (bool): indicate whether to validate the S3 uri.

        Returns (dict): a config dictionary that contains input_config, output_config,
            job_config and role information.

        """
        # JobConfig
        # InputDataConfig
        # OutputConfig

        if isinstance(inputs, AutoMLInput):
            input_config = inputs.to_request_dict()
        else:
            input_config = cls._format_inputs_to_input_config(
                inputs, validate_uri, auto_ml.compression_type, auto_ml.target_attribute_name
            )
        output_config = _Job._prepare_output_config(auto_ml.output_path, auto_ml.output_kms_key)

        role = auto_ml.sagemaker_session.expand_role(auto_ml.role) if expand_role else auto_ml.role

        stop_condition = cls._prepare_auto_ml_stop_condition(
            auto_ml.max_candidate,
            auto_ml.max_runtime_per_training_job_in_seconds,
            auto_ml.total_job_runtime_in_seconds,
        )

        auto_ml_job_config = {
            "CompletionCriteria": stop_condition,
            "SecurityConfig": {
                "EnableInterContainerTrafficEncryption": auto_ml.encrypt_inter_container_traffic
            },
        }

        if auto_ml.volume_kms_key:
            auto_ml_job_config["SecurityConfig"]["VolumeKmsKeyId"] = auto_ml.volume_kms_key
        if auto_ml.vpc_config:
            auto_ml_job_config["SecurityConfig"]["VpcConfig"] = auto_ml.vpc_config

        config = {
            "input_config": input_config,
            "output_config": output_config,
            "auto_ml_job_config": auto_ml_job_config,
            "role": role,
            "generate_candidate_definitions_only": auto_ml.generate_candidate_definitions_only,
        }
        return config

    @classmethod
    def _format_inputs_to_input_config(
        cls, inputs, validate_uri=True, compression=None, target_attribute_name=None
    ):
        """Convert inputs to AutoML InputDataConfig.

        Args:
            inputs (str, list[str]): local path(s) or S3 uri(s) of input datasets.
            validate_uri (bool): indicates whether it is needed to validate S3 uri.
            compression (str): Compression type of the input data.
            target_attribute_name (str): the target attribute name for classification
                or regression.

        Returns (dict): a dict of AutoML InputDataConfig
        """
        if inputs is None:
            return None

        channels = []
        if isinstance(inputs, AutoMLInput):
            channels.append(inputs.to_request_dict())
        elif isinstance(inputs, string_types):
            channel = _Job._format_string_uri_input(
                inputs,
                validate_uri,
                compression=compression,
                target_attribute_name=target_attribute_name,
            ).config
            channels.append(channel)
        elif isinstance(inputs, list):
            for input_entry in inputs:
                channel = _Job._format_string_uri_input(
                    input_entry,
                    validate_uri,
                    compression=compression,
                    target_attribute_name=target_attribute_name,
                ).config
                channels.append(channel)
        else:
            msg = "Cannot format input {}. Expecting a string or a list of strings."
            raise ValueError(msg.format(inputs))

        for channel in channels:
            if channel["TargetAttributeName"] is None:
                raise ValueError("TargetAttributeName cannot be None.")

        return channels

    @classmethod
    def _prepare_auto_ml_stop_condition(
        cls, max_candidates, max_runtime_per_training_job_in_seconds, total_job_runtime_in_seconds
    ):
        """Defines the CompletionCriteria of an AutoMLJob.

        Args:
            max_candidates (int): the maximum number of candidates returned by an
                AutoML job.
            max_runtime_per_training_job_in_seconds (int): the maximum time of each
                training job in seconds.
            total_job_runtime_in_seconds (int): the total wait time of an AutoML job.

        Returns (dict): an AutoML CompletionCriteria.
        """
        stopping_condition = {"MaxCandidates": max_candidates}

        if max_runtime_per_training_job_in_seconds is not None:
            stopping_condition[
                "MaxRuntimePerTrainingJobInSeconds"
            ] = max_runtime_per_training_job_in_seconds
        if total_job_runtime_in_seconds is not None:
            stopping_condition["MaxAutoMLJobRuntimeInSeconds"] = total_job_runtime_in_seconds

        return stopping_condition

    def describe(self):
        """Prints out a response from the DescribeAutoMLJob API call."""
        return self.sagemaker_session.describe_auto_ml_job(self.job_name)

    def wait(self, logs=True):
        """Wait for the AutoML job to finish.

        Args:
            logs (bool): indicate whether to output logs.
        """
        if logs:
            self.sagemaker_session.logs_for_auto_ml_job(self.job_name, wait=True)
        else:
            self.sagemaker_session.wait_for_auto_ml_job(self.job_name)
