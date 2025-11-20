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
"""Inference Recommender mixin for SageMaker model optimization.

This module provides the _InferenceRecommenderMixin class that enables SageMaker models
to use Inference Recommender for right-sizing and optimization recommendations.

Key Features:
- Automatic instance type and configuration recommendations
- Load testing with custom traffic patterns
- Performance optimization based on latency and throughput requirements
- Support for both Default and Advanced recommendation jobs

Example:
    Basic usage with a ModelBuilder::
    
        model_builder = ModelBuilder(model="my-model")
        model = model_builder.build()
        
        # Get right-sizing recommendations
        model.right_size(
            sample_payload_url="s3://my-bucket/sample-payload.json",
            supported_content_types=["application/json"],
            supported_instance_types=["ml.m5.large", "ml.m5.xlarge"]
        )
        
        # Deploy with recommendations
        predictor = model.deploy()
"""
from __future__ import absolute_import

# Standard library imports
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# SageMaker imports
from sagemaker.core.parameter import CategoricalParameter

# ========================================
# Constants
# ========================================

INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING = {
    "xgboost": "XGBOOST",
    "sklearn": "SAGEMAKER-SCIKIT-LEARN", 
    "pytorch": "PYTORCH",
    "tensorflow": "TENSORFLOW",
    "mxnet": "MXNET",
}

# Setting LOGGER for backward compatibility, in case users import it
logger = LOGGER = logging.getLogger("sagemaker")


class Phase:
    """Traffic pattern phase configuration for Advanced Inference Recommendations.

    Defines a phase of load testing with specific duration, user count, and spawn rate.
    Multiple phases can be combined to create complex traffic patterns.
    
    Args:
        duration_in_seconds: How long this phase should run
        initial_number_of_users: Number of concurrent users at start of phase
        spawn_rate: Rate at which new users are added (users per second)
        
    Example:
        Create a ramp-up phase::
        
            phase = Phase(
                duration_in_seconds=300,  # 5 minutes
                initial_number_of_users=1,
                spawn_rate=2  # Add 2 users per second
            )
    """

    def __init__(self, duration_in_seconds: int, initial_number_of_users: int, spawn_rate: int) -> None:
        """Initialize a Phase for load testing.
        
        Args:
            duration_in_seconds: Duration of this phase in seconds
            initial_number_of_users: Starting number of concurrent users
            spawn_rate: Rate of adding new users (users per second)
        """
        self.to_json = {
            "DurationInSeconds": duration_in_seconds,
            "InitialNumberOfUsers": initial_number_of_users,
            "SpawnRate": spawn_rate,
        }


class ModelLatencyThreshold:
    """Latency threshold configuration for Advanced Inference Recommendations.

    Defines acceptable response latency limits for model inference.
    Used to filter recommendations based on performance requirements.
    
    Args:
        percentile: Latency percentile to measure (e.g., "P95", "P99")
        value_in_milliseconds: Maximum acceptable latency in milliseconds
        
    Example:
        Set P95 latency threshold::
        
            threshold = ModelLatencyThreshold(
                percentile="P95",
                value_in_milliseconds=100  # 100ms max P95 latency
            )
    """

    def __init__(self, percentile: str, value_in_milliseconds: int) -> None:
        """Initialize a ModelLatencyThreshold.
        
        Args:
            percentile: Latency percentile (e.g., "P95", "P99")
            value_in_milliseconds: Maximum latency threshold in milliseconds
        """
        self.to_json = {"Percentile": percentile, "ValueInMilliseconds": value_in_milliseconds}


class _InferenceRecommenderMixin:
    """Mixin class providing SageMaker Inference Recommender functionality.
    
    This mixin adds right-sizing capabilities to SageMaker models, enabling
    automatic instance type and configuration recommendations based on model
    performance requirements.
    
    The mixin provides:
    - Automatic framework detection from container images
    - Default and Advanced recommendation job types
    - Load testing with custom traffic patterns
    - Performance-based filtering and optimization
    
    This class is designed to be mixed into Model classes that have:
    - sagemaker_session: SageMaker session for API calls
    - role_arn: IAM role for job execution
    - model_name: Name of the model
    - image_uri: Container image URI (optional, for framework detection)
    """

    def right_size(
        self,
        sample_payload_url: Optional[str] = None,
        supported_content_types: Optional[List[str]] = None,
        supported_instance_types: Optional[List[str]] = None,
        job_name: Optional[str] = None,
        framework: Optional[str] = None,
        framework_version: Optional[str] = None,
        job_duration_in_seconds: Optional[int] = None,
        hyperparameter_ranges: Optional[List[Dict[str, CategoricalParameter]]] = None,
        phases: Optional[List[Phase]] = None,
        traffic_type: Optional[str] = None,
        max_invocations: Optional[int] = None,
        model_latency_thresholds: Optional[List[ModelLatencyThreshold]] = None,
        max_tests: Optional[int] = None,
        max_parallel_tests: Optional[int] = None,
        log_level: Optional[str] = "Verbose",
    ) -> "_InferenceRecommenderMixin":
        """Recommends an instance type for a SageMaker or BYOC model.

        Create a SageMaker ``Model`` or use a registered ``ModelPackage``,
        to start an Inference Recommender job.

        The name of the created model is accessible in the ``name`` field of
        this ``Model`` after right_size returns.

        Args:
            sample_payload_url (str): The S3 path where the sample payload is stored.
            supported_content_types: (list[str]): The supported MIME types for the input data.
            supported_instance_types (list[str]): A list of the instance types that this model
                is expected to work on. (default: None).
            job_name (str): The name of the Inference Recommendations Job. (default: None).
            framework (str): The machine learning framework of the Image URI.
                Only required to specify if you bring your own custom containers (default: None).
            job_duration_in_seconds (int): The maximum job duration that a job can run for.
                (default: None).
            hyperparameter_ranges (list[Dict[str, sagemaker.parameter.CategoricalParameter]]):
                Specifies the hyper parameters to be used during endpoint load tests.
                `instance_type` must be specified as a hyperparameter range.
                `env_vars` can be specified as an optional hyperparameter range. (default: None).
                Example::

                    hyperparameter_ranges = [{
                        'instance_types': CategoricalParameter(['ml.c5.xlarge', 'ml.c5.2xlarge']),
                        'OMP_NUM_THREADS': CategoricalParameter(['1', '2', '3', '4'])
                    }]

            phases (list[Phase]): Shape of the traffic pattern to use in the load test
                (default: None).
            traffic_type (str): Specifies the traffic pattern type. Currently only supports
                one type 'PHASES' (default: None).
            max_invocations (str): defines the minimum invocations per minute for the endpoint
                to support (default: None).
            model_latency_thresholds (list[ModelLatencyThreshold]): defines the maximum response
                latency for endpoints to support (default: None).
            max_tests (int): restricts how many endpoints in total are allowed to be
                spun up for this job (default: None).
            max_parallel_tests (int): restricts how many concurrent endpoints
                this job is allowed to spin up (default: None).
            log_level (str): specifies the inline output when waiting for right_size to complete
                (default: "Verbose").

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """
        # Auto-detect framework from image URI if not provided
        if not framework and hasattr(self, 'image_uri'):
            detected_framework, detected_version = self._extract_framework_from_image_uri()
            if detected_framework:
                # Convert framework enum to string if needed
                framework_str = getattr(detected_framework, 'value', str(detected_framework)).lower()
                framework = INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING.get(
                    framework_str, 
                    str(detected_framework)
                )
            framework_version = framework_version or detected_version

        endpoint_configurations = self._convert_to_endpoint_configurations_json(
            hyperparameter_ranges=hyperparameter_ranges
        )
        traffic_pattern = self._convert_to_traffic_pattern_json(
            traffic_type=traffic_type, phases=phases
        )
        stopping_conditions = self._convert_to_stopping_conditions_json(
            max_invocations=max_invocations, model_latency_thresholds=model_latency_thresholds
        )
        resource_limit = self._convert_to_resource_limit_json(
            max_tests=max_tests, max_parallel_tests=max_parallel_tests
        )

        # Determine job type based on advanced parameters
        if endpoint_configurations or traffic_pattern or stopping_conditions or resource_limit:
            logger.info("Advanced job parameters specified. Running Advanced recommendation job...")
            job_type = "Advanced"
        else:
            logger.info("No advanced parameters specified. Running Default recommendation job...")
            job_type = "Default"

        # Initialize SageMaker session if needed (method from ModelBuilder mixin)
        if hasattr(self, '_init_sagemaker_session_if_does_not_exist'):
            self._init_sagemaker_session_if_does_not_exist()

        # Create inference recommendations job
        ret_name = self.sagemaker_session.create_inference_recommendations_job(
            role=getattr(self, 'role_arn', None),
            job_name=job_name,
            job_type=job_type,
            job_duration_in_seconds=job_duration_in_seconds,
            model_name=getattr(self, 'model_name', None),
            model_package_version_arn=getattr(self, "model_package_arn", None),
            framework=framework,
            framework_version=framework_version,
            sample_payload_url=sample_payload_url,
            supported_content_types=supported_content_types,
            supported_instance_types=supported_instance_types,
            endpoint_configurations=endpoint_configurations,
            traffic_pattern=traffic_pattern,
            stopping_conditions=stopping_conditions,
            resource_limit=resource_limit,
        )

        # Wait for job completion and store results
        self.inference_recommender_job_results = (
            self.sagemaker_session.wait_for_inference_recommendations_job(
                ret_name, log_level=log_level
            )
        )
        self.inference_recommendations = self.inference_recommender_job_results.get(
            "InferenceRecommendations", []
        )

        return self

    def _update_params(self, **kwargs) -> Optional[Tuple[str, int]]:
        """Update deployment parameters based on inference recommendations.
        
        Processes inference recommendation ID or right-size results to determine
        optimal instance type and count for model deployment.
        
        Args:
            **kwargs: Deployment parameters including instance_type, initial_instance_count,
                     inference_recommendation_id, etc.
                     
        Returns:
            Tuple of (instance_type, initial_instance_count) if recommendations found,
            otherwise None to use provided parameters.
        """
        instance_type = kwargs.get("instance_type")
        initial_instance_count = kwargs.get("initial_instance_count")
        accelerator_type = kwargs.get("accelerator_type")
        async_inference_config = kwargs.get("async_inference_config")
        serverless_inference_config = kwargs.get("serverless_inference_config")
        explainer_config = kwargs.get("explainer_config")
        inference_recommendation_id = kwargs.get("inference_recommendation_id")
        inference_recommender_job_results = kwargs.get("inference_recommender_job_results")
        
        inference_recommendation = None
        
        if inference_recommendation_id is not None:
            inference_recommendation = self._update_params_for_recommendation_id(
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                accelerator_type=accelerator_type,
                async_inference_config=async_inference_config,
                serverless_inference_config=serverless_inference_config,
                inference_recommendation_id=inference_recommendation_id,
                explainer_config=explainer_config,
            )
        elif inference_recommender_job_results is not None:
            inference_recommendation = self._update_params_for_right_size(
                instance_type,
                initial_instance_count,
                accelerator_type,
                serverless_inference_config,
                async_inference_config,
                explainer_config,
            )

        return (
            inference_recommendation
            if inference_recommendation
            else (instance_type, initial_instance_count)
        )

    def _update_params_for_right_size(
        self,
        instance_type: Optional[str] = None,
        initial_instance_count: Optional[int] = None,
        accelerator_type: Optional[str] = None,
        serverless_inference_config: Optional[Any] = None,
        async_inference_config: Optional[Any] = None,
        explainer_config: Optional[Any] = None,
    ) -> Optional[Tuple[str, int]]:
        """Validates that Inference Recommendation parameters can be used in `model.deploy()`

        Args:
            instance_type (str): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. If not using
                serverless inference or the model has not called ``right_size()``,
                then it need to be a number larger or equals
                to 1 (default: None)
            initial_instance_count (int):The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference or the model has not called ``right_size()``,
                then it is required to deploy a model.
                (default: None)
            accelerator_type (str): whether accelerator_type has been passed into `model.deploy()`.
            serverless_inference_config (sagemaker.serve.serverless.ServerlessInferenceConfig)):
                whether serverless_inference_config has been passed into `model.deploy()`.
            async_inference_config (sagemaker.model_monitor.AsyncInferenceConfig):
                whether async_inference_config has been passed into `model.deploy()`.
            explainer_config (sagemaker.explainer.ExplainerConfig): whether explainer_config
                has been passed into `model.deploy()`.

        Returns:
            (string, int) or None: Top instance_type and associated initial_instance_count
            if self.inference_recommender_job_results has been generated. Otherwise, return None.
        """
        if accelerator_type:
            raise ValueError("accelerator_type is not compatible with right_size().")
        if instance_type or initial_instance_count:
            logger.warning(
                "instance_type or initial_instance_count specified."
                "Overriding right_size() recommendations."
            )
            return None
        if async_inference_config:
            logger.warning(
                "async_inference_config is specified. Overriding right_size() recommendations."
            )
            return None
        if serverless_inference_config:
            logger.warning(
                "serverless_inference_config is specified. Overriding right_size() recommendations."
            )
            return None
        if explainer_config:
            logger.warning(
                "explainer_config is specified. Overriding right_size() recommendations."
            )
            return None

        return self._filter_recommendations_for_realtime()

    def _update_params_for_recommendation_id(
        self,
        instance_type: Optional[str],
        initial_instance_count: Optional[int],
        accelerator_type: Optional[str],
        async_inference_config: Optional[Any],
        serverless_inference_config: Optional[Any],
        inference_recommendation_id: str,
        explainer_config: Optional[Any],
    ) -> Tuple[str, int]:
        """Update parameters with inference recommendation results.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference, then it is required to deploy a model.
            initial_instance_count (int): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. If not using
                serverless inference, then it need to be a number larger or equals
                to 1.
            accelerator_type (str): Type of Elastic Inference accelerator to
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            async_inference_config (sagemaker.model_monitor.AsyncInferenceConfig): Specifies
                configuration related to async endpoint. Use this configuration when trying
                to create async endpoint and make async inference. If empty config object
                passed through, will use default config to deploy async endpoint. Deploy a
                real-time endpoint if it's None.
            serverless_inference_config (sagemaker.serve.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Use this configuration
                when trying to create serverless endpoint and make serverless inference. If
                empty object passed through, will use pre-defined values in
                ``ServerlessInferenceConfig`` class to deploy serverless endpoint. Deploy an
                instance based endpoint if it's None.
            inference_recommendation_id (str): The recommendation id which specifies
                the recommendation you picked from inference recommendation job
                results and would like to deploy the model and endpoint with
                recommended parameters.
            explainer_config (sagemaker.explainer.ExplainerConfig): Specifies online explainability
                configuration for use with Amazon SageMaker Clarify. Default: None.
        Raises:
            ValueError: If arguments combination check failed in these circumstances:
                - If only one of instance type or instance count specified or
                - If recommendation id does not follow the required format or
                - If recommendation id is not valid or
                - If inference recommendation id is specified along with incompatible parameters
        Returns:
            (string, int): instance type and associated instance count from selected
            inference recommendation id if arguments combination check passed.
        """

        if instance_type is not None and initial_instance_count is not None:
            logger.warning(
                "Both instance_type and initial_instance_count are specified,"
                "overriding the recommendation result."
            )
            return (instance_type, initial_instance_count)

        # Validate non-compatible parameters with recommendation id
        if accelerator_type is not None:
            raise ValueError("accelerator_type is not compatible with inference_recommendation_id.")
        if async_inference_config is not None:
            raise ValueError(
                "async_inference_config is not compatible with inference_recommendation_id."
            )
        if serverless_inference_config is not None:
            raise ValueError(
                "serverless_inference_config is not compatible with inference_recommendation_id."
            )
        if explainer_config is not None:
            raise ValueError("explainer_config is not compatible with inference_recommendation_id.")

        # Validate recommendation ID format
        if not re.match(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,63}\/\w{8}$", inference_recommendation_id):
            raise ValueError(
                f"Invalid inference_recommendation_id format: {inference_recommendation_id}. "
                f"Expected format: <job-or-model-name>/<8-character-id>"
            )
        job_or_model_name = inference_recommendation_id.split("/")[0]

        sage_client = self.sagemaker_session.sagemaker_client
        # Get recommendation from right size job and model
        (
            right_size_recommendation,
            model_recommendation,
            right_size_job_res,
        ) = self._get_recommendation(
            sage_client=sage_client,
            job_or_model_name=job_or_model_name,
            inference_recommendation_id=inference_recommendation_id,
        )

        # Update params based on model recommendation
        if model_recommendation:
            if initial_instance_count is None:
                raise ValueError(
                    "Must specify initial_instance_count when using model recommendation ID."
                )
            # Update environment variables if they exist
            env_vars = getattr(self, 'env_vars', {})
            env_vars.update(model_recommendation.get("Environment", {}))
            instance_type = model_recommendation["InstanceType"]
            return (instance_type, initial_instance_count)

        # Update params based on default inference recommendation
        if bool(instance_type) != bool(initial_instance_count):
            raise ValueError(
                "instance_type and initial_instance_count must both be specified together "
                "to override recommendation, or both omitted to use recommendation values."
            )
            
        input_config = right_size_job_res["InputConfig"]
        model_config = right_size_recommendation["ModelConfiguration"]
        envs = model_config.get("EnvironmentParameters")
        
        # Update environment variables from recommendation
        recommend_envs = {}
        if envs:
            for env in envs:
                recommend_envs[env["Key"]] = env["Value"]
                
        # Safely update env_vars
        current_env_vars = getattr(self, 'env_vars', {})
        current_env_vars.update(recommend_envs)

        # Update params with non-compilation recommendation results
        if (
            "InferenceSpecificationName" not in model_config
            and "CompilationJobName" not in model_config
        ):

            if "ModelPackageVersionArn" in input_config:
                modelpkg_res = sage_client.describe_model_package(
                    ModelPackageName=input_config["ModelPackageVersionArn"]
                )
                self.s3_model_data_url = modelpkg_res["InferenceSpecification"]["Containers"][0][
                    "ModelDataUrl"
                ]
                self.image_uri = modelpkg_res["InferenceSpecification"]["Containers"][0]["Image"]
            elif "ModelName" in input_config:
                model_res = sage_client.describe_model(ModelName=input_config["ModelName"])
                self.s3_model_data_url = model_res["PrimaryContainer"]["ModelDataUrl"]
                self.image_uri = model_res["PrimaryContainer"]["Image"]
        else:
            if "InferenceSpecificationName" in model_config:
                modelpkg_res = sage_client.describe_model_package(
                    ModelPackageName=input_config["ModelPackageVersionArn"]
                )
                self.s3_model_data_url = modelpkg_res["AdditionalInferenceSpecificationDefinition"][
                    "Containers"
                ][0]["ModelDataUrl"]
                self.image_uri = modelpkg_res["AdditionalInferenceSpecificationDefinition"][
                    "Containers"
                ][0]["Image"]
            elif "CompilationJobName" in model_config:
                compilation_res = sage_client.describe_compilation_job(
                    CompilationJobName=model_config["CompilationJobName"]
                )
                self.s3_model_data_url = compilation_res["ModelArtifacts"]["S3ModelArtifacts"]
                self.image_uri = compilation_res["InferenceImage"]

        instance_type = right_size_recommendation["EndpointConfiguration"]["InstanceType"]
        initial_instance_count = right_size_recommendation["EndpointConfiguration"][
            "InitialInstanceCount"
        ]

        return (instance_type, initial_instance_count)

    def _convert_to_endpoint_configurations_json(
        self, hyperparameter_ranges: Optional[List[Dict[str, CategoricalParameter]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert hyperparameter ranges to endpoint configurations for Advanced jobs.
        
        Args:
            hyperparameter_ranges: List of hyperparameter range dictionaries
            
        Returns:
            List of endpoint configuration dictionaries, or None if no ranges provided
            
        Raises:
            ValueError: If instance_types not specified in hyperparameter ranges
        """
        if not hyperparameter_ranges:
            return None

        endpoint_configurations_to_json = []
        for parameter_range in hyperparameter_ranges:
            if not parameter_range.get("instance_types"):
                raise ValueError(
                    "instance_types must be defined as a hyperparameter range for Advanced jobs"
                )
            parameter_range = parameter_range.copy()
            instance_types = parameter_range.get("instance_types").values
            parameter_range.pop("instance_types")

            for instance_type in instance_types:
                parameter_ranges = [
                    {"Name": name, "Value": param.values} for name, param in parameter_range.items()
                ]
                endpoint_configurations_to_json.append(
                    {
                        "EnvironmentParameterRanges": {
                            "CategoricalParameterRanges": parameter_ranges
                        },
                        "InstanceType": instance_type,
                    }
                )

        return endpoint_configurations_to_json

    def _convert_to_traffic_pattern_json(
        self, traffic_type: Optional[str], phases: Optional[List[Phase]]
    ) -> Optional[Dict[str, Any]]:
        """Convert traffic pattern parameters for Advanced jobs.
        
        Args:
            traffic_type: Type of traffic pattern (defaults to "PHASES")
            phases: List of Phase objects defining load test pattern
            
        Returns:
            Traffic pattern dictionary, or None if no phases provided
        """
        if not phases:
            return None
        return {
            "Phases": [phase.to_json for phase in phases],
            "TrafficType": traffic_type if traffic_type else "PHASES",
        }

    def _convert_to_resource_limit_json(
        self, max_tests: Optional[int], max_parallel_tests: Optional[int]
    ) -> Optional[Dict[str, int]]:
        """Convert resource limit parameters for Advanced jobs.
        
        Args:
            max_tests: Maximum number of tests to run
            max_parallel_tests: Maximum number of parallel tests
            
        Returns:
            Resource limit dictionary, or None if no limits specified
        """
        if not max_tests and not max_parallel_tests:
            return None
        resource_limit = {}
        if max_tests:
            resource_limit["MaxNumberOfTests"] = max_tests
        if max_parallel_tests:
            resource_limit["MaxParallelOfTests"] = max_parallel_tests
        return resource_limit

    def _convert_to_stopping_conditions_json(
        self, 
        max_invocations: Optional[int], 
        model_latency_thresholds: Optional[List[ModelLatencyThreshold]]
    ) -> Optional[Dict[str, Any]]:
        """Convert stopping condition parameters for Advanced jobs.
        
        Args:
            max_invocations: Maximum number of invocations per minute
            model_latency_thresholds: List of latency threshold requirements
            
        Returns:
            Stopping conditions dictionary, or None if no conditions specified
        """
        if not max_invocations and not model_latency_thresholds:
            return None
        stopping_conditions = {}
        if max_invocations:
            stopping_conditions["MaxInvocations"] = max_invocations
        if model_latency_thresholds:
            stopping_conditions["ModelLatencyThresholds"] = [
                threshold.to_json for threshold in model_latency_thresholds
            ]
        return stopping_conditions

    def _get_recommendation(
        self, sage_client: Any, job_or_model_name: str, inference_recommendation_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Retrieve recommendation from right-size job or model.
        
        Args:
            sage_client: SageMaker client for API calls
            job_or_model_name: Name of the job or model
            inference_recommendation_id: ID of the specific recommendation
            
        Returns:
            Tuple of (right_size_recommendation, model_recommendation, right_size_job_res)
            
        Raises:
            ValueError: If recommendation ID is not found in any source
        """
        right_size_recommendation, model_recommendation, right_size_job_res = None, None, None
        
        # Try to get recommendation from right-size job first
        right_size_recommendation, right_size_job_res = self._get_right_size_recommendation(
            sage_client=sage_client,
            job_or_model_name=job_or_model_name,
            inference_recommendation_id=inference_recommendation_id,
        )
        
        # If not found in job, try model recommendations
        if right_size_recommendation is None:
            model_recommendation = self._get_model_recommendation(
                sage_client=sage_client,
                job_or_model_name=job_or_model_name,
                inference_recommendation_id=inference_recommendation_id,
            )
            if model_recommendation is None:
                raise ValueError(
                    f"Recommendation ID '{inference_recommendation_id}' not found in "
                    f"job '{job_or_model_name}' or associated model recommendations"
                )

        return right_size_recommendation, model_recommendation, right_size_job_res

    def _get_right_size_recommendation(
        self,
        sage_client: Any,
        job_or_model_name: str,
        inference_recommendation_id: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Get recommendation from right-size job.
        
        Args:
            sage_client: SageMaker client
            job_or_model_name: Name of the inference recommendations job
            inference_recommendation_id: Specific recommendation ID to find
            
        Returns:
            Tuple of (recommendation, job_results) or (None, None) if not found
        """
        right_size_recommendation, right_size_job_res = None, None
        try:
            right_size_job_res = sage_client.describe_inference_recommendations_job(
                JobName=job_or_model_name
            )
            if right_size_job_res:
                right_size_recommendation = self._search_recommendation(
                    recommendation_list=right_size_job_res.get("InferenceRecommendations", []),
                    inference_recommendation_id=inference_recommendation_id,
                )
        except sage_client.exceptions.ResourceNotFound:
            pass

        return right_size_recommendation, right_size_job_res

    def _get_model_recommendation(
        self,
        sage_client: Any,
        job_or_model_name: str,
        inference_recommendation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get recommendation from model deployment recommendations.
        
        Args:
            sage_client: SageMaker client
            job_or_model_name: Name of the model
            inference_recommendation_id: Specific recommendation ID to find
            
        Returns:
            Model recommendation dictionary or None if not found
        """
        model_recommendation = None
        try:
            model_res = sage_client.describe_model(ModelName=job_or_model_name)
            if model_res:
                deployment_rec = model_res.get("DeploymentRecommendation", {})
                realtime_recs = deployment_rec.get("RealTimeInferenceRecommendations", [])
                model_recommendation = self._search_recommendation(
                    recommendation_list=realtime_recs,
                    inference_recommendation_id=inference_recommendation_id,
                )
        except sage_client.exceptions.ResourceNotFound:
            pass

        return model_recommendation

    def _search_recommendation(
        self, recommendation_list: List[Dict[str, Any]], inference_recommendation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Search for specific recommendation by ID.
        
        Args:
            recommendation_list: List of recommendation dictionaries
            inference_recommendation_id: ID to search for
            
        Returns:
            Matching recommendation dictionary or None if not found
        """
        return next(
            (
                rec
                for rec in recommendation_list
                if rec.get("RecommendationId") == inference_recommendation_id
            ),
            None,
        )

    def _filter_recommendations_for_realtime(self) -> Tuple[Optional[str], Optional[int]]:
        """Filter recommendations to find real-time (non-serverless) instance.
        
        Returns:
            Tuple of (instance_type, initial_instance_count) for first real-time
            recommendation found, or (None, None) if none found.
            
        Note:
            TODO: Integrate right_size + deploy with serverless support
        """
        instance_type = None
        initial_instance_count = None
        
        inference_recommendations = getattr(self, 'inference_recommendations', [])
        for recommendation in inference_recommendations:
            endpoint_config = recommendation.get("EndpointConfiguration", {})
            if "ServerlessConfig" not in endpoint_config:
                instance_type = endpoint_config.get("InstanceType")
                initial_instance_count = endpoint_config.get("InitialInstanceCount")
                break
                
        return (instance_type, initial_instance_count)
