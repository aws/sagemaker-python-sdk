"""SageMaker Evaluation Execution Module.

This module provides classes for managing evaluation executions.
"""
from __future__ import absolute_import

# Standard library imports
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

# Third-party imports
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field
from sagemaker.core.common_utils import TagsDict
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import Pipeline, PipelineExecution, Tag
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

# Local imports
from .constants import (
    _TAG_SAGEMAKER_MODEL_EVALUATION,
    EvalType,
    _get_pipeline_name,
    _get_pipeline_name_prefix,
)

logger = logging.getLogger(__name__)


def _create_evaluation_pipeline(
    eval_type: EvalType,
    role_arn: str,
    pipeline_definition: str,
    session: Optional[Any] = None,
    region: Optional[str] = None,
    tags: Optional[List[TagsDict]] = [],
) -> Any:
    """Helper method to create a SageMaker pipeline for evaluation.
    
    Re-renders pipeline_definition with actual pipeline_name before creating.
    
    Args:
        eval_type (EvalType): Type of evaluation.
        role_arn (str): IAM role ARN for pipeline execution.
        pipeline_definition (str): JSON pipeline definition (Jinja2 template).
        session (Optional[Any]): SageMaker session object.
        region (Optional[str]): AWS region.
        tags (Optional[List[TagsDict]]): List of tags to include in pipeline
        
    Returns:
        Any: Created Pipeline instance (ready for execution).
    """
    from jinja2 import Template
    
    pipeline_name = _get_pipeline_name(eval_type)
    client_request_token = str(uuid.uuid4())
    
    logger.info(f"Creating new pipeline: {pipeline_name}")
    
    # Re-render pipeline definition with actual pipeline_name
    template = Template(pipeline_definition)
    resolved_pipeline_definition = template.render(pipeline_name=pipeline_name)
    
    # Create tags for the pipeline
    tags.extend([
        {"key": _TAG_SAGEMAKER_MODEL_EVALUATION, "value": "true"}
    ])
    
    pipeline = Pipeline.create(
        pipeline_name=pipeline_name,
        client_request_token=client_request_token,
        role_arn=role_arn,
        pipeline_definition=resolved_pipeline_definition,
        pipeline_display_name=f"EvaluationPipeline-{eval_type.value}",
        pipeline_description=f"Pipeline for {eval_type.value} evaluation jobs",
        tags=tags,
        session=session,
        region=region
    )
    
    logger.info(f"Successfully created pipeline: {pipeline_name}")
    
    # Wait for pipeline to be ready before returning
    logger.info(f"Waiting for pipeline {pipeline_name} to be ready...")
    try:
        pipeline.wait_for_status(target_status="Active", poll=5, timeout=300)  # Wait up to 5 minutes
        logger.info(f"Pipeline {pipeline_name} is now active and ready for execution")
    except Exception as e:
        logger.warning(f"Failed to wait for pipeline status: {e}. Pipeline may still be initializing...")
    
    return pipeline


def _clean_unassigned_value(value: Any) -> Any:
    """Clean Unassigned object by converting to None.
    
    Args:
        value (Any): Value that may be an Unassigned object.
        
    Returns:
        Any: None if value is Unassigned, otherwise returns the value unchanged.
    """
    if value is not None and hasattr(value, '__class__'):
        if 'Unassigned' in value.__class__.__name__:
            return None
    return value


def _clean_unassigned_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean Unassigned objects from nested dict before pydantic validation.
    
    Args:
        data (Dict[str, Any]): Dictionary that may contain Unassigned objects.
        
    Returns:
        Dict[str, Any]: Cleaned dictionary with Unassigned objects replaced with None.
    """
    if data.get('status', {}).get('failure_reason') is not None:
        data['status']['failure_reason'] = _clean_unassigned_value(data['status']['failure_reason'])
    return data


def _extract_eval_type_from_arn(arn: str) -> Optional[EvalType]:
    """Helper method to extract evaluation type from pipeline or execution ARN.
    
    Extracts eval type from new naming pattern: SagemakerEvaluation-[EvalType]-[uuid]
    
    Args:
        arn (str): Pipeline ARN or Pipeline Execution ARN.
            Pipeline ARN format: arn:aws:sagemaker:region:account:pipeline/pipeline-name
            Execution ARN format: arn:aws:sagemaker:region:account:pipeline/pipeline-name/execution/execution-id
    
    Returns:
        Optional[EvalType]: EvalType if found, None otherwise.
    """
    try:
        # Split ARN and extract pipeline name
        arn_parts = arn.split('/')
        if len(arn_parts) >= 2:
            # For execution ARN, pipeline name is at index -3
            # For pipeline ARN, pipeline name is at index -1
            pipeline_name = arn_parts[-3] if len(arn_parts) >= 4 else arn_parts[-1]
            
            # Check pattern: SagemakerEvaluation-{EvalType}-{uuid}
            for eval_type in EvalType:
                prefix = _get_pipeline_name_prefix(eval_type)
                if pipeline_name.startswith(prefix):
                    logger.debug(f"Extracted eval_type: {eval_type.value} from ARN: {arn}")
                    return eval_type
        
        logger.warning(f"Could not extract eval_type from ARN: {arn}")
        return None
    except Exception as e:
        logger.warning(f"Error extracting eval_type from ARN {arn}: {str(e)}")
        return None


def _get_or_create_pipeline(
    eval_type: EvalType,
    pipeline_definition: str,
    role_arn: str,
    session: Optional[Session] = None,
    region: Optional[str] = None,
    create_tags: Optional[List[TagsDict]] = [],
) -> Pipeline:
    """Get existing pipeline or create/update it.
    
    Searches for existing pipeline using Pipeline.get_all with pipeline_name_prefix.
    Validates tag using Tag.get_all and updates if found. Otherwise creates new pipeline with UUID.
    Re-renders pipeline_definition with actual pipeline_name before create/update.
    
    Args:
        eval_type: Type of evaluation
        pipeline_definition: JSON pipeline definition (Jinja2 template)
        role_arn: IAM role ARN for pipeline execution
        session: Boto3 session (optional)
        region: AWS region (optional)
        create_tags (Optional[List[TagsDict]]): List of tags to include in pipeline
        
    Returns:
        Pipeline instance (existing updated or newly created)
        
    Raises:
        ClientError: If AWS service call fails
    """
    from jinja2 import Template
    
    pipeline_name_prefix = _get_pipeline_name_prefix(eval_type)
    
    try:
        # Use Pipeline.get_all with pipeline_name_prefix to find existing pipelines
        pipelines = Pipeline.get_all(
            pipeline_name_prefix=pipeline_name_prefix,
            session=session,
            region=region
        )
        
        # Check each pipeline for the required tag
        for pipeline in pipelines:
            pipeline_arn = pipeline.pipeline_arn
            
            # Get tags using Tag.get_all
            tags_list = Tag.get_all(resource_arn=pipeline_arn, session=session, region=region)
            tags = {tag.key: tag.value for tag in tags_list}
            
            # Validate tag
            if tags.get(_TAG_SAGEMAKER_MODEL_EVALUATION) == "true":
                pipeline_name = pipeline.pipeline_name
                logger.info(f"Found existing pipeline: {pipeline_name}")
                
                # Re-render pipeline definition with actual pipeline_name
                template = Template(pipeline_definition)
                resolved_pipeline_definition = template.render(pipeline_name=pipeline_name)
                
                # Update pipeline with latest definition
                logger.info(f"Updating pipeline {pipeline_name} with latest definition")
                pipeline.update(
                    pipeline_definition=resolved_pipeline_definition,
                    role_arn=role_arn,
                    pipeline_description=f"Pipeline for {eval_type.value} evaluation jobs (updated)"
                )
                logger.info(f"Successfully updated pipeline: {pipeline_name}")
                return pipeline
        
        # No matching pipeline found, create new one
        logger.info(f"No existing pipeline found with prefix {pipeline_name_prefix}, creating new one")
        return _create_evaluation_pipeline(eval_type, role_arn, pipeline_definition, session, region, create_tags)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if "ResourceNotFound" in error_code:
            return _create_evaluation_pipeline(eval_type, role_arn, pipeline_definition, session, region, create_tags)
        else:
            raise
            
    except Exception as e:
        # If search fails for other reasons, try to create
        logger.info(f"Error searching for pipeline ({str(e)}), attempting to create new pipeline")
        return _create_evaluation_pipeline(eval_type, role_arn, pipeline_definition, session, region, create_tags)


def _start_pipeline_execution(
    pipeline_name: str,
    name: str,
    session: Optional[Session] = None,
    region: Optional[str] = None
) -> str:
    """Start pipeline execution using boto3 client.
    
    Extracted for testability - can be mocked independently in tests.
    
    Args:
        pipeline_name: Name of the pipeline to execute
        name: Base name for the execution
        session: Boto3 session (optional)
        region: AWS region (optional)
        
    Returns:
        ARN of the started pipeline execution
        
    Raises:
        ClientError: If AWS service call fails
    """
    import os
    import boto3
    
    execution_display_name = f"{name}-{int(time.time())}"
    endpoint_url = os.environ.get('SAGEMAKER_ENDPOINT')
    
    # Get boto3 client
    if session:
        sm_client = session.client('sagemaker', region_name=region, endpoint_url=endpoint_url)
    else:
        sm_client = boto3.client('sagemaker', region_name=region, endpoint_url=endpoint_url)
    
    # Start pipeline execution
    logger.info(f"Starting pipeline execution: {execution_display_name}")
    
    response = sm_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=execution_display_name,
        PipelineExecutionDescription=f"Evaluation execution: {name}",
        PipelineParameters=[],  # Empty since all values are pre-substituted
        ClientRequestToken=str(uuid.uuid4())
    )
    
    execution_arn = response['PipelineExecutionArn']
    logger.info(f"Pipeline execution started: {execution_arn}")
    
    return execution_arn


def _create_execution_from_pipeline_execution(
    pe: PipelineExecution,
    eval_type: EvalType
) -> 'EvaluationPipelineExecution':
    """Create EvaluationPipelineExecution from PipelineExecution.
    
    Handles failure_reason Unassigned objects and sets basic properties.
    Extracted for testability - used by both get() and get_all().
    
    Args:
        pe: PipelineExecution object from sagemaker_core
        eval_type: Type of evaluation
        
    Returns:
        EvaluationPipelineExecution with basic properties set
    """
    name = pe.pipeline_execution_arn.split('/')[-1] if pe.pipeline_execution_arn else 'unknown'
    
    # Handle failure_reason which might be an Unassigned object
    failure_reason = pe.failure_reason
    if failure_reason is not None and hasattr(failure_reason, '__class__'):
        if 'Unassigned' in failure_reason.__class__.__name__:
            failure_reason = None
    
    execution = EvaluationPipelineExecution(
        arn=pe.pipeline_execution_arn,
        name=name,
        status=PipelineExecutionStatus(
            overall_status=pe.pipeline_execution_status or 'Unknown',
            failure_reason=failure_reason
        ),
        last_modified_time=pe.last_modified_time,
        eval_type=eval_type
    )
    
    # Store the internal pipeline execution reference
    execution._pipeline_execution = pe
    
    return execution


def _extract_output_s3_location_from_steps(raw_steps: List[Any], session: Optional[Any] = None, region: Optional[str] = None) -> Optional[str]:
    """Helper method to extract S3 output location from training job's OutputDataConfig.
    
    Finds the first evaluation training step (EvaluateCustomModel or EvaluateBaseModel),
    gets its training job ARN, and uses boto3 DescribeTrainingJob to retrieve the S3 output path.
    
    Args:
        raw_steps: List of PipelineExecutionStep objects from SageMaker
        session: Boto3 session (optional)
        region: AWS region (optional)
    
    Returns:
        S3 output location from OutputDataConfig if found, None otherwise
    """
    try:
        import boto3
        import os
        
        # Get endpoint URL from environment variable (for beta endpoint support)
        endpoint_url = os.environ.get('SAGEMAKER_ENDPOINT')
        
        # Get SageMaker client with optional endpoint URL
        if session:
            sm_client = session.client('sagemaker', region_name=region, endpoint_url=endpoint_url)
        else:
            sm_client = boto3.client('sagemaker', region_name=region, endpoint_url=endpoint_url)
        
        for step in raw_steps:
            step_name = getattr(step, 'step_name', '')
            
            # Look for evaluation training steps (custom or base)
            if 'EvaluateCustomModel' in step_name or 'EvaluateBaseModel' in step_name:
                metadata = getattr(step, 'metadata', None)
                if metadata and hasattr(metadata, 'training_job'):
                    training_job_meta = metadata.training_job
                    
                    # Get training job name from ARN
                    if hasattr(training_job_meta, 'arn'):
                        training_job_name = training_job_meta.arn.split('/')[-1]
                        
                        try:
                            # Use boto3 DescribeTrainingJob (avoids pydantic validation issues)
                            response = sm_client.describe_training_job(TrainingJobName=training_job_name)
                            
                            # Get OutputDataConfig.S3OutputPath
                            if 'OutputDataConfig' in response and 'S3OutputPath' in response['OutputDataConfig']:
                                s3_output_path = response['OutputDataConfig']['S3OutputPath']
                                logger.info(f"Extracted s3_output_path from training job {training_job_name}: {s3_output_path}")
                                return s3_output_path
                        except ClientError as e:
                            logger.warning(f"Failed to describe training job {training_job_name}: {e}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error describing training job {training_job_name}: {e}")
                            continue
        
        logger.debug("Could not extract s3_output_path from pipeline steps")
        return None
    except Exception as e:
        logger.warning(f"Error extracting s3_output_path from steps: {str(e)}")
        return None


class StepDetail(BaseModel):
    """Pipeline step details for tracking execution progress.

    Represents the status and timing information for a single step
    in a SageMaker pipeline execution.

    Parameters:
        name (str): Name of the pipeline step.
        status (str): Status of the step (Completed, Executing, Waiting, Failed).
        start_time (Optional[str]): ISO format timestamp when step started.
        end_time (Optional[str]): ISO format timestamp when step ended.
        display_name (Optional[str]): Human-readable display name for the step.
        failure_reason (Optional[str]): Detailed reason if the step failed.
    """

    name: str = Field(..., description="Name of the pipeline step")
    status: str = Field(..., description="Status of the step (Completed, Executing, Waiting, Failed)")
    start_time: Optional[str] = Field(None, description="Step start time")
    end_time: Optional[str] = Field(None, description="Step end time")
    display_name: Optional[str] = Field(None, description="Display name for the step")
    failure_reason: Optional[str] = Field(None, description="Reason for failure if step failed")


class PipelineExecutionStatus(BaseModel):
    """Combined pipeline execution status with step details and failure reason.

    Aggregates the overall execution status along with detailed information
    about individual pipeline steps and any failure reasons.

    Parameters:
        overall_status (str): Overall execution status (Starting, Executing, Completed, Failed, etc.).
        step_details (List[StepDetail]): List of individual pipeline step details.
        failure_reason (Optional[str]): Detailed reason if the execution failed.
    """

    overall_status: str = Field(..., description="Overall execution status (Starting, Running, Completed, Failed, etc.)")
    step_details: List[StepDetail] = Field(default_factory=list, description="List of pipeline step details")
    failure_reason: Optional[str] = Field(None, description="Reason for failure if execution failed")


class EvaluationPipelineExecution(BaseModel):
    """Manages SageMaker pipeline-based evaluation execution lifecycle.

    This class wraps SageMaker Pipeline execution to provide a simplified
    interface for running, monitoring, and managing evaluation jobs. Users
    typically don't instantiate this class directly, but receive instances
    from evaluator classes.

    Example:
        .. code:: python

            from sagemaker.train.evaluate import BenchmarkEvaluator
            from sagemaker.train.evaluate.execution import EvaluationPipelineExecution

            # Start evaluation through evaluator
            evaluator = BenchmarkEvaluator(...)
            execution = evaluator.evaluate()

            # Monitor execution
            print(f"Status: {execution.status.overall_status}")
            print(f"Steps: {len(execution.status.step_details)}")

            # Wait for completion
            execution.wait()

            # Display results
            execution.show_results()

            # Retrieve past executions
            all_executions = list(EvaluationPipelineExecution.get_all())
            specific_execution = EvaluationPipelineExecution.get(arn="arn:...")

    Parameters:
        arn (Optional[str]): ARN of the pipeline execution.
        name (str): Name of the evaluation execution.
        status (PipelineExecutionStatus): Combined status with step details and failure reason.
        last_modified_time (Optional[datetime]): Last modification timestamp.
        eval_type (Optional[EvalType]): Type of evaluation (BENCHMARK, CUSTOM_SCORER, LLM_AS_JUDGE).
        s3_output_path (Optional[str]): S3 location where evaluation results are stored.
        steps (List[Dict[str, Any]]): Raw step information from SageMaker.
    """
    
    # Fields set by underlying SageMaker pipeline operations
    arn: Optional[str] = Field(None, description="ARN of the pipeline execution")
    name: str = Field(..., description="Name of the evaluation execution")
    status: PipelineExecutionStatus = Field(default_factory=lambda: PipelineExecutionStatus(overall_status="Unknown"), description="Combined status, step details, and failure reason")
    last_modified_time: Optional[datetime] = Field(None, description="Last modification timestamp")
    eval_type: Optional[EvalType] = Field(None, description="Evaluation type")
    s3_output_path: Optional[str] = Field(None, description="S3 location where evaluation results are stored")
    
    # Additional fields for internal use
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Raw step information from SageMaker")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._pipeline_execution: Optional[Any] = None
    
    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="EvaluationPipelineExecution.start")
    def start(
        cls,
        eval_type: EvalType,
        name: str,
        pipeline_definition: str,
        role_arn: str,
        s3_output_path: Optional[str] = None,
        session: Optional[Session] = None,
        region: Optional[str] = None,
        tags: Optional[List[TagsDict]] = [],
    ) -> 'EvaluationPipelineExecution':
        """Create sagemaker pipeline execution. Optionally creates pipeline.
        
        Args:
            eval_type (EvalType): Type of evaluation (BENCHMARK, CUSTOM_SCORER, LLM_AS_JUDGE).
            name (str): Name for the evaluation execution.
            pipeline_definition (str): Complete rendered pipeline definition as JSON string.
            role_arn (str): IAM role ARN for pipeline execution.
            s3_output_path (Optional[str]): S3 location where evaluation results are stored.
            session (Optional[Session]): Boto3 session for API calls.
            region (Optional[str]): AWS region for the pipeline.
            tags (Optional[List[TagsDict]]): List of tags to include in pipeline
            
        Returns:
            EvaluationPipelineExecution: Started pipeline execution instance.
            
        Raises:
            ValueError: If pipeline_definition is not valid JSON.
            ClientError: If AWS service call fails.
        """
        # Validate pipeline_definition is valid JSON
        import json
        try:
            json.loads(pipeline_definition)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid pipeline definition JSON: {e}")
        
        # Create execution instance
        execution = cls(
            name=name,
            eval_type=eval_type,
            status=PipelineExecutionStatus(overall_status="Starting"),
            s3_output_path=s3_output_path
        )
        
        try:
            # Get or create pipeline (handles update logic internally)
            pipeline = _get_or_create_pipeline(
                eval_type=eval_type,
                pipeline_definition=pipeline_definition,
                role_arn=role_arn,
                session=session,
                region=region,
                create_tags=tags,
            )
            
            # Start pipeline execution via boto3
            # Use the actual pipeline name from the created/updated pipeline object
            pipeline_name = pipeline.pipeline_name
            execution.arn = _start_pipeline_execution(
                pipeline_name=pipeline_name,
                name=name,
                session=session,
                region=region
            )
            
            # Get the pipeline execution object for future operations
            execution._pipeline_execution = PipelineExecution.get(
                pipeline_execution_arn=execution.arn,
                session=session,
                region=region
            )
            
            # Update execution with initial execution details
            execution.status.overall_status = execution._pipeline_execution.pipeline_execution_status or "Executing"
            execution.last_modified_time = execution._pipeline_execution.creation_time or datetime.now()
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS service error when starting pipeline execution: {error_message}")
            execution.status.overall_status = "Failed"
            execution.status.failure_reason = f"AWS service error: {error_message}"
        except Exception as e:
            logger.error(f"Unexpected error when starting pipeline execution: {str(e)}")
            execution.status.overall_status = "Failed"
            execution.status.failure_reason = f"Unexpected error: {str(e)}"
        
        # Convert to appropriate subclass based on eval_type
        return execution._convert_to_subclass(eval_type)
    
    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="EvaluationPipelineExecution.get_all")
    def get_all(
        cls,
        eval_type: Optional[EvalType] = None,
        session: Optional[Session] = None,
        region: Optional[str] = None
    ):
        """Get all pipeline executions, optionally filtered by evaluation type.
        
        Searches for existing pipelines using prefix and tag validation,
        then retrieves executions from those pipelines.
        
        Args:
            eval_type (Optional[EvalType]): Evaluation type to filter by (e.g., EvalType.BENCHMARK).
                If None, returns executions from all evaluation pipelines.
            session (Optional[Session]): Boto3 session. Will be inferred if not provided.
            region (Optional[str]): AWS region. Will be inferred if not provided.
            
        Yields:
            EvaluationPipelineExecution: Pipeline execution instances.
            
        Example:
            .. code:: python
            
                # Get all evaluation executions as iterator
                iter = EvaluationPipelineExecution.get_all()
                all_executions = list(iter)
                
                # Get only benchmark evaluations
                iter = EvaluationPipelineExecution.get_all(eval_type=EvalType.BENCHMARK)
                benchmark_executions = list(iter)
        """
        
        try:
            # Determine which eval type(s) to search for
            eval_types_to_check = [eval_type] if eval_type else list(EvalType)
            
            for et in eval_types_to_check:
                pipeline_name_prefix = _get_pipeline_name_prefix(et)
                
                try:
                    # Search for pipelines with the prefix
                    pipelines = Pipeline.get_all(
                        pipeline_name_prefix=pipeline_name_prefix,
                        session=session,
                        region=region
                    )
                    
                    # Check each pipeline for the required tag and get its executions
                    for pipeline in pipelines:
                        try:
                            pipeline_arn = pipeline.pipeline_arn
                            
                            # Get tags using Tag.get_all
                            tags_list = Tag.get_all(resource_arn=pipeline_arn, session=session, region=region)
                            tags = {tag.key: tag.value for tag in tags_list}
                            
                            # Validate tag - only process evaluation pipelines
                            if tags.get(_TAG_SAGEMAKER_MODEL_EVALUATION) != "true":
                                logger.debug(f"Skipping pipeline {pipeline.pipeline_name} - missing required tag")
                                continue
                            
                            pipeline_name = pipeline.pipeline_name
                            logger.debug(f"Found evaluation pipeline: {pipeline_name}")
                            
                            # Get all executions for this pipeline
                            pipeline_executions = PipelineExecution.get_all(
                                pipeline_name=pipeline_name,
                                session=session,
                                region=region
                            )
                            
                            # Convert each PipelineExecution to EvaluationPipelineExecution
                            for pe in pipeline_executions:
                                # Create execution from pipeline execution
                                execution = _create_execution_from_pipeline_execution(pe, et)
                                
                                # Enrich with step details and S3 path
                                execution._enrich_with_step_details(session, region)
                                
                                # Convert to appropriate subclass based on eval_type
                                execution = execution._convert_to_subclass(et)
                                
                                yield execution
                        
                        except Exception as e:
                            logger.warning(f"Error processing pipeline {pipeline.pipeline_name}: {str(e)}")
                            continue
                
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    # If no pipelines found with prefix, skip to next eval type
                    if "ResourceNotFound" in error_code or "ValidationException" in error_code:
                        logger.debug(f"No pipelines found with prefix {pipeline_name_prefix}")
                        continue
                    else:
                        logger.warning(f"Error searching for pipelines with prefix {pipeline_name_prefix}: {e}")
                        continue
                except Exception as e:
                    logger.warning(f"Error processing eval type {et.value}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Unexpected error when listing pipeline executions: {str(e)}")
    
    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="EvaluationPipelineExecution.get")
    def get(
        cls,
        arn: str,
        session: Optional[Session] = None,
        region: Optional[str] = None
    ) -> 'EvaluationPipelineExecution':
        """Get a sagemaker pipeline execution instance by ARN.
        
        Args:
            arn (str): ARN of the pipeline execution.
            session (Optional[Session]): Boto3 session. Will be inferred if not provided.
            region (Optional[str]): AWS region. Will be inferred if not provided.
            
        Returns:
            EvaluationPipelineExecution: Retrieved pipeline execution instance.
            
        Raises:
            ClientError: If AWS service call fails.
            
        Example:
            .. code:: python
            
                # Get execution by ARN
                arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/eval-pipeline/execution/abc123"
                execution = EvaluationPipelineExecution.get(arn=arn)
                print(execution.status.overall_status)
        """
        # Create execution instance with basic info
        name = arn.split('/')[-1]
        execution = cls(
            arn=arn,
            name=name,
            status=PipelineExecutionStatus(overall_status="Unknown")
        )
        
        # Try to determine eval_type from execution ARN early (as fallback for error cases)
        execution.eval_type = _extract_eval_type_from_arn(arn)
        
        try:
            # Get pipeline execution details and store internally
            execution._pipeline_execution = PipelineExecution.get(
                pipeline_execution_arn=arn,
                session=session,
                region=region
            )
            
            # Update execution with pipeline execution details
            execution.status.overall_status = execution._pipeline_execution.pipeline_execution_status or "Unknown"
            execution.status.failure_reason = _clean_unassigned_value(execution._pipeline_execution.failure_reason)
            execution.last_modified_time = execution._pipeline_execution.last_modified_time
            
            # Enrich with step details and S3 path
            execution._enrich_with_step_details(session, region)
            
            # Determine eval_type from pipeline ARN (preferred method)
            pipeline_arn = execution._pipeline_execution.pipeline_arn if execution._pipeline_execution else None
            determined_eval_type = execution._determine_eval_type(pipeline_arn)
            if determined_eval_type:
                execution.eval_type = determined_eval_type
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS service error when getting pipeline execution: {error_message}")
            execution.status.overall_status = "Error"
            execution.status.failure_reason = f"AWS service error: {error_code}:{error_message}"
            # eval_type already set from execution ARN fallback above
        except Exception as e:
            logger.error(f"Unexpected error when getting pipeline execution details: {str(e)}")
            execution.status.overall_status = "Error"
            execution.status.failure_reason = f"Unexpected error: {str(e)}"
            # eval_type already set from execution ARN fallback above
        
        # Convert to appropriate subclass based on eval_type
        return execution._convert_to_subclass(execution.eval_type)
    
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="EvaluationPipelineExecution.refresh")
    def refresh(self) -> None:
        """Describe a pipeline execution and update job status"""
        if not self._pipeline_execution:
            return
        
        try:
            # Refresh the pipeline execution instance
            self._pipeline_execution.refresh()
            
            # Update status from refreshed pipeline execution
            self.status.overall_status = self._pipeline_execution.pipeline_execution_status or "Unknown"
            self.status.failure_reason = _clean_unassigned_value(self._pipeline_execution.failure_reason)
            self.last_modified_time = self._pipeline_execution.last_modified_time
            
            # Get updated pipeline execution steps with proper session/region handling
            steps_iterator = self._pipeline_execution.get_all_steps()
            raw_steps = list(steps_iterator)
            self._update_step_details_from_raw_steps(raw_steps)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS service error when refreshing pipeline execution: {error_message}")
        except Exception as e:
            logger.error(f"Unexpected error when refreshing pipeline execution: {str(e)}")
    
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="EvaluationPipelineExecution.stop")
    def stop(self) -> None:
        """Stop a pipeline execution"""
        if not self.arn:
            return
        
        try:
            # TODO: Move to sagemaker_core PipelineExecution.stop() when session handling is fixed
            # For now, use boto3 directly to stop the pipeline execution
            import os
            import boto3
            
            endpoint_url = os.environ.get('SAGEMAKER_ENDPOINT')
            
            # Get boto3 client - extract from pipeline execution if available
            if self._pipeline_execution and hasattr(self._pipeline_execution, '_session'):
                session = self._pipeline_execution._session
                if hasattr(session, 'boto_session'):
                    sm_client = session.boto_session.client('sagemaker', endpoint_url=endpoint_url)
                else:
                    sm_client = session.client('sagemaker', endpoint_url=endpoint_url)
            else:
                # Fallback to default boto3 client
                sm_client = boto3.client('sagemaker', endpoint_url=endpoint_url)
            
            # Stop the pipeline execution using boto3
            sm_client.stop_pipeline_execution(
                PipelineExecutionArn=self.arn
            )
            
            # Update status
            self.status.overall_status = "Stopping"
            logger.info(f"Stopping pipeline execution: {self.arn}")
            
            # Refresh to get updated status
            self.refresh()
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS service error when stopping pipeline execution: {error_message}")
        except Exception as e:
            logger.error(f"Unexpected error when stopping pipeline execution: {str(e)}")
    
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="EvaluationPipelineExecution.wait")
    def wait(
        self, 
        target_status: Literal["Executing", "Stopping", "Stopped", "Failed", "Succeeded"] = "Succeeded", 
        poll: int = 5, 
        timeout: Optional[int] = None
    ) -> None:
        """Wait for a pipeline execution to reach certain status.
        
        This method provides a hybrid implementation that works in both Jupyter notebooks
        and terminal environments, with appropriate visual feedback for each.
        
        Args:
            target_status: The status to wait for
            poll: The number of seconds to wait between each poll
            timeout: The maximum number of seconds to wait before timing out
        """
        if not self._pipeline_execution:
            return
        
        start_time = time.time()
        
        # Detect if running in Jupyter
        is_jupyter = False
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None and 'IPKernelApp' in ipython.config:
                is_jupyter = True
                from IPython.display import display, HTML, clear_output
        except:
            pass
        
        if is_jupyter:
            # Jupyter notebook experience with rich library
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.text import Text
            from rich.layout import Layout
            
            # Create console with Jupyter support
            console = Console(force_jupyter=True)
            
            while True:
                clear_output(wait=True)
                self.refresh()
                current_status = self.status.overall_status
                elapsed = time.time() - start_time
                
                # Create main status table
                status_table = Table(show_header=False, box=None, padding=(0, 1))
                status_table.add_column("Property", style="cyan bold", width=20)
                status_table.add_column("Value", style="white")
                
                status_table.add_row("Overall Status", f"[bold]{current_status}[/bold]")
                status_table.add_row("Target Status", f"[bold]{target_status}[/bold]")
                status_table.add_row("Elapsed Time", f"{elapsed:.1f}s")
                
                if self.status.failure_reason:
                    status_table.add_row("Failure Reason", f"[red]{self.status.failure_reason}[/red]")
                
                # Create steps table if steps exist
                if self.status.step_details:
                    # Check if any step has a failure
                    has_failures = any(step.failure_reason for step in self.status.step_details)
                    
                    steps_table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
                    steps_table.add_column("Step Name", style="cyan", width=30)
                    steps_table.add_column("Status", style="yellow", width=15)
                    steps_table.add_column("Duration", style="green", width=12)
                    
                    failed_steps = []  # Track steps with failures for detailed display
                    
                    for step in self.status.step_details:
                        # Calculate duration if both times are available
                        duration = ""
                        if step.start_time and step.end_time:
                            try:
                                from datetime import datetime
                                start = datetime.fromisoformat(step.start_time.replace('Z', '+00:00'))
                                end = datetime.fromisoformat(step.end_time.replace('Z', '+00:00'))
                                duration_seconds = (end - start).total_seconds()
                                duration = f"{duration_seconds:.1f}s"
                            except:
                                duration = "N/A"
                        elif step.start_time:
                            duration = "Running..."
                        
                        # Color code status
                        status_display = step.status
                        if "succeeded" in step.status.lower() or "completed" in step.status.lower():
                            status_display = f"[green]{step.status}[/green]"
                        elif "failed" in step.status.lower():
                            status_display = f"[red]{step.status}[/red]"
                        elif "executing" in step.status.lower() or "running" in step.status.lower():
                            status_display = f"[yellow]{step.status}[/yellow]"
                        
                        # Build row data
                        row_data = [
                            step.display_name or step.name,
                            status_display,
                            duration
                        ]
                        
                        # Add error indicator if failures exist
                        if has_failures:
                            if step.failure_reason:
                                row_data.append("❌")
                                failed_steps.append(step)
                            else:
                                row_data.append("")
                        
                        steps_table.add_row(*row_data)
                    
                    # Build combined content
                    from rich.console import Group
                    content_parts = [
                        status_table,
                        Text(""),  # Empty line for spacing
                        Text("Pipeline Steps", style="bold magenta"),
                        steps_table
                    ]
                    
                    # Add failure details section if there are any failures
                    if failed_steps:
                        content_parts.append(Text(""))  # Empty line
                        content_parts.append(Text("Step Failure Details", style="bold red"))
                        
                        for step in failed_steps:
                            content_parts.append(Text(""))  # Empty line before each failure
                            content_parts.append(Text(f"• {step.display_name or step.name}:", style="bold red"))
                            content_parts.append(Text(f"  {step.failure_reason}", style="red"))
                    
                    combined_content = Group(*content_parts)
                    
                    # Display combined content in a single panel
                    console.print(Panel(
                        combined_content,
                        title="[bold blue]Pipeline Execution Status[/bold blue]",
                        border_style="blue"
                    ))
                else:
                    # Display only status table if no steps
                    console.print(Panel(
                        status_table,
                        title="[bold blue]Pipeline Execution Status[/bold blue]",
                        border_style="blue"
                    ))
                
                if target_status == current_status:
                    logger.info(f"Final Resource Status: {current_status}")
                    return
                
                if "failed" in current_status.lower():
                    from sagemaker.core.utils.exceptions import FailedStatusError
                    raise FailedStatusError(
                        resource_type="PipelineExecution",
                        status=current_status,
                        reason=self.status.failure_reason,
                    )
                
                if timeout is not None and time.time() - start_time >= timeout:
                    from sagemaker.core.utils.exceptions import TimeoutExceededError
                    raise TimeoutExceededError(
                        resource_type="PipelineExecution", 
                        status=current_status
                    )
                
                time.sleep(poll)
        else:
            # Terminal experience with rich library
            try:
                from rich.live import Live
                from rich.panel import Panel
                from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
                from rich.console import Group
                from rich.status import Status
                from rich.style import Style
                
                progress = Progress(
                    SpinnerColumn("bouncingBar"),
                    TextColumn("{task.description}"),
                    TimeElapsedColumn(),
                )
                progress.add_task(f"Waiting for PipelineExecution to reach [bold]{target_status}[/bold] status...")
                status = Status("Current status:")
                
                with Live(
                    Panel(
                        Group(progress, status),
                        title="Wait Log Panel",
                        border_style=Style(color="blue"),
                    ),
                    transient=True,
                ):
                    while True:
                        self.refresh()
                        current_status = self.status.overall_status
                        status.update(f"Current status: [bold]{current_status}[/bold]")
                        
                        if target_status == current_status:
                            logger.info(f"Final Resource Status: [bold]{current_status}[/bold]")
                            return
                        
                        if "failed" in current_status.lower():
                            from sagemaker.core.utils.exceptions import FailedStatusError
                            raise FailedStatusError(
                                resource_type="PipelineExecution",
                                status=current_status,
                                reason=self.status.failure_reason,
                            )
                        
                        if timeout is not None and time.time() - start_time >= timeout:
                            from sagemaker.core.utils.exceptions import TimeoutExceededError
                            raise TimeoutExceededError(
                                resource_type="PipelineExecution",
                                status=current_status
                            )
                        
                        time.sleep(poll)
            except ImportError:
                # Fallback to simple print-based progress if rich is not available
                logger.info(f"Waiting for PipelineExecution to reach {target_status} status...")
                while True:
                    self.refresh()
                    current_status = self.status.overall_status
                    elapsed = time.time() - start_time
                    print(f"Current status: {current_status} (Elapsed: {elapsed:.1f}s)")
                    
                    if target_status == current_status:
                        logger.info(f"Final Resource Status: {current_status}")
                        return
                    
                    if "failed" in current_status.lower():
                        from sagemaker.core.utils.exceptions import FailedStatusError
                        raise FailedStatusError(
                            resource_type="PipelineExecution",
                            status=current_status,
                            reason=self.status.failure_reason,
                        )
                    
                    if timeout is not None and elapsed >= timeout:
                        from sagemaker.core.utils.exceptions import TimeoutExceededError
                        raise TimeoutExceededError(
                            resource_type="PipelineExecution",
                            status=current_status
                        )
                    
                    time.sleep(poll)
    
    def _enrich_with_step_details(
        self,
        session: Optional[Session] = None,
        region: Optional[str] = None
    ) -> None:
        """Fetch steps, extract S3 path, and update execution with details.
        
        Modifies execution in place. Handles errors gracefully.
        Internal method for use by get() and get_all().
        
        Args:
            session: Boto3 session (optional)
            region: AWS region (optional)
        """
        if not self._pipeline_execution:
            return
        
        try:
            steps_iterator = self._pipeline_execution.get_all_steps(session=session, region=region)
            raw_steps = list(steps_iterator)
            self._update_step_details_from_raw_steps(raw_steps)
            
            # Extract s3_output_path from training job's OutputDataConfig
            if not self.s3_output_path:
                self.s3_output_path = _extract_output_s3_location_from_steps(raw_steps, session, region)
        except Exception as e:
            logger.warning(f"Failed to fetch step details for execution {self.name}: {str(e)}")
    
    def _determine_eval_type(self, pipeline_arn: Optional[str] = None) -> Optional[EvalType]:
        """Determine eval_type from execution or pipeline ARN.
        
        Tries pipeline ARN first (preferred), falls back to execution ARN.
        Internal method for use by get().
        
        Args:
            pipeline_arn: Optional pipeline ARN to check first
            
        Returns:
            EvalType if found, None otherwise
        """
        # Try to determine eval_type from pipeline ARN (preferred method when available)
        if pipeline_arn:
            eval_type_from_pipeline = _extract_eval_type_from_arn(pipeline_arn)
            if eval_type_from_pipeline:
                return eval_type_from_pipeline
        
        # Fall back to execution ARN
        if self.arn:
            return _extract_eval_type_from_arn(self.arn)
        
        return None
    
    def _convert_to_subclass(self, eval_type: EvalType) -> 'EvaluationPipelineExecution':
        """Convert this execution instance to eval-type-specific subclass.
        
        Internal method for use by start(), get(), and get_all().
        
        Args:
            eval_type: Type of evaluation to determine subclass
            
        Returns:
            Execution instance of appropriate subclass
        """
        # Save reference before conversion
        pipeline_execution_ref = self._pipeline_execution
        execution_dict = _clean_unassigned_from_dict(self.dict())
        
        # Convert to appropriate subclass
        if eval_type == EvalType.BENCHMARK or eval_type == EvalType.CUSTOM_SCORER:
            execution = BenchmarkEvaluationExecution(**execution_dict)
        elif eval_type == EvalType.LLM_AS_JUDGE:
            execution = LLMAJEvaluationExecution(**execution_dict)
        else:
            execution = self
        
        # Restore internal pipeline execution reference
        execution._pipeline_execution = pipeline_execution_ref
        
        return execution
    
    def _update_step_details_from_raw_steps(self, raw_steps: List[Any]) -> None:
        """Internal method to update step_details from raw pipeline execution steps
        
        Args:
            raw_steps: List of PipelineExecutionStep objects from SageMaker
        """
        step_details = []
        
        for step in raw_steps:
            try:
                # Convert datetime objects to strings if they exist
                start_time = None
                end_time = None
                
                if hasattr(step, 'start_time') and step.start_time:
                    start_time = step.start_time.isoformat() if hasattr(step.start_time, 'isoformat') else str(step.start_time)
                
                if hasattr(step, 'end_time') and step.end_time:
                    end_time = step.end_time.isoformat() if hasattr(step.end_time, 'isoformat') else str(step.end_time)
                
                # Create StepDetail object
                # Handle step_display_name which might be an Unassigned object
                step_display_name = getattr(step, 'step_display_name', None)
                if step_display_name is not None and hasattr(step_display_name, '__class__'):
                    # Check if it's an Unassigned object from sagemaker_core
                    if 'Unassigned' in step_display_name.__class__.__name__:
                        step_display_name = None
                
                # Get failure reason if available
                failure_reason = getattr(step, 'failure_reason', None)
                if failure_reason is not None and hasattr(failure_reason, '__class__'):
                    # Check if it's an Unassigned object from sagemaker_core
                    if 'Unassigned' in failure_reason.__class__.__name__:
                        failure_reason = None
                
                step_detail = StepDetail(
                    name=getattr(step, 'step_name', 'Unknown Step'),
                    status=getattr(step, 'step_status', 'Unknown'),
                    start_time=start_time,
                    end_time=end_time,
                    display_name=step_display_name,
                    failure_reason=failure_reason
                )
                
                step_details.append(step_detail)
                
            except Exception as e:
                # If there's an error processing a step, log it but continue
                logger.warning(f"Failed to process pipeline step: {str(e)}")
                continue
        
        # Update the job's step details
        self.status.step_details = step_details


# ============================================================================
# Eval-Type-Specific Subclasses
# ============================================================================

class BenchmarkEvaluationExecution(EvaluationPipelineExecution):
    """Benchmark evaluation execution subclass with type-specific show_results().

    Provides benchmark-specific result display functionality for comparing
    custom model performance against a base model.
    """

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="BenchmarkEvaluationExecution.show_results")
    def show_results(self) -> None:
        """Display benchmark evaluation results comparing custom vs base model.

        Shows aggregate metrics with detailed S3 artifact locations.

        Raises:
            ValueError: If execution hasn't succeeded.

        Example:
            .. code:: python

                execution = evaluator.evaluate()
                execution.wait()
                execution.show_results()
        """
        # Refresh and validate status
        self.refresh()

        if self.status.overall_status != "Succeeded":
            raise ValueError(
                f"Cannot show results. Execution status is '{self.status.overall_status}'. "
                f"Results are only available after successful execution. "
                f"Use execution.wait() to wait for completion or check execution.status for details."
            )

        # Delegate to utility
        from ..common_utils.show_results_utils import _show_benchmark_results
        _show_benchmark_results(self)


class LLMAJEvaluationExecution(EvaluationPipelineExecution):
    """LLM As Judge evaluation execution subclass with type-specific show_results().

    Provides LLM-as-Judge-specific result display functionality with pagination
    and detailed judge explanations.
    """

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="LLMAJEvaluationExecution.show_results")
    def show_results(
        self,
        limit: int = 5,
        offset: int = 0,
        show_explanations: bool = False
    ) -> None:
        """Display LLM As Judge evaluation results with pagination.

        Shows per-evaluation results with prompt, response, and scores.

        Args:
            limit (int): Number of evaluation prompts to display. Set to None for all. Defaults to 5.
            offset (int): Starting index for pagination. Defaults to 0.
            show_explanations (bool): Whether to show judge explanations. Defaults to False.

        Raises:
            ValueError: If execution hasn't succeeded.

        Example:
            .. code:: python

                execution = evaluator.evaluate()
                execution.wait()

                # Show first 5 evaluations
                execution.show_results()

                # Show next 5
                execution.show_results(limit=5, offset=5)

                # Show all with explanations
                execution.show_results(limit=None, show_explanations=True)
        """
        # Refresh and validate status
        self.refresh()

        if self.status.overall_status != "Succeeded":
            raise ValueError(
                f"Cannot show results. Execution status is '{self.status.overall_status}'. "
                f"Results are only available after successful execution. "
                f"Use execution.wait() to wait for completion or check execution.status for details."
            )

        # Delegate to utility
        from ..common_utils.show_results_utils import _show_llmaj_results
        _show_llmaj_results(self, limit=limit, offset=offset, show_explanations=show_explanations)
