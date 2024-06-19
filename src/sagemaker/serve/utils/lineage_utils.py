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
"""Holds the util functions used for lineage tracking"""
from __future__ import absolute_import

import os
import time
import re
import logging
from typing import List, Optional, Union

from botocore.exceptions import ClientError

from sagemaker import Session
from sagemaker.lineage._api_types import ArtifactSummary
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association
from sagemaker.lineage.query import LineageSourceEnum
from sagemaker.serve.model_format.mlflow.constants import (
    MLFLOW_RUN_ID_REGEX,
    MODEL_PACKAGE_ARN_REGEX,
    S3_PATH_REGEX,
    MLFLOW_REGISTRY_PATH_REGEX,
)
from sagemaker.serve.utils.lineage_constants import (
    LINEAGE_POLLER_MAX_TIMEOUT_SECS,
    LINEAGE_POLLER_INTERVAL_SECS,
    TRACKING_SERVER_ARN_REGEX,
    TRACKING_SERVER_CREATION_TIME_FORMAT,
    MLFLOW_S3_PATH,
    MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE,
    MLFLOW_LOCAL_PATH,
    MLFLOW_MODEL_PACKAGE_PATH,
    MLFLOW_RUN_ID,
    MLFLOW_REGISTRY_PATH,
    CONTRIBUTED_TO,
    ERROR,
    CODE,
    VALIDATION_EXCEPTION,
)

logger = logging.getLogger(__name__)


def _load_artifact_by_source_uri(
    source_uri: str,
    sagemaker_session: Session,
    source_types_to_match: Optional[List[str]] = None,
    artifact_type: Optional[str] = None,
) -> Optional[ArtifactSummary]:
    """Load lineage artifact by source uri

    Arguments:
        source_uri (str): The s3 uri used for uploading transfomred model artifacts.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
        source_types_to_match (Optional[List[str]]): A list of source type values to match against
            the artifact's source types. If provided, the artifact's source types must match this
            list.
        artifact_type (Optional[str]): The type of the lineage artifact.

    Returns:
        ArtifactSummary: The Artifact Summary for the provided S3 URI.
    """
    artifacts = Artifact.list(source_uri=source_uri, sagemaker_session=sagemaker_session)
    for artifact_summary in artifacts:
        if artifact_type is None or artifact_summary.artifact_type == artifact_type:
            if source_types_to_match:
                if artifact_summary.source.source_types is not None:
                    artifact_source_types = [
                        source_type["Value"] for source_type in artifact_summary.source.source_types
                    ]
                    if set(artifact_source_types) == set(source_types_to_match):
                        return artifact_summary
                else:
                    return None
            else:
                return artifact_summary

    return None


def _poll_lineage_artifact(
    s3_uri: str, artifact_type: str, sagemaker_session: Session
) -> Optional[ArtifactSummary]:
    """Polls lineage artifacts by s3 path.

    Arguments:
        s3_uri (str): The S3 URI to check for artifacts.
        artifact_type (str): The type of the lineage artifact.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.

    Returns:
        Optional[ArtifactSummary]: The artifact summary if found, otherwise None.
    """
    logger.info("Polling lineage artifact for model data in %s", s3_uri)
    start_time = time.time()
    while time.time() - start_time < LINEAGE_POLLER_MAX_TIMEOUT_SECS:
        result = _load_artifact_by_source_uri(
            s3_uri, sagemaker_session, artifact_type=artifact_type
        )
        if result is not None:
            return result
        time.sleep(LINEAGE_POLLER_INTERVAL_SECS)


def _get_mlflow_model_path_type(mlflow_model_path: str) -> str:
    """Identify mlflow model path type.

    Args:
        mlflow_model_path (str): The string to be identified.

    Returns:
        str: Description of what the input string is identified as.
    """
    mlflow_run_id_pattern = MLFLOW_RUN_ID_REGEX
    mlflow_registry_id_pattern = MLFLOW_REGISTRY_PATH_REGEX
    sagemaker_arn_pattern = MODEL_PACKAGE_ARN_REGEX
    s3_pattern = S3_PATH_REGEX

    if re.match(mlflow_run_id_pattern, mlflow_model_path):
        return MLFLOW_RUN_ID
    if re.match(mlflow_registry_id_pattern, mlflow_model_path):
        return MLFLOW_REGISTRY_PATH
    if re.match(sagemaker_arn_pattern, mlflow_model_path):
        return MLFLOW_MODEL_PACKAGE_PATH
    if re.match(s3_pattern, mlflow_model_path):
        return MLFLOW_S3_PATH
    if os.path.exists(mlflow_model_path):
        return MLFLOW_LOCAL_PATH

    raise ValueError(f"Invalid MLflow model path: {mlflow_model_path}")


def _create_mlflow_model_path_lineage_artifact(
    mlflow_model_path: str,
    sagemaker_session: Session,
    source_types_to_match: Optional[List[str]] = None,
) -> Optional[Artifact]:
    """Creates a lineage artifact for the given MLflow model path.

    Args:
        mlflow_model_path (str): The path to the MLflow model.
        sagemaker_session (Session): The SageMaker session object.
        source_types_to_match (Optional[List[str]]): Artifact source types.

    Returns:
        Optional[Artifact]: The created lineage artifact, or None if an error occurred.
    """
    _artifact_name = _get_mlflow_model_path_type(mlflow_model_path)
    properties = dict(
        model_builder_input_model_data_type=_artifact_name,
    )
    try:
        source_types = [dict(SourceIdType="Custom", Value="ModelBuilderInputModelData")]
        if source_types_to_match:
            source_types += [
                dict(SourceIdType="Custom", Value=source_type)
                for source_type in source_types_to_match
                if source_type != "ModelBuilderInputModelData"
            ]

        return Artifact.create(
            source_uri=mlflow_model_path,
            source_types=source_types,
            artifact_type=MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE,
            artifact_name=_artifact_name,
            properties=properties,
            sagemaker_session=sagemaker_session,
        )
    except ClientError as e:
        if e.response[ERROR][CODE] == VALIDATION_EXCEPTION:
            logger.info("Artifact already exists")
        else:
            logger.warning("Failed to create mlflow model path lineage artifact: %s", e)
            raise e


def _retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact(
    mlflow_model_path: str,
    sagemaker_session: Session,
    tracking_server_arn: Optional[str] = None,
) -> Optional[Union[Artifact, ArtifactSummary]]:
    """Retrieves an existing artifact for the given MLflow model path or

    creates a new one if it doesn't exist.

    Args:
        mlflow_model_path (str): The path to the MLflow model.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
        tracking_server_arn (Optional[str]): The MLflow tracking server ARN.

    Returns:
        Optional[Union[Artifact, ArtifactSummary]]: The existing or newly created artifact,
            or None if an error occurred.
    """
    source_types_to_match = ["ModelBuilderInputModelData"]
    input_type = _get_mlflow_model_path_type(mlflow_model_path)
    if tracking_server_arn and input_type in [MLFLOW_RUN_ID, MLFLOW_REGISTRY_PATH]:
        match = re.match(TRACKING_SERVER_ARN_REGEX, tracking_server_arn)
        mlflow_tracking_server_name = match.group(4)
        describe_result = sagemaker_session.sagemaker_client.describe_mlflow_tracking_server(
            TrackingServerName=mlflow_tracking_server_name
        )
        tracking_server_creation_time = describe_result["CreationTime"].strftime(
            TRACKING_SERVER_CREATION_TIME_FORMAT
        )
        source_types_to_match += [tracking_server_arn, tracking_server_creation_time]
    _loaded_artifact = _load_artifact_by_source_uri(
        mlflow_model_path,
        sagemaker_session,
        source_types_to_match,
    )
    if _loaded_artifact is not None:
        return _loaded_artifact
    return _create_mlflow_model_path_lineage_artifact(
        mlflow_model_path,
        sagemaker_session,
        source_types_to_match,
    )


def _add_association_between_artifacts(
    mlflow_model_path_artifact_arn: str,
    autogenerated_model_data_artifact_arn: str,
    sagemaker_session: Session,
) -> None:
    """Add association between mlflow model path artifact and autogenerated model data artifact.

    Arguments:
        mlflow_model_path_artifact_arn (str): The mlflow model path artifact.
        autogenerated_model_data_artifact_arn (str): The autogenerated model data artifact.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    """
    _association_type = CONTRIBUTED_TO
    _source_arn = mlflow_model_path_artifact_arn
    _destination_arn = autogenerated_model_data_artifact_arn
    try:
        logger.info(
            "Adding association with source_arn: "
            "%s, destination_arn: %s and association_type: %s.",
            _source_arn,
            _destination_arn,
            _association_type,
        )
        Association.create(
            source_arn=_source_arn,
            destination_arn=_destination_arn,
            association_type=_association_type,
            sagemaker_session=sagemaker_session,
        )
    except ClientError as e:
        if e.response[ERROR][CODE] == VALIDATION_EXCEPTION:
            logger.info("Association already exists")
        else:
            raise e


def _maintain_lineage_tracking_for_mlflow_model(
    mlflow_model_path: str,
    s3_upload_path: str,
    sagemaker_session: Session,
    tracking_server_arn: Optional[str] = None,
) -> None:
    """Maintains lineage tracking for an MLflow model by creating or retrieving artifacts.

    Args:
        mlflow_model_path (str): The path to the MLflow model.
        s3_upload_path (str): The S3 path where the transformed model data is uploaded.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
        tracking_server_arn (Optional[str]): The MLflow tracking server ARN.
    """
    artifact_for_transformed_model_data = _poll_lineage_artifact(
        s3_uri=s3_upload_path,
        artifact_type=LineageSourceEnum.MODEL_DATA.value,
        sagemaker_session=sagemaker_session,
    )
    if artifact_for_transformed_model_data:
        mlflow_model_artifact = (
            _retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact(
                mlflow_model_path=mlflow_model_path,
                sagemaker_session=sagemaker_session,
                tracking_server_arn=tracking_server_arn,
            )
        )
        if mlflow_model_artifact:
            _mlflow_model_artifact_arn = (
                mlflow_model_artifact.artifact_arn
            )  # pylint: disable=E1101, disable=C0301
            _artifact_for_transformed_model_data_arn = (
                artifact_for_transformed_model_data.artifact_arn
            )  # pylint: disable=C0301
            _add_association_between_artifacts(
                mlflow_model_path_artifact_arn=_mlflow_model_artifact_arn,
                autogenerated_model_data_artifact_arn=_artifact_for_transformed_model_data_arn,
                sagemaker_session=sagemaker_session,
            )
        else:
            logger.warning(
                "Unable to add association between autogenerated lineage "
                "artifact for transformed model data and mlflow model path"
                " lineage artifacts."
            )
    else:
        logger.warning(
            "Lineage artifact for transformed model data is not auto-created within "
            "%s seconds, skipping creation of lineage artifact for mlflow model path",
            LINEAGE_POLLER_MAX_TIMEOUT_SECS,
        )
