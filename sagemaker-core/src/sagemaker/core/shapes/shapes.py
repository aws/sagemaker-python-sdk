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
import datetime
import warnings

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Optional, Any, Union
from sagemaker.core.utils.utils import Unassigned
from sagemaker.core.helper.pipeline_variable import StrPipeVar

# Suppress Pydantic warnings about field names shadowing parent attributes
warnings.filterwarnings("ignore", message=".*shadows an attribute.*")


class Base(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), validate_assignment=True, extra="forbid")


class InternalDependencyException(Base):
    """
    InternalDependencyException
      Your request caused an exception with an internal dependency. Contact customer support.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class InternalFailure(Base):
    """
    InternalFailure
      An internal failure occurred. Try your request again. If the problem persists, contact Amazon Web Services customer support.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class InternalStreamFailure(Base):
    """
    InternalStreamFailure
      The stream processing failed because of an unknown error, exception or failure. Try your request again.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class InvokeEndpointAsyncOutput(Base):
    """
    InvokeEndpointAsyncOutput

    Attributes
    ----------------------
    inference_id: Identifier for an inference request. This will be the same as the InferenceId specified in the input. Amazon SageMaker will generate an identifier for you if you do not specify one.
    output_location: The Amazon S3 URI where the inference response payload is stored.
    failure_location: The Amazon S3 URI where the inference failure response payload is stored.
    """

    inference_id: Optional[StrPipeVar] = Unassigned()
    output_location: Optional[StrPipeVar] = Unassigned()
    failure_location: Optional[StrPipeVar] = Unassigned()


class InvokeEndpointOutput(Base):
    """
    InvokeEndpointOutput

    Attributes
    ----------------------
    body: Includes the inference provided by the model.  For information about the format of the response body, see Common Data Formats-Inference. If the explainer is activated, the body includes the explanations provided by the model. For more information, see the Response section under Invoke the Endpoint in the Developer Guide.
    content_type: The MIME type of the inference returned from the model container.
    invoked_production_variant: Identifies the production variant that was invoked.
    custom_attributes: Provides additional information in the response about the inference returned by a model hosted at an Amazon SageMaker endpoint. The information is an opaque value that is forwarded verbatim. You could use this value, for example, to return an ID received in the CustomAttributes header of a request or other metadata that a service endpoint was programmed to produce. The value must consist of no more than 1024 visible US-ASCII characters as specified in Section 3.3.6. Field Value Components of the Hypertext Transfer Protocol (HTTP/1.1). If the customer wants the custom attribute returned, the model must set the custom attribute to be included on the way back.  The code in your model is responsible for setting or updating any custom attributes in the response. If your code does not set this value in the response, an empty value is returned. For example, if a custom attribute represents the trace ID, your model can prepend the custom attribute with Trace ID: in your post-processing function. This feature is currently supported in the Amazon Web Services SDKs but not in the Amazon SageMaker Python SDK.
    new_session_id: If you created a stateful session with your request, the ID and expiration time that the model assigns to that session.
    closed_session_id: If you closed a stateful session with your request, the ID of that session.
    """

    body: Any
    content_type: Optional[StrPipeVar] = Unassigned()
    invoked_production_variant: Optional[StrPipeVar] = Unassigned()
    custom_attributes: Optional[StrPipeVar] = Unassigned()
    new_session_id: Optional[StrPipeVar] = Unassigned()
    closed_session_id: Optional[StrPipeVar] = Unassigned()


class PayloadPart(Base):
    """
    PayloadPart
      A wrapper for pieces of the payload that's returned in response to a streaming inference request. A streaming inference response consists of one or more payload parts.

    Attributes
    ----------------------
    bytes: A blob that contains part of the response for your streaming inference request.
    """

    bytes: Optional[Any] = Unassigned()


class ModelStreamError(Base):
    """
    ModelStreamError
       An error occurred while streaming the response body. This error can have the following error codes:  ModelInvocationTimeExceeded  The model failed to finish sending the response within the timeout period allowed by Amazon SageMaker.  StreamBroken  The Transmission Control Protocol (TCP) connection between the client and the model was reset or closed.

    Attributes
    ----------------------
    message
    error_code: This error can have the following error codes:  ModelInvocationTimeExceeded  The model failed to finish sending the response within the timeout period allowed by Amazon SageMaker.  StreamBroken  The Transmission Control Protocol (TCP) connection between the client and the model was reset or closed.
    """

    message: Optional[StrPipeVar] = Unassigned()
    error_code: Optional[StrPipeVar] = Unassigned()


class ResponseStream(Base):
    """
    ResponseStream
      A stream of payload parts. Each part contains a portion of the response for a streaming inference request.

    Attributes
    ----------------------
    payload_part: A wrapper for pieces of the payload that's returned in response to a streaming inference request. A streaming inference response consists of one or more payload parts.
    model_stream_error:  An error occurred while streaming the response body. This error can have the following error codes:  ModelInvocationTimeExceeded  The model failed to finish sending the response within the timeout period allowed by Amazon SageMaker.  StreamBroken  The Transmission Control Protocol (TCP) connection between the client and the model was reset or closed.
    internal_stream_failure: The stream processing failed because of an unknown error, exception or failure. Try your request again.
    """

    payload_part: Optional[PayloadPart] = Unassigned()
    model_stream_error: Optional[ModelStreamError] = Unassigned()
    internal_stream_failure: Optional[InternalStreamFailure] = Unassigned()


class InvokeEndpointWithResponseStreamOutput(Base):
    """
    InvokeEndpointWithResponseStreamOutput

    Attributes
    ----------------------
    body
    content_type: The MIME type of the inference returned from the model container.
    invoked_production_variant: Identifies the production variant that was invoked.
    custom_attributes: Provides additional information in the response about the inference returned by a model hosted at an Amazon SageMaker endpoint. The information is an opaque value that is forwarded verbatim. You could use this value, for example, to return an ID received in the CustomAttributes header of a request or other metadata that a service endpoint was programmed to produce. The value must consist of no more than 1024 visible US-ASCII characters as specified in Section 3.3.6. Field Value Components of the Hypertext Transfer Protocol (HTTP/1.1). If the customer wants the custom attribute returned, the model must set the custom attribute to be included on the way back.  The code in your model is responsible for setting or updating any custom attributes in the response. If your code does not set this value in the response, an empty value is returned. For example, if a custom attribute represents the trace ID, your model can prepend the custom attribute with Trace ID: in your post-processing function. This feature is currently supported in the Amazon Web Services SDKs but not in the Amazon SageMaker Python SDK.
    """

    body: ResponseStream
    content_type: Optional[StrPipeVar] = Unassigned()
    invoked_production_variant: Optional[StrPipeVar] = Unassigned()
    custom_attributes: Optional[StrPipeVar] = Unassigned()


class ModelError(Base):
    """
    ModelError
       Model (owned by the customer in the container) returned 4xx or 5xx error code.

    Attributes
    ----------------------
    message
    original_status_code:  Original status code.
    original_message:  Original message.
    log_stream_arn:  The Amazon Resource Name (ARN) of the log stream.
    """

    message: Optional[StrPipeVar] = Unassigned()
    original_status_code: Optional[int] = Unassigned()
    original_message: Optional[StrPipeVar] = Unassigned()
    log_stream_arn: Optional[StrPipeVar] = Unassigned()


class ModelNotReadyException(Base):
    """
    ModelNotReadyException
      Either a serverless endpoint variant's resources are still being provisioned, or a multi-model endpoint is still downloading or loading the target model. Wait and try your request again.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class ServiceUnavailable(Base):
    """
    ServiceUnavailable
      The service is currently unavailable.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class ValidationError(Base):
    """
    ValidationError
      There was an error validating your request.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class AccessForbidden(Base):
    """
    AccessForbidden
      You do not have permission to perform an action.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class BatchGetRecordError(Base):
    """
    BatchGetRecordError
      The error that has occurred when attempting to retrieve a batch of Records.

    Attributes
    ----------------------
    feature_group_name: The name of the feature group that the record belongs to.
    record_identifier_value_as_string: The value for the RecordIdentifier in string format of a Record from a FeatureGroup that is causing an error when attempting to be retrieved.
    error_code: The error code of an error that has occurred when attempting to retrieve a batch of Records. For more information on errors, see Errors.
    error_message: The error message of an error that has occurred when attempting to retrieve a record in the batch.
    """

    feature_group_name: Union[StrPipeVar, object]
    record_identifier_value_as_string: StrPipeVar
    error_code: StrPipeVar
    error_message: StrPipeVar


class BatchGetRecordIdentifier(Base):
    """
    BatchGetRecordIdentifier
      The identifier that identifies the batch of Records you are retrieving in a batch.

    Attributes
    ----------------------
    feature_group_name: The name or Amazon Resource Name (ARN) of the FeatureGroup containing the records you are retrieving in a batch.
    record_identifiers_value_as_string: The value for a list of record identifiers in string format.
    feature_names: List of names of Features to be retrieved. If not specified, the latest value for all the Features are returned.
    """

    feature_group_name: Union[StrPipeVar, object]
    record_identifiers_value_as_string: List[StrPipeVar]
    feature_names: Optional[List[StrPipeVar]] = Unassigned()


class FeatureValue(Base):
    """
    FeatureValue
      The value associated with a feature.

    Attributes
    ----------------------
    feature_name: The name of a feature that a feature value corresponds to.
    value_as_string: The value in string format associated with a feature. Used when your CollectionType is None. Note that features types can be String, Integral, or Fractional. This value represents all three types as a string.
    value_as_string_list: The list of values in string format associated with a feature. Used when your CollectionType is a List, Set, or Vector. Note that features types can be String, Integral, or Fractional. These values represents all three types as a string.
    """

    feature_name: StrPipeVar
    value_as_string: Optional[StrPipeVar] = Unassigned()
    value_as_string_list: Optional[List[StrPipeVar]] = Unassigned()


class BatchGetRecordResultDetail(Base):
    """
    BatchGetRecordResultDetail
      The output of records that have been retrieved in a batch.

    Attributes
    ----------------------
    feature_group_name: The FeatureGroupName containing Records you retrieved in a batch.
    record_identifier_value_as_string: The value of the record identifier in string format.
    record: The Record retrieved.
    expires_at: The ExpiresAt ISO string of the requested record.
    """

    feature_group_name: Union[StrPipeVar, object]
    record_identifier_value_as_string: StrPipeVar
    record: List[FeatureValue]
    expires_at: Optional[StrPipeVar] = Unassigned()


class BatchGetRecordResponse(Base):
    """
    BatchGetRecordResponse

    Attributes
    ----------------------
    records: A list of Records you requested to be retrieved in batch.
    errors: A list of errors that have occurred when retrieving a batch of Records.
    unprocessed_identifiers: A unprocessed list of FeatureGroup names, with their corresponding RecordIdentifier value, and Feature name.
    """

    records: List[BatchGetRecordResultDetail]
    errors: List[BatchGetRecordError]
    unprocessed_identifiers: List[BatchGetRecordIdentifier]


class GetRecordResponse(Base):
    """
    GetRecordResponse

    Attributes
    ----------------------
    record: The record you requested. A list of FeatureValues.
    expires_at: The ExpiresAt ISO string of the requested record.
    """

    record: Optional[List[FeatureValue]] = Unassigned()
    expires_at: Optional[StrPipeVar] = Unassigned()


class TtlDuration(Base):
    """
    TtlDuration
      Time to live duration, where the record is hard deleted after the expiration time is reached; ExpiresAt = EventTime + TtlDuration. For information on HardDelete, see the DeleteRecord API in the Amazon SageMaker API Reference guide.

    Attributes
    ----------------------
    unit:  TtlDuration time unit.
    value:  TtlDuration time value.
    """

    unit: Optional[StrPipeVar] = Unassigned()
    value: Optional[int] = Unassigned()


class ResourceNotFound(Base):
    """
    ResourceNotFound
      Resource being access is not found.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class MetricQuery(Base):
    """
    MetricQuery

    Attributes
    ----------------------
    metric_name
    resource_arn
    metric_stat
    period
    x_axis_type
    start
    end
    start_iteration_number
    end_iteration_number
    """

    metric_name: StrPipeVar
    resource_arn: StrPipeVar
    metric_stat: StrPipeVar
    period: StrPipeVar
    x_axis_type: StrPipeVar
    start: Optional[datetime.datetime] = Unassigned()
    end: Optional[datetime.datetime] = Unassigned()
    start_iteration_number: Optional[int] = Unassigned()
    end_iteration_number: Optional[int] = Unassigned()


class MetricQueryResult(Base):
    """
    MetricQueryResult

    Attributes
    ----------------------
    status
    message
    iteration_numbers
    timestamps
    metric_values
    """

    status: StrPipeVar
    metric_values: List[float]
    message: Optional[StrPipeVar] = Unassigned()
    iteration_numbers: Optional[List[int]] = Unassigned()
    timestamps: Optional[List[datetime.datetime]] = Unassigned()


class BatchGetMetricsResponse(Base):
    """
    BatchGetMetricsResponse

    Attributes
    ----------------------
    metric_query_results
    """

    metric_query_results: Optional[List[MetricQueryResult]] = Unassigned()


class BatchPutMetricsError(Base):
    """
    BatchPutMetricsError

    Attributes
    ----------------------
    code
    message
    metric_index
    """

    code: StrPipeVar
    message: StrPipeVar
    metric_index: int


class RawMetricData(Base):
    """
    RawMetricData

    Attributes
    ----------------------
    metric_name
    timestamp
    iteration_number
    value
    """

    metric_name: StrPipeVar
    timestamp: datetime.datetime
    value: float
    iteration_number: Optional[int] = Unassigned()


class AcceleratorPartitionConfig(Base):
    """
    AcceleratorPartitionConfig

    Attributes
    ----------------------
    type
    count
    """

    type: StrPipeVar
    count: int


class AccessDeniedException(Base):
    """
    AccessDeniedException

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class ActionSource(Base):
    """
    ActionSource
      A structure describing the source of an action.

    Attributes
    ----------------------
    source_uri: The URI of the source.
    source_type: The type of the source.
    source_id: The ID of the source.
    """

    source_uri: StrPipeVar
    source_type: Optional[StrPipeVar] = Unassigned()
    source_id: Optional[StrPipeVar] = Unassigned()


class ActionSummary(Base):
    """
    ActionSummary
      Lists the properties of an action. An action represents an action or activity. Some examples are a workflow step and a model deployment. Generally, an action involves at least one input artifact or output artifact.

    Attributes
    ----------------------
    action_arn: The Amazon Resource Name (ARN) of the action.
    action_name: The name of the action.
    source: The source of the action.
    action_type: The type of the action.
    status: The status of the action.
    creation_time: When the action was created.
    last_modified_time: When the action was last modified.
    """

    action_arn: Optional[StrPipeVar] = Unassigned()
    action_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    source: Optional[ActionSource] = Unassigned()
    action_type: Optional[StrPipeVar] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class ActivationStateV1(Base):
    """
    ActivationStateV1

    Attributes
    ----------------------
    enabled
    """

    enabled: Optional[bool] = Unassigned()


class IamIdentity(Base):
    """
    IamIdentity
      The IAM Identity details associated with the user. These details are associated with model package groups, model packages and project entities only.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the IAM identity.
    principal_id: The ID of the principal that assumes the IAM identity.
    source_identity: The person or application which assumes the IAM identity.
    """

    arn: Optional[StrPipeVar] = Unassigned()
    principal_id: Optional[StrPipeVar] = Unassigned()
    source_identity: Optional[StrPipeVar] = Unassigned()


class UserContext(Base):
    """
    UserContext
      Information about the user who created or modified a SageMaker resource.

    Attributes
    ----------------------
    user_profile_arn: The Amazon Resource Name (ARN) of the user's profile.
    user_profile_name: The name of the user's profile.
    domain_id: The domain associated with the user.
    iam_identity: The IAM Identity details associated with the user. These details are associated with model package groups, model packages, and project entities only.
    """

    user_profile_arn: Optional[StrPipeVar] = Unassigned()
    user_profile_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    domain_id: Optional[StrPipeVar] = Unassigned()
    iam_identity: Optional[IamIdentity] = Unassigned()


class CustomerDetails(Base):
    """
    CustomerDetails

    Attributes
    ----------------------
    account_id
    user_context
    organization_id
    """

    account_id: StrPipeVar
    user_context: Optional[UserContext] = Unassigned()
    organization_id: Optional[StrPipeVar] = Unassigned()


class AddClusterNodeSpecification(Base):
    """
    AddClusterNodeSpecification
      Specifies an instance group and the number of nodes to add to it.

    Attributes
    ----------------------
    instance_group_name: The name of the instance group to which you want to add nodes.
    increment_target_count_by: The number of nodes to add to the specified instance group. The total number of nodes across all instance groups in a single request cannot exceed 50.
    """

    instance_group_name: StrPipeVar
    increment_target_count_by: int


class OnlineStoreSecurityConfig(Base):
    """
    OnlineStoreSecurityConfig
      The security configuration for OnlineStore.

    Attributes
    ----------------------
    kms_key_id: The Amazon Web Services Key Management Service (KMS) key ARN that SageMaker Feature Store uses to encrypt the Amazon S3 objects at rest using Amazon S3 server-side encryption. The caller (either user or IAM role) of CreateFeatureGroup must have below permissions to the OnlineStore KmsKeyId:    "kms:Encrypt"     "kms:Decrypt"     "kms:DescribeKey"     "kms:CreateGrant"     "kms:RetireGrant"     "kms:ReEncryptFrom"     "kms:ReEncryptTo"     "kms:GenerateDataKey"     "kms:ListAliases"     "kms:ListGrants"     "kms:RevokeGrant"    The caller (either user or IAM role) to all DataPlane operations (PutRecord, GetRecord, DeleteRecord) must have the following permissions to the KmsKeyId:    "kms:Decrypt"
    """

    kms_key_id: Optional[StrPipeVar] = Unassigned()


class OnlineStoreReplicaConfig(Base):
    """
    OnlineStoreReplicaConfig

    Attributes
    ----------------------
    security_config
    """

    security_config: Optional[OnlineStoreSecurityConfig] = Unassigned()


class Tag(Base):
    """
    Tag
      A tag object that consists of a key and an optional value, used to manage metadata for SageMaker Amazon Web Services resources. You can add tags to notebook instances, training jobs, hyperparameter tuning jobs, batch transform jobs, models, labeling jobs, work teams, endpoint configurations, and endpoints. For more information on adding tags to SageMaker resources, see AddTags. For more information on adding metadata to your Amazon Web Services resources with tagging, see Tagging Amazon Web Services resources. For advice on best practices for managing Amazon Web Services resources with tagging, see Tagging Best Practices: Implement an Effective Amazon Web Services Resource Tagging Strategy.

    Attributes
    ----------------------
    key: The tag key. Tag keys must be unique per resource.
    value: The tag value.
    """

    key: StrPipeVar
    value: StrPipeVar


class AddOnlineStoreReplicaAction(Base):
    """
    AddOnlineStoreReplicaAction

    Attributes
    ----------------------
    region_name
    online_store_config
    description
    tags
    """

    region_name: StrPipeVar
    online_store_config: Optional[OnlineStoreReplicaConfig] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class AdditionalEnis(Base):
    """
    AdditionalEnis
      Information about additional Elastic Network Interfaces (ENIs) associated with an instance.

    Attributes
    ----------------------
    efa_enis: A list of Elastic Fabric Adapter (EFA) ENIs associated with the instance.
    """

    efa_enis: Optional[List[StrPipeVar]] = Unassigned()


class ModelAccessConfig(Base):
    """
    ModelAccessConfig
      The access configuration file to control access to the ML model. You can explicitly accept the model end-user license agreement (EULA) within the ModelAccessConfig.   If you are a Jumpstart user, see the End-user license agreements section for more details on accepting the EULA.   If you are an AutoML user, see the Optional Parameters section of Create an AutoML job to fine-tune text generation models using the API for details on How to set the EULA acceptance when fine-tuning a model using the AutoML API.

    Attributes
    ----------------------
    accept_eula: Specifies agreement to the model end-user license agreement (EULA). The AcceptEula value must be explicitly defined as True in order to accept the EULA that this model requires. You are responsible for reviewing and complying with any applicable license terms and making sure they are acceptable for your use case before downloading or using a model.
    """

    accept_eula: bool


class InferenceHubAccessConfig(Base):
    """
    InferenceHubAccessConfig
      Configuration information specifying which hub contents have accessible deployment options.

    Attributes
    ----------------------
    hub_content_arn: The ARN of the hub content for which deployment access is allowed.
    """

    hub_content_arn: StrPipeVar


class S3ModelDataSource(Base):
    """
    S3ModelDataSource
      Specifies the S3 location of ML model data to deploy.

    Attributes
    ----------------------
    s3_uri: Specifies the S3 path of ML model data to deploy.
    s3_data_type: Specifies the type of ML model data to deploy. If you choose S3Prefix, S3Uri identifies a key name prefix. SageMaker uses all objects that match the specified key name prefix as part of the ML model data to deploy. A valid key name prefix identified by S3Uri always ends with a forward slash (/). If you choose S3Object, S3Uri identifies an object that is the ML model data to deploy.
    compression_type: Specifies how the ML model data is prepared. If you choose Gzip and choose S3Object as the value of S3DataType, S3Uri identifies an object that is a gzip-compressed TAR archive. SageMaker will attempt to decompress and untar the object during model deployment. If you choose None and chooose S3Object as the value of S3DataType, S3Uri identifies an object that represents an uncompressed ML model to deploy. If you choose None and choose S3Prefix as the value of S3DataType, S3Uri identifies a key name prefix, under which all objects represents the uncompressed ML model to deploy. If you choose None, then SageMaker will follow rules below when creating model data files under /opt/ml/model directory for use by your inference code:   If you choose S3Object as the value of S3DataType, then SageMaker will split the key of the S3 object referenced by S3Uri by slash (/), and use the last part as the filename of the file holding the content of the S3 object.   If you choose S3Prefix as the value of S3DataType, then for each S3 object under the key name pefix referenced by S3Uri, SageMaker will trim its key by the prefix, and use the remainder as the path (relative to /opt/ml/model) of the file holding the content of the S3 object. SageMaker will split the remainder by slash (/), using intermediate parts as directory names and the last part as filename of the file holding the content of the S3 object.   Do not use any of the following as file names or directory names:   An empty or blank string   A string which contains null bytes   A string longer than 255 bytes   A single dot (.)   A double dot (..)     Ambiguous file names will result in model deployment failure. For example, if your uncompressed ML model consists of two S3 objects s3://mybucket/model/weights and s3://mybucket/model/weights/part1 and you specify s3://mybucket/model/ as the value of S3Uri and S3Prefix as the value of S3DataType, then it will result in name clash between /opt/ml/model/weights (a regular file) and /opt/ml/model/weights/ (a directory).   Do not organize the model artifacts in S3 console using folders. When you create a folder in S3 console, S3 creates a 0-byte object with a key set to the folder name you provide. They key of the 0-byte object ends with a slash (/) which violates SageMaker restrictions on model artifact file names, leading to model deployment failure.
    model_access_config: Specifies the access configuration file for the ML model. You can explicitly accept the model end-user license agreement (EULA) within the ModelAccessConfig. You are responsible for reviewing and complying with any applicable license terms and making sure they are acceptable for your use case before downloading or using a model.
    hub_access_config: Configuration information for hub access.
    manifest_s3_uri: The Amazon S3 URI of the manifest file. The manifest file is a CSV file that stores the artifact locations.
    e_tag: The ETag associated with S3 URI.
    manifest_etag: The ETag associated with Manifest S3 URI.
    """

    s3_uri: StrPipeVar
    s3_data_type: StrPipeVar
    compression_type: StrPipeVar
    model_access_config: Optional[ModelAccessConfig] = Unassigned()
    hub_access_config: Optional[InferenceHubAccessConfig] = Unassigned()
    manifest_s3_uri: Optional[StrPipeVar] = Unassigned()
    e_tag: Optional[StrPipeVar] = Unassigned()
    manifest_etag: Optional[StrPipeVar] = Unassigned()


class ModelDataSource(Base):
    """
    ModelDataSource
      Specifies the location of ML model data to deploy. If specified, you must specify one and only one of the available data sources.

    Attributes
    ----------------------
    s3_data_source: Specifies the S3 location of ML model data to deploy.
    """

    s3_data_source: Optional[S3ModelDataSource] = Unassigned()


class ModelInput(Base):
    """
    ModelInput
      Input object for the model.

    Attributes
    ----------------------
    data_input_config: The input configuration object for the model.
    """

    data_input_config: StrPipeVar


class AdditionalS3DataSource(Base):
    """
    AdditionalS3DataSource
      A data source used for training or inference that is in addition to the input dataset or model data.

    Attributes
    ----------------------
    s3_data_type: The data type of the additional data source that you specify for use in inference or training.
    s3_uri: The uniform resource identifier (URI) used to identify an additional data source used in inference or training.
    compression_type: The type of compression used for an additional data source used in inference or training. Specify None if your additional data source is not compressed.
    manifest_s3_uri
    e_tag: The ETag associated with S3 URI.
    manifest_etag
    """

    s3_data_type: StrPipeVar
    s3_uri: StrPipeVar
    compression_type: Optional[StrPipeVar] = Unassigned()
    manifest_s3_uri: Optional[StrPipeVar] = Unassigned()
    e_tag: Optional[StrPipeVar] = Unassigned()
    manifest_etag: Optional[StrPipeVar] = Unassigned()


class BaseModel(Base):
    """
    BaseModel

    Attributes
    ----------------------
    hub_content_name
    hub_content_version
    recipe_name
    """

    hub_content_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    hub_content_version: Optional[StrPipeVar] = Unassigned()
    recipe_name: Optional[StrPipeVar] = Unassigned()


class ModelPackageContainerDefinition(Base):
    """
    ModelPackageContainerDefinition
      Describes the Docker container for the model package.

    Attributes
    ----------------------
    container_hostname: The DNS host name for the Docker container.
    image: The Amazon Elastic Container Registry (Amazon ECR) path where inference code is stored. If you are using your own custom algorithm instead of an algorithm provided by SageMaker, the inference code must meet SageMaker requirements. SageMaker supports both registry/repository[:tag] and registry/repository[@digest] image path formats. For more information, see Using Your Own Algorithms with Amazon SageMaker.
    image_digest: An MD5 hash of the training algorithm that identifies the Docker image used for training.
    model_data_url: The Amazon S3 path where the model artifacts, which result from model training, are stored. This path must point to a single gzip compressed tar archive (.tar.gz suffix).  The model artifacts must be in an S3 bucket that is in the same region as the model package.
    model_data_source: Specifies the location of ML model data to deploy during endpoint creation.
    product_id: The Amazon Web Services Marketplace product ID of the model package.
    environment: The environment variables to set in the Docker container. Each key and value in the Environment string to string map can have length of up to 1024. We support up to 16 entries in the map.
    model_input: A structure with Model Input details.
    framework: The machine learning framework of the model package container image.
    framework_version: The framework version of the Model Package Container Image.
    nearest_model_name: The name of a pre-trained machine learning benchmarked by Amazon SageMaker Inference Recommender model that matches your model. You can find a list of benchmarked models by calling ListModelMetadata.
    sample_payload_url
    additional_s3_data_source: The additional data source that is used during inference in the Docker container for your model package.
    model_data_e_tag: The ETag associated with Model Data URL.
    is_checkpoint
    base_model
    """

    container_hostname: Optional[StrPipeVar] = Unassigned()
    image: Optional[StrPipeVar] = Unassigned() # Revert back to autogen version
    image_digest: Optional[StrPipeVar] = Unassigned()
    model_data_url: Optional[StrPipeVar] = Unassigned()
    model_data_source: Optional[ModelDataSource] = Unassigned()
    product_id: Optional[StrPipeVar] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    model_input: Optional[ModelInput] = Unassigned()
    framework: Optional[StrPipeVar] = Unassigned()
    framework_version: Optional[StrPipeVar] = Unassigned()
    nearest_model_name: Optional[StrPipeVar] = Unassigned()
    sample_payload_url: Optional[StrPipeVar] = Unassigned()
    additional_s3_data_source: Optional[AdditionalS3DataSource] = Unassigned()
    model_data_e_tag: Optional[StrPipeVar] = Unassigned()
    is_checkpoint: Optional[bool] = Unassigned()
    base_model: Optional[BaseModel] = Unassigned()


class AdditionalInferenceSpecificationDefinition(Base):
    """
    AdditionalInferenceSpecificationDefinition
      A structure of additional Inference Specification. Additional Inference Specification specifies details about inference jobs that can be run with models based on this model package

    Attributes
    ----------------------
    name: A unique name to identify the additional inference specification. The name must be unique within the list of your additional inference specifications for a particular model package.
    description: A description of the additional Inference specification
    containers: The Amazon ECR registry path of the Docker image that contains the inference code.
    supported_transform_instance_types: A list of the instance types on which a transformation job can be run or on which an endpoint can be deployed.
    supported_realtime_inference_instance_types: A list of the instance types that are used to generate inferences in real-time.
    supported_content_types: The supported MIME types for the input data.
    supported_response_mime_types: The supported MIME types for the output data.
    """

    name: StrPipeVar
    containers: List[ModelPackageContainerDefinition]
    description: Optional[StrPipeVar] = Unassigned()
    supported_transform_instance_types: Optional[List[StrPipeVar]] = Unassigned()
    supported_realtime_inference_instance_types: Optional[List[StrPipeVar]] = Unassigned()
    supported_content_types: Optional[List[StrPipeVar]] = Unassigned()
    supported_response_mime_types: Optional[List[StrPipeVar]] = Unassigned()


class AdditionalModelDataSource(Base):
    """
    AdditionalModelDataSource
      Data sources that are available to your model in addition to the one that you specify for ModelDataSource when you use the CreateModel action.

    Attributes
    ----------------------
    channel_name: A custom name for this AdditionalModelDataSource object.
    s3_data_source
    """

    channel_name: StrPipeVar
    s3_data_source: S3ModelDataSource


class AgentVersion(Base):
    """
    AgentVersion
      Edge Manager agent version.

    Attributes
    ----------------------
    version: Version of the agent.
    agent_count: The number of Edge Manager agents.
    """

    version: StrPipeVar
    agent_count: int


class AgentsCredentialProvider(Base):
    """
    AgentsCredentialProvider

    Attributes
    ----------------------
    algorithm_container_credential_provider
    algorithm_container_secondary_credential_provider
    training_image_credential_provider
    """

    training_image_credential_provider: StrPipeVar
    algorithm_container_credential_provider: Optional[StrPipeVar] = Unassigned()
    algorithm_container_secondary_credential_provider: Optional[StrPipeVar] = Unassigned()


class Alarm(Base):
    """
    Alarm
      An Amazon CloudWatch alarm configured to monitor metrics on an endpoint.

    Attributes
    ----------------------
    alarm_name: The name of a CloudWatch alarm in your account.
    """

    alarm_name: Optional[StrPipeVar] = Unassigned()


class AlarmDetails(Base):
    """
    AlarmDetails
      The details of the alarm to monitor during the AMI update.

    Attributes
    ----------------------
    alarm_name: The name of the alarm.
    """

    alarm_name: StrPipeVar


class MetricDefinition(Base):
    """
    MetricDefinition
      Specifies a metric that the training algorithm writes to stderr or stdout. You can view these logs to understand how your training job performs and check for any errors encountered during training. SageMaker hyperparameter tuning captures all defined metrics. Specify one of the defined metrics to use as an objective metric using the TuningObjective parameter in the HyperParameterTrainingJobDefinition API to evaluate job performance during hyperparameter tuning.

    Attributes
    ----------------------
    name: The name of the metric.
    regex: A regular expression that searches the output of a training job and gets the value of the metric. For more information about using regular expressions to define metrics, see Defining metrics and environment variables.
    """

    name: StrPipeVar
    regex: StrPipeVar


class TrainingRepositoryAuthConfig(Base):
    """
    TrainingRepositoryAuthConfig
      An object containing authentication information for a private Docker registry.

    Attributes
    ----------------------
    training_repository_credentials_provider_arn: The Amazon Resource Name (ARN) of an Amazon Web Services Lambda function used to give SageMaker access credentials to your private Docker registry.
    """

    training_repository_credentials_provider_arn: StrPipeVar


class TrainingImageConfig(Base):
    """
    TrainingImageConfig
      The configuration to use an image from a private Docker registry for a training job.

    Attributes
    ----------------------
    training_repository_access_mode: The method that your training job will use to gain access to the images in your private Docker registry. For access to an image in a private Docker registry, set to Vpc.
    training_repository_auth_config: An object containing authentication information for a private Docker registry containing your training images.
    """

    training_repository_access_mode: StrPipeVar
    training_repository_auth_config: Optional[TrainingRepositoryAuthConfig] = Unassigned()


class AlgorithmSpecification(Base):
    """
    AlgorithmSpecification
      Specifies the training algorithm to use in a CreateTrainingJob request.  SageMaker uses its own SageMaker account credentials to pull and access built-in algorithms so built-in algorithms are universally accessible across all Amazon Web Services accounts. As a result, built-in algorithms have standard, unrestricted access. You cannot restrict built-in algorithms using IAM roles. Use custom algorithms if you require specific access controls.  For more information about algorithms provided by SageMaker, see Algorithms. For information about using your own algorithms, see Using Your Own Algorithms with Amazon SageMaker.

    Attributes
    ----------------------
    training_image: The registry path of the Docker image that contains the training algorithm. For information about docker registry paths for SageMaker built-in algorithms, see Docker Registry Paths and Example Code in the Amazon SageMaker developer guide. SageMaker supports both registry/repository[:tag] and registry/repository[@digest] image path formats. For more information about using your custom training container, see Using Your Own Algorithms with Amazon SageMaker.  You must specify either the algorithm name to the AlgorithmName parameter or the image URI of the algorithm container to the TrainingImage parameter. For more information, see the note in the AlgorithmName parameter description.
    algorithm_name: The name of the algorithm resource to use for the training job. This must be an algorithm resource that you created or subscribe to on Amazon Web Services Marketplace.  You must specify either the algorithm name to the AlgorithmName parameter or the image URI of the algorithm container to the TrainingImage parameter. Note that the AlgorithmName parameter is mutually exclusive with the TrainingImage parameter. If you specify a value for the AlgorithmName parameter, you can't specify a value for TrainingImage, and vice versa. If you specify values for both parameters, the training job might break; if you don't specify any value for both parameters, the training job might raise a null error.
    training_input_mode
    metric_definitions: A list of metric definition objects. Each object specifies the metric name and regular expressions used to parse algorithm logs. SageMaker publishes each metric to Amazon CloudWatch.
    enable_sage_maker_metrics_time_series: To generate and save time-series metrics during training, set to true. The default is false and time-series metrics aren't generated except in the following cases:   You use one of the SageMaker built-in algorithms   You use one of the following Prebuilt SageMaker Docker Images:   Tensorflow (version &gt;= 1.15)   MXNet (version &gt;= 1.6)   PyTorch (version &gt;= 1.3)     You specify at least one MetricDefinition
    container_entrypoint: The entrypoint script for a Docker container used to run a training job. This script takes precedence over the default train processing instructions. See How Amazon SageMaker Runs Your Training Image for more information.
    container_arguments: The arguments for a container used to run a training job. See How Amazon SageMaker Runs Your Training Image for additional information.
    training_image_config: The configuration to use an image from a private Docker registry for a training job.
    """

    training_input_mode: StrPipeVar
    training_image: Optional[StrPipeVar] = Unassigned()
    algorithm_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    metric_definitions: Optional[List[MetricDefinition]] = Unassigned()
    enable_sage_maker_metrics_time_series: Optional[bool] = Unassigned()
    container_entrypoint: Optional[List[StrPipeVar]] = Unassigned()
    container_arguments: Optional[List[StrPipeVar]] = Unassigned()
    training_image_config: Optional[TrainingImageConfig] = Unassigned()


class AlgorithmStatusItem(Base):
    """
    AlgorithmStatusItem
      Represents the overall status of an algorithm.

    Attributes
    ----------------------
    name: The name of the algorithm for which the overall status is being reported.
    status: The current status.
    failure_reason: if the overall status is Failed, the reason for the failure.
    """

    name: StrPipeVar
    status: StrPipeVar
    failure_reason: Optional[StrPipeVar] = Unassigned()


class AlgorithmStatusDetails(Base):
    """
    AlgorithmStatusDetails
      Specifies the validation and image scan statuses of the algorithm.

    Attributes
    ----------------------
    validation_statuses: The status of algorithm validation.
    image_scan_statuses: The status of the scan of the algorithm's Docker image container.
    """

    validation_statuses: Optional[List[AlgorithmStatusItem]] = Unassigned()
    image_scan_statuses: Optional[List[AlgorithmStatusItem]] = Unassigned()


class AlgorithmSummary(Base):
    """
    AlgorithmSummary
      Provides summary information about an algorithm.

    Attributes
    ----------------------
    algorithm_name: The name of the algorithm that is described by the summary.
    algorithm_arn: The Amazon Resource Name (ARN) of the algorithm.
    algorithm_description: A brief description of the algorithm.
    creation_time: A timestamp that shows when the algorithm was created.
    algorithm_status: The overall status of the algorithm.
    """

    algorithm_name: Union[StrPipeVar, object]
    algorithm_arn: StrPipeVar
    creation_time: datetime.datetime
    algorithm_status: StrPipeVar
    algorithm_description: Optional[StrPipeVar] = Unassigned()


class HubAccessConfig(Base):
    """
    HubAccessConfig
      The configuration for a private hub model reference that points to a public SageMaker JumpStart model. For more information about private hubs, see Private curated hubs for foundation model access control in JumpStart.

    Attributes
    ----------------------
    hub_content_arn: The ARN of your private model hub content. This should be a ModelReference resource type that points to a SageMaker JumpStart public hub model.
    """

    hub_content_arn: StrPipeVar


class S3DataSource(Base):
    """
    S3DataSource
      Describes the S3 data source. Your input bucket must be in the same Amazon Web Services region as your training job.

    Attributes
    ----------------------
    s3_data_type: If you choose S3Prefix, S3Uri identifies a key name prefix. SageMaker uses all objects that match the specified key name prefix for model training.  If you choose ManifestFile, S3Uri identifies an object that is a manifest file containing a list of object keys that you want SageMaker to use for model training.  If you choose AugmentedManifestFile, S3Uri identifies an object that is an augmented manifest file in JSON lines format. This file contains the data you want to use for model training. AugmentedManifestFile can only be used if the Channel's input mode is Pipe. If you choose Converse, S3Uri identifies an Amazon S3 location that contains data formatted according to Converse format. This format structures conversational messages with specific roles and content types used for training and fine-tuning foundational models.
    s3_uri: Depending on the value specified for the S3DataType, identifies either a key name prefix or a manifest. For example:     A key name prefix might look like this: s3://bucketname/exampleprefix/     A manifest might look like this: s3://bucketname/example.manifest   A manifest is an S3 object which is a JSON file consisting of an array of elements. The first element is a prefix which is followed by one or more suffixes. SageMaker appends the suffix elements to the prefix to get a full set of S3Uri. Note that the prefix must be a valid non-empty S3Uri that precludes users from specifying a manifest whose individual S3Uri is sourced from different S3 buckets.  The following code example shows a valid manifest format:   [ {"prefix": "s3://customer_bucket/some/prefix/"},    "relative/path/to/custdata-1",    "relative/path/custdata-2",    ...    "relative/path/custdata-N"   ]   This JSON is equivalent to the following S3Uri list:  s3://customer_bucket/some/prefix/relative/path/to/custdata-1   s3://customer_bucket/some/prefix/relative/path/custdata-2   ...   s3://customer_bucket/some/prefix/relative/path/custdata-N  The complete set of S3Uri in this manifest is the input data for the channel for this data source. The object that each S3Uri points to must be readable by the IAM role that SageMaker uses to perform tasks on your behalf.    Your input bucket must be located in same Amazon Web Services region as your training job.
    s3_data_distribution_type: If you want SageMaker to replicate the entire dataset on each ML compute instance that is launched for model training, specify FullyReplicated.  If you want SageMaker to replicate a subset of data on each ML compute instance that is launched for model training, specify ShardedByS3Key. If there are n ML compute instances launched for a training job, each instance gets approximately 1/n of the number of S3 objects. In this case, model training on each machine uses only the subset of training data.  Don't choose more ML compute instances for training than available S3 objects. If you do, some nodes won't get any data and you will pay for nodes that aren't getting any training data. This applies in both File and Pipe modes. Keep this in mind when developing algorithms.  In distributed training, where you use multiple ML compute EC2 instances, you might choose ShardedByS3Key. If the algorithm requires copying training data to the ML storage volume (when TrainingInputMode is set to File), this copies 1/n of the number of objects.
    attribute_names: A list of one or more attribute names to use that are found in a specified augmented manifest file.
    instance_group_names: A list of names of instance groups that get data from the S3 data source.
    model_access_config
    hub_access_config: The configuration for a private hub model reference that points to a SageMaker JumpStart public hub model.
    """

    s3_data_type: StrPipeVar
    s3_uri: StrPipeVar
    s3_data_distribution_type: Optional[StrPipeVar] = Unassigned()
    attribute_names: Optional[List[StrPipeVar]] = Unassigned()
    instance_group_names: Optional[List[StrPipeVar]] = Unassigned()
    model_access_config: Optional[ModelAccessConfig] = Unassigned()
    hub_access_config: Optional[HubAccessConfig] = Unassigned()


class FileSystemDataSource(Base):
    """
    FileSystemDataSource
      Specifies a file system data source for a channel.

    Attributes
    ----------------------
    file_system_id: The file system id.
    file_system_access_mode: The access mode of the mount of the directory associated with the channel. A directory can be mounted either in ro (read-only) or rw (read-write) mode.
    file_system_type: The file system type.
    directory_path: The full path to the directory to associate with the channel.
    """

    file_system_id: StrPipeVar
    file_system_access_mode: StrPipeVar
    file_system_type: StrPipeVar
    directory_path: StrPipeVar


class DatasetSource(Base):
    """
    DatasetSource

    Attributes
    ----------------------
    dataset_arn
    """

    dataset_arn: StrPipeVar


class DataSource(Base):
    """
    DataSource
      Describes the location of the channel data.

    Attributes
    ----------------------
    s3_data_source: The S3 location of the data source that is associated with a channel.
    file_system_data_source: The file system that is associated with a channel.
    dataset_source
    """

    s3_data_source: Optional[S3DataSource] = Unassigned()
    file_system_data_source: Optional[FileSystemDataSource] = Unassigned()
    dataset_source: Optional[DatasetSource] = Unassigned()


class ShuffleConfig(Base):
    """
    ShuffleConfig
      A configuration for a shuffle option for input data in a channel. If you use S3Prefix for S3DataType, the results of the S3 key prefix matches are shuffled. If you use ManifestFile, the order of the S3 object references in the ManifestFile is shuffled. If you use AugmentedManifestFile, the order of the JSON lines in the AugmentedManifestFile is shuffled. The shuffling order is determined using the Seed value. For Pipe input mode, when ShuffleConfig is specified shuffling is done at the start of every epoch. With large datasets, this ensures that the order of the training data is different for each epoch, and it helps reduce bias and possible overfitting. In a multi-node training job when ShuffleConfig is combined with S3DataDistributionType of ShardedByS3Key, the data is shuffled across nodes so that the content sent to a particular node on the first epoch might be sent to a different node on the second epoch.

    Attributes
    ----------------------
    seed: Determines the shuffling order in ShuffleConfig value.
    """

    seed: int


class Channel(Base):
    """
    Channel
      A channel is a named input source that training algorithms can consume.

    Attributes
    ----------------------
    channel_name: The name of the channel.
    data_source: The location of the channel data.
    content_type: The MIME type of the data.
    compression_type: If training data is compressed, the compression type. The default value is None. CompressionType is used only in Pipe input mode. In File mode, leave this field unset or set it to None.
    record_wrapper_type:  Specify RecordIO as the value when input data is in raw format but the training algorithm requires the RecordIO format. In this case, SageMaker wraps each individual S3 object in a RecordIO record. If the input data is already in RecordIO format, you don't need to set this attribute. For more information, see Create a Dataset Using RecordIO.  In File mode, leave this field unset or set it to None.
    input_mode: (Optional) The input mode to use for the data channel in a training job. If you don't set a value for InputMode, SageMaker uses the value set for TrainingInputMode. Use this parameter to override the TrainingInputMode setting in a AlgorithmSpecification request when you have a channel that needs a different input mode from the training job's general setting. To download the data from Amazon Simple Storage Service (Amazon S3) to the provisioned ML storage volume, and mount the directory to a Docker volume, use File input mode. To stream data directly from Amazon S3 to the container, choose Pipe input mode. To use a model for incremental training, choose File input model.
    shuffle_config: A configuration for a shuffle option for input data in a channel. If you use S3Prefix for S3DataType, this shuffles the results of the S3 key prefix matches. If you use ManifestFile, the order of the S3 object references in the ManifestFile is shuffled. If you use AugmentedManifestFile, the order of the JSON lines in the AugmentedManifestFile is shuffled. The shuffling order is determined using the Seed value. For Pipe input mode, shuffling is done at the start of every epoch. With large datasets this ensures that the order of the training data is different for each epoch, it helps reduce bias and possible overfitting. In a multi-node training job when ShuffleConfig is combined with S3DataDistributionType of ShardedByS3Key, the data is shuffled across nodes so that the content sent to a particular node on the first epoch might be sent to a different node on the second epoch.
    enable_ffm
    """

    channel_name: StrPipeVar
    data_source: DataSource
    content_type: Optional[StrPipeVar] = Unassigned()
    compression_type: Optional[StrPipeVar] = Unassigned()
    record_wrapper_type: Optional[StrPipeVar] = Unassigned()
    input_mode: Optional[StrPipeVar] = Unassigned()
    shuffle_config: Optional[ShuffleConfig] = Unassigned()
    enable_ffm: Optional[bool] = Unassigned()


class OutputChannel(Base):
    """
    OutputChannel

    Attributes
    ----------------------
    channel_name
    local_path
    s3_output_path
    continuous_upload
    kms_key_id
    kms_encryption_context
    """

    channel_name: StrPipeVar
    s3_output_path: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()
    continuous_upload: Optional[bool] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    kms_encryption_context: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class OutputDataConfig(Base):
    """
    OutputDataConfig
      Provides information about how to store model training results (model artifacts).

    Attributes
    ----------------------
    kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that SageMaker uses to encrypt the model artifacts at rest using Amazon S3 server-side encryption. The KmsKeyId can be any of the following formats:    // KMS Key ID  "1234abcd-12ab-34cd-56ef-1234567890ab"    // Amazon Resource Name (ARN) of a KMS Key  "arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"    // KMS Key Alias  "alias/ExampleAlias"    // Amazon Resource Name (ARN) of a KMS Key Alias  "arn:aws:kms:us-west-2:111122223333:alias/ExampleAlias"    If you use a KMS key ID or an alias of your KMS key, the SageMaker execution role must include permissions to call kms:Encrypt. If you don't provide a KMS key ID, SageMaker uses the default KMS key for Amazon S3 for your role's account. For more information, see KMS-Managed Encryption Keys in the Amazon Simple Storage Service Developer Guide. If the output data is stored in Amazon S3 Express One Zone, it is encrypted with server-side encryption with Amazon S3 managed keys (SSE-S3). KMS key is not supported for Amazon S3 Express One Zone The KMS key policy must grant permission to the IAM role that you specify in your CreateTrainingJob, CreateTransformJob, or CreateHyperParameterTuningJob requests. For more information, see Using Key Policies in Amazon Web Services KMS in the Amazon Web Services Key Management Service Developer Guide.
    s3_output_path: Identifies the S3 path where you want SageMaker to store the model artifacts. For example, s3://bucket-name/key-name-prefix.
    compression_type: The model output compression type. Select None to output an uncompressed model, recommended for large model outputs. Defaults to gzip.
    remove_job_name_from_s3_output_path
    disable_model_upload
    channels
    """

    s3_output_path: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    compression_type: Optional[StrPipeVar] = Unassigned()
    remove_job_name_from_s3_output_path: Optional[bool] = Unassigned()
    disable_model_upload: Optional[bool] = Unassigned()
    channels: Optional[List[OutputChannel]] = Unassigned()


class InstanceGroup(Base):
    """
    InstanceGroup
      Defines an instance group for heterogeneous cluster training. When requesting a training job using the CreateTrainingJob API, you can configure multiple instance groups .

    Attributes
    ----------------------
    instance_type: Specifies the instance type of the instance group.
    instance_count: Specifies the number of instances of the instance group.
    instance_group_name: Specifies the name of the instance group.
    """

    instance_type: StrPipeVar
    instance_count: int
    instance_group_name: StrPipeVar


class CapacitySchedule(Base):
    """
    CapacitySchedule

    Attributes
    ----------------------
    capacity_schedule_arn
    """

    capacity_schedule_arn: StrPipeVar


class CapacitySchedulesConfig(Base):
    """
    CapacitySchedulesConfig

    Attributes
    ----------------------
    capacity_fallback_strategy
    capacity_schedules
    """

    capacity_schedules: List[CapacitySchedule]
    capacity_fallback_strategy: Optional[StrPipeVar] = Unassigned()


class PlacementSpecification(Base):
    """
    PlacementSpecification
      Specifies how instances should be placed on a specific UltraServer.

    Attributes
    ----------------------
    ultra_server_id: The unique identifier of the UltraServer where instances should be placed.
    instance_count: The number of ML compute instances required to be placed together on the same UltraServer. Minimum value of 1.
    """

    instance_count: int
    ultra_server_id: Optional[StrPipeVar] = Unassigned()


class InstancePlacementConfig(Base):
    """
    InstancePlacementConfig
      Configuration for how instances are placed and allocated within UltraServers. This is only applicable for UltraServer capacity.

    Attributes
    ----------------------
    enable_multiple_jobs: If set to true, allows multiple jobs to share the same UltraServer instances. If set to false, ensures this job's instances are placed on an UltraServer exclusively, with no other jobs sharing the same UltraServer. Default is false.
    placement_specifications: A list of specifications for how instances should be placed on specific UltraServers. Maximum of 10 items is supported.
    """

    enable_multiple_jobs: Optional[bool] = Unassigned()
    placement_specifications: Optional[List[PlacementSpecification]] = Unassigned()


class ResourceConfig(Base):
    """
    ResourceConfig
      Describes the resources, including machine learning (ML) compute instances and ML storage volumes, to use for model training.

    Attributes
    ----------------------
    instance_type: The ML compute instance type.
    instance_count: The number of ML compute instances to use. For distributed training, provide a value greater than 1.
    volume_size_in_gb: The size of the ML storage volume that you want to provision.  ML storage volumes store model artifacts and incremental states. Training algorithms might also use the ML storage volume for scratch space. If you want to store the training data in the ML storage volume, choose File as the TrainingInputMode in the algorithm specification.  When using an ML instance with NVMe SSD volumes, SageMaker doesn't provision Amazon EBS General Purpose SSD (gp2) storage. Available storage is fixed to the NVMe-type instance's storage capacity. SageMaker configures storage paths for training datasets, checkpoints, model artifacts, and outputs to use the entire capacity of the instance storage. For example, ML instance families with the NVMe-type instance storage include ml.p4d, ml.g4dn, and ml.g5.  When using an ML instance with the EBS-only storage option and without instance storage, you must define the size of EBS volume through VolumeSizeInGB in the ResourceConfig API. For example, ML instance families that use EBS volumes include ml.c5 and ml.p2.  To look up instance types and their instance storage types and volumes, see Amazon EC2 Instance Types. To find the default local paths defined by the SageMaker training platform, see Amazon SageMaker Training Storage Folders for Training Datasets, Checkpoints, Model Artifacts, and Outputs.
    volume_kms_key_id: The Amazon Web Services KMS key that SageMaker uses to encrypt data on the storage volume attached to the ML compute instance(s) that run the training job.  Certain Nitro-based instances include local storage, dependent on the instance type. Local storage volumes are encrypted using a hardware module on the instance. You can't request a VolumeKmsKeyId when using an instance type with local storage. For a list of instance types that support local instance storage, see Instance Store Volumes. For more information about local instance storage encryption, see SSD Instance Store Volumes.  The VolumeKmsKeyId can be in any of the following formats:   // KMS Key ID  "1234abcd-12ab-34cd-56ef-1234567890ab"    // Amazon Resource Name (ARN) of a KMS Key  "arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"
    keep_alive_period_in_seconds: The duration of time in seconds to retain configured resources in a warm pool for subsequent training jobs.
    capacity_reservation_ids
    instance_groups: The configuration of a heterogeneous cluster in JSON format.
    capacity_schedules_config
    training_plan_arn: The Amazon Resource Name (ARN); of the training plan to use for this resource configuration.
    instance_placement_config: Configuration for how training job instances are placed and allocated within UltraServers. Only applicable for UltraServer capacity.
    """

    instance_type: Optional[StrPipeVar] = Unassigned()
    instance_count: Optional[int] = Unassigned()
    volume_size_in_gb: Optional[int] = Unassigned()
    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()
    keep_alive_period_in_seconds: Optional[int] = Unassigned()
    capacity_reservation_ids: Optional[List[StrPipeVar]] = Unassigned()
    instance_groups: Optional[List[InstanceGroup]] = Unassigned()
    capacity_schedules_config: Optional[CapacitySchedulesConfig] = Unassigned()
    training_plan_arn: Optional[StrPipeVar] = Unassigned()
    instance_placement_config: Optional[InstancePlacementConfig] = Unassigned()


class StoppingCondition(Base):
    """
    StoppingCondition
      Specifies a limit to how long a job can run. When the job reaches the time limit, SageMaker ends the job. Use this API to cap costs. To stop a training job, SageMaker sends the algorithm the SIGTERM signal, which delays job termination for 120 seconds. Algorithms can use this 120-second window to save the model artifacts, so the results of training are not lost.  The training algorithms provided by SageMaker automatically save the intermediate results of a model training job when possible. This attempt to save artifacts is only a best effort case as model might not be in a state from which it can be saved. For example, if training has just started, the model might not be ready to save. When saved, this intermediate data is a valid model artifact. You can use it to create a model with CreateModel.  The Neural Topic Model (NTM) currently does not support saving intermediate model artifacts. When training NTMs, make sure that the maximum runtime is sufficient for the training job to complete.

    Attributes
    ----------------------
    max_runtime_in_seconds: The maximum length of time, in seconds, that a training or compilation job can run before it is stopped. For compilation jobs, if the job does not complete during this time, a TimeOut error is generated. We recommend starting with 900 seconds and increasing as necessary based on your model. For all other jobs, if the job does not complete during this time, SageMaker ends the job. When RetryStrategy is specified in the job request, MaxRuntimeInSeconds specifies the maximum time for all of the attempts in total, not each individual attempt. The default value is 1 day. The maximum value is 28 days. The maximum time that a TrainingJob can run in total, including any time spent publishing metrics or archiving and uploading models after it has been stopped, is 30 days.
    max_wait_time_in_seconds: The maximum length of time, in seconds, that a managed Spot training job has to complete. It is the amount of time spent waiting for Spot capacity plus the amount of time the job can run. It must be equal to or greater than MaxRuntimeInSeconds. If the job does not complete during this time, SageMaker ends the job. When RetryStrategy is specified in the job request, MaxWaitTimeInSeconds specifies the maximum time for all of the attempts in total, not each individual attempt.
    max_pending_time_in_seconds: The maximum length of time, in seconds, that a training or compilation job can be pending before it is stopped.  When working with training jobs that use capacity from training plans, not all Pending job states count against the MaxPendingTimeInSeconds limit. The following scenarios do not increment the MaxPendingTimeInSeconds counter:   The plan is in a Scheduled state: Jobs queued (in Pending status) before a plan's start date (waiting for scheduled start time)   Between capacity reservations: Jobs temporarily back to Pending status between two capacity reservation periods    MaxPendingTimeInSeconds only increments when jobs are actively waiting for capacity in an Active plan.
    """

    max_runtime_in_seconds: Optional[int] = Unassigned()
    max_wait_time_in_seconds: Optional[int] = Unassigned()
    max_pending_time_in_seconds: Optional[int] = Unassigned()


class TrainingJobDefinition(Base):
    """
    TrainingJobDefinition
      Defines the input needed to run a training job using the algorithm.

    Attributes
    ----------------------
    training_input_mode
    hyper_parameters: The hyperparameters used for the training job.
    input_data_config: An array of Channel objects, each of which specifies an input source.
    output_data_config: the path to the S3 bucket where you want to store model artifacts. SageMaker creates subfolders for the artifacts.
    resource_config: The resources, including the ML compute instances and ML storage volumes, to use for model training.
    stopping_condition: Specifies a limit to how long a model training job can run. It also specifies how long a managed Spot training job has to complete. When the job reaches the time limit, SageMaker ends the training job. Use this API to cap model training costs. To stop a job, SageMaker sends the algorithm the SIGTERM signal, which delays job termination for 120 seconds. Algorithms can use this 120-second window to save the model artifacts.
    """

    training_input_mode: StrPipeVar
    input_data_config: List[Channel]
    output_data_config: OutputDataConfig
    resource_config: ResourceConfig
    stopping_condition: StoppingCondition
    hyper_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class TransformS3DataSource(Base):
    """
    TransformS3DataSource
      Describes the S3 data source.

    Attributes
    ----------------------
    s3_data_type: If you choose S3Prefix, S3Uri identifies a key name prefix. Amazon SageMaker uses all objects with the specified key name prefix for batch transform.  If you choose ManifestFile, S3Uri identifies an object that is a manifest file containing a list of object keys that you want Amazon SageMaker to use for batch transform.  The following values are compatible: ManifestFile, S3Prefix  The following value is not compatible: AugmentedManifestFile
    s3_uri: Depending on the value specified for the S3DataType, identifies either a key name prefix or a manifest. For example:    A key name prefix might look like this: s3://bucketname/exampleprefix/.     A manifest might look like this: s3://bucketname/example.manifest   The manifest is an S3 object which is a JSON file with the following format:   [ {"prefix": "s3://customer_bucket/some/prefix/"},   "relative/path/to/custdata-1",   "relative/path/custdata-2",   ...   "relative/path/custdata-N"   ]   The preceding JSON matches the following S3Uris:   s3://customer_bucket/some/prefix/relative/path/to/custdata-1   s3://customer_bucket/some/prefix/relative/path/custdata-2   ...   s3://customer_bucket/some/prefix/relative/path/custdata-N   The complete set of S3Uris in this manifest constitutes the input data for the channel for this datasource. The object that each S3Uris points to must be readable by the IAM role that Amazon SageMaker uses to perform tasks on your behalf.
    """

    s3_data_type: StrPipeVar
    s3_uri: StrPipeVar


class TransformDataSource(Base):
    """
    TransformDataSource
      Describes the location of the channel data.

    Attributes
    ----------------------
    s3_data_source: The S3 location of the data source that is associated with a channel.
    """

    s3_data_source: TransformS3DataSource


class TransformInput(Base):
    """
    TransformInput
      Describes the input source of a transform job and the way the transform job consumes it.

    Attributes
    ----------------------
    data_source: Describes the location of the channel data, which is, the S3 location of the input data that the model can consume.
    content_type: The multipurpose internet mail extension (MIME) type of the data. Amazon SageMaker uses the MIME type with each http call to transfer data to the transform job.
    compression_type: If your transform data is compressed, specify the compression type. Amazon SageMaker automatically decompresses the data for the transform job accordingly. The default value is None.
    split_type: The method to use to split the transform job's data files into smaller batches. Splitting is necessary when the total size of each object is too large to fit in a single request. You can also use data splitting to improve performance by processing multiple concurrent mini-batches. The default value for SplitType is None, which indicates that input data files are not split, and request payloads contain the entire contents of an input object. Set the value of this parameter to Line to split records on a newline character boundary. SplitType also supports a number of record-oriented binary data formats. Currently, the supported record formats are:   RecordIO   TFRecord   When splitting is enabled, the size of a mini-batch depends on the values of the BatchStrategy and MaxPayloadInMB parameters. When the value of BatchStrategy is MultiRecord, Amazon SageMaker sends the maximum number of records in each request, up to the MaxPayloadInMB limit. If the value of BatchStrategy is SingleRecord, Amazon SageMaker sends individual records in each request.  Some data formats represent a record as a binary payload wrapped with extra padding bytes. When splitting is applied to a binary data format, padding is removed if the value of BatchStrategy is set to SingleRecord. Padding is not removed if the value of BatchStrategy is set to MultiRecord. For more information about RecordIO, see Create a Dataset Using RecordIO in the MXNet documentation. For more information about TFRecord, see Consuming TFRecord data in the TensorFlow documentation.
    """

    data_source: TransformDataSource
    content_type: Optional[StrPipeVar] = Unassigned()
    compression_type: Optional[StrPipeVar] = Unassigned()
    split_type: Optional[StrPipeVar] = Unassigned()


class TransformOutput(Base):
    """
    TransformOutput
      Describes the results of a transform job.

    Attributes
    ----------------------
    s3_output_path: The Amazon S3 path where you want Amazon SageMaker to store the results of the transform job. For example, s3://bucket-name/key-name-prefix. For every S3 object used as input for the transform job, batch transform stores the transformed data with an .out suffix in a corresponding subfolder in the location in the output prefix. For example, for the input data stored at s3://bucket-name/input-name-prefix/dataset01/data.csv, batch transform stores the transformed data at s3://bucket-name/output-name-prefix/input-name-prefix/data.csv.out. Batch transform doesn't upload partially processed objects. For an input S3 object that contains multiple records, it creates an .out file only if the transform job succeeds on the entire file. When the input contains multiple S3 objects, the batch transform job processes the listed S3 objects and uploads only the output for successfully processed objects. If any object fails in the transform job batch transform marks the job as failed to prompt investigation.
    accept: The MIME type used to specify the output data. Amazon SageMaker uses the MIME type with each http call to transfer data from the transform job.
    assemble_with: Defines how to assemble the results of the transform job as a single S3 object. Choose a format that is most convenient to you. To concatenate the results in binary format, specify None. To add a newline character at the end of every transformed record, specify Line.
    kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt the model artifacts at rest using Amazon S3 server-side encryption. The KmsKeyId can be any of the following formats:    Key ID: 1234abcd-12ab-34cd-56ef-1234567890ab    Key ARN: arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab    Alias name: alias/ExampleAlias    Alias name ARN: arn:aws:kms:us-west-2:111122223333:alias/ExampleAlias    If you don't provide a KMS key ID, Amazon SageMaker uses the default KMS key for Amazon S3 for your role's account. For more information, see KMS-Managed Encryption Keys in the Amazon Simple Storage Service Developer Guide.  The KMS key policy must grant permission to the IAM role that you specify in your CreateModel request. For more information, see Using Key Policies in Amazon Web Services KMS in the Amazon Web Services Key Management Service Developer Guide.
    output_prefix
    output_suffix
    """

    s3_output_path: StrPipeVar
    accept: Optional[StrPipeVar] = Unassigned()
    assemble_with: Optional[StrPipeVar] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    output_prefix: Optional[StrPipeVar] = Unassigned()
    output_suffix: Optional[StrPipeVar] = Unassigned()


class TransformResources(Base):
    """
    TransformResources
      Describes the resources, including ML instance types and ML instance count, to use for transform job.

    Attributes
    ----------------------
    instance_type: The ML compute instance type for the transform job. If you are using built-in algorithms to transform moderately sized datasets, we recommend using ml.m4.xlarge or ml.m5.largeinstance types.
    instance_count: The number of ML compute instances to use in the transform job. The default value is 1, and the maximum is 100. For distributed transform jobs, specify a value greater than 1.
    volume_kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt model data on the storage volume attached to the ML compute instance(s) that run the batch transform job.  Certain Nitro-based instances include local storage, dependent on the instance type. Local storage volumes are encrypted using a hardware module on the instance. You can't request a VolumeKmsKeyId when using an instance type with local storage. For a list of instance types that support local instance storage, see Instance Store Volumes. For more information about local instance storage encryption, see SSD Instance Store Volumes.   The VolumeKmsKeyId can be any of the following formats:   Key ID: 1234abcd-12ab-34cd-56ef-1234567890ab    Key ARN: arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab    Alias name: alias/ExampleAlias    Alias name ARN: arn:aws:kms:us-west-2:111122223333:alias/ExampleAlias
    transform_ami_version: Specifies an option from a collection of preconfigured Amazon Machine Image (AMI) images. Each image is configured by Amazon Web Services with a set of software and driver versions.  al2-ami-sagemaker-batch-gpu-470    Accelerator: GPU   NVIDIA driver version: 470    al2-ami-sagemaker-batch-gpu-535    Accelerator: GPU   NVIDIA driver version: 535
    """

    instance_type: StrPipeVar
    instance_count: int
    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()
    transform_ami_version: Optional[StrPipeVar] = Unassigned()


class TransformJobDefinition(Base):
    """
    TransformJobDefinition
      Defines the input needed to run a transform job using the inference specification specified in the algorithm.

    Attributes
    ----------------------
    max_concurrent_transforms: The maximum number of parallel requests that can be sent to each instance in a transform job. The default value is 1.
    max_payload_in_mb: The maximum payload size allowed, in MB. A payload is the data portion of a record (without metadata).
    batch_strategy: A string that determines the number of records included in a single mini-batch.  SingleRecord means only one record is used per mini-batch. MultiRecord means a mini-batch is set to contain as many records that can fit within the MaxPayloadInMB limit.
    environment: The environment variables to set in the Docker container. We support up to 16 key and values entries in the map.
    transform_input: A description of the input source and the way the transform job consumes it.
    transform_output: Identifies the Amazon S3 location where you want Amazon SageMaker to save the results from the transform job.
    transform_resources: Identifies the ML compute instances for the transform job.
    """

    transform_input: TransformInput
    transform_output: TransformOutput
    transform_resources: TransformResources
    max_concurrent_transforms: Optional[int] = Unassigned()
    max_payload_in_mb: Optional[int] = Unassigned()
    batch_strategy: Optional[StrPipeVar] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class AlgorithmValidationProfile(Base):
    """
    AlgorithmValidationProfile
      Defines a training job and a batch transform job that SageMaker runs to validate your algorithm. The data provided in the validation profile is made available to your buyers on Amazon Web Services Marketplace.

    Attributes
    ----------------------
    profile_name: The name of the profile for the algorithm. The name must have 1 to 63 characters. Valid characters are a-z, A-Z, 0-9, and - (hyphen).
    training_job_definition: The TrainingJobDefinition object that describes the training job that SageMaker runs to validate your algorithm.
    transform_job_definition: The TransformJobDefinition object that describes the transform job that SageMaker runs to validate your algorithm.
    """

    profile_name: StrPipeVar
    training_job_definition: TrainingJobDefinition
    transform_job_definition: Optional[TransformJobDefinition] = Unassigned()


class AlgorithmValidationSpecification(Base):
    """
    AlgorithmValidationSpecification
      Specifies configurations for one or more training jobs that SageMaker runs to test the algorithm.

    Attributes
    ----------------------
    validation_role: The IAM roles that SageMaker uses to run the training jobs.
    validation_profiles: An array of AlgorithmValidationProfile objects, each of which specifies a training job and batch transform job that SageMaker runs to validate your algorithm.
    """

    validation_role: StrPipeVar
    validation_profiles: List[AlgorithmValidationProfile]


class AmazonQSettings(Base):
    """
    AmazonQSettings
      A collection of settings that configure the Amazon Q experience within the domain.

    Attributes
    ----------------------
    status: Whether Amazon Q has been enabled within the domain.
    q_profile_arn: The ARN of the Amazon Q profile used within the domain.
    """

    status: Optional[StrPipeVar] = Unassigned()
    q_profile_arn: Optional[StrPipeVar] = Unassigned()


class AnnotationConsolidationConfig(Base):
    """
    AnnotationConsolidationConfig
      Configures how labels are consolidated across human workers and processes output data.

    Attributes
    ----------------------
    annotation_consolidation_lambda_arn: The Amazon Resource Name (ARN) of a Lambda function implements the logic for annotation consolidation and to process output data. For built-in task types, use one of the following Amazon SageMaker Ground Truth Lambda function ARNs for AnnotationConsolidationLambdaArn. For custom labeling workflows, see Post-annotation Lambda.  Bounding box - Finds the most similar boxes from different workers based on the Jaccard index of the boxes.    arn:aws:lambda:us-east-1:432418664414:function:ACS-BoundingBox     arn:aws:lambda:us-east-2:266458841044:function:ACS-BoundingBox     arn:aws:lambda:us-west-2:081040173940:function:ACS-BoundingBox     arn:aws:lambda:eu-west-1:568282634449:function:ACS-BoundingBox     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-BoundingBox     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-BoundingBox     arn:aws:lambda:ap-south-1:565803892007:function:ACS-BoundingBox     arn:aws:lambda:eu-central-1:203001061592:function:ACS-BoundingBox     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-BoundingBox     arn:aws:lambda:eu-west-2:487402164563:function:ACS-BoundingBox     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-BoundingBox     arn:aws:lambda:ca-central-1:918755190332:function:ACS-BoundingBox     Image classification - Uses a variant of the Expectation Maximization approach to estimate the true class of an image based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:ACS-ImageMultiClass     arn:aws:lambda:us-east-2:266458841044:function:ACS-ImageMultiClass     arn:aws:lambda:us-west-2:081040173940:function:ACS-ImageMultiClass     arn:aws:lambda:eu-west-1:568282634449:function:ACS-ImageMultiClass     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-ImageMultiClass     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-ImageMultiClass     arn:aws:lambda:ap-south-1:565803892007:function:ACS-ImageMultiClass     arn:aws:lambda:eu-central-1:203001061592:function:ACS-ImageMultiClass     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-ImageMultiClass     arn:aws:lambda:eu-west-2:487402164563:function:ACS-ImageMultiClass     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-ImageMultiClass     arn:aws:lambda:ca-central-1:918755190332:function:ACS-ImageMultiClass     Multi-label image classification - Uses a variant of the Expectation Maximization approach to estimate the true classes of an image based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:us-east-2:266458841044:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:us-west-2:081040173940:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:eu-west-1:568282634449:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:ap-south-1:565803892007:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:eu-central-1:203001061592:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:eu-west-2:487402164563:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-ImageMultiClassMultiLabel     arn:aws:lambda:ca-central-1:918755190332:function:ACS-ImageMultiClassMultiLabel     Semantic segmentation - Treats each pixel in an image as a multi-class classification and treats pixel annotations from workers as "votes" for the correct label.    arn:aws:lambda:us-east-1:432418664414:function:ACS-SemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:ACS-SemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:ACS-SemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:ACS-SemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-SemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-SemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:ACS-SemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:ACS-SemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-SemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:ACS-SemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-SemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:ACS-SemanticSegmentation     Text classification - Uses a variant of the Expectation Maximization approach to estimate the true class of text based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:ACS-TextMultiClass     arn:aws:lambda:us-east-2:266458841044:function:ACS-TextMultiClass     arn:aws:lambda:us-west-2:081040173940:function:ACS-TextMultiClass     arn:aws:lambda:eu-west-1:568282634449:function:ACS-TextMultiClass     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-TextMultiClass     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-TextMultiClass     arn:aws:lambda:ap-south-1:565803892007:function:ACS-TextMultiClass     arn:aws:lambda:eu-central-1:203001061592:function:ACS-TextMultiClass     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-TextMultiClass     arn:aws:lambda:eu-west-2:487402164563:function:ACS-TextMultiClass     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-TextMultiClass     arn:aws:lambda:ca-central-1:918755190332:function:ACS-TextMultiClass     Multi-label text classification - Uses a variant of the Expectation Maximization approach to estimate the true classes of text based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:us-east-2:266458841044:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:us-west-2:081040173940:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:eu-west-1:568282634449:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:ap-south-1:565803892007:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:eu-central-1:203001061592:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:eu-west-2:487402164563:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-TextMultiClassMultiLabel     arn:aws:lambda:ca-central-1:918755190332:function:ACS-TextMultiClassMultiLabel     Named entity recognition - Groups similar selections and calculates aggregate boundaries, resolving to most-assigned label.    arn:aws:lambda:us-east-1:432418664414:function:ACS-NamedEntityRecognition     arn:aws:lambda:us-east-2:266458841044:function:ACS-NamedEntityRecognition     arn:aws:lambda:us-west-2:081040173940:function:ACS-NamedEntityRecognition     arn:aws:lambda:eu-west-1:568282634449:function:ACS-NamedEntityRecognition     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-NamedEntityRecognition     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-NamedEntityRecognition     arn:aws:lambda:ap-south-1:565803892007:function:ACS-NamedEntityRecognition     arn:aws:lambda:eu-central-1:203001061592:function:ACS-NamedEntityRecognition     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-NamedEntityRecognition     arn:aws:lambda:eu-west-2:487402164563:function:ACS-NamedEntityRecognition     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-NamedEntityRecognition     arn:aws:lambda:ca-central-1:918755190332:function:ACS-NamedEntityRecognition     Video Classification - Use this task type when you need workers to classify videos using predefined labels that you specify. Workers are shown videos and are asked to choose one label for each video.    arn:aws:lambda:us-east-1:432418664414:function:ACS-VideoMultiClass     arn:aws:lambda:us-east-2:266458841044:function:ACS-VideoMultiClass     arn:aws:lambda:us-west-2:081040173940:function:ACS-VideoMultiClass     arn:aws:lambda:eu-west-1:568282634449:function:ACS-VideoMultiClass     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-VideoMultiClass     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-VideoMultiClass     arn:aws:lambda:ap-south-1:565803892007:function:ACS-VideoMultiClass     arn:aws:lambda:eu-central-1:203001061592:function:ACS-VideoMultiClass     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-VideoMultiClass     arn:aws:lambda:eu-west-2:487402164563:function:ACS-VideoMultiClass     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-VideoMultiClass     arn:aws:lambda:ca-central-1:918755190332:function:ACS-VideoMultiClass     Video Frame Object Detection - Use this task type to have workers identify and locate objects in a sequence of video frames (images extracted from a video) using bounding boxes. For example, you can use this task to ask workers to identify and localize various objects in a series of video frames, such as cars, bikes, and pedestrians.    arn:aws:lambda:us-east-1:432418664414:function:ACS-VideoObjectDetection     arn:aws:lambda:us-east-2:266458841044:function:ACS-VideoObjectDetection     arn:aws:lambda:us-west-2:081040173940:function:ACS-VideoObjectDetection     arn:aws:lambda:eu-west-1:568282634449:function:ACS-VideoObjectDetection     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-VideoObjectDetection     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-VideoObjectDetection     arn:aws:lambda:ap-south-1:565803892007:function:ACS-VideoObjectDetection     arn:aws:lambda:eu-central-1:203001061592:function:ACS-VideoObjectDetection     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-VideoObjectDetection     arn:aws:lambda:eu-west-2:487402164563:function:ACS-VideoObjectDetection     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-VideoObjectDetection     arn:aws:lambda:ca-central-1:918755190332:function:ACS-VideoObjectDetection     Video Frame Object Tracking - Use this task type to have workers track the movement of objects in a sequence of video frames (images extracted from a video) using bounding boxes. For example, you can use this task to ask workers to track the movement of objects, such as cars, bikes, and pedestrians.     arn:aws:lambda:us-east-1:432418664414:function:ACS-VideoObjectTracking     arn:aws:lambda:us-east-2:266458841044:function:ACS-VideoObjectTracking     arn:aws:lambda:us-west-2:081040173940:function:ACS-VideoObjectTracking     arn:aws:lambda:eu-west-1:568282634449:function:ACS-VideoObjectTracking     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-VideoObjectTracking     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-VideoObjectTracking     arn:aws:lambda:ap-south-1:565803892007:function:ACS-VideoObjectTracking     arn:aws:lambda:eu-central-1:203001061592:function:ACS-VideoObjectTracking     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-VideoObjectTracking     arn:aws:lambda:eu-west-2:487402164563:function:ACS-VideoObjectTracking     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-VideoObjectTracking     arn:aws:lambda:ca-central-1:918755190332:function:ACS-VideoObjectTracking     3D Point Cloud Object Detection - Use this task type when you want workers to classify objects in a 3D point cloud by drawing 3D cuboids around objects. For example, you can use this task type to ask workers to identify different types of objects in a point cloud, such as cars, bikes, and pedestrians.    arn:aws:lambda:us-east-1:432418664414:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:us-east-2:266458841044:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:us-west-2:081040173940:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:eu-west-1:568282634449:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:ap-south-1:565803892007:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:eu-central-1:203001061592:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:eu-west-2:487402164563:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-3DPointCloudObjectDetection     arn:aws:lambda:ca-central-1:918755190332:function:ACS-3DPointCloudObjectDetection     3D Point Cloud Object Tracking - Use this task type when you want workers to draw 3D cuboids around objects that appear in a sequence of 3D point cloud frames. For example, you can use this task type to ask workers to track the movement of vehicles across multiple point cloud frames.     arn:aws:lambda:us-east-1:432418664414:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:us-east-2:266458841044:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:us-west-2:081040173940:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:eu-west-1:568282634449:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:ap-south-1:565803892007:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:eu-central-1:203001061592:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:eu-west-2:487402164563:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-3DPointCloudObjectTracking     arn:aws:lambda:ca-central-1:918755190332:function:ACS-3DPointCloudObjectTracking     3D Point Cloud Semantic Segmentation - Use this task type when you want workers to create a point-level semantic segmentation masks by painting objects in a 3D point cloud using different colors where each color is assigned to one of the classes you specify.    arn:aws:lambda:us-east-1:432418664414:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:ACS-3DPointCloudSemanticSegmentation     Use the following ARNs for Label Verification and Adjustment Jobs  Use label verification and adjustment jobs to review and adjust labels. To learn more, see Verify and Adjust Labels .  Semantic Segmentation Adjustment - Treats each pixel in an image as a multi-class classification and treats pixel adjusted annotations from workers as "votes" for the correct label.    arn:aws:lambda:us-east-1:432418664414:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-AdjustmentSemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:ACS-AdjustmentSemanticSegmentation     Semantic Segmentation Verification - Uses a variant of the Expectation Maximization approach to estimate the true class of verification judgment for semantic segmentation labels based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-VerificationSemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:ACS-VerificationSemanticSegmentation     Bounding Box Adjustment - Finds the most similar boxes from different workers based on the Jaccard index of the adjusted annotations.    arn:aws:lambda:us-east-1:432418664414:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:us-east-2:266458841044:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:us-west-2:081040173940:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:eu-west-1:568282634449:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:ap-south-1:565803892007:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:eu-central-1:203001061592:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:eu-west-2:487402164563:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-AdjustmentBoundingBox     arn:aws:lambda:ca-central-1:918755190332:function:ACS-AdjustmentBoundingBox     Bounding Box Verification - Uses a variant of the Expectation Maximization approach to estimate the true class of verification judgement for bounding box labels based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:ACS-VerificationBoundingBox     arn:aws:lambda:us-east-2:266458841044:function:ACS-VerificationBoundingBox     arn:aws:lambda:us-west-2:081040173940:function:ACS-VerificationBoundingBox     arn:aws:lambda:eu-west-1:568282634449:function:ACS-VerificationBoundingBox     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-VerificationBoundingBox     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-VerificationBoundingBox     arn:aws:lambda:ap-south-1:565803892007:function:ACS-VerificationBoundingBox     arn:aws:lambda:eu-central-1:203001061592:function:ACS-VerificationBoundingBox     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-VerificationBoundingBox     arn:aws:lambda:eu-west-2:487402164563:function:ACS-VerificationBoundingBox     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-VerificationBoundingBox     arn:aws:lambda:ca-central-1:918755190332:function:ACS-VerificationBoundingBox     Video Frame Object Detection Adjustment - Use this task type when you want workers to adjust bounding boxes that workers have added to video frames to classify and localize objects in a sequence of video frames.    arn:aws:lambda:us-east-1:432418664414:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:us-east-2:266458841044:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:us-west-2:081040173940:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:eu-west-1:568282634449:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-south-1:565803892007:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:eu-central-1:203001061592:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:eu-west-2:487402164563:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-AdjustmentVideoObjectDetection     arn:aws:lambda:ca-central-1:918755190332:function:ACS-AdjustmentVideoObjectDetection     Video Frame Object Tracking Adjustment - Use this task type when you want workers to adjust bounding boxes that workers have added to video frames to track object movement across a sequence of video frames.    arn:aws:lambda:us-east-1:432418664414:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:us-east-2:266458841044:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:us-west-2:081040173940:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:eu-west-1:568282634449:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-south-1:565803892007:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:eu-central-1:203001061592:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:eu-west-2:487402164563:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-AdjustmentVideoObjectTracking     arn:aws:lambda:ca-central-1:918755190332:function:ACS-AdjustmentVideoObjectTracking     3D Point Cloud Object Detection Adjustment - Use this task type when you want workers to adjust 3D cuboids around objects in a 3D point cloud.     arn:aws:lambda:us-east-1:432418664414:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:us-east-2:266458841044:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:us-west-2:081040173940:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:eu-west-1:568282634449:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-south-1:565803892007:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:eu-central-1:203001061592:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:eu-west-2:487402164563:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ca-central-1:918755190332:function:ACS-Adjustment3DPointCloudObjectDetection     3D Point Cloud Object Tracking Adjustment - Use this task type when you want workers to adjust 3D cuboids around objects that appear in a sequence of 3D point cloud frames.    arn:aws:lambda:us-east-1:432418664414:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:us-east-2:266458841044:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:us-west-2:081040173940:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:eu-west-1:568282634449:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-south-1:565803892007:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:eu-central-1:203001061592:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:eu-west-2:487402164563:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ca-central-1:918755190332:function:ACS-Adjustment3DPointCloudObjectTracking     3D Point Cloud Semantic Segmentation Adjustment - Use this task type when you want workers to adjust a point-level semantic segmentation masks using a paint tool.    arn:aws:lambda:us-east-1:432418664414:function:ACS-3DPointCloudSemanticSegmentation     arn:aws:lambda:us-east-1:432418664414:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:ACS-Adjustment3DPointCloudSemanticSegmentation     Generative AI/Custom - Direct passthrough of output data without any transformation.    arn:aws:lambda:us-east-1:432418664414:function:ACS-PassThrough     arn:aws:lambda:us-east-2:266458841044:function:ACS-PassThrough     arn:aws:lambda:us-west-2:081040173940:function:ACS-PassThrough     arn:aws:lambda:eu-west-1:568282634449:function:ACS-PassThrough     arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-PassThrough     arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-PassThrough     arn:aws:lambda:ap-south-1:565803892007:function:ACS-PassThrough     arn:aws:lambda:eu-central-1:203001061592:function:ACS-PassThrough     arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-PassThrough     arn:aws:lambda:eu-west-2:487402164563:function:ACS-PassThrough     arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-PassThrough     arn:aws:lambda:ca-central-1:918755190332:function:ACS-PassThrough
    """

    annotation_consolidation_lambda_arn: StrPipeVar


class ResourceSpec(Base):
    """
    ResourceSpec
      Specifies the ARN's of a SageMaker AI image and SageMaker AI image version, and the instance type that the version runs on.  When both SageMakerImageVersionArn and SageMakerImageArn are passed, SageMakerImageVersionArn is used. Any updates to SageMakerImageArn will not take effect if SageMakerImageVersionArn already exists in the ResourceSpec because SageMakerImageVersionArn always takes precedence. To clear the value set for SageMakerImageVersionArn, pass None as the value.

    Attributes
    ----------------------
    environment_arn
    environment_version_arn
    sage_maker_image_arn: The ARN of the SageMaker AI image that the image version belongs to.
    sage_maker_image_version_arn: The ARN of the image version created on the instance. To clear the value set for SageMakerImageVersionArn, pass None as the value.
    sage_maker_image_version_alias: The SageMakerImageVersionAlias of the image to launch with. This value is in SemVer 2.0.0 versioning format.
    instance_type: The instance type that the image version runs on.   JupyterServer apps only support the system value. For KernelGateway apps, the system value is translated to ml.t3.medium. KernelGateway apps also support all other values for available instance types.
    lifecycle_config_arn:  The Amazon Resource Name (ARN) of the Lifecycle Configuration attached to the Resource.
    """

    environment_arn: Optional[StrPipeVar] = Unassigned()
    environment_version_arn: Optional[StrPipeVar] = Unassigned()
    sage_maker_image_arn: Optional[StrPipeVar] = Unassigned()
    sage_maker_image_version_arn: Optional[StrPipeVar] = Unassigned()
    sage_maker_image_version_alias: Optional[StrPipeVar] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    lifecycle_config_arn: Optional[StrPipeVar] = Unassigned()


class Service(Base):
    """
    Service

    Attributes
    ----------------------
    environment
    image_uri
    volumes
    entrypoint
    command
    """

    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    image_uri: Optional[StrPipeVar] = Unassigned()
    volumes: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    entrypoint: Optional[List[StrPipeVar]] = Unassigned()
    command: Optional[List[StrPipeVar]] = Unassigned()


class LocalAppLaunchConfiguration(Base):
    """
    LocalAppLaunchConfiguration

    Attributes
    ----------------------
    parent_app_arn
    services
    """

    parent_app_arn: Optional[StrPipeVar] = Unassigned()
    services: Optional[List[Service]] = Unassigned()


class AppLaunchConfiguration(Base):
    """
    AppLaunchConfiguration

    Attributes
    ----------------------
    local_app_launch_configuration
    """

    local_app_launch_configuration: Optional[LocalAppLaunchConfiguration] = Unassigned()


class App(Base):
    """
    App

    Attributes
    ----------------------
    app_arn
    app_type
    app_name
    domain_id
    user_profile_name
    space_name
    status
    effective_trusted_identity_propagation_status
    recovery_mode
    last_health_check_timestamp
    last_user_activity_timestamp
    creation_time
    restart_time
    failure_reason
    resource_spec
    built_in_lifecycle_config_arn
    app_launch_configuration
    tags
    """

    app_arn: Optional[StrPipeVar] = Unassigned()
    app_type: Optional[StrPipeVar] = Unassigned()
    app_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    domain_id: Optional[StrPipeVar] = Unassigned()
    user_profile_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    space_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    effective_trusted_identity_propagation_status: Optional[StrPipeVar] = Unassigned()
    recovery_mode: Optional[bool] = Unassigned()
    last_health_check_timestamp: Optional[datetime.datetime] = Unassigned()
    last_user_activity_timestamp: Optional[datetime.datetime] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    restart_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    resource_spec: Optional[ResourceSpec] = Unassigned()
    built_in_lifecycle_config_arn: Optional[StrPipeVar] = Unassigned()
    app_launch_configuration: Optional[AppLaunchConfiguration] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class AppDetails(Base):
    """
    AppDetails
      Details about an Amazon SageMaker AI app.

    Attributes
    ----------------------
    domain_id: The domain ID.
    user_profile_name: The user profile name.
    space_name: The name of the space.
    app_type: The type of app.
    app_name: The name of the app.
    status: The status.
    creation_time: The creation time.
    resource_spec
    """

    domain_id: Optional[StrPipeVar] = Unassigned()
    user_profile_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    space_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    app_type: Optional[StrPipeVar] = Unassigned()
    app_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    resource_spec: Optional[ResourceSpec] = Unassigned()


class KernelSpec(Base):
    """
    KernelSpec
      The specification of a Jupyter kernel.

    Attributes
    ----------------------
    name: The name of the Jupyter kernel in the image. This value is case sensitive.
    display_name: The display name of the kernel.
    """

    name: StrPipeVar
    display_name: Optional[StrPipeVar] = Unassigned()


class FileSystemConfig(Base):
    """
    FileSystemConfig
      The Amazon Elastic File System storage configuration for a SageMaker AI image.

    Attributes
    ----------------------
    mount_path: The path within the image to mount the user's EFS home directory. The directory should be empty. If not specified, defaults to /home/sagemaker-user.
    default_uid: The default POSIX user ID (UID). If not specified, defaults to 1000.
    default_gid: The default POSIX group ID (GID). If not specified, defaults to 100.
    """

    mount_path: Optional[StrPipeVar] = Unassigned()
    default_uid: Optional[int] = Unassigned()
    default_gid: Optional[int] = Unassigned()


class KernelGatewayImageConfig(Base):
    """
    KernelGatewayImageConfig
      The configuration for the file system and kernels in a SageMaker AI image running as a KernelGateway app.

    Attributes
    ----------------------
    kernel_specs: The specification of the Jupyter kernels in the image.
    file_system_config: The Amazon Elastic File System storage configuration for a SageMaker AI image.
    """

    kernel_specs: List[KernelSpec]
    file_system_config: Optional[FileSystemConfig] = Unassigned()


class ContainerConfig(Base):
    """
    ContainerConfig
      The configuration used to run the application image container.

    Attributes
    ----------------------
    container_arguments: The arguments for the container when you're running the application.
    container_entrypoint: The entrypoint used to run the application in the container.
    container_environment_variables: The environment variables to set in the container
    """

    container_arguments: Optional[List[StrPipeVar]] = Unassigned()
    container_entrypoint: Optional[List[StrPipeVar]] = Unassigned()
    container_environment_variables: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class SaviturAppImageConfig(Base):
    """
    SaviturAppImageConfig

    Attributes
    ----------------------
    file_system_config
    container_config
    """

    file_system_config: Optional[FileSystemConfig] = Unassigned()
    container_config: Optional[ContainerConfig] = Unassigned()


class JupyterLabAppImageConfig(Base):
    """
    JupyterLabAppImageConfig
      The configuration for the file system and kernels in a SageMaker AI image running as a JupyterLab app. The FileSystemConfig object is not supported.

    Attributes
    ----------------------
    file_system_config
    container_config
    """

    file_system_config: Optional[FileSystemConfig] = Unassigned()
    container_config: Optional[ContainerConfig] = Unassigned()


class CodeEditorAppImageConfig(Base):
    """
    CodeEditorAppImageConfig
      The configuration for the file system and kernels in a SageMaker image running as a Code Editor app. The FileSystemConfig object is not supported.

    Attributes
    ----------------------
    file_system_config
    container_config
    """

    file_system_config: Optional[FileSystemConfig] = Unassigned()
    container_config: Optional[ContainerConfig] = Unassigned()


class AppImageConfigDetails(Base):
    """
    AppImageConfigDetails
      The configuration for running a SageMaker AI image as a KernelGateway app.

    Attributes
    ----------------------
    app_image_config_arn: The ARN of the AppImageConfig.
    app_image_config_name: The name of the AppImageConfig. Must be unique to your account.
    creation_time: When the AppImageConfig was created.
    last_modified_time: When the AppImageConfig was last modified.
    kernel_gateway_image_config: The configuration for the file system and kernels in the SageMaker AI image.
    savitur_app_image_config
    jupyter_lab_app_image_config: The configuration for the file system and the runtime, such as the environment variables and entry point.
    code_editor_app_image_config: The configuration for the file system and the runtime, such as the environment variables and entry point.
    """

    app_image_config_arn: Optional[StrPipeVar] = Unassigned()
    app_image_config_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    kernel_gateway_image_config: Optional[KernelGatewayImageConfig] = Unassigned()
    savitur_app_image_config: Optional[SaviturAppImageConfig] = Unassigned()
    jupyter_lab_app_image_config: Optional[JupyterLabAppImageConfig] = Unassigned()
    code_editor_app_image_config: Optional[CodeEditorAppImageConfig] = Unassigned()


class IdleSettings(Base):
    """
    IdleSettings
      Settings related to idle shutdown of Studio applications.

    Attributes
    ----------------------
    lifecycle_management: Indicates whether idle shutdown is activated for the application type.
    idle_timeout_in_minutes: The time that SageMaker waits after the application becomes idle before shutting it down.
    min_idle_timeout_in_minutes: The minimum value in minutes that custom idle shutdown can be set to by the user.
    max_idle_timeout_in_minutes: The maximum value in minutes that custom idle shutdown can be set to by the user.
    """

    lifecycle_management: Optional[StrPipeVar] = Unassigned()
    idle_timeout_in_minutes: Optional[int] = Unassigned()
    min_idle_timeout_in_minutes: Optional[int] = Unassigned()
    max_idle_timeout_in_minutes: Optional[int] = Unassigned()


class AppLifecycleManagement(Base):
    """
    AppLifecycleManagement
      Settings that are used to configure and manage the lifecycle of Amazon SageMaker Studio applications.

    Attributes
    ----------------------
    idle_settings: Settings related to idle shutdown of Studio applications.
    """

    idle_settings: Optional[IdleSettings] = Unassigned()


class AppSpecification(Base):
    """
    AppSpecification
      Configuration to run a processing job in a specified container image.

    Attributes
    ----------------------
    image_uri: The container image to be run by the processing job.
    container_entrypoint: The entrypoint for a container used to run a processing job.
    container_arguments: The arguments for a container used to run a processing job.
    """

    image_uri: StrPipeVar
    container_entrypoint: Optional[List[StrPipeVar]] = Unassigned()
    container_arguments: Optional[List[StrPipeVar]] = Unassigned()


class ArtifactSourceType(Base):
    """
    ArtifactSourceType
      The ID and ID type of an artifact source.

    Attributes
    ----------------------
    source_id_type: The type of ID.
    value: The ID.
    """

    source_id_type: StrPipeVar
    value: StrPipeVar


class ArtifactSource(Base):
    """
    ArtifactSource
      A structure describing the source of an artifact.

    Attributes
    ----------------------
    source_uri: The URI of the source.
    source_types: A list of source types.
    """

    source_uri: StrPipeVar
    source_types: Optional[List[ArtifactSourceType]] = Unassigned()


class ArtifactSummary(Base):
    """
    ArtifactSummary
      Lists a summary of the properties of an artifact. An artifact represents a URI addressable object or data. Some examples are a dataset and a model.

    Attributes
    ----------------------
    artifact_arn: The Amazon Resource Name (ARN) of the artifact.
    artifact_name: The name of the artifact.
    source: The source of the artifact.
    artifact_type: The type of the artifact.
    creation_time: When the artifact was created.
    last_modified_time: When the artifact was last modified.
    """

    artifact_arn: Optional[StrPipeVar] = Unassigned()
    artifact_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    source: Optional[ArtifactSource] = Unassigned()
    artifact_type: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class AssociationInfo(Base):
    """
    AssociationInfo

    Attributes
    ----------------------
    source_arn
    destination_arn
    """

    source_arn: StrPipeVar
    destination_arn: StrPipeVar


class AssociationSummary(Base):
    """
    AssociationSummary
      Lists a summary of the properties of an association. An association is an entity that links other lineage or experiment entities. An example would be an association between a training job and a model.

    Attributes
    ----------------------
    source_arn: The ARN of the source.
    destination_arn: The Amazon Resource Name (ARN) of the destination.
    source_type: The source type.
    destination_type: The destination type.
    association_type: The type of the association.
    source_name: The name of the source.
    destination_name: The name of the destination.
    creation_time: When the association was created.
    created_by
    """

    source_arn: Optional[StrPipeVar] = Unassigned()
    destination_arn: Optional[StrPipeVar] = Unassigned()
    source_type: Optional[StrPipeVar] = Unassigned()
    destination_type: Optional[StrPipeVar] = Unassigned()
    association_type: Optional[StrPipeVar] = Unassigned()
    source_name: Optional[StrPipeVar] = Unassigned()
    destination_name: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()


class AsyncInferenceClientConfig(Base):
    """
    AsyncInferenceClientConfig
      Configures the behavior of the client used by SageMaker to interact with the model container during asynchronous inference.

    Attributes
    ----------------------
    max_concurrent_invocations_per_instance: The maximum number of concurrent requests sent by the SageMaker client to the model container. If no value is provided, SageMaker chooses an optimal value.
    invocation_timeout_in_seconds
    """

    max_concurrent_invocations_per_instance: Optional[int] = Unassigned()
    invocation_timeout_in_seconds: Optional[int] = Unassigned()


class AsyncInferenceNotificationConfig(Base):
    """
    AsyncInferenceNotificationConfig
      Specifies the configuration for notifications of inference results for asynchronous inference.

    Attributes
    ----------------------
    success_topic: Amazon SNS topic to post a notification to when inference completes successfully. If no topic is provided, no notification is sent on success.
    error_topic: Amazon SNS topic to post a notification to when inference fails. If no topic is provided, no notification is sent on failure.
    include_inference_response_in: The Amazon SNS topics where you want the inference response to be included.  The inference response is included only if the response size is less than or equal to 128 KB.
    """

    success_topic: Optional[StrPipeVar] = Unassigned()
    error_topic: Optional[StrPipeVar] = Unassigned()
    include_inference_response_in: Optional[List[StrPipeVar]] = Unassigned()


class AsyncInferenceOutputConfig(Base):
    """
    AsyncInferenceOutputConfig
      Specifies the configuration for asynchronous inference invocation outputs.

    Attributes
    ----------------------
    kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that SageMaker uses to encrypt the asynchronous inference output in Amazon S3.
    s3_output_path: The Amazon S3 location to upload inference responses to.
    notification_config: Specifies the configuration for notifications of inference results for asynchronous inference.
    s3_failure_path: The Amazon S3 location to upload failure inference responses to.
    """

    kms_key_id: Optional[StrPipeVar] = Unassigned()
    s3_output_path: Optional[StrPipeVar] = Unassigned()
    notification_config: Optional[AsyncInferenceNotificationConfig] = Unassigned()
    s3_failure_path: Optional[StrPipeVar] = Unassigned()


class AsyncInferenceConfig(Base):
    """
    AsyncInferenceConfig
      Specifies configuration for how an endpoint performs asynchronous inference.

    Attributes
    ----------------------
    client_config: Configures the behavior of the client used by SageMaker to interact with the model container during asynchronous inference.
    output_config: Specifies the configuration for asynchronous inference invocation outputs.
    """

    output_config: AsyncInferenceOutputConfig
    client_config: Optional[AsyncInferenceClientConfig] = Unassigned()


class AthenaDatasetDefinition(Base):
    """
    AthenaDatasetDefinition
      Configuration for Athena Dataset Definition input.

    Attributes
    ----------------------
    catalog
    database
    query_string
    work_group
    output_s3_uri: The location in Amazon S3 where Athena query results are stored.
    output_dataset_s3_uri
    kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt data generated from an Athena query execution.
    output_format
    output_compression
    """

    catalog: StrPipeVar
    database: StrPipeVar
    query_string: StrPipeVar
    output_s3_uri: StrPipeVar
    output_format: StrPipeVar
    work_group: Optional[StrPipeVar] = Unassigned()
    output_dataset_s3_uri: Optional[StrPipeVar] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    output_compression: Optional[StrPipeVar] = Unassigned()


class AuthorizedUrl(Base):
    """
    AuthorizedUrl
      Contains a presigned URL and its associated local file path for downloading hub content artifacts.

    Attributes
    ----------------------
    url: The presigned S3 URL that provides temporary, secure access to download the file. URLs expire within 15 minutes for security purposes.
    local_path: The recommended local file path where the downloaded file should be stored to maintain proper directory structure and file organization.
    """

    url: Optional[StrPipeVar] = Unassigned()
    local_path: Optional[StrPipeVar] = Unassigned()


class AutoMLAlgorithmConfig(Base):
    """
    AutoMLAlgorithmConfig
      The selection of algorithms trained on your dataset to generate the model candidates for an Autopilot job.

    Attributes
    ----------------------
    auto_ml_algorithms: The selection of algorithms trained on your dataset to generate the model candidates for an Autopilot job.    For the tabular problem type TabularJobConfig:   Selected algorithms must belong to the list corresponding to the training mode set in AutoMLJobConfig.Mode (ENSEMBLING or HYPERPARAMETER_TUNING). Choose a minimum of 1 algorithm.    In ENSEMBLING mode:   "catboost"   "extra-trees"   "fastai"   "lightgbm"   "linear-learner"   "nn-torch"   "randomforest"   "xgboost"     In HYPERPARAMETER_TUNING mode:   "linear-learner"   "mlp"   "xgboost"        For the time-series forecasting problem type TimeSeriesForecastingJobConfig:    Choose your algorithms from this list.   "cnn-qr"   "deepar"   "prophet"   "arima"   "npts"   "ets"
    """

    auto_ml_algorithms: List[StrPipeVar]


class FinalAutoMLJobObjectiveMetric(Base):
    """
    FinalAutoMLJobObjectiveMetric
      The best candidate result from an AutoML training job.

    Attributes
    ----------------------
    type: The type of metric with the best result.
    metric_name: The name of the metric with the best result. For a description of the possible objective metrics, see AutoMLJobObjective$MetricName.
    value: The value of the metric with the best result.
    standard_metric_name: The name of the standard metric. For a description of the standard metrics, see Autopilot candidate metrics.
    """

    metric_name: StrPipeVar
    value: float
    type: Optional[StrPipeVar] = Unassigned()
    standard_metric_name: Optional[StrPipeVar] = Unassigned()


class AutoMLCandidateStep(Base):
    """
    AutoMLCandidateStep
      Information about the steps for a candidate and what step it is working on.

    Attributes
    ----------------------
    candidate_step_type: Whether the candidate is at the transform, training, or processing step.
    candidate_step_arn: The ARN for the candidate's step.
    candidate_step_name: The name for the candidate's step.
    """

    candidate_step_type: StrPipeVar
    candidate_step_arn: StrPipeVar
    candidate_step_name: StrPipeVar


class AutoMLContainerDefinition(Base):
    """
    AutoMLContainerDefinition
      A list of container definitions that describe the different containers that make up an AutoML candidate. For more information, see  ContainerDefinition.

    Attributes
    ----------------------
    image: The Amazon Elastic Container Registry (Amazon ECR) path of the container. For more information, see  ContainerDefinition.
    model_data_url: The location of the model artifacts. For more information, see  ContainerDefinition.
    environment: The environment variables to set in the container. For more information, see  ContainerDefinition.
    """

    image: StrPipeVar
    model_data_url: StrPipeVar
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class CandidateArtifactLocations(Base):
    """
    CandidateArtifactLocations
      The location of artifacts for an AutoML candidate job.

    Attributes
    ----------------------
    explainability: The Amazon S3 prefix to the explainability artifacts generated for the AutoML candidate.
    model_insights: The Amazon S3 prefix to the model insight artifacts generated for the AutoML candidate.
    backtest_results: The Amazon S3 prefix to the accuracy metrics and the inference results observed over the testing window. Available only for the time-series forecasting problem type.
    """

    explainability: StrPipeVar
    model_insights: Optional[StrPipeVar] = Unassigned()
    backtest_results: Optional[StrPipeVar] = Unassigned()


class MetricDatum(Base):
    """
    MetricDatum
      Information about the metric for a candidate produced by an AutoML job.

    Attributes
    ----------------------
    metric_name: The name of the metric.
    standard_metric_name: The name of the standard metric.   For definitions of the standard metrics, see  Autopilot candidate metrics .
    value: The value of the metric.
    set: The dataset split from which the AutoML job produced the metric.
    """

    metric_name: Optional[StrPipeVar] = Unassigned()
    standard_metric_name: Optional[StrPipeVar] = Unassigned()
    value: Optional[float] = Unassigned()
    set: Optional[StrPipeVar] = Unassigned()


class CandidateProperties(Base):
    """
    CandidateProperties
      The properties of an AutoML candidate job.

    Attributes
    ----------------------
    candidate_artifact_locations: The Amazon S3 prefix to the artifacts generated for an AutoML candidate.
    candidate_metrics: Information about the candidate metrics for an AutoML job.
    """

    candidate_artifact_locations: Optional[CandidateArtifactLocations] = Unassigned()
    candidate_metrics: Optional[List[MetricDatum]] = Unassigned()


class AutoMLCandidate(Base):
    """
    AutoMLCandidate
      Information about a candidate produced by an AutoML training job, including its status, steps, and other properties.

    Attributes
    ----------------------
    candidate_name: The name of the candidate.
    final_auto_ml_job_objective_metric
    objective_status: The objective's status.
    candidate_steps: Information about the candidate's steps.
    candidate_status: The candidate's status.
    inference_containers: Information about the recommended inference container definitions.
    creation_time: The creation time.
    end_time: The end time.
    last_modified_time: The last modified time.
    failure_reason: The failure reason.
    candidate_properties: The properties of an AutoML candidate job.
    local_mode_enabled
    inference_container_definitions: The mapping of all supported processing unit (CPU, GPU, etc...) to inference container definitions for the candidate. This field is populated for the AutoML jobs V2 (for example, for jobs created by calling CreateAutoMLJobV2) related to image or text classification problem types only.
    """

    candidate_name: StrPipeVar
    objective_status: StrPipeVar
    candidate_steps: List[AutoMLCandidateStep]
    candidate_status: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    final_auto_ml_job_objective_metric: Optional[FinalAutoMLJobObjectiveMetric] = Unassigned()
    inference_containers: Optional[List[AutoMLContainerDefinition]] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    candidate_properties: Optional[CandidateProperties] = Unassigned()
    local_mode_enabled: Optional[bool] = Unassigned()
    inference_container_definitions: Optional[Dict[StrPipeVar, List[AutoMLContainerDefinition]]] = (
        Unassigned()
    )


class Transformer(Base):
    """
    Transformer

    Attributes
    ----------------------
    name
    """

    name: StrPipeVar


class ColumnConfig(Base):
    """
    ColumnConfig

    Attributes
    ----------------------
    column_type
    column_names
    transformers
    """

    transformers: List[Transformer]
    column_type: Optional[StrPipeVar] = Unassigned()
    column_names: Optional[List[StrPipeVar]] = Unassigned()


class CandidateSpecification(Base):
    """
    CandidateSpecification

    Attributes
    ----------------------
    algorithm
    columns_config
    """

    columns_config: List[ColumnConfig]
    algorithm: Optional[StrPipeVar] = Unassigned()


class AutoMLCandidateGenerationConfig(Base):
    """
    AutoMLCandidateGenerationConfig
      Stores the configuration information for how a candidate is generated (optional).

    Attributes
    ----------------------
    generate_candidates_mode
    algorithms
    transformers
    feature_specification_s3_uri: A URL to the Amazon S3 data source containing selected features from the input data source to run an Autopilot job. You can input FeatureAttributeNames (optional) in JSON format as shown below:   { "FeatureAttributeNames":["col1", "col2", ...] }. You can also specify the data type of the feature (optional) in the format shown below:  { "FeatureDataTypes":{"col1":"numeric", "col2":"categorical" ... } }   These column keys may not include the target column.  In ensembling mode, Autopilot only supports the following data types: numeric, categorical, text, and datetime. In HPO mode, Autopilot can support numeric, categorical, text, datetime, and sequence. If only FeatureDataTypes is provided, the column keys (col1, col2,..) should be a subset of the column names in the input data.  If both FeatureDataTypes and FeatureAttributeNames are provided, then the column keys should be a subset of the column names provided in FeatureAttributeNames.  The key name FeatureAttributeNames is fixed. The values listed in ["col1", "col2", ...] are case sensitive and should be a list of strings containing unique values that are a subset of the column names in the input data. The list of columns provided must not include the target column.
    candidates_specification
    algorithms_config: Stores the configuration information for the selection of algorithms trained on tabular data. The list of available algorithms to choose from depends on the training mode set in  TabularJobConfig.Mode .    AlgorithmsConfig should not be set if the training mode is set on AUTO.   When AlgorithmsConfig is provided, one AutoMLAlgorithms attribute must be set and one only. If the list of algorithms provided as values for AutoMLAlgorithms is empty, CandidateGenerationConfig uses the full set of algorithms for the given training mode.   When AlgorithmsConfig is not provided, CandidateGenerationConfig uses the full set of algorithms for the given training mode.   For the list of all algorithms per problem type and training mode, see  AutoMLAlgorithmConfig. For more information on each algorithm, see the Algorithm support section in Autopilot developer guide.
    """

    generate_candidates_mode: Optional[StrPipeVar] = Unassigned()
    algorithms: Optional[List[StrPipeVar]] = Unassigned()
    transformers: Optional[List[StrPipeVar]] = Unassigned()
    feature_specification_s3_uri: Optional[StrPipeVar] = Unassigned()
    candidates_specification: Optional[List[CandidateSpecification]] = Unassigned()
    algorithms_config: Optional[List[AutoMLAlgorithmConfig]] = Unassigned()


class AutoMLS3DataSource(Base):
    """
    AutoMLS3DataSource
      Describes the Amazon S3 data source.

    Attributes
    ----------------------
    s3_data_type: The data type.    If you choose S3Prefix, S3Uri identifies a key name prefix. SageMaker AI uses all objects that match the specified key name prefix for model training. The S3Prefix should have the following format:  s3://DOC-EXAMPLE-BUCKET/DOC-EXAMPLE-FOLDER-OR-FILE    If you choose ManifestFile, S3Uri identifies an object that is a manifest file containing a list of object keys that you want SageMaker AI to use for model training. A ManifestFile should have the format shown below:  [ {"prefix": "s3://DOC-EXAMPLE-BUCKET/DOC-EXAMPLE-FOLDER/DOC-EXAMPLE-PREFIX/"},    "DOC-EXAMPLE-RELATIVE-PATH/DOC-EXAMPLE-FOLDER/DATA-1",   "DOC-EXAMPLE-RELATIVE-PATH/DOC-EXAMPLE-FOLDER/DATA-2",   ... "DOC-EXAMPLE-RELATIVE-PATH/DOC-EXAMPLE-FOLDER/DATA-N" ]    If you choose AugmentedManifestFile, S3Uri identifies an object that is an augmented manifest file in JSON lines format. This file contains the data you want to use for model training. AugmentedManifestFile is available for V2 API jobs only (for example, for jobs created by calling CreateAutoMLJobV2). Here is a minimal, single-record example of an AugmentedManifestFile:  {"source-ref": "s3://DOC-EXAMPLE-BUCKET/DOC-EXAMPLE-FOLDER/cats/cat.jpg",   "label-metadata": {"class-name": "cat" } For more information on AugmentedManifestFile, see Provide Dataset Metadata to Training Jobs with an Augmented Manifest File.
    s3_uri: The URL to the Amazon S3 data source. The Uri refers to the Amazon S3 prefix or ManifestFile depending on the data type.
    """

    s3_data_type: StrPipeVar
    s3_uri: StrPipeVar


class AutoMLFileSystemDataSource(Base):
    """
    AutoMLFileSystemDataSource

    Attributes
    ----------------------
    file_system_id
    file_system_access_mode
    file_system_type
    directory_path
    """

    file_system_id: StrPipeVar
    file_system_access_mode: StrPipeVar
    file_system_type: StrPipeVar
    directory_path: StrPipeVar


class AutoMLDataSource(Base):
    """
    AutoMLDataSource
      The data source for the Autopilot job.

    Attributes
    ----------------------
    s3_data_source: The Amazon S3 location of the input data.
    file_system_data_source
    """

    s3_data_source: AutoMLS3DataSource
    file_system_data_source: Optional[AutoMLFileSystemDataSource] = Unassigned()


class AutoMLSnowflakeDatasetDefinition(Base):
    """
    AutoMLSnowflakeDatasetDefinition

    Attributes
    ----------------------
    warehouse
    database
    schema
    table_name
    snowflake_role
    secret_arn
    output_s3_uri
    storage_integration
    kms_key_id
    """

    warehouse: StrPipeVar
    database: StrPipeVar
    schema: StrPipeVar
    table_name: StrPipeVar
    secret_arn: StrPipeVar
    output_s3_uri: StrPipeVar
    storage_integration: StrPipeVar
    snowflake_role: Optional[StrPipeVar] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class AutoMLDatasetDefinition(Base):
    """
    AutoMLDatasetDefinition

    Attributes
    ----------------------
    auto_ml_snowflake_dataset_definition
    """

    auto_ml_snowflake_dataset_definition: Optional[AutoMLSnowflakeDatasetDefinition] = Unassigned()


class AutoMLChannel(Base):
    """
    AutoMLChannel
      A channel is a named input source that training algorithms can consume. The validation dataset size is limited to less than 2 GB. The training dataset size must be less than 100 GB. For more information, see  Channel.  A validation dataset must contain the same headers as the training dataset.

    Attributes
    ----------------------
    data_source: The data source for an AutoML channel.
    compression_type: You can use Gzip or None. The default value is None.
    target_attribute_name: The name of the target variable in supervised learning, usually represented by 'y'.
    feature_attribute_s3_uri
    auto_ml_dataset_definition
    content_type: The content type of the data from the input source. You can use text/csv;header=present or x-application/vnd.amazon+parquet. The default value is text/csv;header=present.
    channel_type: The channel type (optional) is an enum string. The default value is training. Channels for training and validation must share the same ContentType and TargetAttributeName. For information on specifying training and validation channel types, see How to specify training and validation datasets.
    sample_weight_attribute_name: If specified, this column name indicates which column of the dataset should be treated as sample weights for use by the objective metric during the training, evaluation, and the selection of the best model. This column is not considered as a predictive feature. For more information on Autopilot metrics, see Metrics and validation. Sample weights should be numeric, non-negative, with larger values indicating which rows are more important than others. Data points that have invalid or no weight value are excluded. Support for sample weights is available in Ensembling mode only.
    """

    target_attribute_name: StrPipeVar
    data_source: Optional[AutoMLDataSource] = Unassigned()
    compression_type: Optional[StrPipeVar] = Unassigned()
    feature_attribute_s3_uri: Optional[StrPipeVar] = Unassigned()
    auto_ml_dataset_definition: Optional[AutoMLDatasetDefinition] = Unassigned()
    content_type: Optional[StrPipeVar] = Unassigned()
    channel_type: Optional[StrPipeVar] = Unassigned()
    sample_weight_attribute_name: Optional[StrPipeVar] = Unassigned()


class EmrServerlessComputeConfig(Base):
    """
    EmrServerlessComputeConfig
       This data type is intended for use exclusively by SageMaker Canvas and cannot be used in other contexts at the moment.  Specifies the compute configuration for the EMR Serverless job.

    Attributes
    ----------------------
    execution_role_arn: The ARN of the IAM role granting the AutoML job V2 the necessary permissions access policies to list, connect to, or manage EMR Serverless jobs. For detailed information about the required permissions of this role, see "How to configure AutoML to initiate a remote job on EMR Serverless for large datasets" in Create a regression or classification job for tabular data using the AutoML API or Create an AutoML job for time-series forecasting using the API.
    """

    execution_role_arn: StrPipeVar


class AutoMLComputeConfig(Base):
    """
    AutoMLComputeConfig
       This data type is intended for use exclusively by SageMaker Canvas and cannot be used in other contexts at the moment.  Specifies the compute configuration for an AutoML job V2.

    Attributes
    ----------------------
    emr_serverless_compute_config: The configuration for using  EMR Serverless to run the AutoML job V2. To allow your AutoML job V2 to automatically initiate a remote job on EMR Serverless when additional compute resources are needed to process large datasets, you need to provide an EmrServerlessComputeConfig object, which includes an ExecutionRoleARN attribute, to the AutoMLComputeConfig of the AutoML job V2 input request. By seamlessly transitioning to EMR Serverless when required, the AutoML job can handle datasets that would otherwise exceed the initially provisioned resources, without any manual intervention from you.  EMR Serverless is available for the tabular and time series problem types. We recommend setting up this option for tabular datasets larger than 5 GB and time series datasets larger than 30 GB.
    """

    emr_serverless_compute_config: Optional[EmrServerlessComputeConfig] = Unassigned()


class AutoMLDataSplitConfig(Base):
    """
    AutoMLDataSplitConfig
      This structure specifies how to split the data into train and validation datasets. The validation and training datasets must contain the same headers. For jobs created by calling CreateAutoMLJob, the validation dataset must be less than 2 GB in size.

    Attributes
    ----------------------
    validation_fraction: The validation fraction (optional) is a float that specifies the portion of the training dataset to be used for validation. The default value is 0.2, and values must be greater than 0 and less than 1. We recommend setting this value to be less than 0.5.
    """

    validation_fraction: Optional[float] = Unassigned()


class AutoMLEndpointConfigDefinition(Base):
    """
    AutoMLEndpointConfigDefinition

    Attributes
    ----------------------
    endpoint_config_name
    initial_instance_count
    instance_type
    """

    endpoint_config_name: Union[StrPipeVar, object]
    initial_instance_count: int
    instance_type: StrPipeVar


class AutoMLEndpointDeletionCondition(Base):
    """
    AutoMLEndpointDeletionCondition

    Attributes
    ----------------------
    max_runtime_in_seconds
    """

    max_runtime_in_seconds: int


class AutoMLEndpointDefinition(Base):
    """
    AutoMLEndpointDefinition

    Attributes
    ----------------------
    endpoint_name
    endpoint_config_name
    deletion_condition
    """

    endpoint_name: Union[StrPipeVar, object]
    endpoint_config_name: Union[StrPipeVar, object]
    deletion_condition: Optional[AutoMLEndpointDeletionCondition] = Unassigned()


class AutoMLExternalFeatureTransformers(Base):
    """
    AutoMLExternalFeatureTransformers

    Attributes
    ----------------------
    pre_feature_transformers
    """

    pre_feature_transformers: Optional[List[AutoMLContainerDefinition]] = Unassigned()


class AutoMLJobArtifacts(Base):
    """
    AutoMLJobArtifacts
      The artifacts that are generated during an AutoML job.

    Attributes
    ----------------------
    candidate_definition_notebook_location: The URL of the notebook location.
    data_exploration_notebook_location: The URL of the notebook location.
    """

    candidate_definition_notebook_location: Optional[StrPipeVar] = Unassigned()
    data_exploration_notebook_location: Optional[StrPipeVar] = Unassigned()


class AutoMLJobChannel(Base):
    """
    AutoMLJobChannel
      A channel is a named input source that training algorithms can consume. This channel is used for AutoML jobs V2 (jobs created by calling CreateAutoMLJobV2).

    Attributes
    ----------------------
    channel_type: The type of channel. Defines whether the data are used for training or validation. The default value is training. Channels for training and validation must share the same ContentType   The type of channel defaults to training for the time-series forecasting problem type.
    content_type: The content type of the data from the input source. The following are the allowed content types for different problems:   For tabular problem types: text/csv;header=present or x-application/vnd.amazon+parquet. The default value is text/csv;header=present.   For image classification: image/png, image/jpeg, or image/*. The default value is image/*.   For text classification: text/csv;header=present or x-application/vnd.amazon+parquet. The default value is text/csv;header=present.   For time-series forecasting: text/csv;header=present or x-application/vnd.amazon+parquet. The default value is text/csv;header=present.   For text generation (LLMs fine-tuning): text/csv;header=present or x-application/vnd.amazon+parquet. The default value is text/csv;header=present.
    compression_type: The allowed compression types depend on the input format and problem type. We allow the compression type Gzip for S3Prefix inputs on tabular data only. For all other inputs, the compression type should be None. If no compression type is provided, we default to None.
    data_source: The data source for an AutoML channel (Required).
    dataset_definition
    """

    channel_type: Optional[StrPipeVar] = Unassigned()
    content_type: Optional[StrPipeVar] = Unassigned()
    compression_type: Optional[StrPipeVar] = Unassigned()
    data_source: Optional[AutoMLDataSource] = Unassigned()
    dataset_definition: Optional[AutoMLDatasetDefinition] = Unassigned()


class AutoMLJobCompletionCriteria(Base):
    """
    AutoMLJobCompletionCriteria
      How long a job is allowed to run, or how many candidates a job is allowed to generate.

    Attributes
    ----------------------
    max_candidates: The maximum number of times a training job is allowed to run. For text and image classification, time-series forecasting, as well as text generation (LLMs fine-tuning) problem types, the supported value is 1. For tabular problem types, the maximum value is 750.
    max_runtime_per_training_job_in_seconds: The maximum time, in seconds, that each training job executed inside hyperparameter tuning is allowed to run as part of a hyperparameter tuning job. For more information, see the StoppingCondition used by the CreateHyperParameterTuningJob action. For job V2s (jobs created by calling CreateAutoMLJobV2), this field controls the runtime of the job candidate. For TextGenerationJobConfig problem types, the maximum time defaults to 72 hours (259200 seconds).
    max_auto_ml_job_runtime_in_seconds: The maximum runtime, in seconds, an AutoML job has to complete. If an AutoML job exceeds the maximum runtime, the job is stopped automatically and its processing is ended gracefully. The AutoML job identifies the best model whose training was completed and marks it as the best-performing model. Any unfinished steps of the job, such as automatic one-click Autopilot model deployment, are not completed.
    """

    max_candidates: Optional[int] = Unassigned()
    max_runtime_per_training_job_in_seconds: Optional[int] = Unassigned()
    max_auto_ml_job_runtime_in_seconds: Optional[int] = Unassigned()


class VpcConfig(Base):
    """
    VpcConfig
      Specifies an Amazon Virtual Private Cloud (VPC) that your SageMaker jobs, hosted models, and compute resources have access to. You can control access to and from your resources by configuring a VPC. For more information, see Give SageMaker Access to Resources in your Amazon VPC.

    Attributes
    ----------------------
    security_group_ids: The VPC security group IDs, in the form sg-xxxxxxxx. Specify the security groups for the VPC that is specified in the Subnets field.
    subnets: The ID of the subnets in the VPC to which you want to connect your training job or model. For information about the availability of specific instance types, see Supported Instance Types and Availability Zones.
    """

    security_group_ids: List[StrPipeVar]
    subnets: List[StrPipeVar]


class AutoMLSecurityConfig(Base):
    """
    AutoMLSecurityConfig
      Security options.

    Attributes
    ----------------------
    volume_kms_key_id: The key used to encrypt stored data.
    enable_inter_container_traffic_encryption: Whether to use traffic encryption between the container layers.
    vpc_config: The VPC configuration.
    """

    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()
    enable_inter_container_traffic_encryption: Optional[bool] = Unassigned()
    vpc_config: Optional[VpcConfig] = Unassigned()


class AutoMLJobConfig(Base):
    """
    AutoMLJobConfig
      A collection of settings used for an AutoML job.

    Attributes
    ----------------------
    completion_criteria: How long an AutoML job is allowed to run, or how many candidates a job is allowed to generate.
    security_config: The security configuration for traffic encryption or Amazon VPC settings.
    candidate_generation_config: The configuration for generating a candidate for an AutoML job (optional).
    data_split_config: The configuration for splitting the input training dataset. Type: AutoMLDataSplitConfig
    engine
    mode: The method that Autopilot uses to train the data. You can either specify the mode manually or let Autopilot choose for you based on the dataset size by selecting AUTO. In AUTO mode, Autopilot chooses ENSEMBLING for datasets smaller than 100 MB, and HYPERPARAMETER_TUNING for larger ones. The ENSEMBLING mode uses a multi-stack ensemble model to predict classification and regression tasks directly from your dataset. This machine learning mode combines several base models to produce an optimal predictive model. It then uses a stacking ensemble method to combine predictions from contributing members. A multi-stack ensemble model can provide better performance over a single model by combining the predictive capabilities of multiple models. See Autopilot algorithm support for a list of algorithms supported by ENSEMBLING mode. The HYPERPARAMETER_TUNING (HPO) mode uses the best hyperparameters to train the best version of a model. HPO automatically selects an algorithm for the type of problem you want to solve. Then HPO finds the best hyperparameters according to your objective metric. See Autopilot algorithm support for a list of algorithms supported by HYPERPARAMETER_TUNING mode.
    local_mode_enabled
    external_feature_transformers
    """

    completion_criteria: Optional[AutoMLJobCompletionCriteria] = Unassigned()
    security_config: Optional[AutoMLSecurityConfig] = Unassigned()
    candidate_generation_config: Optional[AutoMLCandidateGenerationConfig] = Unassigned()
    data_split_config: Optional[AutoMLDataSplitConfig] = Unassigned()
    engine: Optional[StrPipeVar] = Unassigned()
    mode: Optional[StrPipeVar] = Unassigned()
    local_mode_enabled: Optional[bool] = Unassigned()
    external_feature_transformers: Optional[AutoMLExternalFeatureTransformers] = Unassigned()


class AutoMLJobObjective(Base):
    """
    AutoMLJobObjective
      Specifies a metric to minimize or maximize as the objective of an AutoML job.

    Attributes
    ----------------------
    metric_name: The name of the objective metric used to measure the predictive quality of a machine learning system. During training, the model's parameters are updated iteratively to optimize its performance based on the feedback provided by the objective metric when evaluating the model on the validation dataset. The list of available metrics supported by Autopilot and the default metric applied when you do not specify a metric name explicitly depend on the problem type.   For tabular problem types:   List of available metrics:     Regression: MAE, MSE, R2, RMSE     Binary classification: Accuracy, AUC, BalancedAccuracy, F1, Precision, Recall     Multiclass classification: Accuracy, BalancedAccuracy, F1macro, PrecisionMacro, RecallMacro    For a description of each metric, see Autopilot metrics for classification and regression.   Default objective metrics:   Regression: MSE.   Binary classification: F1.   Multiclass classification: Accuracy.       For image or text classification problem types:   List of available metrics: Accuracy  For a description of each metric, see Autopilot metrics for text and image classification.   Default objective metrics: Accuracy      For time-series forecasting problem types:   List of available metrics: RMSE, wQL, Average wQL, MASE, MAPE, WAPE  For a description of each metric, see Autopilot metrics for time-series forecasting.   Default objective metrics: AverageWeightedQuantileLoss      For text generation problem types (LLMs fine-tuning): Fine-tuning language models in Autopilot does not require setting the AutoMLJobObjective field. Autopilot fine-tunes LLMs without requiring multiple candidates to be trained and evaluated. Instead, using your dataset, Autopilot directly fine-tunes your target model to enhance a default objective metric, the cross-entropy loss. After fine-tuning a language model, you can evaluate the quality of its generated text using different metrics. For a list of the available metrics, see Metrics for fine-tuning LLMs in Autopilot.
    """

    metric_name: StrPipeVar


class AutoMLJobStepMetadata(Base):
    """
    AutoMLJobStepMetadata
      Metadata for an AutoML job step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the AutoML job.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class AutoMLPartialFailureReason(Base):
    """
    AutoMLPartialFailureReason
      The reason for a partial failure of an AutoML job.

    Attributes
    ----------------------
    partial_failure_message: The message containing the reason for a partial failure of an AutoML job.
    """

    partial_failure_message: Optional[StrPipeVar] = Unassigned()


class AutoMLJobSummary(Base):
    """
    AutoMLJobSummary
      Provides a summary about an AutoML job.

    Attributes
    ----------------------
    auto_ml_job_name: The name of the AutoML job you are requesting.
    auto_ml_job_arn: The ARN of the AutoML job.
    auto_ml_job_status: The status of the AutoML job.
    auto_ml_job_secondary_status: The secondary status of the AutoML job.
    creation_time: When the AutoML job was created.
    end_time: The end time of an AutoML job.
    last_modified_time: When the AutoML job was last modified.
    failure_reason: The failure reason of an AutoML job.
    partial_failure_reasons: The list of reasons for partial failures within an AutoML job.
    """

    auto_ml_job_name: StrPipeVar
    auto_ml_job_arn: StrPipeVar
    auto_ml_job_status: StrPipeVar
    auto_ml_job_secondary_status: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    end_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    partial_failure_reasons: Optional[List[AutoMLPartialFailureReason]] = Unassigned()


class AutoMLOutputDataConfig(Base):
    """
    AutoMLOutputDataConfig
      The output data configuration.

    Attributes
    ----------------------
    kms_key_id: The Key Management Service encryption key ID.
    s3_output_path: The Amazon S3 output path. Must be 512 characters or less.
    """

    s3_output_path: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class ImageClassificationJobConfig(Base):
    """
    ImageClassificationJobConfig
      The collection of settings used by an AutoML job V2 for the image classification problem type.

    Attributes
    ----------------------
    completion_criteria: How long a job is allowed to run, or how many candidates a job is allowed to generate.
    multi_label_enabled
    """

    completion_criteria: Optional[AutoMLJobCompletionCriteria] = Unassigned()
    multi_label_enabled: Optional[bool] = Unassigned()


class TextClassificationJobConfig(Base):
    """
    TextClassificationJobConfig
      The collection of settings used by an AutoML job V2 for the text classification problem type.

    Attributes
    ----------------------
    completion_criteria: How long a job is allowed to run, or how many candidates a job is allowed to generate.
    content_column: The name of the column used to provide the sentences to be classified. It should not be the same as the target column.
    target_label_column: The name of the column used to provide the class labels. It should not be same as the content column.
    """

    content_column: StrPipeVar
    target_label_column: StrPipeVar
    completion_criteria: Optional[AutoMLJobCompletionCriteria] = Unassigned()


class TimeSeriesTransformations(Base):
    """
    TimeSeriesTransformations
      Transformations allowed on the dataset. Supported transformations are Filling and Aggregation. Filling specifies how to add values to missing values in the dataset. Aggregation defines how to aggregate data that does not align with forecast frequency.

    Attributes
    ----------------------
    filling: A key value pair defining the filling method for a column, where the key is the column name and the value is an object which defines the filling logic. You can specify multiple filling methods for a single column. The supported filling methods and their corresponding options are:    frontfill: none (Supported only for target column)    middlefill: zero, value, median, mean, min, max     backfill: zero, value, median, mean, min, max     futurefill: zero, value, median, mean, min, max    To set a filling method to a specific value, set the fill parameter to the chosen filling method value (for example "backfill" : "value"), and define the filling value in an additional parameter prefixed with "_value". For example, to set backfill to a value of 2, you must include two parameters: "backfill": "value" and "backfill_value":"2".
    aggregation: A key value pair defining the aggregation method for a column, where the key is the column name and the value is the aggregation method. The supported aggregation methods are sum (default), avg, first, min, max.  Aggregation is only supported for the target column.
    """

    filling: Optional[Dict[StrPipeVar, Dict[StrPipeVar, StrPipeVar]]] = Unassigned()
    aggregation: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class TimeSeriesConfig(Base):
    """
    TimeSeriesConfig
      The collection of components that defines the time-series.

    Attributes
    ----------------------
    target_attribute_name: The name of the column representing the target variable that you want to predict for each item in your dataset. The data type of the target variable must be numerical.
    timestamp_attribute_name: The name of the column indicating a point in time at which the target value of a given item is recorded.
    item_identifier_attribute_name: The name of the column that represents the set of item identifiers for which you want to predict the target value.
    grouping_attribute_names: A set of columns names that can be grouped with the item identifier column to create a composite key for which a target value is predicted.
    """

    target_attribute_name: StrPipeVar
    timestamp_attribute_name: StrPipeVar
    item_identifier_attribute_name: StrPipeVar
    grouping_attribute_names: Optional[List[StrPipeVar]] = Unassigned()


class HolidayConfigAttributes(Base):
    """
    HolidayConfigAttributes
      Stores the holiday featurization attributes applicable to each item of time-series datasets during the training of a forecasting model. This allows the model to identify patterns associated with specific holidays.

    Attributes
    ----------------------
    country_code: The country code for the holiday calendar. For the list of public holiday calendars supported by AutoML job V2, see Country Codes. Use the country code corresponding to the country of your choice.
    """

    country_code: Optional[StrPipeVar] = Unassigned()


class CandidateGenerationConfig(Base):
    """
    CandidateGenerationConfig
      Stores the configuration information for how model candidates are generated using an AutoML job V2.

    Attributes
    ----------------------
    algorithms_config: Your Autopilot job trains a default set of algorithms on your dataset. For tabular and time-series data, you can customize the algorithm list by selecting a subset of algorithms for your problem type.  AlgorithmsConfig stores the customized selection of algorithms to train on your data.    For the tabular problem type TabularJobConfig, the list of available algorithms to choose from depends on the training mode set in  AutoMLJobConfig.Mode .    AlgorithmsConfig should not be set when the training mode AutoMLJobConfig.Mode is set to AUTO.   When AlgorithmsConfig is provided, one AutoMLAlgorithms attribute must be set and one only. If the list of algorithms provided as values for AutoMLAlgorithms is empty, CandidateGenerationConfig uses the full set of algorithms for the given training mode.   When AlgorithmsConfig is not provided, CandidateGenerationConfig uses the full set of algorithms for the given training mode.   For the list of all algorithms per training mode, see  AlgorithmConfig. For more information on each algorithm, see the Algorithm support section in the Autopilot developer guide.    For the time-series forecasting problem type TimeSeriesForecastingJobConfig, choose your algorithms from the list provided in  AlgorithmConfig. For more information on each algorithm, see the Algorithms support for time-series forecasting section in the Autopilot developer guide.   When AlgorithmsConfig is provided, one AutoMLAlgorithms attribute must be set and one only. If the list of algorithms provided as values for AutoMLAlgorithms is empty, CandidateGenerationConfig uses the full set of algorithms for time-series forecasting.   When AlgorithmsConfig is not provided, CandidateGenerationConfig uses the full set of algorithms for time-series forecasting.
    generate_candidates_mode
    transformers
    candidates_specification
    """

    algorithms_config: Optional[List[AutoMLAlgorithmConfig]] = Unassigned()
    generate_candidates_mode: Optional[StrPipeVar] = Unassigned()
    transformers: Optional[List[StrPipeVar]] = Unassigned()
    candidates_specification: Optional[List[CandidateSpecification]] = Unassigned()


class TimeSeriesForecastingJobConfig(Base):
    """
    TimeSeriesForecastingJobConfig
      The collection of settings used by an AutoML job V2 for the time-series forecasting problem type.

    Attributes
    ----------------------
    feature_specification_s3_uri: A URL to the Amazon S3 data source containing additional selected features that complement the target, itemID, timestamp, and grouped columns set in TimeSeriesConfig. When not provided, the AutoML job V2 includes all the columns from the original dataset that are not already declared in TimeSeriesConfig. If provided, the AutoML job V2 only considers these additional columns as a complement to the ones declared in TimeSeriesConfig. You can input FeatureAttributeNames (optional) in JSON format as shown below:   { "FeatureAttributeNames":["col1", "col2", ...] }. You can also specify the data type of the feature (optional) in the format shown below:  { "FeatureDataTypes":{"col1":"numeric", "col2":"categorical" ... } }  Autopilot supports the following data types: numeric, categorical, text, and datetime.  These column keys must not include any column set in TimeSeriesConfig.
    completion_criteria
    forecast_frequency: The frequency of predictions in a forecast. Valid intervals are an integer followed by Y (Year), M (Month), W (Week), D (Day), H (Hour), and min (Minute). For example, 1D indicates every day and 15min indicates every 15 minutes. The value of a frequency must not overlap with the next larger frequency. For example, you must use a frequency of 1H instead of 60min. The valid values for each frequency are the following:   Minute - 1-59   Hour - 1-23   Day - 1-6   Week - 1-4   Month - 1-11   Year - 1
    forecast_horizon: The number of time-steps that the model predicts. The forecast horizon is also called the prediction length. The maximum forecast horizon is the lesser of 500 time-steps or 1/4 of the time-steps in the dataset.
    forecast_quantiles: The quantiles used to train the model for forecasts at a specified quantile. You can specify quantiles from 0.01 (p1) to 0.99 (p99), by increments of 0.01 or higher. Up to five forecast quantiles can be specified. When ForecastQuantiles is not provided, the AutoML job uses the quantiles p10, p50, and p90 as default.
    transformations: The transformations modifying specific attributes of the time-series, such as filling strategies for missing values.
    time_series_config: The collection of components that defines the time-series.
    holiday_config: The collection of holiday featurization attributes used to incorporate national holiday information into your forecasting model.
    candidate_generation_config
    """

    forecast_frequency: StrPipeVar
    forecast_horizon: int
    time_series_config: TimeSeriesConfig
    feature_specification_s3_uri: Optional[StrPipeVar] = Unassigned()
    completion_criteria: Optional[AutoMLJobCompletionCriteria] = Unassigned()
    forecast_quantiles: Optional[List[StrPipeVar]] = Unassigned()
    transformations: Optional[TimeSeriesTransformations] = Unassigned()
    holiday_config: Optional[List[HolidayConfigAttributes]] = Unassigned()
    candidate_generation_config: Optional[CandidateGenerationConfig] = Unassigned()


class TabularJobConfig(Base):
    """
    TabularJobConfig
      The collection of settings used by an AutoML job V2 for the tabular problem type.

    Attributes
    ----------------------
    candidate_generation_config: The configuration information of how model candidates are generated.
    completion_criteria
    feature_specification_s3_uri: A URL to the Amazon S3 data source containing selected features from the input data source to run an Autopilot job V2. You can input FeatureAttributeNames (optional) in JSON format as shown below:   { "FeatureAttributeNames":["col1", "col2", ...] }. You can also specify the data type of the feature (optional) in the format shown below:  { "FeatureDataTypes":{"col1":"numeric", "col2":"categorical" ... } }   These column keys may not include the target column.  In ensembling mode, Autopilot only supports the following data types: numeric, categorical, text, and datetime. In HPO mode, Autopilot can support numeric, categorical, text, datetime, and sequence. If only FeatureDataTypes is provided, the column keys (col1, col2,..) should be a subset of the column names in the input data.  If both FeatureDataTypes and FeatureAttributeNames are provided, then the column keys should be a subset of the column names provided in FeatureAttributeNames.  The key name FeatureAttributeNames is fixed. The values listed in ["col1", "col2", ...] are case sensitive and should be a list of strings containing unique values that are a subset of the column names in the input data. The list of columns provided must not include the target column.
    mode: The method that Autopilot uses to train the data. You can either specify the mode manually or let Autopilot choose for you based on the dataset size by selecting AUTO. In AUTO mode, Autopilot chooses ENSEMBLING for datasets smaller than 100 MB, and HYPERPARAMETER_TUNING for larger ones. The ENSEMBLING mode uses a multi-stack ensemble model to predict classification and regression tasks directly from your dataset. This machine learning mode combines several base models to produce an optimal predictive model. It then uses a stacking ensemble method to combine predictions from contributing members. A multi-stack ensemble model can provide better performance over a single model by combining the predictive capabilities of multiple models. See Autopilot algorithm support for a list of algorithms supported by ENSEMBLING mode. The HYPERPARAMETER_TUNING (HPO) mode uses the best hyperparameters to train the best version of a model. HPO automatically selects an algorithm for the type of problem you want to solve. Then HPO finds the best hyperparameters according to your objective metric. See Autopilot algorithm support for a list of algorithms supported by HYPERPARAMETER_TUNING mode.
    generate_candidate_definitions_only: Generates possible candidates without training the models. A model candidate is a combination of data preprocessors, algorithms, and algorithm parameter settings.
    problem_type: The type of supervised learning problem available for the model candidates of the AutoML job V2. For more information, see  SageMaker Autopilot problem types.  You must either specify the type of supervised learning problem in ProblemType and provide the AutoMLJobObjective metric, or none at all.
    target_attribute_name: The name of the target variable in supervised learning, usually represented by 'y'.
    sample_weight_attribute_name: If specified, this column name indicates which column of the dataset should be treated as sample weights for use by the objective metric during the training, evaluation, and the selection of the best model. This column is not considered as a predictive feature. For more information on Autopilot metrics, see Metrics and validation. Sample weights should be numeric, non-negative, with larger values indicating which rows are more important than others. Data points that have invalid or no weight value are excluded. Support for sample weights is available in Ensembling mode only.
    """

    target_attribute_name: StrPipeVar
    candidate_generation_config: Optional[CandidateGenerationConfig] = Unassigned()
    completion_criteria: Optional[AutoMLJobCompletionCriteria] = Unassigned()
    feature_specification_s3_uri: Optional[StrPipeVar] = Unassigned()
    mode: Optional[StrPipeVar] = Unassigned()
    generate_candidate_definitions_only: Optional[bool] = Unassigned()
    problem_type: Optional[StrPipeVar] = Unassigned()
    sample_weight_attribute_name: Optional[StrPipeVar] = Unassigned()


class TextGenerationJobConfig(Base):
    """
    TextGenerationJobConfig
      The collection of settings used by an AutoML job V2 for the text generation problem type.  The text generation models that support fine-tuning in Autopilot are currently accessible exclusively in regions supported by Canvas. Refer to the documentation of Canvas for the full list of its supported Regions.

    Attributes
    ----------------------
    completion_criteria: How long a fine-tuning job is allowed to run. For TextGenerationJobConfig problem types, the MaxRuntimePerTrainingJobInSeconds attribute of AutoMLJobCompletionCriteria defaults to 72h (259200s).
    base_model_name: The name of the base model to fine-tune. Autopilot supports fine-tuning a variety of large language models. For information on the list of supported models, see Text generation models supporting fine-tuning in Autopilot. If no BaseModelName is provided, the default model used is Falcon7BInstruct.
    text_generation_hyper_parameters: The hyperparameters used to configure and optimize the learning process of the base model. You can set any combination of the following hyperparameters for all base models. For more information on each supported hyperparameter, see Optimize the learning process of your text generation models with hyperparameters.    "epochCount": The number of times the model goes through the entire training dataset. Its value should be a string containing an integer value within the range of "1" to "10".    "batchSize": The number of data samples used in each iteration of training. Its value should be a string containing an integer value within the range of "1" to "64".    "learningRate": The step size at which a model's parameters are updated during training. Its value should be a string containing a floating-point value within the range of "0" to "1".    "learningRateWarmupSteps": The number of training steps during which the learning rate gradually increases before reaching its target or maximum value. Its value should be a string containing an integer value within the range of "0" to "250".   Here is an example where all four hyperparameters are configured.  { "epochCount":"5", "learningRate":"0.5", "batchSize": "32", "learningRateWarmupSteps": "10" }
    model_access_config
    """

    completion_criteria: Optional[AutoMLJobCompletionCriteria] = Unassigned()
    base_model_name: Optional[StrPipeVar] = Unassigned()
    text_generation_hyper_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    model_access_config: Optional[ModelAccessConfig] = Unassigned()


class AutoMLProblemTypeConfig(Base):
    """
    AutoMLProblemTypeConfig
      A collection of settings specific to the problem type used to configure an AutoML job V2. There must be one and only one config of the following type.

    Attributes
    ----------------------
    image_classification_job_config: Settings used to configure an AutoML job V2 for the image classification problem type.
    text_classification_job_config: Settings used to configure an AutoML job V2 for the text classification problem type.
    time_series_forecasting_job_config: Settings used to configure an AutoML job V2 for the time-series forecasting problem type.
    tabular_job_config: Settings used to configure an AutoML job V2 for the tabular problem type (regression, classification).
    text_generation_job_config: Settings used to configure an AutoML job V2 for the text generation (LLMs fine-tuning) problem type.  The text generation models that support fine-tuning in Autopilot are currently accessible exclusively in regions supported by Canvas. Refer to the documentation of Canvas for the full list of its supported Regions.
    """

    image_classification_job_config: Optional[ImageClassificationJobConfig] = Unassigned()
    text_classification_job_config: Optional[TextClassificationJobConfig] = Unassigned()
    time_series_forecasting_job_config: Optional[TimeSeriesForecastingJobConfig] = Unassigned()
    tabular_job_config: Optional[TabularJobConfig] = Unassigned()
    text_generation_job_config: Optional[TextGenerationJobConfig] = Unassigned()


class TabularResolvedAttributes(Base):
    """
    TabularResolvedAttributes
      The resolved attributes specific to the tabular problem type.

    Attributes
    ----------------------
    problem_type: The type of supervised learning problem available for the model candidates of the AutoML job V2 (Binary Classification, Multiclass Classification, Regression). For more information, see  SageMaker Autopilot problem types.
    local_mode_enabled
    """

    problem_type: Optional[StrPipeVar] = Unassigned()
    local_mode_enabled: Optional[bool] = Unassigned()


class TextGenerationResolvedAttributes(Base):
    """
    TextGenerationResolvedAttributes
      The resolved attributes specific to the text generation problem type.

    Attributes
    ----------------------
    base_model_name: The name of the base model to fine-tune.
    """

    base_model_name: Optional[StrPipeVar] = Unassigned()


class AutoMLProblemTypeResolvedAttributes(Base):
    """
    AutoMLProblemTypeResolvedAttributes
      Stores resolved attributes specific to the problem type of an AutoML job V2.

    Attributes
    ----------------------
    tabular_resolved_attributes: The resolved attributes for the tabular problem type.
    text_generation_resolved_attributes: The resolved attributes for the text generation problem type.
    """

    tabular_resolved_attributes: Optional[TabularResolvedAttributes] = Unassigned()
    text_generation_resolved_attributes: Optional[TextGenerationResolvedAttributes] = Unassigned()


class AutoMLResolvedAttributes(Base):
    """
    AutoMLResolvedAttributes
      The resolved attributes used to configure an AutoML job V2.

    Attributes
    ----------------------
    auto_ml_job_objective
    completion_criteria
    auto_ml_problem_type_resolved_attributes: Defines the resolved attributes specific to a problem type.
    """

    auto_ml_job_objective: Optional[AutoMLJobObjective] = Unassigned()
    completion_criteria: Optional[AutoMLJobCompletionCriteria] = Unassigned()
    auto_ml_problem_type_resolved_attributes: Optional[AutoMLProblemTypeResolvedAttributes] = (
        Unassigned()
    )


class AutoMLTask(Base):
    """
    AutoMLTask

    Attributes
    ----------------------
    auto_ml_job_arn
    auto_ml_task_arn
    candidate_name
    auto_ml_task_type
    auto_ml_task_status
    creation_time
    end_time
    last_modified_time
    """

    auto_ml_job_arn: StrPipeVar
    auto_ml_task_arn: StrPipeVar
    candidate_name: StrPipeVar
    auto_ml_task_type: StrPipeVar
    auto_ml_task_status: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    end_time: Optional[datetime.datetime] = Unassigned()


class ExplainabilityTaskContext(Base):
    """
    ExplainabilityTaskContext

    Attributes
    ----------------------
    candidate_name
    include_pdp
    overwrite_artifacts
    """

    candidate_name: StrPipeVar
    include_pdp: Optional[bool] = Unassigned()
    overwrite_artifacts: Optional[bool] = Unassigned()


class ModelInsightsTaskContext(Base):
    """
    ModelInsightsTaskContext

    Attributes
    ----------------------
    candidate_name
    """

    candidate_name: StrPipeVar


class AutoMLTaskContext(Base):
    """
    AutoMLTaskContext

    Attributes
    ----------------------
    explainability_task_context
    model_insights_task_context
    """

    explainability_task_context: Optional[ExplainabilityTaskContext] = Unassigned()
    model_insights_task_context: Optional[ModelInsightsTaskContext] = Unassigned()


class AutoParameter(Base):
    """
    AutoParameter
      The name and an example value of the hyperparameter that you want to use in Autotune. If Automatic model tuning (AMT) determines that your hyperparameter is eligible for Autotune, an optimal hyperparameter range is selected for you.

    Attributes
    ----------------------
    name: The name of the hyperparameter to optimize using Autotune.
    value_hint: An example value of the hyperparameter to optimize using Autotune.
    """

    name: StrPipeVar
    value_hint: StrPipeVar


class AutoRollbackConfig(Base):
    """
    AutoRollbackConfig
      Automatic rollback configuration for handling endpoint deployment failures and recovery.

    Attributes
    ----------------------
    alarms: List of CloudWatch alarms in your account that are configured to monitor metrics on an endpoint. If any alarms are tripped during a deployment, SageMaker rolls back the deployment.
    """

    alarms: Optional[List[Alarm]] = Unassigned()


class Autotune(Base):
    """
    Autotune
      A flag to indicate if you want to use Autotune to automatically find optimal values for the following fields:    ParameterRanges: The names and ranges of parameters that a hyperparameter tuning job can optimize.    ResourceLimits: The maximum resources that can be used for a training job. These resources include the maximum number of training jobs, the maximum runtime of a tuning job, and the maximum number of training jobs to run at the same time.    TrainingJobEarlyStoppingType: A flag that specifies whether or not to use early stopping for training jobs launched by a hyperparameter tuning job.    RetryStrategy: The number of times to retry a training job.    Strategy: Specifies how hyperparameter tuning chooses the combinations of hyperparameter values to use for the training jobs that it launches.    ConvergenceDetected: A flag to indicate that Automatic model tuning (AMT) has detected model convergence.

    Attributes
    ----------------------
    mode: Set Mode to Enabled if you want to use Autotune.
    """

    mode: StrPipeVar


class AvailableUpgrade(Base):
    """
    AvailableUpgrade
      Contains information about an available upgrade for a SageMaker Partner AI App, including the version number and release notes.

    Attributes
    ----------------------
    version: The semantic version number of the available upgrade for the SageMaker Partner AI App.
    release_notes: A list of release notes describing the changes and improvements included in the available upgrade version.
    """

    version: Optional[StrPipeVar] = Unassigned()
    release_notes: Optional[List[StrPipeVar]] = Unassigned()


class BatchAddClusterNodesError(Base):
    """
    BatchAddClusterNodesError
      Information about an error that occurred during the node addition operation.

    Attributes
    ----------------------
    instance_group_name: The name of the instance group for which the error occurred.
    error_code: The error code associated with the failure. Possible values include InstanceGroupNotFound and InvalidInstanceGroupState.
    failed_count: The number of nodes that failed to be added to the specified instance group.
    message: A descriptive message providing additional details about the error.
    """

    instance_group_name: StrPipeVar
    error_code: StrPipeVar
    failed_count: int
    message: Optional[StrPipeVar] = Unassigned()


class NodeAdditionResult(Base):
    """
    NodeAdditionResult
      Information about a node that was successfully added to the cluster.

    Attributes
    ----------------------
    node_logical_id: A unique identifier assigned to the node that can be used to track its provisioning status through the DescribeClusterNode operation.
    instance_group_name: The name of the instance group to which the node was added.
    status: The current status of the node. Possible values include Pending, Running, Failed, ShuttingDown, SystemUpdating, DeepHealthCheckInProgress, and NotFound.
    """

    node_logical_id: StrPipeVar
    instance_group_name: StrPipeVar
    status: StrPipeVar


class BatchDataCaptureConfig(Base):
    """
    BatchDataCaptureConfig
      Configuration to control how SageMaker captures inference data for batch transform jobs.

    Attributes
    ----------------------
    destination_s3_uri: The Amazon S3 location being used to capture the data.
    kms_key_id: The Amazon Resource Name (ARN) of a Amazon Web Services Key Management Service key that SageMaker uses to encrypt data on the storage volume attached to the ML compute instance that hosts the batch transform job. The KmsKeyId can be any of the following formats:    Key ID: 1234abcd-12ab-34cd-56ef-1234567890ab    Key ARN: arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab    Alias name: alias/ExampleAlias    Alias name ARN: arn:aws:kms:us-west-2:111122223333:alias/ExampleAlias
    generate_inference_id: Flag that indicates whether to append inference id to the output.
    """

    destination_s3_uri: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    generate_inference_id: Optional[bool] = Unassigned()


class BatchDeleteClusterNodeLogicalIdsError(Base):
    """
    BatchDeleteClusterNodeLogicalIdsError
      Information about an error that occurred when attempting to delete a node identified by its NodeLogicalId.

    Attributes
    ----------------------
    code: The error code associated with the failure. Possible values include NodeLogicalIdNotFound, InvalidNodeStatus, and InternalError.
    message: A descriptive message providing additional details about the error.
    node_logical_id: The NodeLogicalId of the node that could not be deleted.
    """

    code: StrPipeVar
    message: StrPipeVar
    node_logical_id: StrPipeVar


class BatchDeleteClusterNodesError(Base):
    """
    BatchDeleteClusterNodesError
      Represents an error encountered when deleting a node from a SageMaker HyperPod cluster.

    Attributes
    ----------------------
    code: The error code associated with the error encountered when deleting a node. The code provides information about the specific issue encountered, such as the node not being found, the node's status being invalid for deletion, or the node ID being in use by another process.
    message: A message describing the error encountered when deleting a node.
    node_id: The ID of the node that encountered an error during the deletion process.
    """

    code: StrPipeVar
    message: StrPipeVar
    node_id: StrPipeVar


class BatchDeleteClusterNodesResponse(Base):
    """
    BatchDeleteClusterNodesResponse

    Attributes
    ----------------------
    failed: A list of errors encountered when deleting the specified nodes.
    successful: A list of node IDs that were successfully deleted from the specified cluster.
    failed_node_logical_ids: A list of NodeLogicalIds that could not be deleted, along with error information explaining why the deletion failed.
    successful_node_logical_ids: A list of NodeLogicalIds that were successfully deleted from the cluster.
    """

    failed: Optional[List[BatchDeleteClusterNodesError]] = Unassigned()
    successful: Optional[List[StrPipeVar]] = Unassigned()
    failed_node_logical_ids: Optional[List[BatchDeleteClusterNodeLogicalIdsError]] = Unassigned()
    successful_node_logical_ids: Optional[List[StrPipeVar]] = Unassigned()


class BatchDescribeModelPackageError(Base):
    """
    BatchDescribeModelPackageError
      The error code and error description associated with the resource.

    Attributes
    ----------------------
    error_code:
    error_response:
    """

    error_code: StrPipeVar
    error_response: StrPipeVar


class InferenceSpecification(Base):
    """
    InferenceSpecification
      Defines how to perform inference generation after a training job is run.

    Attributes
    ----------------------
    containers: The Amazon ECR registry path of the Docker image that contains the inference code.
    supported_transform_instance_types: A list of the instance types on which a transformation job can be run or on which an endpoint can be deployed. This parameter is required for unversioned models, and optional for versioned models.
    supported_realtime_inference_instance_types: A list of the instance types that are used to generate inferences in real-time. This parameter is required for unversioned models, and optional for versioned models.
    supported_content_types: The supported MIME types for the input data.
    supported_response_mime_types: The supported MIME types for the output data.
    """

    containers: List[ModelPackageContainerDefinition]
    supported_transform_instance_types: Optional[List[StrPipeVar]] = Unassigned()
    supported_realtime_inference_instance_types: Optional[List[StrPipeVar]] = Unassigned()
    supported_content_types: Optional[List[StrPipeVar]] = Unassigned()
    supported_response_mime_types: Optional[List[StrPipeVar]] = Unassigned()


class BatchDescribeModelPackageSummary(Base):
    """
    BatchDescribeModelPackageSummary
      Provides summary information about the model package.

    Attributes
    ----------------------
    model_package_group_name: The group name for the model package
    model_package_version: The version number of a versioned model.
    model_package_arn: The Amazon Resource Name (ARN) of the model package.
    model_package_description: The description of the model package.
    creation_time: The creation time of the mortgage package summary.
    inference_specification
    model_package_status: The status of the mortgage package.
    model_approval_status: The approval status of the model.
    model_package_registration_type
    """

    model_package_group_name: Union[StrPipeVar, object]
    model_package_arn: StrPipeVar
    creation_time: datetime.datetime
    inference_specification: InferenceSpecification
    model_package_status: StrPipeVar
    model_package_version: Optional[int] = Unassigned()
    model_package_description: Optional[StrPipeVar] = Unassigned()
    model_approval_status: Optional[StrPipeVar] = Unassigned()
    model_package_registration_type: Optional[StrPipeVar] = Unassigned()


class BatchDescribeModelPackageOutput(Base):
    """
    BatchDescribeModelPackageOutput

    Attributes
    ----------------------
    model_package_summaries: The summaries for the model package versions
    batch_describe_model_package_error_map: A map of the resource and BatchDescribeModelPackageError objects reporting the error associated with describing the model package.
    """

    model_package_summaries: Optional[Dict[StrPipeVar, BatchDescribeModelPackageSummary]] = (
        Unassigned()
    )
    batch_describe_model_package_error_map: Optional[
        Dict[StrPipeVar, BatchDescribeModelPackageError]
    ] = Unassigned()


class BatchRebootClusterNodeLogicalIdsError(Base):
    """
    BatchRebootClusterNodeLogicalIdsError

    Attributes
    ----------------------
    node_logical_id
    error_code
    message
    """

    node_logical_id: StrPipeVar
    error_code: StrPipeVar
    message: StrPipeVar


class BatchRebootClusterNodesError(Base):
    """
    BatchRebootClusterNodesError

    Attributes
    ----------------------
    node_id
    error_code
    message
    """

    node_id: StrPipeVar
    error_code: StrPipeVar
    message: StrPipeVar


class BatchRepairClusterNodesError(Base):
    """
    BatchRepairClusterNodesError

    Attributes
    ----------------------
    repair_action
    node_id
    message
    code
    """

    repair_action: StrPipeVar
    node_id: StrPipeVar
    message: StrPipeVar
    code: StrPipeVar


class RepairNodeItem(Base):
    """
    RepairNodeItem

    Attributes
    ----------------------
    node_ids
    repair_action
    """

    node_ids: List[StrPipeVar]
    repair_action: StrPipeVar


class BatchRepairClusterNodesSuccess(Base):
    """
    BatchRepairClusterNodesSuccess

    Attributes
    ----------------------
    repair_action
    node_id
    """

    repair_action: StrPipeVar
    node_id: StrPipeVar


class BatchReplaceClusterNodeLogicalIdsError(Base):
    """
    BatchReplaceClusterNodeLogicalIdsError

    Attributes
    ----------------------
    node_logical_id
    error_code
    message
    """

    node_logical_id: StrPipeVar
    error_code: StrPipeVar
    message: StrPipeVar


class BatchReplaceClusterNodesError(Base):
    """
    BatchReplaceClusterNodesError

    Attributes
    ----------------------
    node_id
    error_code
    message
    """

    node_id: StrPipeVar
    error_code: StrPipeVar
    message: StrPipeVar


class MonitoringCsvDatasetFormat(Base):
    """
    MonitoringCsvDatasetFormat
      Represents the CSV dataset format used when running a monitoring job.

    Attributes
    ----------------------
    header: Indicates if the CSV data has a header.
    compressed
    """

    header: Optional[bool] = Unassigned()
    compressed: Optional[bool] = Unassigned()


class MonitoringJsonDatasetFormat(Base):
    """
    MonitoringJsonDatasetFormat
      Represents the JSON dataset format used when running a monitoring job.

    Attributes
    ----------------------
    line: Indicates if the file should be read as a JSON object per line.
    compressed
    """

    line: Optional[bool] = Unassigned()
    compressed: Optional[bool] = Unassigned()


class MonitoringParquetDatasetFormat(Base):
    """
    MonitoringParquetDatasetFormat
      Represents the Parquet dataset format used when running a monitoring job.

    Attributes
    ----------------------
    """


class MonitoringDatasetFormat(Base):
    """
    MonitoringDatasetFormat
      Represents the dataset format used when running a monitoring job.

    Attributes
    ----------------------
    csv: The CSV dataset used in the monitoring job.
    json: The JSON dataset used in the monitoring job
    parquet: The Parquet dataset used in the monitoring job
    """

    csv: Optional[MonitoringCsvDatasetFormat] = Unassigned()
    json: Optional[MonitoringJsonDatasetFormat] = Unassigned()
    parquet: Optional[MonitoringParquetDatasetFormat] = Unassigned()


class BatchTransformInput(Base):
    """
    BatchTransformInput
      Input object for the batch transform job.

    Attributes
    ----------------------
    data_captured_destination_s3_uri: The Amazon S3 location being used to capture the data.
    dataset_format: The dataset format for your batch transform job.
    local_path: Path to the filesystem where the batch transform data is available to the container.
    s3_input_mode: Whether the Pipe or File is used as the input mode for transferring data for the monitoring job. Pipe mode is recommended for large datasets. File mode is useful for small files that fit in memory. Defaults to File.
    s3_data_distribution_type: Whether input data distributed in Amazon S3 is fully replicated or sharded by an S3 key. Defaults to FullyReplicated
    features_attribute: The attributes of the input data that are the input features.
    inference_attribute: The attribute of the input data that represents the ground truth label.
    probability_attribute: In a classification problem, the attribute that represents the class probability.
    probability_threshold_attribute: The threshold for the class probability to be evaluated as a positive result.
    start_time_offset: If specified, monitoring jobs substract this time from the start time. For information about using offsets for scheduling monitoring jobs, see Schedule Model Quality Monitoring Jobs.
    end_time_offset: If specified, monitoring jobs subtract this time from the end time. For information about using offsets for scheduling monitoring jobs, see Schedule Model Quality Monitoring Jobs.
    exclude_features_attribute: The attributes of the input data to exclude from the analysis.
    """

    data_captured_destination_s3_uri: StrPipeVar
    dataset_format: MonitoringDatasetFormat
    local_path: StrPipeVar
    s3_input_mode: Optional[StrPipeVar] = Unassigned()
    s3_data_distribution_type: Optional[StrPipeVar] = Unassigned()
    features_attribute: Optional[StrPipeVar] = Unassigned()
    inference_attribute: Optional[StrPipeVar] = Unassigned()
    probability_attribute: Optional[StrPipeVar] = Unassigned()
    probability_threshold_attribute: Optional[float] = Unassigned()
    start_time_offset: Optional[StrPipeVar] = Unassigned()
    end_time_offset: Optional[StrPipeVar] = Unassigned()
    exclude_features_attribute: Optional[StrPipeVar] = Unassigned()


class BedrockCustomModelDeploymentMetadata(Base):
    """
    BedrockCustomModelDeploymentMetadata

    Attributes
    ----------------------
    arn
    """

    arn: Optional[StrPipeVar] = Unassigned()


class BedrockCustomModelMetadata(Base):
    """
    BedrockCustomModelMetadata

    Attributes
    ----------------------
    arn
    """

    arn: Optional[StrPipeVar] = Unassigned()


class BedrockModelImportMetadata(Base):
    """
    BedrockModelImportMetadata

    Attributes
    ----------------------
    arn
    """

    arn: Optional[StrPipeVar] = Unassigned()


class BedrockProvisionedModelThroughputMetadata(Base):
    """
    BedrockProvisionedModelThroughputMetadata

    Attributes
    ----------------------
    arn
    """

    arn: Optional[StrPipeVar] = Unassigned()


class BenchmarkResultsOutputConfig(Base):
    """
    BenchmarkResultsOutputConfig

    Attributes
    ----------------------
    s3_output_uri
    """

    s3_output_uri: Optional[StrPipeVar] = Unassigned()


class BestObjectiveNotImproving(Base):
    """
    BestObjectiveNotImproving
      A structure that keeps track of which training jobs launched by your hyperparameter tuning job are not improving model performance as evaluated against an objective function.

    Attributes
    ----------------------
    max_number_of_training_jobs_not_improving: The number of training jobs that have failed to improve model performance by 1% or greater over prior training jobs as evaluated against an objective function.
    """

    max_number_of_training_jobs_not_improving: Optional[int] = Unassigned()


class MetricsSource(Base):
    """
    MetricsSource
      Details about the metrics source.

    Attributes
    ----------------------
    content_type: The metric source content type.
    content_digest: The hash key used for the metrics source.
    s3_uri: The S3 URI for the metrics source.
    """

    content_type: StrPipeVar
    s3_uri: StrPipeVar
    content_digest: Optional[StrPipeVar] = Unassigned()


class Bias(Base):
    """
    Bias
      Contains bias metrics for a model.

    Attributes
    ----------------------
    report: The bias report for a model
    pre_training_report: The pre-training bias report for a model.
    post_training_report: The post-training bias report for a model.
    """

    report: Optional[MetricsSource] = Unassigned()
    pre_training_report: Optional[MetricsSource] = Unassigned()
    post_training_report: Optional[MetricsSource] = Unassigned()


class CapacitySize(Base):
    """
    CapacitySize
      Specifies the type and size of the endpoint capacity to activate for a blue/green deployment, a rolling deployment, or a rollback strategy. You can specify your batches as either instance count or the overall percentage or your fleet. For a rollback strategy, if you don't specify the fields in this object, or if you set the Value to 100%, then SageMaker uses a blue/green rollback strategy and rolls all traffic back to the blue fleet.

    Attributes
    ----------------------
    type: Specifies the endpoint capacity type.    INSTANCE_COUNT: The endpoint activates based on the number of instances.    CAPACITY_PERCENT: The endpoint activates based on the specified percentage of capacity.
    value: Defines the capacity size, either as a number of instances or a capacity percentage.
    """

    type: StrPipeVar
    value: int


class TrafficRoutingConfig(Base):
    """
    TrafficRoutingConfig
      Defines the traffic routing strategy during an endpoint deployment to shift traffic from the old fleet to the new fleet.

    Attributes
    ----------------------
    type: Traffic routing strategy type.    ALL_AT_ONCE: Endpoint traffic shifts to the new fleet in a single step.     CANARY: Endpoint traffic shifts to the new fleet in two steps. The first step is the canary, which is a small portion of the traffic. The second step is the remainder of the traffic.     LINEAR: Endpoint traffic shifts to the new fleet in n steps of a configurable size.
    wait_interval_in_seconds: The waiting time (in seconds) between incremental steps to turn on traffic on the new endpoint fleet.
    canary_size: Batch size for the first step to turn on traffic on the new endpoint fleet. Value must be less than or equal to 50% of the variant's total instance count.
    linear_step_size: Batch size for each step to turn on traffic on the new endpoint fleet. Value must be 10-50% of the variant's total instance count.
    """

    type: StrPipeVar
    wait_interval_in_seconds: int
    canary_size: Optional[CapacitySize] = Unassigned()
    linear_step_size: Optional[CapacitySize] = Unassigned()


class BlueGreenUpdatePolicy(Base):
    """
    BlueGreenUpdatePolicy
      Update policy for a blue/green deployment. If this update policy is specified, SageMaker creates a new fleet during the deployment while maintaining the old fleet. SageMaker flips traffic to the new fleet according to the specified traffic routing configuration. Only one update policy should be used in the deployment configuration. If no update policy is specified, SageMaker uses a blue/green deployment strategy with all at once traffic shifting by default.

    Attributes
    ----------------------
    traffic_routing_configuration: Defines the traffic routing strategy to shift traffic from the old fleet to the new fleet during an endpoint deployment.
    termination_wait_in_seconds: Additional waiting time in seconds after the completion of an endpoint deployment before terminating the old endpoint fleet. Default is 0.
    maximum_execution_timeout_in_seconds: Maximum execution timeout for the deployment. Note that the timeout value should be larger than the total waiting time specified in TerminationWaitInSeconds and WaitIntervalInSeconds.
    """

    traffic_routing_configuration: TrafficRoutingConfig
    termination_wait_in_seconds: Optional[int] = Unassigned()
    maximum_execution_timeout_in_seconds: Optional[int] = Unassigned()


class BurstLimit(Base):
    """
    BurstLimit

    Attributes
    ----------------------
    allow_unlimited_burst
    burst_multiplier
    """

    allow_unlimited_burst: Optional[bool] = Unassigned()
    burst_multiplier: Optional[int] = Unassigned()


class CacheHitResult(Base):
    """
    CacheHitResult
      Details on the cache hit of a pipeline execution step.

    Attributes
    ----------------------
    source_pipeline_execution_arn: The Amazon Resource Name (ARN) of the pipeline execution.
    """

    source_pipeline_execution_arn: Optional[StrPipeVar] = Unassigned()


class OutputParameter(Base):
    """
    OutputParameter
      An output parameter of a pipeline step.

    Attributes
    ----------------------
    name: The name of the output parameter.
    value: The value of the output parameter.
    """

    name: StrPipeVar
    value: StrPipeVar


class CallbackStepMetadata(Base):
    """
    CallbackStepMetadata
      Metadata about a callback step.

    Attributes
    ----------------------
    callback_token: The pipeline generated token from the Amazon SQS queue.
    sqs_queue_url: The URL of the Amazon Simple Queue Service (Amazon SQS) queue used by the callback step.
    output_parameters: A list of the output parameters of the callback step.
    """

    callback_token: Optional[StrPipeVar] = Unassigned()
    sqs_queue_url: Optional[StrPipeVar] = Unassigned()
    output_parameters: Optional[List[OutputParameter]] = Unassigned()


class TimeSeriesForecastingSettings(Base):
    """
    TimeSeriesForecastingSettings
      Time series forecast settings for the SageMaker Canvas application.

    Attributes
    ----------------------
    status: Describes whether time series forecasting is enabled or disabled in the Canvas application.
    amazon_forecast_role_arn: The IAM role that Canvas passes to Amazon Forecast for time series forecasting. By default, Canvas uses the execution role specified in the UserProfile that launches the Canvas application. If an execution role is not specified in the UserProfile, Canvas uses the execution role specified in the Domain that owns the UserProfile. To allow time series forecasting, this IAM role should have the  AmazonSageMakerCanvasForecastAccess policy attached and forecast.amazonaws.com added in the trust relationship as a service principal.
    """

    status: Optional[StrPipeVar] = Unassigned()
    amazon_forecast_role_arn: Optional[StrPipeVar] = Unassigned()


class ModelRegisterSettings(Base):
    """
    ModelRegisterSettings
      The model registry settings for the SageMaker Canvas application.

    Attributes
    ----------------------
    status: Describes whether the integration to the model registry is enabled or disabled in the Canvas application.
    cross_account_model_register_role_arn: The Amazon Resource Name (ARN) of the SageMaker model registry account. Required only to register model versions created by a different SageMaker Canvas Amazon Web Services account than the Amazon Web Services account in which SageMaker model registry is set up.
    """

    status: Optional[StrPipeVar] = Unassigned()
    cross_account_model_register_role_arn: Optional[StrPipeVar] = Unassigned()


class WorkspaceSettings(Base):
    """
    WorkspaceSettings
      The workspace settings for the SageMaker Canvas application.

    Attributes
    ----------------------
    s3_artifact_path: The Amazon S3 bucket used to store artifacts generated by Canvas. Updating the Amazon S3 location impacts existing configuration settings, and Canvas users no longer have access to their artifacts. Canvas users must log out and log back in to apply the new location.
    s3_kms_key_id: The Amazon Web Services Key Management Service (KMS) encryption key ID that is used to encrypt artifacts generated by Canvas in the Amazon S3 bucket.
    """

    s3_artifact_path: Optional[StrPipeVar] = Unassigned()
    s3_kms_key_id: Optional[StrPipeVar] = Unassigned()


class IdentityProviderOAuthSetting(Base):
    """
    IdentityProviderOAuthSetting
      The Amazon SageMaker Canvas application setting where you configure OAuth for connecting to an external data source, such as Snowflake.

    Attributes
    ----------------------
    data_source_name: The name of the data source that you're connecting to. Canvas currently supports OAuth for Snowflake and Salesforce Data Cloud.
    status: Describes whether OAuth for a data source is enabled or disabled in the Canvas application.
    secret_arn: The ARN of an Amazon Web Services Secrets Manager secret that stores the credentials from your identity provider, such as the client ID and secret, authorization URL, and token URL.
    """

    data_source_name: Optional[StrPipeVar] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    secret_arn: Optional[StrPipeVar] = Unassigned()


class DirectDeploySettings(Base):
    """
    DirectDeploySettings
      The model deployment settings for the SageMaker Canvas application.  In order to enable model deployment for Canvas, the SageMaker Domain's or user profile's Amazon Web Services IAM execution role must have the AmazonSageMakerCanvasDirectDeployAccess policy attached. You can also turn on model deployment permissions through the SageMaker Domain's or user profile's settings in the SageMaker console.

    Attributes
    ----------------------
    status: Describes whether model deployment permissions are enabled or disabled in the Canvas application.
    """

    status: Optional[StrPipeVar] = Unassigned()


class KendraSettings(Base):
    """
    KendraSettings
      The Amazon SageMaker Canvas application setting where you configure document querying.

    Attributes
    ----------------------
    status: Describes whether the document querying feature is enabled or disabled in the Canvas application.
    index_id_list
    """

    status: Optional[StrPipeVar] = Unassigned()
    index_id_list: Optional[List[StrPipeVar]] = Unassigned()


class GenerativeAiSettings(Base):
    """
    GenerativeAiSettings
      The generative AI settings for the SageMaker Canvas application. Configure these settings for Canvas users starting chats with generative AI foundation models. For more information, see  Use generative AI with foundation models.

    Attributes
    ----------------------
    amazon_bedrock_role_arn: The ARN of an Amazon Web Services IAM role that allows fine-tuning of large language models (LLMs) in Amazon Bedrock. The IAM role should have Amazon S3 read and write permissions, as well as a trust relationship that establishes bedrock.amazonaws.com as a service principal.
    """

    amazon_bedrock_role_arn: Optional[StrPipeVar] = Unassigned()


class EmrServerlessSettings(Base):
    """
    EmrServerlessSettings
      The settings for running Amazon EMR Serverless jobs in SageMaker Canvas.

    Attributes
    ----------------------
    execution_role_arn: The Amazon Resource Name (ARN) of the Amazon Web Services IAM role that is assumed for running Amazon EMR Serverless jobs in SageMaker Canvas. This role should have the necessary permissions to read and write data attached and a trust relationship with EMR Serverless.
    status: Describes whether Amazon EMR Serverless job capabilities are enabled or disabled in the SageMaker Canvas application.
    """

    execution_role_arn: Optional[StrPipeVar] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()


class DataScienceAssistantSettings(Base):
    """
    DataScienceAssistantSettings

    Attributes
    ----------------------
    status
    cross_region_q_service_status
    """

    status: Optional[StrPipeVar] = Unassigned()
    cross_region_q_service_status: Optional[StrPipeVar] = Unassigned()


class CanvasAppSettings(Base):
    """
    CanvasAppSettings
      The SageMaker Canvas application settings.

    Attributes
    ----------------------
    time_series_forecasting_settings: Time series forecast settings for the SageMaker Canvas application.
    model_register_settings: The model registry settings for the SageMaker Canvas application.
    workspace_settings: The workspace settings for the SageMaker Canvas application.
    identity_provider_o_auth_settings: The settings for connecting to an external data source with OAuth.
    direct_deploy_settings: The model deployment settings for the SageMaker Canvas application.
    kendra_settings: The settings for document querying.
    generative_ai_settings: The generative AI settings for the SageMaker Canvas application.
    emr_serverless_settings: The settings for running Amazon EMR Serverless data processing jobs in SageMaker Canvas.
    data_science_assistant_settings
    """

    time_series_forecasting_settings: Optional[TimeSeriesForecastingSettings] = Unassigned()
    model_register_settings: Optional[ModelRegisterSettings] = Unassigned()
    workspace_settings: Optional[WorkspaceSettings] = Unassigned()
    identity_provider_o_auth_settings: Optional[List[IdentityProviderOAuthSetting]] = Unassigned()
    direct_deploy_settings: Optional[DirectDeploySettings] = Unassigned()
    kendra_settings: Optional[KendraSettings] = Unassigned()
    generative_ai_settings: Optional[GenerativeAiSettings] = Unassigned()
    emr_serverless_settings: Optional[EmrServerlessSettings] = Unassigned()
    data_science_assistant_settings: Optional[DataScienceAssistantSettings] = Unassigned()


class CapacityBlockOffering(Base):
    """
    CapacityBlockOffering

    Attributes
    ----------------------
    capacity_block_duration_in_hours
    start_time
    end_time
    upfront_fee
    currency_code
    availability_zone
    """

    capacity_block_duration_in_hours: int
    upfront_fee: StrPipeVar
    currency_code: StrPipeVar
    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()
    availability_zone: Optional[StrPipeVar] = Unassigned()


class CapacityReservation(Base):
    """
    CapacityReservation
      Information about the Capacity Reservation used by an instance or instance group.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the Capacity Reservation.
    type: The type of Capacity Reservation. Valid values are ODCR (On-Demand Capacity Reservation) or CRG (Capacity Reservation Group).
    """

    arn: Optional[StrPipeVar] = Unassigned()
    type: Optional[StrPipeVar] = Unassigned()


class CapacityResources(Base):
    """
    CapacityResources

    Attributes
    ----------------------
    capacity_block_offerings
    capacity_resource_arn
    """

    capacity_block_offerings: Optional[List[CapacityBlockOffering]] = Unassigned()
    capacity_resource_arn: Optional[StrPipeVar] = Unassigned()


class CapacityScheduleStatusTransition(Base):
    """
    CapacityScheduleStatusTransition

    Attributes
    ----------------------
    status
    start_time
    end_time
    status_message
    """

    status: StrPipeVar
    start_time: datetime.datetime
    status_message: StrPipeVar
    end_time: Optional[datetime.datetime] = Unassigned()


class CapacityScheduleDetail(Base):
    """
    CapacityScheduleDetail

    Attributes
    ----------------------
    capacity_schedule_arn
    owner_account_id
    capacity_schedule_type
    instance_type
    total_instance_count
    available_instance_count
    availability_zone_distribution
    placement
    availability_zone
    status
    requested_start_time
    requested_end_time
    start_time
    end_time
    duration_in_hours
    capacity_block_offerings
    capacity_resources
    target_resources
    capacity_schedule_status_transitions
    """

    capacity_schedule_arn: StrPipeVar
    capacity_schedule_type: StrPipeVar
    instance_type: StrPipeVar
    total_instance_count: int
    placement: StrPipeVar
    status: StrPipeVar
    requested_start_time: datetime.datetime
    owner_account_id: Optional[StrPipeVar] = Unassigned()
    available_instance_count: Optional[int] = Unassigned()
    availability_zone_distribution: Optional[StrPipeVar] = Unassigned()
    availability_zone: Optional[StrPipeVar] = Unassigned()
    requested_end_time: Optional[datetime.datetime] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()
    duration_in_hours: Optional[int] = Unassigned()
    capacity_block_offerings: Optional[List[CapacityBlockOffering]] = Unassigned()
    capacity_resources: Optional[CapacityResources] = Unassigned()
    target_resources: Optional[List[StrPipeVar]] = Unassigned()
    capacity_schedule_status_transitions: Optional[List[CapacityScheduleStatusTransition]] = (
        Unassigned()
    )


class CapacityScheduleFilter(Base):
    """
    CapacityScheduleFilter

    Attributes
    ----------------------
    name
    value
    """

    name: StrPipeVar
    value: StrPipeVar


class CapacityScheduleOffering(Base):
    """
    CapacityScheduleOffering

    Attributes
    ----------------------
    capacity_schedule_offering_id
    capacity_schedule_type
    eligible_resources
    instance_type
    instance_count
    placement
    requested_start_time
    requested_end_time
    availability_zones
    availability_zone_distribution
    duration_in_hours
    capacity_block_offerings
    """

    capacity_schedule_offering_id: StrPipeVar
    capacity_schedule_type: StrPipeVar
    instance_type: StrPipeVar
    instance_count: int
    requested_start_time: datetime.datetime
    eligible_resources: Optional[List[StrPipeVar]] = Unassigned()
    placement: Optional[StrPipeVar] = Unassigned()
    requested_end_time: Optional[datetime.datetime] = Unassigned()
    availability_zones: Optional[List[StrPipeVar]] = Unassigned()
    availability_zone_distribution: Optional[StrPipeVar] = Unassigned()
    duration_in_hours: Optional[int] = Unassigned()
    capacity_block_offerings: Optional[List[CapacityBlockOffering]] = Unassigned()


class CapacitySizeConfig(Base):
    """
    CapacitySizeConfig
      The configuration of the size measurements of the AMI update. Using this configuration, you can specify whether SageMaker should update your instance group by an amount or percentage of instances.

    Attributes
    ----------------------
    type: Specifies whether SageMaker should process the update by amount or percentage of instances.
    value: Specifies the amount or percentage of instances SageMaker updates at a time.
    """

    type: StrPipeVar
    value: int


class CaptureContainerConfig(Base):
    """
    CaptureContainerConfig

    Attributes
    ----------------------
    container_hostname
    """

    container_hostname: StrPipeVar


class CaptureContentTypeHeader(Base):
    """
    CaptureContentTypeHeader
      Configuration specifying how to treat different headers. If no headers are specified Amazon SageMaker AI will by default base64 encode when capturing the data.

    Attributes
    ----------------------
    csv_content_types: The list of all content type headers that Amazon SageMaker AI will treat as CSV and capture accordingly.
    json_content_types: The list of all content type headers that SageMaker AI will treat as JSON and capture accordingly.
    """

    csv_content_types: Optional[List[StrPipeVar]] = Unassigned()
    json_content_types: Optional[List[StrPipeVar]] = Unassigned()


class CaptureOption(Base):
    """
    CaptureOption
      Specifies data Model Monitor will capture.

    Attributes
    ----------------------
    capture_mode: Specify the boundary of data to capture.
    capture_boundary
    capture_containers
    """

    capture_mode: StrPipeVar
    capture_boundary: Optional[StrPipeVar] = Unassigned()
    capture_containers: Optional[List[CaptureContainerConfig]] = Unassigned()


class CategoricalParameter(Base):
    """
    CategoricalParameter
      Environment parameters you want to benchmark your load test against.

    Attributes
    ----------------------
    name: The Name of the environment variable.
    value: The list of values you can pass.
    """

    name: StrPipeVar
    value: List[StrPipeVar]


class CategoricalParameterRange(Base):
    """
    CategoricalParameterRange
      A list of categorical hyperparameters to tune.

    Attributes
    ----------------------
    name: The name of the categorical hyperparameter to tune.
    values: A list of the categories for the hyperparameter.
    """

    name: StrPipeVar
    values: List[StrPipeVar]


class CategoricalParameterRangeSpecification(Base):
    """
    CategoricalParameterRangeSpecification
      Defines the possible values for a categorical hyperparameter.

    Attributes
    ----------------------
    values: The allowed categories for the hyperparameter.
    """

    values: List[StrPipeVar]


class CfnStackCreateParameter(Base):
    """
    CfnStackCreateParameter
       A key-value pair that represents a parameter for the CloudFormation stack.

    Attributes
    ----------------------
    key:  The name of the CloudFormation parameter.
    value:  The value of the CloudFormation parameter.
    """

    key: StrPipeVar
    value: Optional[StrPipeVar] = Unassigned()


class CfnCreateTemplateProvider(Base):
    """
    CfnCreateTemplateProvider
       The CloudFormation template provider configuration for creating infrastructure resources.

    Attributes
    ----------------------
    template_name:  A unique identifier for the template within the project.
    template_url:  The Amazon S3 URL of the CloudFormation template.
    role_arn:  The IAM role that CloudFormation assumes when creating the stack.
    parameters:  An array of CloudFormation stack parameters.
    """

    template_name: StrPipeVar
    template_url: StrPipeVar
    role_arn: Optional[StrPipeVar] = Unassigned()
    parameters: Optional[List[CfnStackCreateParameter]] = Unassigned()


class CfnStackDetail(Base):
    """
    CfnStackDetail
       Details about the CloudFormation stack.

    Attributes
    ----------------------
    name:  The name of the CloudFormation stack.
    id:  The unique identifier of the CloudFormation stack.
    status_message:  A human-readable message about the stack's current status.
    """

    status_message: StrPipeVar
    name: Optional[StrPipeVar] = Unassigned()
    id: Optional[StrPipeVar] = Unassigned()


class CfnStackParameter(Base):
    """
    CfnStackParameter
       A key-value pair representing a parameter used in the CloudFormation stack.

    Attributes
    ----------------------
    key:  The name of the CloudFormation parameter.
    value:  The value of the CloudFormation parameter.
    """

    key: StrPipeVar
    value: Optional[StrPipeVar] = Unassigned()


class CfnStackUpdateParameter(Base):
    """
    CfnStackUpdateParameter
       A key-value pair representing a parameter used in the CloudFormation stack.

    Attributes
    ----------------------
    key:  The name of the CloudFormation parameter.
    value:  The value of the CloudFormation parameter.
    """

    key: StrPipeVar
    value: Optional[StrPipeVar] = Unassigned()


class CfnTemplateProviderDetail(Base):
    """
    CfnTemplateProviderDetail
       Details about a CloudFormation template provider configuration and associated provisioning information.

    Attributes
    ----------------------
    template_name:  The unique identifier of the template within the project.
    template_url:  The Amazon S3 URL of the CloudFormation template.
    role_arn:  The IAM role used by CloudFormation to create the stack.
    parameters:  An array of CloudFormation stack parameters.
    stack_detail:  Information about the CloudFormation stack created by the template provider.
    """

    template_name: StrPipeVar
    template_url: StrPipeVar
    role_arn: Optional[StrPipeVar] = Unassigned()
    parameters: Optional[List[CfnStackParameter]] = Unassigned()
    stack_detail: Optional[CfnStackDetail] = Unassigned()


class CfnUpdateTemplateProvider(Base):
    """
    CfnUpdateTemplateProvider
       Contains configuration details for updating an existing CloudFormation template provider in the project.

    Attributes
    ----------------------
    template_name:  The unique identifier of the template to update within the project.
    template_url:  The Amazon S3 URL of the CloudFormation template.
    parameters:  An array of CloudFormation stack parameters.
    """

    template_name: StrPipeVar
    template_url: StrPipeVar
    parameters: Optional[List[CfnStackUpdateParameter]] = Unassigned()


class ChannelSpecification(Base):
    """
    ChannelSpecification
      Defines a named input source, called a channel, to be used by an algorithm.

    Attributes
    ----------------------
    name: The name of the channel.
    description: A brief description of the channel.
    is_required: Indicates whether the channel is required by the algorithm.
    supported_content_types: The supported MIME types for the data.
    supported_compression_types: The allowed compression types, if data compression is used.
    supported_input_modes: The allowed input mode, either FILE or PIPE. In FILE mode, Amazon SageMaker copies the data from the input source onto the local Amazon Elastic Block Store (Amazon EBS) volumes before starting your training algorithm. This is the most commonly used input mode. In PIPE mode, Amazon SageMaker streams input data from the source directly to your algorithm without using the EBS volume.
    """

    name: StrPipeVar
    supported_content_types: List[StrPipeVar]
    supported_input_modes: List[StrPipeVar]
    description: Optional[StrPipeVar] = Unassigned()
    is_required: Optional[bool] = Unassigned()
    supported_compression_types: Optional[List[StrPipeVar]] = Unassigned()


class CheckpointConfig(Base):
    """
    CheckpointConfig
      Contains information about the output location for managed spot training checkpoint data.

    Attributes
    ----------------------
    s3_uri: Identifies the S3 path where you want SageMaker to store checkpoints. For example, s3://bucket-name/key-name-prefix.
    local_path: (Optional) The local directory where checkpoints are written. The default directory is /opt/ml/checkpoints/.
    """

    s3_uri: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()


class ClarifyCheckStepMetadata(Base):
    """
    ClarifyCheckStepMetadata
      The container for the metadata for the ClarifyCheck step. For more information, see the topic on ClarifyCheck step in the Amazon SageMaker Developer Guide.

    Attributes
    ----------------------
    check_type: The type of the Clarify Check step
    baseline_used_for_drift_check_constraints: The Amazon S3 URI of baseline constraints file to be used for the drift check.
    calculated_baseline_constraints: The Amazon S3 URI of the newly calculated baseline constraints file.
    model_package_group_name: The model package group name.
    violation_report: The Amazon S3 URI of the violation report if violations are detected.
    check_job_arn: The Amazon Resource Name (ARN) of the check processing job that was run by this step's execution.
    skip_check: This flag indicates if the drift check against the previous baseline will be skipped or not. If it is set to False, the previous baseline of the configured check type must be available.
    register_new_baseline: This flag indicates if a newly calculated baseline can be accessed through step properties BaselineUsedForDriftCheckConstraints and BaselineUsedForDriftCheckStatistics. If it is set to False, the previous baseline of the configured check type must also be available. These can be accessed through the BaselineUsedForDriftCheckConstraints property.
    """

    check_type: Optional[StrPipeVar] = Unassigned()
    baseline_used_for_drift_check_constraints: Optional[StrPipeVar] = Unassigned()
    calculated_baseline_constraints: Optional[StrPipeVar] = Unassigned()
    model_package_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    violation_report: Optional[StrPipeVar] = Unassigned()
    check_job_arn: Optional[StrPipeVar] = Unassigned()
    skip_check: Optional[bool] = Unassigned()
    register_new_baseline: Optional[bool] = Unassigned()


class ClarifyInferenceConfig(Base):
    """
    ClarifyInferenceConfig
      The inference configuration parameter for the model container.

    Attributes
    ----------------------
    features_attribute: Provides the JMESPath expression to extract the features from a model container input in JSON Lines format. For example, if FeaturesAttribute is the JMESPath expression 'myfeatures', it extracts a list of features [1,2,3] from request data '{"myfeatures":[1,2,3]}'.
    content_template: A template string used to format a JSON record into an acceptable model container input. For example, a ContentTemplate string '{"myfeatures":$features}' will format a list of features [1,2,3] into the record string '{"myfeatures":[1,2,3]}'. Required only when the model container input is in JSON Lines format.
    record_template
    max_record_count: The maximum number of records in a request that the model container can process when querying the model container for the predictions of a synthetic dataset. A record is a unit of input data that inference can be made on, for example, a single line in CSV data. If MaxRecordCount is 1, the model container expects one record per request. A value of 2 or greater means that the model expects batch requests, which can reduce overhead and speed up the inferencing process. If this parameter is not provided, the explainer will tune the record count per request according to the model container's capacity at runtime.
    max_payload_in_mb: The maximum payload size (MB) allowed of a request from the explainer to the model container. Defaults to 6 MB.
    probability_index: A zero-based index used to extract a probability value (score) or list from model container output in CSV format. If this value is not provided, the entire model container output will be treated as a probability value (score) or list.  Example for a single class model: If the model container output consists of a string-formatted prediction label followed by its probability: '1,0.6', set ProbabilityIndex to 1 to select the probability value 0.6.  Example for a multiclass model: If the model container output consists of a string-formatted prediction label followed by its probability: '"[\'cat\',\'dog\',\'fish\']","[0.1,0.6,0.3]"', set ProbabilityIndex to 1 to select the probability values [0.1,0.6,0.3].
    label_index: A zero-based index used to extract a label header or list of label headers from model container output in CSV format.  Example for a multiclass model: If the model container output consists of label headers followed by probabilities: '"[\'cat\',\'dog\',\'fish\']","[0.1,0.6,0.3]"', set LabelIndex to 0 to select the label headers ['cat','dog','fish'].
    probability_attribute: A JMESPath expression used to extract the probability (or score) from the model container output if the model container is in JSON Lines format.  Example: If the model container output of a single request is '{"predicted_label":1,"probability":0.6}', then set ProbabilityAttribute to 'probability'.
    label_attribute: A JMESPath expression used to locate the list of label headers in the model container output.  Example: If the model container output of a batch request is '{"labels":["cat","dog","fish"],"probability":[0.6,0.3,0.1]}', then set LabelAttribute to 'labels' to extract the list of label headers ["cat","dog","fish"]
    label_headers: For multiclass classification problems, the label headers are the names of the classes. Otherwise, the label header is the name of the predicted label. These are used to help readability for the output of the InvokeEndpoint API. See the response section under Invoke the endpoint in the Developer Guide for more information. If there are no label headers in the model container output, provide them manually using this parameter.
    feature_headers: The names of the features. If provided, these are included in the endpoint response payload to help readability of the InvokeEndpoint output. See the Response section under Invoke the endpoint in the Developer Guide for more information.
    feature_types: A list of data types of the features (optional). Applicable only to NLP explainability. If provided, FeatureTypes must have at least one 'text' string (for example, ['text']). If FeatureTypes is not provided, the explainer infers the feature types based on the baseline data. The feature types are included in the endpoint response payload. For additional information see the response section under Invoke the endpoint in the Developer Guide for more information.
    """

    features_attribute: Optional[StrPipeVar] = Unassigned()
    content_template: Optional[StrPipeVar] = Unassigned()
    record_template: Optional[StrPipeVar] = Unassigned()
    max_record_count: Optional[int] = Unassigned()
    max_payload_in_mb: Optional[int] = Unassigned()
    probability_index: Optional[int] = Unassigned()
    label_index: Optional[int] = Unassigned()
    probability_attribute: Optional[StrPipeVar] = Unassigned()
    label_attribute: Optional[StrPipeVar] = Unassigned()
    label_headers: Optional[List[StrPipeVar]] = Unassigned()
    feature_headers: Optional[List[StrPipeVar]] = Unassigned()
    feature_types: Optional[List[StrPipeVar]] = Unassigned()


class ClarifyShapBaselineConfig(Base):
    """
    ClarifyShapBaselineConfig
      The configuration for the SHAP baseline (also called the background or reference dataset) of the Kernal SHAP algorithm.    The number of records in the baseline data determines the size of the synthetic dataset, which has an impact on latency of explainability requests. For more information, see the Synthetic data of Configure and create an endpoint.    ShapBaseline and ShapBaselineUri are mutually exclusive parameters. One or the either is required to configure a SHAP baseline.

    Attributes
    ----------------------
    mime_type: The MIME type of the baseline data. Choose from 'text/csv' or 'application/jsonlines'. Defaults to 'text/csv'.
    shap_baseline: The inline SHAP baseline data in string format. ShapBaseline can have one or multiple records to be used as the baseline dataset. The format of the SHAP baseline file should be the same format as the training dataset. For example, if the training dataset is in CSV format and each record contains four features, and all features are numerical, then the format of the baseline data should also share these characteristics. For natural language processing (NLP) of text columns, the baseline value should be the value used to replace the unit of text specified by the Granularity of the TextConfig parameter. The size limit for ShapBasline is 4 KB. Use the ShapBaselineUri parameter if you want to provide more than 4 KB of baseline data.
    shap_baseline_uri: The uniform resource identifier (URI) of the S3 bucket where the SHAP baseline file is stored. The format of the SHAP baseline file should be the same format as the format of the training dataset. For example, if the training dataset is in CSV format, and each record in the training dataset has four features, and all features are numerical, then the baseline file should also have this same format. Each record should contain only the features. If you are using a virtual private cloud (VPC), the ShapBaselineUri should be accessible to the VPC. For more information about setting up endpoints with Amazon Virtual Private Cloud, see Give SageMaker access to Resources in your Amazon Virtual Private Cloud.
    """

    mime_type: Optional[StrPipeVar] = Unassigned()
    shap_baseline: Optional[StrPipeVar] = Unassigned()
    shap_baseline_uri: Optional[StrPipeVar] = Unassigned()


class ClarifyTextConfig(Base):
    """
    ClarifyTextConfig
      A parameter used to configure the SageMaker Clarify explainer to treat text features as text so that explanations are provided for individual units of text. Required only for natural language processing (NLP) explainability.

    Attributes
    ----------------------
    language: Specifies the language of the text features in ISO 639-1 or ISO 639-3 code of a supported language.   For a mix of multiple languages, use code 'xx'.
    granularity: The unit of granularity for the analysis of text features. For example, if the unit is 'token', then each token (like a word in English) of the text is treated as a feature. SHAP values are computed for each unit/feature.
    """

    language: StrPipeVar
    granularity: StrPipeVar


class ClarifyShapConfig(Base):
    """
    ClarifyShapConfig
      The configuration for SHAP analysis using SageMaker Clarify Explainer.

    Attributes
    ----------------------
    shap_baseline_config: The configuration for the SHAP baseline of the Kernal SHAP algorithm.
    number_of_samples: The number of samples to be used for analysis by the Kernal SHAP algorithm.   The number of samples determines the size of the synthetic dataset, which has an impact on latency of explainability requests. For more information, see the Synthetic data of Configure and create an endpoint.
    use_logit: A Boolean toggle to indicate if you want to use the logit function (true) or log-odds units (false) for model predictions. Defaults to false.
    seed: The starting value used to initialize the random number generator in the explainer. Provide a value for this parameter to obtain a deterministic SHAP result.
    text_config: A parameter that indicates if text features are treated as text and explanations are provided for individual units of text. Required for natural language processing (NLP) explainability only.
    """

    shap_baseline_config: ClarifyShapBaselineConfig
    number_of_samples: Optional[int] = Unassigned()
    use_logit: Optional[bool] = Unassigned()
    seed: Optional[int] = Unassigned()
    text_config: Optional[ClarifyTextConfig] = Unassigned()


class ClarifyExplainerConfig(Base):
    """
    ClarifyExplainerConfig
      The configuration parameters for the SageMaker Clarify explainer.

    Attributes
    ----------------------
    enable_explanations: A JMESPath boolean expression used to filter which records to explain. Explanations are activated by default. See  EnableExplanations for additional information.
    inference_config: The inference configuration parameter for the model container.
    shap_config: The configuration for SHAP analysis.
    """

    shap_config: ClarifyShapConfig
    enable_explanations: Optional[StrPipeVar] = Unassigned()
    inference_config: Optional[ClarifyInferenceConfig] = Unassigned()


class ClusterAutoScalingConfig(Base):
    """
    ClusterAutoScalingConfig
      Specifies the autoscaling configuration for a HyperPod cluster.

    Attributes
    ----------------------
    mode: Describes whether autoscaling is enabled or disabled for the cluster. Valid values are Enable and Disable.
    auto_scaler_type: The type of autoscaler to use. Currently supported value is Karpenter.
    """

    mode: StrPipeVar
    auto_scaler_type: Optional[StrPipeVar] = Unassigned()


class ClusterAutoScalingConfigOutput(Base):
    """
    ClusterAutoScalingConfigOutput
      The autoscaling configuration and status information for a HyperPod cluster.

    Attributes
    ----------------------
    mode: Describes whether autoscaling is enabled or disabled for the cluster.
    auto_scaler_type: The type of autoscaler configured for the cluster.
    status: The current status of the autoscaling configuration. Valid values are InService, Failed, Creating, and Deleting.
    failure_message: If the autoscaling status is Failed, this field contains a message describing the failure.
    """

    mode: StrPipeVar
    status: StrPipeVar
    auto_scaler_type: Optional[StrPipeVar] = Unassigned()
    failure_message: Optional[StrPipeVar] = Unassigned()


class ClusterSpotOptions(Base):
    """
    ClusterSpotOptions

    Attributes
    ----------------------
    """


class ClusterOnDemandOptions(Base):
    """
    ClusterOnDemandOptions

    Attributes
    ----------------------
    """


class ClusterCapacityRequirements(Base):
    """
    ClusterCapacityRequirements

    Attributes
    ----------------------
    spot
    on_demand
    """

    spot: Optional[ClusterSpotOptions] = Unassigned()
    on_demand: Optional[ClusterOnDemandOptions] = Unassigned()


class ClusterEbsVolumeConfig(Base):
    """
    ClusterEbsVolumeConfig
      Defines the configuration for attaching an additional Amazon Elastic Block Store (EBS) volume to each instance of the SageMaker HyperPod cluster instance group. To learn more, see SageMaker HyperPod release notes: June 20, 2024.

    Attributes
    ----------------------
    volume_size_in_gb: The size in gigabytes (GB) of the additional EBS volume to be attached to the instances in the SageMaker HyperPod cluster instance group. The additional EBS volume is attached to each instance within the SageMaker HyperPod cluster instance group and mounted to /opt/sagemaker.
    volume_kms_key_id: The ID of a KMS key to encrypt the Amazon EBS volume.
    root_volume: Specifies whether the configuration is for the cluster's root or secondary Amazon EBS volume. You can specify two ClusterEbsVolumeConfig fields to configure both the root and secondary volumes. Set the value to True if you'd like to provide your own customer managed Amazon Web Services KMS key to encrypt the root volume. When True:   The configuration is applied to the root volume.   You can't specify the VolumeSizeInGB field. The size of the root volume is determined for you.   You must specify a KMS key ID for VolumeKmsKeyId to encrypt the root volume with your own KMS key instead of an Amazon Web Services owned KMS key.   Otherwise, by default, the value is False, and the following applies:   The configuration is applied to the secondary volume, while the root volume is encrypted with an Amazon Web Services owned key.   You must specify the VolumeSizeInGB field.   You can optionally specify the VolumeKmsKeyId to encrypt the secondary volume with your own KMS key instead of an Amazon Web Services owned KMS key.
    """

    volume_size_in_gb: Optional[int] = Unassigned()
    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()
    root_volume: Optional[bool] = Unassigned()


class ClusterMetadata(Base):
    """
    ClusterMetadata
      Metadata information about a HyperPod cluster showing information about the cluster level operations, such as creating, updating, and deleting.

    Attributes
    ----------------------
    failure_message: An error message describing why the cluster level operation (such as creating, updating, or deleting) failed.
    eks_role_access_entries: A list of Amazon EKS IAM role ARNs associated with the cluster. This is created by HyperPod on your behalf and only applies for EKS orchestrated clusters.
    slr_access_entry: The Service-Linked Role (SLR) associated with the cluster. This is created by HyperPod on your behalf and only applies for EKS orchestrated clusters.
    """

    failure_message: Optional[StrPipeVar] = Unassigned()
    eks_role_access_entries: Optional[List[StrPipeVar]] = Unassigned()
    slr_access_entry: Optional[StrPipeVar] = Unassigned()


class InstanceGroupDeepHealthCheck(Base):
    """
    InstanceGroupDeepHealthCheck

    Attributes
    ----------------------
    operation_status
    requested_checks
    """

    operation_status: Optional[StrPipeVar] = Unassigned()
    requested_checks: Optional[List[StrPipeVar]] = Unassigned()


class InstanceGroupMetadata(Base):
    """
    InstanceGroupMetadata
      Metadata information about an instance group in a HyperPod cluster.

    Attributes
    ----------------------
    failure_message: An error message describing why the instance group level operation (such as creating, scaling, or deleting) failed.
    availability_zone_id: The ID of the Availability Zone where the instance group is located.
    capacity_reservation: Information about the Capacity Reservation used by the instance group.
    subnet_id: The ID of the subnet where the instance group is located.
    security_group_ids: A list of security group IDs associated with the instance group.
    ami_override: If you use a custom Amazon Machine Image (AMI) for the instance group, this field shows the ID of the custom AMI.
    instance_group_deep_health_check
    """

    failure_message: Optional[StrPipeVar] = Unassigned()
    availability_zone_id: Optional[StrPipeVar] = Unassigned()
    capacity_reservation: Optional[CapacityReservation] = Unassigned()
    subnet_id: Optional[StrPipeVar] = Unassigned()
    security_group_ids: Optional[List[StrPipeVar]] = Unassigned()
    ami_override: Optional[StrPipeVar] = Unassigned()
    instance_group_deep_health_check: Optional[InstanceGroupDeepHealthCheck] = Unassigned()


class InstanceGroupScalingMetadata(Base):
    """
    InstanceGroupScalingMetadata
      Metadata information about scaling operations for an instance group.

    Attributes
    ----------------------
    instance_count: The current number of instances in the group.
    target_count: The desired number of instances for the group after scaling.
    min_count
    failure_message: An error message describing why the scaling operation failed, if applicable.
    """

    instance_count: Optional[int] = Unassigned()
    target_count: Optional[int] = Unassigned()
    min_count: Optional[int] = Unassigned()
    failure_message: Optional[StrPipeVar] = Unassigned()


class HealthInfo(Base):
    """
    HealthInfo

    Attributes
    ----------------------
    health_status
    health_status_reason
    repair_action
    recommendation
    """

    health_status: Optional[StrPipeVar] = Unassigned()
    health_status_reason: Optional[StrPipeVar] = Unassigned()
    repair_action: Optional[StrPipeVar] = Unassigned()
    recommendation: Optional[StrPipeVar] = Unassigned()


class InstanceDeepHealthCheck(Base):
    """
    InstanceDeepHealthCheck

    Attributes
    ----------------------
    operation_status
    requested_checks
    completed_checks
    message
    """

    operation_status: Optional[StrPipeVar] = Unassigned()
    requested_checks: Optional[List[StrPipeVar]] = Unassigned()
    completed_checks: Optional[List[StrPipeVar]] = Unassigned()
    message: Optional[StrPipeVar] = Unassigned()


class InstanceMetadata(Base):
    """
    InstanceMetadata
      Metadata information about an instance in a HyperPod cluster.

    Attributes
    ----------------------
    customer_eni: The ID of the customer-managed Elastic Network Interface (ENI) associated with the instance.
    additional_enis: Information about additional Elastic Network Interfaces (ENIs) associated with the instance.
    capacity_reservation: Information about the Capacity Reservation used by the instance.
    failure_message: An error message describing why the instance creation or update failed, if applicable.
    lcs_execution_state: The execution state of the Lifecycle Script (LCS) for the instance.
    node_logical_id: The unique logical identifier of the node within the cluster. The ID used here is the same object as in the BatchAddClusterNodes API.
    node_health_info
    instance_deep_health_check
    """

    customer_eni: Optional[StrPipeVar] = Unassigned()
    additional_enis: Optional[AdditionalEnis] = Unassigned()
    capacity_reservation: Optional[CapacityReservation] = Unassigned()
    failure_message: Optional[StrPipeVar] = Unassigned()
    lcs_execution_state: Optional[StrPipeVar] = Unassigned()
    node_logical_id: Optional[StrPipeVar] = Unassigned()
    node_health_info: Optional[HealthInfo] = Unassigned()
    instance_deep_health_check: Optional[InstanceDeepHealthCheck] = Unassigned()


class InstanceMonitorMetadata(Base):
    """
    InstanceMonitorMetadata

    Attributes
    ----------------------
    instance_ready_count
    target_count
    failure_message
    """

    instance_ready_count: Optional[int] = Unassigned()
    target_count: Optional[int] = Unassigned()
    failure_message: Optional[StrPipeVar] = Unassigned()


class InstanceHealthMetadata(Base):
    """
    InstanceHealthMetadata

    Attributes
    ----------------------
    orchestrator_health_state
    failure_message
    """

    orchestrator_health_state: Optional[StrPipeVar] = Unassigned()
    failure_message: Optional[StrPipeVar] = Unassigned()


class EventMetadata(Base):
    """
    EventMetadata
      Metadata associated with a cluster event, which may include details about various resource types.

    Attributes
    ----------------------
    cluster: Metadata specific to cluster-level events.
    instance_group: Metadata specific to instance group-level events.
    instance_group_scaling: Metadata related to instance group scaling events.
    instance: Metadata specific to instance-level events.
    instance_monitor
    instance_health
    """

    cluster: Optional[ClusterMetadata] = Unassigned()
    instance_group: Optional[InstanceGroupMetadata] = Unassigned()
    instance_group_scaling: Optional[InstanceGroupScalingMetadata] = Unassigned()
    instance: Optional[InstanceMetadata] = Unassigned()
    instance_monitor: Optional[InstanceMonitorMetadata] = Unassigned()
    instance_health: Optional[InstanceHealthMetadata] = Unassigned()


class EventDetails(Base):
    """
    EventDetails
      Detailed information about a specific event, including event metadata.

    Attributes
    ----------------------
    event_metadata: Metadata specific to the event, which may include information about the cluster, instance group, or instance involved.
    """

    event_metadata: Optional[EventMetadata] = Unassigned()


class ClusterEventDetail(Base):
    """
    ClusterEventDetail
      Detailed information about a specific event in a HyperPod cluster.

    Attributes
    ----------------------
    event_id: The unique identifier (UUID) of the event.
    cluster_arn: The Amazon Resource Name (ARN) of the HyperPod cluster associated with the event.
    cluster_name: The name of the HyperPod cluster associated with the event.
    instance_group_name: The name of the instance group associated with the event, if applicable.
    instance_id: The EC2 instance ID associated with the event, if applicable.
    resource_type: The type of resource associated with the event. Valid values are Cluster, InstanceGroup, or Instance.
    event_time: The timestamp when the event occurred.
    event_details: Additional details about the event, including event-specific metadata.
    description: A human-readable description of the event.
    """

    event_id: StrPipeVar
    cluster_arn: StrPipeVar
    cluster_name: Union[StrPipeVar, object]
    resource_type: StrPipeVar
    event_time: datetime.datetime
    instance_group_name: Optional[StrPipeVar] = Unassigned()
    instance_id: Optional[StrPipeVar] = Unassigned()
    event_details: Optional[EventDetails] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()


class ClusterEventSummary(Base):
    """
    ClusterEventSummary
      A summary of an event in a HyperPod cluster.

    Attributes
    ----------------------
    event_id: The unique identifier (UUID) of the event.
    cluster_arn: The Amazon Resource Name (ARN) of the HyperPod cluster associated with the event.
    cluster_name: The name of the HyperPod cluster associated with the event.
    instance_group_name: The name of the instance group associated with the event, if applicable.
    instance_id: The Amazon Elastic Compute Cloud (EC2) instance ID associated with the event, if applicable.
    resource_type: The type of resource associated with the event. Valid values are Cluster, InstanceGroup, or Instance.
    event_time: The timestamp when the event occurred.
    description: A brief, human-readable description of the event.
    """

    event_id: StrPipeVar
    cluster_arn: StrPipeVar
    cluster_name: Union[StrPipeVar, object]
    resource_type: StrPipeVar
    event_time: datetime.datetime
    instance_group_name: Optional[StrPipeVar] = Unassigned()
    instance_id: Optional[StrPipeVar] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()


class ClusterLifeCycleConfig(Base):
    """
    ClusterLifeCycleConfig
      The lifecycle configuration for a SageMaker HyperPod cluster.

    Attributes
    ----------------------
    source_s3_uri: An Amazon S3 bucket path where your lifecycle scripts are stored.  Make sure that the S3 bucket path starts with s3://sagemaker-. The IAM role for SageMaker HyperPod has the managed  AmazonSageMakerClusterInstanceRolePolicy  attached, which allows access to S3 buckets with the specific prefix sagemaker-.
    on_create: The file name of the entrypoint script of lifecycle scripts under SourceS3Uri. This entrypoint script runs during cluster creation.
    """

    source_s3_uri: StrPipeVar
    on_create: StrPipeVar


class ClusterInstanceStorageConfig(Base):
    """
    ClusterInstanceStorageConfig
      Defines the configuration for attaching additional storage to the instances in the SageMaker HyperPod cluster instance group. To learn more, see SageMaker HyperPod release notes: June 20, 2024.

    Attributes
    ----------------------
    ebs_volume_config: Defines the configuration for attaching additional Amazon Elastic Block Store (EBS) volumes to the instances in the SageMaker HyperPod cluster instance group. The additional EBS volume is attached to each instance within the SageMaker HyperPod cluster instance group and mounted to /opt/sagemaker.
    """

    ebs_volume_config: Optional[ClusterEbsVolumeConfig] = Unassigned()


class ScalingConfig(Base):
    """
    ScalingConfig
      Defines how an instance group should be scaled and provisioned in SageMaker HyperPod.

    Attributes
    ----------------------
    best_effort_provisioning: Specifies whether to turn on best-effort provisioning. The default value is false. If set to true, SageMaker HyperPod will attempt to provision as many instances as possible, even if some instances fail to provision due to faulty nodes or configuration issues. This allows for partial provisioning of the requested number of instances when the full target cannot be achieved. Note that for provisioning with on-demand instances, billing begins as soon as healthy instances become available and enter the InService status.
    """

    best_effort_provisioning: bool


class RollingDeploymentPolicy(Base):
    """
    RollingDeploymentPolicy
      The configurations that SageMaker uses when updating the AMI versions.

    Attributes
    ----------------------
    maximum_batch_size: The maximum amount of instances in the cluster that SageMaker can update at a time.
    rollback_maximum_batch_size: The maximum amount of instances in the cluster that SageMaker can roll back at a time.
    """

    maximum_batch_size: CapacitySizeConfig
    rollback_maximum_batch_size: Optional[CapacitySizeConfig] = Unassigned()


class DeploymentConfiguration(Base):
    """
    DeploymentConfiguration
      The configuration to use when updating the AMI versions.

    Attributes
    ----------------------
    rolling_update_policy: The policy that SageMaker uses when updating the AMI versions of the cluster.
    wait_interval_in_seconds: The duration in seconds that SageMaker waits before updating more instances in the cluster.
    auto_rollback_configuration: An array that contains the alarms that SageMaker monitors to know whether to roll back the AMI update.
    """

    rolling_update_policy: Optional[RollingDeploymentPolicy] = Unassigned()
    wait_interval_in_seconds: Optional[int] = Unassigned()
    auto_rollback_configuration: Optional[List[AlarmDetails]] = Unassigned()


class ScheduledUpdateConfig(Base):
    """
    ScheduledUpdateConfig
      The configuration object of the schedule that SageMaker follows when updating the AMI.

    Attributes
    ----------------------
    schedule_expression: A cron expression that specifies the schedule that SageMaker follows when updating the AMI.
    deployment_config: The configuration to use when updating the AMI versions.
    """

    schedule_expression: StrPipeVar
    deployment_config: Optional[DeploymentConfiguration] = Unassigned()


class ClusterKubernetesTaint(Base):
    """
    ClusterKubernetesTaint

    Attributes
    ----------------------
    key
    value
    effect
    """

    key: StrPipeVar
    effect: StrPipeVar
    value: Optional[StrPipeVar] = Unassigned()


class ClusterKubernetesConfigDetails(Base):
    """
    ClusterKubernetesConfigDetails

    Attributes
    ----------------------
    current_labels
    desired_labels
    current_taints
    desired_taints
    """

    current_labels: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    desired_labels: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    current_taints: Optional[List[ClusterKubernetesTaint]] = Unassigned()
    desired_taints: Optional[List[ClusterKubernetesTaint]] = Unassigned()


class ClusterInstanceGroupDetails(Base):
    """
    ClusterInstanceGroupDetails
      Details of an instance group in a SageMaker HyperPod cluster.

    Attributes
    ----------------------
    current_count: The number of instances that are currently in the instance group of a SageMaker HyperPod cluster.
    target_count: The number of instances you specified to add to the instance group of a SageMaker HyperPod cluster.
    min_count
    instance_group_name: The name of the instance group of a SageMaker HyperPod cluster.
    instance_type: The instance type of the instance group of a SageMaker HyperPod cluster.
    life_cycle_config: Details of LifeCycle configuration for the instance group.
    execution_role: The execution role for the instance group to assume.
    threads_per_core: The number you specified to TreadsPerCore in CreateCluster for enabling or disabling multithreading. For instance types that support multithreading, you can specify 1 for disabling multithreading and 2 for enabling multithreading. For more information, see the reference table of CPU cores and threads per CPU core per instance type in the Amazon Elastic Compute Cloud User Guide.
    instance_storage_configs: The additional storage configurations for the instances in the SageMaker HyperPod cluster instance group.
    enable_burn_in_test
    on_start_deep_health_check
    on_start_deep_health_checks: A flag indicating whether deep health checks should be performed when the cluster instance group is created or updated.
    status: The current status of the cluster instance group.    InService: The instance group is active and healthy.    Creating: The instance group is being provisioned.    Updating: The instance group is being updated.    Failed: The instance group has failed to provision or is no longer healthy.    Degraded: The instance group is degraded, meaning that some instances have failed to provision or are no longer healthy.    Deleting: The instance group is being deleted.
    failure_messages: If the instance group is in a Failed or Degraded state, this field contains a list of failure messages that explain why the instances failed to provision or are no longer healthy. Each message includes a description of the issue.
    scaling_config: The actual scaling configuration applied to an existing instance group, reflecting the current provisioning state and scaling characteristics.
    training_plan_arn: The Amazon Resource Name (ARN); of the training plan associated with this cluster instance group. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .
    training_plan_status: The current status of the training plan associated with this cluster instance group.
    override_vpc_config: The customized Amazon VPC configuration at the instance group level that overrides the default Amazon VPC configuration of the SageMaker HyperPod cluster.
    custom_metadata
    scheduled_update_config: The configuration object of the schedule that SageMaker follows when updating the AMI.
    current_image_id: The ID of the Amazon Machine Image (AMI) currently in use by the instance group.
    desired_image_id: The ID of the Amazon Machine Image (AMI) desired for the instance group.
    active_operations
    kubernetes_config
    capacity_type
    capacity_requirements
    target_state_count: The number of nodes running a specific image ID since the last software update request.
    software_update_status: Status of the last software udpate request.
    active_software_update_config
    """

    current_count: Optional[int] = Unassigned()
    target_count: Optional[int] = Unassigned()
    min_count: Optional[int] = Unassigned()
    instance_group_name: Optional[StrPipeVar] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    life_cycle_config: Optional[ClusterLifeCycleConfig] = Unassigned()
    execution_role: Optional[StrPipeVar] = Unassigned()
    threads_per_core: Optional[int] = Unassigned()
    instance_storage_configs: Optional[List[ClusterInstanceStorageConfig]] = Unassigned()
    enable_burn_in_test: Optional[bool] = Unassigned()
    on_start_deep_health_check: Optional[List[StrPipeVar]] = Unassigned()
    on_start_deep_health_checks: Optional[List[StrPipeVar]] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    failure_messages: Optional[List[StrPipeVar]] = Unassigned()
    scaling_config: Optional[ScalingConfig] = Unassigned()
    training_plan_arn: Optional[StrPipeVar] = Unassigned()
    training_plan_status: Optional[StrPipeVar] = Unassigned()
    override_vpc_config: Optional[VpcConfig] = Unassigned()
    custom_metadata: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    scheduled_update_config: Optional[ScheduledUpdateConfig] = Unassigned()
    current_image_id: Optional[StrPipeVar] = Unassigned()
    desired_image_id: Optional[StrPipeVar] = Unassigned()
    active_operations: Optional[Dict[StrPipeVar, int]] = Unassigned()
    kubernetes_config: Optional[ClusterKubernetesConfigDetails] = Unassigned()
    capacity_type: Optional[StrPipeVar] = Unassigned()
    capacity_requirements: Optional[ClusterCapacityRequirements] = Unassigned()
    target_state_count: Optional[int] = Unassigned()
    software_update_status: Optional[StrPipeVar] = Unassigned()
    active_software_update_config: Optional[DeploymentConfiguration] = Unassigned()


class ClusterKubernetesConfig(Base):
    """
    ClusterKubernetesConfig

    Attributes
    ----------------------
    labels
    taints
    """

    labels: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    taints: Optional[List[ClusterKubernetesTaint]] = Unassigned()


class ClusterInstanceGroupSpecification(Base):
    """
    ClusterInstanceGroupSpecification
      The specifications of an instance group that you need to define.

    Attributes
    ----------------------
    instance_count: Specifies the number of instances to add to the instance group of a SageMaker HyperPod cluster.
    min_instance_count
    instance_group_name: Specifies the name of the instance group.
    instance_type: Specifies the instance type of the instance group.
    life_cycle_config: Specifies the LifeCycle configuration for the instance group.
    execution_role: Specifies an IAM execution role to be assumed by the instance group.
    threads_per_core: Specifies the value for Threads per core. For instance types that support multithreading, you can specify 1 for disabling multithreading and 2 for enabling multithreading. For instance types that doesn't support multithreading, specify 1. For more information, see the reference table of CPU cores and threads per CPU core per instance type in the Amazon Elastic Compute Cloud User Guide.
    instance_storage_configs: Specifies the additional storage configurations for the instances in the SageMaker HyperPod cluster instance group.
    enable_burn_in_test
    on_start_deep_health_check
    on_start_deep_health_checks: A flag indicating whether deep health checks should be performed when the cluster instance group is created or updated.
    scaling_config: The scaling and provisioning strategy for a planned instance group, specifying how instances should be allocated and handled during cluster creation.
    training_plan_arn: The Amazon Resource Name (ARN); of the training plan to use for this cluster instance group. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .
    override_vpc_config: To configure multi-AZ deployments, customize the Amazon VPC configuration at the instance group level. You can specify different subnets and security groups across different AZs in the instance group specification to override a SageMaker HyperPod cluster's default Amazon VPC configuration. For more information about deploying a cluster in multiple AZs, see Setting up SageMaker HyperPod clusters across multiple AZs.  When your Amazon VPC and subnets support IPv6, network communications differ based on the cluster orchestration platform:   Slurm-orchestrated clusters automatically configure nodes with dual IPv6 and IPv4 addresses, allowing immediate IPv6 network communications.   In Amazon EKS-orchestrated clusters, nodes receive dual-stack addressing, but pods can only use IPv6 when the Amazon EKS cluster is explicitly IPv6-enabled. For information about deploying an IPv6 Amazon EKS cluster, see Amazon EKS IPv6 Cluster Deployment.   Additional resources for IPv6 configuration:   For information about adding IPv6 support to your VPC, see to IPv6 Support for VPC.   For information about creating a new IPv6-compatible VPC, see Amazon VPC Creation Guide.   To configure SageMaker HyperPod with a custom Amazon VPC, see Custom Amazon VPC Setup for SageMaker HyperPod.
    custom_metadata
    scheduled_update_config: The configuration object of the schedule that SageMaker uses to update the AMI.
    image_id: When configuring your HyperPod cluster, you can specify an image ID using one of the following options:    HyperPodPublicAmiId: Use a HyperPod public AMI    CustomAmiId: Use your custom AMI    default: Use the default latest system image   If you choose to use a custom AMI (CustomAmiId), ensure it meets the following requirements:   Encryption: The custom AMI must be unencrypted.   Ownership: The custom AMI must be owned by the same Amazon Web Services account that is creating the HyperPod cluster.   Volume support: Only the primary AMI snapshot volume is supported; additional AMI volumes are not supported.   When updating the instance group's AMI through the UpdateClusterSoftware operation, if an instance group uses a custom AMI, you must provide an ImageId or use the default as input. Note that if you don't specify an instance group in your UpdateClusterSoftware request, then all of the instance groups are patched with the specified image.
    kubernetes_config
    capacity_type
    capacity_requirements
    """

    instance_count: int
    instance_group_name: StrPipeVar
    instance_type: StrPipeVar
    life_cycle_config: ClusterLifeCycleConfig
    execution_role: StrPipeVar
    min_instance_count: Optional[int] = Unassigned()
    threads_per_core: Optional[int] = Unassigned()
    instance_storage_configs: Optional[List[ClusterInstanceStorageConfig]] = Unassigned()
    enable_burn_in_test: Optional[bool] = Unassigned()
    on_start_deep_health_check: Optional[List[StrPipeVar]] = Unassigned()
    on_start_deep_health_checks: Optional[List[StrPipeVar]] = Unassigned()
    scaling_config: Optional[ScalingConfig] = Unassigned()
    training_plan_arn: Optional[StrPipeVar] = Unassigned()
    override_vpc_config: Optional[VpcConfig] = Unassigned()
    custom_metadata: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    scheduled_update_config: Optional[ScheduledUpdateConfig] = Unassigned()
    image_id: Optional[StrPipeVar] = Unassigned()
    kubernetes_config: Optional[ClusterKubernetesConfig] = Unassigned()
    capacity_type: Optional[StrPipeVar] = Unassigned()
    capacity_requirements: Optional[ClusterCapacityRequirements] = Unassigned()


class ClusterInstancePlacement(Base):
    """
    ClusterInstancePlacement
      Specifies the placement details for the node in the SageMaker HyperPod cluster, including the Availability Zone and the unique identifier (ID) of the Availability Zone.

    Attributes
    ----------------------
    availability_zone: The Availability Zone where the node in the SageMaker HyperPod cluster is launched.
    availability_zone_id: The unique identifier (ID) of the Availability Zone where the node in the SageMaker HyperPod cluster is launched.
    """

    availability_zone: Optional[StrPipeVar] = Unassigned()
    availability_zone_id: Optional[StrPipeVar] = Unassigned()


class ClusterInstanceStatusDetails(Base):
    """
    ClusterInstanceStatusDetails
      Details of an instance in a SageMaker HyperPod cluster.

    Attributes
    ----------------------
    status: The status of an instance in a SageMaker HyperPod cluster.
    message: The message from an instance in a SageMaker HyperPod cluster.
    """

    status: StrPipeVar
    message: Optional[StrPipeVar] = Unassigned()


class ClusterKubernetesConfigNodeDetails(Base):
    """
    ClusterKubernetesConfigNodeDetails

    Attributes
    ----------------------
    current_labels
    desired_labels
    current_taints
    desired_taints
    """

    current_labels: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    desired_labels: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    current_taints: Optional[List[ClusterKubernetesTaint]] = Unassigned()
    desired_taints: Optional[List[ClusterKubernetesTaint]] = Unassigned()


class UltraServerInfo(Base):
    """
    UltraServerInfo
      Contains information about the UltraServer object.

    Attributes
    ----------------------
    id: The unique identifier of the UltraServer.
    """

    id: Optional[StrPipeVar] = Unassigned()


class ClusterNodeDetails(Base):
    """
    ClusterNodeDetails
      Details of an instance (also called a node interchangeably) in a SageMaker HyperPod cluster.

    Attributes
    ----------------------
    instance_group_name: The instance group name in which the instance is.
    instance_id: The ID of the instance.
    node_logical_id: A unique identifier for the node that persists throughout its lifecycle, from provisioning request to termination. This identifier can be used to track the node even before it has an assigned InstanceId.
    instance_status: The status of the instance.
    instance_type: The type of the instance.
    launch_time: The time when the instance is launched.
    last_software_update_time: The time when the cluster was last updated.
    life_cycle_config: The LifeCycle configuration applied to the instance.
    override_vpc_config: The customized Amazon VPC configuration at the instance group level that overrides the default Amazon VPC configuration of the SageMaker HyperPod cluster.
    threads_per_core: The number of threads per CPU core you specified under CreateCluster.
    instance_storage_configs: The configurations of additional storage specified to the instance group where the instance (node) is launched.
    private_primary_ip: The private primary IP address of the SageMaker HyperPod cluster node.
    private_primary_ipv6: The private primary IPv6 address of the SageMaker HyperPod cluster node when configured with an Amazon VPC that supports IPv6 and includes subnets with IPv6 addressing enabled in either the cluster Amazon VPC configuration or the instance group Amazon VPC configuration.
    private_dns_hostname: The private DNS hostname of the SageMaker HyperPod cluster node.
    placement: The placement details of the SageMaker HyperPod cluster node.
    health_info
    current_image_id: The ID of the Amazon Machine Image (AMI) currently in use by the node.
    desired_image_id: The ID of the Amazon Machine Image (AMI) desired for the node.
    ultra_server_info: Contains information about the UltraServer.
    kubernetes_config
    capacity_type
    """

    instance_group_name: Optional[StrPipeVar] = Unassigned()
    instance_id: Optional[StrPipeVar] = Unassigned()
    node_logical_id: Optional[StrPipeVar] = Unassigned()
    instance_status: Optional[ClusterInstanceStatusDetails] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    launch_time: Optional[datetime.datetime] = Unassigned()
    last_software_update_time: Optional[datetime.datetime] = Unassigned()
    life_cycle_config: Optional[ClusterLifeCycleConfig] = Unassigned()
    override_vpc_config: Optional[VpcConfig] = Unassigned()
    threads_per_core: Optional[int] = Unassigned()
    instance_storage_configs: Optional[List[ClusterInstanceStorageConfig]] = Unassigned()
    private_primary_ip: Optional[StrPipeVar] = Unassigned()
    private_primary_ipv6: Optional[StrPipeVar] = Unassigned()
    private_dns_hostname: Optional[StrPipeVar] = Unassigned()
    placement: Optional[ClusterInstancePlacement] = Unassigned()
    health_info: Optional[HealthInfo] = Unassigned()
    current_image_id: Optional[StrPipeVar] = Unassigned()
    desired_image_id: Optional[StrPipeVar] = Unassigned()
    ultra_server_info: Optional[UltraServerInfo] = Unassigned()
    kubernetes_config: Optional[ClusterKubernetesConfigNodeDetails] = Unassigned()
    capacity_type: Optional[StrPipeVar] = Unassigned()


class ClusterNodeSummaryHealthInfo(Base):
    """
    ClusterNodeSummaryHealthInfo

    Attributes
    ----------------------
    health_status
    health_status_reason
    """

    health_status: Optional[StrPipeVar] = Unassigned()
    health_status_reason: Optional[StrPipeVar] = Unassigned()


class ClusterNodeSummary(Base):
    """
    ClusterNodeSummary
      Lists a summary of the properties of an instance (also called a node interchangeably) of a SageMaker HyperPod cluster.

    Attributes
    ----------------------
    instance_group_name: The name of the instance group in which the instance is.
    instance_id: The ID of the instance.
    node_logical_id: A unique identifier for the node that persists throughout its lifecycle, from provisioning request to termination. This identifier can be used to track the node even before it has an assigned InstanceId. This field is only included when IncludeNodeLogicalIds is set to True in the ListClusterNodes request.
    instance_type: The type of the instance.
    launch_time: The time when the instance is launched.
    last_software_update_time: The time when SageMaker last updated the software of the instances in the cluster.
    instance_status: The status of the instance.
    health_info
    ultra_server_info: Contains information about the UltraServer.
    private_dns_hostname
    """

    instance_group_name: StrPipeVar
    instance_id: StrPipeVar
    instance_type: StrPipeVar
    launch_time: datetime.datetime
    instance_status: ClusterInstanceStatusDetails
    node_logical_id: Optional[StrPipeVar] = Unassigned()
    last_software_update_time: Optional[datetime.datetime] = Unassigned()
    health_info: Optional[ClusterNodeSummaryHealthInfo] = Unassigned()
    ultra_server_info: Optional[UltraServerInfo] = Unassigned()
    private_dns_hostname: Optional[StrPipeVar] = Unassigned()


class ClusterOrchestratorEksConfig(Base):
    """
    ClusterOrchestratorEksConfig
      The configuration settings for the Amazon EKS cluster used as the orchestrator for the SageMaker HyperPod cluster.

    Attributes
    ----------------------
    cluster_arn: The Amazon Resource Name (ARN) of the Amazon EKS cluster associated with the SageMaker HyperPod cluster.
    """

    cluster_arn: StrPipeVar


class ClusterOrchestrator(Base):
    """
    ClusterOrchestrator
      The type of orchestrator used for the SageMaker HyperPod cluster.

    Attributes
    ----------------------
    eks: The Amazon EKS cluster used as the orchestrator for the SageMaker HyperPod cluster.
    """

    eks: ClusterOrchestratorEksConfig


class ClusterResilienceConfig(Base):
    """
    ClusterResilienceConfig

    Attributes
    ----------------------
    enable_node_auto_recovery
    """

    enable_node_auto_recovery: Optional[bool] = Unassigned()


class FSxLustreConfig(Base):
    """
    FSxLustreConfig
      Configuration settings for an Amazon FSx for Lustre file system to be used with the cluster.

    Attributes
    ----------------------
    size_in_gi_b: The storage capacity of the Amazon FSx for Lustre file system, specified in gibibytes (GiB).
    per_unit_storage_throughput: The throughput capacity of the Amazon FSx for Lustre file system, measured in MB/s per TiB of storage.
    """

    size_in_gi_b: int
    per_unit_storage_throughput: int


class TrustedEnvironmentDetails(Base):
    """
    TrustedEnvironmentDetails

    Attributes
    ----------------------
    f_sx_lustre_config
    s3_output_path
    """

    f_sx_lustre_config: Optional[FSxLustreConfig] = Unassigned()
    s3_output_path: Optional[StrPipeVar] = Unassigned()


class EnvironmentConfigDetails(Base):
    """
    EnvironmentConfigDetails
      The configuration details for the restricted instance groups (RIG) environment.

    Attributes
    ----------------------
    f_sx_lustre_config: Configuration settings for an Amazon FSx for Lustre file system to be used with the cluster.
    s3_output_path: The Amazon S3 path where output data from the restricted instance group (RIG) environment will be stored.
    """

    f_sx_lustre_config: Optional[FSxLustreConfig] = Unassigned()
    s3_output_path: Optional[StrPipeVar] = Unassigned()


class ClusterRestrictedInstanceGroupDetails(Base):
    """
    ClusterRestrictedInstanceGroupDetails
      The instance group details of the restricted instance group (RIG).

    Attributes
    ----------------------
    current_count: The number of instances that are currently in the restricted instance group of a SageMaker HyperPod cluster.
    target_count: The number of instances you specified to add to the restricted instance group of a SageMaker HyperPod cluster.
    instance_group_name: The name of the restricted instance group of a SageMaker HyperPod cluster.
    instance_type: The instance type of the restricted instance group of a SageMaker HyperPod cluster.
    execution_role: The execution role for the restricted instance group to assume.
    threads_per_core: The number you specified to TreadsPerCore in CreateCluster for enabling or disabling multithreading. For instance types that support multithreading, you can specify 1 for disabling multithreading and 2 for enabling multithreading. For more information, see the reference table of CPU cores and threads per CPU core per instance type in the Amazon Elastic Compute Cloud User Guide.
    instance_storage_configs: The additional storage configurations for the instances in the SageMaker HyperPod cluster restricted instance group.
    enable_burn_in_test
    on_start_deep_health_check
    on_start_deep_health_checks: A flag indicating whether deep health checks should be performed when the cluster's restricted instance group is created or updated.
    status: The current status of the cluster's restricted instance group.    InService: The restricted instance group is active and healthy.    Creating: The restricted instance group is being provisioned.    Updating: The restricted instance group is being updated.    Failed: The restricted instance group has failed to provision or is no longer healthy.    Degraded: The restricted instance group is degraded, meaning that some instances have failed to provision or are no longer healthy.    Deleting: The restricted instance group is being deleted.
    failure_messages
    scaling_config
    training_plan_arn: The Amazon Resource Name (ARN) of the training plan to filter clusters by. For more information about reserving GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .
    training_plan_status: The current status of the training plan associated with this cluster restricted instance group.
    override_vpc_config
    custom_metadata
    scheduled_update_config
    trusted_environment
    environment_config: The configuration for the restricted instance groups (RIG) environment.
    """

    current_count: Optional[int] = Unassigned()
    target_count: Optional[int] = Unassigned()
    instance_group_name: Optional[StrPipeVar] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    execution_role: Optional[StrPipeVar] = Unassigned()
    threads_per_core: Optional[int] = Unassigned()
    instance_storage_configs: Optional[List[ClusterInstanceStorageConfig]] = Unassigned()
    enable_burn_in_test: Optional[bool] = Unassigned()
    on_start_deep_health_check: Optional[List[StrPipeVar]] = Unassigned()
    on_start_deep_health_checks: Optional[List[StrPipeVar]] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    failure_messages: Optional[List[StrPipeVar]] = Unassigned()
    scaling_config: Optional[ScalingConfig] = Unassigned()
    training_plan_arn: Optional[StrPipeVar] = Unassigned()
    training_plan_status: Optional[StrPipeVar] = Unassigned()
    override_vpc_config: Optional[VpcConfig] = Unassigned()
    custom_metadata: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    scheduled_update_config: Optional[ScheduledUpdateConfig] = Unassigned()
    trusted_environment: Optional[TrustedEnvironmentDetails] = Unassigned()
    environment_config: Optional[EnvironmentConfigDetails] = Unassigned()


class TrustedEnvironmentConfig(Base):
    """
    TrustedEnvironmentConfig

    Attributes
    ----------------------
    f_sx_lustre_config
    """

    f_sx_lustre_config: Optional[FSxLustreConfig] = Unassigned()


class TrustedEnvironment(Base):
    """
    TrustedEnvironment

    Attributes
    ----------------------
    config
    """

    config: Optional[TrustedEnvironmentConfig] = Unassigned()


class EnvironmentConfig(Base):
    """
    EnvironmentConfig
      The configuration for the restricted instance groups (RIG) environment.

    Attributes
    ----------------------
    f_sx_lustre_config: Configuration settings for an Amazon FSx for Lustre file system to be used with the cluster.
    """

    f_sx_lustre_config: Optional[FSxLustreConfig] = Unassigned()


class ClusterRestrictedInstanceGroupSpecification(Base):
    """
    ClusterRestrictedInstanceGroupSpecification
      The specifications of a restricted instance group that you need to define.

    Attributes
    ----------------------
    instance_count: Specifies the number of instances to add to the restricted instance group of a SageMaker HyperPod cluster.
    instance_group_name: Specifies the name of the restricted instance group.
    instance_type: Specifies the instance type of the restricted instance group.
    execution_role: Specifies an IAM execution role to be assumed by the restricted instance group.
    threads_per_core: The number you specified to TreadsPerCore in CreateCluster for enabling or disabling multithreading. For instance types that support multithreading, you can specify 1 for disabling multithreading and 2 for enabling multithreading. For more information, see the reference table of CPU cores and threads per CPU core per instance type in the Amazon Elastic Compute Cloud User Guide.
    instance_storage_configs: Specifies the additional storage configurations for the instances in the SageMaker HyperPod cluster restricted instance group.
    enable_burn_in_test
    on_start_deep_health_check
    on_start_deep_health_checks: A flag indicating whether deep health checks should be performed when the cluster restricted instance group is created or updated.
    scaling_config
    training_plan_arn: The Amazon Resource Name (ARN) of the training plan to filter clusters by. For more information about reserving GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .
    override_vpc_config
    custom_metadata
    scheduled_update_config
    trusted_environment
    environment_config: The configuration for the restricted instance groups (RIG) environment.
    """

    instance_count: int
    instance_group_name: StrPipeVar
    instance_type: StrPipeVar
    execution_role: StrPipeVar
    environment_config: EnvironmentConfig
    threads_per_core: Optional[int] = Unassigned()
    instance_storage_configs: Optional[List[ClusterInstanceStorageConfig]] = Unassigned()
    enable_burn_in_test: Optional[bool] = Unassigned()
    on_start_deep_health_check: Optional[List[StrPipeVar]] = Unassigned()
    on_start_deep_health_checks: Optional[List[StrPipeVar]] = Unassigned()
    scaling_config: Optional[ScalingConfig] = Unassigned()
    training_plan_arn: Optional[StrPipeVar] = Unassigned()
    override_vpc_config: Optional[VpcConfig] = Unassigned()
    custom_metadata: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    scheduled_update_config: Optional[ScheduledUpdateConfig] = Unassigned()
    trusted_environment: Optional[TrustedEnvironment] = Unassigned()


class ClusterSchedulerConfigSummary(Base):
    """
    ClusterSchedulerConfigSummary
      Summary of the cluster policy.

    Attributes
    ----------------------
    cluster_scheduler_config_arn: ARN of the cluster policy.
    cluster_scheduler_config_id: ID of the cluster policy.
    cluster_scheduler_config_version: Version of the cluster policy.
    name: Name of the cluster policy.
    creation_time: Creation time of the cluster policy.
    last_modified_time: Last modified time of the cluster policy.
    status: Status of the cluster policy.
    cluster_arn: ARN of the cluster.
    """

    cluster_scheduler_config_arn: StrPipeVar
    cluster_scheduler_config_id: StrPipeVar
    name: StrPipeVar
    creation_time: datetime.datetime
    status: StrPipeVar
    cluster_scheduler_config_version: Optional[int] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    cluster_arn: Optional[StrPipeVar] = Unassigned()


class ClusterSummary(Base):
    """
    ClusterSummary
      Lists a summary of the properties of a SageMaker HyperPod cluster.

    Attributes
    ----------------------
    cluster_arn: The Amazon Resource Name (ARN) of the SageMaker HyperPod cluster.
    cluster_name: The name of the SageMaker HyperPod cluster.
    creation_time: The time when the SageMaker HyperPod cluster is created.
    cluster_status: The status of the SageMaker HyperPod cluster.
    training_plan_arns: A list of Amazon Resource Names (ARNs) of the training plans associated with this cluster. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .
    """

    cluster_arn: StrPipeVar
    cluster_name: Union[StrPipeVar, object]
    creation_time: datetime.datetime
    cluster_status: StrPipeVar
    training_plan_arns: Optional[List[StrPipeVar]] = Unassigned()


class ClusterTieredStorageConfig(Base):
    """
    ClusterTieredStorageConfig
      Defines the configuration for managed tier checkpointing in a HyperPod cluster. Managed tier checkpointing uses multiple storage tiers, including cluster CPU memory, to provide faster checkpoint operations and improved fault tolerance for large-scale model training. The system automatically saves checkpoints at high frequency to memory and periodically persists them to durable storage, like Amazon S3.

    Attributes
    ----------------------
    mode: Specifies whether managed tier checkpointing is enabled or disabled for the HyperPod cluster. When set to Enable, the system installs a memory management daemon that provides disaggregated memory as a service for checkpoint storage. When set to Disable, the feature is turned off and the memory management daemon is removed from the cluster.
    instance_memory_allocation_percentage: The percentage (int) of cluster memory to allocate for checkpointing.
    """

    mode: StrPipeVar
    instance_memory_allocation_percentage: Optional[int] = Unassigned()


class CustomImage(Base):
    """
    CustomImage
      A custom SageMaker AI image. For more information, see Bring your own SageMaker AI image.

    Attributes
    ----------------------
    image_name: The name of the CustomImage. Must be unique to your account.
    image_version_number: The version number of the CustomImage.
    app_image_config_name: The name of the AppImageConfig.
    """

    image_name: Union[StrPipeVar, object]
    app_image_config_name: Union[StrPipeVar, object]
    image_version_number: Optional[int] = Unassigned()


class CodeEditorAppSettings(Base):
    """
    CodeEditorAppSettings
      The Code Editor application settings. For more information about Code Editor, see Get started with Code Editor in Amazon SageMaker.

    Attributes
    ----------------------
    default_resource_spec
    custom_images: A list of custom SageMaker images that are configured to run as a Code Editor app.
    lifecycle_config_arns: The Amazon Resource Name (ARN) of the Code Editor application lifecycle configuration.
    app_lifecycle_management: Settings that are used to configure and manage the lifecycle of CodeEditor applications.
    built_in_lifecycle_config_arn: The lifecycle configuration that runs before the default lifecycle configuration. It can override changes made in the default lifecycle configuration.
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    custom_images: Optional[List[CustomImage]] = Unassigned()
    lifecycle_config_arns: Optional[List[StrPipeVar]] = Unassigned()
    app_lifecycle_management: Optional[AppLifecycleManagement] = Unassigned()
    built_in_lifecycle_config_arn: Optional[StrPipeVar] = Unassigned()


class CodeRepository(Base):
    """
    CodeRepository
      A Git repository that SageMaker AI automatically displays to users for cloning in the JupyterServer application.

    Attributes
    ----------------------
    repository_url: The URL of the Git repository.
    """

    repository_url: StrPipeVar


class GitConfig(Base):
    """
    GitConfig
      Specifies configuration details for a Git repository in your Amazon Web Services account.

    Attributes
    ----------------------
    repository_url: The URL where the Git repository is located.
    branch: The default branch for the Git repository.
    secret_arn: The Amazon Resource Name (ARN) of the Amazon Web Services Secrets Manager secret that contains the credentials used to access the git repository. The secret must have a staging label of AWSCURRENT and must be in the following format:  {"username": UserName, "password": Password}
    """

    repository_url: StrPipeVar
    branch: Optional[StrPipeVar] = Unassigned()
    secret_arn: Optional[StrPipeVar] = Unassigned()


class CodeRepositorySummary(Base):
    """
    CodeRepositorySummary
      Specifies summary information about a Git repository.

    Attributes
    ----------------------
    code_repository_name: The name of the Git repository.
    code_repository_arn: The Amazon Resource Name (ARN) of the Git repository.
    creation_time: The date and time that the Git repository was created.
    last_modified_time: The date and time that the Git repository was last modified.
    git_config: Configuration details for the Git repository, including the URL where it is located and the ARN of the Amazon Web Services Secrets Manager secret that contains the credentials used to access the repository.
    """

    code_repository_name: Union[StrPipeVar, object]
    code_repository_arn: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    git_config: Optional[GitConfig] = Unassigned()


class CognitoConfig(Base):
    """
    CognitoConfig
      Use this parameter to configure your Amazon Cognito workforce. A single Cognito workforce is created using and corresponds to a single  Amazon Cognito user pool.

    Attributes
    ----------------------
    user_pool: A  user pool is a user directory in Amazon Cognito. With a user pool, your users can sign in to your web or mobile app through Amazon Cognito. Your users can also sign in through social identity providers like Google, Facebook, Amazon, or Apple, and through SAML identity providers.
    client_id: The client ID for your Amazon Cognito user pool.
    """

    user_pool: StrPipeVar
    client_id: StrPipeVar


class CognitoMemberDefinition(Base):
    """
    CognitoMemberDefinition
      Identifies a Amazon Cognito user group. A user group can be used in on or more work teams.

    Attributes
    ----------------------
    user_pool: An identifier for a user pool. The user pool must be in the same region as the service that you are calling.
    user_group: An identifier for a user group.
    client_id: An identifier for an application client. You must create the app client ID using Amazon Cognito.
    member_definition_id
    """

    user_pool: StrPipeVar
    user_group: StrPipeVar
    client_id: StrPipeVar
    member_definition_id: Optional[StrPipeVar] = Unassigned()


class VectorConfig(Base):
    """
    VectorConfig
      Configuration for your vector collection type.

    Attributes
    ----------------------
    dimension: The number of elements in your vector.
    """

    dimension: int


class CollectionConfig(Base):
    """
    CollectionConfig
      Configuration for your collection.

    Attributes
    ----------------------
    vector_config: Configuration for your vector collection type.    Dimension: The number of elements in your vector.
    """

    vector_config: Optional[VectorConfig] = Unassigned()


class CollectionConfiguration(Base):
    """
    CollectionConfiguration
      Configuration information for the Amazon SageMaker Debugger output tensor collections.

    Attributes
    ----------------------
    collection_name: The name of the tensor collection. The name must be unique relative to other rule configuration names.
    collection_parameters: Parameter values for the tensor collection. The allowed parameters are "name", "include_regex", "reduction_config", "save_config", "tensor_names", and "save_histogram".
    """

    collection_name: Optional[StrPipeVar] = Unassigned()
    collection_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class CommentEntity(Base):
    """
    CommentEntity

    Attributes
    ----------------------
    publisher
    comment
    creation_time
    """

    publisher: Optional[StrPipeVar] = Unassigned()
    comment: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()


class CompilationJobStepMetadata(Base):
    """
    CompilationJobStepMetadata

    Attributes
    ----------------------
    arn
    """

    arn: Optional[StrPipeVar] = Unassigned()


class CompilationJobSummary(Base):
    """
    CompilationJobSummary
      A summary of a model compilation job.

    Attributes
    ----------------------
    compilation_job_name: The name of the model compilation job that you want a summary for.
    compilation_job_arn: The Amazon Resource Name (ARN) of the model compilation job.
    creation_time: The time when the model compilation job was created.
    compilation_start_time: The time when the model compilation job started.
    compilation_end_time: The time when the model compilation job completed.
    compilation_target_device: The type of device that the model will run on after the compilation job has completed.
    compilation_target_platform_os: The type of OS that the model will run on after the compilation job has completed.
    compilation_target_platform_arch: The type of architecture that the model will run on after the compilation job has completed.
    compilation_target_platform_accelerator: The type of accelerator that the model will run on after the compilation job has completed.
    last_modified_time: The time when the model compilation job was last modified.
    compilation_job_status: The status of the model compilation job.
    """

    compilation_job_name: Union[StrPipeVar, object]
    compilation_job_arn: StrPipeVar
    creation_time: datetime.datetime
    compilation_job_status: StrPipeVar
    compilation_start_time: Optional[datetime.datetime] = Unassigned()
    compilation_end_time: Optional[datetime.datetime] = Unassigned()
    compilation_target_device: Optional[StrPipeVar] = Unassigned()
    compilation_target_platform_os: Optional[StrPipeVar] = Unassigned()
    compilation_target_platform_arch: Optional[StrPipeVar] = Unassigned()
    compilation_target_platform_accelerator: Optional[StrPipeVar] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class ComponentJobSummary(Base):
    """
    ComponentJobSummary

    Attributes
    ----------------------
    auto_ml_job_name
    auto_ml_job_arn
    last_modified_time
    status
    creation_time
    component_job_type
    component_job_name
    component_job_arn
    end_time
    failure_reason
    description
    """

    auto_ml_job_name: Optional[StrPipeVar] = Unassigned()
    auto_ml_job_arn: Optional[StrPipeVar] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    component_job_type: Optional[StrPipeVar] = Unassigned()
    component_job_name: Optional[StrPipeVar] = Unassigned()
    component_job_arn: Optional[StrPipeVar] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()


class ComputeQuotaResourceConfig(Base):
    """
    ComputeQuotaResourceConfig
      Configuration of the resources used for the compute allocation definition.

    Attributes
    ----------------------
    instance_type: The instance type of the instance group for the cluster.
    count: The number of instances to add to the instance group of a SageMaker HyperPod cluster.
    accelerators: The number of accelerators to allocate. If you don't specify a value for vCPU and MemoryInGiB, SageMaker AI automatically allocates ratio-based values for those parameters based on the number of accelerators you provide. For example, if you allocate 16 out of 32 total accelerators, SageMaker AI uses the ratio of 0.5 and allocates values to vCPU and MemoryInGiB.
    v_cpu: The number of vCPU to allocate. If you specify a value only for vCPU, SageMaker AI automatically allocates ratio-based values for MemoryInGiB based on this vCPU parameter. For example, if you allocate 20 out of 40 total vCPU, SageMaker AI uses the ratio of 0.5 and allocates values to MemoryInGiB. Accelerators are set to 0.
    memory_in_gi_b: The amount of memory in GiB to allocate. If you specify a value only for this parameter, SageMaker AI automatically allocates a ratio-based value for vCPU based on this memory that you provide. For example, if you allocate 200 out of 400 total memory in GiB, SageMaker AI uses the ratio of 0.5 and allocates values to vCPU. Accelerators are set to 0.
    accelerator_partition
    """

    instance_type: StrPipeVar
    count: Optional[int] = Unassigned()
    accelerators: Optional[int] = Unassigned()
    v_cpu: Optional[float] = Unassigned()
    memory_in_gi_b: Optional[float] = Unassigned()
    accelerator_partition: Optional[AcceleratorPartitionConfig] = Unassigned()


class ResourceSharingConfig(Base):
    """
    ResourceSharingConfig
      Resource sharing configuration.

    Attributes
    ----------------------
    strategy: The strategy of how idle compute is shared within the cluster. The following are the options of strategies.    DontLend: entities do not lend idle compute.    Lend: entities can lend idle compute to entities that can borrow.    LendandBorrow: entities can lend idle compute and borrow idle compute from other entities.   Default is LendandBorrow.
    borrow_limit: The limit on how much idle compute can be borrowed.The values can be 1 - 500 percent of idle compute that the team is allowed to borrow. Default is 50.
    """

    strategy: StrPipeVar
    borrow_limit: Optional[int] = Unassigned()


class ComputeQuotaConfig(Base):
    """
    ComputeQuotaConfig
      Configuration of the compute allocation definition for an entity. This includes the resource sharing option and the setting to preempt low priority tasks.

    Attributes
    ----------------------
    compute_quota_resources: Allocate compute resources by instance types.
    resource_sharing_config: Resource sharing configuration. This defines how an entity can lend and borrow idle compute with other entities within the cluster.
    preempt_team_tasks: Allows workloads from within an entity to preempt same-team workloads. When set to LowerPriority, the entity's lower priority tasks are preempted by their own higher priority tasks. Default is LowerPriority.
    """

    compute_quota_resources: Optional[List[ComputeQuotaResourceConfig]] = Unassigned()
    resource_sharing_config: Optional[ResourceSharingConfig] = Unassigned()
    preempt_team_tasks: Optional[StrPipeVar] = Unassigned()


class ComputeQuotaTarget(Base):
    """
    ComputeQuotaTarget
      The target entity to allocate compute resources to.

    Attributes
    ----------------------
    team_name: Name of the team to allocate compute resources to.
    fair_share_weight: Assigned entity fair-share weight. Idle compute will be shared across entities based on these assigned weights. This weight is only used when FairShare is enabled. A weight of 0 is the lowest priority and 100 is the highest. Weight 0 is the default.
    """

    team_name: StrPipeVar
    fair_share_weight: Optional[int] = Unassigned()


class ComputeQuotaSummary(Base):
    """
    ComputeQuotaSummary
      Summary of the compute allocation definition.

    Attributes
    ----------------------
    compute_quota_arn: ARN of the compute allocation definition.
    compute_quota_id: ID of the compute allocation definition.
    name: Name of the compute allocation definition.
    compute_quota_version: Version of the compute allocation definition.
    status: Status of the compute allocation definition.
    cluster_arn: ARN of the cluster.
    compute_quota_config: Configuration of the compute allocation definition. This includes the resource sharing option, and the setting to preempt low priority tasks.
    compute_quota_target: The target entity to allocate compute resources to.
    activation_state: The state of the compute allocation being described. Use to enable or disable compute allocation. Default is Enabled.
    creation_time: Creation time of the compute allocation definition.
    last_modified_time: Last modified time of the compute allocation definition.
    """

    compute_quota_arn: StrPipeVar
    compute_quota_id: StrPipeVar
    name: StrPipeVar
    status: StrPipeVar
    compute_quota_target: ComputeQuotaTarget
    creation_time: datetime.datetime
    compute_quota_version: Optional[int] = Unassigned()
    cluster_arn: Optional[StrPipeVar] = Unassigned()
    compute_quota_config: Optional[ComputeQuotaConfig] = Unassigned()
    activation_state: Optional[StrPipeVar] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class Concurrency(Base):
    """
    Concurrency

    Attributes
    ----------------------
    number_of_concurrent_users
    duration_in_seconds
    """

    number_of_concurrent_users: Optional[int] = Unassigned()
    duration_in_seconds: Optional[int] = Unassigned()


class ConditionStepMetadata(Base):
    """
    ConditionStepMetadata
      Metadata for a Condition step.

    Attributes
    ----------------------
    outcome: The outcome of the Condition step evaluation.
    """

    outcome: Optional[StrPipeVar] = Unassigned()


class ConflictException(Base):
    """
    ConflictException
      There was a conflict when you attempted to modify a SageMaker entity such as an Experiment or Artifact.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class RepositoryAuthConfig(Base):
    """
    RepositoryAuthConfig
      Specifies an authentication configuration for the private docker registry where your model image is hosted. Specify a value for this property only if you specified Vpc as the value for the RepositoryAccessMode field of the ImageConfig object that you passed to a call to CreateModel and the private Docker registry where the model image is hosted requires authentication.

    Attributes
    ----------------------
    repository_credentials_provider_arn: The Amazon Resource Name (ARN) of an Amazon Web Services Lambda function that provides credentials to authenticate to the private Docker registry where your model image is hosted. For information about how to create an Amazon Web Services Lambda function, see Create a Lambda function with the console in the Amazon Web Services Lambda Developer Guide.
    """

    repository_credentials_provider_arn: StrPipeVar


class ImageConfig(Base):
    """
    ImageConfig
      Specifies whether the model container is in Amazon ECR or a private Docker registry accessible from your Amazon Virtual Private Cloud (VPC).

    Attributes
    ----------------------
    repository_access_mode: Set this to one of the following values:    Platform - The model image is hosted in Amazon ECR.    Vpc - The model image is hosted in a private Docker registry in your VPC.
    repository_auth_config: (Optional) Specifies an authentication configuration for the private docker registry where your model image is hosted. Specify a value for this property only if you specified Vpc as the value for the RepositoryAccessMode field, and the private Docker registry where the model image is hosted requires authentication.
    """

    repository_access_mode: StrPipeVar
    repository_auth_config: Optional[RepositoryAuthConfig] = Unassigned()


class MultiModelConfig(Base):
    """
    MultiModelConfig
      Specifies additional configuration for hosting multi-model endpoints.

    Attributes
    ----------------------
    model_cache_setting: Whether to cache models for a multi-model endpoint. By default, multi-model endpoints cache models so that a model does not have to be loaded into memory each time it is invoked. Some use cases do not benefit from model caching. For example, if an endpoint hosts a large number of models that are each invoked infrequently, the endpoint might perform better if you disable model caching. To disable model caching, set the value of this parameter to Disabled.
    model_load_concurrency_factor
    """

    model_cache_setting: Optional[StrPipeVar] = Unassigned()
    model_load_concurrency_factor: Optional[int] = Unassigned()


class ContainerDefinition(Base):
    """
    ContainerDefinition
      Describes the container, as part of model definition.

    Attributes
    ----------------------
    container_hostname: This parameter is ignored for models that contain only a PrimaryContainer. When a ContainerDefinition is part of an inference pipeline, the value of the parameter uniquely identifies the container for the purposes of logging and metrics. For information, see Use Logs and Metrics to Monitor an Inference Pipeline. If you don't specify a value for this parameter for a ContainerDefinition that is part of an inference pipeline, a unique name is automatically assigned based on the position of the ContainerDefinition in the pipeline. If you specify a value for the ContainerHostName for any ContainerDefinition that is part of an inference pipeline, you must specify a value for the ContainerHostName parameter of every ContainerDefinition in that pipeline.
    image: The path where inference code is stored. This can be either in Amazon EC2 Container Registry or in a Docker registry that is accessible from the same VPC that you configure for your endpoint. If you are using your own custom algorithm instead of an algorithm provided by SageMaker, the inference code must meet SageMaker requirements. SageMaker supports both registry/repository[:tag] and registry/repository[@digest] image path formats. For more information, see Using Your Own Algorithms with Amazon SageMaker.   The model artifacts in an Amazon S3 bucket and the Docker image for inference container in Amazon EC2 Container Registry must be in the same region as the model or endpoint you are creating.
    image_config: Specifies whether the model container is in Amazon ECR or a private Docker registry accessible from your Amazon Virtual Private Cloud (VPC). For information about storing containers in a private Docker registry, see Use a Private Docker Registry for Real-Time Inference Containers.   The model artifacts in an Amazon S3 bucket and the Docker image for inference container in Amazon EC2 Container Registry must be in the same region as the model or endpoint you are creating.
    mode: Whether the container hosts a single model or multiple models.
    model_data_url: The S3 path where the model artifacts, which result from model training, are stored. This path must point to a single gzip compressed tar archive (.tar.gz suffix). The S3 path is required for SageMaker built-in algorithms, but not if you use your own algorithms. For more information on built-in algorithms, see Common Parameters.   The model artifacts must be in an S3 bucket that is in the same region as the model or endpoint you are creating.  If you provide a value for this parameter, SageMaker uses Amazon Web Services Security Token Service to download model artifacts from the S3 path you provide. Amazon Web Services STS is activated in your Amazon Web Services account by default. If you previously deactivated Amazon Web Services STS for a region, you need to reactivate Amazon Web Services STS for that region. For more information, see Activating and Deactivating Amazon Web Services STS in an Amazon Web Services Region in the Amazon Web Services Identity and Access Management User Guide.  If you use a built-in algorithm to create a model, SageMaker requires that you provide a S3 path to the model artifacts in ModelDataUrl.
    model_data_source: Specifies the location of ML model data to deploy.  Currently you cannot use ModelDataSource in conjunction with SageMaker batch transform, SageMaker serverless endpoints, SageMaker multi-model endpoints, and SageMaker Marketplace.
    additional_model_data_sources: Data sources that are available to your model in addition to the one that you specify for ModelDataSource when you use the CreateModel action.
    environment: The environment variables to set in the Docker container. Don't include any sensitive data in your environment variables. The maximum length of each key and value in the Environment map is 1024 bytes. The maximum length of all keys and values in the map, combined, is 32 KB. If you pass multiple containers to a CreateModel request, then the maximum length of all of their maps, combined, is also 32 KB.
    model_package_name: The name or Amazon Resource Name (ARN) of the model package to use to create the model.
    inference_specification_name: The inference specification name in the model package version.
    multi_model_config: Specifies additional configuration for multi-model endpoints.
    """

    container_hostname: Optional[StrPipeVar] = Unassigned()
    image: Optional[StrPipeVar] = Unassigned()
    image_config: Optional[ImageConfig] = Unassigned()
    mode: Optional[StrPipeVar] = Unassigned()
    model_data_url: Optional[StrPipeVar] = Unassigned()
    model_data_source: Optional[ModelDataSource] = Unassigned()
    additional_model_data_sources: Optional[List[AdditionalModelDataSource]] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    model_package_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    inference_specification_name: Optional[StrPipeVar] = Unassigned()
    multi_model_config: Optional[MultiModelConfig] = Unassigned()


class ContextSource(Base):
    """
    ContextSource
      A structure describing the source of a context.

    Attributes
    ----------------------
    source_uri: The URI of the source.
    source_type: The type of the source.
    source_id: The ID of the source.
    """

    source_uri: StrPipeVar
    source_type: Optional[StrPipeVar] = Unassigned()
    source_id: Optional[StrPipeVar] = Unassigned()


class ContextSummary(Base):
    """
    ContextSummary
      Lists a summary of the properties of a context. A context provides a logical grouping of other entities.

    Attributes
    ----------------------
    context_arn: The Amazon Resource Name (ARN) of the context.
    context_name: The name of the context.
    source: The source of the context.
    context_type: The type of the context.
    creation_time: When the context was created.
    last_modified_time: When the context was last modified.
    """

    context_arn: Optional[StrPipeVar] = Unassigned()
    context_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    source: Optional[ContextSource] = Unassigned()
    context_type: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class ContinuousParameter(Base):
    """
    ContinuousParameter

    Attributes
    ----------------------
    name
    min_value
    max_value
    scaling_type
    """

    name: Optional[StrPipeVar] = Unassigned()
    min_value: Optional[float] = Unassigned()
    max_value: Optional[float] = Unassigned()
    scaling_type: Optional[StrPipeVar] = Unassigned()


class ContinuousParameterRange(Base):
    """
    ContinuousParameterRange
      A list of continuous hyperparameters to tune.

    Attributes
    ----------------------
    name: The name of the continuous hyperparameter to tune.
    min_value: The minimum value for the hyperparameter. The tuning job uses floating-point values between this value and MaxValuefor tuning.
    max_value: The maximum value for the hyperparameter. The tuning job uses floating-point values between MinValue value and this value for tuning.
    scaling_type: The scale that hyperparameter tuning uses to search the hyperparameter range. For information about choosing a hyperparameter scale, see Hyperparameter Scaling. One of the following values:  Auto  SageMaker hyperparameter tuning chooses the best scale for the hyperparameter.  Linear  Hyperparameter tuning searches the values in the hyperparameter range by using a linear scale.  Logarithmic  Hyperparameter tuning searches the values in the hyperparameter range by using a logarithmic scale. Logarithmic scaling works only for ranges that have only values greater than 0.  ReverseLogarithmic  Hyperparameter tuning searches the values in the hyperparameter range by using a reverse logarithmic scale. Reverse logarithmic scaling works only for ranges that are entirely within the range 0&lt;=x&lt;1.0.
    """

    name: StrPipeVar
    min_value: StrPipeVar
    max_value: StrPipeVar
    scaling_type: Optional[StrPipeVar] = Unassigned()


class ContinuousParameterRangeSpecification(Base):
    """
    ContinuousParameterRangeSpecification
      Defines the possible values for a continuous hyperparameter.

    Attributes
    ----------------------
    min_value: The minimum floating-point value allowed.
    max_value: The maximum floating-point value allowed.
    """

    min_value: StrPipeVar
    max_value: StrPipeVar


class ConvergenceDetected(Base):
    """
    ConvergenceDetected
      A flag to indicating that automatic model tuning (AMT) has detected model convergence, defined as a lack of significant improvement (1% or less) against an objective metric.

    Attributes
    ----------------------
    complete_on_convergence: A flag to stop a tuning job once AMT has detected that the job has converged.
    """

    complete_on_convergence: Optional[StrPipeVar] = Unassigned()


class MetadataProperties(Base):
    """
    MetadataProperties
      Metadata properties of the tracking entity, trial, or trial component.

    Attributes
    ----------------------
    commit_id: The commit ID.
    repository: The repository.
    generated_by: The entity this entity was generated by.
    project_id: The project ID.
    branch_name
    """

    commit_id: Optional[StrPipeVar] = Unassigned()
    repository: Optional[StrPipeVar] = Unassigned()
    generated_by: Optional[StrPipeVar] = Unassigned()
    project_id: Optional[StrPipeVar] = Unassigned()
    branch_name: Optional[StrPipeVar] = Unassigned()


class IntegerParameterRangeSpecification(Base):
    """
    IntegerParameterRangeSpecification
      Defines the possible values for an integer hyperparameter.

    Attributes
    ----------------------
    min_value: The minimum integer value allowed.
    max_value: The maximum integer value allowed.
    """

    min_value: StrPipeVar
    max_value: StrPipeVar


class ParameterRange(Base):
    """
    ParameterRange
      Defines the possible values for categorical, continuous, and integer hyperparameters to be used by an algorithm.

    Attributes
    ----------------------
    integer_parameter_range_specification: A IntegerParameterRangeSpecification object that defines the possible values for an integer hyperparameter.
    continuous_parameter_range_specification: A ContinuousParameterRangeSpecification object that defines the possible values for a continuous hyperparameter.
    categorical_parameter_range_specification: A CategoricalParameterRangeSpecification object that defines the possible values for a categorical hyperparameter.
    """

    integer_parameter_range_specification: Optional[IntegerParameterRangeSpecification] = (
        Unassigned()
    )
    continuous_parameter_range_specification: Optional[ContinuousParameterRangeSpecification] = (
        Unassigned()
    )
    categorical_parameter_range_specification: Optional[CategoricalParameterRangeSpecification] = (
        Unassigned()
    )


class HyperParameterSpecification(Base):
    """
    HyperParameterSpecification
      Defines a hyperparameter to be used by an algorithm.

    Attributes
    ----------------------
    name: The name of this hyperparameter. The name must be unique.
    description: A brief description of the hyperparameter.
    type: The type of this hyperparameter. The valid types are Integer, Continuous, Categorical, and FreeText.
    range: The allowed range for this hyperparameter.
    is_tunable: Indicates whether this hyperparameter is tunable in a hyperparameter tuning job.
    is_required: Indicates whether this hyperparameter is required.
    default_value: The default value for this hyperparameter. If a default value is specified, a hyperparameter cannot be required.
    default_scaling_type
    """

    name: StrPipeVar
    type: StrPipeVar
    description: Optional[StrPipeVar] = Unassigned()
    range: Optional[ParameterRange] = Unassigned()
    is_tunable: Optional[bool] = Unassigned()
    is_required: Optional[bool] = Unassigned()
    default_value: Optional[StrPipeVar] = Unassigned()
    default_scaling_type: Optional[StrPipeVar] = Unassigned()


class HyperParameterTuningJobObjective(Base):
    """
    HyperParameterTuningJobObjective
      Defines the objective metric for a hyperparameter tuning job. Hyperparameter tuning uses the value of this metric to evaluate the training jobs it launches, and returns the training job that results in either the highest or lowest value for this metric, depending on the value you specify for the Type parameter. If you want to define a custom objective metric, see Define metrics and environment variables.

    Attributes
    ----------------------
    type: Whether to minimize or maximize the objective metric.
    metric_name: The name of the metric to use for the objective metric.
    """

    type: StrPipeVar
    metric_name: StrPipeVar


class TrainingSpecification(Base):
    """
    TrainingSpecification
      Defines how the algorithm is used for a training job.

    Attributes
    ----------------------
    training_image: The Amazon ECR registry path of the Docker image that contains the training algorithm.
    training_image_digest: An MD5 hash of the training algorithm that identifies the Docker image used for training.
    supported_hyper_parameters: A list of the HyperParameterSpecification objects, that define the supported hyperparameters. This is required if the algorithm supports automatic model tuning.&gt;
    supported_training_instance_types: A list of the instance types that this algorithm can use for training.
    supports_distributed_training: Indicates whether the algorithm supports distributed training. If set to false, buyers can't request more than one instance during training.
    metric_definitions: A list of MetricDefinition objects, which are used for parsing metrics generated by the algorithm.
    training_channels: A list of ChannelSpecification objects, which specify the input sources to be used by the algorithm.
    supported_tuning_job_objective_metrics: A list of the metrics that the algorithm emits that can be used as the objective metric in a hyperparameter tuning job.
    additional_s3_data_source: The additional data source used during the training job.
    """

    training_image: StrPipeVar
    supported_training_instance_types: List[StrPipeVar]
    training_channels: List[ChannelSpecification]
    training_image_digest: Optional[StrPipeVar] = Unassigned()
    supported_hyper_parameters: Optional[List[HyperParameterSpecification]] = Unassigned()
    supports_distributed_training: Optional[bool] = Unassigned()
    metric_definitions: Optional[List[MetricDefinition]] = Unassigned()
    supported_tuning_job_objective_metrics: Optional[List[HyperParameterTuningJobObjective]] = (
        Unassigned()
    )
    additional_s3_data_source: Optional[AdditionalS3DataSource] = Unassigned()


class ImageUrlOverrides(Base):
    """
    ImageUrlOverrides

    Attributes
    ----------------------
    data_builder_image_url
    data_processing_image_url
    pipeline_recommender_image_url
    agt_image_url
    multimodal_pretraining_image_url
    robotorch_image_url
    time_series_pre_training_image_url
    time_series_training_image_url
    thundera_image_url
    """

    data_builder_image_url: Optional[StrPipeVar] = Unassigned()
    data_processing_image_url: Optional[StrPipeVar] = Unassigned()
    pipeline_recommender_image_url: Optional[StrPipeVar] = Unassigned()
    agt_image_url: Optional[StrPipeVar] = Unassigned()
    multimodal_pretraining_image_url: Optional[StrPipeVar] = Unassigned()
    robotorch_image_url: Optional[StrPipeVar] = Unassigned()
    time_series_pre_training_image_url: Optional[StrPipeVar] = Unassigned()
    time_series_training_image_url: Optional[StrPipeVar] = Unassigned()
    thundera_image_url: Optional[StrPipeVar] = Unassigned()


class ModelDeployConfig(Base):
    """
    ModelDeployConfig
      Specifies how to generate the endpoint name for an automatic one-click Autopilot model deployment.

    Attributes
    ----------------------
    model_deploy_mode
    auto_generate_endpoint_name: Set to True to automatically generate an endpoint name for a one-click Autopilot model deployment; set to False otherwise. The default value is False.  If you set AutoGenerateEndpointName to True, do not specify the EndpointName; otherwise a 400 error is thrown.
    endpoint_name: Specifies the endpoint name to use for a one-click Autopilot model deployment if the endpoint name is not generated automatically.  Specify the EndpointName if and only if you set AutoGenerateEndpointName to False; otherwise a 400 error is thrown.
    endpoint_config_definitions
    endpoint_definitions
    """

    model_deploy_mode: Optional[StrPipeVar] = Unassigned()
    auto_generate_endpoint_name: Optional[bool] = Unassigned()
    endpoint_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    endpoint_config_definitions: Optional[List[AutoMLEndpointConfigDefinition]] = Unassigned()
    endpoint_definitions: Optional[List[AutoMLEndpointDefinition]] = Unassigned()


class PriorityClass(Base):
    """
    PriorityClass
      Priority class configuration. When included in PriorityClasses, these class configurations define how tasks are queued.

    Attributes
    ----------------------
    name: Name of the priority class.
    weight: Weight of the priority class. The value is within a range from 0 to 100, where 0 is the default. A weight of 0 is the lowest priority and 100 is the highest. Weight 0 is the default.
    """

    name: StrPipeVar
    weight: int


class SchedulerConfig(Base):
    """
    SchedulerConfig
      Cluster policy configuration. This policy is used for task prioritization and fair-share allocation. This helps prioritize critical workloads and distributes idle compute across entities.

    Attributes
    ----------------------
    priority_classes: List of the priority classes, PriorityClass, of the cluster policy. When specified, these class configurations define how tasks are queued.
    fair_share: When enabled, entities borrow idle compute based on their assigned FairShareWeight. When disabled, entities borrow idle compute based on a first-come first-serve basis. Default is Enabled.
    """

    priority_classes: Optional[List[PriorityClass]] = Unassigned()
    fair_share: Optional[StrPipeVar] = Unassigned()


class InputConfig(Base):
    """
    InputConfig
      Contains information about the location of input model artifacts, the name and shape of the expected data inputs, and the framework in which the model was trained.

    Attributes
    ----------------------
    s3_uri: The S3 path where the model artifacts, which result from model training, are stored. This path must point to a single gzip compressed tar archive (.tar.gz suffix).
    data_input_config: Specifies the name and shape of the expected data inputs for your trained model with a JSON dictionary form. The data inputs are Framework specific.     TensorFlow: You must specify the name and shape (NHWC format) of the expected data inputs using a dictionary format for your trained model. The dictionary formats required for the console and CLI are different.   Examples for one input:   If using the console, {"input":[1,1024,1024,3]}    If using the CLI, {\"input\":[1,1024,1024,3]}      Examples for two inputs:   If using the console, {"data1": [1,28,28,1], "data2":[1,28,28,1]}    If using the CLI, {\"data1\": [1,28,28,1], \"data2\":[1,28,28,1]}         KERAS: You must specify the name and shape (NCHW format) of expected data inputs using a dictionary format for your trained model. Note that while Keras model artifacts should be uploaded in NHWC (channel-last) format, DataInputConfig should be specified in NCHW (channel-first) format. The dictionary formats required for the console and CLI are different.   Examples for one input:   If using the console, {"input_1":[1,3,224,224]}    If using the CLI, {\"input_1\":[1,3,224,224]}      Examples for two inputs:   If using the console, {"input_1": [1,3,224,224], "input_2":[1,3,224,224]}     If using the CLI, {\"input_1\": [1,3,224,224], \"input_2\":[1,3,224,224]}         MXNET/ONNX/DARKNET: You must specify the name and shape (NCHW format) of the expected data inputs in order using a dictionary format for your trained model. The dictionary formats required for the console and CLI are different.   Examples for one input:   If using the console, {"data":[1,3,1024,1024]}    If using the CLI, {\"data\":[1,3,1024,1024]}      Examples for two inputs:   If using the console, {"var1": [1,1,28,28], "var2":[1,1,28,28]}     If using the CLI, {\"var1\": [1,1,28,28], \"var2\":[1,1,28,28]}         PyTorch: You can either specify the name and shape (NCHW format) of expected data inputs in order using a dictionary format for your trained model or you can specify the shape only using a list format. The dictionary formats required for the console and CLI are different. The list formats for the console and CLI are the same.   Examples for one input in dictionary format:   If using the console, {"input0":[1,3,224,224]}    If using the CLI, {\"input0\":[1,3,224,224]}      Example for one input in list format: [[1,3,224,224]]    Examples for two inputs in dictionary format:   If using the console, {"input0":[1,3,224,224], "input1":[1,3,224,224]}    If using the CLI, {\"input0\":[1,3,224,224], \"input1\":[1,3,224,224]}       Example for two inputs in list format: [[1,3,224,224], [1,3,224,224]]       XGBOOST: input data name and shape are not needed.    DataInputConfig supports the following parameters for CoreML TargetDevice (ML Model format):    shape: Input shape, for example {"input_1": {"shape": [1,224,224,3]}}. In addition to static input shapes, CoreML converter supports Flexible input shapes:   Range Dimension. You can use the Range Dimension feature if you know the input shape will be within some specific interval in that dimension, for example: {"input_1": {"shape": ["1..10", 224, 224, 3]}}    Enumerated shapes. Sometimes, the models are trained to work only on a select set of inputs. You can enumerate all supported input shapes, for example: {"input_1": {"shape": [[1, 224, 224, 3], [1, 160, 160, 3]]}}       default_shape: Default input shape. You can set a default shape during conversion for both Range Dimension and Enumerated Shapes. For example {"input_1": {"shape": ["1..10", 224, 224, 3], "default_shape": [1, 224, 224, 3]}}     type: Input type. Allowed values: Image and Tensor. By default, the converter generates an ML Model with inputs of type Tensor (MultiArray). User can set input type to be Image. Image input type requires additional input parameters such as bias and scale.    bias: If the input type is an Image, you need to provide the bias vector.    scale: If the input type is an Image, you need to provide a scale factor.   CoreML ClassifierConfig parameters can be specified using OutputConfig CompilerOptions. CoreML converter supports Tensorflow and PyTorch models. CoreML conversion examples:   Tensor type input:    "DataInputConfig": {"input_1": {"shape": [[1,224,224,3], [1,160,160,3]], "default_shape": [1,224,224,3]}}      Tensor type input without input name (PyTorch):    "DataInputConfig": [{"shape": [[1,3,224,224], [1,3,160,160]], "default_shape": [1,3,224,224]}]      Image type input:    "DataInputConfig": {"input_1": {"shape": [[1,224,224,3], [1,160,160,3]], "default_shape": [1,224,224,3], "type": "Image", "bias": [-1,-1,-1], "scale": 0.007843137255}}     "CompilerOptions": {"class_labels": "imagenet_labels_1000.txt"}      Image type input without input name (PyTorch):    "DataInputConfig": [{"shape": [[1,3,224,224], [1,3,160,160]], "default_shape": [1,3,224,224], "type": "Image", "bias": [-1,-1,-1], "scale": 0.007843137255}]     "CompilerOptions": {"class_labels": "imagenet_labels_1000.txt"}      Depending on the model format, DataInputConfig requires the following parameters for ml_eia2 OutputConfig:TargetDevice.   For TensorFlow models saved in the SavedModel format, specify the input names from signature_def_key and the input model shapes for DataInputConfig. Specify the signature_def_key in  OutputConfig:CompilerOptions  if the model does not use TensorFlow's default signature def key. For example:    "DataInputConfig": {"inputs": [1, 224, 224, 3]}     "CompilerOptions": {"signature_def_key": "serving_custom"}      For TensorFlow models saved as a frozen graph, specify the input tensor names and shapes in DataInputConfig and the output tensor names for output_names in  OutputConfig:CompilerOptions . For example:    "DataInputConfig": {"input_tensor:0": [1, 224, 224, 3]}     "CompilerOptions": {"output_names": ["output_tensor:0"]}
    framework: Identifies the framework in which the model was trained. For example: TENSORFLOW.
    framework_version: Specifies the framework version to use. This API field is only supported for the MXNet, PyTorch, TensorFlow and TensorFlow Lite frameworks. For information about framework versions supported for cloud targets and edge devices, see Cloud Supported Instance Types and Frameworks and Edge Supported Frameworks.
    """

    s3_uri: StrPipeVar
    framework: StrPipeVar
    data_input_config: Optional[StrPipeVar] = Unassigned()
    framework_version: Optional[StrPipeVar] = Unassigned()


class TargetPlatform(Base):
    """
    TargetPlatform
      Contains information about a target platform that you want your model to run on, such as OS, architecture, and accelerators. It is an alternative of TargetDevice.

    Attributes
    ----------------------
    os: Specifies a target platform OS.    LINUX: Linux-based operating systems.    ANDROID: Android operating systems. Android API level can be specified using the ANDROID_PLATFORM compiler option. For example, "CompilerOptions": {'ANDROID_PLATFORM': 28}
    arch: Specifies a target platform architecture.    X86_64: 64-bit version of the x86 instruction set.    X86: 32-bit version of the x86 instruction set.    ARM64: ARMv8 64-bit CPU.    ARM_EABIHF: ARMv7 32-bit, Hard Float.    ARM_EABI: ARMv7 32-bit, Soft Float. Used by Android 32-bit ARM platform.
    accelerator: Specifies a target platform accelerator (optional).    NVIDIA: Nvidia graphics processing unit. It also requires gpu-code, trt-ver, cuda-ver compiler options    MALI: ARM Mali graphics processor    INTEL_GRAPHICS: Integrated Intel graphics
    """

    os: StrPipeVar
    arch: StrPipeVar
    accelerator: Optional[StrPipeVar] = Unassigned()


class OutputConfig(Base):
    """
    OutputConfig
      Contains information about the output location for the compiled model and the target device that the model runs on. TargetDevice and TargetPlatform are mutually exclusive, so you need to choose one between the two to specify your target device or platform. If you cannot find your device you want to use from the TargetDevice list, use TargetPlatform to describe the platform of your edge device and CompilerOptions if there are specific settings that are required or recommended to use for particular TargetPlatform.

    Attributes
    ----------------------
    s3_output_location: Identifies the S3 bucket where you want Amazon SageMaker AI to store the model artifacts. For example, s3://bucket-name/key-name-prefix.
    target_device: Identifies the target device or the machine learning instance that you want to run your model on after the compilation has completed. Alternatively, you can specify OS, architecture, and accelerator using TargetPlatform fields. It can be used instead of TargetPlatform.  Currently ml_trn1 is available only in US East (N. Virginia) Region, and ml_inf2 is available only in US East (Ohio) Region.
    target_platform: Contains information about a target platform that you want your model to run on, such as OS, architecture, and accelerators. It is an alternative of TargetDevice. The following examples show how to configure the TargetPlatform and CompilerOptions JSON strings for popular target platforms:    Raspberry Pi 3 Model B+  "TargetPlatform": {"Os": "LINUX", "Arch": "ARM_EABIHF"},    "CompilerOptions": {'mattr': ['+neon']}    Jetson TX2  "TargetPlatform": {"Os": "LINUX", "Arch": "ARM64", "Accelerator": "NVIDIA"},    "CompilerOptions": {'gpu-code': 'sm_62', 'trt-ver': '6.0.1', 'cuda-ver': '10.0'}    EC2 m5.2xlarge instance OS  "TargetPlatform": {"Os": "LINUX", "Arch": "X86_64", "Accelerator": "NVIDIA"},    "CompilerOptions": {'mcpu': 'skylake-avx512'}    RK3399  "TargetPlatform": {"Os": "LINUX", "Arch": "ARM64", "Accelerator": "MALI"}    ARMv7 phone (CPU)  "TargetPlatform": {"Os": "ANDROID", "Arch": "ARM_EABI"},    "CompilerOptions": {'ANDROID_PLATFORM': 25, 'mattr': ['+neon']}    ARMv8 phone (CPU)  "TargetPlatform": {"Os": "ANDROID", "Arch": "ARM64"},    "CompilerOptions": {'ANDROID_PLATFORM': 29}
    compiler_options: Specifies additional parameters for compiler options in JSON format. The compiler options are TargetPlatform specific. It is required for NVIDIA accelerators and highly recommended for CPU compilations. For any other cases, it is optional to specify CompilerOptions.     DTYPE: Specifies the data type for the input. When compiling for ml_* (except for ml_inf) instances using PyTorch framework, provide the data type (dtype) of the model's input. "float32" is used if "DTYPE" is not specified. Options for data type are:   float32: Use either "float" or "float32".   int64: Use either "int64" or "long".    For example, {"dtype" : "float32"}.    CPU: Compilation for CPU supports the following compiler options.    mcpu: CPU micro-architecture. For example, {'mcpu': 'skylake-avx512'}     mattr: CPU flags. For example, {'mattr': ['+neon', '+vfpv4']}       ARM: Details of ARM CPU compilations.    NEON: NEON is an implementation of the Advanced SIMD extension used in ARMv7 processors. For example, add {'mattr': ['+neon']} to the compiler options if compiling for ARM 32-bit platform with the NEON support.      NVIDIA: Compilation for NVIDIA GPU supports the following compiler options.    gpu_code: Specifies the targeted architecture.    trt-ver: Specifies the TensorRT versions in x.y.z. format.    cuda-ver: Specifies the CUDA version in x.y format.   For example, {'gpu-code': 'sm_72', 'trt-ver': '6.0.1', 'cuda-ver': '10.1'}     ANDROID: Compilation for the Android OS supports the following compiler options:    ANDROID_PLATFORM: Specifies the Android API levels. Available levels range from 21 to 29. For example, {'ANDROID_PLATFORM': 28}.    mattr: Add {'mattr': ['+neon']} to compiler options if compiling for ARM 32-bit platform with NEON support.      INFERENTIA: Compilation for target ml_inf1 uses compiler options passed in as a JSON string. For example, "CompilerOptions": "\"--verbose 1 --num-neuroncores 2 -O2\"".  For information about supported compiler options, see  Neuron Compiler CLI Reference Guide.     CoreML: Compilation for the CoreML OutputConfig TargetDevice supports the following compiler options:    class_labels: Specifies the classification labels file name inside input tar.gz file. For example, {"class_labels": "imagenet_labels_1000.txt"}. Labels inside the txt file should be separated by newlines.
    kms_key_id: The Amazon Web Services Key Management Service key (Amazon Web Services KMS) that Amazon SageMaker AI uses to encrypt your output models with Amazon S3 server-side encryption after compilation job. If you don't provide a KMS key ID, Amazon SageMaker AI uses the default KMS key for Amazon S3 for your role's account. For more information, see KMS-Managed Encryption Keys in the Amazon Simple Storage Service Developer Guide.  The KmsKeyId can be any of the following formats:    Key ID: 1234abcd-12ab-34cd-56ef-1234567890ab    Key ARN: arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab    Alias name: alias/ExampleAlias    Alias name ARN: arn:aws:kms:us-west-2:111122223333:alias/ExampleAlias
    """

    s3_output_location: StrPipeVar
    target_device: Optional[StrPipeVar] = Unassigned()
    target_platform: Optional[TargetPlatform] = Unassigned()
    compiler_options: Optional[StrPipeVar] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class NeoResourceConfig(Base):
    """
    NeoResourceConfig

    Attributes
    ----------------------
    volume_kms_key_id
    """

    volume_kms_key_id: StrPipeVar


class NeoVpcConfig(Base):
    """
    NeoVpcConfig
      The VpcConfig configuration object that specifies the VPC that you want the compilation jobs to connect to. For more information on controlling access to your Amazon S3 buckets used for compilation job, see Give Amazon SageMaker AI Compilation Jobs Access to Resources in Your Amazon VPC.

    Attributes
    ----------------------
    security_group_ids: The VPC security group IDs. IDs have the form of sg-xxxxxxxx. Specify the security groups for the VPC that is specified in the Subnets field.
    subnets: The ID of the subnets in the VPC that you want to connect the compilation job to for accessing the model in Amazon S3.
    """

    security_group_ids: List[StrPipeVar]
    subnets: List[StrPipeVar]


class CustomMonitoringAppSpecification(Base):
    """
    CustomMonitoringAppSpecification

    Attributes
    ----------------------
    image_uri
    container_entrypoint
    container_arguments
    environment
    record_preprocessor_source_uri
    post_analytics_processor_source_uri
    """

    image_uri: StrPipeVar
    container_entrypoint: Optional[List[StrPipeVar]] = Unassigned()
    container_arguments: Optional[List[StrPipeVar]] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    record_preprocessor_source_uri: Optional[StrPipeVar] = Unassigned()
    post_analytics_processor_source_uri: Optional[StrPipeVar] = Unassigned()


class ProcessingS3Input(Base):
    """
    ProcessingS3Input
      Configuration for downloading input data from Amazon S3 into the processing container.

    Attributes
    ----------------------
    s3_uri: The URI of the Amazon S3 prefix Amazon SageMaker downloads data required to run a processing job.
    local_path: The local path in your container where you want Amazon SageMaker to write input data to. LocalPath is an absolute path to the input data and must begin with /opt/ml/processing/. LocalPath is a required parameter when AppManaged is False (default).
    s3_data_type: Whether you use an S3Prefix or a ManifestFile for the data type. If you choose S3Prefix, S3Uri identifies a key name prefix. Amazon SageMaker uses all objects with the specified key name prefix for the processing job. If you choose ManifestFile, S3Uri identifies an object that is a manifest file containing a list of object keys that you want Amazon SageMaker to use for the processing job.
    s3_input_mode: Whether to use File or Pipe input mode. In File mode, Amazon SageMaker copies the data from the input source onto the local ML storage volume before starting your processing container. This is the most commonly used input mode. In Pipe mode, Amazon SageMaker streams input data from the source directly to your processing container into named pipes without using the ML storage volume.
    s3_data_distribution_type: Whether to distribute the data from Amazon S3 to all processing instances with FullyReplicated, or whether the data from Amazon S3 is sharded by Amazon S3 key, downloading one shard of data to each processing instance.
    s3_compression_type: Whether to GZIP-decompress the data in Amazon S3 as it is streamed into the processing container. Gzip can only be used when Pipe mode is specified as the S3InputMode. In Pipe mode, Amazon SageMaker streams input data from the source directly to your container without using the EBS volume.
    """

    s3_uri: StrPipeVar
    s3_data_type: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()
    s3_input_mode: Optional[StrPipeVar] = Unassigned()
    s3_data_distribution_type: Optional[StrPipeVar] = Unassigned()
    s3_compression_type: Optional[StrPipeVar] = Unassigned()


class RedshiftDatasetDefinition(Base):
    """
    RedshiftDatasetDefinition
      Configuration for Redshift Dataset Definition input.

    Attributes
    ----------------------
    cluster_id
    database
    db_user
    query_string
    cluster_role_arn: The IAM role attached to your Redshift cluster that Amazon SageMaker uses to generate datasets.
    output_s3_uri: The location in Amazon S3 where the Redshift query results are stored.
    output_dataset_s3_uri
    kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt data from a Redshift execution.
    output_format
    output_compression
    """

    cluster_id: StrPipeVar
    database: StrPipeVar
    db_user: StrPipeVar
    query_string: StrPipeVar
    cluster_role_arn: StrPipeVar
    output_s3_uri: StrPipeVar
    output_format: StrPipeVar
    output_dataset_s3_uri: Optional[StrPipeVar] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    output_compression: Optional[StrPipeVar] = Unassigned()


class SnowflakeQueryVariable(Base):
    """
    SnowflakeQueryVariable

    Attributes
    ----------------------
    value
    """

    value: StrPipeVar


class SnowflakeDatasetDefinition(Base):
    """
    SnowflakeDatasetDefinition

    Attributes
    ----------------------
    warehouse
    database
    schema
    snowflake_role
    secret_arn
    query_string
    query_variables
    output_s3_uri
    output_dataset_s3_uri
    storage_integration
    output_format_type
    output_compression
    output_format_name
    kms_key_id
    """

    warehouse: StrPipeVar
    secret_arn: StrPipeVar
    query_string: StrPipeVar
    output_s3_uri: StrPipeVar
    storage_integration: StrPipeVar
    database: Optional[StrPipeVar] = Unassigned()
    schema: Optional[StrPipeVar] = Unassigned()
    snowflake_role: Optional[StrPipeVar] = Unassigned()
    query_variables: Optional[List[SnowflakeQueryVariable]] = Unassigned()
    output_dataset_s3_uri: Optional[StrPipeVar] = Unassigned()
    output_format_type: Optional[StrPipeVar] = Unassigned()
    output_compression: Optional[StrPipeVar] = Unassigned()
    output_format_name: Optional[StrPipeVar] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class DatasetDefinition(Base):
    """
    DatasetDefinition
      Configuration for Dataset Definition inputs. The Dataset Definition input must specify exactly one of either AthenaDatasetDefinition or RedshiftDatasetDefinition types.

    Attributes
    ----------------------
    athena_dataset_definition
    redshift_dataset_definition
    local_path: The local path where you want Amazon SageMaker to download the Dataset Definition inputs to run a processing job. LocalPath is an absolute path to the input data. This is a required parameter when AppManaged is False (default).
    data_distribution_type: Whether the generated dataset is FullyReplicated or ShardedByS3Key (default).
    input_mode: Whether to use File or Pipe input mode. In File (default) mode, Amazon SageMaker copies the data from the input source onto the local Amazon Elastic Block Store (Amazon EBS) volumes before starting your training algorithm. This is the most commonly used input mode. In Pipe mode, Amazon SageMaker streams input data from the source directly to your algorithm without using the EBS volume.
    snowflake_dataset_definition
    """

    athena_dataset_definition: Optional[AthenaDatasetDefinition] = Unassigned()
    redshift_dataset_definition: Optional[RedshiftDatasetDefinition] = Unassigned()
    local_path: Optional[StrPipeVar] = Unassigned()
    data_distribution_type: Optional[StrPipeVar] = Unassigned()
    input_mode: Optional[StrPipeVar] = Unassigned()
    snowflake_dataset_definition: Optional[SnowflakeDatasetDefinition] = Unassigned()


class ProcessingInput(Base):
    """
    ProcessingInput
      The inputs for a processing job. The processing input must specify exactly one of either S3Input or DatasetDefinition types.

    Attributes
    ----------------------
    input_name: The name for the processing job input.
    app_managed: When True, input operations such as data download are managed natively by the processing job application. When False (default), input operations are managed by Amazon SageMaker.
    s3_input: Configuration for downloading input data from Amazon S3 into the processing container.
    dataset_definition: Configuration for a Dataset Definition input.
    """

    input_name: StrPipeVar
    app_managed: Optional[bool] = Unassigned()
    s3_input: Optional[ProcessingS3Input] = Unassigned()
    dataset_definition: Optional[DatasetDefinition] = Unassigned()


class EndpointInput(Base):
    """
    EndpointInput
      Input object for the endpoint

    Attributes
    ----------------------
    endpoint_name: An endpoint in customer's account which has enabled DataCaptureConfig enabled.
    local_path: Path to the filesystem where the endpoint data is available to the container.
    s3_input_mode: Whether the Pipe or File is used as the input mode for transferring data for the monitoring job. Pipe mode is recommended for large datasets. File mode is useful for small files that fit in memory. Defaults to File.
    s3_data_distribution_type: Whether input data distributed in Amazon S3 is fully replicated or sharded by an Amazon S3 key. Defaults to FullyReplicated
    features_attribute: The attributes of the input data that are the input features.
    inference_attribute: The attribute of the input data that represents the ground truth label.
    probability_attribute: In a classification problem, the attribute that represents the class probability.
    probability_threshold_attribute: The threshold for the class probability to be evaluated as a positive result.
    start_time_offset: If specified, monitoring jobs substract this time from the start time. For information about using offsets for scheduling monitoring jobs, see Schedule Model Quality Monitoring Jobs.
    end_time_offset: If specified, monitoring jobs substract this time from the end time. For information about using offsets for scheduling monitoring jobs, see Schedule Model Quality Monitoring Jobs.
    variant_name
    exclude_features_attribute: The attributes of the input data to exclude from the analysis.
    """

    endpoint_name: Union[StrPipeVar, object]
    local_path: StrPipeVar
    s3_input_mode: Optional[StrPipeVar] = Unassigned()
    s3_data_distribution_type: Optional[StrPipeVar] = Unassigned()
    features_attribute: Optional[StrPipeVar] = Unassigned()
    inference_attribute: Optional[StrPipeVar] = Unassigned()
    probability_attribute: Optional[StrPipeVar] = Unassigned()
    probability_threshold_attribute: Optional[float] = Unassigned()
    start_time_offset: Optional[StrPipeVar] = Unassigned()
    end_time_offset: Optional[StrPipeVar] = Unassigned()
    variant_name: Optional[StrPipeVar] = Unassigned()
    exclude_features_attribute: Optional[StrPipeVar] = Unassigned()


class MonitoringGroundTruthS3Input(Base):
    """
    MonitoringGroundTruthS3Input
      The ground truth labels for the dataset used for the monitoring job.

    Attributes
    ----------------------
    s3_uri: The address of the Amazon S3 location of the ground truth labels.
    """

    s3_uri: Optional[StrPipeVar] = Unassigned()


class CustomMonitoringJobInput(Base):
    """
    CustomMonitoringJobInput

    Attributes
    ----------------------
    processing_inputs
    endpoint_input
    batch_transform_input
    ground_truth_s3_input
    """

    processing_inputs: Optional[List[ProcessingInput]] = Unassigned()
    endpoint_input: Optional[EndpointInput] = Unassigned()
    batch_transform_input: Optional[BatchTransformInput] = Unassigned()
    ground_truth_s3_input: Optional[MonitoringGroundTruthS3Input] = Unassigned()


class MonitoringS3Output(Base):
    """
    MonitoringS3Output
      Information about where and how you want to store the results of a monitoring job.

    Attributes
    ----------------------
    s3_uri: A URI that identifies the Amazon S3 storage location where Amazon SageMaker AI saves the results of a monitoring job.
    local_path: The local path to the Amazon S3 storage location where Amazon SageMaker AI saves the results of a monitoring job. LocalPath is an absolute path for the output data.
    s3_upload_mode: Whether to upload the results of the monitoring job continuously or after the job completes.
    """

    s3_uri: StrPipeVar
    local_path: StrPipeVar
    s3_upload_mode: Optional[StrPipeVar] = Unassigned()


class MonitoringOutput(Base):
    """
    MonitoringOutput
      The output object for a monitoring job.

    Attributes
    ----------------------
    s3_output: The Amazon S3 storage location where the results of a monitoring job are saved.
    """

    s3_output: MonitoringS3Output


class MonitoringOutputConfig(Base):
    """
    MonitoringOutputConfig
      The output configuration for monitoring jobs.

    Attributes
    ----------------------
    monitoring_outputs: Monitoring outputs for monitoring jobs. This is where the output of the periodic monitoring jobs is uploaded.
    kms_key_id: The Key Management Service (KMS) key that Amazon SageMaker AI uses to encrypt the model artifacts at rest using Amazon S3 server-side encryption.
    """

    monitoring_outputs: List[MonitoringOutput]
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class MonitoringClusterConfig(Base):
    """
    MonitoringClusterConfig
      Configuration for the cluster used to run model monitoring jobs.

    Attributes
    ----------------------
    instance_count: The number of ML compute instances to use in the model monitoring job. For distributed processing jobs, specify a value greater than 1. The default value is 1.
    instance_type: The ML compute instance type for the processing job.
    volume_size_in_gb: The size of the ML storage volume, in gigabytes, that you want to provision. You must specify sufficient ML storage for your scenario.
    volume_kms_key_id: The Key Management Service (KMS) key that Amazon SageMaker AI uses to encrypt data on the storage volume attached to the ML compute instance(s) that run the model monitoring job.
    """

    instance_count: int
    instance_type: StrPipeVar
    volume_size_in_gb: int
    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()


class MonitoringResources(Base):
    """
    MonitoringResources
      Identifies the resources to deploy for a monitoring job.

    Attributes
    ----------------------
    cluster_config: The configuration for the cluster resources used to run the processing job.
    """

    cluster_config: MonitoringClusterConfig


class MonitoringNetworkConfig(Base):
    """
    MonitoringNetworkConfig
      The networking configuration for the monitoring job.

    Attributes
    ----------------------
    enable_inter_container_traffic_encryption: Whether to encrypt all communications between the instances used for the monitoring jobs. Choose True to encrypt communications. Encryption provides greater security for distributed jobs, but the processing might take longer.
    enable_network_isolation: Whether to allow inbound and outbound network calls to and from the containers used for the monitoring job.
    vpc_config
    """

    enable_inter_container_traffic_encryption: Optional[bool] = Unassigned()
    enable_network_isolation: Optional[bool] = Unassigned()
    vpc_config: Optional[VpcConfig] = Unassigned()


class MonitoringStoppingCondition(Base):
    """
    MonitoringStoppingCondition
      A time limit for how long the monitoring job is allowed to run before stopping.

    Attributes
    ----------------------
    max_runtime_in_seconds: The maximum runtime allowed in seconds.  The MaxRuntimeInSeconds cannot exceed the frequency of the job. For data quality and model explainability, this can be up to 3600 seconds for an hourly schedule. For model bias and model quality hourly schedules, this can be up to 1800 seconds.
    """

    max_runtime_in_seconds: int


class MonitoringConstraintsResource(Base):
    """
    MonitoringConstraintsResource
      The constraints resource for a monitoring job.

    Attributes
    ----------------------
    s3_uri: The Amazon S3 URI for the constraints resource.
    """

    s3_uri: Optional[StrPipeVar] = Unassigned()


class MonitoringStatisticsResource(Base):
    """
    MonitoringStatisticsResource
      The statistics resource for a monitoring job.

    Attributes
    ----------------------
    s3_uri: The Amazon S3 URI for the statistics resource.
    """

    s3_uri: Optional[StrPipeVar] = Unassigned()


class DataQualityBaselineConfig(Base):
    """
    DataQualityBaselineConfig
      Configuration for monitoring constraints and monitoring statistics. These baseline resources are compared against the results of the current job from the series of jobs scheduled to collect data periodically.

    Attributes
    ----------------------
    baselining_job_name: The name of the job that performs baselining for the data quality monitoring job.
    constraints_resource
    statistics_resource
    """

    baselining_job_name: Optional[StrPipeVar] = Unassigned()
    constraints_resource: Optional[MonitoringConstraintsResource] = Unassigned()
    statistics_resource: Optional[MonitoringStatisticsResource] = Unassigned()


class DataQualityAppSpecification(Base):
    """
    DataQualityAppSpecification
      Information about the container that a data quality monitoring job runs.

    Attributes
    ----------------------
    image_uri: The container image that the data quality monitoring job runs.
    container_entrypoint: The entrypoint for a container used to run a monitoring job.
    container_arguments: The arguments to send to the container that the monitoring job runs.
    record_preprocessor_source_uri: An Amazon S3 URI to a script that is called per row prior to running analysis. It can base64 decode the payload and convert it into a flattened JSON so that the built-in container can use the converted data. Applicable only for the built-in (first party) containers.
    post_analytics_processor_source_uri: An Amazon S3 URI to a script that is called after analysis has been performed. Applicable only for the built-in (first party) containers.
    environment: Sets the environment variables in the container that the monitoring job runs.
    """

    image_uri: StrPipeVar
    container_entrypoint: Optional[List[StrPipeVar]] = Unassigned()
    container_arguments: Optional[List[StrPipeVar]] = Unassigned()
    record_preprocessor_source_uri: Optional[StrPipeVar] = Unassigned()
    post_analytics_processor_source_uri: Optional[StrPipeVar] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class DataQualityJobInput(Base):
    """
    DataQualityJobInput
      The input for the data quality monitoring job. Currently endpoints are supported for input.

    Attributes
    ----------------------
    endpoint_input
    batch_transform_input: Input object for the batch transform job.
    """

    endpoint_input: Optional[EndpointInput] = Unassigned()
    batch_transform_input: Optional[BatchTransformInput] = Unassigned()


class EdgeOutputConfig(Base):
    """
    EdgeOutputConfig
      The output configuration.

    Attributes
    ----------------------
    s3_output_location: The Amazon Simple Storage (S3) bucker URI.
    kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt data on the storage volume after compilation job. If you don't provide a KMS key ID, Amazon SageMaker uses the default KMS key for Amazon S3 for your role's account.
    preset_deployment_type: The deployment type SageMaker Edge Manager will create. Currently only supports Amazon Web Services IoT Greengrass Version 2 components.
    preset_deployment_config: The configuration used to create deployment artifacts. Specify configuration options with a JSON string. The available configuration options for each type are:    ComponentName (optional) - Name of the GreenGrass V2 component. If not specified, the default name generated consists of "SagemakerEdgeManager" and the name of your SageMaker Edge Manager packaging job.    ComponentDescription (optional) - Description of the component.    ComponentVersion (optional) - The version of the component.  Amazon Web Services IoT Greengrass uses semantic versions for components. Semantic versions follow a major.minor.patch number system. For example, version 1.0.0 represents the first major release for a component. For more information, see the semantic version specification.     PlatformOS (optional) - The name of the operating system for the platform. Supported platforms include Windows and Linux.    PlatformArchitecture (optional) - The processor architecture for the platform.  Supported architectures Windows include: Windows32_x86, Windows64_x64. Supported architectures for Linux include: Linux x86_64, Linux ARMV8.
    """

    s3_output_location: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    preset_deployment_type: Optional[StrPipeVar] = Unassigned()
    preset_deployment_config: Optional[StrPipeVar] = Unassigned()


class EnvironmentSettings(Base):
    """
    EnvironmentSettings

    Attributes
    ----------------------
    default_s3_artifact_path
    default_s3_kms_key_id
    """

    default_s3_artifact_path: Optional[StrPipeVar] = Unassigned()
    default_s3_kms_key_id: Optional[StrPipeVar] = Unassigned()


class SharingSettings(Base):
    """
    SharingSettings
      Specifies options for sharing Amazon SageMaker AI Studio notebooks. These settings are specified as part of DefaultUserSettings when the CreateDomain API is called, and as part of UserSettings when the CreateUserProfile API is called. When SharingSettings is not specified, notebook sharing isn't allowed.

    Attributes
    ----------------------
    notebook_output_option: Whether to include the notebook cell output when sharing the notebook. The default is Disabled.
    s3_output_path: When NotebookOutputOption is Allowed, the Amazon S3 bucket used to store the shared notebook snapshots.
    s3_kms_key_id: When NotebookOutputOption is Allowed, the Amazon Web Services Key Management Service (KMS) encryption key ID used to encrypt the notebook cell output in the Amazon S3 bucket.
    """

    notebook_output_option: Optional[StrPipeVar] = Unassigned()
    s3_output_path: Optional[StrPipeVar] = Unassigned()
    s3_kms_key_id: Optional[StrPipeVar] = Unassigned()


class JupyterServerAppSettings(Base):
    """
    JupyterServerAppSettings
      The JupyterServer app settings.

    Attributes
    ----------------------
    default_resource_spec: The default instance type and the Amazon Resource Name (ARN) of the default SageMaker AI image used by the JupyterServer app. If you use the LifecycleConfigArns parameter, then this parameter is also required.
    lifecycle_config_arns:  The Amazon Resource Name (ARN) of the Lifecycle Configurations attached to the JupyterServerApp. If you use this parameter, the DefaultResourceSpec parameter is also required.  To remove a Lifecycle Config, you must set LifecycleConfigArns to an empty list.
    code_repositories: A list of Git repositories that SageMaker AI automatically displays to users for cloning in the JupyterServer application.
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    lifecycle_config_arns: Optional[List[StrPipeVar]] = Unassigned()
    code_repositories: Optional[List[CodeRepository]] = Unassigned()


class KernelGatewayAppSettings(Base):
    """
    KernelGatewayAppSettings
      The KernelGateway app settings.

    Attributes
    ----------------------
    default_resource_spec: The default instance type and the Amazon Resource Name (ARN) of the default SageMaker AI image used by the KernelGateway app.  The Amazon SageMaker AI Studio UI does not use the default instance type value set here. The default instance type set here is used when Apps are created using the CLI or CloudFormation and the instance type parameter value is not passed.
    custom_images: A list of custom SageMaker AI images that are configured to run as a KernelGateway app. The maximum number of custom images are as follows.   On a domain level: 200   On a space level: 5   On a user profile level: 5
    lifecycle_config_arns:  The Amazon Resource Name (ARN) of the Lifecycle Configurations attached to the the user profile or domain.  To remove a Lifecycle Config, you must set LifecycleConfigArns to an empty list.
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    custom_images: Optional[List[CustomImage]] = Unassigned()
    lifecycle_config_arns: Optional[List[StrPipeVar]] = Unassigned()


class TensorBoardAppSettings(Base):
    """
    TensorBoardAppSettings
      The TensorBoard app settings.

    Attributes
    ----------------------
    default_resource_spec: The default instance type and the Amazon Resource Name (ARN) of the SageMaker AI image created on the instance.
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()


class RStudioServerProAppSettings(Base):
    """
    RStudioServerProAppSettings
      A collection of settings that configure user interaction with the RStudioServerPro app.

    Attributes
    ----------------------
    access_status: Indicates whether the current user has access to the RStudioServerPro app.
    user_group: The level of permissions that the user has within the RStudioServerPro app. This value defaults to `User`. The `Admin` value allows the user access to the RStudio Administrative Dashboard.
    """

    access_status: Optional[StrPipeVar] = Unassigned()
    user_group: Optional[StrPipeVar] = Unassigned()


class RSessionAppSettings(Base):
    """
    RSessionAppSettings
      A collection of settings that apply to an RSessionGateway app.

    Attributes
    ----------------------
    default_resource_spec
    custom_images: A list of custom SageMaker AI images that are configured to run as a RSession app.
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    custom_images: Optional[List[CustomImage]] = Unassigned()


class VSCodeAppSettings(Base):
    """
    VSCodeAppSettings

    Attributes
    ----------------------
    default_resource_spec
    custom_images
    lifecycle_config_arns
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    custom_images: Optional[List[CustomImage]] = Unassigned()
    lifecycle_config_arns: Optional[List[StrPipeVar]] = Unassigned()


class SaviturAppSettings(Base):
    """
    SaviturAppSettings

    Attributes
    ----------------------
    default_resource_spec
    custom_images
    lifecycle_config_arns
    code_repositories
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    custom_images: Optional[List[CustomImage]] = Unassigned()
    lifecycle_config_arns: Optional[List[StrPipeVar]] = Unassigned()
    code_repositories: Optional[List[CodeRepository]] = Unassigned()


class EmrSettings(Base):
    """
    EmrSettings
      The configuration parameters that specify the IAM roles assumed by the execution role of SageMaker (assumable roles) and the cluster instances or job execution environments (execution roles or runtime roles) to manage and access resources required for running Amazon EMR clusters or Amazon EMR Serverless applications.

    Attributes
    ----------------------
    assumable_role_arns: An array of Amazon Resource Names (ARNs) of the IAM roles that the execution role of SageMaker can assume for performing operations or tasks related to Amazon EMR clusters or Amazon EMR Serverless applications. These roles define the permissions and access policies required when performing Amazon EMR-related operations, such as listing, connecting to, or terminating Amazon EMR clusters or Amazon EMR Serverless applications. They are typically used in cross-account access scenarios, where the Amazon EMR resources (clusters or serverless applications) are located in a different Amazon Web Services account than the SageMaker domain.
    execution_role_arns: An array of Amazon Resource Names (ARNs) of the IAM roles used by the Amazon EMR cluster instances or job execution environments to access other Amazon Web Services services and resources needed during the runtime of your Amazon EMR or Amazon EMR Serverless workloads, such as Amazon S3 for data access, Amazon CloudWatch for logging, or other Amazon Web Services services based on the particular workload requirements.
    """

    assumable_role_arns: Optional[List[StrPipeVar]] = Unassigned()
    execution_role_arns: Optional[List[StrPipeVar]] = Unassigned()


class JupyterLabAppSettings(Base):
    """
    JupyterLabAppSettings
      The settings for the JupyterLab application.

    Attributes
    ----------------------
    default_resource_spec
    custom_images: A list of custom SageMaker images that are configured to run as a JupyterLab app.
    lifecycle_config_arns: The Amazon Resource Name (ARN) of the lifecycle configurations attached to the user profile or domain. To remove a lifecycle config, you must set LifecycleConfigArns to an empty list.
    code_repositories: A list of Git repositories that SageMaker automatically displays to users for cloning in the JupyterLab application.
    app_lifecycle_management: Indicates whether idle shutdown is activated for JupyterLab applications.
    emr_settings: The configuration parameters that specify the IAM roles assumed by the execution role of SageMaker (assumable roles) and the cluster instances or job execution environments (execution roles or runtime roles) to manage and access resources required for running Amazon EMR clusters or Amazon EMR Serverless applications.
    built_in_lifecycle_config_arn: The lifecycle configuration that runs before the default lifecycle configuration. It can override changes made in the default lifecycle configuration.
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    custom_images: Optional[List[CustomImage]] = Unassigned()
    lifecycle_config_arns: Optional[List[StrPipeVar]] = Unassigned()
    code_repositories: Optional[List[CodeRepository]] = Unassigned()
    app_lifecycle_management: Optional[AppLifecycleManagement] = Unassigned()
    emr_settings: Optional[EmrSettings] = Unassigned()
    built_in_lifecycle_config_arn: Optional[StrPipeVar] = Unassigned()


class DefaultEbsStorageSettings(Base):
    """
    DefaultEbsStorageSettings
      A collection of default EBS storage settings that apply to spaces created within a domain or user profile.

    Attributes
    ----------------------
    default_ebs_volume_size_in_gb: The default size of the EBS storage volume for a space.
    maximum_ebs_volume_size_in_gb: The maximum size of the EBS storage volume for a space.
    """

    default_ebs_volume_size_in_gb: int
    maximum_ebs_volume_size_in_gb: int


class DefaultSpaceStorageSettings(Base):
    """
    DefaultSpaceStorageSettings
      The default storage settings for a space.

    Attributes
    ----------------------
    default_ebs_storage_settings: The default EBS storage settings for a space.
    """

    default_ebs_storage_settings: Optional[DefaultEbsStorageSettings] = Unassigned()


class CustomPosixUserConfig(Base):
    """
    CustomPosixUserConfig
      Details about the POSIX identity that is used for file system operations.

    Attributes
    ----------------------
    uid: The POSIX user ID.
    gid: The POSIX group ID.
    """

    uid: int
    gid: int


class EFSFileSystemConfig(Base):
    """
    EFSFileSystemConfig
      The settings for assigning a custom Amazon EFS file system to a user profile or space for an Amazon SageMaker AI Domain.

    Attributes
    ----------------------
    file_system_id: The ID of your Amazon EFS file system.
    file_system_path: The path to the file system directory that is accessible in Amazon SageMaker AI Studio. Permitted users can access only this directory and below.
    """

    file_system_id: StrPipeVar
    file_system_path: Optional[StrPipeVar] = Unassigned()


class FSxLustreFileSystemConfig(Base):
    """
    FSxLustreFileSystemConfig
      The settings for assigning a custom Amazon FSx for Lustre file system to a user profile or space for an Amazon SageMaker Domain.

    Attributes
    ----------------------
    file_system_id: The globally unique, 17-digit, ID of the file system, assigned by Amazon FSx for Lustre.
    file_system_path: The path to the file system directory that is accessible in Amazon SageMaker Studio. Permitted users can access only this directory and below.
    """

    file_system_id: StrPipeVar
    file_system_path: Optional[StrPipeVar] = Unassigned()


class S3FileSystemConfig(Base):
    """
    S3FileSystemConfig
      Configuration for the custom Amazon S3 file system.

    Attributes
    ----------------------
    mount_path: The file system path where the Amazon S3 storage location will be mounted within the Amazon SageMaker Studio environment.
    s3_uri: The Amazon S3 URI of the S3 file system configuration.
    """

    s3_uri: StrPipeVar
    mount_path: Optional[StrPipeVar] = Unassigned()


class CustomFileSystemConfig(Base):
    """
    CustomFileSystemConfig
      The settings for assigning a custom file system to a user profile or space for an Amazon SageMaker AI Domain. Permitted users can access this file system in Amazon SageMaker AI Studio.

    Attributes
    ----------------------
    efs_file_system_config: The settings for a custom Amazon EFS file system.
    f_sx_lustre_file_system_config: The settings for a custom Amazon FSx for Lustre file system.
    s3_file_system_config: Configuration settings for a custom Amazon S3 file system.
    """

    efs_file_system_config: Optional[EFSFileSystemConfig] = Unassigned()
    f_sx_lustre_file_system_config: Optional[FSxLustreFileSystemConfig] = Unassigned()
    s3_file_system_config: Optional[S3FileSystemConfig] = Unassigned()


class HiddenSageMakerImage(Base):
    """
    HiddenSageMakerImage
      The SageMaker images that are hidden from the Studio user interface. You must specify the SageMaker image name and version aliases.

    Attributes
    ----------------------
    sage_maker_image_name:  The SageMaker image name that you are hiding from the Studio user interface.
    version_aliases:  The version aliases you are hiding from the Studio user interface.
    """

    sage_maker_image_name: Optional[StrPipeVar] = Unassigned()
    version_aliases: Optional[List[StrPipeVar]] = Unassigned()


class StudioWebPortalSettings(Base):
    """
    StudioWebPortalSettings
      Studio settings. If these settings are applied on a user level, they take priority over the settings applied on a domain level.

    Attributes
    ----------------------
    hidden_ml_tools: The machine learning tools that are hidden from the Studio left navigation pane.
    hidden_app_types: The Applications supported in Studio that are hidden from the Studio left navigation pane.
    hidden_instance_types:  The instance types you are hiding from the Studio user interface.
    hidden_sage_maker_image_version_aliases:  The version aliases you are hiding from the Studio user interface.
    """

    hidden_ml_tools: Optional[List[StrPipeVar]] = Unassigned()
    hidden_app_types: Optional[List[StrPipeVar]] = Unassigned()
    hidden_instance_types: Optional[List[StrPipeVar]] = Unassigned()
    hidden_sage_maker_image_version_aliases: Optional[List[HiddenSageMakerImage]] = Unassigned()


class UserSettings(Base):
    """
    UserSettings
      A collection of settings that apply to users in a domain. These settings are specified when the CreateUserProfile API is called, and as DefaultUserSettings when the CreateDomain API is called.  SecurityGroups is aggregated when specified in both calls. For all other settings in UserSettings, the values specified in CreateUserProfile take precedence over those specified in CreateDomain.

    Attributes
    ----------------------
    execution_role: The execution role for the user. SageMaker applies this setting only to private spaces that the user creates in the domain. SageMaker doesn't apply this setting to shared spaces.
    environment_settings: The environment settings.
    security_groups: The security groups for the Amazon Virtual Private Cloud (VPC) that the domain uses for communication. Optional when the CreateDomain.AppNetworkAccessType parameter is set to PublicInternetOnly. Required when the CreateDomain.AppNetworkAccessType parameter is set to VpcOnly, unless specified as part of the DefaultUserSettings for the domain. Amazon SageMaker AI adds a security group to allow NFS traffic from Amazon SageMaker AI Studio. Therefore, the number of security groups that you can specify is one less than the maximum number shown. SageMaker applies these settings only to private spaces that the user creates in the domain. SageMaker doesn't apply these settings to shared spaces.
    sharing_settings: Specifies options for sharing Amazon SageMaker AI Studio notebooks.
    jupyter_server_app_settings: The Jupyter server's app settings.
    kernel_gateway_app_settings: The kernel gateway app settings.
    tensor_board_app_settings: The TensorBoard app settings.
    r_studio_server_pro_app_settings: A collection of settings that configure user interaction with the RStudioServerPro app.
    r_session_app_settings: A collection of settings that configure the RSessionGateway app.
    canvas_app_settings: The Canvas app settings. SageMaker applies these settings only to private spaces that SageMaker creates for the Canvas app.
    vs_code_app_settings
    savitur_app_settings
    code_editor_app_settings: The Code Editor application settings. SageMaker applies these settings only to private spaces that the user creates in the domain. SageMaker doesn't apply these settings to shared spaces.
    jupyter_lab_app_settings: The settings for the JupyterLab application. SageMaker applies these settings only to private spaces that the user creates in the domain. SageMaker doesn't apply these settings to shared spaces.
    space_storage_settings: The storage settings for a space. SageMaker applies these settings only to private spaces that the user creates in the domain. SageMaker doesn't apply these settings to shared spaces.
    default_landing_uri: The default experience that the user is directed to when accessing the domain. The supported values are:    studio::: Indicates that Studio is the default experience. This value can only be passed if StudioWebPortal is set to ENABLED.    app:JupyterServer:: Indicates that Studio Classic is the default experience.
    studio_web_portal: Whether the user can access Studio. If this value is set to DISABLED, the user cannot access Studio, even if that is the default experience for the domain.
    custom_posix_user_config: Details about the POSIX identity that is used for file system operations. SageMaker applies these settings only to private spaces that the user creates in the domain. SageMaker doesn't apply these settings to shared spaces.
    custom_file_system_configs: The settings for assigning a custom file system to a user profile. Permitted users can access this file system in Amazon SageMaker AI Studio. SageMaker applies these settings only to private spaces that the user creates in the domain. SageMaker doesn't apply these settings to shared spaces.
    emr_settings
    studio_web_portal_settings: Studio settings. If these settings are applied on a user level, they take priority over the settings applied on a domain level.
    auto_mount_home_efs: Indicates whether auto-mounting of an EFS volume is supported for the user profile. The DefaultAsDomain value is only supported for user profiles. Do not use the DefaultAsDomain value when setting this parameter for a domain. SageMaker applies this setting only to private spaces that the user creates in the domain. SageMaker doesn't apply this setting to shared spaces.
    """

    execution_role: Optional[StrPipeVar] = Unassigned()
    environment_settings: Optional[EnvironmentSettings] = Unassigned()
    security_groups: Optional[List[StrPipeVar]] = Unassigned()
    sharing_settings: Optional[SharingSettings] = Unassigned()
    jupyter_server_app_settings: Optional[JupyterServerAppSettings] = Unassigned()
    kernel_gateway_app_settings: Optional[KernelGatewayAppSettings] = Unassigned()
    tensor_board_app_settings: Optional[TensorBoardAppSettings] = Unassigned()
    r_studio_server_pro_app_settings: Optional[RStudioServerProAppSettings] = Unassigned()
    r_session_app_settings: Optional[RSessionAppSettings] = Unassigned()
    canvas_app_settings: Optional[CanvasAppSettings] = Unassigned()
    vs_code_app_settings: Optional[VSCodeAppSettings] = Unassigned()
    savitur_app_settings: Optional[SaviturAppSettings] = Unassigned()
    code_editor_app_settings: Optional[CodeEditorAppSettings] = Unassigned()
    jupyter_lab_app_settings: Optional[JupyterLabAppSettings] = Unassigned()
    space_storage_settings: Optional[DefaultSpaceStorageSettings] = Unassigned()
    default_landing_uri: Optional[StrPipeVar] = Unassigned()
    studio_web_portal: Optional[StrPipeVar] = Unassigned()
    custom_posix_user_config: Optional[CustomPosixUserConfig] = Unassigned()
    custom_file_system_configs: Optional[List[CustomFileSystemConfig]] = Unassigned()
    emr_settings: Optional[EmrSettings] = Unassigned()
    studio_web_portal_settings: Optional[StudioWebPortalSettings] = Unassigned()
    auto_mount_home_efs: Optional[StrPipeVar] = Unassigned()


class RStudioServerProDomainSettings(Base):
    """
    RStudioServerProDomainSettings
      A collection of settings that configure the RStudioServerPro Domain-level app.

    Attributes
    ----------------------
    domain_execution_role_arn: The ARN of the execution role for the RStudioServerPro Domain-level app.
    r_studio_connect_url: A URL pointing to an RStudio Connect server.
    r_studio_package_manager_url: A URL pointing to an RStudio Package Manager server.
    default_resource_spec
    """

    domain_execution_role_arn: StrPipeVar
    r_studio_connect_url: Optional[StrPipeVar] = Unassigned()
    r_studio_package_manager_url: Optional[StrPipeVar] = Unassigned()
    default_resource_spec: Optional[ResourceSpec] = Unassigned()


class TrustedIdentityPropagationSettings(Base):
    """
    TrustedIdentityPropagationSettings
      The Trusted Identity Propagation (TIP) settings for the SageMaker domain. These settings determine how user identities from IAM Identity Center are propagated through the domain to TIP enabled Amazon Web Services services.

    Attributes
    ----------------------
    status: The status of Trusted Identity Propagation (TIP) at the SageMaker domain level.  When disabled, standard IAM role-based access is used.  When enabled:   User identities from IAM Identity Center are propagated through the application to TIP enabled Amazon Web Services services.   New applications or existing applications that are automatically patched, will use the domain level configuration.
    """

    status: StrPipeVar


class DockerSettings(Base):
    """
    DockerSettings
      A collection of settings that configure the domain's Docker interaction.

    Attributes
    ----------------------
    enable_docker_access: Indicates whether the domain can access Docker.
    vpc_only_trusted_accounts: The list of Amazon Web Services accounts that are trusted when the domain is created in VPC-only mode.
    rootless_docker: Indicates whether to use rootless Docker.
    """

    enable_docker_access: Optional[StrPipeVar] = Unassigned()
    vpc_only_trusted_accounts: Optional[List[StrPipeVar]] = Unassigned()
    rootless_docker: Optional[StrPipeVar] = Unassigned()


class UnifiedStudioSettings(Base):
    """
    UnifiedStudioSettings
      The settings that apply to an Amazon SageMaker AI domain when you use it in Amazon SageMaker Unified Studio.

    Attributes
    ----------------------
    studio_web_portal_access: Sets whether you can access the domain in Amazon SageMaker Studio:  ENABLED  You can access the domain in Amazon SageMaker Studio. If you migrate the domain to Amazon SageMaker Unified Studio, you can access it in both studio interfaces.  DISABLED  You can't access the domain in Amazon SageMaker Studio. If you migrate the domain to Amazon SageMaker Unified Studio, you can access it only in that studio interface.   To migrate a domain to Amazon SageMaker Unified Studio, you specify the UnifiedStudioSettings data type when you use the UpdateDomain action.
    domain_account_id: The ID of the Amazon Web Services account that has the Amazon SageMaker Unified Studio domain. The default value, if you don't specify an ID, is the ID of the account that has the Amazon SageMaker AI domain.
    domain_region: The Amazon Web Services Region where the domain is located in Amazon SageMaker Unified Studio. The default value, if you don't specify a Region, is the Region where the Amazon SageMaker AI domain is located.
    domain_id: The ID of the Amazon SageMaker Unified Studio domain associated with this domain.
    project_id: The ID of the Amazon SageMaker Unified Studio project that corresponds to the domain.
    environment_id: The ID of the environment that Amazon SageMaker Unified Studio associates with the domain.
    project_s3_path: The location where Amazon S3 stores temporary execution data and other artifacts for the project that corresponds to the domain.
    single_sign_on_application_arn: The ARN of the Amazon DataZone application managed by Amazon SageMaker Unified Studio in the Amazon Web Services IAM Identity Center.
    """

    studio_web_portal_access: Optional[StrPipeVar] = Unassigned()
    domain_account_id: Optional[StrPipeVar] = Unassigned()
    domain_region: Optional[StrPipeVar] = Unassigned()
    domain_id: Optional[StrPipeVar] = Unassigned()
    project_id: Optional[StrPipeVar] = Unassigned()
    environment_id: Optional[StrPipeVar] = Unassigned()
    project_s3_path: Optional[StrPipeVar] = Unassigned()
    single_sign_on_application_arn: Optional[StrPipeVar] = Unassigned()


class DomainSettings(Base):
    """
    DomainSettings
      A collection of settings that apply to the SageMaker Domain. These settings are specified through the CreateDomain API call.

    Attributes
    ----------------------
    security_group_ids: The security groups for the Amazon Virtual Private Cloud that the Domain uses for communication between Domain-level apps and user apps.
    logout_redirection_url
    r_studio_server_pro_domain_settings: A collection of settings that configure the RStudioServerPro Domain-level app.
    execution_role_identity_config: The configuration for attaching a SageMaker AI user profile name to the execution role as a sts:SourceIdentity key.
    trusted_identity_propagation_settings: The Trusted Identity Propagation (TIP) settings for the SageMaker domain. These settings determine how user identities from IAM Identity Center are propagated through the domain to TIP enabled Amazon Web Services services.
    docker_settings: A collection of settings that configure the domain's Docker interaction.
    amazon_q_settings: A collection of settings that configure the Amazon Q experience within the domain. The AuthMode that you use to create the domain must be SSO.
    unified_studio_settings: The settings that apply to an SageMaker AI domain when you use it in Amazon SageMaker Unified Studio.
    ip_address_type: The IP address type for the domain. Specify ipv4 for IPv4-only connectivity or dualstack for both IPv4 and IPv6 connectivity. When you specify dualstack, the subnet must support IPv6 CIDR blocks. If not specified, defaults to ipv4.
    """

    security_group_ids: Optional[List[StrPipeVar]] = Unassigned()
    logout_redirection_url: Optional[StrPipeVar] = Unassigned()
    r_studio_server_pro_domain_settings: Optional[RStudioServerProDomainSettings] = Unassigned()
    execution_role_identity_config: Optional[StrPipeVar] = Unassigned()
    trusted_identity_propagation_settings: Optional[TrustedIdentityPropagationSettings] = (
        Unassigned()
    )
    docker_settings: Optional[DockerSettings] = Unassigned()
    amazon_q_settings: Optional[AmazonQSettings] = Unassigned()
    unified_studio_settings: Optional[UnifiedStudioSettings] = Unassigned()
    ip_address_type: Optional[StrPipeVar] = Unassigned()


class DefaultSpaceSettings(Base):
    """
    DefaultSpaceSettings
      The default settings for shared spaces that users create in the domain. SageMaker applies these settings only to shared spaces. It doesn't apply them to private spaces.

    Attributes
    ----------------------
    execution_role: The ARN of the execution role for the space.
    security_groups: The security group IDs for the Amazon VPC that the space uses for communication.
    jupyter_server_app_settings
    kernel_gateway_app_settings
    jupyter_lab_app_settings
    space_storage_settings
    custom_posix_user_config
    custom_file_system_configs: The settings for assigning a custom file system to a domain. Permitted users can access this file system in Amazon SageMaker AI Studio.
    """

    execution_role: Optional[StrPipeVar] = Unassigned()
    security_groups: Optional[List[StrPipeVar]] = Unassigned()
    jupyter_server_app_settings: Optional[JupyterServerAppSettings] = Unassigned()
    kernel_gateway_app_settings: Optional[KernelGatewayAppSettings] = Unassigned()
    jupyter_lab_app_settings: Optional[JupyterLabAppSettings] = Unassigned()
    space_storage_settings: Optional[DefaultSpaceStorageSettings] = Unassigned()
    custom_posix_user_config: Optional[CustomPosixUserConfig] = Unassigned()
    custom_file_system_configs: Optional[List[CustomFileSystemConfig]] = Unassigned()


class EdgeDeploymentModelConfig(Base):
    """
    EdgeDeploymentModelConfig
      Contains information about the configuration of a model in a deployment.

    Attributes
    ----------------------
    model_handle: The name the device application uses to reference this model.
    edge_packaging_job_name: The edge packaging job associated with this deployment.
    """

    model_handle: StrPipeVar
    edge_packaging_job_name: Union[StrPipeVar, object]


class DeviceSelectionConfig(Base):
    """
    DeviceSelectionConfig
      Contains information about the configurations of selected devices.

    Attributes
    ----------------------
    device_subset_type: Type of device subsets to deploy to the current stage.
    percentage: Percentage of devices in the fleet to deploy to the current stage.
    device_names: List of devices chosen to deploy.
    device_name_contains: A filter to select devices with names containing this name.
    """

    device_subset_type: StrPipeVar
    percentage: Optional[int] = Unassigned()
    device_names: Optional[List[StrPipeVar]] = Unassigned()
    device_name_contains: Optional[StrPipeVar] = Unassigned()


class EdgeDeploymentConfig(Base):
    """
    EdgeDeploymentConfig
      Contains information about the configuration of a deployment.

    Attributes
    ----------------------
    failure_handling_policy: Toggle that determines whether to rollback to previous configuration if the current deployment fails. By default this is turned on. You may turn this off if you want to investigate the errors yourself.
    """

    failure_handling_policy: StrPipeVar


class DeploymentStage(Base):
    """
    DeploymentStage
      Contains information about a stage in an edge deployment plan.

    Attributes
    ----------------------
    stage_name: The name of the stage.
    device_selection_config: Configuration of the devices in the stage.
    deployment_config: Configuration of the deployment details.
    """

    stage_name: StrPipeVar
    device_selection_config: DeviceSelectionConfig
    deployment_config: Optional[EdgeDeploymentConfig] = Unassigned()


class ProductionVariantCoreDumpConfig(Base):
    """
    ProductionVariantCoreDumpConfig
      Specifies configuration for a core dump from the model container when the process crashes.

    Attributes
    ----------------------
    destination_s3_uri: The Amazon S3 bucket to send the core dump to.
    kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that SageMaker uses to encrypt the core dump data at rest using Amazon S3 server-side encryption. The KmsKeyId can be any of the following formats:    // KMS Key ID  "1234abcd-12ab-34cd-56ef-1234567890ab"    // Amazon Resource Name (ARN) of a KMS Key  "arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"    // KMS Key Alias  "alias/ExampleAlias"    // Amazon Resource Name (ARN) of a KMS Key Alias  "arn:aws:kms:us-west-2:111122223333:alias/ExampleAlias"    If you use a KMS key ID or an alias of your KMS key, the SageMaker execution role must include permissions to call kms:Encrypt. If you don't provide a KMS key ID, SageMaker uses the default KMS key for Amazon S3 for your role's account. SageMaker uses server-side encryption with KMS-managed keys for OutputDataConfig. If you use a bucket policy with an s3:PutObject permission that only allows objects with server-side encryption, set the condition key of s3:x-amz-server-side-encryption to "aws:kms". For more information, see KMS-Managed Encryption Keys in the Amazon Simple Storage Service Developer Guide.  The KMS key policy must grant permission to the IAM role that you specify in your CreateEndpoint and UpdateEndpoint requests. For more information, see Using Key Policies in Amazon Web Services KMS in the Amazon Web Services Key Management Service Developer Guide.
    """

    destination_s3_uri: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class ProductionVariantServerlessConfig(Base):
    """
    ProductionVariantServerlessConfig
      Specifies the serverless configuration for an endpoint variant.

    Attributes
    ----------------------
    memory_size_in_mb: The memory size of your serverless endpoint. Valid values are in 1 GB increments: 1024 MB, 2048 MB, 3072 MB, 4096 MB, 5120 MB, or 6144 MB.
    max_concurrency: The maximum number of concurrent invocations your serverless endpoint can process.
    provisioned_concurrency: The amount of provisioned concurrency to allocate for the serverless endpoint. Should be less than or equal to MaxConcurrency.  This field is not supported for serverless endpoint recommendations for Inference Recommender jobs. For more information about creating an Inference Recommender job, see CreateInferenceRecommendationsJobs.
    """

    memory_size_in_mb: int
    max_concurrency: int
    provisioned_concurrency: Optional[int] = Unassigned()


class ProductionVariantManagedInstanceScaling(Base):
    """
    ProductionVariantManagedInstanceScaling
      Settings that control the range in the number of instances that the endpoint provisions as it scales up or down to accommodate traffic.

    Attributes
    ----------------------
    status: Indicates whether managed instance scaling is enabled.
    min_instance_count: The minimum number of instances that the endpoint must retain when it scales down to accommodate a decrease in traffic.
    max_instance_count: The maximum number of instances that the endpoint can provision when it scales up to accommodate an increase in traffic.
    """

    status: Optional[StrPipeVar] = Unassigned()
    min_instance_count: Optional[int] = Unassigned()
    max_instance_count: Optional[int] = Unassigned()


class ProductionVariantRoutingConfig(Base):
    """
    ProductionVariantRoutingConfig
      Settings that control how the endpoint routes incoming traffic to the instances that the endpoint hosts.

    Attributes
    ----------------------
    routing_strategy: Sets how the endpoint routes incoming traffic:    LEAST_OUTSTANDING_REQUESTS: The endpoint routes requests to the specific instances that have more capacity to process them.    RANDOM: The endpoint routes each request to a randomly chosen instance.
    """

    routing_strategy: StrPipeVar


class ProductionVariantCapacitySchedulesConfig(Base):
    """
    ProductionVariantCapacitySchedulesConfig

    Attributes
    ----------------------
    capacity_fallback_strategy
    capacity_schedules
    """

    capacity_schedules: List[CapacitySchedule]
    capacity_fallback_strategy: Optional[StrPipeVar] = Unassigned()


class ProductionVariantHyperPodConfig(Base):
    """
    ProductionVariantHyperPodConfig

    Attributes
    ----------------------
    ingress_address
    """

    ingress_address: StrPipeVar


class ProductionVariantCapacityReservationConfig(Base):
    """
    ProductionVariantCapacityReservationConfig
      Settings for the capacity reservation for the compute instances that SageMaker AI reserves for an endpoint.

    Attributes
    ----------------------
    ec2_capacity_reservations
    capacity_reservation_preference: Options that you can choose for the capacity reservation. SageMaker AI supports the following options:  capacity-reservations-only  SageMaker AI launches instances only into an ML capacity reservation. If no capacity is available, the instances fail to launch.
    ml_reservation_arn: The Amazon Resource Name (ARN) that uniquely identifies the ML capacity reservation that SageMaker AI applies when it deploys the endpoint.
    """

    ec2_capacity_reservations: Optional[List[StrPipeVar]] = Unassigned()
    capacity_reservation_preference: Optional[StrPipeVar] = Unassigned()
    ml_reservation_arn: Optional[StrPipeVar] = Unassigned()


class ProductionVariant(Base):
    """
    ProductionVariant
       Identifies a model that you want to host and the resources chosen to deploy for hosting it. If you are deploying multiple models, tell SageMaker how to distribute traffic among the models by specifying variant weights. For more information on production variants, check  Production variants.

    Attributes
    ----------------------
    variant_name: The name of the production variant.
    model_name: The name of the model that you want to host. This is the name that you specified when creating the model.
    initial_instance_count: Number of instances to launch initially.
    instance_type: The ML compute instance type.
    initial_variant_weight: Determines initial traffic distribution among all of the models that you specify in the endpoint configuration. The traffic to a production variant is determined by the ratio of the VariantWeight to the sum of all VariantWeight values across all ProductionVariants. If unspecified, it defaults to 1.0.
    accelerator_type: This parameter is no longer supported. Elastic Inference (EI) is no longer available. This parameter was used to specify the size of the EI instance to use for the production variant.
    core_dump_config: Specifies configuration for a core dump from the model container when the process crashes.
    serverless_config: The serverless configuration for an endpoint. Specifies a serverless endpoint configuration instead of an instance-based endpoint configuration.
    volume_size_in_gb: The size, in GB, of the ML storage volume attached to individual inference instance associated with the production variant. Currently only Amazon EBS gp2 storage volumes are supported.
    model_data_download_timeout_in_seconds: The timeout value, in seconds, to download and extract the model that you want to host from Amazon S3 to the individual inference instance associated with this production variant.
    container_startup_health_check_timeout_in_seconds: The timeout value, in seconds, for your inference container to pass health check by SageMaker Hosting. For more information about health check, see How Your Container Should Respond to Health Check (Ping) Requests.
    enable_ssm_access:  You can use this parameter to turn on native Amazon Web Services Systems Manager (SSM) access for a production variant behind an endpoint. By default, SSM access is disabled for all production variants behind an endpoint. You can turn on or turn off SSM access for a production variant behind an existing endpoint by creating a new endpoint configuration and calling UpdateEndpoint.
    managed_instance_scaling: Settings that control the range in the number of instances that the endpoint provisions as it scales up or down to accommodate traffic.
    routing_config: Settings that control how the endpoint routes incoming traffic to the instances that the endpoint hosts.
    capacity_schedules_config
    inference_ami_version: Specifies an option from a collection of preconfigured Amazon Machine Image (AMI) images. Each image is configured by Amazon Web Services with a set of software and driver versions. Amazon Web Services optimizes these configurations for different machine learning workloads. By selecting an AMI version, you can ensure that your inference environment is compatible with specific software requirements, such as CUDA driver versions, Linux kernel versions, or Amazon Web Services Neuron driver versions. The AMI version names, and their configurations, are the following:  al2-ami-sagemaker-inference-gpu-2    Accelerator: GPU   NVIDIA driver version: 535   CUDA version: 12.2    al2-ami-sagemaker-inference-gpu-2-1    Accelerator: GPU   NVIDIA driver version: 535   CUDA version: 12.2   NVIDIA Container Toolkit with disabled CUDA-compat mounting    al2-ami-sagemaker-inference-gpu-3-1    Accelerator: GPU   NVIDIA driver version: 550   CUDA version: 12.4   NVIDIA Container Toolkit with disabled CUDA-compat mounting    al2-ami-sagemaker-inference-neuron-2    Accelerator: Inferentia2 and Trainium   Neuron driver version: 2.19
    hyper_pod_config
    capacity_reservation_config: Settings for the capacity reservation for the compute instances that SageMaker AI reserves for an endpoint.
    """

    variant_name: StrPipeVar
    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    initial_instance_count: Optional[int] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    initial_variant_weight: Optional[float] = Unassigned()
    accelerator_type: Optional[StrPipeVar] = Unassigned()
    core_dump_config: Optional[ProductionVariantCoreDumpConfig] = Unassigned()
    serverless_config: Optional[ProductionVariantServerlessConfig] = Unassigned()
    volume_size_in_gb: Optional[int] = Unassigned()
    model_data_download_timeout_in_seconds: Optional[int] = Unassigned()
    container_startup_health_check_timeout_in_seconds: Optional[int] = Unassigned()
    enable_ssm_access: Optional[bool] = Unassigned()
    managed_instance_scaling: Optional[ProductionVariantManagedInstanceScaling] = Unassigned()
    routing_config: Optional[ProductionVariantRoutingConfig] = Unassigned()
    capacity_schedules_config: Optional[ProductionVariantCapacitySchedulesConfig] = Unassigned()
    inference_ami_version: Optional[StrPipeVar] = Unassigned()
    hyper_pod_config: Optional[ProductionVariantHyperPodConfig] = Unassigned()
    capacity_reservation_config: Optional[ProductionVariantCapacityReservationConfig] = Unassigned()


class DataCaptureConfig(Base):
    """
    DataCaptureConfig
      Configuration to control how SageMaker AI captures inference data.

    Attributes
    ----------------------
    enable_capture: Whether data capture should be enabled or disabled (defaults to enabled).
    initial_sampling_percentage: The percentage of requests SageMaker AI will capture. A lower value is recommended for Endpoints with high traffic.
    destination_s3_uri: The Amazon S3 location used to capture the data.
    kms_key_id: The Amazon Resource Name (ARN) of an Key Management Service key that SageMaker AI uses to encrypt the captured data at rest using Amazon S3 server-side encryption. The KmsKeyId can be any of the following formats:    Key ID: 1234abcd-12ab-34cd-56ef-1234567890ab    Key ARN: arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab    Alias name: alias/ExampleAlias    Alias name ARN: arn:aws:kms:us-west-2:111122223333:alias/ExampleAlias
    capture_options: Specifies data Model Monitor will capture. You can configure whether to collect only input, only output, or both
    capture_content_type_header: Configuration specifying how to treat different headers. If no headers are specified SageMaker AI will by default base64 encode when capturing the data.
    """

    initial_sampling_percentage: int
    destination_s3_uri: StrPipeVar
    capture_options: List[CaptureOption]
    enable_capture: Optional[bool] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    capture_content_type_header: Optional[CaptureContentTypeHeader] = Unassigned()


class ExplainerConfig(Base):
    """
    ExplainerConfig
      A parameter to activate explainers.

    Attributes
    ----------------------
    clarify_explainer_config: A member of ExplainerConfig that contains configuration parameters for the SageMaker Clarify explainer.
    """

    clarify_explainer_config: Optional[ClarifyExplainerConfig] = Unassigned()


class MetricsConfig(Base):
    """
    MetricsConfig

    Attributes
    ----------------------
    enable_enhanced_metrics: Specifies whether to enable enhanced metrics for the endpoint. Enhanced metrics provide utilization data at instance and container granularity. Container granularity is supported for Inference Components. The default is False.
    metric_publish_frequency_in_seconds: The frequency, in seconds, at which Utilization Metrics are published to Amazon CloudWatch. The default is 60 seconds.
    """

    enable_enhanced_metrics: Optional[bool] = Unassigned()
    metric_publish_frequency_in_seconds: Optional[int] = Unassigned()


class EndpointDeletionCondition(Base):
    """
    EndpointDeletionCondition

    Attributes
    ----------------------
    max_runtime_in_seconds
    """

    max_runtime_in_seconds: int


class RollingUpdatePolicy(Base):
    """
    RollingUpdatePolicy
      Specifies a rolling deployment strategy for updating a SageMaker endpoint.

    Attributes
    ----------------------
    maximum_batch_size: Batch size for each rolling step to provision capacity and turn on traffic on the new endpoint fleet, and terminate capacity on the old endpoint fleet. Value must be between 5% to 50% of the variant's total instance count.
    wait_interval_in_seconds: The length of the baking period, during which SageMaker monitors alarms for each batch on the new fleet.
    maximum_execution_timeout_in_seconds: The time limit for the total deployment. Exceeding this limit causes a timeout.
    wait_for_instance_termination
    rollback_maximum_batch_size: Batch size for rollback to the old endpoint fleet. Each rolling step to provision capacity and turn on traffic on the old endpoint fleet, and terminate capacity on the new endpoint fleet. If this field is absent, the default value will be set to 100% of total capacity which means to bring up the whole capacity of the old fleet at once during rollback.
    """

    maximum_batch_size: CapacitySize
    wait_interval_in_seconds: int
    maximum_execution_timeout_in_seconds: Optional[int] = Unassigned()
    wait_for_instance_termination: Optional[bool] = Unassigned()
    rollback_maximum_batch_size: Optional[CapacitySize] = Unassigned()


class DeploymentConfig(Base):
    """
    DeploymentConfig
      The deployment configuration for an endpoint, which contains the desired deployment strategy and rollback configurations.

    Attributes
    ----------------------
    blue_green_update_policy: Update policy for a blue/green deployment. If this update policy is specified, SageMaker creates a new fleet during the deployment while maintaining the old fleet. SageMaker flips traffic to the new fleet according to the specified traffic routing configuration. Only one update policy should be used in the deployment configuration. If no update policy is specified, SageMaker uses a blue/green deployment strategy with all at once traffic shifting by default.
    rolling_update_policy: Specifies a rolling deployment strategy for updating a SageMaker endpoint.
    auto_rollback_configuration: Automatic rollback configuration for handling endpoint deployment failures and recovery.
    """

    blue_green_update_policy: Optional[BlueGreenUpdatePolicy] = Unassigned()
    rolling_update_policy: Optional[RollingUpdatePolicy] = Unassigned()
    auto_rollback_configuration: Optional[AutoRollbackConfig] = Unassigned()


class EvaluationJobModel(Base):
    """
    EvaluationJobModel

    Attributes
    ----------------------
    model_identifier
    model_type
    endpoint_arn
    """

    model_identifier: StrPipeVar
    model_type: StrPipeVar
    endpoint_arn: Optional[StrPipeVar] = Unassigned()


class EvaluationJobModelConfig(Base):
    """
    EvaluationJobModelConfig

    Attributes
    ----------------------
    models
    """

    models: List[EvaluationJobModel]


class EvaluationJobOutputDataConfig(Base):
    """
    EvaluationJobOutputDataConfig

    Attributes
    ----------------------
    s3_uri
    kms_key_id
    """

    s3_uri: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class EvaluationJobCustomDataset(Base):
    """
    EvaluationJobCustomDataset

    Attributes
    ----------------------
    dataset_name
    s3_uri
    """

    dataset_name: Optional[StrPipeVar] = Unassigned()
    s3_uri: Optional[StrPipeVar] = Unassigned()


class EvaluationJobInputDataConfig(Base):
    """
    EvaluationJobInputDataConfig

    Attributes
    ----------------------
    custom_datasets
    """

    custom_datasets: Optional[List[EvaluationJobCustomDataset]] = Unassigned()


class EvaluationJobHumanTaskConfig(Base):
    """
    EvaluationJobHumanTaskConfig

    Attributes
    ----------------------
    flow_definition_arn
    task_instructions
    """

    flow_definition_arn: StrPipeVar
    task_instructions: StrPipeVar


class EvaluationJobHumanWorkflowConfig(Base):
    """
    EvaluationJobHumanWorkflowConfig

    Attributes
    ----------------------
    flow_definition_arn
    task_instructions
    """

    flow_definition_arn: StrPipeVar
    task_instructions: StrPipeVar


class EvaluationJobHumanEvaluationMetric(Base):
    """
    EvaluationJobHumanEvaluationMetric

    Attributes
    ----------------------
    metric_name
    rating_method
    metric_type
    description
    """

    metric_name: StrPipeVar
    rating_method: Optional[StrPipeVar] = Unassigned()
    metric_type: Optional[StrPipeVar] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()


class EvaluationJobHumanEvaluationConfig(Base):
    """
    EvaluationJobHumanEvaluationConfig

    Attributes
    ----------------------
    human_task_config
    human_workflow_config
    human_evaluation_metrics
    """

    human_evaluation_metrics: List[EvaluationJobHumanEvaluationMetric]
    human_task_config: Optional[EvaluationJobHumanTaskConfig] = Unassigned()
    human_workflow_config: Optional[EvaluationJobHumanWorkflowConfig] = Unassigned()


class EvaluationJobEvaluationConfig(Base):
    """
    EvaluationJobEvaluationConfig

    Attributes
    ----------------------
    human_evaluation_config
    """

    human_evaluation_config: EvaluationJobHumanEvaluationConfig


class EvaluationJobCredentialProxyConfig(Base):
    """
    EvaluationJobCredentialProxyConfig

    Attributes
    ----------------------
    upstream_platform_customer_credential_token
    credential_provider_function
    """

    upstream_platform_customer_credential_token: StrPipeVar
    credential_provider_function: StrPipeVar


class EvaluationJobUpstreamPlatformCustomerOutputDataConfig(Base):
    """
    EvaluationJobUpstreamPlatformCustomerOutputDataConfig

    Attributes
    ----------------------
    kms_key_id
    s3_kms_encryption_context
    kms_encryption_context
    s3_uri
    """

    s3_uri: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    s3_kms_encryption_context: Optional[StrPipeVar] = Unassigned()
    kms_encryption_context: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class EvaluationJobUpstreamPlatformConfig(Base):
    """
    EvaluationJobUpstreamPlatformConfig

    Attributes
    ----------------------
    credential_proxy_config
    upstream_platform_customer_output_data_config
    upstream_platform_customer_account_id
    upstream_platform_customer_evaluation_job_arn
    upstream_platform_customer_execution_role
    """

    credential_proxy_config: EvaluationJobCredentialProxyConfig
    upstream_platform_customer_output_data_config: (
        EvaluationJobUpstreamPlatformCustomerOutputDataConfig
    )
    upstream_platform_customer_account_id: StrPipeVar
    upstream_platform_customer_execution_role: StrPipeVar
    upstream_platform_customer_evaluation_job_arn: Optional[StrPipeVar] = Unassigned()


class InputExperimentSource(Base):
    """
    InputExperimentSource

    Attributes
    ----------------------
    source_arn
    """

    source_arn: StrPipeVar


class FeatureDefinition(Base):
    """
    FeatureDefinition
      A list of features. You must include FeatureName and FeatureType. Valid feature FeatureTypes are Integral, Fractional and String.

    Attributes
    ----------------------
    feature_name: The name of a feature. The type must be a string. FeatureName cannot be any of the following: is_deleted, write_time, api_invocation_time. The name:   Must start with an alphanumeric character.   Can only include alphanumeric characters, underscores, and hyphens. Spaces are not allowed.
    feature_type: The value type of a feature. Valid values are Integral, Fractional, or String.
    collection_type: A grouping of elements where each element within the collection must have the same feature type (String, Integral, or Fractional).    List: An ordered collection of elements.    Set: An unordered collection of unique elements.    Vector: A specialized list that represents a fixed-size array of elements. The vector dimension is determined by you. Must have elements with fractional feature types.
    collection_config: Configuration for your collection.
    """

    feature_name: StrPipeVar
    feature_type: StrPipeVar
    collection_type: Optional[StrPipeVar] = Unassigned()
    collection_config: Optional[CollectionConfig] = Unassigned()


class OnlineStoreConfig(Base):
    """
    OnlineStoreConfig
      Use this to specify the Amazon Web Services Key Management Service (KMS) Key ID, or KMSKeyId, for at rest data encryption. You can turn OnlineStore on or off by specifying the EnableOnlineStore flag at General Assembly. The default value is False.

    Attributes
    ----------------------
    security_config: Use to specify KMS Key ID (KMSKeyId) for at-rest encryption of your OnlineStore.
    enable_online_store: Turn OnlineStore off by specifying False for the EnableOnlineStore flag. Turn OnlineStore on by specifying True for the EnableOnlineStore flag.  The default value is False.
    ttl_duration: Time to live duration, where the record is hard deleted after the expiration time is reached; ExpiresAt = EventTime + TtlDuration. For information on HardDelete, see the DeleteRecord API in the Amazon SageMaker API Reference guide.
    storage_type: Option for different tiers of low latency storage for real-time data retrieval.    Standard: A managed low latency data store for feature groups.    InMemory: A managed data store for feature groups that supports very low latency retrieval.
    """

    security_config: Optional[OnlineStoreSecurityConfig] = Unassigned()
    enable_online_store: Optional[bool] = Unassigned()
    ttl_duration: Optional[TtlDuration] = Unassigned()
    storage_type: Optional[StrPipeVar] = Unassigned()


class S3StorageConfig(Base):
    """
    S3StorageConfig
      The Amazon Simple Storage (Amazon S3) location and security configuration for OfflineStore.

    Attributes
    ----------------------
    s3_uri: The S3 URI, or location in Amazon S3, of OfflineStore. S3 URIs have a format similar to the following: s3://example-bucket/prefix/.
    kms_key_id: The Amazon Web Services Key Management Service (KMS) key ARN of the key used to encrypt any objects written into the OfflineStore S3 location. The IAM roleARN that is passed as a parameter to CreateFeatureGroup must have below permissions to the KmsKeyId:    "kms:GenerateDataKey"
    resolved_output_s3_uri: The S3 path where offline records are written.
    """

    s3_uri: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    resolved_output_s3_uri: Optional[StrPipeVar] = Unassigned()


class DataCatalogConfig(Base):
    """
    DataCatalogConfig
      The meta data of the Glue table which serves as data catalog for the OfflineStore.

    Attributes
    ----------------------
    table_name: The name of the Glue table.
    catalog: The name of the Glue table catalog.
    database: The name of the Glue table database.
    """

    table_name: StrPipeVar
    catalog: StrPipeVar
    database: StrPipeVar


class OfflineStoreConfig(Base):
    """
    OfflineStoreConfig
      The configuration of an OfflineStore. Provide an OfflineStoreConfig in a request to CreateFeatureGroup to create an OfflineStore. To encrypt an OfflineStore using at rest data encryption, specify Amazon Web Services Key Management Service (KMS) key ID, or KMSKeyId, in S3StorageConfig.

    Attributes
    ----------------------
    s3_storage_config: The Amazon Simple Storage (Amazon S3) location of OfflineStore.
    disable_glue_table_creation: Set to True to disable the automatic creation of an Amazon Web Services Glue table when configuring an OfflineStore. If set to False, Feature Store will name the OfflineStore Glue table following Athena's naming recommendations. The default value is False.
    data_catalog_config: The meta data of the Glue table that is autogenerated when an OfflineStore is created.
    table_format: Format for the offline store table. Supported formats are Glue (Default) and Apache Iceberg.
    """

    s3_storage_config: S3StorageConfig
    disable_glue_table_creation: Optional[bool] = Unassigned()
    data_catalog_config: Optional[DataCatalogConfig] = Unassigned()
    table_format: Optional[StrPipeVar] = Unassigned()


class OnlineStoreReplicaMetadata(Base):
    """
    OnlineStoreReplicaMetadata

    Attributes
    ----------------------
    source_region_name
    source_table_name
    source_feature_group_arn
    """

    source_region_name: StrPipeVar
    source_table_name: StrPipeVar
    source_feature_group_arn: StrPipeVar


class OnlineStoreMetadata(Base):
    """
    OnlineStoreMetadata

    Attributes
    ----------------------
    storage_account_id
    is_online_store_replica
    online_store_replica_metadata
    """

    storage_account_id: Optional[StrPipeVar] = Unassigned()
    is_online_store_replica: Optional[bool] = Unassigned()
    online_store_replica_metadata: Optional[OnlineStoreReplicaMetadata] = Unassigned()


class ThroughputConfig(Base):
    """
    ThroughputConfig
      Used to set feature group throughput configuration. There are two modes: ON_DEMAND and PROVISIONED. With on-demand mode, you are charged for data reads and writes that your application performs on your feature group. You do not need to specify read and write throughput because Feature Store accommodates your workloads as they ramp up and down. You can switch a feature group to on-demand only once in a 24 hour period. With provisioned throughput mode, you specify the read and write capacity per second that you expect your application to require, and you are billed based on those limits. Exceeding provisioned throughput will result in your requests being throttled.  Note: PROVISIONED throughput mode is supported only for feature groups that are offline-only, or use the  Standard  tier online store.

    Attributes
    ----------------------
    throughput_mode: The mode used for your feature group throughput: ON_DEMAND or PROVISIONED.
    provisioned_read_capacity_units:  For provisioned feature groups with online store enabled, this indicates the read throughput you are billed for and can consume without throttling.  This field is not applicable for on-demand feature groups.
    provisioned_write_capacity_units:  For provisioned feature groups, this indicates the write throughput you are billed for and can consume without throttling.  This field is not applicable for on-demand feature groups.
    """

    throughput_mode: StrPipeVar
    provisioned_read_capacity_units: Optional[int] = Unassigned()
    provisioned_write_capacity_units: Optional[int] = Unassigned()


class HumanLoopRequestSource(Base):
    """
    HumanLoopRequestSource
      Container for configuring the source of human task requests.

    Attributes
    ----------------------
    aws_managed_human_loop_request_source: Specifies whether Amazon Rekognition or Amazon Textract are used as the integration source. The default field settings and JSON parsing rules are different based on the integration source. Valid values:
    """

    aws_managed_human_loop_request_source: StrPipeVar


class HumanLoopActivationConditionsConfig(Base):
    """
    HumanLoopActivationConditionsConfig
      Defines under what conditions SageMaker creates a human loop. Used within CreateFlowDefinition. See HumanLoopActivationConditionsConfig for the required format of activation conditions.

    Attributes
    ----------------------
    human_loop_activation_conditions: JSON expressing use-case specific conditions declaratively. If any condition is matched, atomic tasks are created against the configured work team. The set of conditions is different for Rekognition and Textract. For more information about how to structure the JSON, see JSON Schema for Human Loop Activation Conditions in Amazon Augmented AI in the Amazon SageMaker Developer Guide.
    """

    human_loop_activation_conditions: StrPipeVar


class HumanLoopActivationConfig(Base):
    """
    HumanLoopActivationConfig
      Provides information about how and under what conditions SageMaker creates a human loop. If HumanLoopActivationConfig is not given, then all requests go to humans.

    Attributes
    ----------------------
    human_loop_request_source
    human_loop_activation_conditions_config: Container structure for defining under what conditions SageMaker creates a human loop.
    """

    human_loop_activation_conditions_config: HumanLoopActivationConditionsConfig
    human_loop_request_source: Optional[HumanLoopRequestSource] = Unassigned()


class USD(Base):
    """
    USD
      Represents an amount of money in United States dollars.

    Attributes
    ----------------------
    dollars: The whole number of dollars in the amount.
    cents: The fractional portion, in cents, of the amount.
    tenth_fractions_of_a_cent: Fractions of a cent, in tenths.
    """

    dollars: Optional[int] = Unassigned()
    cents: Optional[int] = Unassigned()
    tenth_fractions_of_a_cent: Optional[int] = Unassigned()


class PublicWorkforceTaskPrice(Base):
    """
    PublicWorkforceTaskPrice
      Defines the amount of money paid to an Amazon Mechanical Turk worker for each task performed.  Use one of the following prices for bounding box tasks. Prices are in US dollars and should be based on the complexity of the task; the longer it takes in your initial testing, the more you should offer.   0.036   0.048   0.060   0.072   0.120   0.240   0.360   0.480   0.600   0.720   0.840   0.960   1.080   1.200   Use one of the following prices for image classification, text classification, and custom tasks. Prices are in US dollars.   0.012   0.024   0.036   0.048   0.060   0.072   0.120   0.240   0.360   0.480   0.600   0.720   0.840   0.960   1.080   1.200   Use one of the following prices for semantic segmentation tasks. Prices are in US dollars.   0.840   0.960   1.080   1.200   Use one of the following prices for Textract AnalyzeDocument Important Form Key Amazon Augmented AI review tasks. Prices are in US dollars.   2.400    2.280    2.160    2.040    1.920    1.800    1.680    1.560    1.440    1.320    1.200    1.080    0.960    0.840    0.720    0.600    0.480    0.360    0.240    0.120    0.072    0.060    0.048    0.036    0.024    0.012    Use one of the following prices for Rekognition DetectModerationLabels Amazon Augmented AI review tasks. Prices are in US dollars.   1.200    1.080    0.960    0.840    0.720    0.600    0.480    0.360    0.240    0.120    0.072    0.060    0.048    0.036    0.024    0.012    Use one of the following prices for Amazon Augmented AI custom human review tasks. Prices are in US dollars.   1.200    1.080    0.960    0.840    0.720    0.600    0.480    0.360    0.240    0.120    0.072    0.060    0.048    0.036    0.024    0.012

    Attributes
    ----------------------
    amount_in_usd: Defines the amount of money paid to an Amazon Mechanical Turk worker in United States dollars.
    """

    amount_in_usd: Optional[USD] = Unassigned()


class HumanLoopConfig(Base):
    """
    HumanLoopConfig
      Describes the work to be performed by human workers.

    Attributes
    ----------------------
    workteam_arn: Amazon Resource Name (ARN) of a team of workers. To learn more about the types of workforces and work teams you can create and use with Amazon A2I, see Create and Manage Workforces.
    human_task_ui_arn: The Amazon Resource Name (ARN) of the human task user interface. You can use standard HTML and Crowd HTML Elements to create a custom worker task template. You use this template to create a human task UI. To learn how to create a custom HTML template, see Create Custom Worker Task Template. To learn how to create a human task UI, which is a worker task template that can be used in a flow definition, see Create and Delete a Worker Task Templates.
    task_title: A title for the human worker task.
    task_description: A description for the human worker task.
    task_count: The number of distinct workers who will perform the same task on each object. For example, if TaskCount is set to 3 for an image classification labeling job, three workers will classify each input image. Increasing TaskCount can improve label accuracy.
    task_availability_lifetime_in_seconds: The length of time that a task remains available for review by human workers.
    task_time_limit_in_seconds: The amount of time that a worker has to complete a task. The default value is 3,600 seconds (1 hour).
    task_keywords: Keywords used to describe the task so that workers can discover the task.
    public_workforce_task_price
    """

    workteam_arn: StrPipeVar
    human_task_ui_arn: StrPipeVar
    task_title: StrPipeVar
    task_description: StrPipeVar
    task_count: int
    task_availability_lifetime_in_seconds: Optional[int] = Unassigned()
    task_time_limit_in_seconds: Optional[int] = Unassigned()
    task_keywords: Optional[List[StrPipeVar]] = Unassigned()
    public_workforce_task_price: Optional[PublicWorkforceTaskPrice] = Unassigned()


class FlowDefinitionOutputConfig(Base):
    """
    FlowDefinitionOutputConfig
      Contains information about where human output will be stored.

    Attributes
    ----------------------
    s3_output_path: The Amazon S3 path where the object containing human output will be made available. To learn more about the format of Amazon A2I output data, see Amazon A2I Output Data.
    kms_key_id: The Amazon Key Management Service (KMS) key ID for server-side encryption.
    """

    s3_output_path: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class GroundTruthJobDataAttributes(Base):
    """
    GroundTruthJobDataAttributes

    Attributes
    ----------------------
    content_classifiers
    """

    content_classifiers: Optional[List[StrPipeVar]] = Unassigned()


class GroundTruthJobS3DataSource(Base):
    """
    GroundTruthJobS3DataSource

    Attributes
    ----------------------
    s3_uri
    """

    s3_uri: Optional[StrPipeVar] = Unassigned()


class GroundTruthJobDataSource(Base):
    """
    GroundTruthJobDataSource

    Attributes
    ----------------------
    s3_data_source
    """

    s3_data_source: Optional[GroundTruthJobS3DataSource] = Unassigned()


class GroundTruthJobInputConfig(Base):
    """
    GroundTruthJobInputConfig

    Attributes
    ----------------------
    data_attributes
    data_source
    """

    data_attributes: Optional[GroundTruthJobDataAttributes] = Unassigned()
    data_source: Optional[GroundTruthJobDataSource] = Unassigned()


class GroundTruthJobOutputConfig(Base):
    """
    GroundTruthJobOutputConfig

    Attributes
    ----------------------
    s3_output_path
    """

    s3_output_path: Optional[StrPipeVar] = Unassigned()


class GroundTruthProjectPointOfContact(Base):
    """
    GroundTruthProjectPointOfContact

    Attributes
    ----------------------
    name
    email
    """

    name: StrPipeVar
    email: StrPipeVar


class PresignedUrlAccessConfig(Base):
    """
    PresignedUrlAccessConfig
      Configuration for accessing hub content through presigned URLs, including license agreement acceptance and URL validation settings.

    Attributes
    ----------------------
    accept_eula: Indicates acceptance of the End User License Agreement (EULA) for gated models. Set to true to acknowledge acceptance of the license terms required for accessing gated content.
    expected_s3_url: The expected S3 URL prefix for validation purposes. This parameter helps ensure consistency between the resolved S3 URIs and the deployment configuration, reducing potential compatibility issues.
    """

    accept_eula: Optional[bool] = Unassigned()
    expected_s3_url: Optional[StrPipeVar] = Unassigned()


class HubS3StorageConfig(Base):
    """
    HubS3StorageConfig
      The Amazon S3 storage configuration of a hub.

    Attributes
    ----------------------
    s3_output_path: The Amazon S3 bucket prefix for hosting hub content.
    """

    s3_output_path: Optional[StrPipeVar] = Unassigned()


class UiTemplate(Base):
    """
    UiTemplate
      The Liquid template for the worker user interface.

    Attributes
    ----------------------
    content: The content of the Liquid template for the worker user interface.
    """

    content: StrPipeVar


class HyperbandStrategyConfig(Base):
    """
    HyperbandStrategyConfig
      The configuration for Hyperband, a multi-fidelity based hyperparameter tuning strategy. Hyperband uses the final and intermediate results of a training job to dynamically allocate resources to utilized hyperparameter configurations while automatically stopping under-performing configurations. This parameter should be provided only if Hyperband is selected as the StrategyConfig under the HyperParameterTuningJobConfig API.

    Attributes
    ----------------------
    number_of_brackets
    reduction_factor
    variant
    min_resource: The minimum number of resources (such as epochs) that can be used by a training job launched by a hyperparameter tuning job. If the value for MinResource has not been reached, the training job is not stopped by Hyperband.
    max_resource: The maximum number of resources (such as epochs) that can be used by a training job launched by a hyperparameter tuning job. Once a job reaches the MaxResource value, it is stopped. If a value for MaxResource is not provided, and Hyperband is selected as the hyperparameter tuning strategy, HyperbandTraining attempts to infer MaxResource from the following keys (if present) in StaticsHyperParameters:    epochs     numepochs     n-epochs     n_epochs     num_epochs    If HyperbandStrategyConfig is unable to infer a value for MaxResource, it generates a validation error. The maximum value is 20,000 epochs. All metrics that correspond to an objective metric are used to derive early stopping decisions. For distributed training jobs, ensure that duplicate metrics are not printed in the logs across the individual nodes in a training job. If multiple nodes are publishing duplicate or incorrect metrics, training jobs may make an incorrect stopping decision and stop the job prematurely.
    """

    number_of_brackets: Optional[int] = Unassigned()
    reduction_factor: Optional[int] = Unassigned()
    variant: Optional[StrPipeVar] = Unassigned()
    min_resource: Optional[int] = Unassigned()
    max_resource: Optional[int] = Unassigned()


class HyperParameterTuningJobStrategyConfig(Base):
    """
    HyperParameterTuningJobStrategyConfig
      The configuration for a training job launched by a hyperparameter tuning job. Choose Bayesian for Bayesian optimization, and Random for random search optimization. For more advanced use cases, use Hyperband, which evaluates objective metrics for training jobs after every epoch. For more information about strategies, see How Hyperparameter Tuning Works.

    Attributes
    ----------------------
    hyperband_strategy_config: The configuration for the object that specifies the Hyperband strategy. This parameter is only supported for the Hyperband selection for Strategy within the HyperParameterTuningJobConfig API.
    """

    hyperband_strategy_config: Optional[HyperbandStrategyConfig] = Unassigned()


class ResourceLimits(Base):
    """
    ResourceLimits
      Specifies the maximum number of training jobs and parallel training jobs that a hyperparameter tuning job can launch.

    Attributes
    ----------------------
    max_number_of_training_jobs: The maximum number of training jobs that a hyperparameter tuning job can launch.
    max_parallel_training_jobs: The maximum number of concurrent training jobs that a hyperparameter tuning job can launch.
    max_wall_clock_time_in_minutes
    max_total_compute_time_in_minutes
    max_runtime_in_seconds: The maximum time in seconds that a hyperparameter tuning job can run.
    max_billable_time_in_seconds
    """

    max_parallel_training_jobs: int
    max_number_of_training_jobs: Optional[int] = Unassigned()
    max_wall_clock_time_in_minutes: Optional[int] = Unassigned()
    max_total_compute_time_in_minutes: Optional[int] = Unassigned()
    max_runtime_in_seconds: Optional[int] = Unassigned()
    max_billable_time_in_seconds: Optional[int] = Unassigned()


class IntegerParameterRange(Base):
    """
    IntegerParameterRange
      For a hyperparameter of the integer type, specifies the range that a hyperparameter tuning job searches.

    Attributes
    ----------------------
    name: The name of the hyperparameter to search.
    min_value: The minimum value of the hyperparameter to search.
    max_value: The maximum value of the hyperparameter to search.
    scaling_type: The scale that hyperparameter tuning uses to search the hyperparameter range. For information about choosing a hyperparameter scale, see Hyperparameter Scaling. One of the following values:  Auto  SageMaker hyperparameter tuning chooses the best scale for the hyperparameter.  Linear  Hyperparameter tuning searches the values in the hyperparameter range by using a linear scale.  Logarithmic  Hyperparameter tuning searches the values in the hyperparameter range by using a logarithmic scale. Logarithmic scaling works only for ranges that have only values greater than 0.
    """

    name: StrPipeVar
    min_value: StrPipeVar
    max_value: StrPipeVar
    scaling_type: Optional[StrPipeVar] = Unassigned()


class ParameterRanges(Base):
    """
    ParameterRanges
      Specifies ranges of integer, continuous, and categorical hyperparameters that a hyperparameter tuning job searches. The hyperparameter tuning job launches training jobs with hyperparameter values within these ranges to find the combination of values that result in the training job with the best performance as measured by the objective metric of the hyperparameter tuning job.  The maximum number of items specified for Array Members refers to the maximum number of hyperparameters for each range and also the maximum for the hyperparameter tuning job itself. That is, the sum of the number of hyperparameters for all the ranges can't exceed the maximum number specified.

    Attributes
    ----------------------
    integer_parameter_ranges: The array of IntegerParameterRange objects that specify ranges of integer hyperparameters that a hyperparameter tuning job searches.
    continuous_parameter_ranges: The array of ContinuousParameterRange objects that specify ranges of continuous hyperparameters that a hyperparameter tuning job searches.
    categorical_parameter_ranges: The array of CategoricalParameterRange objects that specify ranges of categorical hyperparameters that a hyperparameter tuning job searches.
    auto_parameters: A list containing hyperparameter names and example values to be used by Autotune to determine optimal ranges for your tuning job.
    """

    integer_parameter_ranges: Optional[List[IntegerParameterRange]] = Unassigned()
    continuous_parameter_ranges: Optional[List[ContinuousParameterRange]] = Unassigned()
    categorical_parameter_ranges: Optional[List[CategoricalParameterRange]] = Unassigned()
    auto_parameters: Optional[List[AutoParameter]] = Unassigned()


class HyperParameterTrainingJobInstancePool(Base):
    """
    HyperParameterTrainingJobInstancePool

    Attributes
    ----------------------
    instance_type
    pool_size
    """

    instance_type: StrPipeVar
    pool_size: int


class TuningJobCompletionCriteria(Base):
    """
    TuningJobCompletionCriteria
      The job completion criteria.

    Attributes
    ----------------------
    target_objective_metric_value: The value of the objective metric.
    best_objective_not_improving: A flag to stop your hyperparameter tuning job if model performance fails to improve as evaluated against an objective function.
    convergence_detected: A flag to top your hyperparameter tuning job if automatic model tuning (AMT) has detected that your model has converged as evaluated against your objective function.
    """

    target_objective_metric_value: Optional[float] = Unassigned()
    best_objective_not_improving: Optional[BestObjectiveNotImproving] = Unassigned()
    convergence_detected: Optional[ConvergenceDetected] = Unassigned()


class HyperParameterTuningJobCompletionConfig(Base):
    """
    HyperParameterTuningJobCompletionConfig

    Attributes
    ----------------------
    in_progress_training_jobs_handling
    """

    in_progress_training_jobs_handling: Optional[StrPipeVar] = Unassigned()


class HyperParameterTuningJobConfig(Base):
    """
    HyperParameterTuningJobConfig
      Configures a hyperparameter tuning job.

    Attributes
    ----------------------
    strategy: Specifies how hyperparameter tuning chooses the combinations of hyperparameter values to use for the training job it launches. For information about search strategies, see How Hyperparameter Tuning Works.
    strategy_config: The configuration for the Hyperband optimization strategy. This parameter should be provided only if Hyperband is selected as the strategy for HyperParameterTuningJobConfig.
    hyper_parameter_tuning_job_objective: The HyperParameterTuningJobObjective specifies the objective metric used to evaluate the performance of training jobs launched by this tuning job.
    resource_limits: The ResourceLimits object that specifies the maximum number of training and parallel training jobs that can be used for this hyperparameter tuning job.
    parameter_ranges: The ParameterRanges object that specifies the ranges of hyperparameters that this tuning job searches over to find the optimal configuration for the highest model performance against your chosen objective metric.
    training_job_early_stopping_type: Specifies whether to use early stopping for training jobs launched by the hyperparameter tuning job. Because the Hyperband strategy has its own advanced internal early stopping mechanism, TrainingJobEarlyStoppingType must be OFF to use Hyperband. This parameter can take on one of the following values (the default value is OFF):  OFF  Training jobs launched by the hyperparameter tuning job do not use early stopping.  AUTO  SageMaker stops training jobs launched by the hyperparameter tuning job when they are unlikely to perform better than previously completed training jobs. For more information, see Stop Training Jobs Early.
    training_job_instance_pools
    tuning_job_completion_criteria: The tuning job's completion criteria.
    completion_config
    random_seed: A value used to initialize a pseudo-random number generator. Setting a random seed and using the same seed later for the same tuning job will allow hyperparameter optimization to find more a consistent hyperparameter configuration between the two runs.
    """

    strategy: StrPipeVar
    resource_limits: ResourceLimits
    strategy_config: Optional[HyperParameterTuningJobStrategyConfig] = Unassigned()
    hyper_parameter_tuning_job_objective: Optional[HyperParameterTuningJobObjective] = Unassigned()
    parameter_ranges: Optional[ParameterRanges] = Unassigned()
    training_job_early_stopping_type: Optional[StrPipeVar] = Unassigned()
    training_job_instance_pools: Optional[List[HyperParameterTrainingJobInstancePool]] = (
        Unassigned()
    )
    tuning_job_completion_criteria: Optional[TuningJobCompletionCriteria] = Unassigned()
    completion_config: Optional[HyperParameterTuningJobCompletionConfig] = Unassigned()
    random_seed: Optional[int] = Unassigned()


class HyperParameterAlgorithmSpecification(Base):
    """
    HyperParameterAlgorithmSpecification
      Specifies which training algorithm to use for training jobs that a hyperparameter tuning job launches and the metrics to monitor.

    Attributes
    ----------------------
    training_image:  The registry path of the Docker image that contains the training algorithm. For information about Docker registry paths for built-in algorithms, see Algorithms Provided by Amazon SageMaker: Common Parameters. SageMaker supports both registry/repository[:tag] and registry/repository[@digest] image path formats. For more information, see Using Your Own Algorithms with Amazon SageMaker.
    training_input_mode
    algorithm_name: The name of the resource algorithm to use for the hyperparameter tuning job. If you specify a value for this parameter, do not specify a value for TrainingImage.
    metric_definitions: An array of MetricDefinition objects that specify the metrics that the algorithm emits.
    """

    training_input_mode: StrPipeVar
    training_image: Optional[StrPipeVar] = Unassigned()
    algorithm_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    metric_definitions: Optional[List[MetricDefinition]] = Unassigned()


class HyperParameterTuningInstanceGroup(Base):
    """
    HyperParameterTuningInstanceGroup

    Attributes
    ----------------------
    instance_type
    instance_count
    instance_group_name
    """

    instance_type: StrPipeVar
    instance_count: int
    instance_group_name: StrPipeVar


class HyperParameterTuningInstanceConfig(Base):
    """
    HyperParameterTuningInstanceConfig
      The configuration for hyperparameter tuning resources for use in training jobs launched by the tuning job. These resources include compute instances and storage volumes. Specify one or more compute instance configurations and allocation strategies to select resources (optional).

    Attributes
    ----------------------
    instance_type: The instance type used for processing of hyperparameter optimization jobs. Choose from general purpose (no GPUs) instance types: ml.m5.xlarge, ml.m5.2xlarge, and ml.m5.4xlarge or compute optimized (no GPUs) instance types: ml.c5.xlarge and ml.c5.2xlarge. For more information about instance types, see instance type descriptions.
    instance_count: The number of instances of the type specified by InstanceType. Choose an instance count larger than 1 for distributed training algorithms. See Step 2: Launch a SageMaker Distributed Training Job Using the SageMaker Python SDK for more information.
    volume_size_in_gb: The volume size in GB of the data to be processed for hyperparameter optimization (optional).
    """

    instance_type: StrPipeVar
    instance_count: int
    volume_size_in_gb: int


class HyperParameterTuningResourceConfig(Base):
    """
    HyperParameterTuningResourceConfig
      The configuration of resources, including compute instances and storage volumes for use in training jobs launched by hyperparameter tuning jobs. HyperParameterTuningResourceConfig is similar to ResourceConfig, but has the additional InstanceConfigs and AllocationStrategy fields to allow for flexible instance management. Specify one or more instance types, count, and the allocation strategy for instance selection.   HyperParameterTuningResourceConfig supports the capabilities of ResourceConfig with the exception of KeepAlivePeriodInSeconds. Hyperparameter tuning jobs use warm pools by default, which reuse clusters between training jobs.

    Attributes
    ----------------------
    instance_type: The instance type used to run hyperparameter optimization tuning jobs. See  descriptions of instance types for more information.
    instance_count: The number of compute instances of type InstanceType to use. For distributed training, select a value greater than 1.
    volume_size_in_gb: The volume size in GB for the storage volume to be used in processing hyperparameter optimization jobs (optional). These volumes store model artifacts, incremental states and optionally, scratch space for training algorithms. Do not provide a value for this parameter if a value for InstanceConfigs is also specified. Some instance types have a fixed total local storage size. If you select one of these instances for training, VolumeSizeInGB cannot be greater than this total size. For a list of instance types with local instance storage and their sizes, see instance store volumes.  SageMaker supports only the General Purpose SSD (gp2) storage volume type.
    volume_kms_key_id: A key used by Amazon Web Services Key Management Service to encrypt data on the storage volume attached to the compute instances used to run the training job. You can use either of the following formats to specify a key. KMS Key ID:  "1234abcd-12ab-34cd-56ef-1234567890ab"  Amazon Resource Name (ARN) of a KMS key:  "arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"  Some instances use local storage, which use a hardware module to encrypt storage volumes. If you choose one of these instance types, you cannot request a VolumeKmsKeyId. For a list of instance types that use local storage, see instance store volumes. For more information about Amazon Web Services Key Management Service, see KMS encryption for more information.
    instance_groups
    allocation_strategy: The strategy that determines the order of preference for resources specified in InstanceConfigs used in hyperparameter optimization.
    instance_configs: A list containing the configuration(s) for one or more resources for processing hyperparameter jobs. These resources include compute instances and storage volumes to use in model training jobs launched by hyperparameter tuning jobs. The AllocationStrategy controls the order in which multiple configurations provided in InstanceConfigs are used.  If you only want to use a single instance configuration inside the HyperParameterTuningResourceConfig API, do not provide a value for InstanceConfigs. Instead, use InstanceType, VolumeSizeInGB and InstanceCount. If you use InstanceConfigs, do not provide values for InstanceType, VolumeSizeInGB or InstanceCount.
    """

    instance_type: Optional[StrPipeVar] = Unassigned()
    instance_count: Optional[int] = Unassigned()
    volume_size_in_gb: Optional[int] = Unassigned()
    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()
    instance_groups: Optional[List[HyperParameterTuningInstanceGroup]] = Unassigned()
    allocation_strategy: Optional[StrPipeVar] = Unassigned()
    instance_configs: Optional[List[HyperParameterTuningInstanceConfig]] = Unassigned()


class RetryStrategy(Base):
    """
    RetryStrategy
      The retry strategy to use when a training job fails due to an InternalServerError. RetryStrategy is specified as part of the CreateTrainingJob and CreateHyperParameterTuningJob requests. You can add the StoppingCondition parameter to the request to limit the training time for the complete job.

    Attributes
    ----------------------
    maximum_retry_attempts: The number of times to retry the job. When the job is retried, it's SecondaryStatus is changed to STARTING.
    """

    maximum_retry_attempts: int


class HyperParameterTrainingJobDefinition(Base):
    """
    HyperParameterTrainingJobDefinition
      Defines the training jobs launched by a hyperparameter tuning job.

    Attributes
    ----------------------
    definition_name: The job definition name.
    tuning_objective
    hyper_parameter_ranges
    static_hyper_parameters: Specifies the values of hyperparameters that do not change for the tuning job.
    initial_hyper_parameter_configurations
    algorithm_specification: The HyperParameterAlgorithmSpecification object that specifies the resource algorithm to use for the training jobs that the tuning job launches.
    role_arn: The Amazon Resource Name (ARN) of the IAM role associated with the training jobs that the tuning job launches.
    input_data_config: An array of Channel objects that specify the input for the training jobs that the tuning job launches.
    vpc_config: The VpcConfig object that specifies the VPC that you want the training jobs that this hyperparameter tuning job launches to connect to. Control access to and from your training container by configuring the VPC. For more information, see Protect Training Jobs by Using an Amazon Virtual Private Cloud.
    output_data_config: Specifies the path to the Amazon S3 bucket where you store model artifacts from the training jobs that the tuning job launches.
    resource_config: The resources, including the compute instances and storage volumes, to use for the training jobs that the tuning job launches. Storage volumes store model artifacts and incremental states. Training algorithms might also use storage volumes for scratch space. If you want SageMaker to use the storage volume to store the training data, choose File as the TrainingInputMode in the algorithm specification. For distributed training algorithms, specify an instance count greater than 1.  If you want to use hyperparameter optimization with instance type flexibility, use HyperParameterTuningResourceConfig instead.
    hyper_parameter_tuning_resource_config: The configuration for the hyperparameter tuning resources, including the compute instances and storage volumes, used for training jobs launched by the tuning job. By default, storage volumes hold model artifacts and incremental states. Choose File for TrainingInputMode in the AlgorithmSpecification parameter to additionally store training data in the storage volume (optional).
    stopping_condition: Specifies a limit to how long a model hyperparameter training job can run. It also specifies how long a managed spot training job has to complete. When the job reaches the time limit, SageMaker ends the training job. Use this API to cap model training costs.
    enable_network_isolation: Isolates the training container. No inbound or outbound network calls can be made, except for calls between peers within a training cluster for distributed training. If network isolation is used for training jobs that are configured to use a VPC, SageMaker downloads and uploads customer data and model artifacts through the specified VPC, but the training container does not have network access.
    enable_inter_container_traffic_encryption: To encrypt all communications between ML compute instances in distributed training, choose True. Encryption provides greater security for distributed training, but training might take longer. How long it takes depends on the amount of communication between compute instances, especially if you use a deep learning algorithm in distributed training.
    enable_managed_spot_training: A Boolean indicating whether managed spot training is enabled (True) or not (False).
    checkpoint_config
    retry_strategy: The number of times to retry the job when the job fails due to an InternalServerError.
    environment: An environment variable that you can pass into the SageMaker CreateTrainingJob API. You can use an existing environment variable from the training container or use your own. See Define metrics and variables for more information.  The maximum number of items specified for Map Entries refers to the maximum number of environment variables for each TrainingJobDefinition and also the maximum for the hyperparameter tuning job itself. That is, the sum of the number of environment variables for all the training job definitions can't exceed the maximum number specified.
    """

    algorithm_specification: HyperParameterAlgorithmSpecification
    role_arn: StrPipeVar
    output_data_config: OutputDataConfig
    stopping_condition: StoppingCondition
    definition_name: Optional[StrPipeVar] = Unassigned()
    tuning_objective: Optional[HyperParameterTuningJobObjective] = Unassigned()
    hyper_parameter_ranges: Optional[ParameterRanges] = Unassigned()
    static_hyper_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    initial_hyper_parameter_configurations: Optional[List[Dict[StrPipeVar, StrPipeVar]]] = (
        Unassigned()
    )
    input_data_config: Optional[List[Channel]] = Unassigned()
    vpc_config: Optional[VpcConfig] = Unassigned()
    resource_config: Optional[ResourceConfig] = Unassigned()
    hyper_parameter_tuning_resource_config: Optional[HyperParameterTuningResourceConfig] = (
        Unassigned()
    )
    enable_network_isolation: Optional[bool] = Unassigned()
    enable_inter_container_traffic_encryption: Optional[bool] = Unassigned()
    enable_managed_spot_training: Optional[bool] = Unassigned()
    checkpoint_config: Optional[CheckpointConfig] = Unassigned()
    retry_strategy: Optional[RetryStrategy] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class ParentHyperParameterTuningJob(Base):
    """
    ParentHyperParameterTuningJob
      A previously completed or stopped hyperparameter tuning job to be used as a starting point for a new hyperparameter tuning job.

    Attributes
    ----------------------
    hyper_parameter_tuning_job_name: The name of the hyperparameter tuning job to be used as a starting point for a new hyperparameter tuning job.
    """

    hyper_parameter_tuning_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()


class HyperParameterTuningJobWarmStartConfig(Base):
    """
    HyperParameterTuningJobWarmStartConfig
      Specifies the configuration for a hyperparameter tuning job that uses one or more previous hyperparameter tuning jobs as a starting point. The results of previous tuning jobs are used to inform which combinations of hyperparameters to search over in the new tuning job. All training jobs launched by the new hyperparameter tuning job are evaluated by using the objective metric, and the training job that performs the best is compared to the best training jobs from the parent tuning jobs. From these, the training job that performs the best as measured by the objective metric is returned as the overall best training job.  All training jobs launched by parent hyperparameter tuning jobs and the new hyperparameter tuning jobs count against the limit of training jobs for the tuning job.

    Attributes
    ----------------------
    parent_hyper_parameter_tuning_jobs: An array of hyperparameter tuning jobs that are used as the starting point for the new hyperparameter tuning job. For more information about warm starting a hyperparameter tuning job, see Using a Previous Hyperparameter Tuning Job as a Starting Point. Hyperparameter tuning jobs created before October 1, 2018 cannot be used as parent jobs for warm start tuning jobs.
    warm_start_type: Specifies one of the following:  IDENTICAL_DATA_AND_ALGORITHM  The new hyperparameter tuning job uses the same input data and training image as the parent tuning jobs. You can change the hyperparameter ranges to search and the maximum number of training jobs that the hyperparameter tuning job launches. You cannot use a new version of the training algorithm, unless the changes in the new version do not affect the algorithm itself. For example, changes that improve logging or adding support for a different data format are allowed. You can also change hyperparameters from tunable to static, and from static to tunable, but the total number of static plus tunable hyperparameters must remain the same as it is in all parent jobs. The objective metric for the new tuning job must be the same as for all parent jobs.  TRANSFER_LEARNING  The new hyperparameter tuning job can include input data, hyperparameter ranges, maximum number of concurrent training jobs, and maximum number of training jobs that are different than those of its parent hyperparameter tuning jobs. The training image can also be a different version from the version used in the parent hyperparameter tuning job. You can also change hyperparameters from tunable to static, and from static to tunable, but the total number of static plus tunable hyperparameters must remain the same as it is in all parent jobs. The objective metric for the new tuning job must be the same as for all parent jobs.
    """

    parent_hyper_parameter_tuning_jobs: List[ParentHyperParameterTuningJob]
    warm_start_type: StrPipeVar


class IdentityCenterUserToken(Base):
    """
    IdentityCenterUserToken

    Attributes
    ----------------------
    encrypted_refresh_token
    client_id
    idc_user_id
    skip_revoke_token_after_complete
    """

    encrypted_refresh_token: StrPipeVar
    client_id: StrPipeVar
    idc_user_id: StrPipeVar
    skip_revoke_token_after_complete: Optional[bool] = Unassigned()


class InferenceComponentContainerSpecification(Base):
    """
    InferenceComponentContainerSpecification
      Defines a container that provides the runtime environment for a model that you deploy with an inference component.

    Attributes
    ----------------------
    image: The Amazon Elastic Container Registry (Amazon ECR) path where the Docker image for the model is stored.
    artifact_url: The Amazon S3 path where the model artifacts, which result from model training, are stored. This path must point to a single gzip compressed tar archive (.tar.gz suffix).
    environment: The environment variables to set in the Docker container. Each key and value in the Environment string-to-string map can have length of up to 1024. We support up to 16 entries in the map.
    """

    image: Optional[StrPipeVar] = Unassigned()
    artifact_url: Optional[StrPipeVar] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class InferenceComponentStartupParameters(Base):
    """
    InferenceComponentStartupParameters
      Settings that take effect while the model container starts up.

    Attributes
    ----------------------
    model_data_download_timeout_in_seconds: The timeout value, in seconds, to download and extract the model that you want to host from Amazon S3 to the individual inference instance associated with this inference component.
    container_startup_health_check_timeout_in_seconds: The timeout value, in seconds, for your inference container to pass health check by Amazon S3 Hosting. For more information about health check, see How Your Container Should Respond to Health Check (Ping) Requests.
    """

    model_data_download_timeout_in_seconds: Optional[int] = Unassigned()
    container_startup_health_check_timeout_in_seconds: Optional[int] = Unassigned()


class InferenceComponentComputeResourceRequirements(Base):
    """
    InferenceComponentComputeResourceRequirements
      Defines the compute resources to allocate to run a model, plus any adapter models, that you assign to an inference component. These resources include CPU cores, accelerators, and memory.

    Attributes
    ----------------------
    number_of_cpu_cores_required: The number of CPU cores to allocate to run a model that you assign to an inference component.
    number_of_accelerator_devices_required: The number of accelerators to allocate to run a model that you assign to an inference component. Accelerators include GPUs and Amazon Web Services Inferentia.
    min_memory_required_in_mb: The minimum MB of memory to allocate to run a model that you assign to an inference component.
    max_memory_required_in_mb: The maximum MB of memory to allocate to run a model that you assign to an inference component.
    """

    min_memory_required_in_mb: int
    number_of_cpu_cores_required: Optional[float] = Unassigned()
    number_of_accelerator_devices_required: Optional[float] = Unassigned()
    max_memory_required_in_mb: Optional[int] = Unassigned()


class InferenceComponentDataCacheConfig(Base):
    """
    InferenceComponentDataCacheConfig
      Settings that affect how the inference component caches data.

    Attributes
    ----------------------
    enable_caching: Sets whether the endpoint that hosts the inference component caches the model artifacts and container image. With caching enabled, the endpoint caches this data in each instance that it provisions for the inference component. That way, the inference component deploys faster during the auto scaling process. If caching isn't enabled, the inference component takes longer to deploy because of the time it spends downloading the data.
    """

    enable_caching: bool


class InferenceComponentSpecification(Base):
    """
    InferenceComponentSpecification
      Details about the resources to deploy with this inference component, including the model, container, and compute resources.

    Attributes
    ----------------------
    model_name: The name of an existing SageMaker AI model object in your account that you want to deploy with the inference component.
    container: Defines a container that provides the runtime environment for a model that you deploy with an inference component.
    startup_parameters: Settings that take effect while the model container starts up.
    compute_resource_requirements: The compute resources allocated to run the model, plus any adapter models, that you assign to the inference component. Omit this parameter if your request is meant to create an adapter inference component. An adapter inference component is loaded by a base inference component, and it uses the compute resources of the base inference component.
    base_inference_component_name: The name of an existing inference component that is to contain the inference component that you're creating with your request. Specify this parameter only if your request is meant to create an adapter inference component. An adapter inference component contains the path to an adapter model. The purpose of the adapter model is to tailor the inference output of a base foundation model, which is hosted by the base inference component. The adapter inference component uses the compute resources that you assigned to the base inference component. When you create an adapter inference component, use the Container parameter to specify the location of the adapter artifacts. In the parameter value, use the ArtifactUrl parameter of the InferenceComponentContainerSpecification data type. Before you can create an adapter inference component, you must have an existing inference component that contains the foundation model that you want to adapt.
    data_cache_config: Settings that affect how the inference component caches data.
    """

    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    container: Optional[InferenceComponentContainerSpecification] = Unassigned()
    startup_parameters: Optional[InferenceComponentStartupParameters] = Unassigned()
    compute_resource_requirements: Optional[InferenceComponentComputeResourceRequirements] = (
        Unassigned()
    )
    base_inference_component_name: Optional[StrPipeVar] = Unassigned()
    data_cache_config: Optional[InferenceComponentDataCacheConfig] = Unassigned()


class InferenceComponentRuntimeConfig(Base):
    """
    InferenceComponentRuntimeConfig
      Runtime settings for a model that is deployed with an inference component.

    Attributes
    ----------------------
    copy_count: The number of runtime copies of the model container to deploy with the inference component. Each copy can serve inference requests.
    """

    copy_count: int


class InferenceExperimentSchedule(Base):
    """
    InferenceExperimentSchedule
      The start and end times of an inference experiment. The maximum duration that you can set for an inference experiment is 30 days.

    Attributes
    ----------------------
    start_time: The timestamp at which the inference experiment started or will start.
    end_time: The timestamp at which the inference experiment ended or will end.
    """

    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()


class RealTimeInferenceConfig(Base):
    """
    RealTimeInferenceConfig
      The infrastructure configuration for deploying the model to a real-time inference endpoint.

    Attributes
    ----------------------
    instance_type: The instance type the model is deployed to.
    instance_count: The number of instances of the type specified by InstanceType.
    """

    instance_type: StrPipeVar
    instance_count: int


class ModelInfrastructureConfig(Base):
    """
    ModelInfrastructureConfig
      The configuration for the infrastructure that the model will be deployed to.

    Attributes
    ----------------------
    infrastructure_type: The inference option to which to deploy your model. Possible values are the following:    RealTime: Deploy to real-time inference.
    real_time_inference_config: The infrastructure configuration for deploying the model to real-time inference.
    """

    infrastructure_type: StrPipeVar
    real_time_inference_config: RealTimeInferenceConfig


class ModelVariantConfig(Base):
    """
    ModelVariantConfig
      Contains information about the deployment options of a model.

    Attributes
    ----------------------
    model_name: The name of the Amazon SageMaker Model entity.
    variant_name: The name of the variant.
    infrastructure_config: The configuration for the infrastructure that the model will be deployed to.
    """

    model_name: Union[StrPipeVar, object]
    variant_name: StrPipeVar
    infrastructure_config: ModelInfrastructureConfig


class InferenceExperimentDataStorageConfig(Base):
    """
    InferenceExperimentDataStorageConfig
      The Amazon S3 location and configuration for storing inference request and response data.

    Attributes
    ----------------------
    destination: The Amazon S3 bucket where the inference request and response data is stored.
    kms_key:  The Amazon Web Services Key Management Service key that Amazon SageMaker uses to encrypt captured data at rest using Amazon S3 server-side encryption.
    content_type
    """

    destination: StrPipeVar
    kms_key: Optional[StrPipeVar] = Unassigned()
    content_type: Optional[CaptureContentTypeHeader] = Unassigned()


class ShadowModelVariantConfig(Base):
    """
    ShadowModelVariantConfig
      The name and sampling percentage of a shadow variant.

    Attributes
    ----------------------
    shadow_model_variant_name: The name of the shadow variant.
    sampling_percentage:  The percentage of inference requests that Amazon SageMaker replicates from the production variant to the shadow variant.
    """

    shadow_model_variant_name: StrPipeVar
    sampling_percentage: int


class ShadowModeConfig(Base):
    """
    ShadowModeConfig
       The configuration of ShadowMode inference experiment type, which specifies a production variant to take all the inference requests, and a shadow variant to which Amazon SageMaker replicates a percentage of the inference requests. For the shadow variant it also specifies the percentage of requests that Amazon SageMaker replicates.

    Attributes
    ----------------------
    source_model_variant_name:  The name of the production variant, which takes all the inference requests.
    shadow_model_variants: List of shadow variant configurations.
    """

    source_model_variant_name: StrPipeVar
    shadow_model_variants: List[ShadowModelVariantConfig]


class Phase(Base):
    """
    Phase
      Defines the traffic pattern.

    Attributes
    ----------------------
    initial_number_of_users: Specifies how many concurrent users to start with. The value should be between 1 and 3.
    spawn_rate: Specified how many new users to spawn in a minute.
    duration_in_seconds: Specifies how long a traffic phase should be. For custom load tests, the value should be between 120 and 3600. This value should not exceed JobDurationInSeconds.
    """

    initial_number_of_users: Optional[int] = Unassigned()
    spawn_rate: Optional[int] = Unassigned()
    duration_in_seconds: Optional[int] = Unassigned()


class Stairs(Base):
    """
    Stairs
      Defines the stairs traffic pattern for an Inference Recommender load test. This pattern type consists of multiple steps where the number of users increases at each step. Specify either the stairs or phases traffic pattern.

    Attributes
    ----------------------
    duration_in_seconds: Defines how long each traffic step should be.
    number_of_steps: Specifies how many steps to perform during traffic.
    users_per_step: Specifies how many new users to spawn in each step.
    """

    duration_in_seconds: Optional[int] = Unassigned()
    number_of_steps: Optional[int] = Unassigned()
    users_per_step: Optional[int] = Unassigned()


class InferenceInvocationTypes(Base):
    """
    InferenceInvocationTypes

    Attributes
    ----------------------
    invocation_type
    """

    invocation_type: Optional[StrPipeVar] = Unassigned()


class PayloadSampling(Base):
    """
    PayloadSampling

    Attributes
    ----------------------
    sampling_type
    sampling_seed
    """

    sampling_type: Optional[StrPipeVar] = Unassigned()
    sampling_seed: Optional[int] = Unassigned()


class TrafficPattern(Base):
    """
    TrafficPattern
      Defines the traffic pattern of the load test.

    Attributes
    ----------------------
    traffic_type: Defines the traffic patterns. Choose either PHASES or STAIRS.
    phases: Defines the phases traffic specification.
    stairs: Defines the stairs traffic pattern.
    concurrencies
    inference_invocation_types
    payload_sampling
    """

    traffic_type: Optional[StrPipeVar] = Unassigned()
    phases: Optional[List[Phase]] = Unassigned()
    stairs: Optional[Stairs] = Unassigned()
    concurrencies: Optional[List[Concurrency]] = Unassigned()
    inference_invocation_types: Optional[InferenceInvocationTypes] = Unassigned()
    payload_sampling: Optional[PayloadSampling] = Unassigned()


class RecommendationJobResourceLimit(Base):
    """
    RecommendationJobResourceLimit
      Specifies the maximum number of jobs that can run in parallel and the maximum number of jobs that can run.

    Attributes
    ----------------------
    max_number_of_tests: Defines the maximum number of load tests.
    max_parallel_of_tests: Defines the maximum number of parallel load tests.
    """

    max_number_of_tests: Optional[int] = Unassigned()
    max_parallel_of_tests: Optional[int] = Unassigned()


class IntegerParameter(Base):
    """
    IntegerParameter

    Attributes
    ----------------------
    name
    min_value
    max_value
    scaling_type
    """

    name: Optional[StrPipeVar] = Unassigned()
    min_value: Optional[int] = Unassigned()
    max_value: Optional[int] = Unassigned()
    scaling_type: Optional[StrPipeVar] = Unassigned()


class EnvironmentParameterRanges(Base):
    """
    EnvironmentParameterRanges
      Specifies the range of environment parameters

    Attributes
    ----------------------
    categorical_parameter_ranges: Specified a list of parameters for each category.
    integer_parameter_ranges
    continuous_parameter_ranges
    """

    categorical_parameter_ranges: Optional[List[CategoricalParameter]] = Unassigned()
    integer_parameter_ranges: Optional[List[IntegerParameter]] = Unassigned()
    continuous_parameter_ranges: Optional[List[ContinuousParameter]] = Unassigned()


class EndpointInputConfiguration(Base):
    """
    EndpointInputConfiguration
      The endpoint configuration for the load test.

    Attributes
    ----------------------
    instance_type: The instance types to use for the load test.
    serverless_config
    inference_specification_name: The inference specification name in the model package version.
    environment_parameter_ranges:  The parameter you want to benchmark against.
    """

    instance_type: Optional[StrPipeVar] = Unassigned()
    serverless_config: Optional[ProductionVariantServerlessConfig] = Unassigned()
    inference_specification_name: Optional[StrPipeVar] = Unassigned()
    environment_parameter_ranges: Optional[EnvironmentParameterRanges] = Unassigned()


class RecommendationJobPayloadConfig(Base):
    """
    RecommendationJobPayloadConfig
      The configuration for the payload for a recommendation job.

    Attributes
    ----------------------
    sample_payload_url: The Amazon Simple Storage Service (Amazon S3) path where the sample payload is stored. This path must point to a single gzip compressed tar archive (.tar.gz suffix).
    supported_content_types: The supported MIME types for the input data.
    """

    sample_payload_url: Optional[StrPipeVar] = Unassigned()
    supported_content_types: Optional[List[StrPipeVar]] = Unassigned()


class RecommendationJobContainerConfig(Base):
    """
    RecommendationJobContainerConfig
      Specifies mandatory fields for running an Inference Recommender job directly in the CreateInferenceRecommendationsJob API. The fields specified in ContainerConfig override the corresponding fields in the model package. Use ContainerConfig if you want to specify these fields for the recommendation job but don't want to edit them in your model package.

    Attributes
    ----------------------
    domain: The machine learning domain of the model and its components. Valid Values: COMPUTER_VISION \| NATURAL_LANGUAGE_PROCESSING \| MACHINE_LEARNING
    task: The machine learning task that the model accomplishes. Valid Values: IMAGE_CLASSIFICATION \| OBJECT_DETECTION \| TEXT_GENERATION \| IMAGE_SEGMENTATION \| FILL_MASK \| CLASSIFICATION \| REGRESSION \| OTHER
    framework: The machine learning framework of the container image. Valid Values: TENSORFLOW \| PYTORCH \| XGBOOST \| SAGEMAKER-SCIKIT-LEARN
    framework_version: The framework version of the container image.
    payload_config: Specifies the SamplePayloadUrl and all other sample payload-related fields.
    nearest_model_name: The name of a pre-trained machine learning model benchmarked by Amazon SageMaker Inference Recommender that matches your model. Valid Values: efficientnetb7 \| unet \| xgboost \| faster-rcnn-resnet101 \| nasnetlarge \| vgg16 \| inception-v3 \| mask-rcnn \| sagemaker-scikit-learn \| densenet201-gluon \| resnet18v2-gluon \| xception \| densenet201 \| yolov4 \| resnet152 \| bert-base-cased \| xceptionV1-keras \| resnet50 \| retinanet
    supported_instance_types: A list of the instance types that are used to generate inferences in real-time.
    supported_endpoint_type: The endpoint type to receive recommendations for. By default this is null, and the results of the inference recommendation job return a combined list of both real-time and serverless benchmarks. By specifying a value for this field, you can receive a longer list of benchmarks for the desired endpoint type.
    data_input_config: Specifies the name and shape of the expected data inputs for your trained model with a JSON dictionary form. This field is used for optimizing your model using SageMaker Neo. For more information, see DataInputConfig.
    supported_response_mime_types: The supported MIME types for the output data.
    """

    domain: Optional[StrPipeVar] = Unassigned()
    task: Optional[StrPipeVar] = Unassigned()
    framework: Optional[StrPipeVar] = Unassigned()
    framework_version: Optional[StrPipeVar] = Unassigned()
    payload_config: Optional[RecommendationJobPayloadConfig] = Unassigned()
    nearest_model_name: Optional[StrPipeVar] = Unassigned()
    supported_instance_types: Optional[List[StrPipeVar]] = Unassigned()
    supported_endpoint_type: Optional[StrPipeVar] = Unassigned()
    data_input_config: Optional[StrPipeVar] = Unassigned()
    supported_response_mime_types: Optional[List[StrPipeVar]] = Unassigned()


class EndpointInfo(Base):
    """
    EndpointInfo
      Details about a customer endpoint that was compared in an Inference Recommender job.

    Attributes
    ----------------------
    endpoint_name: The name of a customer's endpoint.
    """

    endpoint_name: Optional[Union[StrPipeVar, object]] = Unassigned()


class RecommendationJobVpcConfig(Base):
    """
    RecommendationJobVpcConfig
      Inference Recommender provisions SageMaker endpoints with access to VPC in the inference recommendation job.

    Attributes
    ----------------------
    security_group_ids: The VPC security group IDs. IDs have the form of sg-xxxxxxxx. Specify the security groups for the VPC that is specified in the Subnets field.
    subnets: The ID of the subnets in the VPC to which you want to connect your model.
    """

    security_group_ids: List[StrPipeVar]
    subnets: List[StrPipeVar]


class TokenizerConfig(Base):
    """
    TokenizerConfig

    Attributes
    ----------------------
    model_id
    accept_eula
    """

    model_id: Optional[StrPipeVar] = Unassigned()
    accept_eula: Optional[bool] = Unassigned()


class RecommendationJobInputConfig(Base):
    """
    RecommendationJobInputConfig
      The input configuration of the recommendation job.

    Attributes
    ----------------------
    model_package_version_arn: The Amazon Resource Name (ARN) of a versioned model package.
    model_name: The name of the created model.
    job_duration_in_seconds: Specifies the maximum duration of the job, in seconds. The maximum value is 18,000 seconds.
    traffic_pattern: Specifies the traffic pattern of the job.
    resource_limit: Defines the resource limit of the job.
    endpoint_configurations: Specifies the endpoint configuration to use for a job.
    volume_kms_key_id: The Amazon Resource Name (ARN) of a Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt data on the storage volume attached to the ML compute instance that hosts the endpoint. This key will be passed to SageMaker Hosting for endpoint creation.  The SageMaker execution role must have kms:CreateGrant permission in order to encrypt data on the storage volume of the endpoints created for inference recommendation. The inference recommendation job will fail asynchronously during endpoint configuration creation if the role passed does not have kms:CreateGrant permission. The KmsKeyId can be any of the following formats:   // KMS Key ID  "1234abcd-12ab-34cd-56ef-1234567890ab"    // Amazon Resource Name (ARN) of a KMS Key  "arn:aws:kms:&lt;region&gt;:&lt;account&gt;:key/&lt;key-id-12ab-34cd-56ef-1234567890ab&gt;"    // KMS Key Alias  "alias/ExampleAlias"    // Amazon Resource Name (ARN) of a KMS Key Alias  "arn:aws:kms:&lt;region&gt;:&lt;account&gt;:alias/&lt;ExampleAlias&gt;"    For more information about key identifiers, see Key identifiers (KeyID) in the Amazon Web Services Key Management Service (Amazon Web Services KMS) documentation.
    container_config: Specifies mandatory fields for running an Inference Recommender job. The fields specified in ContainerConfig override the corresponding fields in the model package.
    endpoints: Existing customer endpoints on which to run an Inference Recommender job.
    vpc_config: Inference Recommender provisions SageMaker endpoints with access to VPC in the inference recommendation job.
    tokenizer_config
    """

    model_package_version_arn: Optional[StrPipeVar] = Unassigned()
    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    job_duration_in_seconds: Optional[int] = Unassigned()
    traffic_pattern: Optional[TrafficPattern] = Unassigned()
    resource_limit: Optional[RecommendationJobResourceLimit] = Unassigned()
    endpoint_configurations: Optional[List[EndpointInputConfiguration]] = Unassigned()
    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()
    container_config: Optional[RecommendationJobContainerConfig] = Unassigned()
    endpoints: Optional[List[EndpointInfo]] = Unassigned()
    vpc_config: Optional[RecommendationJobVpcConfig] = Unassigned()
    tokenizer_config: Optional[TokenizerConfig] = Unassigned()


class ModelLatencyThreshold(Base):
    """
    ModelLatencyThreshold
      The model latency threshold.

    Attributes
    ----------------------
    percentile: The model latency percentile threshold. Acceptable values are P95 and P99. For custom load tests, specify the value as P95.
    value_in_milliseconds: The model latency percentile value in milliseconds.
    """

    percentile: Optional[StrPipeVar] = Unassigned()
    value_in_milliseconds: Optional[int] = Unassigned()


class RecommendationJobStoppingConditions(Base):
    """
    RecommendationJobStoppingConditions
      Specifies conditions for stopping a job. When a job reaches a stopping condition limit, SageMaker ends the job.

    Attributes
    ----------------------
    max_invocations: The maximum number of requests per minute expected for the endpoint.
    model_latency_thresholds: The interval of time taken by a model to respond as viewed from SageMaker. The interval includes the local communication time taken to send the request and to fetch the response from the container of a model and the time taken to complete the inference in the container.
    flat_invocations: Stops a load test when the number of invocations (TPS) peaks and flattens, which means that the instance has reached capacity. The default value is Stop. If you want the load test to continue after invocations have flattened, set the value to Continue.
    """

    max_invocations: Optional[int] = Unassigned()
    model_latency_thresholds: Optional[List[ModelLatencyThreshold]] = Unassigned()
    flat_invocations: Optional[StrPipeVar] = Unassigned()


class RecommendationJobTuningJob(Base):
    """
    RecommendationJobTuningJob

    Attributes
    ----------------------
    job_name
    """

    job_name: Optional[StrPipeVar] = Unassigned()


class RecommendationJobTuningWarmStartConfig(Base):
    """
    RecommendationJobTuningWarmStartConfig

    Attributes
    ----------------------
    jobs
    """

    jobs: Optional[List[RecommendationJobTuningJob]] = Unassigned()


class RecommendationJobTuningConvergenceDetected(Base):
    """
    RecommendationJobTuningConvergenceDetected

    Attributes
    ----------------------
    complete_on_convergence
    """

    complete_on_convergence: Optional[StrPipeVar] = Unassigned()


class RecommendationJobTuningBestObjectiveNotImproving(Base):
    """
    RecommendationJobTuningBestObjectiveNotImproving

    Attributes
    ----------------------
    max_number_of_tests_not_improving
    """

    max_number_of_tests_not_improving: Optional[int] = Unassigned()


class RecommendationJobTuningCompletionCriteria(Base):
    """
    RecommendationJobTuningCompletionCriteria

    Attributes
    ----------------------
    convergence_detected
    best_objective_not_improving
    """

    convergence_detected: Optional[RecommendationJobTuningConvergenceDetected] = Unassigned()
    best_objective_not_improving: Optional[RecommendationJobTuningBestObjectiveNotImproving] = (
        Unassigned()
    )


class RecommendationJobTuningObjectiveMetric(Base):
    """
    RecommendationJobTuningObjectiveMetric

    Attributes
    ----------------------
    name
    """

    name: Optional[StrPipeVar] = Unassigned()


class RecommendationJobEndpointConfigurationTuning(Base):
    """
    RecommendationJobEndpointConfigurationTuning

    Attributes
    ----------------------
    warm_start_config
    random_seed
    strategy
    completion_criteria
    objective_metric
    """

    warm_start_config: Optional[RecommendationJobTuningWarmStartConfig] = Unassigned()
    random_seed: Optional[int] = Unassigned()
    strategy: Optional[StrPipeVar] = Unassigned()
    completion_criteria: Optional[RecommendationJobTuningCompletionCriteria] = Unassigned()
    objective_metric: Optional[RecommendationJobTuningObjectiveMetric] = Unassigned()


class RecommendationJobCompiledOutputConfig(Base):
    """
    RecommendationJobCompiledOutputConfig
      Provides information about the output configuration for the compiled model.

    Attributes
    ----------------------
    s3_output_uri: Identifies the Amazon S3 bucket where you want SageMaker to store the compiled model artifacts.
    """

    s3_output_uri: Optional[StrPipeVar] = Unassigned()


class RecommendationJobOutputConfig(Base):
    """
    RecommendationJobOutputConfig
      Provides information about the output configuration for the compiled model.

    Attributes
    ----------------------
    kms_key_id: The Amazon Resource Name (ARN) of a Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt your output artifacts with Amazon S3 server-side encryption. The SageMaker execution role must have kms:GenerateDataKey permission. The KmsKeyId can be any of the following formats:   // KMS Key ID  "1234abcd-12ab-34cd-56ef-1234567890ab"    // Amazon Resource Name (ARN) of a KMS Key  "arn:aws:kms:&lt;region&gt;:&lt;account&gt;:key/&lt;key-id-12ab-34cd-56ef-1234567890ab&gt;"    // KMS Key Alias  "alias/ExampleAlias"    // Amazon Resource Name (ARN) of a KMS Key Alias  "arn:aws:kms:&lt;region&gt;:&lt;account&gt;:alias/&lt;ExampleAlias&gt;"    For more information about key identifiers, see Key identifiers (KeyID) in the Amazon Web Services Key Management Service (Amazon Web Services KMS) documentation.
    compiled_output_config: Provides information about the output configuration for the compiled model.
    benchmark_results_output_config
    """

    kms_key_id: Optional[StrPipeVar] = Unassigned()
    compiled_output_config: Optional[RecommendationJobCompiledOutputConfig] = Unassigned()
    benchmark_results_output_config: Optional[BenchmarkResultsOutputConfig] = Unassigned()


class LabelingJobS3DataSource(Base):
    """
    LabelingJobS3DataSource
      The Amazon S3 location of the input data objects.

    Attributes
    ----------------------
    manifest_s3_uri: The Amazon S3 location of the manifest file that describes the input data objects.  The input manifest file referenced in ManifestS3Uri must contain one of the following keys: source-ref or source. The value of the keys are interpreted as follows:    source-ref: The source of the object is the Amazon S3 object specified in the value. Use this value when the object is a binary object, such as an image.    source: The source of the object is the value. Use this value when the object is a text value.   If you are a new user of Ground Truth, it is recommended you review Use an Input Manifest File  in the Amazon SageMaker Developer Guide to learn how to create an input manifest file.
    """

    manifest_s3_uri: StrPipeVar


class LabelingJobSnsDataSource(Base):
    """
    LabelingJobSnsDataSource
      An Amazon SNS data source used for streaming labeling jobs.

    Attributes
    ----------------------
    sns_topic_arn: The Amazon SNS input topic Amazon Resource Name (ARN). Specify the ARN of the input topic you will use to send new data objects to a streaming labeling job.
    """

    sns_topic_arn: StrPipeVar


class LabelingJobDataSource(Base):
    """
    LabelingJobDataSource
      Provides information about the location of input data. You must specify at least one of the following: S3DataSource or SnsDataSource. Use SnsDataSource to specify an SNS input topic for a streaming labeling job. If you do not specify and SNS input topic ARN, Ground Truth will create a one-time labeling job. Use S3DataSource to specify an input manifest file for both streaming and one-time labeling jobs. Adding an S3DataSource is optional if you use SnsDataSource to create a streaming labeling job.

    Attributes
    ----------------------
    s3_data_source: The Amazon S3 location of the input data objects.
    sns_data_source: An Amazon SNS data source used for streaming labeling jobs. To learn more, see Send Data to a Streaming Labeling Job.
    """

    s3_data_source: Optional[LabelingJobS3DataSource] = Unassigned()
    sns_data_source: Optional[LabelingJobSnsDataSource] = Unassigned()


class LabelingJobDataAttributes(Base):
    """
    LabelingJobDataAttributes
      Attributes of the data specified by the customer. Use these to describe the data to be labeled.

    Attributes
    ----------------------
    content_classifiers: Declares that your content is free of personally identifiable information or adult content. SageMaker may restrict the Amazon Mechanical Turk workers that can view your task based on this information.
    """

    content_classifiers: Optional[List[StrPipeVar]] = Unassigned()


class LabelingJobInputConfig(Base):
    """
    LabelingJobInputConfig
      Input configuration information for a labeling job.

    Attributes
    ----------------------
    data_source: The location of the input data.
    data_attributes: Attributes of the data specified by the customer.
    """

    data_source: LabelingJobDataSource
    data_attributes: Optional[LabelingJobDataAttributes] = Unassigned()


class LabelingJobOutputConfig(Base):
    """
    LabelingJobOutputConfig
      Output configuration information for a labeling job.

    Attributes
    ----------------------
    s3_output_path: The Amazon S3 location to write output data.
    kms_key_id: The Amazon Web Services Key Management Service ID of the key used to encrypt the output data, if any. If you provide your own KMS key ID, you must add the required permissions to your KMS key described in Encrypt Output Data and Storage Volume with Amazon Web Services KMS. If you don't provide a KMS key ID, Amazon SageMaker uses the default Amazon Web Services KMS key for Amazon S3 for your role's account to encrypt your output data. If you use a bucket policy with an s3:PutObject permission that only allows objects with server-side encryption, set the condition key of s3:x-amz-server-side-encryption to "aws:kms". For more information, see KMS-Managed Encryption Keys in the Amazon Simple Storage Service Developer Guide.
    sns_topic_arn: An Amazon Simple Notification Service (Amazon SNS) output topic ARN. Provide a SnsTopicArn if you want to do real time chaining to another streaming job and receive an Amazon SNS notifications each time a data object is submitted by a worker. If you provide an SnsTopicArn in OutputConfig, when workers complete labeling tasks, Ground Truth will send labeling task output data to the SNS output topic you specify here.  To learn more, see Receive Output Data from a Streaming Labeling Job.
    """

    s3_output_path: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    sns_topic_arn: Optional[StrPipeVar] = Unassigned()


class LabelingJobStoppingConditions(Base):
    """
    LabelingJobStoppingConditions
      A set of conditions for stopping a labeling job. If any of the conditions are met, the job is automatically stopped. You can use these conditions to control the cost of data labeling.  Labeling jobs fail after 30 days with an appropriate client error message.

    Attributes
    ----------------------
    max_human_labeled_object_count: The maximum number of objects that can be labeled by human workers.
    max_percentage_of_input_dataset_labeled: The maximum number of input data objects that should be labeled.
    """

    max_human_labeled_object_count: Optional[int] = Unassigned()
    max_percentage_of_input_dataset_labeled: Optional[int] = Unassigned()


class LabelingJobResourceConfig(Base):
    """
    LabelingJobResourceConfig
      Configure encryption on the storage volume attached to the ML compute instance used to run automated data labeling model training and inference.

    Attributes
    ----------------------
    volume_kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt data on the storage volume attached to the ML compute instance(s) that run the training and inference jobs used for automated data labeling.  You can only specify a VolumeKmsKeyId when you create a labeling job with automated data labeling enabled using the API operation CreateLabelingJob. You cannot specify an Amazon Web Services KMS key to encrypt the storage volume used for automated data labeling model training and inference when you create a labeling job using the console. To learn more, see Output Data and Storage Volume Encryption. The VolumeKmsKeyId can be any of the following formats:   KMS Key ID  "1234abcd-12ab-34cd-56ef-1234567890ab"    Amazon Resource Name (ARN) of a KMS Key  "arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"
    vpc_config
    """

    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()
    vpc_config: Optional[VpcConfig] = Unassigned()


class LabelingJobAlgorithmsConfig(Base):
    """
    LabelingJobAlgorithmsConfig
      Provides configuration information for auto-labeling of your data objects. A LabelingJobAlgorithmsConfig object must be supplied in order to use auto-labeling.

    Attributes
    ----------------------
    labeling_job_algorithm_specification_arn: Specifies the Amazon Resource Name (ARN) of the algorithm used for auto-labeling. You must select one of the following ARNs:    Image classification   arn:aws:sagemaker:region:027400017018:labeling-job-algorithm-specification/image-classification     Text classification   arn:aws:sagemaker:region:027400017018:labeling-job-algorithm-specification/text-classification     Object detection   arn:aws:sagemaker:region:027400017018:labeling-job-algorithm-specification/object-detection     Semantic Segmentation   arn:aws:sagemaker:region:027400017018:labeling-job-algorithm-specification/semantic-segmentation
    initial_active_learning_model_arn: At the end of an auto-label job Ground Truth sends the Amazon Resource Name (ARN) of the final model used for auto-labeling. You can use this model as the starting point for subsequent similar jobs by providing the ARN of the model here.
    labeling_job_resource_config: Provides configuration information for a labeling job.
    """

    labeling_job_algorithm_specification_arn: StrPipeVar
    initial_active_learning_model_arn: Optional[StrPipeVar] = Unassigned()
    labeling_job_resource_config: Optional[LabelingJobResourceConfig] = Unassigned()


class UiConfig(Base):
    """
    UiConfig
      Provided configuration information for the worker UI for a labeling job. Provide either HumanTaskUiArn or UiTemplateS3Uri. For named entity recognition, 3D point cloud and video frame labeling jobs, use HumanTaskUiArn. For all other Ground Truth built-in task types and custom task types, use UiTemplateS3Uri to specify the location of a worker task template in Amazon S3.

    Attributes
    ----------------------
    ui_template_s3_uri: The Amazon S3 bucket location of the UI template, or worker task template. This is the template used to render the worker UI and tools for labeling job tasks. For more information about the contents of a UI template, see  Creating Your Custom Labeling Task Template.
    human_task_ui_arn: The ARN of the worker task template used to render the worker UI and tools for labeling job tasks. Use this parameter when you are creating a labeling job for named entity recognition, 3D point cloud and video frame labeling jobs. Use your labeling job task type to select one of the following ARNs and use it with this parameter when you create a labeling job. Replace aws-region with the Amazon Web Services Region you are creating your labeling job in. For example, replace aws-region with us-west-1 if you create a labeling job in US West (N. California).  Named Entity Recognition  Use the following HumanTaskUiArn for named entity recognition labeling jobs:  arn:aws:sagemaker:aws-region:394669845002:human-task-ui/NamedEntityRecognition   3D Point Cloud HumanTaskUiArns  Use this HumanTaskUiArn for 3D point cloud object detection and 3D point cloud object detection adjustment labeling jobs.     arn:aws:sagemaker:aws-region:394669845002:human-task-ui/PointCloudObjectDetection     Use this HumanTaskUiArn for 3D point cloud object tracking and 3D point cloud object tracking adjustment labeling jobs.     arn:aws:sagemaker:aws-region:394669845002:human-task-ui/PointCloudObjectTracking     Use this HumanTaskUiArn for 3D point cloud semantic segmentation and 3D point cloud semantic segmentation adjustment labeling jobs.    arn:aws:sagemaker:aws-region:394669845002:human-task-ui/PointCloudSemanticSegmentation     Video Frame HumanTaskUiArns  Use this HumanTaskUiArn for video frame object detection and video frame object detection adjustment labeling jobs.     arn:aws:sagemaker:region:394669845002:human-task-ui/VideoObjectDetection     Use this HumanTaskUiArn for video frame object tracking and video frame object tracking adjustment labeling jobs.     arn:aws:sagemaker:aws-region:394669845002:human-task-ui/VideoObjectTracking
    """

    ui_template_s3_uri: Optional[StrPipeVar] = Unassigned()
    human_task_ui_arn: Optional[StrPipeVar] = Unassigned()


class HumanTaskConfig(Base):
    """
    HumanTaskConfig
      Information required for human workers to complete a labeling task.

    Attributes
    ----------------------
    workteam_arn: The Amazon Resource Name (ARN) of the work team assigned to complete the tasks.
    ui_config: Information about the user interface that workers use to complete the labeling task.
    pre_human_task_lambda_arn: The Amazon Resource Name (ARN) of a Lambda function that is run before a data object is sent to a human worker. Use this function to provide input to a custom labeling job. For built-in task types, use one of the following Amazon SageMaker Ground Truth Lambda function ARNs for PreHumanTaskLambdaArn. For custom labeling workflows, see Pre-annotation Lambda.   Bounding box - Finds the most similar boxes from different workers based on the Jaccard index of the boxes.    arn:aws:lambda:us-east-1:432418664414:function:PRE-BoundingBox     arn:aws:lambda:us-east-2:266458841044:function:PRE-BoundingBox     arn:aws:lambda:us-west-2:081040173940:function:PRE-BoundingBox     arn:aws:lambda:ca-central-1:918755190332:function:PRE-BoundingBox     arn:aws:lambda:eu-west-1:568282634449:function:PRE-BoundingBox     arn:aws:lambda:eu-west-2:487402164563:function:PRE-BoundingBox     arn:aws:lambda:eu-central-1:203001061592:function:PRE-BoundingBox     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-BoundingBox     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-BoundingBox     arn:aws:lambda:ap-south-1:565803892007:function:PRE-BoundingBox     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-BoundingBox     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-BoundingBox     Image classification - Uses a variant of the Expectation Maximization approach to estimate the true class of an image based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:PRE-ImageMultiClass     arn:aws:lambda:us-east-2:266458841044:function:PRE-ImageMultiClass     arn:aws:lambda:us-west-2:081040173940:function:PRE-ImageMultiClass     arn:aws:lambda:ca-central-1:918755190332:function:PRE-ImageMultiClass     arn:aws:lambda:eu-west-1:568282634449:function:PRE-ImageMultiClass     arn:aws:lambda:eu-west-2:487402164563:function:PRE-ImageMultiClass     arn:aws:lambda:eu-central-1:203001061592:function:PRE-ImageMultiClass     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-ImageMultiClass     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-ImageMultiClass     arn:aws:lambda:ap-south-1:565803892007:function:PRE-ImageMultiClass     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-ImageMultiClass     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-ImageMultiClass     Multi-label image classification - Uses a variant of the Expectation Maximization approach to estimate the true classes of an image based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:us-east-2:266458841044:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:us-west-2:081040173940:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:ca-central-1:918755190332:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:eu-west-1:568282634449:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:eu-west-2:487402164563:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:eu-central-1:203001061592:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:ap-south-1:565803892007:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-ImageMultiClassMultiLabel     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-ImageMultiClassMultiLabel     Semantic segmentation - Treats each pixel in an image as a multi-class classification and treats pixel annotations from workers as "votes" for the correct label.    arn:aws:lambda:us-east-1:432418664414:function:PRE-SemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:PRE-SemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:PRE-SemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:PRE-SemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:PRE-SemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:PRE-SemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:PRE-SemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-SemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-SemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:PRE-SemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-SemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-SemanticSegmentation     Text classification - Uses a variant of the Expectation Maximization approach to estimate the true class of text based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:PRE-TextMultiClass     arn:aws:lambda:us-east-2:266458841044:function:PRE-TextMultiClass     arn:aws:lambda:us-west-2:081040173940:function:PRE-TextMultiClass     arn:aws:lambda:ca-central-1:918755190332:function:PRE-TextMultiClass     arn:aws:lambda:eu-west-1:568282634449:function:PRE-TextMultiClass     arn:aws:lambda:eu-west-2:487402164563:function:PRE-TextMultiClass     arn:aws:lambda:eu-central-1:203001061592:function:PRE-TextMultiClass     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-TextMultiClass     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-TextMultiClass     arn:aws:lambda:ap-south-1:565803892007:function:PRE-TextMultiClass     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-TextMultiClass     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-TextMultiClass     Multi-label text classification - Uses a variant of the Expectation Maximization approach to estimate the true classes of text based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:us-east-2:266458841044:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:us-west-2:081040173940:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:ca-central-1:918755190332:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:eu-west-1:568282634449:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:eu-west-2:487402164563:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:eu-central-1:203001061592:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:ap-south-1:565803892007:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-TextMultiClassMultiLabel     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-TextMultiClassMultiLabel     Named entity recognition - Groups similar selections and calculates aggregate boundaries, resolving to most-assigned label.    arn:aws:lambda:us-east-1:432418664414:function:PRE-NamedEntityRecognition     arn:aws:lambda:us-east-2:266458841044:function:PRE-NamedEntityRecognition     arn:aws:lambda:us-west-2:081040173940:function:PRE-NamedEntityRecognition     arn:aws:lambda:ca-central-1:918755190332:function:PRE-NamedEntityRecognition     arn:aws:lambda:eu-west-1:568282634449:function:PRE-NamedEntityRecognition     arn:aws:lambda:eu-west-2:487402164563:function:PRE-NamedEntityRecognition     arn:aws:lambda:eu-central-1:203001061592:function:PRE-NamedEntityRecognition     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-NamedEntityRecognition     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-NamedEntityRecognition     arn:aws:lambda:ap-south-1:565803892007:function:PRE-NamedEntityRecognition     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-NamedEntityRecognition     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-NamedEntityRecognition     Video Classification - Use this task type when you need workers to classify videos using predefined labels that you specify. Workers are shown videos and are asked to choose one label for each video.    arn:aws:lambda:us-east-1:432418664414:function:PRE-VideoMultiClass     arn:aws:lambda:us-east-2:266458841044:function:PRE-VideoMultiClass     arn:aws:lambda:us-west-2:081040173940:function:PRE-VideoMultiClass     arn:aws:lambda:eu-west-1:568282634449:function:PRE-VideoMultiClass     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-VideoMultiClass     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-VideoMultiClass     arn:aws:lambda:ap-south-1:565803892007:function:PRE-VideoMultiClass     arn:aws:lambda:eu-central-1:203001061592:function:PRE-VideoMultiClass     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-VideoMultiClass     arn:aws:lambda:eu-west-2:487402164563:function:PRE-VideoMultiClass     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-VideoMultiClass     arn:aws:lambda:ca-central-1:918755190332:function:PRE-VideoMultiClass     Video Frame Object Detection - Use this task type to have workers identify and locate objects in a sequence of video frames (images extracted from a video) using bounding boxes. For example, you can use this task to ask workers to identify and localize various objects in a series of video frames, such as cars, bikes, and pedestrians.    arn:aws:lambda:us-east-1:432418664414:function:PRE-VideoObjectDetection     arn:aws:lambda:us-east-2:266458841044:function:PRE-VideoObjectDetection     arn:aws:lambda:us-west-2:081040173940:function:PRE-VideoObjectDetection     arn:aws:lambda:eu-west-1:568282634449:function:PRE-VideoObjectDetection     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-VideoObjectDetection     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-VideoObjectDetection     arn:aws:lambda:ap-south-1:565803892007:function:PRE-VideoObjectDetection     arn:aws:lambda:eu-central-1:203001061592:function:PRE-VideoObjectDetection     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-VideoObjectDetection     arn:aws:lambda:eu-west-2:487402164563:function:PRE-VideoObjectDetection     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-VideoObjectDetection     arn:aws:lambda:ca-central-1:918755190332:function:PRE-VideoObjectDetection     Video Frame Object Tracking - Use this task type to have workers track the movement of objects in a sequence of video frames (images extracted from a video) using bounding boxes. For example, you can use this task to ask workers to track the movement of objects, such as cars, bikes, and pedestrians.     arn:aws:lambda:us-east-1:432418664414:function:PRE-VideoObjectTracking     arn:aws:lambda:us-east-2:266458841044:function:PRE-VideoObjectTracking     arn:aws:lambda:us-west-2:081040173940:function:PRE-VideoObjectTracking     arn:aws:lambda:eu-west-1:568282634449:function:PRE-VideoObjectTracking     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-VideoObjectTracking     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-VideoObjectTracking     arn:aws:lambda:ap-south-1:565803892007:function:PRE-VideoObjectTracking     arn:aws:lambda:eu-central-1:203001061592:function:PRE-VideoObjectTracking     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-VideoObjectTracking     arn:aws:lambda:eu-west-2:487402164563:function:PRE-VideoObjectTracking     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-VideoObjectTracking     arn:aws:lambda:ca-central-1:918755190332:function:PRE-VideoObjectTracking     3D Point Cloud Modalities  Use the following pre-annotation lambdas for 3D point cloud labeling modality tasks. See 3D Point Cloud Task types  to learn more.   3D Point Cloud Object Detection - Use this task type when you want workers to classify objects in a 3D point cloud by drawing 3D cuboids around objects. For example, you can use this task type to ask workers to identify different types of objects in a point cloud, such as cars, bikes, and pedestrians.    arn:aws:lambda:us-east-1:432418664414:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:us-east-2:266458841044:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:us-west-2:081040173940:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:eu-west-1:568282634449:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:ap-south-1:565803892007:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:eu-central-1:203001061592:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:eu-west-2:487402164563:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-3DPointCloudObjectDetection     arn:aws:lambda:ca-central-1:918755190332:function:PRE-3DPointCloudObjectDetection     3D Point Cloud Object Tracking - Use this task type when you want workers to draw 3D cuboids around objects that appear in a sequence of 3D point cloud frames. For example, you can use this task type to ask workers to track the movement of vehicles across multiple point cloud frames.     arn:aws:lambda:us-east-1:432418664414:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:us-east-2:266458841044:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:us-west-2:081040173940:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:eu-west-1:568282634449:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:ap-south-1:565803892007:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:eu-central-1:203001061592:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:eu-west-2:487402164563:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-3DPointCloudObjectTracking     arn:aws:lambda:ca-central-1:918755190332:function:PRE-3DPointCloudObjectTracking     3D Point Cloud Semantic Segmentation - Use this task type when you want workers to create a point-level semantic segmentation masks by painting objects in a 3D point cloud using different colors where each color is assigned to one of the classes you specify.    arn:aws:lambda:us-east-1:432418664414:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-3DPointCloudSemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:PRE-3DPointCloudSemanticSegmentation     Use the following ARNs for Label Verification and Adjustment Jobs  Use label verification and adjustment jobs to review and adjust labels. To learn more, see Verify and Adjust Labels .  Bounding box verification - Uses a variant of the Expectation Maximization approach to estimate the true class of verification judgement for bounding box labels based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:PRE-VerificationBoundingBox     arn:aws:lambda:us-east-2:266458841044:function:PRE-VerificationBoundingBox     arn:aws:lambda:us-west-2:081040173940:function:PRE-VerificationBoundingBox     arn:aws:lambda:eu-west-1:568282634449:function:PRE-VerificationBoundingBox     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-VerificationBoundingBox     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-VerificationBoundingBox     arn:aws:lambda:ap-south-1:565803892007:function:PRE-VerificationBoundingBox     arn:aws:lambda:eu-central-1:203001061592:function:PRE-VerificationBoundingBox     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-VerificationBoundingBox     arn:aws:lambda:eu-west-2:487402164563:function:PRE-VerificationBoundingBox     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-VerificationBoundingBox     arn:aws:lambda:ca-central-1:918755190332:function:PRE-VerificationBoundingBox     Bounding box adjustment - Finds the most similar boxes from different workers based on the Jaccard index of the adjusted annotations.    arn:aws:lambda:us-east-1:432418664414:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:us-east-2:266458841044:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:us-west-2:081040173940:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:ca-central-1:918755190332:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:eu-west-1:568282634449:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:eu-west-2:487402164563:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:eu-central-1:203001061592:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:ap-south-1:565803892007:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-AdjustmentBoundingBox     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-AdjustmentBoundingBox     Semantic segmentation verification - Uses a variant of the Expectation Maximization approach to estimate the true class of verification judgment for semantic segmentation labels based on annotations from individual workers.    arn:aws:lambda:us-east-1:432418664414:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-VerificationSemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-VerificationSemanticSegmentation     Semantic segmentation adjustment - Treats each pixel in an image as a multi-class classification and treats pixel adjusted annotations from workers as "votes" for the correct label.    arn:aws:lambda:us-east-1:432418664414:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-AdjustmentSemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-AdjustmentSemanticSegmentation     Video Frame Object Detection Adjustment - Use this task type when you want workers to adjust bounding boxes that workers have added to video frames to classify and localize objects in a sequence of video frames.    arn:aws:lambda:us-east-1:432418664414:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:us-east-2:266458841044:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:us-west-2:081040173940:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:eu-west-1:568282634449:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-south-1:565803892007:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:eu-central-1:203001061592:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:eu-west-2:487402164563:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-AdjustmentVideoObjectDetection     arn:aws:lambda:ca-central-1:918755190332:function:PRE-AdjustmentVideoObjectDetection     Video Frame Object Tracking Adjustment - Use this task type when you want workers to adjust bounding boxes that workers have added to video frames to track object movement across a sequence of video frames.    arn:aws:lambda:us-east-1:432418664414:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:us-east-2:266458841044:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:us-west-2:081040173940:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:eu-west-1:568282634449:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-south-1:565803892007:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:eu-central-1:203001061592:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:eu-west-2:487402164563:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-AdjustmentVideoObjectTracking     arn:aws:lambda:ca-central-1:918755190332:function:PRE-AdjustmentVideoObjectTracking     3D point cloud object detection adjustment - Adjust 3D cuboids in a point cloud frame.     arn:aws:lambda:us-east-1:432418664414:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:us-east-2:266458841044:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:us-west-2:081040173940:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:eu-west-1:568282634449:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-south-1:565803892007:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:eu-central-1:203001061592:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:eu-west-2:487402164563:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-Adjustment3DPointCloudObjectDetection     arn:aws:lambda:ca-central-1:918755190332:function:PRE-Adjustment3DPointCloudObjectDetection     3D point cloud object tracking adjustment - Adjust 3D cuboids across a sequence of point cloud frames.     arn:aws:lambda:us-east-1:432418664414:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:us-east-2:266458841044:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:us-west-2:081040173940:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:eu-west-1:568282634449:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-south-1:565803892007:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:eu-central-1:203001061592:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:eu-west-2:487402164563:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-Adjustment3DPointCloudObjectTracking     arn:aws:lambda:ca-central-1:918755190332:function:PRE-Adjustment3DPointCloudObjectTracking     3D point cloud semantic segmentation adjustment - Adjust semantic segmentation masks in a 3D point cloud.     arn:aws:lambda:us-east-1:432418664414:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:us-east-2:266458841044:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:us-west-2:081040173940:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-west-1:568282634449:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-south-1:565803892007:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-central-1:203001061592:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:eu-west-2:487402164563:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-Adjustment3DPointCloudSemanticSegmentation     arn:aws:lambda:ca-central-1:918755190332:function:PRE-Adjustment3DPointCloudSemanticSegmentation     Generative AI/Custom - Direct passthrough of input data without any transformation.    arn:aws:lambda:us-east-1:432418664414:function:PRE-PassThrough     arn:aws:lambda:us-east-2:266458841044:function:PRE-PassThrough     arn:aws:lambda:us-west-2:081040173940:function:PRE-PassThrough     arn:aws:lambda:ca-central-1:918755190332:function:PRE-PassThrough     arn:aws:lambda:eu-west-1:568282634449:function:PRE-PassThrough     arn:aws:lambda:eu-west-2:487402164563:function:PRE-PassThrough     arn:aws:lambda:eu-central-1:203001061592:function:PRE-PassThrough     arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-PassThrough     arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-PassThrough     arn:aws:lambda:ap-south-1:565803892007:function:PRE-PassThrough     arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-PassThrough     arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-PassThrough
    task_keywords: Keywords used to describe the task so that workers on Amazon Mechanical Turk can discover the task.
    task_title: A title for the task for your human workers.
    task_description: A description of the task for your human workers.
    number_of_human_workers_per_data_object: The number of human workers that will label an object.
    task_time_limit_in_seconds: The amount of time that a worker has to complete a task.  If you create a custom labeling job, the maximum value for this parameter is 8 hours (28,800 seconds). If you create a labeling job using a built-in task type the maximum for this parameter depends on the task type you use:   For image and text labeling jobs, the maximum is 8 hours (28,800 seconds).   For 3D point cloud and video frame labeling jobs, the maximum is 30 days (2952,000 seconds) for non-AL mode. For most users, the maximum is also 30 days.
    task_availability_lifetime_in_seconds: The length of time that a task remains available for labeling by human workers. The default and maximum values for this parameter depend on the type of workforce you use.   If you choose the Amazon Mechanical Turk workforce, the maximum is 12 hours (43,200 seconds). The default is 6 hours (21,600 seconds).   If you choose a private or vendor workforce, the default value is 30 days (2592,000 seconds) for non-AL mode. For most users, the maximum is also 30 days.
    max_concurrent_task_count: Defines the maximum number of data objects that can be labeled by human workers at the same time. Also referred to as batch size. Each object may have more than one worker at one time. The default value is 1000 objects. To increase the maximum value to 5000 objects, contact Amazon Web Services Support.
    annotation_consolidation_config: Configures how labels are consolidated across human workers.
    public_workforce_task_price: The price that you pay for each task performed by an Amazon Mechanical Turk worker.
    """

    workteam_arn: StrPipeVar
    ui_config: UiConfig
    task_title: StrPipeVar
    task_description: StrPipeVar
    number_of_human_workers_per_data_object: int
    task_time_limit_in_seconds: int
    pre_human_task_lambda_arn: Optional[StrPipeVar] = Unassigned()
    task_keywords: Optional[List[StrPipeVar]] = Unassigned()
    task_availability_lifetime_in_seconds: Optional[int] = Unassigned()
    max_concurrent_task_count: Optional[int] = Unassigned()
    annotation_consolidation_config: Optional[AnnotationConsolidationConfig] = Unassigned()
    public_workforce_task_price: Optional[PublicWorkforceTaskPrice] = Unassigned()


class ModelBiasBaselineConfig(Base):
    """
    ModelBiasBaselineConfig
      The configuration for a baseline model bias job.

    Attributes
    ----------------------
    baselining_job_name: The name of the baseline model bias job.
    constraints_resource
    """

    baselining_job_name: Optional[StrPipeVar] = Unassigned()
    constraints_resource: Optional[MonitoringConstraintsResource] = Unassigned()


class ModelBiasAppSpecification(Base):
    """
    ModelBiasAppSpecification
      Docker container image configuration object for the model bias job.

    Attributes
    ----------------------
    image_uri: The container image to be run by the model bias job.
    config_uri: JSON formatted S3 file that defines bias parameters. For more information on this JSON configuration file, see Configure bias parameters.
    environment: Sets the environment variables in the Docker container.
    """

    image_uri: StrPipeVar
    config_uri: StrPipeVar
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class ModelBiasJobInput(Base):
    """
    ModelBiasJobInput
      Inputs for the model bias job.

    Attributes
    ----------------------
    endpoint_input
    batch_transform_input: Input object for the batch transform job.
    ground_truth_s3_input: Location of ground truth labels to use in model bias job.
    """

    ground_truth_s3_input: MonitoringGroundTruthS3Input
    endpoint_input: Optional[EndpointInput] = Unassigned()
    batch_transform_input: Optional[BatchTransformInput] = Unassigned()


class ModelCardExportOutputConfig(Base):
    """
    ModelCardExportOutputConfig
      Configure the export output details for an Amazon SageMaker Model Card.

    Attributes
    ----------------------
    s3_output_path: The Amazon S3 output path to export your model card PDF.
    """

    s3_output_path: StrPipeVar


class ModelCardSecurityConfig(Base):
    """
    ModelCardSecurityConfig
      Configure the security settings to protect model card data.

    Attributes
    ----------------------
    kms_key_id: A Key Management Service key ID to use for encrypting a model card.
    """

    kms_key_id: Optional[StrPipeVar] = Unassigned()


class ModelExplainabilityBaselineConfig(Base):
    """
    ModelExplainabilityBaselineConfig
      The configuration for a baseline model explainability job.

    Attributes
    ----------------------
    baselining_job_name: The name of the baseline model explainability job.
    constraints_resource
    """

    baselining_job_name: Optional[StrPipeVar] = Unassigned()
    constraints_resource: Optional[MonitoringConstraintsResource] = Unassigned()


class ModelExplainabilityAppSpecification(Base):
    """
    ModelExplainabilityAppSpecification
      Docker container image configuration object for the model explainability job.

    Attributes
    ----------------------
    image_uri: The container image to be run by the model explainability job.
    config_uri: JSON formatted Amazon S3 file that defines explainability parameters. For more information on this JSON configuration file, see Configure model explainability parameters.
    environment: Sets the environment variables in the Docker container.
    """

    image_uri: StrPipeVar
    config_uri: StrPipeVar
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class ModelExplainabilityJobInput(Base):
    """
    ModelExplainabilityJobInput
      Inputs for the model explainability job.

    Attributes
    ----------------------
    endpoint_input
    batch_transform_input: Input object for the batch transform job.
    """

    endpoint_input: Optional[EndpointInput] = Unassigned()
    batch_transform_input: Optional[BatchTransformInput] = Unassigned()


class InferenceExecutionConfig(Base):
    """
    InferenceExecutionConfig
      Specifies details about how containers in a multi-container endpoint are run.

    Attributes
    ----------------------
    mode: How containers in a multi-container are run. The following values are valid.    SERIAL - Containers run as a serial pipeline.    DIRECT - Only the individual container that you specify is run.
    """

    mode: StrPipeVar


class ModelPackageValidationProfile(Base):
    """
    ModelPackageValidationProfile
      Contains data, such as the inputs and targeted instance types that are used in the process of validating the model package. The data provided in the validation profile is made available to your buyers on Amazon Web Services Marketplace.

    Attributes
    ----------------------
    profile_name: The name of the profile for the model package.
    transform_job_definition: The TransformJobDefinition object that describes the transform job used for the validation of the model package.
    """

    profile_name: StrPipeVar
    transform_job_definition: TransformJobDefinition


class ModelPackageValidationSpecification(Base):
    """
    ModelPackageValidationSpecification
      Specifies batch transform jobs that SageMaker runs to validate your model package.

    Attributes
    ----------------------
    validation_role: The IAM roles to be used for the validation of the model package.
    validation_profiles: An array of ModelPackageValidationProfile objects, each of which specifies a batch transform job that SageMaker runs to validate your model package.
    """

    validation_role: StrPipeVar
    validation_profiles: List[ModelPackageValidationProfile]


class SourceAlgorithm(Base):
    """
    SourceAlgorithm
      Specifies an algorithm that was used to create the model package. The algorithm must be either an algorithm resource in your SageMaker account or an algorithm in Amazon Web Services Marketplace that you are subscribed to.

    Attributes
    ----------------------
    model_data_url: The Amazon S3 path where the model artifacts, which result from model training, are stored. This path must point to a single gzip compressed tar archive (.tar.gz suffix).  The model artifacts must be in an S3 bucket that is in the same Amazon Web Services region as the algorithm.
    model_data_source: Specifies the location of ML model data to deploy during endpoint creation.
    model_data_e_tag: The ETag associated with Model Data URL.
    algorithm_name: The name of an algorithm that was used to create the model package. The algorithm must be either an algorithm resource in your SageMaker account or an algorithm in Amazon Web Services Marketplace that you are subscribed to.
    """

    algorithm_name: Union[StrPipeVar, object]
    model_data_url: Optional[StrPipeVar] = Unassigned()
    model_data_source: Optional[ModelDataSource] = Unassigned()
    model_data_e_tag: Optional[StrPipeVar] = Unassigned()


class SourceAlgorithmSpecification(Base):
    """
    SourceAlgorithmSpecification
      A list of algorithms that were used to create a model package.

    Attributes
    ----------------------
    source_algorithms: A list of the algorithms that were used to create a model package.
    """

    source_algorithms: List[SourceAlgorithm]


class ModelQuality(Base):
    """
    ModelQuality
      Model quality statistics and constraints.

    Attributes
    ----------------------
    statistics: Model quality statistics.
    constraints: Model quality constraints.
    """

    statistics: Optional[MetricsSource] = Unassigned()
    constraints: Optional[MetricsSource] = Unassigned()


class ModelDataQuality(Base):
    """
    ModelDataQuality
      Data quality constraints and statistics for a model.

    Attributes
    ----------------------
    statistics: Data quality statistics for a model.
    constraints: Data quality constraints for a model.
    """

    statistics: Optional[MetricsSource] = Unassigned()
    constraints: Optional[MetricsSource] = Unassigned()


class Explainability(Base):
    """
    Explainability
      Contains explainability metrics for a model.

    Attributes
    ----------------------
    report: The explainability report for a model.
    """

    report: Optional[MetricsSource] = Unassigned()


class ModelMetrics(Base):
    """
    ModelMetrics
      Contains metrics captured from a model.

    Attributes
    ----------------------
    model_quality: Metrics that measure the quality of a model.
    model_data_quality: Metrics that measure the quality of the input data for a model.
    bias: Metrics that measure bias in a model.
    explainability: Metrics that help explain a model.
    """

    model_quality: Optional[ModelQuality] = Unassigned()
    model_data_quality: Optional[ModelDataQuality] = Unassigned()
    bias: Optional[Bias] = Unassigned()
    explainability: Optional[Explainability] = Unassigned()


class TestInput(Base):
    """
    TestInput

    Attributes
    ----------------------
    data_source
    content_type
    compression_type
    split_type
    """

    data_source: Optional[DataSource] = Unassigned()
    content_type: Optional[StrPipeVar] = Unassigned()
    compression_type: Optional[StrPipeVar] = Unassigned()
    split_type: Optional[StrPipeVar] = Unassigned()


class HealthCheckConfig(Base):
    """
    HealthCheckConfig

    Attributes
    ----------------------
    num_payload
    num_failures_allowed
    """

    num_payload: Optional[int] = Unassigned()
    num_failures_allowed: Optional[int] = Unassigned()


class DeploymentSpecification(Base):
    """
    DeploymentSpecification

    Attributes
    ----------------------
    test_input
    health_check_config
    """

    test_input: Optional[TestInput] = Unassigned()
    health_check_config: Optional[HealthCheckConfig] = Unassigned()


class FileSource(Base):
    """
    FileSource
      Contains details regarding the file source.

    Attributes
    ----------------------
    content_type: The type of content stored in the file source.
    content_digest: The digest of the file source.
    s3_uri: The Amazon S3 URI for the file source.
    """

    s3_uri: StrPipeVar
    content_type: Optional[StrPipeVar] = Unassigned()
    content_digest: Optional[StrPipeVar] = Unassigned()


class DriftCheckBias(Base):
    """
    DriftCheckBias
      Represents the drift check bias baselines that can be used when the model monitor is set using the model package.

    Attributes
    ----------------------
    config_file: The bias config file for a model.
    pre_training_constraints: The pre-training constraints.
    post_training_constraints: The post-training constraints.
    """

    config_file: Optional[FileSource] = Unassigned()
    pre_training_constraints: Optional[MetricsSource] = Unassigned()
    post_training_constraints: Optional[MetricsSource] = Unassigned()


class DriftCheckExplainability(Base):
    """
    DriftCheckExplainability
      Represents the drift check explainability baselines that can be used when the model monitor is set using the model package.

    Attributes
    ----------------------
    constraints: The drift check explainability constraints.
    config_file: The explainability config file for the model.
    """

    constraints: Optional[MetricsSource] = Unassigned()
    config_file: Optional[FileSource] = Unassigned()


class DriftCheckModelQuality(Base):
    """
    DriftCheckModelQuality
      Represents the drift check model quality baselines that can be used when the model monitor is set using the model package.

    Attributes
    ----------------------
    statistics: The drift check model quality statistics.
    constraints: The drift check model quality constraints.
    """

    statistics: Optional[MetricsSource] = Unassigned()
    constraints: Optional[MetricsSource] = Unassigned()


class DriftCheckModelDataQuality(Base):
    """
    DriftCheckModelDataQuality
      Represents the drift check data quality baselines that can be used when the model monitor is set using the model package.

    Attributes
    ----------------------
    statistics: The drift check model data quality statistics.
    constraints: The drift check model data quality constraints.
    """

    statistics: Optional[MetricsSource] = Unassigned()
    constraints: Optional[MetricsSource] = Unassigned()


class DriftCheckBaselines(Base):
    """
    DriftCheckBaselines
      Represents the drift check baselines that can be used when the model monitor is set using the model package.

    Attributes
    ----------------------
    bias: Represents the drift check bias baselines that can be used when the model monitor is set using the model package.
    explainability: Represents the drift check explainability baselines that can be used when the model monitor is set using the model package.
    model_quality: Represents the drift check model quality baselines that can be used when the model monitor is set using the model package.
    model_data_quality: Represents the drift check model data quality baselines that can be used when the model monitor is set using the model package.
    """

    bias: Optional[DriftCheckBias] = Unassigned()
    explainability: Optional[DriftCheckExplainability] = Unassigned()
    model_quality: Optional[DriftCheckModelQuality] = Unassigned()
    model_data_quality: Optional[DriftCheckModelDataQuality] = Unassigned()


class ModelPackageSecurityConfig(Base):
    """
    ModelPackageSecurityConfig
      An optional Key Management Service key to encrypt, decrypt, and re-encrypt model package information for regulated workloads with highly sensitive data.

    Attributes
    ----------------------
    kms_key_id: The KMS Key ID (KMSKeyId) used for encryption of model package information.
    """

    kms_key_id: Optional[str] = Unassigned()


class ModelPackageModelCard(Base):
    """
    ModelPackageModelCard
      The model card associated with the model package. Since ModelPackageModelCard is tied to a model package, it is a specific usage of a model card and its schema is simplified compared to the schema of ModelCard. The ModelPackageModelCard schema does not include model_package_details, and model_overview is composed of the model_creator and model_artifact properties. For more information about the model package model card schema, see Model package model card schema. For more information about the model card associated with the model package, see View the Details of a Model Version.

    Attributes
    ----------------------
    model_card_content: The content of the model card. The content must follow the schema described in Model Package Model Card Schema.
    model_card_status: The approval status of the model card within your organization. Different organizations might have different criteria for model card review and approval.    Draft: The model card is a work in progress.    PendingReview: The model card is pending review.    Approved: The model card is approved.    Archived: The model card is archived. No more updates can be made to the model card content. If you try to update the model card content, you will receive the message Model Card is in Archived state.
    """

    model_card_content: Optional[StrPipeVar] = Unassigned()
    model_card_status: Optional[StrPipeVar] = Unassigned()


class ModelLifeCycle(Base):
    """
    ModelLifeCycle
       A structure describing the current state of the model in its life cycle.

    Attributes
    ----------------------
    stage:  The current stage in the model life cycle.
    stage_status:  The current status of a stage in model life cycle.
    stage_description:  Describes the stage related details.
    """

    stage: StrPipeVar
    stage_status: StrPipeVar
    stage_description: Optional[StrPipeVar] = Unassigned()


class ModelQualityBaselineConfig(Base):
    """
    ModelQualityBaselineConfig
      Configuration for monitoring constraints and monitoring statistics. These baseline resources are compared against the results of the current job from the series of jobs scheduled to collect data periodically.

    Attributes
    ----------------------
    baselining_job_name: The name of the job that performs baselining for the monitoring job.
    constraints_resource
    """

    baselining_job_name: Optional[StrPipeVar] = Unassigned()
    constraints_resource: Optional[MonitoringConstraintsResource] = Unassigned()


class ModelQualityAppSpecification(Base):
    """
    ModelQualityAppSpecification
      Container image configuration object for the monitoring job.

    Attributes
    ----------------------
    image_uri: The address of the container image that the monitoring job runs.
    container_entrypoint: Specifies the entrypoint for a container that the monitoring job runs.
    container_arguments: An array of arguments for the container used to run the monitoring job.
    record_preprocessor_source_uri: An Amazon S3 URI to a script that is called per row prior to running analysis. It can base64 decode the payload and convert it into a flattened JSON so that the built-in container can use the converted data. Applicable only for the built-in (first party) containers.
    post_analytics_processor_source_uri: An Amazon S3 URI to a script that is called after analysis has been performed. Applicable only for the built-in (first party) containers.
    problem_type: The machine learning problem type of the model that the monitoring job monitors.
    environment: Sets the environment variables in the container that the monitoring job runs.
    """

    image_uri: StrPipeVar
    container_entrypoint: Optional[List[StrPipeVar]] = Unassigned()
    container_arguments: Optional[List[StrPipeVar]] = Unassigned()
    record_preprocessor_source_uri: Optional[StrPipeVar] = Unassigned()
    post_analytics_processor_source_uri: Optional[StrPipeVar] = Unassigned()
    problem_type: Optional[StrPipeVar] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class ModelQualityJobInput(Base):
    """
    ModelQualityJobInput
      The input for the model quality monitoring job. Currently endpoints are supported for input for model quality monitoring jobs.

    Attributes
    ----------------------
    endpoint_input
    batch_transform_input: Input object for the batch transform job.
    ground_truth_s3_input: The ground truth label provided for the model.
    """

    ground_truth_s3_input: MonitoringGroundTruthS3Input
    endpoint_input: Optional[EndpointInput] = Unassigned()
    batch_transform_input: Optional[BatchTransformInput] = Unassigned()


class ScheduleConfig(Base):
    """
    ScheduleConfig
      Configuration details about the monitoring schedule.

    Attributes
    ----------------------
    schedule_expression: A cron expression that describes details about the monitoring schedule. The supported cron expressions are:   If you want to set the job to start every hour, use the following:  Hourly: cron(0 \* ? \* \* \*)    If you want to start the job daily:  cron(0 [00-23] ? \* \* \*)    If you want to run the job one time, immediately, use the following keyword:  NOW    For example, the following are valid cron expressions:   Daily at noon UTC: cron(0 12 ? \* \* \*)    Daily at midnight UTC: cron(0 0 ? \* \* \*)    To support running every 6, 12 hours, the following are also supported:  cron(0 [00-23]/[01-24] ? \* \* \*)  For example, the following are valid cron expressions:   Every 12 hours, starting at 5pm UTC: cron(0 17/12 ? \* \* \*)    Every two hours starting at midnight: cron(0 0/2 ? \* \* \*)       Even though the cron expression is set to start at 5PM UTC, note that there could be a delay of 0-20 minutes from the actual requested time to run the execution.    We recommend that if you would like a daily schedule, you do not provide this parameter. Amazon SageMaker AI will pick a time for running every day.    You can also specify the keyword NOW to run the monitoring job immediately, one time, without recurring.
    data_analysis_start_time: Sets the start time for a monitoring job window. Express this time as an offset to the times that you schedule your monitoring jobs to run. You schedule monitoring jobs with the ScheduleExpression parameter. Specify this offset in ISO 8601 duration format. For example, if you want to monitor the five hours of data in your dataset that precede the start of each monitoring job, you would specify: "-PT5H". The start time that you specify must not precede the end time that you specify by more than 24 hours. You specify the end time with the DataAnalysisEndTime parameter. If you set ScheduleExpression to NOW, this parameter is required.
    data_analysis_end_time: Sets the end time for a monitoring job window. Express this time as an offset to the times that you schedule your monitoring jobs to run. You schedule monitoring jobs with the ScheduleExpression parameter. Specify this offset in ISO 8601 duration format. For example, if you want to end the window one hour before the start of each monitoring job, you would specify: "-PT1H". The end time that you specify must not follow the start time that you specify by more than 24 hours. You specify the start time with the DataAnalysisStartTime parameter. If you set ScheduleExpression to NOW, this parameter is required.
    """

    schedule_expression: StrPipeVar
    data_analysis_start_time: Optional[StrPipeVar] = Unassigned()
    data_analysis_end_time: Optional[StrPipeVar] = Unassigned()


class MonitoringBaselineConfig(Base):
    """
    MonitoringBaselineConfig
      Configuration for monitoring constraints and monitoring statistics. These baseline resources are compared against the results of the current job from the series of jobs scheduled to collect data periodically.

    Attributes
    ----------------------
    baselining_job_name: The name of the job that performs baselining for the monitoring job.
    constraints_resource: The baseline constraint file in Amazon S3 that the current monitoring job should validated against.
    statistics_resource: The baseline statistics file in Amazon S3 that the current monitoring job should be validated against.
    """

    baselining_job_name: Optional[StrPipeVar] = Unassigned()
    constraints_resource: Optional[MonitoringConstraintsResource] = Unassigned()
    statistics_resource: Optional[MonitoringStatisticsResource] = Unassigned()


class MonitoringInput(Base):
    """
    MonitoringInput
      The inputs for a monitoring job.

    Attributes
    ----------------------
    processing_inputs
    endpoint_input: The endpoint for a monitoring job.
    batch_transform_input: Input object for the batch transform job.
    """

    processing_inputs: Optional[List[ProcessingInput]] = Unassigned()
    endpoint_input: Optional[EndpointInput] = Unassigned()
    batch_transform_input: Optional[BatchTransformInput] = Unassigned()


class MonitoringAppSpecification(Base):
    """
    MonitoringAppSpecification
      Container image configuration object for the monitoring job.

    Attributes
    ----------------------
    image_uri: The container image to be run by the monitoring job.
    container_entrypoint: Specifies the entrypoint for a container used to run the monitoring job.
    container_arguments: An array of arguments for the container used to run the monitoring job.
    record_preprocessor_source_uri: An Amazon S3 URI to a script that is called per row prior to running analysis. It can base64 decode the payload and convert it into a flattened JSON so that the built-in container can use the converted data. Applicable only for the built-in (first party) containers.
    post_analytics_processor_source_uri: An Amazon S3 URI to a script that is called after analysis has been performed. Applicable only for the built-in (first party) containers.
    """

    image_uri: StrPipeVar
    container_entrypoint: Optional[List[StrPipeVar]] = Unassigned()
    container_arguments: Optional[List[StrPipeVar]] = Unassigned()
    record_preprocessor_source_uri: Optional[StrPipeVar] = Unassigned()
    post_analytics_processor_source_uri: Optional[StrPipeVar] = Unassigned()


class NetworkConfig(Base):
    """
    NetworkConfig
      Networking options for a job, such as network traffic encryption between containers, whether to allow inbound and outbound network calls to and from containers, and the VPC subnets and security groups to use for VPC-enabled jobs.

    Attributes
    ----------------------
    enable_inter_container_traffic_encryption: Whether to encrypt all communications between distributed processing jobs. Choose True to encrypt communications. Encryption provides greater security for distributed processing jobs, but the processing might take longer.
    enable_network_isolation: Whether to allow inbound and outbound network calls to and from the containers used for the processing job.
    vpc_config
    """

    enable_inter_container_traffic_encryption: Optional[bool] = Unassigned()
    enable_network_isolation: Optional[bool] = Unassigned()
    vpc_config: Optional[VpcConfig] = Unassigned()


class MonitoringJobDefinition(Base):
    """
    MonitoringJobDefinition
      Defines the monitoring job.

    Attributes
    ----------------------
    baseline_config: Baseline configuration used to validate that the data conforms to the specified constraints and statistics
    monitoring_inputs: The array of inputs for the monitoring job. Currently we support monitoring an Amazon SageMaker AI Endpoint.
    monitoring_output_config: The array of outputs from the monitoring job to be uploaded to Amazon S3.
    monitoring_resources: Identifies the resources, ML compute instances, and ML storage volumes to deploy for a monitoring job. In distributed processing, you specify more than one instance.
    monitoring_app_specification: Configures the monitoring job to run a specified Docker container image.
    stopping_condition: Specifies a time limit for how long the monitoring job is allowed to run.
    environment: Sets the environment variables in the Docker container.
    network_config: Specifies networking options for an monitoring job.
    role_arn: The Amazon Resource Name (ARN) of an IAM role that Amazon SageMaker AI can assume to perform tasks on your behalf.
    """

    monitoring_inputs: List[MonitoringInput]
    monitoring_output_config: MonitoringOutputConfig
    monitoring_resources: MonitoringResources
    monitoring_app_specification: MonitoringAppSpecification
    role_arn: StrPipeVar
    baseline_config: Optional[MonitoringBaselineConfig] = Unassigned()
    stopping_condition: Optional[MonitoringStoppingCondition] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    network_config: Optional[NetworkConfig] = Unassigned()


class MonitoringScheduleConfig(Base):
    """
    MonitoringScheduleConfig
      Configures the monitoring schedule and defines the monitoring job.

    Attributes
    ----------------------
    schedule_config: Configures the monitoring schedule.
    monitoring_job_definition: Defines the monitoring job.
    monitoring_job_definition_name: The name of the monitoring job definition to schedule.
    monitoring_type: The type of the monitoring job definition to schedule.
    """

    schedule_config: Optional[ScheduleConfig] = Unassigned()
    monitoring_job_definition: Optional[MonitoringJobDefinition] = Unassigned()
    monitoring_job_definition_name: Optional[StrPipeVar] = Unassigned()
    monitoring_type: Optional[StrPipeVar] = Unassigned()


class InstanceMetadataServiceConfiguration(Base):
    """
    InstanceMetadataServiceConfiguration
      Information on the IMDS configuration of the notebook instance

    Attributes
    ----------------------
    minimum_instance_metadata_service_version: Indicates the minimum IMDS version that the notebook instance supports. When passed as part of CreateNotebookInstance, if no value is selected, then it defaults to IMDSv1. This means that both IMDSv1 and IMDSv2 are supported. If passed as part of UpdateNotebookInstance, there is no default.
    """

    minimum_instance_metadata_service_version: StrPipeVar


class NotebookInstanceLifecycleHook(Base):
    """
    NotebookInstanceLifecycleHook
      Contains the notebook instance lifecycle configuration script. Each lifecycle configuration script has a limit of 16384 characters. The value of the $PATH environment variable that is available to both scripts is /sbin:bin:/usr/sbin:/usr/bin. View Amazon CloudWatch Logs for notebook instance lifecycle configurations in log group /aws/sagemaker/NotebookInstances in log stream [notebook-instance-name]/[LifecycleConfigHook]. Lifecycle configuration scripts cannot run for longer than 5 minutes. If a script runs for longer than 5 minutes, it fails and the notebook instance is not created or started. For information about notebook instance lifestyle configurations, see Step 2.1: (Optional) Customize a Notebook Instance.

    Attributes
    ----------------------
    content: A base64-encoded string that contains a shell script for a notebook instance lifecycle configuration.
    """

    content: Optional[StrPipeVar] = Unassigned()


class OptimizationModelAccessConfig(Base):
    """
    OptimizationModelAccessConfig
      The access configuration settings for the source ML model for an optimization job, where you can accept the model end-user license agreement (EULA).

    Attributes
    ----------------------
    accept_eula: Specifies agreement to the model end-user license agreement (EULA). The AcceptEula value must be explicitly defined as True in order to accept the EULA that this model requires. You are responsible for reviewing and complying with any applicable license terms and making sure they are acceptable for your use case before downloading or using a model.
    """

    accept_eula: bool


class OptimizationJobModelSourceS3(Base):
    """
    OptimizationJobModelSourceS3
      The Amazon S3 location of a source model to optimize with an optimization job.

    Attributes
    ----------------------
    s3_uri: An Amazon S3 URI that locates a source model to optimize with an optimization job.
    model_access_config: The access configuration settings for the source ML model for an optimization job, where you can accept the model end-user license agreement (EULA).
    """

    s3_uri: Optional[StrPipeVar] = Unassigned()
    model_access_config: Optional[OptimizationModelAccessConfig] = Unassigned()


class OptimizationSageMakerModel(Base):
    """
    OptimizationSageMakerModel

    Attributes
    ----------------------
    model_name
    """

    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()


class OptimizationJobModelSource(Base):
    """
    OptimizationJobModelSource
      The location of the source model to optimize with an optimization job.

    Attributes
    ----------------------
    s3: The Amazon S3 location of a source model to optimize with an optimization job.
    sage_maker_model
    """

    s3: Optional[OptimizationJobModelSourceS3] = Unassigned()
    sage_maker_model: Optional[OptimizationSageMakerModel] = Unassigned()


class ModelQuantizationConfig(Base):
    """
    ModelQuantizationConfig
      Settings for the model quantization technique that's applied by a model optimization job.

    Attributes
    ----------------------
    image: The URI of an LMI DLC in Amazon ECR. SageMaker uses this image to run the optimization.
    override_environment: Environment variables that override the default ones in the model container.
    """

    image: Optional[StrPipeVar] = Unassigned()
    override_environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class ModelCompilationConfig(Base):
    """
    ModelCompilationConfig
      Settings for the model compilation technique that's applied by a model optimization job.

    Attributes
    ----------------------
    image: The URI of an LMI DLC in Amazon ECR. SageMaker uses this image to run the optimization.
    override_environment: Environment variables that override the default ones in the model container.
    """

    image: Optional[StrPipeVar] = Unassigned()
    override_environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class OptimizationJobDraftModel(Base):
    """
    OptimizationJobDraftModel

    Attributes
    ----------------------
    s3_uri
    model_access_config
    """

    s3_uri: Optional[StrPipeVar] = Unassigned()
    model_access_config: Optional[OptimizationModelAccessConfig] = Unassigned()


class SpeculativeDecodingConfig(Base):
    """
    SpeculativeDecodingConfig

    Attributes
    ----------------------
    draft_model
    """

    draft_model: Optional[OptimizationJobDraftModel] = Unassigned()


class ModelShardingConfig(Base):
    """
    ModelShardingConfig
      Settings for the model sharding technique that's applied by a model optimization job.

    Attributes
    ----------------------
    image: The URI of an LMI DLC in Amazon ECR. SageMaker uses this image to run the optimization.
    override_environment: Environment variables that override the default ones in the model container.
    """

    image: Optional[StrPipeVar] = Unassigned()
    override_environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class ModelSpeculativeDecodingTrainingDataSource(Base):
    """
    ModelSpeculativeDecodingTrainingDataSource

    Attributes
    ----------------------
    s3_uri
    s3_data_type
    """

    s3_uri: StrPipeVar
    s3_data_type: StrPipeVar


class ModelSpeculativeDecodingConfig(Base):
    """
    ModelSpeculativeDecodingConfig

    Attributes
    ----------------------
    technique
    training_data_source
    """

    technique: StrPipeVar
    training_data_source: Optional[ModelSpeculativeDecodingTrainingDataSource] = Unassigned()


class OptimizationConfig(Base):
    """
    OptimizationConfig
      Settings for an optimization technique that you apply with a model optimization job.

    Attributes
    ----------------------
    model_quantization_config: Settings for the model quantization technique that's applied by a model optimization job.
    model_compilation_config: Settings for the model compilation technique that's applied by a model optimization job.
    speculative_decoding_config
    model_sharding_config: Settings for the model sharding technique that's applied by a model optimization job.
    model_speculative_decoding_config
    """

    model_quantization_config: Optional[ModelQuantizationConfig] = Unassigned()
    model_compilation_config: Optional[ModelCompilationConfig] = Unassigned()
    speculative_decoding_config: Optional[SpeculativeDecodingConfig] = Unassigned()
    model_sharding_config: Optional[ModelShardingConfig] = Unassigned()
    model_speculative_decoding_config: Optional[ModelSpeculativeDecodingConfig] = Unassigned()


class OptimizationJobOutputConfig(Base):
    """
    OptimizationJobOutputConfig
      Details for where to store the optimized model that you create with the optimization job.

    Attributes
    ----------------------
    kms_key_id: The Amazon Resource Name (ARN) of a key in Amazon Web Services KMS. SageMaker uses they key to encrypt the artifacts of the optimized model when SageMaker uploads the model to Amazon S3.
    s3_output_location: The Amazon S3 URI for where to store the optimized model that you create with an optimization job.
    sage_maker_model
    """

    s3_output_location: StrPipeVar
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    sage_maker_model: Optional[OptimizationSageMakerModel] = Unassigned()


class OptimizationVpcConfig(Base):
    """
    OptimizationVpcConfig
      A VPC in Amazon VPC that's accessible to an optimized that you create with an optimization job. You can control access to and from your resources by configuring a VPC. For more information, see Give SageMaker Access to Resources in your Amazon VPC.

    Attributes
    ----------------------
    security_group_ids: The VPC security group IDs, in the form sg-xxxxxxxx. Specify the security groups for the VPC that is specified in the Subnets field.
    subnets: The ID of the subnets in the VPC to which you want to connect your optimized model.
    """

    security_group_ids: List[StrPipeVar]
    subnets: List[StrPipeVar]


class PartnerAppMaintenanceConfig(Base):
    """
    PartnerAppMaintenanceConfig
      Maintenance configuration settings for the SageMaker Partner AI App.

    Attributes
    ----------------------
    maintenance_window_start: The day and time of the week in Coordinated Universal Time (UTC) 24-hour standard time that weekly maintenance updates are scheduled. This value must take the following format: 3-letter-day:24-h-hour:minute. For example: TUE:03:30.
    """

    maintenance_window_start: Optional[StrPipeVar] = Unassigned()


class RoleGroupAssignment(Base):
    """
    RoleGroupAssignment
      Defines the mapping between an in-app role and the Amazon Web Services IAM Identity Center group patterns that should be assigned to that role within the SageMaker Partner AI App.

    Attributes
    ----------------------
    role_name: The name of the in-app role within the SageMaker Partner AI App. The specific roles available depend on the app type and version.
    group_patterns: A list of Amazon Web Services IAM Identity Center group patterns that should be assigned to the specified role. Group patterns support wildcard matching using \*.
    """

    role_name: StrPipeVar
    group_patterns: List[StrPipeVar]


class PartnerAppConfig(Base):
    """
    PartnerAppConfig
      Configuration settings for the SageMaker Partner AI App.

    Attributes
    ----------------------
    admin_users: The list of users that are given admin access to the SageMaker Partner AI App.
    arguments: This is a map of required inputs for a SageMaker Partner AI App. Based on the application type, the map is populated with a key and value pair that is specific to the user and application.
    assigned_group_patterns: A list of Amazon Web Services IAM Identity Center group patterns that can access the SageMaker Partner AI App. Group names support wildcard matching using \*. An empty list indicates the app will not use Identity Center group features. All groups specified in RoleGroupAssignments must match patterns in this list.
    role_group_assignments: A map of in-app roles to Amazon Web Services IAM Identity Center group patterns. Groups assigned to specific roles receive those permissions, while groups in AssignedGroupPatterns but not in this map receive default in-app role depending on app type. Group patterns support wildcard matching using \*. Currently supported by Fiddler version 1.3 and later with roles: ORG_MEMBER (default) and ORG_ADMIN.
    """

    admin_users: Optional[List[StrPipeVar]] = Unassigned()
    arguments: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    assigned_group_patterns: Optional[List[StrPipeVar]] = Unassigned()
    role_group_assignments: Optional[List[RoleGroupAssignment]] = Unassigned()


class PersistentVolumeConfiguration(Base):
    """
    PersistentVolumeConfiguration

    Attributes
    ----------------------
    size_in_gb
    """

    size_in_gb: Optional[int] = Unassigned()


class PipelineDefinitionS3Location(Base):
    """
    PipelineDefinitionS3Location
      The location of the pipeline definition stored in Amazon S3.

    Attributes
    ----------------------
    bucket: Name of the S3 bucket.
    object_key: The object key (or key name) uniquely identifies the object in an S3 bucket.
    version_id: Version Id of the pipeline definition file. If not specified, Amazon SageMaker will retrieve the latest version.
    """

    bucket: StrPipeVar
    object_key: StrPipeVar
    version_id: Optional[StrPipeVar] = Unassigned()


class ParallelismConfiguration(Base):
    """
    ParallelismConfiguration
      Configuration that controls the parallelism of the pipeline. By default, the parallelism configuration specified applies to all executions of the pipeline unless overridden.

    Attributes
    ----------------------
    max_parallel_execution_steps: The max number of steps that can be executed in parallel.
    """

    max_parallel_execution_steps: int


class ProcessingS3InputInternal(Base):
    """
    ProcessingS3InputInternal

    Attributes
    ----------------------
    s3_uri
    local_path
    s3_data_type
    s3_input_mode
    s3_download_mode
    s3_data_distribution_type
    s3_compression_type
    """

    s3_uri: StrPipeVar
    s3_data_type: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()
    s3_input_mode: Optional[StrPipeVar] = Unassigned()
    s3_download_mode: Optional[StrPipeVar] = Unassigned()
    s3_data_distribution_type: Optional[StrPipeVar] = Unassigned()
    s3_compression_type: Optional[StrPipeVar] = Unassigned()


class ProcessingInputInternal(Base):
    """
    ProcessingInputInternal

    Attributes
    ----------------------
    input_name
    app_managed
    s3_input
    dataset_definition
    """

    input_name: Optional[StrPipeVar] = Unassigned()
    app_managed: Optional[bool] = Unassigned()
    s3_input: Optional[ProcessingS3InputInternal] = Unassigned()
    dataset_definition: Optional[DatasetDefinition] = Unassigned()


class ProcessingS3Output(Base):
    """
    ProcessingS3Output
      Configuration for uploading output data to Amazon S3 from the processing container.

    Attributes
    ----------------------
    s3_uri: A URI that identifies the Amazon S3 bucket where you want Amazon SageMaker to save the results of a processing job.
    local_path: The local path of a directory where you want Amazon SageMaker to upload its contents to Amazon S3. LocalPath is an absolute path to a directory containing output files. This directory will be created by the platform and exist when your container's entrypoint is invoked.
    s3_upload_mode: Whether to upload the results of the processing job continuously or after the job completes.
    """

    s3_uri: StrPipeVar
    s3_upload_mode: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()


class ProcessingFeatureStoreOutput(Base):
    """
    ProcessingFeatureStoreOutput
      Configuration for processing job outputs in Amazon SageMaker Feature Store.

    Attributes
    ----------------------
    feature_group_name: The name of the Amazon SageMaker FeatureGroup to use as the destination for processing job output. Note that your processing script is responsible for putting records into your Feature Store.
    """

    feature_group_name: Union[StrPipeVar, object]


class ProcessingOutput(Base):
    """
    ProcessingOutput
      Describes the results of a processing job. The processing output must specify exactly one of either S3Output or FeatureStoreOutput types.

    Attributes
    ----------------------
    output_name: The name for the processing job output.
    s3_output: Configuration for processing job outputs in Amazon S3.
    feature_store_output: Configuration for processing job outputs in Amazon SageMaker Feature Store. This processing output type is only supported when AppManaged is specified.
    app_managed: When True, output operations such as data upload are managed natively by the processing job application. When False (default), output operations are managed by Amazon SageMaker.
    """

    output_name: StrPipeVar
    s3_output: Optional[ProcessingS3Output] = Unassigned()
    feature_store_output: Optional[ProcessingFeatureStoreOutput] = Unassigned()
    app_managed: Optional[bool] = Unassigned()


class ProcessingOutputConfig(Base):
    """
    ProcessingOutputConfig
      Configuration for uploading output from the processing container.

    Attributes
    ----------------------
    outputs: An array of outputs configuring the data to upload from the processing container.
    kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt the processing job output. KmsKeyId can be an ID of a KMS key, ARN of a KMS key, or alias of a KMS key. The KmsKeyId is applied to all outputs.
    """

    outputs: List[ProcessingOutput]
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class ProcessingClusterConfig(Base):
    """
    ProcessingClusterConfig
      Configuration for the cluster used to run a processing job.

    Attributes
    ----------------------
    instance_count: The number of ML compute instances to use in the processing job. For distributed processing jobs, specify a value greater than 1. The default value is 1.
    instance_type: The ML compute instance type for the processing job.
    volume_size_in_gb: The size of the ML storage volume in gigabytes that you want to provision. You must specify sufficient ML storage for your scenario.  Certain Nitro-based instances include local storage with a fixed total size, dependent on the instance type. When using these instances for processing, Amazon SageMaker mounts the local instance storage instead of Amazon EBS gp2 storage. You can't request a VolumeSizeInGB greater than the total size of the local instance storage. For a list of instance types that support local instance storage, including the total size per instance type, see Instance Store Volumes.
    volume_kms_key_id: The Amazon Web Services Key Management Service (Amazon Web Services KMS) key that Amazon SageMaker uses to encrypt data on the storage volume attached to the ML compute instance(s) that run the processing job.   Certain Nitro-based instances include local storage, dependent on the instance type. Local storage volumes are encrypted using a hardware module on the instance. You can't request a VolumeKmsKeyId when using an instance type with local storage. For a list of instance types that support local instance storage, see Instance Store Volumes. For more information about local instance storage encryption, see SSD Instance Store Volumes.
    """

    instance_count: int
    instance_type: StrPipeVar
    volume_size_in_gb: int
    volume_kms_key_id: Optional[StrPipeVar] = Unassigned()


class ProcessingResources(Base):
    """
    ProcessingResources
      Identifies the resources, ML compute instances, and ML storage volumes to deploy for a processing job. In distributed training, you specify more than one instance.

    Attributes
    ----------------------
    cluster_config: The configuration for the resources in a cluster used to run the processing job.
    """

    cluster_config: ProcessingClusterConfig


class ProcessingStoppingCondition(Base):
    """
    ProcessingStoppingCondition
      Configures conditions under which the processing job should be stopped, such as how long the processing job has been running. After the condition is met, the processing job is stopped.

    Attributes
    ----------------------
    max_runtime_in_seconds: Specifies the maximum runtime in seconds.
    """

    max_runtime_in_seconds: int


class ProcessingUpstreamS3Output(Base):
    """
    ProcessingUpstreamS3Output

    Attributes
    ----------------------
    s3_uri
    local_path
    s3_upload_mode
    role_arn
    """

    s3_uri: StrPipeVar
    local_path: StrPipeVar
    s3_upload_mode: StrPipeVar
    role_arn: Optional[StrPipeVar] = Unassigned()


class UpstreamProcessingOutput(Base):
    """
    UpstreamProcessingOutput

    Attributes
    ----------------------
    output_name
    upstream_s3_output
    """

    output_name: StrPipeVar
    upstream_s3_output: ProcessingUpstreamS3Output


class UpstreamProcessingOutputConfig(Base):
    """
    UpstreamProcessingOutputConfig

    Attributes
    ----------------------
    outputs
    kms_key_id
    """

    outputs: List[UpstreamProcessingOutput]
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class ExperimentConfig(Base):
    """
    ExperimentConfig
      Associates a SageMaker job as a trial component with an experiment and trial. Specified when you call the following APIs:    CreateProcessingJob     CreateTrainingJob     CreateTransformJob

    Attributes
    ----------------------
    experiment_name: The name of an existing experiment to associate with the trial component.
    trial_name: The name of an existing trial to associate the trial component with. If not specified, a new trial is created.
    trial_component_display_name: The display name for the trial component. If this key isn't specified, the display name is the trial component name.
    run_name: The name of the experiment run to associate with the trial component.
    """

    experiment_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    trial_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    trial_component_display_name: Optional[StrPipeVar] = Unassigned()
    run_name: Optional[StrPipeVar] = Unassigned()


class ProvisioningParameter(Base):
    """
    ProvisioningParameter
      A key value pair used when you provision a project as a service catalog product. For information, see What is Amazon Web Services Service Catalog.

    Attributes
    ----------------------
    key: The key that identifies a provisioning parameter.
    value: The value of the provisioning parameter.
    """

    key: Optional[StrPipeVar] = Unassigned()
    value: Optional[StrPipeVar] = Unassigned()


class ServiceCatalogProvisioningDetails(Base):
    """
    ServiceCatalogProvisioningDetails
      Details that you specify to provision a service catalog product. For information about service catalog, see What is Amazon Web Services Service Catalog.

    Attributes
    ----------------------
    product_id: The ID of the product to provision.
    provisioning_artifact_id: The ID of the provisioning artifact.
    path_id: The path identifier of the product. This value is optional if the product has a default path, and required if the product has more than one path.
    provisioning_parameters: A list of key value pairs that you specify when you provision a product.
    """

    product_id: StrPipeVar
    provisioning_artifact_id: Optional[StrPipeVar] = Unassigned()
    path_id: Optional[StrPipeVar] = Unassigned()
    provisioning_parameters: Optional[List[ProvisioningParameter]] = Unassigned()


class CreateTemplateProvider(Base):
    """
    CreateTemplateProvider
       Contains configuration details for a template provider. Only one type of template provider can be specified.

    Attributes
    ----------------------
    cfn_template_provider:  The CloudFormation template provider configuration for creating infrastructure resources.
    """

    cfn_template_provider: Optional[CfnCreateTemplateProvider] = Unassigned()


class QuotaResourceConfig(Base):
    """
    QuotaResourceConfig

    Attributes
    ----------------------
    instance_type
    count
    """

    instance_type: Optional[StrPipeVar] = Unassigned()
    count: Optional[int] = Unassigned()


class OverQuota(Base):
    """
    OverQuota

    Attributes
    ----------------------
    allow_over_quota
    use_dedicated_capacity
    fair_share_weight
    burst_limit
    """

    allow_over_quota: Optional[bool] = Unassigned()
    use_dedicated_capacity: Optional[bool] = Unassigned()
    fair_share_weight: Optional[int] = Unassigned()
    burst_limit: Optional[BurstLimit] = Unassigned()


class QuotaAllocationTarget(Base):
    """
    QuotaAllocationTarget

    Attributes
    ----------------------
    id
    type
    roles
    """

    id: Optional[StrPipeVar] = Unassigned()
    type: Optional[StrPipeVar] = Unassigned()
    roles: Optional[List[StrPipeVar]] = Unassigned()


class PreemptionConfig(Base):
    """
    PreemptionConfig

    Attributes
    ----------------------
    allow_same_team_preemption
    """

    allow_same_team_preemption: bool


class SpaceIdleSettings(Base):
    """
    SpaceIdleSettings
      Settings related to idle shutdown of Studio applications in a space.

    Attributes
    ----------------------
    idle_timeout_in_minutes: The time that SageMaker waits after the application becomes idle before shutting it down.
    """

    idle_timeout_in_minutes: Optional[int] = Unassigned()


class SpaceAppLifecycleManagement(Base):
    """
    SpaceAppLifecycleManagement
      Settings that are used to configure and manage the lifecycle of Amazon SageMaker Studio applications in a space.

    Attributes
    ----------------------
    idle_settings: Settings related to idle shutdown of Studio applications.
    """

    idle_settings: Optional[SpaceIdleSettings] = Unassigned()


class SpaceCodeEditorAppSettings(Base):
    """
    SpaceCodeEditorAppSettings
      The application settings for a Code Editor space.

    Attributes
    ----------------------
    default_resource_spec
    app_lifecycle_management: Settings that are used to configure and manage the lifecycle of CodeEditor applications in a space.
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    app_lifecycle_management: Optional[SpaceAppLifecycleManagement] = Unassigned()


class SpaceJupyterLabAppSettings(Base):
    """
    SpaceJupyterLabAppSettings
      The settings for the JupyterLab application within a space.

    Attributes
    ----------------------
    default_resource_spec
    code_repositories: A list of Git repositories that SageMaker automatically displays to users for cloning in the JupyterLab application.
    app_lifecycle_management: Settings that are used to configure and manage the lifecycle of JupyterLab applications in a space.
    """

    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    code_repositories: Optional[List[CodeRepository]] = Unassigned()
    app_lifecycle_management: Optional[SpaceAppLifecycleManagement] = Unassigned()


class EbsStorageSettings(Base):
    """
    EbsStorageSettings
      A collection of EBS storage settings that apply to both private and shared spaces.

    Attributes
    ----------------------
    ebs_volume_size_in_gb: The size of an EBS storage volume for a space.
    """

    ebs_volume_size_in_gb: int


class SpaceStorageSettings(Base):
    """
    SpaceStorageSettings
      The storage settings for a space.

    Attributes
    ----------------------
    ebs_storage_settings: A collection of EBS storage settings for a space.
    """

    ebs_storage_settings: Optional[EbsStorageSettings] = Unassigned()


class EFSFileSystem(Base):
    """
    EFSFileSystem
      A file system, created by you in Amazon EFS, that you assign to a user profile or space for an Amazon SageMaker AI Domain. Permitted users can access this file system in Amazon SageMaker AI Studio.

    Attributes
    ----------------------
    file_system_id: The ID of your Amazon EFS file system.
    """

    file_system_id: StrPipeVar


class FSxLustreFileSystem(Base):
    """
    FSxLustreFileSystem
      A custom file system in Amazon FSx for Lustre.

    Attributes
    ----------------------
    file_system_id: Amazon FSx for Lustre file system ID.
    """

    file_system_id: StrPipeVar


class S3FileSystem(Base):
    """
    S3FileSystem
      A custom file system in Amazon S3. This is only supported in Amazon SageMaker Unified Studio.

    Attributes
    ----------------------
    s3_uri: The Amazon S3 URI that specifies the location in S3 where files are stored, which is mounted within the Studio environment. For example: s3://&lt;bucket-name&gt;/&lt;prefix&gt;/.
    """

    s3_uri: StrPipeVar


class CustomFileSystem(Base):
    """
    CustomFileSystem
      A file system, created by you, that you assign to a user profile or space for an Amazon SageMaker AI Domain. Permitted users can access this file system in Amazon SageMaker AI Studio.

    Attributes
    ----------------------
    efs_file_system: A custom file system in Amazon EFS.
    f_sx_lustre_file_system: A custom file system in Amazon FSx for Lustre.
    s3_file_system: A custom file system in Amazon S3. This is only supported in Amazon SageMaker Unified Studio.
    """

    efs_file_system: Optional[EFSFileSystem] = Unassigned()
    f_sx_lustre_file_system: Optional[FSxLustreFileSystem] = Unassigned()
    s3_file_system: Optional[S3FileSystem] = Unassigned()


class SpaceSettings(Base):
    """
    SpaceSettings
      A collection of space settings.

    Attributes
    ----------------------
    jupyter_server_app_settings
    kernel_gateway_app_settings
    vs_code_app_settings
    savitur_app_settings
    code_editor_app_settings: The Code Editor application settings.
    jupyter_lab_app_settings: The settings for the JupyterLab application.
    app_type: The type of app created within the space. If using the  UpdateSpace API, you can't change the app type of your space by specifying a different value for this field.
    space_storage_settings: The storage settings for a space.
    space_managed_resources: If you enable this option, SageMaker AI creates the following resources on your behalf when you create the space:   The user profile that possesses the space.   The app that the space contains.
    custom_file_systems: A file system, created by you, that you assign to a space for an Amazon SageMaker AI Domain. Permitted users can access this file system in Amazon SageMaker AI Studio.
    remote_access: A setting that enables or disables remote access for a SageMaker space. When enabled, this allows you to connect to the remote space from your local IDE.
    """

    jupyter_server_app_settings: Optional[JupyterServerAppSettings] = Unassigned()
    kernel_gateway_app_settings: Optional[KernelGatewayAppSettings] = Unassigned()
    vs_code_app_settings: Optional[VSCodeAppSettings] = Unassigned()
    savitur_app_settings: Optional[SaviturAppSettings] = Unassigned()
    code_editor_app_settings: Optional[SpaceCodeEditorAppSettings] = Unassigned()
    jupyter_lab_app_settings: Optional[SpaceJupyterLabAppSettings] = Unassigned()
    app_type: Optional[StrPipeVar] = Unassigned()
    space_storage_settings: Optional[SpaceStorageSettings] = Unassigned()
    space_managed_resources: Optional[StrPipeVar] = Unassigned()
    custom_file_systems: Optional[List[CustomFileSystem]] = Unassigned()
    remote_access: Optional[StrPipeVar] = Unassigned()


class OwnershipSettings(Base):
    """
    OwnershipSettings
      The collection of ownership settings for a space.

    Attributes
    ----------------------
    owner_user_profile_name: The user profile who is the owner of the space.
    """

    owner_user_profile_name: StrPipeVar


class SpaceSharingSettings(Base):
    """
    SpaceSharingSettings
      A collection of space sharing settings.

    Attributes
    ----------------------
    sharing_type: Specifies the sharing type of the space.
    """

    sharing_type: StrPipeVar


class ResourceTags(Base):
    """
    ResourceTags

    Attributes
    ----------------------
    network_interface_tags
    """

    network_interface_tags: Optional[List[Tag]] = Unassigned()


class ProcessingOutputTraining(Base):
    """
    ProcessingOutputTraining

    Attributes
    ----------------------
    output_name
    s3_output
    feature_store_output
    app_managed
    """

    output_name: StrPipeVar
    s3_output: Optional[ProcessingS3Output] = Unassigned()
    feature_store_output: Optional[ProcessingFeatureStoreOutput] = Unassigned()
    app_managed: Optional[bool] = Unassigned()


class ProcessingOutputConfigTraining(Base):
    """
    ProcessingOutputConfigTraining

    Attributes
    ----------------------
    outputs
    kms_key_id
    """

    outputs: List[ProcessingOutputTraining]
    kms_key_id: Optional[StrPipeVar] = Unassigned()


class ProcessingResult(Base):
    """
    ProcessingResult

    Attributes
    ----------------------
    exit_message
    internal_failure_reason
    fault_entity
    payer
    """

    exit_message: Optional[StrPipeVar] = Unassigned()
    internal_failure_reason: Optional[StrPipeVar] = Unassigned()
    fault_entity: Optional[StrPipeVar] = Unassigned()
    payer: Optional[StrPipeVar] = Unassigned()


class ProcessingUpstreamSvcConfig(Base):
    """
    ProcessingUpstreamSvcConfig
      Populated only for a Processing Job running in Training platform. Has fields to represent the Upstream Service Resource ARNs for a Processing Job. (Upstream to a Processing Job). These fields are used to determine the sourceArn and sourceAccount headers to be used for assume-role service calls to prevent confused deputy attacks

    Attributes
    ----------------------
    auto_ml_job_arn
    monitoring_schedule_arn
    training_job_arn
    """

    auto_ml_job_arn: Optional[StrPipeVar] = Unassigned()
    monitoring_schedule_arn: Optional[StrPipeVar] = Unassigned()
    training_job_arn: Optional[StrPipeVar] = Unassigned()


class ProcessingJobConfig(Base):
    """
    ProcessingJobConfig

    Attributes
    ----------------------
    processing_inputs
    processing_output_config
    upstream_processing_output_config
    processing_result
    processing_upstream_svc_config
    """

    processing_inputs: Optional[List[ProcessingInputInternal]] = Unassigned()
    processing_output_config: Optional[ProcessingOutputConfigTraining] = Unassigned()
    upstream_processing_output_config: Optional[UpstreamProcessingOutputConfig] = Unassigned()
    processing_result: Optional[ProcessingResult] = Unassigned()
    processing_upstream_svc_config: Optional[ProcessingUpstreamSvcConfig] = Unassigned()


class CredentialProxyConfig(Base):
    """
    CredentialProxyConfig

    Attributes
    ----------------------
    platform_credential_token
    customer_credential_token
    credential_provider_function
    platform_credential_provider_function
    customer_credential_provider_encryption_key
    platform_credential_provider_encryption_key
    customer_credential_provider_kms_key_id
    platform_credential_provider_kms_key_id
    """

    customer_credential_token: StrPipeVar
    credential_provider_function: StrPipeVar
    platform_credential_token: Optional[StrPipeVar] = Unassigned()
    platform_credential_provider_function: Optional[StrPipeVar] = Unassigned()
    customer_credential_provider_encryption_key: Optional[StrPipeVar] = Unassigned()
    platform_credential_provider_encryption_key: Optional[StrPipeVar] = Unassigned()
    customer_credential_provider_kms_key_id: Optional[StrPipeVar] = Unassigned()
    platform_credential_provider_kms_key_id: Optional[StrPipeVar] = Unassigned()


class LogRoutingConfig(Base):
    """
    LogRoutingConfig

    Attributes
    ----------------------
    log_group
    log_stream_prefix
    metrics_namespace
    metrics_host_dimension_value
    """

    log_group: Optional[StrPipeVar] = Unassigned()
    log_stream_prefix: Optional[StrPipeVar] = Unassigned()
    metrics_namespace: Optional[StrPipeVar] = Unassigned()
    metrics_host_dimension_value: Optional[StrPipeVar] = Unassigned()


class UpstreamPlatformOutputDataConfig(Base):
    """
    UpstreamPlatformOutputDataConfig

    Attributes
    ----------------------
    kms_key_id
    kms_encryption_context
    channels
    """

    kms_key_id: Optional[StrPipeVar] = Unassigned()
    kms_encryption_context: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    channels: Optional[List[OutputChannel]] = Unassigned()


class UpstreamPlatformConfig(Base):
    """
    UpstreamPlatformConfig

    Attributes
    ----------------------
    credential_proxy_config
    log_routing_config
    vpc_config
    agents_credential_provider
    output_data_config
    checkpoint_config
    upstream_customer_account_id
    upstream_customer_arn
    enable_s3_context_keys_on_input_data
    execution_role
    """

    credential_proxy_config: Optional[CredentialProxyConfig] = Unassigned()
    log_routing_config: Optional[LogRoutingConfig] = Unassigned()
    vpc_config: Optional[VpcConfig] = Unassigned()
    agents_credential_provider: Optional[AgentsCredentialProvider] = Unassigned()
    output_data_config: Optional[UpstreamPlatformOutputDataConfig] = Unassigned()
    checkpoint_config: Optional[CheckpointConfig] = Unassigned()
    upstream_customer_account_id: Optional[StrPipeVar] = Unassigned()
    upstream_customer_arn: Optional[StrPipeVar] = Unassigned()
    enable_s3_context_keys_on_input_data: Optional[bool] = Unassigned()
    execution_role: Optional[StrPipeVar] = Unassigned()


class DebugHookConfig(Base):
    """
    DebugHookConfig
      Configuration information for the Amazon SageMaker Debugger hook parameters, metric and tensor collections, and storage paths. To learn more about how to configure the DebugHookConfig parameter, see Use the SageMaker and Debugger Configuration API Operations to Create, Update, and Debug Your Training Job.

    Attributes
    ----------------------
    local_path: Path to local storage location for metrics and tensors. Defaults to /opt/ml/output/tensors/.
    s3_output_path: Path to Amazon S3 storage location for metrics and tensors.
    hook_parameters: Configuration information for the Amazon SageMaker Debugger hook parameters.
    collection_configurations: Configuration information for Amazon SageMaker Debugger tensor collections. To learn more about how to configure the CollectionConfiguration parameter, see Use the SageMaker and Debugger Configuration API Operations to Create, Update, and Debug Your Training Job.
    """

    s3_output_path: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()
    hook_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    collection_configurations: Optional[List[CollectionConfiguration]] = Unassigned()


class DebugRuleConfiguration(Base):
    """
    DebugRuleConfiguration
      Configuration information for SageMaker Debugger rules for debugging. To learn more about how to configure the DebugRuleConfiguration parameter, see Use the SageMaker and Debugger Configuration API Operations to Create, Update, and Debug Your Training Job.

    Attributes
    ----------------------
    rule_configuration_name: The name of the rule configuration. It must be unique relative to other rule configuration names.
    local_path: Path to local storage location for output of rules. Defaults to /opt/ml/processing/output/rule/.
    s3_output_path: Path to Amazon S3 storage location for rules.
    rule_evaluator_image: The Amazon Elastic Container (ECR) Image for the managed rule evaluation.
    instance_type: The instance type to deploy a custom rule for debugging a training job.
    volume_size_in_gb: The size, in GB, of the ML storage volume attached to the processing instance.
    rule_parameters: Runtime configuration for rule container.
    """

    rule_configuration_name: StrPipeVar
    rule_evaluator_image: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()
    s3_output_path: Optional[StrPipeVar] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    volume_size_in_gb: Optional[int] = Unassigned()
    rule_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class TensorBoardOutputConfig(Base):
    """
    TensorBoardOutputConfig
      Configuration of storage locations for the Amazon SageMaker Debugger TensorBoard output data.

    Attributes
    ----------------------
    local_path: Path to local storage location for tensorBoard output. Defaults to /opt/ml/output/tensorboard.
    s3_output_path: Path to Amazon S3 storage location for TensorBoard output.
    """

    s3_output_path: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()


class ProfilerConfig(Base):
    """
    ProfilerConfig
      Configuration information for Amazon SageMaker Debugger system monitoring, framework profiling, and storage paths.

    Attributes
    ----------------------
    s3_output_path: Path to Amazon S3 storage location for system and framework metrics.
    profiling_interval_in_milliseconds: A time interval for capturing system metrics in milliseconds. Available values are 100, 200, 500, 1000 (1 second), 5000 (5 seconds), and 60000 (1 minute) milliseconds. The default value is 500 milliseconds.
    profiling_parameters: Configuration information for capturing framework metrics. Available key strings for different profiling options are DetailedProfilingConfig, PythonProfilingConfig, and DataLoaderProfilingConfig. The following codes are configuration structures for the ProfilingParameters parameter. To learn more about how to configure the ProfilingParameters parameter, see Use the SageMaker and Debugger Configuration API Operations to Create, Update, and Debug Your Training Job.
    disable_profiler: Configuration to turn off Amazon SageMaker Debugger's system monitoring and profiling functionality. To turn it off, set to True.
    """

    s3_output_path: Optional[StrPipeVar] = Unassigned()
    profiling_interval_in_milliseconds: Optional[int] = Unassigned()
    profiling_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    disable_profiler: Optional[bool] = Unassigned()


class ProfilerRuleConfiguration(Base):
    """
    ProfilerRuleConfiguration
      Configuration information for profiling rules.

    Attributes
    ----------------------
    rule_configuration_name: The name of the rule configuration. It must be unique relative to other rule configuration names.
    local_path: Path to local storage location for output of rules. Defaults to /opt/ml/processing/output/rule/.
    s3_output_path: Path to Amazon S3 storage location for rules.
    rule_evaluator_image: The Amazon Elastic Container Registry Image for the managed rule evaluation.
    instance_type: The instance type to deploy a custom rule for profiling a training job.
    volume_size_in_gb: The size, in GB, of the ML storage volume attached to the processing instance.
    rule_parameters: Runtime configuration for rule container.
    """

    rule_configuration_name: StrPipeVar
    rule_evaluator_image: StrPipeVar
    local_path: Optional[StrPipeVar] = Unassigned()
    s3_output_path: Optional[StrPipeVar] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    volume_size_in_gb: Optional[int] = Unassigned()
    rule_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class RemoteDebugConfig(Base):
    """
    RemoteDebugConfig
      Configuration for remote debugging for the CreateTrainingJob API. To learn more about the remote debugging functionality of SageMaker, see Access a training container through Amazon Web Services Systems Manager (SSM) for remote debugging.

    Attributes
    ----------------------
    enable_remote_debug: If set to True, enables remote debugging.
    """

    enable_remote_debug: Optional[bool] = Unassigned()


class InfraCheckConfig(Base):
    """
    InfraCheckConfig
      Configuration information for the infrastructure health check of a training job. A SageMaker-provided health check tests the health of instance hardware and cluster network connectivity.

    Attributes
    ----------------------
    enable_infra_check: Enables an infrastructure health check.
    """

    enable_infra_check: Optional[bool] = Unassigned()


class SessionChainingConfig(Base):
    """
    SessionChainingConfig
      Contains information about attribute-based access control (ABAC) for a training job. The session chaining configuration uses Amazon Security Token Service (STS) for your training job to request temporary, limited-privilege credentials to tenants. For more information, see Attribute-based access control (ABAC) for multi-tenancy training.

    Attributes
    ----------------------
    enable_session_tag_chaining: Set to True to allow SageMaker to extract session tags from a training job creation role and reuse these tags when assuming the training job execution role.
    """

    enable_session_tag_chaining: Optional[bool] = Unassigned()


class ServerlessJobConfig(Base):
    """
    ServerlessJobConfig

    Attributes
    ----------------------
    base_model_arn
    accept_eula
    job_type
    customization_technique
    peft
    evaluation_type
    evaluator_arn
    job_spec
    """
    
    base_model_arn: StrPipeVar
    job_type: StrPipeVar
    accept_eula: Optional[bool] = Unassigned()
    customization_technique: Optional[StrPipeVar] = Unassigned()
    peft: Optional[StrPipeVar] = Unassigned()
    evaluation_type: Optional[StrPipeVar] = Unassigned()
    evaluator_arn: Optional[StrPipeVar] = Unassigned()
    job_spec: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class MlflowConfig(Base):
    """
    MlflowConfig

    Attributes
    ----------------------
    mlflow_tracking_server_arn
    mlflow_resource_arn
    mlflow_experiment_name
    mlflow_run_name
    """

    mlflow_resource_arn: StrPipeVar
    mlflow_tracking_server_arn: Optional[StrPipeVar] = Unassigned()
    mlflow_experiment_name: Optional[StrPipeVar] = Unassigned()
    mlflow_run_name: Optional[StrPipeVar] = Unassigned()


class ModelPackageConfig(Base):
    """
    ModelPackageConfig

    Attributes
    ----------------------
    model_package_group_arn
    source_model_package_arn
    """

    model_package_group_arn: StrPipeVar
    source_model_package_arn: Optional[StrPipeVar] = Unassigned()


class ModelClientConfig(Base):
    """
    ModelClientConfig
      Configures the timeout and maximum number of retries for processing a transform job invocation.

    Attributes
    ----------------------
    invocations_timeout_in_seconds: The timeout value in seconds for an invocation request. The default value is 600.
    invocations_max_retries: The maximum number of retries when invocation requests are failing. The default value is 3.
    """

    invocations_timeout_in_seconds: Optional[int] = Unassigned()
    invocations_max_retries: Optional[int] = Unassigned()


class DataProcessing(Base):
    """
    DataProcessing
      The data structure used to specify the data to be used for inference in a batch transform job and to associate the data that is relevant to the prediction results in the output. The input filter provided allows you to exclude input data that is not needed for inference in a batch transform job. The output filter provided allows you to include input data relevant to interpreting the predictions in the output from the job. For more information, see Associate Prediction Results with their Corresponding Input Records.

    Attributes
    ----------------------
    input_filter: A JSONPath expression used to select a portion of the input data to pass to the algorithm. Use the InputFilter parameter to exclude fields, such as an ID column, from the input. If you want SageMaker to pass the entire input dataset to the algorithm, accept the default value $. Examples: "$", "$[1:]", "$.features"
    output_filter: A JSONPath expression used to select a portion of the joined dataset to save in the output file for a batch transform job. If you want SageMaker to store the entire input dataset in the output file, leave the default value, $. If you specify indexes that aren't within the dimension size of the joined dataset, you get an error. Examples: "$", "$[0,5:]", "$['id','SageMakerOutput']"
    join_source: Specifies the source of the data to join with the transformed data. The valid values are None and Input. The default value is None, which specifies not to join the input with the transformed data. If you want the batch transform job to join the original input data with the transformed data, set JoinSource to Input. You can specify OutputFilter as an additional filter to select a portion of the joined dataset and store it in the output file. For JSON or JSONLines objects, such as a JSON array, SageMaker adds the transformed data to the input JSON object in an attribute called SageMakerOutput. The joined result for JSON must be a key-value pair object. If the input is not a key-value pair object, SageMaker creates a new JSON file. In the new JSON file, and the input data is stored under the SageMakerInput key and the results are stored in SageMakerOutput. For CSV data, SageMaker takes each row as a JSON array and joins the transformed data with the input by appending each transformed row to the end of the input. The joined data has the original input data followed by the transformed data and the output is a CSV file. For information on how joining in applied, see Workflow for Associating Inferences with Input Records.
    """

    input_filter: Optional[StrPipeVar] = Unassigned()
    output_filter: Optional[StrPipeVar] = Unassigned()
    join_source: Optional[StrPipeVar] = Unassigned()


class InputTrialComponentSource(Base):
    """
    InputTrialComponentSource

    Attributes
    ----------------------
    source_arn
    """

    source_arn: StrPipeVar


class TrialComponentStatus(Base):
    """
    TrialComponentStatus
      The status of the trial component.

    Attributes
    ----------------------
    primary_status: The status of the trial component.
    message: If the component failed, a message describing why.
    """

    primary_status: Optional[StrPipeVar] = Unassigned()
    message: Optional[StrPipeVar] = Unassigned()


class TrialComponentParameterValue(Base):
    """
    TrialComponentParameterValue
      The value of a hyperparameter. Only one of NumberValue or StringValue can be specified. This object is specified in the CreateTrialComponent request.

    Attributes
    ----------------------
    string_value: The string value of a categorical hyperparameter. If you specify a value for this parameter, you can't specify the NumberValue parameter.
    number_value: The numeric value of a numeric hyperparameter. If you specify a value for this parameter, you can't specify the StringValue parameter.
    """

    string_value: Optional[StrPipeVar] = Unassigned()
    number_value: Optional[float] = Unassigned()


class TrialComponentArtifact(Base):
    """
    TrialComponentArtifact
      Represents an input or output artifact of a trial component. You specify TrialComponentArtifact as part of the InputArtifacts and OutputArtifacts parameters in the CreateTrialComponent request. Examples of input artifacts are datasets, algorithms, hyperparameters, source code, and instance types. Examples of output artifacts are metrics, snapshots, logs, and images.

    Attributes
    ----------------------
    media_type: The media type of the artifact, which indicates the type of data in the artifact file. The media type consists of a type and a subtype concatenated with a slash (/) character, for example, text/csv, image/jpeg, and s3/uri. The type specifies the category of the media. The subtype specifies the kind of data.
    value: The location of the artifact.
    """

    value: StrPipeVar
    media_type: Optional[StrPipeVar] = Unassigned()


class InputTrialSource(Base):
    """
    InputTrialSource

    Attributes
    ----------------------
    source_arn
    """

    source_arn: StrPipeVar


class OidcConfig(Base):
    """
    OidcConfig
      Use this parameter to configure your OIDC Identity Provider (IdP).

    Attributes
    ----------------------
    client_id: The OIDC IdP client ID used to configure your private workforce.
    client_secret: The OIDC IdP client secret used to configure your private workforce.
    issuer: The OIDC IdP issuer used to configure your private workforce.
    authorization_endpoint: The OIDC IdP authorization endpoint used to configure your private workforce.
    token_endpoint: The OIDC IdP token endpoint used to configure your private workforce.
    user_info_endpoint: The OIDC IdP user information endpoint used to configure your private workforce.
    logout_endpoint: The OIDC IdP logout endpoint used to configure your private workforce.
    jwks_uri: The OIDC IdP JSON Web Key Set (Jwks) URI used to configure your private workforce.
    scope: An array of string identifiers used to refer to the specific pieces of user data or claims that the client application wants to access.
    authentication_request_extra_params: A string to string map of identifiers specific to the custom identity provider (IdP) being used.
    """

    client_id: StrPipeVar
    client_secret: StrPipeVar
    issuer: StrPipeVar
    authorization_endpoint: StrPipeVar
    token_endpoint: StrPipeVar
    user_info_endpoint: StrPipeVar
    logout_endpoint: StrPipeVar
    jwks_uri: StrPipeVar
    scope: Optional[StrPipeVar] = Unassigned()
    authentication_request_extra_params: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class SourceIpConfig(Base):
    """
    SourceIpConfig
      A list of IP address ranges (CIDRs). Used to create an allow list of IP addresses for a private workforce. Workers will only be able to log in to their worker portal from an IP address within this range. By default, a workforce isn't restricted to specific IP addresses.

    Attributes
    ----------------------
    cidrs: A list of one to ten Classless Inter-Domain Routing (CIDR) values. Maximum: Ten CIDR values  The following Length Constraints apply to individual CIDR values in the CIDR value list.
    """

    cidrs: List[StrPipeVar]


class WorkforceVpcConfigRequest(Base):
    """
    WorkforceVpcConfigRequest
      The VPC object you use to create or update a workforce.

    Attributes
    ----------------------
    vpc_id: The ID of the VPC that the workforce uses for communication.
    security_group_ids: The VPC security group IDs, in the form sg-xxxxxxxx. The security groups must be for the same VPC as specified in the subnet.
    subnets: The ID of the subnets in the VPC that you want to connect.
    """

    vpc_id: Optional[StrPipeVar] = Unassigned()
    security_group_ids: Optional[List[StrPipeVar]] = Unassigned()
    subnets: Optional[List[StrPipeVar]] = Unassigned()


class OidcMemberDefinition(Base):
    """
    OidcMemberDefinition
      A list of user groups that exist in your OIDC Identity Provider (IdP). One to ten groups can be used to create a single private work team. When you add a user group to the list of Groups, you can add that user group to one or more private work teams. If you add a user group to a private work team, all workers in that user group are added to the work team.

    Attributes
    ----------------------
    groups: A list of comma seperated strings that identifies user groups in your OIDC IdP. Each user group is made up of a group of private workers.
    group
    member_definition_id
    """

    groups: Optional[List[StrPipeVar]] = Unassigned()
    group: Optional[StrPipeVar] = Unassigned()
    member_definition_id: Optional[StrPipeVar] = Unassigned()


class MemberDefinition(Base):
    """
    MemberDefinition
      Defines an Amazon Cognito or your own OIDC IdP user group that is part of a work team.

    Attributes
    ----------------------
    cognito_member_definition: The Amazon Cognito user group that is part of the work team.
    oidc_member_definition: A list user groups that exist in your OIDC Identity Provider (IdP). One to ten groups can be used to create a single private work team. When you add a user group to the list of Groups, you can add that user group to one or more private work teams. If you add a user group to a private work team, all workers in that user group are added to the work team.
    """

    cognito_member_definition: Optional[CognitoMemberDefinition] = Unassigned()
    oidc_member_definition: Optional[OidcMemberDefinition] = Unassigned()


class MembershipRule(Base):
    """
    MembershipRule

    Attributes
    ----------------------
    target_member_definition
    filter_expression
    """

    target_member_definition: Optional[StrPipeVar] = Unassigned()
    filter_expression: Optional[StrPipeVar] = Unassigned()


class NotificationConfiguration(Base):
    """
    NotificationConfiguration
      Configures Amazon SNS notifications of available or expiring work items for work teams.

    Attributes
    ----------------------
    notification_topic_arn: The ARN for the Amazon SNS topic to which notifications should be published.
    """

    notification_topic_arn: Optional[StrPipeVar] = Unassigned()


class IamPolicyConstraints(Base):
    """
    IamPolicyConstraints
      Use this parameter to specify a supported global condition key that is added to the IAM policy.

    Attributes
    ----------------------
    source_ip: When SourceIp is Enabled the worker's IP address when a task is rendered in the worker portal is added to the IAM policy as a Condition used to generate the Amazon S3 presigned URL. This IP address is checked by Amazon S3 and must match in order for the Amazon S3 resource to be rendered in the worker portal.
    vpc_source_ip: When VpcSourceIp is Enabled the worker's IP address when a task is rendered in private worker portal inside the VPC is added to the IAM policy as a Condition used to generate the Amazon S3 presigned URL. To render the task successfully Amazon S3 checks that the presigned URL is being accessed over an Amazon S3 VPC Endpoint, and that the worker's IP address matches the IP address in the IAM policy. To learn more about configuring private worker portal, see Use Amazon VPC mode from a private worker portal.
    """

    source_ip: Optional[StrPipeVar] = Unassigned()
    vpc_source_ip: Optional[StrPipeVar] = Unassigned()


class S3Presign(Base):
    """
    S3Presign
      This object defines the access restrictions to Amazon S3 resources that are included in custom worker task templates using the Liquid filter, grant_read_access. To learn more about how custom templates are created, see Create custom worker task templates.

    Attributes
    ----------------------
    iam_policy_constraints: Use this parameter to specify the allowed request source. Possible sources are either SourceIp or VpcSourceIp.
    """

    iam_policy_constraints: Optional[IamPolicyConstraints] = Unassigned()


class WorkerAccessConfiguration(Base):
    """
    WorkerAccessConfiguration
      Use this optional parameter to constrain access to an Amazon S3 resource based on the IP address using supported IAM global condition keys. The Amazon S3 resource is accessed in the worker portal using a Amazon S3 presigned URL.

    Attributes
    ----------------------
    s3_presign: Defines any Amazon S3 resource constraints.
    """

    s3_presign: Optional[S3Presign] = Unassigned()


class CustomMonitoringJobDefinition(Base):
    """
    CustomMonitoringJobDefinition

    Attributes
    ----------------------
    job_definition_arn
    job_definition_name
    creation_time
    custom_monitoring_app_specification
    custom_monitoring_job_input
    custom_monitoring_job_output_config
    job_resources
    network_config
    role_arn
    stopping_condition
    """

    job_definition_arn: StrPipeVar
    job_definition_name: StrPipeVar
    creation_time: datetime.datetime
    custom_monitoring_app_specification: CustomMonitoringAppSpecification
    custom_monitoring_job_input: CustomMonitoringJobInput
    custom_monitoring_job_output_config: MonitoringOutputConfig
    job_resources: MonitoringResources
    role_arn: StrPipeVar
    network_config: Optional[MonitoringNetworkConfig] = Unassigned()
    stopping_condition: Optional[MonitoringStoppingCondition] = Unassigned()


class CustomizedMetricSpecification(Base):
    """
    CustomizedMetricSpecification
      A customized metric.

    Attributes
    ----------------------
    metric_name: The name of the customized metric.
    namespace: The namespace of the customized metric.
    statistic: The statistic of the customized metric.
    """

    metric_name: Optional[StrPipeVar] = Unassigned()
    namespace: Optional[StrPipeVar] = Unassigned()
    statistic: Optional[StrPipeVar] = Unassigned()


class DataCaptureConfigSummary(Base):
    """
    DataCaptureConfigSummary
      The currently active data capture configuration used by your Endpoint.

    Attributes
    ----------------------
    enable_capture: Whether data capture is enabled or disabled.
    capture_status: Whether data capture is currently functional.
    current_sampling_percentage: The percentage of requests being captured by your Endpoint.
    destination_s3_uri: The Amazon S3 location being used to capture the data.
    kms_key_id: The KMS key being used to encrypt the data in Amazon S3.
    """

    enable_capture: bool
    capture_status: StrPipeVar
    current_sampling_percentage: int
    destination_s3_uri: StrPipeVar
    kms_key_id: StrPipeVar


class DataQualityJobDefinition(Base):
    """
    DataQualityJobDefinition

    Attributes
    ----------------------
    job_definition_arn
    job_definition_name
    creation_time
    data_quality_baseline_config
    data_quality_app_specification
    data_quality_job_input
    data_quality_job_output_config
    job_resources
    network_config
    role_arn
    stopping_condition
    """

    job_definition_arn: StrPipeVar
    job_definition_name: StrPipeVar
    creation_time: datetime.datetime
    data_quality_app_specification: DataQualityAppSpecification
    data_quality_job_input: DataQualityJobInput
    data_quality_job_output_config: MonitoringOutputConfig
    job_resources: MonitoringResources
    role_arn: StrPipeVar
    data_quality_baseline_config: Optional[DataQualityBaselineConfig] = Unassigned()
    network_config: Optional[MonitoringNetworkConfig] = Unassigned()
    stopping_condition: Optional[MonitoringStoppingCondition] = Unassigned()


class DebugRuleEvaluationStatus(Base):
    """
    DebugRuleEvaluationStatus
      Information about the status of the rule evaluation.

    Attributes
    ----------------------
    rule_configuration_name: The name of the rule configuration.
    rule_evaluation_job_arn: The Amazon Resource Name (ARN) of the rule evaluation job.
    rule_evaluation_status: Status of the rule evaluation.
    status_details: Details from the rule evaluation.
    last_modified_time: Timestamp when the rule evaluation status was last modified.
    """

    rule_configuration_name: Optional[StrPipeVar] = Unassigned()
    rule_evaluation_job_arn: Optional[StrPipeVar] = Unassigned()
    rule_evaluation_status: Optional[StrPipeVar] = Unassigned()
    status_details: Optional[StrPipeVar] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class RetentionPolicy(Base):
    """
    RetentionPolicy
      The retention policy for data stored on an Amazon Elastic File System volume.

    Attributes
    ----------------------
    home_efs_file_system: The default is Retain, which specifies to keep the data stored on the Amazon EFS volume. Specify Delete to delete the data stored on the Amazon EFS volume.
    """

    home_efs_file_system: Optional[StrPipeVar] = Unassigned()


class DeployedImage(Base):
    """
    DeployedImage
      Gets the Amazon EC2 Container Registry path of the docker image of the model that is hosted in this ProductionVariant. If you used the registry/repository[:tag] form to specify the image path of the primary container when you created the model hosted in this ProductionVariant, the path resolves to a path of the form registry/repository[@digest]. A digest is a hash value that identifies a specific version of an image. For information about Amazon ECR paths, see Pulling an Image in the Amazon ECR User Guide.

    Attributes
    ----------------------
    specified_image: The image path you specified when you created the model.
    resolved_image: The specific digest path of the image hosted in this ProductionVariant.
    resolution_time: The date and time when the image path for the model resolved to the ResolvedImage
    """

    specified_image: Optional[StrPipeVar] = Unassigned()
    resolved_image: Optional[StrPipeVar] = Unassigned()
    resolution_time: Optional[datetime.datetime] = Unassigned()


class RealTimeInferenceRecommendation(Base):
    """
    RealTimeInferenceRecommendation
      The recommended configuration to use for Real-Time Inference.

    Attributes
    ----------------------
    recommendation_id: The recommendation ID which uniquely identifies each recommendation.
    instance_type: The recommended instance type for Real-Time Inference.
    environment: The recommended environment variables to set in the model container for Real-Time Inference.
    """

    recommendation_id: StrPipeVar
    instance_type: StrPipeVar
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class DeploymentRecommendation(Base):
    """
    DeploymentRecommendation
      A set of recommended deployment configurations for the model. To get more advanced recommendations, see CreateInferenceRecommendationsJob to create an inference recommendation job.

    Attributes
    ----------------------
    recommendation_status: Status of the deployment recommendation. The status NOT_APPLICABLE means that SageMaker is unable to provide a default recommendation for the model using the information provided. If the deployment status is IN_PROGRESS, retry your API call after a few seconds to get a COMPLETED deployment recommendation.
    real_time_inference_recommendations: A list of RealTimeInferenceRecommendation items.
    """

    recommendation_status: StrPipeVar
    real_time_inference_recommendations: Optional[List[RealTimeInferenceRecommendation]] = (
        Unassigned()
    )


class EdgeDeploymentStatus(Base):
    """
    EdgeDeploymentStatus
      Contains information summarizing the deployment stage results.

    Attributes
    ----------------------
    stage_status: The general status of the current stage.
    edge_deployment_success_in_stage: The number of edge devices with the successful deployment in the current stage.
    edge_deployment_pending_in_stage: The number of edge devices yet to pick up the deployment in current stage, or in progress.
    edge_deployment_failed_in_stage: The number of edge devices that failed the deployment in current stage.
    edge_deployment_status_message: A detailed message about deployment status in current stage.
    edge_deployment_stage_start_time: The time when the deployment API started.
    """

    stage_status: StrPipeVar
    edge_deployment_success_in_stage: int
    edge_deployment_pending_in_stage: int
    edge_deployment_failed_in_stage: int
    edge_deployment_status_message: Optional[StrPipeVar] = Unassigned()
    edge_deployment_stage_start_time: Optional[datetime.datetime] = Unassigned()


class DeploymentStageStatusSummary(Base):
    """
    DeploymentStageStatusSummary
      Contains information summarizing the deployment stage results.

    Attributes
    ----------------------
    stage_name: The name of the stage.
    device_selection_config: Configuration of the devices in the stage.
    deployment_config: Configuration of the deployment details.
    deployment_status: General status of the current state.
    """

    stage_name: StrPipeVar
    device_selection_config: DeviceSelectionConfig
    deployment_config: EdgeDeploymentConfig
    deployment_status: EdgeDeploymentStatus


class DerivedInformation(Base):
    """
    DerivedInformation
      Information that SageMaker Neo automatically derived about the model.

    Attributes
    ----------------------
    derived_data_input_config: The data input configuration that SageMaker Neo automatically derived for the model. When SageMaker Neo derives this information, you don't need to specify the data input configuration when you create a compilation job.
    derived_framework
    derived_framework_version
    """

    derived_data_input_config: Optional[StrPipeVar] = Unassigned()
    derived_framework: Optional[StrPipeVar] = Unassigned()
    derived_framework_version: Optional[StrPipeVar] = Unassigned()


class ResolvedAttributes(Base):
    """
    ResolvedAttributes
      The resolved attributes.

    Attributes
    ----------------------
    auto_ml_job_objective
    problem_type: The problem type.
    completion_criteria
    """

    auto_ml_job_objective: Optional[AutoMLJobObjective] = Unassigned()
    problem_type: Optional[StrPipeVar] = Unassigned()
    completion_criteria: Optional[AutoMLJobCompletionCriteria] = Unassigned()


class ModelDeployEndpointConfig(Base):
    """
    ModelDeployEndpointConfig

    Attributes
    ----------------------
    endpoint_config_name
    endpoint_config_arn
    """

    endpoint_config_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    endpoint_config_arn: Optional[StrPipeVar] = Unassigned()


class ModelDeployEndpoint(Base):
    """
    ModelDeployEndpoint

    Attributes
    ----------------------
    endpoint_name
    endpoint_arn
    """

    endpoint_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    endpoint_arn: Optional[StrPipeVar] = Unassigned()


class ModelDeployResult(Base):
    """
    ModelDeployResult
      Provides information about the endpoint of the model deployment.

    Attributes
    ----------------------
    endpoint_name: The name of the endpoint to which the model has been deployed.  If model deployment fails, this field is omitted from the response.
    endpoint_configs
    endpoints
    """

    endpoint_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    endpoint_configs: Optional[List[ModelDeployEndpointConfig]] = Unassigned()
    endpoints: Optional[List[ModelDeployEndpoint]] = Unassigned()


class ModelArtifacts(Base):
    """
    ModelArtifacts
      Provides information about the location that is configured for storing model artifacts.  Model artifacts are outputs that result from training a model. They typically consist of trained parameters, a model definition that describes how to compute inferences, and other metadata. A SageMaker container stores your trained model artifacts in the /opt/ml/model directory. After training has completed, by default, these artifacts are uploaded to your Amazon S3 bucket as compressed files.

    Attributes
    ----------------------
    s3_model_artifacts: The path of the S3 object that contains the model artifacts. For example, s3://bucket-name/keynameprefix/model.tar.gz.
    """

    s3_model_artifacts: StrPipeVar


class ModelDigests(Base):
    """
    ModelDigests
      Provides information to verify the integrity of stored model artifacts.

    Attributes
    ----------------------
    artifact_digest: Provides a hash value that uniquely identifies the stored model artifacts.
    """

    artifact_digest: Optional[StrPipeVar] = Unassigned()


class EdgeModel(Base):
    """
    EdgeModel
      The model on the edge device.

    Attributes
    ----------------------
    model_name: The name of the model.
    model_version: The model version.
    latest_sample_time: The timestamp of the last data sample taken.
    latest_inference: The timestamp of the last inference that was made.
    """

    model_name: Union[StrPipeVar, object]
    model_version: StrPipeVar
    latest_sample_time: Optional[datetime.datetime] = Unassigned()
    latest_inference: Optional[datetime.datetime] = Unassigned()


class EdgePresetDeploymentOutput(Base):
    """
    EdgePresetDeploymentOutput
      The output of a SageMaker Edge Manager deployable resource.

    Attributes
    ----------------------
    type: The deployment type created by SageMaker Edge Manager. Currently only supports Amazon Web Services IoT Greengrass Version 2 components.
    artifact: The Amazon Resource Name (ARN) of the generated deployable resource.
    status: The status of the deployable resource.
    status_message: Returns a message describing the status of the deployed resource.
    """

    type: StrPipeVar
    artifact: Optional[StrPipeVar] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    status_message: Optional[StrPipeVar] = Unassigned()


class ProductionVariantStatus(Base):
    """
    ProductionVariantStatus
      Describes the status of the production variant.

    Attributes
    ----------------------
    status: The endpoint variant status which describes the current deployment stage status or operational status.    Creating: Creating inference resources for the production variant.    Deleting: Terminating inference resources for the production variant.    Updating: Updating capacity for the production variant.    ActivatingTraffic: Turning on traffic for the production variant.    Baking: Waiting period to monitor the CloudWatch alarms in the automatic rollback configuration.
    status_message: A message that describes the status of the production variant.
    start_time: The start time of the current status change.
    """

    status: StrPipeVar
    status_message: Optional[StrPipeVar] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()


class Ec2CapacityReservation(Base):
    """
    Ec2CapacityReservation
      The EC2 capacity reservations that are shared to an ML capacity reservation.

    Attributes
    ----------------------
    ec2_capacity_reservation_id: The unique identifier for an EC2 capacity reservation that's part of the ML capacity reservation.
    total_instance_count: The number of instances that you allocated to the EC2 capacity reservation.
    available_instance_count: The number of instances that are currently available in the EC2 capacity reservation.
    used_by_current_endpoint: The number of instances from the EC2 capacity reservation that are being used by the endpoint.
    """

    ec2_capacity_reservation_id: Optional[StrPipeVar] = Unassigned()
    total_instance_count: Optional[int] = Unassigned()
    available_instance_count: Optional[int] = Unassigned()
    used_by_current_endpoint: Optional[int] = Unassigned()


class ProductionVariantCapacityReservationSummary(Base):
    """
    ProductionVariantCapacityReservationSummary
      Details about an ML capacity reservation.

    Attributes
    ----------------------
    ml_reservation_arn: The Amazon Resource Name (ARN) that uniquely identifies the ML capacity reservation that SageMaker AI applies when it deploys the endpoint.
    capacity_reservation_preference: The option that you chose for the capacity reservation. SageMaker AI supports the following options:  capacity-reservations-only  SageMaker AI launches instances only into an ML capacity reservation. If no capacity is available, the instances fail to launch.
    total_instance_count: The number of instances that you allocated to the ML capacity reservation.
    available_instance_count: The number of instances that are currently available in the ML capacity reservation.
    used_by_current_endpoint: The number of instances from the ML capacity reservation that are being used by the endpoint.
    ec2_capacity_reservations: The EC2 capacity reservations that are shared to this ML capacity reservation, if any.
    """

    ml_reservation_arn: Optional[StrPipeVar] = Unassigned()
    capacity_reservation_preference: Optional[StrPipeVar] = Unassigned()
    total_instance_count: Optional[int] = Unassigned()
    available_instance_count: Optional[int] = Unassigned()
    used_by_current_endpoint: Optional[int] = Unassigned()
    ec2_capacity_reservations: Optional[List[Ec2CapacityReservation]] = Unassigned()


class ProductionVariantSummary(Base):
    """
    ProductionVariantSummary
      Describes weight and capacities for a production variant associated with an endpoint. If you sent a request to the UpdateEndpointWeightsAndCapacities API and the endpoint status is Updating, you get different desired and current values.

    Attributes
    ----------------------
    variant_name: The name of the variant.
    deployed_images: An array of DeployedImage objects that specify the Amazon EC2 Container Registry paths of the inference images deployed on instances of this ProductionVariant.
    current_weight: The weight associated with the variant.
    desired_weight: The requested weight, as specified in the UpdateEndpointWeightsAndCapacities request.
    current_instance_count: The number of instances associated with the variant.
    desired_instance_count: The number of instances requested in the UpdateEndpointWeightsAndCapacities request.
    variant_status: The endpoint variant status which describes the current deployment stage status or operational status.
    current_serverless_config: The serverless configuration for the endpoint.
    desired_serverless_config: The serverless configuration requested for the endpoint update.
    managed_instance_scaling: Settings that control the range in the number of instances that the endpoint provisions as it scales up or down to accommodate traffic.
    routing_config: Settings that control how the endpoint routes incoming traffic to the instances that the endpoint hosts.
    capacity_schedules_config
    hyper_pod_config
    capacity_reservation_config: Settings for the capacity reservation for the compute instances that SageMaker AI reserves for an endpoint.
    """

    variant_name: StrPipeVar
    deployed_images: Optional[List[DeployedImage]] = Unassigned()
    current_weight: Optional[float] = Unassigned()
    desired_weight: Optional[float] = Unassigned()
    current_instance_count: Optional[int] = Unassigned()
    desired_instance_count: Optional[int] = Unassigned()
    variant_status: Optional[List[ProductionVariantStatus]] = Unassigned()
    current_serverless_config: Optional[ProductionVariantServerlessConfig] = Unassigned()
    desired_serverless_config: Optional[ProductionVariantServerlessConfig] = Unassigned()
    managed_instance_scaling: Optional[ProductionVariantManagedInstanceScaling] = Unassigned()
    routing_config: Optional[ProductionVariantRoutingConfig] = Unassigned()
    capacity_schedules_config: Optional[ProductionVariantCapacitySchedulesConfig] = Unassigned()
    hyper_pod_config: Optional[ProductionVariantHyperPodConfig] = Unassigned()
    capacity_reservation_config: Optional[ProductionVariantCapacityReservationSummary] = (
        Unassigned()
    )


class PendingProductionVariantSummary(Base):
    """
    PendingProductionVariantSummary
      The production variant summary for a deployment when an endpoint is creating or updating with the CreateEndpoint or UpdateEndpoint operations. Describes the VariantStatus , weight and capacity for a production variant associated with an endpoint.

    Attributes
    ----------------------
    variant_name: The name of the variant.
    deployed_images: An array of DeployedImage objects that specify the Amazon EC2 Container Registry paths of the inference images deployed on instances of this ProductionVariant.
    current_weight: The weight associated with the variant.
    desired_weight: The requested weight for the variant in this deployment, as specified in the endpoint configuration for the endpoint. The value is taken from the request to the CreateEndpointConfig operation.
    current_instance_count: The number of instances associated with the variant.
    desired_instance_count: The number of instances requested in this deployment, as specified in the endpoint configuration for the endpoint. The value is taken from the request to the CreateEndpointConfig operation.
    instance_type: The type of instances associated with the variant.
    accelerator_type: This parameter is no longer supported. Elastic Inference (EI) is no longer available. This parameter was used to specify the size of the EI instance to use for the production variant.
    variant_status: The endpoint variant status which describes the current deployment stage status or operational status.
    current_serverless_config: The serverless configuration for the endpoint.
    desired_serverless_config: The serverless configuration requested for this deployment, as specified in the endpoint configuration for the endpoint.
    managed_instance_scaling: Settings that control the range in the number of instances that the endpoint provisions as it scales up or down to accommodate traffic.
    routing_config: Settings that control how the endpoint routes incoming traffic to the instances that the endpoint hosts.
    capacity_schedules_config
    capacity_reservation_config: Settings for the capacity reservation for the compute instances that SageMaker AI reserves for an endpoint.
    """

    variant_name: StrPipeVar
    deployed_images: Optional[List[DeployedImage]] = Unassigned()
    current_weight: Optional[float] = Unassigned()
    desired_weight: Optional[float] = Unassigned()
    current_instance_count: Optional[int] = Unassigned()
    desired_instance_count: Optional[int] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    accelerator_type: Optional[StrPipeVar] = Unassigned()
    variant_status: Optional[List[ProductionVariantStatus]] = Unassigned()
    current_serverless_config: Optional[ProductionVariantServerlessConfig] = Unassigned()
    desired_serverless_config: Optional[ProductionVariantServerlessConfig] = Unassigned()
    managed_instance_scaling: Optional[ProductionVariantManagedInstanceScaling] = Unassigned()
    routing_config: Optional[ProductionVariantRoutingConfig] = Unassigned()
    capacity_schedules_config: Optional[ProductionVariantCapacitySchedulesConfig] = Unassigned()
    capacity_reservation_config: Optional[ProductionVariantCapacityReservationSummary] = (
        Unassigned()
    )


class PendingDeploymentSummary(Base):
    """
    PendingDeploymentSummary
      The summary of an in-progress deployment when an endpoint is creating or updating with a new endpoint configuration.

    Attributes
    ----------------------
    endpoint_config_name: The name of the endpoint configuration used in the deployment.
    production_variants: An array of PendingProductionVariantSummary objects, one for each model hosted behind this endpoint for the in-progress deployment.
    start_time: The start time of the deployment.
    shadow_production_variants: An array of PendingProductionVariantSummary objects, one for each model hosted behind this endpoint in shadow mode with production traffic replicated from the model specified on ProductionVariants for the in-progress deployment.
    graph_config_name
    """

    endpoint_config_name: Union[StrPipeVar, object]
    production_variants: Optional[List[PendingProductionVariantSummary]] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    shadow_production_variants: Optional[List[PendingProductionVariantSummary]] = Unassigned()
    graph_config_name: Optional[StrPipeVar] = Unassigned()


class ExperimentSource(Base):
    """
    ExperimentSource
      The source of the experiment.

    Attributes
    ----------------------
    source_arn: The Amazon Resource Name (ARN) of the source.
    source_type: The source type.
    """

    source_arn: StrPipeVar
    source_type: Optional[StrPipeVar] = Unassigned()


class ThroughputConfigDescription(Base):
    """
    ThroughputConfigDescription
      Active throughput configuration of the feature group. There are two modes: ON_DEMAND and PROVISIONED. With on-demand mode, you are charged for data reads and writes that your application performs on your feature group. You do not need to specify read and write throughput because Feature Store accommodates your workloads as they ramp up and down. You can switch a feature group to on-demand only once in a 24 hour period. With provisioned throughput mode, you specify the read and write capacity per second that you expect your application to require, and you are billed based on those limits. Exceeding provisioned throughput will result in your requests being throttled.  Note: PROVISIONED throughput mode is supported only for feature groups that are offline-only, or use the  Standard  tier online store.

    Attributes
    ----------------------
    throughput_mode: The mode used for your feature group throughput: ON_DEMAND or PROVISIONED.
    provisioned_read_capacity_units:  For provisioned feature groups with online store enabled, this indicates the read throughput you are billed for and can consume without throttling.  This field is not applicable for on-demand feature groups.
    provisioned_write_capacity_units:  For provisioned feature groups, this indicates the write throughput you are billed for and can consume without throttling.  This field is not applicable for on-demand feature groups.
    """

    throughput_mode: StrPipeVar
    provisioned_read_capacity_units: Optional[int] = Unassigned()
    provisioned_write_capacity_units: Optional[int] = Unassigned()


class OfflineStoreStatus(Base):
    """
    OfflineStoreStatus
      The status of OfflineStore.

    Attributes
    ----------------------
    status: An OfflineStore status.
    blocked_reason: The justification for why the OfflineStoreStatus is Blocked (if applicable).
    """

    status: StrPipeVar
    blocked_reason: Optional[StrPipeVar] = Unassigned()


class LastUpdateStatus(Base):
    """
    LastUpdateStatus
      A value that indicates whether the update was successful.

    Attributes
    ----------------------
    status: A value that indicates whether the update was made successful.
    failure_reason: If the update wasn't successful, indicates the reason why it failed.
    """

    status: StrPipeVar
    failure_reason: Optional[StrPipeVar] = Unassigned()


class OnlineStoreReplicaStatus(Base):
    """
    OnlineStoreReplicaStatus

    Attributes
    ----------------------
    status
    failure_reason
    """

    status: StrPipeVar
    failure_reason: Optional[StrPipeVar] = Unassigned()


class OnlineStoreReplica(Base):
    """
    OnlineStoreReplica

    Attributes
    ----------------------
    region_name
    online_store_replica_status
    """

    region_name: StrPipeVar
    online_store_replica_status: OnlineStoreReplicaStatus


class FeatureParameter(Base):
    """
    FeatureParameter
      A key-value pair that you specify to describe the feature.

    Attributes
    ----------------------
    key: A key that must contain a value to describe the feature.
    value: The value that belongs to a key.
    """

    key: Optional[StrPipeVar] = Unassigned()
    value: Optional[StrPipeVar] = Unassigned()


class HubContentDependency(Base):
    """
    HubContentDependency
      Any dependencies related to hub content, such as scripts, model artifacts, datasets, or notebooks.

    Attributes
    ----------------------
    dependency_origin_path: The hub content dependency origin path.
    dependency_copy_path: The hub content dependency copy path.
    """

    dependency_origin_path: Optional[StrPipeVar] = Unassigned()
    dependency_copy_path: Optional[StrPipeVar] = Unassigned()


class UiTemplateInfo(Base):
    """
    UiTemplateInfo
      Container for user interface template information.

    Attributes
    ----------------------
    url: The URL for the user interface template.
    content_sha256: The SHA-256 digest of the contents of the template.
    """

    url: Optional[StrPipeVar] = Unassigned()
    content_sha256: Optional[StrPipeVar] = Unassigned()


class TrainingJobStatusCounters(Base):
    """
    TrainingJobStatusCounters
      The numbers of training jobs launched by a hyperparameter tuning job, categorized by status.

    Attributes
    ----------------------
    completed: The number of completed training jobs launched by the hyperparameter tuning job.
    in_progress: The number of in-progress training jobs launched by a hyperparameter tuning job.
    retryable_error: The number of training jobs that failed, but can be retried. A failed training job can be retried only if it failed because an internal service error occurred.
    non_retryable_error: The number of training jobs that failed and can't be retried. A failed training job can't be retried if it failed because a client error occurred.
    stopped: The number of training jobs launched by a hyperparameter tuning job that were manually stopped.
    """

    completed: Optional[int] = Unassigned()
    in_progress: Optional[int] = Unassigned()
    retryable_error: Optional[int] = Unassigned()
    non_retryable_error: Optional[int] = Unassigned()
    stopped: Optional[int] = Unassigned()


class ObjectiveStatusCounters(Base):
    """
    ObjectiveStatusCounters
      Specifies the number of training jobs that this hyperparameter tuning job launched, categorized by the status of their objective metric. The objective metric status shows whether the final objective metric for the training job has been evaluated by the tuning job and used in the hyperparameter tuning process.

    Attributes
    ----------------------
    succeeded: The number of training jobs whose final objective metric was evaluated by the hyperparameter tuning job and used in the hyperparameter tuning process.
    pending: The number of training jobs that are in progress and pending evaluation of their final objective metric.
    failed: The number of training jobs whose final objective metric was not evaluated and used in the hyperparameter tuning process. This typically occurs when the training job failed or did not emit an objective metric.
    """

    succeeded: Optional[int] = Unassigned()
    pending: Optional[int] = Unassigned()
    failed: Optional[int] = Unassigned()


class FinalHyperParameterTuningJobObjectiveMetric(Base):
    """
    FinalHyperParameterTuningJobObjectiveMetric
      Shows the latest objective metric emitted by a training job that was launched by a hyperparameter tuning job. You define the objective metric in the HyperParameterTuningJobObjective parameter of HyperParameterTuningJobConfig.

    Attributes
    ----------------------
    type: Select if you want to minimize or maximize the objective metric during hyperparameter tuning.
    metric_name: The name of the objective metric. For SageMaker built-in algorithms, metrics are defined per algorithm. See the metrics for XGBoost as an example. You can also use a custom algorithm for training and define your own metrics. For more information, see Define metrics and environment variables.
    value: The value of the objective metric.
    """

    metric_name: StrPipeVar
    value: float
    type: Optional[StrPipeVar] = Unassigned()


class HyperParameterTrainingJobSummary(Base):
    """
    HyperParameterTrainingJobSummary
      The container for the summary information about a training job.

    Attributes
    ----------------------
    training_job_definition_name: The training job definition name.
    training_job_name: The name of the training job.
    training_job_arn: The Amazon Resource Name (ARN) of the training job.
    tuning_job_name: The HyperParameter tuning job that launched the training job.
    creation_time: The date and time that the training job was created.
    training_start_time: The date and time that the training job started.
    training_end_time: Specifies the time when the training job ends on training instances. You are billed for the time interval between the value of TrainingStartTime and this time. For successful jobs and stopped jobs, this is the time after model artifacts are uploaded. For failed jobs, this is the time when SageMaker detects a job failure.
    training_job_status: The status of the training job.
    tuned_hyper_parameters: A list of the hyperparameters for which you specified ranges to search.
    failure_reason: The reason that the training job failed.
    final_hyper_parameter_tuning_job_objective_metric: The FinalHyperParameterTuningJobObjectiveMetric object that specifies the value of the objective metric of the tuning job that launched this training job.
    objective_status: The status of the objective metric for the training job:   Succeeded: The final objective metric for the training job was evaluated by the hyperparameter tuning job and used in the hyperparameter tuning process.     Pending: The training job is in progress and evaluation of its final objective metric is pending.     Failed: The final objective metric for the training job was not evaluated, and was not used in the hyperparameter tuning process. This typically occurs when the training job failed or did not emit an objective metric.
    """

    training_job_name: Union[StrPipeVar, object]
    training_job_arn: StrPipeVar
    creation_time: datetime.datetime
    training_job_status: StrPipeVar
    tuned_hyper_parameters: Dict[StrPipeVar, StrPipeVar]
    training_job_definition_name: Optional[StrPipeVar] = Unassigned()
    tuning_job_name: Optional[StrPipeVar] = Unassigned()
    training_start_time: Optional[datetime.datetime] = Unassigned()
    training_end_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    final_hyper_parameter_tuning_job_objective_metric: Optional[
        FinalHyperParameterTuningJobObjectiveMetric
    ] = Unassigned()
    objective_status: Optional[StrPipeVar] = Unassigned()


class HyperParameterTuningJobCompletionDetails(Base):
    """
    HyperParameterTuningJobCompletionDetails
      A structure that contains runtime information about both current and completed hyperparameter tuning jobs.

    Attributes
    ----------------------
    number_of_training_jobs_objective_not_improving: The number of training jobs launched by a tuning job that are not improving (1% or less) as measured by model performance evaluated against an objective function.
    convergence_detected_time: The time in timestamp format that AMT detected model convergence, as defined by a lack of significant improvement over time based on criteria developed over a wide range of diverse benchmarking tests.
    """

    number_of_training_jobs_objective_not_improving: Optional[int] = Unassigned()
    convergence_detected_time: Optional[datetime.datetime] = Unassigned()


class HyperParameterTuningJobConsumedResources(Base):
    """
    HyperParameterTuningJobConsumedResources
      The total resources consumed by your hyperparameter tuning job.

    Attributes
    ----------------------
    runtime_in_seconds: The wall clock runtime in seconds used by your hyperparameter tuning job.
    billable_time_in_seconds
    """

    runtime_in_seconds: Optional[int] = Unassigned()
    billable_time_in_seconds: Optional[int] = Unassigned()


class InferenceComponentContainerSpecificationSummary(Base):
    """
    InferenceComponentContainerSpecificationSummary
      Details about the resources that are deployed with this inference component.

    Attributes
    ----------------------
    deployed_image
    artifact_url: The Amazon S3 path where the model artifacts are stored.
    environment: The environment variables to set in the Docker container.
    """

    deployed_image: Optional[DeployedImage] = Unassigned()
    artifact_url: Optional[StrPipeVar] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class InferenceComponentDataCacheConfigSummary(Base):
    """
    InferenceComponentDataCacheConfigSummary
      Settings that affect how the inference component caches data.

    Attributes
    ----------------------
    enable_caching: Indicates whether the inference component caches model artifacts as part of the auto scaling process.
    """

    enable_caching: bool


class InferenceComponentSpecificationSummary(Base):
    """
    InferenceComponentSpecificationSummary
      Details about the resources that are deployed with this inference component.

    Attributes
    ----------------------
    model_name: The name of the SageMaker AI model object that is deployed with the inference component.
    container: Details about the container that provides the runtime environment for the model that is deployed with the inference component.
    startup_parameters: Settings that take effect while the model container starts up.
    compute_resource_requirements: The compute resources allocated to run the model, plus any adapter models, that you assign to the inference component.
    base_inference_component_name: The name of the base inference component that contains this inference component.
    data_cache_config: Settings that affect how the inference component caches data.
    """

    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    container: Optional[InferenceComponentContainerSpecificationSummary] = Unassigned()
    startup_parameters: Optional[InferenceComponentStartupParameters] = Unassigned()
    compute_resource_requirements: Optional[InferenceComponentComputeResourceRequirements] = (
        Unassigned()
    )
    base_inference_component_name: Optional[StrPipeVar] = Unassigned()
    data_cache_config: Optional[InferenceComponentDataCacheConfigSummary] = Unassigned()


class InferenceComponentRuntimeConfigSummary(Base):
    """
    InferenceComponentRuntimeConfigSummary
      Details about the runtime settings for the model that is deployed with the inference component.

    Attributes
    ----------------------
    desired_copy_count: The number of runtime copies of the model container that you requested to deploy with the inference component.
    current_copy_count: The number of runtime copies of the model container that are currently deployed.
    """

    desired_copy_count: Optional[int] = Unassigned()
    current_copy_count: Optional[int] = Unassigned()


class InferenceComponentCapacitySize(Base):
    """
    InferenceComponentCapacitySize
      Specifies the type and size of the endpoint capacity to activate for a rolling deployment or a rollback strategy. You can specify your batches as either of the following:   A count of inference component copies    The overall percentage or your fleet    For a rollback strategy, if you don't specify the fields in this object, or if you set the Value parameter to 100%, then SageMaker AI uses a blue/green rollback strategy and rolls all traffic back to the blue fleet.

    Attributes
    ----------------------
    type: Specifies the endpoint capacity type.  COPY_COUNT  The endpoint activates based on the number of inference component copies.  CAPACITY_PERCENT  The endpoint activates based on the specified percentage of capacity.
    value: Defines the capacity size, either as a number of inference component copies or a capacity percentage.
    """

    type: StrPipeVar
    value: int


class InferenceComponentRollingUpdatePolicy(Base):
    """
    InferenceComponentRollingUpdatePolicy
      Specifies a rolling deployment strategy for updating a SageMaker AI inference component.

    Attributes
    ----------------------
    maximum_batch_size: The batch size for each rolling step in the deployment process. For each step, SageMaker AI provisions capacity on the new endpoint fleet, routes traffic to that fleet, and terminates capacity on the old endpoint fleet. The value must be between 5% to 50% of the copy count of the inference component.
    wait_interval_in_seconds: The length of the baking period, during which SageMaker AI monitors alarms for each batch on the new fleet.
    maximum_execution_timeout_in_seconds: The time limit for the total deployment. Exceeding this limit causes a timeout.
    rollback_maximum_batch_size: The batch size for a rollback to the old endpoint fleet. If this field is absent, the value is set to the default, which is 100% of the total capacity. When the default is used, SageMaker AI provisions the entire capacity of the old fleet at once during rollback.
    """

    maximum_batch_size: InferenceComponentCapacitySize
    wait_interval_in_seconds: int
    maximum_execution_timeout_in_seconds: Optional[int] = Unassigned()
    rollback_maximum_batch_size: Optional[InferenceComponentCapacitySize] = Unassigned()


class InferenceComponentDeploymentConfig(Base):
    """
    InferenceComponentDeploymentConfig
      The deployment configuration for an endpoint that hosts inference components. The configuration includes the desired deployment strategy and rollback settings.

    Attributes
    ----------------------
    rolling_update_policy: Specifies a rolling deployment strategy for updating a SageMaker AI endpoint.
    auto_rollback_configuration
    """

    rolling_update_policy: InferenceComponentRollingUpdatePolicy
    auto_rollback_configuration: Optional[AutoRollbackConfig] = Unassigned()


class EndpointMetadata(Base):
    """
    EndpointMetadata
      The metadata of the endpoint.

    Attributes
    ----------------------
    endpoint_name: The name of the endpoint.
    endpoint_config_name: The name of the endpoint configuration.
    endpoint_status:  The status of the endpoint. For possible values of the status of an endpoint, see EndpointSummary.
    failure_reason:  If the status of the endpoint is Failed, or the status is InService but update operation fails, this provides the reason why it failed.
    """

    endpoint_name: Union[StrPipeVar, object]
    endpoint_config_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    endpoint_status: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()


class ModelVariantConfigSummary(Base):
    """
    ModelVariantConfigSummary
      Summary of the deployment configuration of a model.

    Attributes
    ----------------------
    model_name: The name of the Amazon SageMaker Model entity.
    variant_name: The name of the variant.
    infrastructure_config: The configuration of the infrastructure that the model has been deployed to.
    status: The status of deployment for the model variant on the hosted inference endpoint.    Creating - Amazon SageMaker is preparing the model variant on the hosted inference endpoint.     InService - The model variant is running on the hosted inference endpoint.     Updating - Amazon SageMaker is updating the model variant on the hosted inference endpoint.     Deleting - Amazon SageMaker is deleting the model variant on the hosted inference endpoint.     Deleted - The model variant has been deleted on the hosted inference endpoint. This can only happen after stopping the experiment.
    """

    model_name: Union[StrPipeVar, object]
    variant_name: StrPipeVar
    infrastructure_config: ModelInfrastructureConfig
    status: StrPipeVar


class RecommendationMetrics(Base):
    """
    RecommendationMetrics
      The metrics of recommendations.

    Attributes
    ----------------------
    cost_per_hour: Defines the cost per hour for the instance.
    cost_per_inference: Defines the cost per inference for the instance .
    max_invocations: The expected maximum number of requests per minute for the instance.
    model_latency: The expected model latency at maximum invocation per minute for the instance.
    cpu_utilization: The expected CPU utilization at maximum invocations per minute for the instance.  NaN indicates that the value is not available.
    memory_utilization: The expected memory utilization at maximum invocations per minute for the instance.  NaN indicates that the value is not available.
    model_setup_time: The time it takes to launch new compute resources for a serverless endpoint. The time can vary depending on the model size, how long it takes to download the model, and the start-up time of the container.  NaN indicates that the value is not available.
    input_tokens_per_second_per_request
    output_tokens_per_second_per_request
    time_to_first_token
    cost_per_million_tokens
    cost_per_million_input_tokens
    cost_per_million_output_tokens
    intertoken_latency
    max_concurrency
    """

    cost_per_hour: Optional[float] = Unassigned()
    cost_per_inference: Optional[float] = Unassigned()
    max_invocations: Optional[int] = Unassigned()
    model_latency: Optional[int] = Unassigned()
    cpu_utilization: Optional[float] = Unassigned()
    memory_utilization: Optional[float] = Unassigned()
    model_setup_time: Optional[int] = Unassigned()
    input_tokens_per_second_per_request: Optional[float] = Unassigned()
    output_tokens_per_second_per_request: Optional[float] = Unassigned()
    time_to_first_token: Optional[float] = Unassigned()
    cost_per_million_tokens: Optional[float] = Unassigned()
    cost_per_million_input_tokens: Optional[float] = Unassigned()
    cost_per_million_output_tokens: Optional[float] = Unassigned()
    intertoken_latency: Optional[float] = Unassigned()
    max_concurrency: Optional[int] = Unassigned()


class EndpointOutputConfiguration(Base):
    """
    EndpointOutputConfiguration
      The endpoint configuration made by Inference Recommender during a recommendation job.

    Attributes
    ----------------------
    endpoint_name: The name of the endpoint made during a recommendation job.
    variant_name: The name of the production variant (deployed model) made during a recommendation job.
    instance_type: The instance type recommended by Amazon SageMaker Inference Recommender.
    initial_instance_count: The number of instances recommended to launch initially.
    serverless_config
    """

    endpoint_name: Union[StrPipeVar, object]
    variant_name: StrPipeVar
    instance_type: Optional[StrPipeVar] = Unassigned()
    initial_instance_count: Optional[int] = Unassigned()
    serverless_config: Optional[ProductionVariantServerlessConfig] = Unassigned()


class EnvironmentParameter(Base):
    """
    EnvironmentParameter
      A list of environment parameters suggested by the Amazon SageMaker Inference Recommender.

    Attributes
    ----------------------
    key: The environment key suggested by the Amazon SageMaker Inference Recommender.
    value_type: The value type suggested by the Amazon SageMaker Inference Recommender.
    value: The value suggested by the Amazon SageMaker Inference Recommender.
    """

    key: StrPipeVar
    value_type: StrPipeVar
    value: StrPipeVar


class ModelConfiguration(Base):
    """
    ModelConfiguration
      Defines the model configuration. Includes the specification name and environment parameters.

    Attributes
    ----------------------
    inference_specification_name: The inference specification name in the model package version.
    environment_parameters: Defines the environment parameters that includes key, value types, and values.
    compilation_job_name: The name of the compilation job used to create the recommended model artifacts.
    image
    """

    inference_specification_name: Optional[StrPipeVar] = Unassigned()
    environment_parameters: Optional[List[EnvironmentParameter]] = Unassigned()
    compilation_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    image: Optional[StrPipeVar] = Unassigned()


class InferenceRecommendation(Base):
    """
    InferenceRecommendation
      A list of recommendations made by Amazon SageMaker Inference Recommender.

    Attributes
    ----------------------
    recommendation_id: The recommendation ID which uniquely identifies each recommendation.
    metrics: The metrics used to decide what recommendation to make.
    endpoint_configuration: Defines the endpoint configuration parameters.
    model_configuration: Defines the model configuration.
    endpoint_arn
    invocation_end_time: A timestamp that shows when the benchmark completed.
    invocation_start_time: A timestamp that shows when the benchmark started.
    """

    endpoint_configuration: EndpointOutputConfiguration
    model_configuration: ModelConfiguration
    recommendation_id: Optional[StrPipeVar] = Unassigned()
    metrics: Optional[RecommendationMetrics] = Unassigned()
    endpoint_arn: Optional[StrPipeVar] = Unassigned()
    invocation_end_time: Optional[datetime.datetime] = Unassigned()
    invocation_start_time: Optional[datetime.datetime] = Unassigned()


class InferenceMetrics(Base):
    """
    InferenceMetrics
      The metrics for an existing endpoint compared in an Inference Recommender job.

    Attributes
    ----------------------
    max_invocations: The expected maximum number of requests per minute for the instance.
    model_latency: The expected model latency at maximum invocations per minute for the instance.
    input_tokens_per_second_per_request
    output_tokens_per_second_per_request
    time_to_first_token
    intertoken_latency
    max_concurrency
    """

    max_invocations: int
    model_latency: int
    input_tokens_per_second_per_request: Optional[float] = Unassigned()
    output_tokens_per_second_per_request: Optional[float] = Unassigned()
    time_to_first_token: Optional[float] = Unassigned()
    intertoken_latency: Optional[float] = Unassigned()
    max_concurrency: Optional[int] = Unassigned()


class EndpointPerformance(Base):
    """
    EndpointPerformance
      The performance results from running an Inference Recommender job on an existing endpoint.

    Attributes
    ----------------------
    metrics: The metrics for an existing endpoint.
    endpoint_info
    """

    metrics: InferenceMetrics
    endpoint_info: EndpointInfo


class LabelCounters(Base):
    """
    LabelCounters
      Provides a breakdown of the number of objects labeled.

    Attributes
    ----------------------
    total_labeled: The total number of objects labeled.
    human_labeled: The total number of objects labeled by a human worker.
    machine_labeled: The total number of objects labeled by automated data labeling.
    failed_non_retryable_error: The total number of objects that could not be labeled due to an error.
    unlabeled: The total number of objects not yet labeled.
    """

    total_labeled: Optional[int] = Unassigned()
    human_labeled: Optional[int] = Unassigned()
    machine_labeled: Optional[int] = Unassigned()
    failed_non_retryable_error: Optional[int] = Unassigned()
    unlabeled: Optional[int] = Unassigned()


class LabelingJobOutput(Base):
    """
    LabelingJobOutput
      Specifies the location of the output produced by the labeling job.

    Attributes
    ----------------------
    output_dataset_s3_uri: The Amazon S3 bucket location of the manifest file for labeled data.
    final_active_learning_model_arn: The Amazon Resource Name (ARN) for the most recent SageMaker model trained as part of automated data labeling.
    """

    output_dataset_s3_uri: StrPipeVar
    final_active_learning_model_arn: Optional[StrPipeVar] = Unassigned()


class UpgradeRollbackVersionDetails(Base):
    """
    UpgradeRollbackVersionDetails

    Attributes
    ----------------------
    snapshot_time
    previous_version
    """

    snapshot_time: Optional[datetime.datetime] = Unassigned()
    previous_version: Optional[StrPipeVar] = Unassigned()


class ModelCardExportArtifacts(Base):
    """
    ModelCardExportArtifacts
      The artifacts of the model card export job.

    Attributes
    ----------------------
    s3_export_artifacts: The Amazon S3 URI of the exported model artifacts.
    """

    s3_export_artifacts: StrPipeVar


class ModelPackageStatusItem(Base):
    """
    ModelPackageStatusItem
      Represents the overall status of a model package.

    Attributes
    ----------------------
    name: The name of the model package for which the overall status is being reported.
    status: The current status.
    failure_reason: if the overall status is Failed, the reason for the failure.
    """

    name: StrPipeVar
    status: StrPipeVar
    failure_reason: Optional[StrPipeVar] = Unassigned()


class ModelPackageStatusDetails(Base):
    """
    ModelPackageStatusDetails
      Specifies the validation and image scan statuses of the model package.

    Attributes
    ----------------------
    validation_statuses: The validation status of the model package.
    image_scan_statuses: The status of the scan of the Docker image container for the model package.
    """

    validation_statuses: List[ModelPackageStatusItem]
    image_scan_statuses: Optional[List[ModelPackageStatusItem]] = Unassigned()


class MonitoringExecutionSummary(Base):
    """
    MonitoringExecutionSummary
      Summary of information about the last monitoring job to run.

    Attributes
    ----------------------
    monitoring_schedule_name: The name of the monitoring schedule.
    scheduled_time: The time the monitoring job was scheduled.
    creation_time: The time at which the monitoring job was created.
    last_modified_time: A timestamp that indicates the last time the monitoring job was modified.
    monitoring_execution_status: The status of the monitoring job.
    processing_job_arn: The Amazon Resource Name (ARN) of the monitoring job.
    endpoint_name: The name of the endpoint used to run the monitoring job.
    failure_reason: Contains the reason a monitoring job failed, if it failed.
    monitoring_job_definition_name: The name of the monitoring job.
    monitoring_type: The type of the monitoring job.
    variant_name
    monitoring_execution_id
    """

    monitoring_schedule_name: Union[StrPipeVar, object]
    scheduled_time: datetime.datetime
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    monitoring_execution_status: StrPipeVar
    processing_job_arn: Optional[StrPipeVar] = Unassigned()
    endpoint_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    monitoring_job_definition_name: Optional[StrPipeVar] = Unassigned()
    monitoring_type: Optional[StrPipeVar] = Unassigned()
    variant_name: Optional[StrPipeVar] = Unassigned()
    monitoring_execution_id: Optional[StrPipeVar] = Unassigned()


class ModelQualityJobDefinition(Base):
    """
    ModelQualityJobDefinition

    Attributes
    ----------------------
    job_definition_arn
    job_definition_name
    creation_time
    model_quality_baseline_config
    model_quality_app_specification
    model_quality_job_input
    model_quality_job_output_config
    job_resources
    network_config
    role_arn
    stopping_condition
    """

    job_definition_arn: StrPipeVar
    job_definition_name: StrPipeVar
    creation_time: datetime.datetime
    model_quality_app_specification: ModelQualityAppSpecification
    model_quality_job_input: ModelQualityJobInput
    model_quality_job_output_config: MonitoringOutputConfig
    job_resources: MonitoringResources
    role_arn: StrPipeVar
    model_quality_baseline_config: Optional[ModelQualityBaselineConfig] = Unassigned()
    network_config: Optional[MonitoringNetworkConfig] = Unassigned()
    stopping_condition: Optional[MonitoringStoppingCondition] = Unassigned()


class ModelBiasJobDefinition(Base):
    """
    ModelBiasJobDefinition

    Attributes
    ----------------------
    job_definition_arn
    job_definition_name
    creation_time
    model_bias_baseline_config
    model_bias_app_specification
    model_bias_job_input
    model_bias_job_output_config
    job_resources
    network_config
    role_arn
    stopping_condition
    """

    job_definition_arn: StrPipeVar
    job_definition_name: StrPipeVar
    creation_time: datetime.datetime
    model_bias_app_specification: ModelBiasAppSpecification
    model_bias_job_input: ModelBiasJobInput
    model_bias_job_output_config: MonitoringOutputConfig
    job_resources: MonitoringResources
    role_arn: StrPipeVar
    model_bias_baseline_config: Optional[ModelBiasBaselineConfig] = Unassigned()
    network_config: Optional[MonitoringNetworkConfig] = Unassigned()
    stopping_condition: Optional[MonitoringStoppingCondition] = Unassigned()


class ModelExplainabilityJobDefinition(Base):
    """
    ModelExplainabilityJobDefinition

    Attributes
    ----------------------
    job_definition_arn
    job_definition_name
    creation_time
    model_explainability_baseline_config
    model_explainability_app_specification
    model_explainability_job_input
    model_explainability_job_output_config
    job_resources
    network_config
    role_arn
    stopping_condition
    """

    job_definition_arn: StrPipeVar
    job_definition_name: StrPipeVar
    creation_time: datetime.datetime
    model_explainability_app_specification: ModelExplainabilityAppSpecification
    model_explainability_job_input: ModelExplainabilityJobInput
    model_explainability_job_output_config: MonitoringOutputConfig
    job_resources: MonitoringResources
    role_arn: StrPipeVar
    model_explainability_baseline_config: Optional[ModelExplainabilityBaselineConfig] = Unassigned()
    network_config: Optional[MonitoringNetworkConfig] = Unassigned()
    stopping_condition: Optional[MonitoringStoppingCondition] = Unassigned()


class OptimizationOutput(Base):
    """
    OptimizationOutput
      Output values produced by an optimization job.

    Attributes
    ----------------------
    recommended_inference_image: The image that SageMaker recommends that you use to host the optimized model that you created with an optimization job.
    """

    recommended_inference_image: Optional[StrPipeVar] = Unassigned()


class ErrorInfo(Base):
    """
    ErrorInfo
      This is an error field object that contains the error code and the reason for an operation failure.

    Attributes
    ----------------------
    code: The error code for an invalid or failed operation.
    reason: The failure reason for the operation.
    """

    code: Optional[StrPipeVar] = Unassigned()
    reason: Optional[StrPipeVar] = Unassigned()


class DescribePipelineDefinitionForExecutionResponse(Base):
    """
    DescribePipelineDefinitionForExecutionResponse

    Attributes
    ----------------------
    pipeline_definition: The JSON pipeline definition.
    creation_time: The time when the pipeline was created.
    """

    pipeline_definition: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()


class PipelineExperimentConfig(Base):
    """
    PipelineExperimentConfig
      Specifies the names of the experiment and trial created by a pipeline.

    Attributes
    ----------------------
    experiment_name: The name of the experiment.
    trial_name: The name of the trial.
    """

    experiment_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    trial_name: Optional[Union[StrPipeVar, object]] = Unassigned()


class SelectedStep(Base):
    """
    SelectedStep
      A step selected to run in selective execution mode.

    Attributes
    ----------------------
    step_name: The name of the pipeline step.
    """

    step_name: StrPipeVar


class SelectiveExecutionConfig(Base):
    """
    SelectiveExecutionConfig
      The selective execution configuration applied to the pipeline run.

    Attributes
    ----------------------
    source_pipeline_execution_arn: The ARN from a reference execution of the current pipeline. Used to copy input collaterals needed for the selected steps to run. The execution status of the pipeline can be either Failed or Success. This field is required if the steps you specify for SelectedSteps depend on output collaterals from any non-specified pipeline steps. For more information, see Selective Execution for Pipeline Steps.
    selected_steps: A list of pipeline steps to run. All step(s) in all path(s) between two selected steps should be included.
    """

    selected_steps: List[SelectedStep]
    source_pipeline_execution_arn: Optional[StrPipeVar] = Unassigned()


class MLflowConfiguration(Base):
    """
    MLflowConfiguration

    Attributes
    ----------------------
    mlflow_resource_arn
    mlflow_experiment_name
    """

    mlflow_resource_arn: Optional[StrPipeVar] = Unassigned()
    mlflow_experiment_name: Optional[StrPipeVar] = Unassigned()


class ServiceCatalogProvisionedProductDetails(Base):
    """
    ServiceCatalogProvisionedProductDetails
      Details of a provisioned service catalog product. For information about service catalog, see What is Amazon Web Services Service Catalog.

    Attributes
    ----------------------
    provisioned_product_id: The ID of the provisioned product.
    provisioned_product_status_message: The current status of the product.    AVAILABLE - Stable state, ready to perform any operation. The most recent operation succeeded and completed.    UNDER_CHANGE - Transitive state. Operations performed might not have valid results. Wait for an AVAILABLE status before performing operations.    TAINTED - Stable state, ready to perform any operation. The stack has completed the requested operation but is not exactly what was requested. For example, a request to update to a new version failed and the stack rolled back to the current version.    ERROR - An unexpected error occurred. The provisioned product exists but the stack is not running. For example, CloudFormation received a parameter value that was not valid and could not launch the stack.    PLAN_IN_PROGRESS - Transitive state. The plan operations were performed to provision a new product, but resources have not yet been created. After reviewing the list of resources to be created, execute the plan. Wait for an AVAILABLE status before performing operations.
    """

    provisioned_product_id: Optional[StrPipeVar] = Unassigned()
    provisioned_product_status_message: Optional[StrPipeVar] = Unassigned()


class TemplateProviderDetail(Base):
    """
    TemplateProviderDetail
       Details about a template provider configuration and associated provisioning information.

    Attributes
    ----------------------
    cfn_template_provider_detail:  Details about a CloudFormation template provider configuration and associated provisioning information.
    """

    cfn_template_provider_detail: Optional[CfnTemplateProviderDetail] = Unassigned()


class UltraServerSummary(Base):
    """
    UltraServerSummary
      A summary of UltraServer resources and their current status.

    Attributes
    ----------------------
    ultra_server_type: The type of UltraServer, such as ml.u-p6e-gb200x72.
    instance_type: The Amazon EC2 instance type used in the UltraServer.
    ultra_server_count: The number of UltraServers of this type.
    available_spare_instance_count: The number of available spare instances in the UltraServers.
    unhealthy_instance_count: The total number of instances across all UltraServers of this type that are currently in an unhealthy state.
    """

    ultra_server_type: StrPipeVar
    instance_type: StrPipeVar
    ultra_server_count: Optional[int] = Unassigned()
    available_spare_instance_count: Optional[int] = Unassigned()
    unhealthy_instance_count: Optional[int] = Unassigned()


class SubscribedWorkteam(Base):
    """
    SubscribedWorkteam
      Describes a work team of a vendor that does the labelling job.

    Attributes
    ----------------------
    workteam_arn: The Amazon Resource Name (ARN) of the vendor that you have subscribed.
    marketplace_title: The title of the service provided by the vendor in the Amazon Marketplace.
    seller_name: The name of the vendor in the Amazon Marketplace.
    marketplace_description: The description of the vendor from the Amazon Marketplace.
    listing_id: Marketplace product listing ID.
    """

    workteam_arn: StrPipeVar
    marketplace_title: Optional[StrPipeVar] = Unassigned()
    seller_name: Optional[StrPipeVar] = Unassigned()
    marketplace_description: Optional[StrPipeVar] = Unassigned()
    listing_id: Optional[StrPipeVar] = Unassigned()


class TrainingJobOutput(Base):
    """
    TrainingJobOutput
      Provides information about the location that is configured for storing optional output.

    Attributes
    ----------------------
    s3_training_job_output: Provides information about the S3 bucket where training job output (model artifacts) is stored. For example, s3://bucket-name/keyname-prefix/output.tar.gz.
    """

    s3_training_job_output: StrPipeVar


class WarmPoolStatus(Base):
    """
    WarmPoolStatus
      Status and billing information about the warm pool.

    Attributes
    ----------------------
    status: The status of the warm pool.    InUse: The warm pool is in use for the training job.    Available: The warm pool is available to reuse for a matching training job.    Reused: The warm pool moved to a matching training job for reuse.    Terminated: The warm pool is no longer available. Warm pools are unavailable if they are terminated by a user, terminated for a patch update, or terminated for exceeding the specified KeepAlivePeriodInSeconds.
    resource_retained_billable_time_in_seconds: The billable time in seconds used by the warm pool. Billable time refers to the absolute wall-clock time. Multiply ResourceRetainedBillableTimeInSeconds by the number of instances (InstanceCount) in your training cluster to get the total compute time SageMaker bills you if you run warm pool training. The formula is as follows: ResourceRetainedBillableTimeInSeconds \* InstanceCount.
    reused_by_job: The name of the matching training job that reused the warm pool.
    """

    status: StrPipeVar
    resource_retained_billable_time_in_seconds: Optional[int] = Unassigned()
    reused_by_job: Optional[StrPipeVar] = Unassigned()


class SecondaryStatusTransition(Base):
    """
    SecondaryStatusTransition
      An array element of SecondaryStatusTransitions for DescribeTrainingJob. It provides additional details about a status that the training job has transitioned through. A training job can be in one of several states, for example, starting, downloading, training, or uploading. Within each state, there are a number of intermediate states. For example, within the starting state, SageMaker could be starting the training job or launching the ML instances. These transitional states are referred to as the job's secondary status.

    Attributes
    ----------------------
    status: Contains a secondary status information from a training job. Status might be one of the following secondary statuses:  InProgress     Starting - Starting the training job.    Downloading - An optional stage for algorithms that support File training input mode. It indicates that data is being downloaded to the ML storage volumes.    Training - Training is in progress.    Uploading - Training is complete and the model artifacts are being uploaded to the S3 location.    Completed     Completed - The training job has completed.    Failed     Failed - The training job has failed. The reason for the failure is returned in the FailureReason field of DescribeTrainingJobResponse.    Stopped     MaxRuntimeExceeded - The job stopped because it exceeded the maximum allowed runtime.    Stopped - The training job has stopped.    Stopping     Stopping - Stopping the training job.     We no longer support the following secondary statuses:    LaunchingMLInstances     PreparingTrainingStack     DownloadingTrainingImage
    start_time: A timestamp that shows when the training job transitioned to the current secondary status state.
    end_time: A timestamp that shows when the training job transitioned out of this secondary status state into another secondary status state or when the training job has ended.
    status_message: A detailed description of the progress within a secondary status.  SageMaker provides secondary statuses and status messages that apply to each of them:  Starting    Starting the training job.   Launching requested ML instances.   Insufficient capacity error from EC2 while launching instances, retrying!   Launched instance was unhealthy, replacing it!   Preparing the instances for training.    Training    Training image download completed. Training in progress.      Status messages are subject to change. Therefore, we recommend not including them in code that programmatically initiates actions. For examples, don't use status messages in if statements.  To have an overview of your training job's progress, view TrainingJobStatus and SecondaryStatus in DescribeTrainingJob, and StatusMessage together. For example, at the start of a training job, you might see the following:    TrainingJobStatus - InProgress    SecondaryStatus - Training    StatusMessage - Downloading the training image
    """

    status: StrPipeVar
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = Unassigned()
    status_message: Optional[StrPipeVar] = Unassigned()


class MetricData(Base):
    """
    MetricData
      The name, value, and date and time of a metric that was emitted to Amazon CloudWatch.

    Attributes
    ----------------------
    metric_name: The name of the metric.
    value: The value of the metric.
    timestamp: The date and time that the algorithm emitted the metric.
    """

    metric_name: Optional[StrPipeVar] = Unassigned()
    value: Optional[float] = Unassigned()
    timestamp: Optional[datetime.datetime] = Unassigned()


class ProfilerRuleEvaluationStatus(Base):
    """
    ProfilerRuleEvaluationStatus
      Information about the status of the rule evaluation.

    Attributes
    ----------------------
    rule_configuration_name: The name of the rule configuration.
    rule_evaluation_job_arn: The Amazon Resource Name (ARN) of the rule evaluation job.
    rule_evaluation_status: Status of the rule evaluation.
    status_details: Details from the rule evaluation.
    last_modified_time: Timestamp when the rule evaluation status was last modified.
    """

    rule_configuration_name: Optional[StrPipeVar] = Unassigned()
    rule_evaluation_job_arn: Optional[StrPipeVar] = Unassigned()
    rule_evaluation_status: Optional[StrPipeVar] = Unassigned()
    status_details: Optional[StrPipeVar] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class ImageMetadata(Base):
    """
    ImageMetadata

    Attributes
    ----------------------
    image_type
    """

    image_type: Optional[StrPipeVar] = Unassigned()


class MlflowDetails(Base):
    """
    MlflowDetails

    Attributes
    ----------------------
    mlflow_experiment_id
    mlflow_run_id
    """

    mlflow_experiment_id: Optional[StrPipeVar] = Unassigned()
    mlflow_run_id: Optional[StrPipeVar] = Unassigned()


class TrainingProgressInfo(Base):
    """
    TrainingProgressInfo

    Attributes
    ----------------------
    total_step_count_per_epoch
    current_step
    current_epoch
    max_epoch
    """

    total_step_count_per_epoch: Optional[int] = Unassigned()
    current_step: Optional[int] = Unassigned()
    current_epoch: Optional[int] = Unassigned()
    max_epoch: Optional[int] = Unassigned()


class ReservedCapacitySummary(Base):
    """
    ReservedCapacitySummary
      Details of a reserved capacity for the training plan. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .

    Attributes
    ----------------------
    reserved_capacity_arn: The Amazon Resource Name (ARN); of the reserved capacity.
    reserved_capacity_type: The type of reserved capacity.
    ultra_server_type: The type of UltraServer included in this reserved capacity, such as ml.u-p6e-gb200x72.
    ultra_server_count: The number of UltraServers included in this reserved capacity.
    instance_type: The instance type for the reserved capacity.
    total_instance_count: The total number of instances in the reserved capacity.
    status: The current status of the reserved capacity.
    availability_zone: The availability zone for the reserved capacity.
    availability_zone_id
    duration_hours: The number of whole hours in the total duration for this reserved capacity.
    duration_minutes: The additional minutes beyond whole hours in the total duration for this reserved capacity.
    start_time: The start time of the reserved capacity.
    end_time: The end time of the reserved capacity.
    """

    reserved_capacity_arn: StrPipeVar
    instance_type: StrPipeVar
    total_instance_count: int
    status: StrPipeVar
    reserved_capacity_type: Optional[StrPipeVar] = Unassigned()
    ultra_server_type: Optional[StrPipeVar] = Unassigned()
    ultra_server_count: Optional[int] = Unassigned()
    availability_zone: Optional[StrPipeVar] = Unassigned()
    availability_zone_id: Optional[StrPipeVar] = Unassigned()
    duration_hours: Optional[int] = Unassigned()
    duration_minutes: Optional[int] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()


class TrainingPlanStatusTransition(Base):
    """
    TrainingPlanStatusTransition

    Attributes
    ----------------------
    status
    start_time
    end_time
    status_message
    """

    status: StrPipeVar
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = Unassigned()
    status_message: Optional[StrPipeVar] = Unassigned()


class S3JobProgress(Base):
    """
    S3JobProgress

    Attributes
    ----------------------
    completed_objects
    failed_objects
    """

    completed_objects: int
    failed_objects: int


class TransformJobProgress(Base):
    """
    TransformJobProgress

    Attributes
    ----------------------
    s3_job_progress
    """

    s3_job_progress: Optional[S3JobProgress] = Unassigned()


class TrialComponentSource(Base):
    """
    TrialComponentSource
      The Amazon Resource Name (ARN) and job type of the source of a trial component.

    Attributes
    ----------------------
    source_arn: The source Amazon Resource Name (ARN).
    source_type: The source job type.
    """

    source_arn: StrPipeVar
    source_type: Optional[StrPipeVar] = Unassigned()


class TrialComponentMetricSummary(Base):
    """
    TrialComponentMetricSummary
      A summary of the metrics of a trial component.

    Attributes
    ----------------------
    metric_name: The name of the metric.
    source_arn: The Amazon Resource Name (ARN) of the source.
    time_stamp: When the metric was last updated.
    max: The maximum value of the metric.
    min: The minimum value of the metric.
    last: The most recent value of the metric.
    count: The number of samples used to generate the metric.
    avg: The average value of the metric.
    std_dev: The standard deviation of the metric.
    """

    metric_name: Optional[StrPipeVar] = Unassigned()
    source_arn: Optional[StrPipeVar] = Unassigned()
    time_stamp: Optional[datetime.datetime] = Unassigned()
    max: Optional[float] = Unassigned()
    min: Optional[float] = Unassigned()
    last: Optional[float] = Unassigned()
    count: Optional[int] = Unassigned()
    avg: Optional[float] = Unassigned()
    std_dev: Optional[float] = Unassigned()


class TrialSource(Base):
    """
    TrialSource
      The source of the trial.

    Attributes
    ----------------------
    source_arn: The Amazon Resource Name (ARN) of the source.
    source_type: The source job type.
    """

    source_arn: StrPipeVar
    source_type: Optional[StrPipeVar] = Unassigned()


class OidcConfigForResponse(Base):
    """
    OidcConfigForResponse
      Your OIDC IdP workforce configuration.

    Attributes
    ----------------------
    client_id: The OIDC IdP client ID used to configure your private workforce.
    issuer: The OIDC IdP issuer used to configure your private workforce.
    authorization_endpoint: The OIDC IdP authorization endpoint used to configure your private workforce.
    token_endpoint: The OIDC IdP token endpoint used to configure your private workforce.
    user_info_endpoint: The OIDC IdP user information endpoint used to configure your private workforce.
    logout_endpoint: The OIDC IdP logout endpoint used to configure your private workforce.
    jwks_uri: The OIDC IdP JSON Web Key Set (Jwks) URI used to configure your private workforce.
    scope: An array of string identifiers used to refer to the specific pieces of user data or claims that the client application wants to access.
    authentication_request_extra_params: A string to string map of identifiers specific to the custom identity provider (IdP) being used.
    """

    client_id: Optional[StrPipeVar] = Unassigned()
    issuer: Optional[StrPipeVar] = Unassigned()
    authorization_endpoint: Optional[StrPipeVar] = Unassigned()
    token_endpoint: Optional[StrPipeVar] = Unassigned()
    user_info_endpoint: Optional[StrPipeVar] = Unassigned()
    logout_endpoint: Optional[StrPipeVar] = Unassigned()
    jwks_uri: Optional[StrPipeVar] = Unassigned()
    scope: Optional[StrPipeVar] = Unassigned()
    authentication_request_extra_params: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class WorkforceVpcConfigResponse(Base):
    """
    WorkforceVpcConfigResponse
      A VpcConfig object that specifies the VPC that you want your workforce to connect to.

    Attributes
    ----------------------
    vpc_id: The ID of the VPC that the workforce uses for communication.
    security_group_ids: The VPC security group IDs, in the form sg-xxxxxxxx. The security groups must be for the same VPC as specified in the subnet.
    subnets: The ID of the subnets in the VPC that you want to connect.
    vpc_endpoint_id: The IDs for the VPC service endpoints of your VPC workforce when it is created and updated.
    """

    vpc_id: StrPipeVar
    security_group_ids: List[StrPipeVar]
    subnets: List[StrPipeVar]
    vpc_endpoint_id: Optional[StrPipeVar] = Unassigned()


class Workforce(Base):
    """
    Workforce
      A single private workforce, which is automatically created when you create your first private work team. You can create one private work force in each Amazon Web Services Region. By default, any workforce-related API operation used in a specific region will apply to the workforce created in that region. To learn how to create a private workforce, see Create a Private Workforce.

    Attributes
    ----------------------
    workforce_name: The name of the private workforce.
    workforce_arn: The Amazon Resource Name (ARN) of the private workforce.
    last_updated_date: The most recent date that UpdateWorkforce was used to successfully add one or more IP address ranges (CIDRs) to a private workforce's allow list.
    source_ip_config: A list of one to ten IP address ranges (CIDRs) to be added to the workforce allow list. By default, a workforce isn't restricted to specific IP addresses.
    sub_domain: The subdomain for your OIDC Identity Provider.
    cognito_config: The configuration of an Amazon Cognito workforce. A single Cognito workforce is created using and corresponds to a single  Amazon Cognito user pool.
    oidc_config: The configuration of an OIDC Identity Provider (IdP) private workforce.
    create_date: The date that the workforce is created.
    workforce_vpc_config: The configuration of a VPC workforce.
    status: The status of your workforce.
    failure_reason: The reason your workforce failed.
    ip_address_type: The IP address type you specify - either IPv4 only or dualstack (IPv4 and IPv6) - to support your labeling workforce.
    """

    workforce_name: Union[StrPipeVar, object]
    workforce_arn: StrPipeVar
    last_updated_date: Optional[datetime.datetime] = Unassigned()
    source_ip_config: Optional[SourceIpConfig] = Unassigned()
    sub_domain: Optional[StrPipeVar] = Unassigned()
    cognito_config: Optional[CognitoConfig] = Unassigned()
    oidc_config: Optional[OidcConfigForResponse] = Unassigned()
    create_date: Optional[datetime.datetime] = Unassigned()
    workforce_vpc_config: Optional[WorkforceVpcConfigResponse] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    ip_address_type: Optional[StrPipeVar] = Unassigned()


class Workteam(Base):
    """
    Workteam
      Provides details about a labeling work team.

    Attributes
    ----------------------
    workteam_name: The name of the work team.
    member_definitions: A list of MemberDefinition objects that contains objects that identify the workers that make up the work team.  Workforces can be created using Amazon Cognito or your own OIDC Identity Provider (IdP). For private workforces created using Amazon Cognito use CognitoMemberDefinition. For workforces created using your own OIDC identity provider (IdP) use OidcMemberDefinition.
    workteam_arn: The Amazon Resource Name (ARN) that identifies the work team.
    workforce_arn: The Amazon Resource Name (ARN) of the workforce.
    product_listing_ids: The Amazon Marketplace identifier for a vendor's work team.
    description: A description of the work team.
    sub_domain: The URI of the labeling job's user interface. Workers open this URI to start labeling your data objects.
    create_date: The date and time that the work team was created (timestamp).
    last_updated_date: The date and time that the work team was last updated (timestamp).
    notification_configuration: Configures SNS notifications of available or expiring work items for work teams.
    membership_rule
    membership_type
    worker_access_configuration: Describes any access constraints that have been defined for Amazon S3 resources.
    """

    workteam_name: Union[StrPipeVar, object]
    member_definitions: List[MemberDefinition]
    workteam_arn: StrPipeVar
    description: StrPipeVar
    workforce_arn: Optional[StrPipeVar] = Unassigned()
    product_listing_ids: Optional[List[StrPipeVar]] = Unassigned()
    sub_domain: Optional[StrPipeVar] = Unassigned()
    create_date: Optional[datetime.datetime] = Unassigned()
    last_updated_date: Optional[datetime.datetime] = Unassigned()
    notification_configuration: Optional[NotificationConfiguration] = Unassigned()
    membership_rule: Optional[MembershipRule] = Unassigned()
    membership_type: Optional[StrPipeVar] = Unassigned()
    worker_access_configuration: Optional[WorkerAccessConfiguration] = Unassigned()


class ProductionVariantServerlessUpdateConfig(Base):
    """
    ProductionVariantServerlessUpdateConfig
      Specifies the serverless update concurrency configuration for an endpoint variant.

    Attributes
    ----------------------
    max_concurrency: The updated maximum number of concurrent invocations your serverless endpoint can process.
    provisioned_concurrency: The updated amount of provisioned concurrency to allocate for the serverless endpoint. Should be less than or equal to MaxConcurrency.
    """

    max_concurrency: Optional[int] = Unassigned()
    provisioned_concurrency: Optional[int] = Unassigned()


class DesiredWeightAndCapacity(Base):
    """
    DesiredWeightAndCapacity
      Specifies weight and capacity values for a production variant.

    Attributes
    ----------------------
    variant_name: The name of the variant to update.
    desired_weight: The variant's weight.
    desired_instance_count: The variant's capacity.
    serverless_update_config: Specifies the serverless update concurrency configuration for an endpoint variant.
    """

    variant_name: StrPipeVar
    desired_weight: Optional[float] = Unassigned()
    desired_instance_count: Optional[int] = Unassigned()
    serverless_update_config: Optional[ProductionVariantServerlessUpdateConfig] = Unassigned()


class Device(Base):
    """
    Device
      Information of a particular device.

    Attributes
    ----------------------
    device_name: The name of the device.
    description: Description of the device.
    iot_thing_name: Amazon Web Services Internet of Things (IoT) object name.
    """

    device_name: Union[StrPipeVar, object]
    description: Optional[StrPipeVar] = Unassigned()
    iot_thing_name: Optional[StrPipeVar] = Unassigned()


class DeviceDeploymentSummary(Base):
    """
    DeviceDeploymentSummary
      Contains information summarizing device details and deployment status.

    Attributes
    ----------------------
    edge_deployment_plan_arn: The ARN of the edge deployment plan.
    edge_deployment_plan_name: The name of the edge deployment plan.
    stage_name: The name of the stage in the edge deployment plan.
    deployed_stage_name: The name of the deployed stage.
    device_fleet_name: The name of the fleet to which the device belongs to.
    device_name: The name of the device.
    device_arn: The ARN of the device.
    device_deployment_status: The deployment status of the device.
    device_deployment_status_message: The detailed error message for the deployoment status result.
    description: The description of the device.
    deployment_start_time: The time when the deployment on the device started.
    """

    edge_deployment_plan_arn: StrPipeVar
    edge_deployment_plan_name: Union[StrPipeVar, object]
    stage_name: StrPipeVar
    device_name: Union[StrPipeVar, object]
    device_arn: StrPipeVar
    deployed_stage_name: Optional[StrPipeVar] = Unassigned()
    device_fleet_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    device_deployment_status: Optional[StrPipeVar] = Unassigned()
    device_deployment_status_message: Optional[StrPipeVar] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    deployment_start_time: Optional[datetime.datetime] = Unassigned()


class DeviceFleetSummary(Base):
    """
    DeviceFleetSummary
      Summary of the device fleet.

    Attributes
    ----------------------
    device_fleet_arn: Amazon Resource Name (ARN) of the device fleet.
    device_fleet_name: Name of the device fleet.
    creation_time: Timestamp of when the device fleet was created.
    last_modified_time: Timestamp of when the device fleet was last updated.
    """

    device_fleet_arn: StrPipeVar
    device_fleet_name: Union[StrPipeVar, object]
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class DeviceStats(Base):
    """
    DeviceStats
      Status of devices.

    Attributes
    ----------------------
    connected_device_count: The number of devices connected with a heartbeat.
    registered_device_count: The number of registered devices.
    """

    connected_device_count: int
    registered_device_count: int


class EdgeModelSummary(Base):
    """
    EdgeModelSummary
      Summary of model on edge device.

    Attributes
    ----------------------
    model_name: The name of the model.
    model_version: The version model.
    """

    model_name: Union[StrPipeVar, object]
    model_version: StrPipeVar


class DeviceSummary(Base):
    """
    DeviceSummary
      Summary of the device.

    Attributes
    ----------------------
    device_name: The unique identifier of the device.
    device_arn: Amazon Resource Name (ARN) of the device.
    description: A description of the device.
    device_fleet_name: The name of the fleet the device belongs to.
    iot_thing_name: The Amazon Web Services Internet of Things (IoT) object thing name associated with the device..
    registration_time: The timestamp of the last registration or de-reregistration.
    latest_heartbeat: The last heartbeat received from the device.
    models: Models on the device.
    agent_version: Edge Manager agent version.
    """

    device_name: Union[StrPipeVar, object]
    device_arn: StrPipeVar
    description: Optional[StrPipeVar] = Unassigned()
    device_fleet_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    iot_thing_name: Optional[StrPipeVar] = Unassigned()
    registration_time: Optional[datetime.datetime] = Unassigned()
    latest_heartbeat: Optional[datetime.datetime] = Unassigned()
    models: Optional[List[EdgeModelSummary]] = Unassigned()
    agent_version: Optional[StrPipeVar] = Unassigned()


class Domain(Base):
    """
    Domain

    Attributes
    ----------------------
    domain_arn
    domain_id
    domain_name
    home_efs_file_system_id
    single_sign_on_managed_application_instance_id
    single_sign_on_application_arn
    status
    creation_time
    last_modified_time
    failure_reason
    security_group_id_for_domain_boundary
    auth_mode
    default_user_settings
    domain_settings
    app_network_access
    app_network_access_type
    home_efs_file_system_kms_key_id
    subnet_ids
    url
    vpc_id
    kms_key_id
    app_security_group_management
    app_storage_type
    tag_propagation
    default_space_settings
    tags
    """

    domain_arn: Optional[StrPipeVar] = Unassigned()
    domain_id: Optional[StrPipeVar] = Unassigned()
    domain_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    home_efs_file_system_id: Optional[StrPipeVar] = Unassigned()
    single_sign_on_managed_application_instance_id: Optional[StrPipeVar] = Unassigned()
    single_sign_on_application_arn: Optional[StrPipeVar] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    security_group_id_for_domain_boundary: Optional[StrPipeVar] = Unassigned()
    auth_mode: Optional[StrPipeVar] = Unassigned()
    default_user_settings: Optional[UserSettings] = Unassigned()
    domain_settings: Optional[DomainSettings] = Unassigned()
    app_network_access: Optional[StrPipeVar] = Unassigned()
    app_network_access_type: Optional[StrPipeVar] = Unassigned()
    home_efs_file_system_kms_key_id: Optional[StrPipeVar] = Unassigned()
    subnet_ids: Optional[List[StrPipeVar]] = Unassigned()
    url: Optional[StrPipeVar] = Unassigned()
    vpc_id: Optional[StrPipeVar] = Unassigned()
    kms_key_id: Optional[StrPipeVar] = Unassigned()
    app_security_group_management: Optional[StrPipeVar] = Unassigned()
    app_storage_type: Optional[StrPipeVar] = Unassigned()
    tag_propagation: Optional[StrPipeVar] = Unassigned()
    default_space_settings: Optional[DefaultSpaceSettings] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class DomainDetails(Base):
    """
    DomainDetails
      The domain's details.

    Attributes
    ----------------------
    domain_arn: The domain's Amazon Resource Name (ARN).
    domain_id: The domain ID.
    domain_name: The domain name.
    status: The status.
    creation_time: The creation time.
    last_modified_time: The last modified time.
    url: The domain's URL.
    """

    domain_arn: Optional[StrPipeVar] = Unassigned()
    domain_id: Optional[StrPipeVar] = Unassigned()
    domain_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    url: Optional[StrPipeVar] = Unassigned()


class RStudioServerProDomainSettingsForUpdate(Base):
    """
    RStudioServerProDomainSettingsForUpdate
      A collection of settings that update the current configuration for the RStudioServerPro Domain-level app.

    Attributes
    ----------------------
    domain_execution_role_arn: The execution role for the RStudioServerPro Domain-level app.
    default_resource_spec
    r_studio_connect_url: A URL pointing to an RStudio Connect server.
    r_studio_package_manager_url: A URL pointing to an RStudio Package Manager server.
    """

    domain_execution_role_arn: StrPipeVar
    default_resource_spec: Optional[ResourceSpec] = Unassigned()
    r_studio_connect_url: Optional[StrPipeVar] = Unassigned()
    r_studio_package_manager_url: Optional[StrPipeVar] = Unassigned()


class DomainSettingsForUpdate(Base):
    """
    DomainSettingsForUpdate
      A collection of Domain configuration settings to update.

    Attributes
    ----------------------
    r_studio_server_pro_domain_settings_for_update: A collection of RStudioServerPro Domain-level app settings to update. A single RStudioServerPro application is created for a domain.
    execution_role_identity_config: The configuration for attaching a SageMaker AI user profile name to the execution role as a sts:SourceIdentity key. This configuration can only be modified if there are no apps in the InService or Pending state.
    security_group_ids: The security groups for the Amazon Virtual Private Cloud that the Domain uses for communication between Domain-level apps and user apps.
    trusted_identity_propagation_settings: The Trusted Identity Propagation (TIP) settings for the SageMaker domain. These settings determine how user identities from IAM Identity Center are propagated through the domain to TIP enabled Amazon Web Services services.
    docker_settings: A collection of settings that configure the domain's Docker interaction.
    amazon_q_settings: A collection of settings that configure the Amazon Q experience within the domain.
    unified_studio_settings: The settings that apply to an SageMaker AI domain when you use it in Amazon SageMaker Unified Studio.
    ip_address_type: The IP address type for the domain. Specify ipv4 for IPv4-only connectivity or dualstack for both IPv4 and IPv6 connectivity. When you specify dualstack, the subnet must support IPv6 CIDR blocks.
    """

    r_studio_server_pro_domain_settings_for_update: Optional[
        RStudioServerProDomainSettingsForUpdate
    ] = Unassigned()
    execution_role_identity_config: Optional[StrPipeVar] = Unassigned()
    security_group_ids: Optional[List[StrPipeVar]] = Unassigned()
    trusted_identity_propagation_settings: Optional[TrustedIdentityPropagationSettings] = (
        Unassigned()
    )
    docker_settings: Optional[DockerSettings] = Unassigned()
    amazon_q_settings: Optional[AmazonQSettings] = Unassigned()
    unified_studio_settings: Optional[UnifiedStudioSettings] = Unassigned()
    ip_address_type: Optional[StrPipeVar] = Unassigned()


class DryRunOperation(Base):
    """
    DryRunOperation

    Attributes
    ----------------------
    error_code
    message
    """

    error_code: Optional[StrPipeVar] = Unassigned()
    message: Optional[StrPipeVar] = Unassigned()


class PredefinedMetricSpecification(Base):
    """
    PredefinedMetricSpecification
      A specification for a predefined metric.

    Attributes
    ----------------------
    predefined_metric_type: The metric type. You can only apply SageMaker metric types to SageMaker endpoints.
    """

    predefined_metric_type: Optional[StrPipeVar] = Unassigned()


class MetricSpecification(Base):
    """
    MetricSpecification
      An object containing information about a metric.

    Attributes
    ----------------------
    predefined: Information about a predefined metric.
    customized: Information about a customized metric.
    """

    predefined: Optional[PredefinedMetricSpecification] = Unassigned()
    customized: Optional[CustomizedMetricSpecification] = Unassigned()


class TargetTrackingScalingPolicyConfiguration(Base):
    """
    TargetTrackingScalingPolicyConfiguration
      A target tracking scaling policy. Includes support for predefined or customized metrics. When using the PutScalingPolicy API, this parameter is required when you are creating a policy with the policy type TargetTrackingScaling.

    Attributes
    ----------------------
    metric_specification: An object containing information about a metric.
    target_value: The recommended target value to specify for the metric when creating a scaling policy.
    """

    metric_specification: Optional[MetricSpecification] = Unassigned()
    target_value: Optional[float] = Unassigned()


class ScalingPolicy(Base):
    """
    ScalingPolicy
      An object containing a recommended scaling policy.

    Attributes
    ----------------------
    target_tracking: A target tracking scaling policy. Includes support for predefined or customized metrics.
    """

    target_tracking: Optional[TargetTrackingScalingPolicyConfiguration] = Unassigned()


class DynamicScalingConfiguration(Base):
    """
    DynamicScalingConfiguration
      An object with the recommended values for you to specify when creating an autoscaling policy.

    Attributes
    ----------------------
    min_capacity: The recommended minimum capacity to specify for your autoscaling policy.
    max_capacity: The recommended maximum capacity to specify for your autoscaling policy.
    scale_in_cooldown: The recommended scale in cooldown time for your autoscaling policy.
    scale_out_cooldown: The recommended scale out cooldown time for your autoscaling policy.
    scaling_policies: An object of the scaling policies for each metric.
    """

    min_capacity: Optional[int] = Unassigned()
    max_capacity: Optional[int] = Unassigned()
    scale_in_cooldown: Optional[int] = Unassigned()
    scale_out_cooldown: Optional[int] = Unassigned()
    scaling_policies: Optional[List[ScalingPolicy]] = Unassigned()


class EMRStepMetadata(Base):
    """
    EMRStepMetadata
      The configurations and outcomes of an Amazon EMR step execution.

    Attributes
    ----------------------
    cluster_id: The identifier of the EMR cluster.
    step_id: The identifier of the EMR cluster step.
    step_name: The name of the EMR cluster step.
    log_file_path: The path to the log file where the cluster step's failure root cause is recorded.
    """

    cluster_id: Optional[StrPipeVar] = Unassigned()
    step_id: Optional[StrPipeVar] = Unassigned()
    step_name: Optional[StrPipeVar] = Unassigned()
    log_file_path: Optional[StrPipeVar] = Unassigned()


class Edge(Base):
    """
    Edge
      A directed edge connecting two lineage entities.

    Attributes
    ----------------------
    source_arn: The Amazon Resource Name (ARN) of the source lineage entity of the directed edge.
    destination_arn: The Amazon Resource Name (ARN) of the destination lineage entity of the directed edge.
    association_type: The type of the Association(Edge) between the source and destination. For example ContributedTo, Produced, or DerivedFrom.
    """

    source_arn: Optional[StrPipeVar] = Unassigned()
    destination_arn: Optional[StrPipeVar] = Unassigned()
    association_type: Optional[StrPipeVar] = Unassigned()


class EdgeDeploymentPlanSummary(Base):
    """
    EdgeDeploymentPlanSummary
      Contains information summarizing an edge deployment plan.

    Attributes
    ----------------------
    edge_deployment_plan_arn: The ARN of the edge deployment plan.
    edge_deployment_plan_name: The name of the edge deployment plan.
    device_fleet_name: The name of the device fleet used for the deployment.
    edge_deployment_success: The number of edge devices with the successful deployment.
    edge_deployment_pending: The number of edge devices yet to pick up the deployment, or in progress.
    edge_deployment_failed: The number of edge devices that failed the deployment.
    creation_time: The time when the edge deployment plan was created.
    last_modified_time: The time when the edge deployment plan was last updated.
    """

    edge_deployment_plan_arn: StrPipeVar
    edge_deployment_plan_name: Union[StrPipeVar, object]
    device_fleet_name: Union[StrPipeVar, object]
    edge_deployment_success: int
    edge_deployment_pending: int
    edge_deployment_failed: int
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class EdgeModelStat(Base):
    """
    EdgeModelStat
      Status of edge devices with this model.

    Attributes
    ----------------------
    model_name: The name of the model.
    model_version: The model version.
    offline_device_count: The number of devices that have this model version and do not have a heart beat.
    connected_device_count: The number of devices that have this model version and have a heart beat.
    active_device_count: The number of devices that have this model version, a heart beat, and are currently running.
    sampling_device_count: The number of devices with this model version and are producing sample data.
    """

    model_name: Union[StrPipeVar, object]
    model_version: StrPipeVar
    offline_device_count: int
    connected_device_count: int
    active_device_count: int
    sampling_device_count: int


class EdgePackagingJobSummary(Base):
    """
    EdgePackagingJobSummary
      Summary of edge packaging job.

    Attributes
    ----------------------
    edge_packaging_job_arn: The Amazon Resource Name (ARN) of the edge packaging job.
    edge_packaging_job_name: The name of the edge packaging job.
    edge_packaging_job_status: The status of the edge packaging job.
    compilation_job_name: The name of the SageMaker Neo compilation job.
    model_name: The name of the model.
    model_version: The version of the model.
    creation_time: The timestamp of when the job was created.
    last_modified_time: The timestamp of when the edge packaging job was last updated.
    """

    edge_packaging_job_arn: StrPipeVar
    edge_packaging_job_name: Union[StrPipeVar, object]
    edge_packaging_job_status: StrPipeVar
    compilation_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_version: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class MonitoringSchedule(Base):
    """
    MonitoringSchedule
      A schedule for a model monitoring job. For information about model monitor, see Amazon SageMaker Model Monitor.

    Attributes
    ----------------------
    monitoring_schedule_arn: The Amazon Resource Name (ARN) of the monitoring schedule.
    monitoring_schedule_name: The name of the monitoring schedule.
    monitoring_schedule_status: The status of the monitoring schedule. This can be one of the following values.    PENDING - The schedule is pending being created.    FAILED - The schedule failed.    SCHEDULED - The schedule was successfully created.    STOPPED - The schedule was stopped.
    monitoring_type: The type of the monitoring job definition to schedule.
    failure_reason: If the monitoring schedule failed, the reason it failed.
    creation_time: The time that the monitoring schedule was created.
    last_modified_time: The last time the monitoring schedule was changed.
    monitoring_schedule_config
    endpoint_name: The endpoint that hosts the model being monitored.
    last_monitoring_execution_summary
    custom_monitoring_job_definition
    data_quality_job_definition
    model_quality_job_definition
    model_bias_job_definition
    model_explainability_job_definition
    variant_name
    tags: A list of the tags associated with the monitoring schedlue. For more information, see Tagging Amazon Web Services resources in the Amazon Web Services General Reference Guide.
    """

    monitoring_schedule_arn: Optional[StrPipeVar] = Unassigned()
    monitoring_schedule_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    monitoring_schedule_status: Optional[StrPipeVar] = Unassigned()
    monitoring_type: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    monitoring_schedule_config: Optional[MonitoringScheduleConfig] = Unassigned()
    endpoint_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    last_monitoring_execution_summary: Optional[MonitoringExecutionSummary] = Unassigned()
    custom_monitoring_job_definition: Optional[CustomMonitoringJobDefinition] = Unassigned()
    data_quality_job_definition: Optional[DataQualityJobDefinition] = Unassigned()
    model_quality_job_definition: Optional[ModelQualityJobDefinition] = Unassigned()
    model_bias_job_definition: Optional[ModelBiasJobDefinition] = Unassigned()
    model_explainability_job_definition: Optional[ModelExplainabilityJobDefinition] = Unassigned()
    variant_name: Optional[StrPipeVar] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class Endpoint(Base):
    """
    Endpoint
      A hosted endpoint for real-time inference.

    Attributes
    ----------------------
    endpoint_name: The name of the endpoint.
    endpoint_arn: The Amazon Resource Name (ARN) of the endpoint.
    endpoint_config_name: The endpoint configuration associated with the endpoint.
    deletion_condition
    production_variants: A list of the production variants hosted on the endpoint. Each production variant is a model.
    data_capture_config
    endpoint_status: The status of the endpoint.
    failure_reason: If the endpoint failed, the reason it failed.
    creation_time: The time that the endpoint was created.
    last_modified_time: The last time the endpoint was modified.
    monitoring_schedules: A list of monitoring schedules for the endpoint. For information about model monitoring, see Amazon SageMaker Model Monitor.
    tags: A list of the tags associated with the endpoint. For more information, see Tagging Amazon Web Services resources in the Amazon Web Services General Reference Guide.
    shadow_production_variants: A list of the shadow variants hosted on the endpoint. Each shadow variant is a model in shadow mode with production traffic replicated from the production variant.
    """

    endpoint_name: Union[StrPipeVar, object]
    endpoint_arn: StrPipeVar
    endpoint_config_name: Union[StrPipeVar, object]
    endpoint_status: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    deletion_condition: Optional[EndpointDeletionCondition] = Unassigned()
    production_variants: Optional[List[ProductionVariantSummary]] = Unassigned()
    data_capture_config: Optional[DataCaptureConfigSummary] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    monitoring_schedules: Optional[List[MonitoringSchedule]] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    shadow_production_variants: Optional[List[ProductionVariantSummary]] = Unassigned()


class EndpointConfigStepMetadata(Base):
    """
    EndpointConfigStepMetadata
      Metadata for an endpoint configuration step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the endpoint configuration used in the step.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class EndpointConfigSummary(Base):
    """
    EndpointConfigSummary
      Provides summary information for an endpoint configuration.

    Attributes
    ----------------------
    endpoint_config_name: The name of the endpoint configuration.
    endpoint_config_arn: The Amazon Resource Name (ARN) of the endpoint configuration.
    creation_time: A timestamp that shows when the endpoint configuration was created.
    """

    endpoint_config_name: Union[StrPipeVar, object]
    endpoint_config_arn: StrPipeVar
    creation_time: datetime.datetime


class EndpointStepMetadata(Base):
    """
    EndpointStepMetadata
      Metadata for an endpoint step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the endpoint in the step.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class EndpointSummary(Base):
    """
    EndpointSummary
      Provides summary information for an endpoint.

    Attributes
    ----------------------
    endpoint_name: The name of the endpoint.
    endpoint_arn: The Amazon Resource Name (ARN) of the endpoint.
    creation_time: A timestamp that shows when the endpoint was created.
    last_modified_time: A timestamp that shows when the endpoint was last modified.
    endpoint_status: The status of the endpoint.    OutOfService: Endpoint is not available to take incoming requests.    Creating: CreateEndpoint is executing.    Updating: UpdateEndpoint or UpdateEndpointWeightsAndCapacities is executing.    SystemUpdating: Endpoint is undergoing maintenance and cannot be updated or deleted or re-scaled until it has completed. This maintenance operation does not change any customer-specified values such as VPC config, KMS encryption, model, instance type, or instance count.    RollingBack: Endpoint fails to scale up or down or change its variant weight and is in the process of rolling back to its previous configuration. Once the rollback completes, endpoint returns to an InService status. This transitional status only applies to an endpoint that has autoscaling enabled and is undergoing variant weight or capacity changes as part of an UpdateEndpointWeightsAndCapacities call or when the UpdateEndpointWeightsAndCapacities operation is called explicitly.    InService: Endpoint is available to process incoming requests.    Deleting: DeleteEndpoint is executing.    Failed: Endpoint could not be created, updated, or re-scaled. Use DescribeEndpointOutput$FailureReason for information about the failure. DeleteEndpoint is the only operation that can be performed on a failed endpoint.   To get a list of endpoints with a specified status, use the StatusEquals filter with a call to ListEndpoints.
    """

    endpoint_name: Union[StrPipeVar, object]
    endpoint_arn: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    endpoint_status: StrPipeVar


class EvaluationJobSummary(Base):
    """
    EvaluationJobSummary

    Attributes
    ----------------------
    evaluation_job_name
    evaluation_job_arn
    evaluation_job_status
    creation_time
    evaluation_method
    failure_reason
    model_identifiers
    """

    evaluation_job_name: Union[StrPipeVar, object]
    evaluation_job_arn: StrPipeVar
    evaluation_job_status: StrPipeVar
    creation_time: datetime.datetime
    evaluation_method: StrPipeVar
    failure_reason: Optional[StrPipeVar] = Unassigned()
    model_identifiers: Optional[List[StrPipeVar]] = Unassigned()


class EventEntity(Base):
    """
    EventEntity

    Attributes
    ----------------------
    event_sender
    event_id
    shared_model_id
    shared_model_version
    event_type
    read
    """

    event_sender: Optional[StrPipeVar] = Unassigned()
    event_id: Optional[StrPipeVar] = Unassigned()
    shared_model_id: Optional[StrPipeVar] = Unassigned()
    shared_model_version: Optional[StrPipeVar] = Unassigned()
    event_type: Optional[StrPipeVar] = Unassigned()
    read: Optional[bool] = Unassigned()


class Experiment(Base):
    """
    Experiment
      The properties of an experiment as returned by the Search API. For information about experiments, see the CreateExperiment API.

    Attributes
    ----------------------
    experiment_name: The name of the experiment.
    experiment_arn: The Amazon Resource Name (ARN) of the experiment.
    display_name: The name of the experiment as displayed. If DisplayName isn't specified, ExperimentName is displayed.
    source
    description: The description of the experiment.
    creation_time: When the experiment was created.
    created_by: Who created the experiment.
    last_modified_time: When the experiment was last modified.
    last_modified_by
    tags: The list of tags that are associated with the experiment. You can use Search API to search on the tags.
    """

    experiment_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    experiment_arn: Optional[StrPipeVar] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    source: Optional[ExperimentSource] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class ExperimentSummary(Base):
    """
    ExperimentSummary
      A summary of the properties of an experiment. To get the complete set of properties, call the DescribeExperiment API and provide the ExperimentName.

    Attributes
    ----------------------
    experiment_arn: The Amazon Resource Name (ARN) of the experiment.
    experiment_name: The name of the experiment.
    display_name: The name of the experiment as displayed. If DisplayName isn't specified, ExperimentName is displayed.
    experiment_source
    creation_time: When the experiment was created.
    last_modified_time: When the experiment was last modified.
    """

    experiment_arn: Optional[StrPipeVar] = Unassigned()
    experiment_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    experiment_source: Optional[ExperimentSource] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class FailStepMetadata(Base):
    """
    FailStepMetadata
      The container for the metadata for Fail step.

    Attributes
    ----------------------
    error_message: A message that you define and then is processed and rendered by the Fail step when the error occurs.
    """

    error_message: Optional[StrPipeVar] = Unassigned()


class FeatureGroup(Base):
    """
    FeatureGroup
      Amazon SageMaker Feature Store stores features in a collection called Feature Group. A Feature Group can be visualized as a table which has rows, with a unique identifier for each row where each column in the table is a feature. In principle, a Feature Group is composed of features and values per features.

    Attributes
    ----------------------
    feature_group_arn: The Amazon Resource Name (ARN) of a FeatureGroup.
    feature_group_name: The name of the FeatureGroup.
    record_identifier_feature_name: The name of the Feature whose value uniquely identifies a Record defined in the FeatureGroup FeatureDefinitions.
    event_time_feature_name: The name of the feature that stores the EventTime of a Record in a FeatureGroup. A EventTime is point in time when a new event occurs that corresponds to the creation or update of a Record in FeatureGroup. All Records in the FeatureGroup must have a corresponding EventTime.
    feature_definitions: A list of Features. Each Feature must include a FeatureName and a FeatureType.  Valid FeatureTypes are Integral, Fractional and String.   FeatureNames cannot be any of the following: is_deleted, write_time, api_invocation_time. You can create up to 2,500 FeatureDefinitions per FeatureGroup.
    creation_time: The time a FeatureGroup was created.
    last_modified_time: A timestamp indicating the last time you updated the feature group.
    online_store_config
    offline_store_config
    role_arn: The Amazon Resource Name (ARN) of the IAM execution role used to create the feature group.
    feature_group_status: A FeatureGroup status.
    offline_store_status
    last_update_status: A value that indicates whether the feature group was updated successfully.
    failure_reason: The reason that the FeatureGroup failed to be replicated in the OfflineStore. This is failure may be due to a failure to create a FeatureGroup in or delete a FeatureGroup from the OfflineStore.
    description: A free form description of a FeatureGroup.
    online_store_replicas
    online_store_read_write_type
    last_modified_by
    created_by
    tags: Tags used to define a FeatureGroup.
    all_tags
    """

    feature_group_arn: Optional[StrPipeVar] = Unassigned()
    feature_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    record_identifier_feature_name: Optional[StrPipeVar] = Unassigned()
    event_time_feature_name: Optional[StrPipeVar] = Unassigned()
    feature_definitions: Optional[List[FeatureDefinition]] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    online_store_config: Optional[OnlineStoreConfig] = Unassigned()
    offline_store_config: Optional[OfflineStoreConfig] = Unassigned()
    role_arn: Optional[StrPipeVar] = Unassigned()
    feature_group_status: Optional[StrPipeVar] = Unassigned()
    offline_store_status: Optional[OfflineStoreStatus] = Unassigned()
    last_update_status: Optional[LastUpdateStatus] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    online_store_replicas: Optional[List[OnlineStoreReplica]] = Unassigned()
    online_store_read_write_type: Optional[StrPipeVar] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    all_tags: Optional[StrPipeVar] = Unassigned()


class FeatureGroupSummary(Base):
    """
    FeatureGroupSummary
      The name, ARN, CreationTime, FeatureGroup values, LastUpdatedTime and EnableOnlineStorage status of a FeatureGroup.

    Attributes
    ----------------------
    feature_group_name: The name of FeatureGroup.
    feature_group_arn: Unique identifier for the FeatureGroup.
    creation_time: A timestamp indicating the time of creation time of the FeatureGroup.
    feature_group_status: The status of a FeatureGroup. The status can be any of the following: Creating, Created, CreateFail, Deleting or DetailFail.
    offline_store_status: Notifies you if replicating data into the OfflineStore has failed. Returns either: Active or Blocked.
    """

    feature_group_name: Union[StrPipeVar, object]
    feature_group_arn: StrPipeVar
    creation_time: datetime.datetime
    feature_group_status: Optional[StrPipeVar] = Unassigned()
    offline_store_status: Optional[OfflineStoreStatus] = Unassigned()


class FeatureMetadata(Base):
    """
    FeatureMetadata
      The metadata for a feature. It can either be metadata that you specify, or metadata that is updated automatically.

    Attributes
    ----------------------
    feature_group_arn: The Amazon Resource Number (ARN) of the feature group.
    feature_group_name: The name of the feature group containing the feature.
    feature_name: The name of feature.
    feature_type: The data type of the feature.
    creation_time: A timestamp indicating when the feature was created.
    last_modified_time: A timestamp indicating when the feature was last modified.
    description: An optional description that you specify to better describe the feature.
    parameters: Optional key-value pairs that you specify to better describe the feature.
    all_parameters
    """

    feature_group_arn: Optional[StrPipeVar] = Unassigned()
    feature_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    feature_name: Optional[StrPipeVar] = Unassigned()
    feature_type: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    parameters: Optional[List[FeatureParameter]] = Unassigned()
    all_parameters: Optional[StrPipeVar] = Unassigned()


class Filter(Base):
    """
    Filter
      A conditional statement for a search expression that includes a resource property, a Boolean operator, and a value. Resources that match the statement are returned in the results from the Search API. If you specify a Value, but not an Operator, SageMaker uses the equals operator. In search, there are several property types:  Metrics  To define a metric filter, enter a value using the form "Metrics.&lt;name&gt;", where &lt;name&gt; is a metric name. For example, the following filter searches for training jobs with an "accuracy" metric greater than "0.9":  {   "Name": "Metrics.accuracy",   "Operator": "GreaterThan",   "Value": "0.9"   }   HyperParameters  To define a hyperparameter filter, enter a value with the form "HyperParameters.&lt;name&gt;". Decimal hyperparameter values are treated as a decimal in a comparison if the specified Value is also a decimal value. If the specified Value is an integer, the decimal hyperparameter values are treated as integers. For example, the following filter is satisfied by training jobs with a "learning_rate" hyperparameter that is less than "0.5":   {    "Name": "HyperParameters.learning_rate",    "Operator": "LessThan",    "Value": "0.5"    }   Tags  To define a tag filter, enter a value with the form Tags.&lt;key&gt;.

    Attributes
    ----------------------
    name: A resource property name. For example, TrainingJobName. For valid property names, see SearchRecord. You must specify a valid property for the resource.
    operator: A Boolean binary operator that is used to evaluate the filter. The operator field contains one of the following values:  Equals  The value of Name equals Value.  NotEquals  The value of Name doesn't equal Value.  Exists  The Name property exists.  NotExists  The Name property does not exist.  GreaterThan  The value of Name is greater than Value. Not supported for text properties.  GreaterThanOrEqualTo  The value of Name is greater than or equal to Value. Not supported for text properties.  LessThan  The value of Name is less than Value. Not supported for text properties.  LessThanOrEqualTo  The value of Name is less than or equal to Value. Not supported for text properties.  In  The value of Name is one of the comma delimited strings in Value. Only supported for text properties.  Contains  The value of Name contains the string Value. Only supported for text properties. A SearchExpression can include the Contains operator multiple times when the value of Name is one of the following:    Experiment.DisplayName     Experiment.ExperimentName     Experiment.Tags     Trial.DisplayName     Trial.TrialName     Trial.Tags     TrialComponent.DisplayName     TrialComponent.TrialComponentName     TrialComponent.Tags     TrialComponent.InputArtifacts     TrialComponent.OutputArtifacts    A SearchExpression can include only one Contains operator for all other values of Name. In these cases, if you include multiple Contains operators in the SearchExpression, the result is the following error message: "'CONTAINS' operator usage limit of 1 exceeded."
    value: A value used with Name and Operator to determine which resources satisfy the filter's condition. For numerical properties, Value must be an integer or floating-point decimal. For timestamp properties, Value must be an ISO 8601 date-time string of the following format: YYYY-mm-dd'T'HH:MM:SS.
    """

    name: StrPipeVar
    operator: Optional[StrPipeVar] = Unassigned()
    value: Optional[StrPipeVar] = Unassigned()


class FlowDefinitionSummary(Base):
    """
    FlowDefinitionSummary
      Contains summary information about the flow definition.

    Attributes
    ----------------------
    flow_definition_name: The name of the flow definition.
    flow_definition_arn: The Amazon Resource Name (ARN) of the flow definition.
    flow_definition_status: The status of the flow definition. Valid values:
    creation_time: The timestamp when SageMaker created the flow definition.
    failure_reason: The reason why the flow definition creation failed. A failure reason is returned only when the flow definition status is Failed.
    """

    flow_definition_name: Union[StrPipeVar, object]
    flow_definition_arn: StrPipeVar
    flow_definition_status: StrPipeVar
    creation_time: datetime.datetime
    failure_reason: Optional[StrPipeVar] = Unassigned()


class GetDeviceFleetReportResponse(Base):
    """
    GetDeviceFleetReportResponse

    Attributes
    ----------------------
    device_fleet_arn: The Amazon Resource Name (ARN) of the device.
    device_fleet_name: The name of the fleet.
    output_config: The output configuration for storing sample data collected by the fleet.
    description: Description of the fleet.
    report_generated: Timestamp of when the report was generated.
    device_stats: Status of devices.
    agent_versions: The versions of Edge Manager agent deployed on the fleet.
    model_stats: Status of model on device.
    """

    device_fleet_arn: StrPipeVar
    device_fleet_name: Union[StrPipeVar, object]
    output_config: Optional[EdgeOutputConfig] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    report_generated: Optional[datetime.datetime] = Unassigned()
    device_stats: Optional[DeviceStats] = Unassigned()
    agent_versions: Optional[List[AgentVersion]] = Unassigned()
    model_stats: Optional[List[EdgeModelStat]] = Unassigned()


class LabelingPortalPolicyStatement(Base):
    """
    LabelingPortalPolicyStatement

    Attributes
    ----------------------
    labeling_portal_policy_groups
    labeling_portal_policy_action
    labeling_portal_policy_resources
    """

    labeling_portal_policy_groups: List[StrPipeVar]
    labeling_portal_policy_action: StrPipeVar
    labeling_portal_policy_resources: List[StrPipeVar]


class LabelingPortalPolicy(Base):
    """
    LabelingPortalPolicy

    Attributes
    ----------------------
    labeling_portal_policy_statements
    """

    labeling_portal_policy_statements: List[LabelingPortalPolicyStatement]


class GetLineageGroupPolicyResponse(Base):
    """
    GetLineageGroupPolicyResponse

    Attributes
    ----------------------
    lineage_group_arn: The Amazon Resource Name (ARN) of the lineage group.
    resource_policy: The resource policy that gives access to the lineage group in another account.
    """

    lineage_group_arn: Optional[StrPipeVar] = Unassigned()
    resource_policy: Optional[StrPipeVar] = Unassigned()


class ScalingPolicyObjective(Base):
    """
    ScalingPolicyObjective
      An object where you specify the anticipated traffic pattern for an endpoint.

    Attributes
    ----------------------
    min_invocations_per_minute: The minimum number of expected requests to your endpoint per minute.
    max_invocations_per_minute: The maximum number of expected requests to your endpoint per minute.
    """

    min_invocations_per_minute: Optional[int] = Unassigned()
    max_invocations_per_minute: Optional[int] = Unassigned()


class ScalingPolicyMetric(Base):
    """
    ScalingPolicyMetric
      The metric for a scaling policy.

    Attributes
    ----------------------
    invocations_per_instance: The number of invocations sent to a model, normalized by InstanceCount in each ProductionVariant. 1/numberOfInstances is sent as the value on each request, where numberOfInstances is the number of active instances for the ProductionVariant behind the endpoint at the time of the request.
    model_latency: The interval of time taken by a model to respond as viewed from SageMaker. This interval includes the local communication times taken to send the request and to fetch the response from the container of a model and the time taken to complete the inference in the container.
    """

    invocations_per_instance: Optional[int] = Unassigned()
    model_latency: Optional[int] = Unassigned()


class PropertyNameQuery(Base):
    """
    PropertyNameQuery
      Part of the SuggestionQuery type. Specifies a hint for retrieving property names that begin with the specified text.

    Attributes
    ----------------------
    property_name_hint: Text that begins a property's name.
    """

    property_name_hint: StrPipeVar


class SuggestionQuery(Base):
    """
    SuggestionQuery
      Specified in the GetSearchSuggestions request. Limits the property names that are included in the response.

    Attributes
    ----------------------
    property_name_query: Defines a property name hint. Only property names that begin with the specified hint are included in the response.
    """

    property_name_query: Optional[PropertyNameQuery] = Unassigned()


class PropertyNameSuggestion(Base):
    """
    PropertyNameSuggestion
      A property name returned from a GetSearchSuggestions call that specifies a value in the PropertyNameQuery field.

    Attributes
    ----------------------
    property_name: A suggested property name based on what you entered in the search textbox in the SageMaker console.
    """

    property_name: Optional[StrPipeVar] = Unassigned()


class GitConfigForUpdate(Base):
    """
    GitConfigForUpdate
      Specifies configuration details for a Git repository when the repository is updated.

    Attributes
    ----------------------
    secret_arn: The Amazon Resource Name (ARN) of the Amazon Web Services Secrets Manager secret that contains the credentials used to access the git repository. The secret must have a staging label of AWSCURRENT and must be in the following format:  {"username": UserName, "password": Password}
    """

    secret_arn: Optional[StrPipeVar] = Unassigned()


class GroundTruthJobSummary(Base):
    """
    GroundTruthJobSummary

    Attributes
    ----------------------
    ground_truth_project_arn
    ground_truth_workflow_arn
    ground_truth_job_arn
    ground_truth_job_name
    ground_truth_job_status
    created_at
    """

    ground_truth_project_arn: Optional[StrPipeVar] = Unassigned()
    ground_truth_workflow_arn: Optional[StrPipeVar] = Unassigned()
    ground_truth_job_arn: Optional[StrPipeVar] = Unassigned()
    ground_truth_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    ground_truth_job_status: Optional[StrPipeVar] = Unassigned()
    created_at: Optional[datetime.datetime] = Unassigned()


class GroundTruthProjectSummary(Base):
    """
    GroundTruthProjectSummary

    Attributes
    ----------------------
    ground_truth_project_name
    ground_truth_project_description
    ground_truth_project_arn
    ground_truth_project_status
    created_at
    """

    ground_truth_project_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    ground_truth_project_description: Optional[StrPipeVar] = Unassigned()
    ground_truth_project_arn: Optional[StrPipeVar] = Unassigned()
    ground_truth_project_status: Optional[StrPipeVar] = Unassigned()
    created_at: Optional[datetime.datetime] = Unassigned()


class GroundTruthWorkflowSummary(Base):
    """
    GroundTruthWorkflowSummary

    Attributes
    ----------------------
    ground_truth_project_arn
    ground_truth_workflow_arn
    ground_truth_workflow_name
    created_at
    """

    ground_truth_project_arn: Optional[StrPipeVar] = Unassigned()
    ground_truth_workflow_arn: Optional[StrPipeVar] = Unassigned()
    ground_truth_workflow_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    created_at: Optional[datetime.datetime] = Unassigned()


class HubContentInfo(Base):
    """
    HubContentInfo
      Information about hub content.

    Attributes
    ----------------------
    hub_content_name: The name of the hub content.
    hub_content_arn: The Amazon Resource Name (ARN) of the hub content.
    sage_maker_public_hub_content_arn: The ARN of the public hub content.
    hub_content_version: The version of the hub content.
    hub_content_type: The type of hub content.
    document_schema_version: The version of the hub content document schema.
    hub_content_display_name: The display name of the hub content.
    hub_content_description: A description of the hub content.
    support_status: The support status of the hub content.
    hub_content_search_keywords: The searchable keywords for the hub content.
    hub_content_status: The status of the hub content.
    creation_time: The date and time that the hub content was created.
    original_creation_time: The date and time when the hub content was originally created, before any updates or revisions.
    """

    hub_content_name: Union[StrPipeVar, object]
    hub_content_arn: StrPipeVar
    hub_content_version: StrPipeVar
    hub_content_type: StrPipeVar
    document_schema_version: StrPipeVar
    hub_content_status: StrPipeVar
    creation_time: datetime.datetime
    sage_maker_public_hub_content_arn: Optional[StrPipeVar] = Unassigned()
    hub_content_display_name: Optional[StrPipeVar] = Unassigned()
    hub_content_description: Optional[StrPipeVar] = Unassigned()
    support_status: Optional[StrPipeVar] = Unassigned()
    hub_content_search_keywords: Optional[List[StrPipeVar]] = Unassigned()
    original_creation_time: Optional[datetime.datetime] = Unassigned()


class HubInfo(Base):
    """
    HubInfo
      Information about a hub.

    Attributes
    ----------------------
    hub_name: The name of the hub.
    hub_arn: The Amazon Resource Name (ARN) of the hub.
    hub_display_name: The display name of the hub.
    hub_description: A description of the hub.
    hub_search_keywords: The searchable keywords for the hub.
    hub_status: The status of the hub.
    creation_time: The date and time that the hub was created.
    last_modified_time: The date and time that the hub was last modified.
    """

    hub_name: Union[StrPipeVar, object]
    hub_arn: StrPipeVar
    hub_status: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    hub_display_name: Optional[StrPipeVar] = Unassigned()
    hub_description: Optional[StrPipeVar] = Unassigned()
    hub_search_keywords: Optional[List[StrPipeVar]] = Unassigned()


class HumanTaskUiSummary(Base):
    """
    HumanTaskUiSummary
      Container for human task user interface information.

    Attributes
    ----------------------
    human_task_ui_name: The name of the human task user interface.
    human_task_ui_arn: The Amazon Resource Name (ARN) of the human task user interface.
    human_task_ui_status
    creation_time: A timestamp when SageMaker created the human task user interface.
    """

    human_task_ui_name: Union[StrPipeVar, object]
    human_task_ui_arn: StrPipeVar
    creation_time: datetime.datetime
    human_task_ui_status: Optional[StrPipeVar] = Unassigned()


class HyperParameterTuningJobSearchEntity(Base):
    """
    HyperParameterTuningJobSearchEntity
      An entity returned by the SearchRecord API containing the properties of a hyperparameter tuning job.

    Attributes
    ----------------------
    hyper_parameter_tuning_job_name: The name of a hyperparameter tuning job.
    hyper_parameter_tuning_job_arn: The Amazon Resource Name (ARN) of a hyperparameter tuning job.
    hyper_parameter_tuning_job_config
    training_job_definition
    training_job_definitions: The job definitions included in a hyperparameter tuning job.
    hyper_parameter_tuning_job_status: The status of a hyperparameter tuning job.
    creation_time: The time that a hyperparameter tuning job was created.
    hyper_parameter_tuning_end_time: The time that a hyperparameter tuning job ended.
    last_modified_time: The time that a hyperparameter tuning job was last modified.
    training_job_status_counters
    objective_status_counters
    best_training_job
    overall_best_training_job
    warm_start_config
    failure_reason: The error that was created when a hyperparameter tuning job failed.
    tuning_job_completion_details: Information about either a current or completed hyperparameter tuning job.
    consumed_resources: The total amount of resources consumed by a hyperparameter tuning job.
    tags: The tags associated with a hyperparameter tuning job. For more information see Tagging Amazon Web Services resources.
    """

    hyper_parameter_tuning_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    hyper_parameter_tuning_job_arn: Optional[StrPipeVar] = Unassigned()
    hyper_parameter_tuning_job_config: Optional[HyperParameterTuningJobConfig] = Unassigned()
    training_job_definition: Optional[HyperParameterTrainingJobDefinition] = Unassigned()
    training_job_definitions: Optional[List[HyperParameterTrainingJobDefinition]] = Unassigned()
    hyper_parameter_tuning_job_status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    hyper_parameter_tuning_end_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    training_job_status_counters: Optional[TrainingJobStatusCounters] = Unassigned()
    objective_status_counters: Optional[ObjectiveStatusCounters] = Unassigned()
    best_training_job: Optional[HyperParameterTrainingJobSummary] = Unassigned()
    overall_best_training_job: Optional[HyperParameterTrainingJobSummary] = Unassigned()
    warm_start_config: Optional[HyperParameterTuningJobWarmStartConfig] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    tuning_job_completion_details: Optional[HyperParameterTuningJobCompletionDetails] = Unassigned()
    consumed_resources: Optional[HyperParameterTuningJobConsumedResources] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class HyperParameterTuningJobSummary(Base):
    """
    HyperParameterTuningJobSummary
      Provides summary information about a hyperparameter tuning job.

    Attributes
    ----------------------
    hyper_parameter_tuning_job_name: The name of the tuning job.
    hyper_parameter_tuning_job_arn: The Amazon Resource Name (ARN) of the tuning job.
    hyper_parameter_tuning_job_status: The status of the tuning job.
    strategy: Specifies the search strategy hyperparameter tuning uses to choose which hyperparameters to evaluate at each iteration.
    creation_time: The date and time that the tuning job was created.
    hyper_parameter_tuning_end_time: The date and time that the tuning job ended.
    last_modified_time: The date and time that the tuning job was modified.
    training_job_status_counters: The TrainingJobStatusCounters object that specifies the numbers of training jobs, categorized by status, that this tuning job launched.
    objective_status_counters: The ObjectiveStatusCounters object that specifies the numbers of training jobs, categorized by objective metric status, that this tuning job launched.
    resource_limits: The ResourceLimits object that specifies the maximum number of training jobs and parallel training jobs allowed for this tuning job.
    """

    hyper_parameter_tuning_job_name: Union[StrPipeVar, object]
    hyper_parameter_tuning_job_arn: StrPipeVar
    hyper_parameter_tuning_job_status: StrPipeVar
    strategy: StrPipeVar
    creation_time: datetime.datetime
    training_job_status_counters: TrainingJobStatusCounters
    objective_status_counters: ObjectiveStatusCounters
    hyper_parameter_tuning_end_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    resource_limits: Optional[ResourceLimits] = Unassigned()


class Image(Base):
    """
    Image
      A SageMaker AI image. A SageMaker AI image represents a set of container images that are derived from a common base container image. Each of these container images is represented by a SageMaker AI ImageVersion.

    Attributes
    ----------------------
    creation_time: When the image was created.
    description: The description of the image.
    display_name: The name of the image as displayed.
    failure_reason: When a create, update, or delete operation fails, the reason for the failure.
    image_arn: The ARN of the image.
    image_name: The name of the image.
    image_status: The status of the image.
    last_modified_time: When the image was last modified.
    """

    creation_time: datetime.datetime
    image_arn: StrPipeVar
    image_name: Union[StrPipeVar, object]
    image_status: StrPipeVar
    last_modified_time: datetime.datetime
    description: Optional[StrPipeVar] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()


class ImageSearchShape(Base):
    """
    ImageSearchShape

    Attributes
    ----------------------
    creation_time
    description
    display_name
    failure_reason
    image_arn
    image_name
    image_status
    last_modified_time
    role_arn
    tags
    """

    creation_time: Optional[datetime.datetime] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    image_arn: Optional[StrPipeVar] = Unassigned()
    image_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    image_status: Optional[StrPipeVar] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    role_arn: Optional[StrPipeVar] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class ImageVersion(Base):
    """
    ImageVersion
      A version of a SageMaker AI Image. A version represents an existing container image.

    Attributes
    ----------------------
    creation_time: When the version was created.
    failure_reason: When a create or delete operation fails, the reason for the failure.
    image_arn: The ARN of the image the version is based on.
    image_version_arn: The ARN of the version.
    image_version_status: The status of the version.
    last_modified_time: When the version was last modified.
    version: The version number.
    """

    creation_time: datetime.datetime
    image_arn: StrPipeVar
    image_version_arn: StrPipeVar
    image_version_status: StrPipeVar
    last_modified_time: datetime.datetime
    version: int
    failure_reason: Optional[StrPipeVar] = Unassigned()


class ImageVersionSearchShape(Base):
    """
    ImageVersionSearchShape

    Attributes
    ----------------------
    base_image
    container_image
    creation_time
    failure_reason
    image_arn
    image_version_arn
    image_version_status
    last_modified_time
    version
    vendor_guidance
    job_type
    ml_framework
    programming_lang
    processor
    horovod
    soci_image
    release_notes
    override_alias_image_version
    """

    base_image: Optional[StrPipeVar] = Unassigned()
    container_image: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    image_arn: Optional[StrPipeVar] = Unassigned()
    image_version_arn: Optional[StrPipeVar] = Unassigned()
    image_version_status: Optional[StrPipeVar] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    version: Optional[int] = Unassigned()
    vendor_guidance: Optional[StrPipeVar] = Unassigned()
    job_type: Optional[StrPipeVar] = Unassigned()
    ml_framework: Optional[StrPipeVar] = Unassigned()
    programming_lang: Optional[StrPipeVar] = Unassigned()
    processor: Optional[StrPipeVar] = Unassigned()
    horovod: Optional[bool] = Unassigned()
    soci_image: Optional[bool] = Unassigned()
    release_notes: Optional[StrPipeVar] = Unassigned()
    override_alias_image_version: Optional[bool] = Unassigned()


class InferenceComponentMetadata(Base):
    """
    InferenceComponentMetadata

    Attributes
    ----------------------
    arn
    """

    arn: Optional[StrPipeVar] = Unassigned()


class InferenceComponentSummary(Base):
    """
    InferenceComponentSummary
      A summary of the properties of an inference component.

    Attributes
    ----------------------
    creation_time: The time when the inference component was created.
    inference_component_arn: The Amazon Resource Name (ARN) of the inference component.
    inference_component_name: The name of the inference component.
    endpoint_arn: The Amazon Resource Name (ARN) of the endpoint that hosts the inference component.
    endpoint_name: The name of the endpoint that hosts the inference component.
    variant_name: The name of the production variant that hosts the inference component.
    inference_component_status: The status of the inference component.
    last_modified_time: The time when the inference component was last updated.
    """

    creation_time: datetime.datetime
    inference_component_arn: StrPipeVar
    inference_component_name: Union[StrPipeVar, object]
    endpoint_arn: StrPipeVar
    endpoint_name: Union[StrPipeVar, object]
    variant_name: StrPipeVar
    last_modified_time: datetime.datetime
    inference_component_status: Optional[StrPipeVar] = Unassigned()


class InferenceExperimentSummary(Base):
    """
    InferenceExperimentSummary
      Lists a summary of properties of an inference experiment.

    Attributes
    ----------------------
    name: The name of the inference experiment.
    type: The type of the inference experiment.
    schedule: The duration for which the inference experiment ran or will run. The maximum duration that you can set for an inference experiment is 30 days.
    status: The status of the inference experiment.
    status_reason: The error message for the inference experiment status result.
    description: The description of the inference experiment.
    creation_time: The timestamp at which the inference experiment was created.
    completion_time: The timestamp at which the inference experiment was completed.
    last_modified_time: The timestamp when you last modified the inference experiment.
    role_arn:  The ARN of the IAM role that Amazon SageMaker can assume to access model artifacts and container images, and manage Amazon SageMaker Inference endpoints for model deployment.
    arn
    """

    name: StrPipeVar
    type: StrPipeVar
    status: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    schedule: Optional[InferenceExperimentSchedule] = Unassigned()
    status_reason: Optional[StrPipeVar] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    completion_time: Optional[datetime.datetime] = Unassigned()
    role_arn: Optional[StrPipeVar] = Unassigned()
    arn: Optional[StrPipeVar] = Unassigned()


class InferenceRecommendationsJob(Base):
    """
    InferenceRecommendationsJob
      A structure that contains a list of recommendation jobs.

    Attributes
    ----------------------
    job_name: The name of the job.
    job_description: The job description.
    job_type: The recommendation job type.
    job_arn: The Amazon Resource Name (ARN) of the recommendation job.
    status: The status of the job.
    creation_time: A timestamp that shows when the job was created.
    completion_time: A timestamp that shows when the job completed.
    role_arn: The Amazon Resource Name (ARN) of an IAM role that enables Amazon SageMaker to perform tasks on your behalf.
    last_modified_time: A timestamp that shows when the job was last modified.
    failure_reason: If the job fails, provides information why the job failed.
    model_name: The name of the created model.
    sample_payload_url: The Amazon Simple Storage Service (Amazon S3) path where the sample payload is stored. This path must point to a single gzip compressed tar archive (.tar.gz suffix).
    model_package_version_arn: The Amazon Resource Name (ARN) of a versioned model package.
    benchmark_results_output_config
    """

    job_name: StrPipeVar
    job_description: StrPipeVar
    job_type: StrPipeVar
    job_arn: StrPipeVar
    status: StrPipeVar
    creation_time: datetime.datetime
    role_arn: StrPipeVar
    last_modified_time: datetime.datetime
    completion_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    sample_payload_url: Optional[StrPipeVar] = Unassigned()
    model_package_version_arn: Optional[StrPipeVar] = Unassigned()
    benchmark_results_output_config: Optional[BenchmarkResultsOutputConfig] = Unassigned()


class RecommendationJobInferenceBenchmark(Base):
    """
    RecommendationJobInferenceBenchmark
      The details for a specific benchmark from an Inference Recommender job.

    Attributes
    ----------------------
    metrics
    endpoint_metrics
    endpoint_configuration
    model_configuration
    failure_reason: The reason why a benchmark failed.
    invocation_end_time: A timestamp that shows when the benchmark completed.
    invocation_start_time: A timestamp that shows when the benchmark started.
    """

    model_configuration: ModelConfiguration
    metrics: Optional[RecommendationMetrics] = Unassigned()
    endpoint_metrics: Optional[InferenceMetrics] = Unassigned()
    endpoint_configuration: Optional[EndpointOutputConfiguration] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    invocation_end_time: Optional[datetime.datetime] = Unassigned()
    invocation_start_time: Optional[datetime.datetime] = Unassigned()


class InferenceRecommendationsJobStep(Base):
    """
    InferenceRecommendationsJobStep
      A returned array object for the Steps response field in the ListInferenceRecommendationsJobSteps API command.

    Attributes
    ----------------------
    step_type: The type of the subtask.  BENCHMARK: Evaluate the performance of your model on different instance types.
    job_name: The name of the Inference Recommender job.
    status: The current status of the benchmark.
    inference_benchmark: The details for a specific benchmark.
    """

    step_type: StrPipeVar
    job_name: StrPipeVar
    status: StrPipeVar
    inference_benchmark: Optional[RecommendationJobInferenceBenchmark] = Unassigned()


class InferenceServiceConfig(Base):
    """
    InferenceServiceConfig

    Attributes
    ----------------------
    request_status
    execution_role_arn
    """

    request_status: StrPipeVar
    execution_role_arn: Optional[StrPipeVar] = Unassigned()


class InstanceGroupHealthCheckConfiguration(Base):
    """
    InstanceGroupHealthCheckConfiguration

    Attributes
    ----------------------
    instance_group_name
    instance_ids
    deep_health_checks
    """

    instance_group_name: StrPipeVar
    instance_ids: Optional[List[StrPipeVar]] = Unassigned()
    deep_health_checks: Optional[List[StrPipeVar]] = Unassigned()


class LabelCountersForWorkteam(Base):
    """
    LabelCountersForWorkteam
      Provides counts for human-labeled tasks in the labeling job.

    Attributes
    ----------------------
    human_labeled: The total number of data objects labeled by a human worker.
    pending_human: The total number of data objects that need to be labeled by a human worker.
    total: The total number of tasks in the labeling job.
    """

    human_labeled: Optional[int] = Unassigned()
    pending_human: Optional[int] = Unassigned()
    total: Optional[int] = Unassigned()


class LabelingJobForWorkteamSummary(Base):
    """
    LabelingJobForWorkteamSummary
      Provides summary information for a work team.

    Attributes
    ----------------------
    labeling_job_name: The name of the labeling job that the work team is assigned to.
    job_reference_code: A unique identifier for a labeling job. You can use this to refer to a specific labeling job.
    work_requester_account_id: The Amazon Web Services account ID of the account used to start the labeling job.
    creation_time: The date and time that the labeling job was created.
    label_counters: Provides information about the progress of a labeling job.
    number_of_human_workers_per_data_object: The configured number of workers per data object.
    """

    job_reference_code: StrPipeVar
    work_requester_account_id: StrPipeVar
    creation_time: datetime.datetime
    labeling_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    label_counters: Optional[LabelCountersForWorkteam] = Unassigned()
    number_of_human_workers_per_data_object: Optional[int] = Unassigned()


class LabelingJobSummary(Base):
    """
    LabelingJobSummary
      Provides summary information about a labeling job.

    Attributes
    ----------------------
    labeling_job_name: The name of the labeling job.
    labeling_job_arn: The Amazon Resource Name (ARN) assigned to the labeling job when it was created.
    creation_time: The date and time that the job was created (timestamp).
    last_modified_time: The date and time that the job was last modified (timestamp).
    labeling_job_status: The current status of the labeling job.
    label_counters: Counts showing the progress of the labeling job.
    workteam_arn: The Amazon Resource Name (ARN) of the work team assigned to the job.
    pre_human_task_lambda_arn: The Amazon Resource Name (ARN) of a Lambda function. The function is run before each data object is sent to a worker.
    annotation_consolidation_lambda_arn: The Amazon Resource Name (ARN) of the Lambda function used to consolidate the annotations from individual workers into a label for a data object. For more information, see Annotation Consolidation.
    failure_reason: If the LabelingJobStatus field is Failed, this field contains a description of the error.
    labeling_job_output: The location of the output produced by the labeling job.
    input_config: Input configuration for the labeling job.
    """

    labeling_job_name: Union[StrPipeVar, object]
    labeling_job_arn: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    labeling_job_status: StrPipeVar
    label_counters: LabelCounters
    workteam_arn: StrPipeVar
    pre_human_task_lambda_arn: Optional[StrPipeVar] = Unassigned()
    annotation_consolidation_lambda_arn: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    labeling_job_output: Optional[LabelingJobOutput] = Unassigned()
    input_config: Optional[LabelingJobInputConfig] = Unassigned()


class LambdaStepMetadata(Base):
    """
    LambdaStepMetadata
      Metadata for a Lambda step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the Lambda function that was run by this step execution.
    output_parameters: A list of the output parameters of the Lambda step.
    """

    arn: Optional[StrPipeVar] = Unassigned()
    output_parameters: Optional[List[OutputParameter]] = Unassigned()


class LineageGroupSummary(Base):
    """
    LineageGroupSummary
      Lists a summary of the properties of a lineage group. A lineage group provides a group of shareable lineage entity resources.

    Attributes
    ----------------------
    lineage_group_arn: The Amazon Resource Name (ARN) of the lineage group resource.
    lineage_group_name: The name or Amazon Resource Name (ARN) of the lineage group.
    display_name: The display name of the lineage group summary.
    creation_time: The creation time of the lineage group summary.
    last_modified_time: The last modified time of the lineage group summary.
    """

    lineage_group_arn: Optional[StrPipeVar] = Unassigned()
    lineage_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class LineageMetadata(Base):
    """
    LineageMetadata

    Attributes
    ----------------------
    action_arns
    artifact_arns
    context_arns
    associations
    """

    action_arns: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    artifact_arns: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    context_arns: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    associations: Optional[List[AssociationInfo]] = Unassigned()


class MonitoringJobDefinitionSummary(Base):
    """
    MonitoringJobDefinitionSummary
      Summary information about a monitoring job.

    Attributes
    ----------------------
    monitoring_job_definition_name: The name of the monitoring job.
    monitoring_job_definition_arn: The Amazon Resource Name (ARN) of the monitoring job.
    creation_time: The time that the monitoring job was created.
    endpoint_name: The name of the endpoint that the job monitors.
    variant_name
    """

    monitoring_job_definition_name: StrPipeVar
    monitoring_job_definition_arn: StrPipeVar
    creation_time: datetime.datetime
    endpoint_name: Union[StrPipeVar, object]
    variant_name: Optional[StrPipeVar] = Unassigned()


class MlflowAppSummary(Base):
    """
    MlflowAppSummary

    Attributes
    ----------------------
    arn
    name
    status
    creation_time
    last_modified_time
    mlflow_version
    """

    arn: Optional[StrPipeVar] = Unassigned()
    name: Optional[StrPipeVar] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    mlflow_version: Optional[StrPipeVar] = Unassigned()


class TrackingServerSummary(Base):
    """
    TrackingServerSummary
      The summary of the tracking server to list.

    Attributes
    ----------------------
    tracking_server_arn: The ARN of a listed tracking server.
    tracking_server_name: The name of a listed tracking server.
    creation_time: The creation time of a listed tracking server.
    last_modified_time: The last modified time of a listed tracking server.
    tracking_server_status: The creation status of a listed tracking server.
    is_active: The activity status of a listed tracking server.
    mlflow_version: The MLflow version used for a listed tracking server.
    """

    tracking_server_arn: Optional[StrPipeVar] = Unassigned()
    tracking_server_name: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    tracking_server_status: Optional[StrPipeVar] = Unassigned()
    is_active: Optional[StrPipeVar] = Unassigned()
    mlflow_version: Optional[StrPipeVar] = Unassigned()


class ModelCardExportJobSummary(Base):
    """
    ModelCardExportJobSummary
      The summary of the Amazon SageMaker Model Card export job.

    Attributes
    ----------------------
    model_card_export_job_name: The name of the model card export job.
    model_card_export_job_arn: The Amazon Resource Name (ARN) of the model card export job.
    status: The completion status of the model card export job.
    model_card_name: The name of the model card that the export job exports.
    model_card_version: The version of the model card that the export job exports.
    created_at: The date and time that the model card export job was created.
    last_modified_at: The date and time that the model card export job was last modified..
    """

    model_card_export_job_name: Union[StrPipeVar, object]
    model_card_export_job_arn: StrPipeVar
    status: StrPipeVar
    model_card_name: Union[StrPipeVar, object]
    model_card_version: int
    created_at: datetime.datetime
    last_modified_at: datetime.datetime


class ModelCardVersionSummary(Base):
    """
    ModelCardVersionSummary
      A summary of a specific version of the model card.

    Attributes
    ----------------------
    model_card_name: The name of the model card.
    model_card_arn: The Amazon Resource Name (ARN) of the model card.
    model_card_status: The approval status of the model card version within your organization. Different organizations might have different criteria for model card review and approval.    Draft: The model card is a work in progress.    PendingReview: The model card is pending review.    Approved: The model card is approved.    Archived: The model card is archived. No more updates should be made to the model card, but it can still be exported.
    model_card_version: A version of the model card.
    creation_time: The date and time that the model card version was created.
    last_modified_time: The time date and time that the model card version was last modified.
    """

    model_card_name: Union[StrPipeVar, object]
    model_card_arn: StrPipeVar
    model_card_status: StrPipeVar
    model_card_version: int
    creation_time: datetime.datetime
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class ModelCardSummary(Base):
    """
    ModelCardSummary
      A summary of the model card.

    Attributes
    ----------------------
    model_card_name: The name of the model card.
    model_card_arn: The Amazon Resource Name (ARN) of the model card.
    model_card_status: The approval status of the model card within your organization. Different organizations might have different criteria for model card review and approval.    Draft: The model card is a work in progress.    PendingReview: The model card is pending review.    Approved: The model card is approved.    Archived: The model card is archived. No more updates should be made to the model card, but it can still be exported.
    creation_time: The date and time that the model card was created.
    last_modified_time: The date and time that the model card was last modified.
    """

    model_card_name: Union[StrPipeVar, object]
    model_card_arn: StrPipeVar
    model_card_status: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class ModelMetadataFilter(Base):
    """
    ModelMetadataFilter
      Part of the search expression. You can specify the name and value (domain, task, framework, framework version, task, and model).

    Attributes
    ----------------------
    name: The name of the of the model to filter by.
    value: The value to filter the model metadata.
    """

    name: StrPipeVar
    value: StrPipeVar


class ModelMetadataSearchExpression(Base):
    """
    ModelMetadataSearchExpression
      One or more filters that searches for the specified resource or resources in a search. All resource objects that satisfy the expression's condition are included in the search results

    Attributes
    ----------------------
    filters: A list of filter objects.
    """

    filters: Optional[List[ModelMetadataFilter]] = Unassigned()


class ModelMetadataSummary(Base):
    """
    ModelMetadataSummary
      A summary of the model metadata.

    Attributes
    ----------------------
    domain: The machine learning domain of the model.
    framework: The machine learning framework of the model.
    task: The machine learning task of the model.
    model: The name of the model.
    framework_version: The framework version of the model.
    """

    domain: StrPipeVar
    framework: StrPipeVar
    task: StrPipeVar
    model: StrPipeVar
    framework_version: StrPipeVar


class ModelPackageGroupSummary(Base):
    """
    ModelPackageGroupSummary
      Summary information about a model group.

    Attributes
    ----------------------
    model_package_group_name: The name of the model group.
    model_package_group_arn: The Amazon Resource Name (ARN) of the model group.
    model_package_group_description: A description of the model group.
    creation_time: The time that the model group was created.
    model_package_group_status: The status of the model group.
    """

    model_package_group_name: Union[StrPipeVar, object]
    model_package_group_arn: StrPipeVar
    creation_time: datetime.datetime
    model_package_group_status: StrPipeVar
    model_package_group_description: Optional[StrPipeVar] = Unassigned()


class ModelPackageSummary(Base):
    """
    ModelPackageSummary
      Provides summary information about a model package.

    Attributes
    ----------------------
    model_package_name: The name of the model package.
    model_package_group_name: If the model package is a versioned model, the model group that the versioned model belongs to.
    model_package_version: If the model package is a versioned model, the version of the model.
    model_package_arn: The Amazon Resource Name (ARN) of the model package.
    model_package_description: A brief description of the model package.
    creation_time: A timestamp that shows when the model package was created.
    model_package_status: The overall status of the model package.
    model_approval_status: The approval status of the model. This can be one of the following values.    APPROVED - The model is approved    REJECTED - The model is rejected.    PENDING_MANUAL_APPROVAL - The model is waiting for manual approval.
    model_life_cycle
    model_package_registration_type
    """

    model_package_arn: StrPipeVar
    creation_time: datetime.datetime
    model_package_status: StrPipeVar
    model_package_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_package_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_package_version: Optional[int] = Unassigned()
    model_package_description: Optional[StrPipeVar] = Unassigned()
    model_approval_status: Optional[StrPipeVar] = Unassigned()
    model_life_cycle: Optional[ModelLifeCycle] = Unassigned()
    model_package_registration_type: Optional[StrPipeVar] = Unassigned()


class ModelSummary(Base):
    """
    ModelSummary
      Provides summary information about a model.

    Attributes
    ----------------------
    model_name: The name of the model that you want a summary for.
    model_arn: The Amazon Resource Name (ARN) of the model.
    creation_time: A timestamp that indicates when the model was created.
    """

    model_name: Union[StrPipeVar, object]
    model_arn: StrPipeVar
    creation_time: datetime.datetime


class MonitoringAlertHistorySummary(Base):
    """
    MonitoringAlertHistorySummary
      Provides summary information of an alert's history.

    Attributes
    ----------------------
    monitoring_schedule_name: The name of a monitoring schedule.
    monitoring_alert_name: The name of a monitoring alert.
    creation_time: A timestamp that indicates when the first alert transition occurred in an alert history. An alert transition can be from status InAlert to OK, or from OK to InAlert.
    alert_status: The current alert status of an alert.
    """

    monitoring_schedule_name: Union[StrPipeVar, object]
    monitoring_alert_name: Union[StrPipeVar, object]
    creation_time: datetime.datetime
    alert_status: StrPipeVar


class ModelDashboardIndicatorAction(Base):
    """
    ModelDashboardIndicatorAction
      An alert action taken to light up an icon on the Amazon SageMaker Model Dashboard when an alert goes into InAlert status.

    Attributes
    ----------------------
    enabled: Indicates whether the alert action is turned on.
    """

    enabled: Optional[bool] = Unassigned()


class MonitoringAlertActions(Base):
    """
    MonitoringAlertActions
      A list of alert actions taken in response to an alert going into InAlert status.

    Attributes
    ----------------------
    model_dashboard_indicator: An alert action taken to light up an icon on the Model Dashboard when an alert goes into InAlert status.
    """

    model_dashboard_indicator: Optional[ModelDashboardIndicatorAction] = Unassigned()


class MonitoringAlertSummary(Base):
    """
    MonitoringAlertSummary
      Provides summary information about a monitor alert.

    Attributes
    ----------------------
    monitoring_alert_name: The name of a monitoring alert.
    creation_time: A timestamp that indicates when a monitor alert was created.
    last_modified_time: A timestamp that indicates when a monitor alert was last updated.
    alert_status: The current status of an alert.
    datapoints_to_alert: Within EvaluationPeriod, how many execution failures will raise an alert.
    evaluation_period: The number of most recent monitoring executions to consider when evaluating alert status.
    actions: A list of alert actions taken in response to an alert going into InAlert status.
    """

    monitoring_alert_name: Union[StrPipeVar, object]
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    alert_status: StrPipeVar
    datapoints_to_alert: int
    evaluation_period: int
    actions: MonitoringAlertActions


class MonitoringScheduleSummary(Base):
    """
    MonitoringScheduleSummary
      Summarizes the monitoring schedule.

    Attributes
    ----------------------
    monitoring_schedule_name: The name of the monitoring schedule.
    monitoring_schedule_arn: The Amazon Resource Name (ARN) of the monitoring schedule.
    creation_time: The creation time of the monitoring schedule.
    last_modified_time: The last time the monitoring schedule was modified.
    monitoring_schedule_status: The status of the monitoring schedule.
    endpoint_name: The name of the endpoint using the monitoring schedule.
    monitoring_job_definition_name: The name of the monitoring job definition that the schedule is for.
    monitoring_type: The type of the monitoring job definition that the schedule is for.
    variant_name
    """

    monitoring_schedule_name: Union[StrPipeVar, object]
    monitoring_schedule_arn: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    monitoring_schedule_status: StrPipeVar
    endpoint_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    monitoring_job_definition_name: Optional[StrPipeVar] = Unassigned()
    monitoring_type: Optional[StrPipeVar] = Unassigned()
    variant_name: Optional[StrPipeVar] = Unassigned()


class NotebookInstanceLifecycleConfigSummary(Base):
    """
    NotebookInstanceLifecycleConfigSummary
      Provides a summary of a notebook instance lifecycle configuration.

    Attributes
    ----------------------
    notebook_instance_lifecycle_config_name: The name of the lifecycle configuration.
    notebook_instance_lifecycle_config_arn: The Amazon Resource Name (ARN) of the lifecycle configuration.
    creation_time: A timestamp that tells when the lifecycle configuration was created.
    last_modified_time: A timestamp that tells when the lifecycle configuration was last modified.
    """

    notebook_instance_lifecycle_config_name: Union[StrPipeVar, object]
    notebook_instance_lifecycle_config_arn: StrPipeVar
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class NotebookInstanceSummary(Base):
    """
    NotebookInstanceSummary
      Provides summary information for an SageMaker AI notebook instance.

    Attributes
    ----------------------
    notebook_instance_name: The name of the notebook instance that you want a summary for.
    notebook_instance_arn: The Amazon Resource Name (ARN) of the notebook instance.
    notebook_instance_status: The status of the notebook instance.
    url: The URL that you use to connect to the Jupyter notebook running in your notebook instance.
    instance_type: The type of ML compute instance that the notebook instance is running on.
    creation_time: A timestamp that shows when the notebook instance was created.
    last_modified_time: A timestamp that shows when the notebook instance was last modified.
    notebook_instance_lifecycle_config_name: The name of a notebook instance lifecycle configuration associated with this notebook instance. For information about notebook instance lifestyle configurations, see Step 2.1: (Optional) Customize a Notebook Instance.
    default_code_repository: The Git repository associated with the notebook instance as its default code repository. This can be either the name of a Git repository stored as a resource in your account, or the URL of a Git repository in Amazon Web Services CodeCommit or in any other Git repository. When you open a notebook instance, it opens in the directory that contains this repository. For more information, see Associating Git Repositories with SageMaker AI Notebook Instances.
    additional_code_repositories: An array of up to three Git repositories associated with the notebook instance. These can be either the names of Git repositories stored as resources in your account, or the URL of Git repositories in Amazon Web Services CodeCommit or in any other Git repository. These repositories are cloned at the same level as the default repository of your notebook instance. For more information, see Associating Git Repositories with SageMaker AI Notebook Instances.
    """

    notebook_instance_name: Union[StrPipeVar, object]
    notebook_instance_arn: StrPipeVar
    notebook_instance_status: Optional[StrPipeVar] = Unassigned()
    url: Optional[StrPipeVar] = Unassigned()
    instance_type: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    notebook_instance_lifecycle_config_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    default_code_repository: Optional[StrPipeVar] = Unassigned()
    additional_code_repositories: Optional[List[StrPipeVar]] = Unassigned()


class OptimizationJobSummary(Base):
    """
    OptimizationJobSummary
      Summarizes an optimization job by providing some of its key properties.

    Attributes
    ----------------------
    optimization_job_name: The name that you assigned to the optimization job.
    optimization_job_arn: The Amazon Resource Name (ARN) of the optimization job.
    creation_time: The time when you created the optimization job.
    optimization_job_status: The current status of the optimization job.
    optimization_start_time: The time when the optimization job started.
    optimization_end_time: The time when the optimization job finished processing.
    last_modified_time: The time when the optimization job was last updated.
    deployment_instance_type: The type of instance that hosts the optimized model that you create with the optimization job.
    max_instance_count
    optimization_types: The optimization techniques that are applied by the optimization job.
    """

    optimization_job_name: Union[StrPipeVar, object]
    optimization_job_arn: StrPipeVar
    creation_time: datetime.datetime
    optimization_job_status: StrPipeVar
    deployment_instance_type: StrPipeVar
    optimization_types: List[StrPipeVar]
    optimization_start_time: Optional[datetime.datetime] = Unassigned()
    optimization_end_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    max_instance_count: Optional[int] = Unassigned()


class PartnerAppSummary(Base):
    """
    PartnerAppSummary
      A subset of information related to a SageMaker Partner AI App. This information is used as part of the ListPartnerApps API response.

    Attributes
    ----------------------
    arn: The ARN of the SageMaker Partner AI App.
    name: The name of the SageMaker Partner AI App.
    type: The type of SageMaker Partner AI App to create. Must be one of the following: lakera-guard, comet, deepchecks-llm-evaluation, or fiddler.
    status: The status of the SageMaker Partner AI App.
    creation_time: The creation time of the SageMaker Partner AI App.
    """

    arn: Optional[StrPipeVar] = Unassigned()
    name: Optional[StrPipeVar] = Unassigned()
    type: Optional[StrPipeVar] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()


class TrainingJobStepMetadata(Base):
    """
    TrainingJobStepMetadata
      Metadata for a training job step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the training job that was run by this step execution.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class ProcessingJobStepMetadata(Base):
    """
    ProcessingJobStepMetadata
      Metadata for a processing job step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the processing job.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class TransformJobStepMetadata(Base):
    """
    TransformJobStepMetadata
      Metadata for a transform job step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the transform job that was run by this step execution.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class TuningJobStepMetaData(Base):
    """
    TuningJobStepMetaData
      Metadata for a tuning step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the tuning job that was run by this step execution.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class ModelStepMetadata(Base):
    """
    ModelStepMetadata
      Metadata for Model steps.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the created model.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class RegisterModelStepMetadata(Base):
    """
    RegisterModelStepMetadata
      Metadata for a register model job step.

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the model package.
    """

    arn: Optional[StrPipeVar] = Unassigned()


class QualityCheckStepMetadata(Base):
    """
    QualityCheckStepMetadata
      Container for the metadata for a Quality check step. For more information, see the topic on QualityCheck step in the Amazon SageMaker Developer Guide.

    Attributes
    ----------------------
    check_type: The type of the Quality check step.
    baseline_used_for_drift_check_statistics: The Amazon S3 URI of the baseline statistics file used for the drift check.
    baseline_used_for_drift_check_constraints: The Amazon S3 URI of the baseline constraints file used for the drift check.
    calculated_baseline_statistics: The Amazon S3 URI of the newly calculated baseline statistics file.
    calculated_baseline_constraints: The Amazon S3 URI of the newly calculated baseline constraints file.
    model_package_group_name: The model package group name.
    violation_report: The Amazon S3 URI of violation report if violations are detected.
    check_job_arn: The Amazon Resource Name (ARN) of the Quality check processing job that was run by this step execution.
    skip_check: This flag indicates if the drift check against the previous baseline will be skipped or not. If it is set to False, the previous baseline of the configured check type must be available.
    register_new_baseline: This flag indicates if a newly calculated baseline can be accessed through step properties BaselineUsedForDriftCheckConstraints and BaselineUsedForDriftCheckStatistics. If it is set to False, the previous baseline of the configured check type must also be available. These can be accessed through the BaselineUsedForDriftCheckConstraints and  BaselineUsedForDriftCheckStatistics properties.
    """

    check_type: Optional[StrPipeVar] = Unassigned()
    baseline_used_for_drift_check_statistics: Optional[StrPipeVar] = Unassigned()
    baseline_used_for_drift_check_constraints: Optional[StrPipeVar] = Unassigned()
    calculated_baseline_statistics: Optional[StrPipeVar] = Unassigned()
    calculated_baseline_constraints: Optional[StrPipeVar] = Unassigned()
    model_package_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    violation_report: Optional[StrPipeVar] = Unassigned()
    check_job_arn: Optional[StrPipeVar] = Unassigned()
    skip_check: Optional[bool] = Unassigned()
    register_new_baseline: Optional[bool] = Unassigned()


class PipelineExecutionStepMetadata(Base):
    """
    PipelineExecutionStepMetadata
      Metadata for a step execution.

    Attributes
    ----------------------
    training_job: The Amazon Resource Name (ARN) of the training job that was run by this step execution.
    processing_job: The Amazon Resource Name (ARN) of the processing job that was run by this step execution.
    transform_job: The Amazon Resource Name (ARN) of the transform job that was run by this step execution.
    tuning_job: The Amazon Resource Name (ARN) of the tuning job that was run by this step execution.
    compilation_job
    model: The Amazon Resource Name (ARN) of the model that was created by this step execution.
    register_model: The Amazon Resource Name (ARN) of the model package that the model was registered to by this step execution.
    condition: The outcome of the condition evaluation that was run by this step execution.
    callback: The URL of the Amazon SQS queue used by this step execution, the pipeline generated token, and a list of output parameters.
    lambda: The Amazon Resource Name (ARN) of the Lambda function that was run by this step execution and a list of output parameters.
    emr: The configurations and outcomes of an Amazon EMR step execution.
    quality_check: The configurations and outcomes of the check step execution. This includes:    The type of the check conducted.   The Amazon S3 URIs of baseline constraints and statistics files to be used for the drift check.   The Amazon S3 URIs of newly calculated baseline constraints and statistics.   The model package group name provided.   The Amazon S3 URI of the violation report if violations detected.   The Amazon Resource Name (ARN) of check processing job initiated by the step execution.   The Boolean flags indicating if the drift check is skipped.   If step property BaselineUsedForDriftCheck is set the same as CalculatedBaseline.
    clarify_check: Container for the metadata for a Clarify check step. The configurations and outcomes of the check step execution. This includes:    The type of the check conducted,   The Amazon S3 URIs of baseline constraints and statistics files to be used for the drift check.   The Amazon S3 URIs of newly calculated baseline constraints and statistics.   The model package group name provided.   The Amazon S3 URI of the violation report if violations detected.   The Amazon Resource Name (ARN) of check processing job initiated by the step execution.   The boolean flags indicating if the drift check is skipped.   If step property BaselineUsedForDriftCheck is set the same as CalculatedBaseline.
    fail: The configurations and outcomes of a Fail step execution.
    auto_ml_job: The Amazon Resource Name (ARN) of the AutoML job that was run by this step.
    endpoint: The endpoint that was invoked during this step execution.
    endpoint_config: The endpoint configuration used to create an endpoint during this step execution.
    bedrock_custom_model
    bedrock_custom_model_deployment
    bedrock_provisioned_model_throughput
    bedrock_model_import
    inference_component
    lineage
    """

    training_job: Optional[TrainingJobStepMetadata] = Unassigned()
    processing_job: Optional[ProcessingJobStepMetadata] = Unassigned()
    transform_job: Optional[TransformJobStepMetadata] = Unassigned()
    tuning_job: Optional[TuningJobStepMetaData] = Unassigned()
    compilation_job: Optional[CompilationJobStepMetadata] = Unassigned()
    model: Optional[ModelStepMetadata] = Unassigned()
    register_model: Optional[RegisterModelStepMetadata] = Unassigned()
    condition: Optional[ConditionStepMetadata] = Unassigned()
    callback: Optional[CallbackStepMetadata] = Unassigned()
    # lambda: Optional[LambdaStepMetadata] = Unassigned()
    emr: Optional[EMRStepMetadata] = Unassigned()
    quality_check: Optional[QualityCheckStepMetadata] = Unassigned()
    clarify_check: Optional[ClarifyCheckStepMetadata] = Unassigned()
    fail: Optional[FailStepMetadata] = Unassigned()
    auto_ml_job: Optional[AutoMLJobStepMetadata] = Unassigned()
    endpoint: Optional[EndpointStepMetadata] = Unassigned()
    endpoint_config: Optional[EndpointConfigStepMetadata] = Unassigned()
    bedrock_custom_model: Optional[BedrockCustomModelMetadata] = Unassigned()
    bedrock_custom_model_deployment: Optional[BedrockCustomModelDeploymentMetadata] = Unassigned()
    bedrock_provisioned_model_throughput: Optional[BedrockProvisionedModelThroughputMetadata] = (
        Unassigned()
    )
    bedrock_model_import: Optional[BedrockModelImportMetadata] = Unassigned()
    inference_component: Optional[InferenceComponentMetadata] = Unassigned()
    lineage: Optional[LineageMetadata] = Unassigned()


class SelectiveExecutionResult(Base):
    """
    SelectiveExecutionResult
      The ARN from an execution of the current pipeline.

    Attributes
    ----------------------
    source_pipeline_execution_arn: The ARN from an execution of the current pipeline.
    """

    source_pipeline_execution_arn: Optional[StrPipeVar] = Unassigned()


class PipelineExecutionStep(Base):
    """
    PipelineExecutionStep
      An execution of a step in a pipeline.

    Attributes
    ----------------------
    step_name: The name of the step that is executed.
    step_display_name: The display name of the step.
    step_description: The description of the step.
    start_time: The time that the step started executing.
    end_time: The time that the step stopped executing.
    step_status: The status of the step execution.
    cache_hit_result: If this pipeline execution step was cached, details on the cache hit.
    failure_reason: The reason why the step failed execution. This is only returned if the step failed its execution.
    metadata: Metadata to run the pipeline step.
    attempt_count: The current attempt of the execution step. For more information, see Retry Policy for SageMaker Pipelines steps.
    selective_execution_result: The ARN from an execution of the current pipeline from which results are reused for this step.
    """

    step_name: Optional[StrPipeVar] = Unassigned()
    step_display_name: Optional[StrPipeVar] = Unassigned()
    step_description: Optional[StrPipeVar] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()
    step_status: Optional[StrPipeVar] = Unassigned()
    cache_hit_result: Optional[CacheHitResult] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    metadata: Optional[PipelineExecutionStepMetadata] = Unassigned()
    attempt_count: Optional[int] = Unassigned()
    selective_execution_result: Optional[SelectiveExecutionResult] = Unassigned()


class PipelineExecutionSummary(Base):
    """
    PipelineExecutionSummary
      A pipeline execution summary.

    Attributes
    ----------------------
    pipeline_execution_arn: The Amazon Resource Name (ARN) of the pipeline execution.
    start_time: The start time of the pipeline execution.
    pipeline_execution_status: The status of the pipeline execution.
    pipeline_execution_description: The description of the pipeline execution.
    pipeline_execution_display_name: The display name of the pipeline execution.
    pipeline_execution_failure_reason: A message generated by SageMaker Pipelines describing why the pipeline execution failed.
    """

    pipeline_execution_arn: Optional[StrPipeVar] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    pipeline_execution_status: Optional[StrPipeVar] = Unassigned()
    pipeline_execution_description: Optional[StrPipeVar] = Unassigned()
    pipeline_execution_display_name: Optional[StrPipeVar] = Unassigned()
    pipeline_execution_failure_reason: Optional[StrPipeVar] = Unassigned()


class Parameter(Base):
    """
    Parameter
      Assigns a value to a named Pipeline parameter.

    Attributes
    ----------------------
    name: The name of the parameter to assign a value to. This parameter name must match a named parameter in the pipeline definition.
    value: The literal value for the parameter.
    """

    name: StrPipeVar
    value: StrPipeVar


class PipelineVersionSummary(Base):
    """
    PipelineVersionSummary
      The summary of the pipeline version.

    Attributes
    ----------------------
    pipeline_arn: The Amazon Resource Name (ARN) of the pipeline.
    pipeline_version_id: The ID of the pipeline version.
    creation_time: The creation time of the pipeline version.
    pipeline_version_description: The description of the pipeline version.
    pipeline_version_display_name: The display name of the pipeline version.
    last_execution_pipeline_execution_arn: The Amazon Resource Name (ARN) of the most recent pipeline execution created from this pipeline version.
    """

    pipeline_arn: Optional[StrPipeVar] = Unassigned()
    pipeline_version_id: Optional[int] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    pipeline_version_description: Optional[StrPipeVar] = Unassigned()
    pipeline_version_display_name: Optional[StrPipeVar] = Unassigned()
    last_execution_pipeline_execution_arn: Optional[StrPipeVar] = Unassigned()


class PipelineSummary(Base):
    """
    PipelineSummary
      A summary of a pipeline.

    Attributes
    ----------------------
    pipeline_arn:  The Amazon Resource Name (ARN) of the pipeline.
    pipeline_name: The name of the pipeline.
    pipeline_display_name: The display name of the pipeline.
    pipeline_description: The description of the pipeline.
    role_arn: The Amazon Resource Name (ARN) that the pipeline used to execute.
    creation_time: The creation time of the pipeline.
    last_modified_time: The time that the pipeline was last modified.
    last_execution_time: The last time that a pipeline execution began.
    """

    pipeline_arn: Optional[StrPipeVar] = Unassigned()
    pipeline_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    pipeline_display_name: Optional[StrPipeVar] = Unassigned()
    pipeline_description: Optional[StrPipeVar] = Unassigned()
    role_arn: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_execution_time: Optional[datetime.datetime] = Unassigned()


class ProcessingJobSummary(Base):
    """
    ProcessingJobSummary
      Summary of information about a processing job.

    Attributes
    ----------------------
    processing_job_name: The name of the processing job.
    processing_job_arn: The Amazon Resource Name (ARN) of the processing job..
    creation_time: The time at which the processing job was created.
    processing_end_time: The time at which the processing job completed.
    last_modified_time: A timestamp that indicates the last time the processing job was modified.
    processing_job_status: The status of the processing job.
    failure_reason: A string, up to one KB in size, that contains the reason a processing job failed, if it failed.
    exit_message: An optional string, up to one KB in size, that contains metadata from the processing container when the processing job exits.
    """

    processing_job_name: Union[StrPipeVar, object]
    processing_job_arn: StrPipeVar
    creation_time: datetime.datetime
    processing_job_status: StrPipeVar
    processing_end_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    exit_message: Optional[StrPipeVar] = Unassigned()


class ProjectSummary(Base):
    """
    ProjectSummary
      Information about a project.

    Attributes
    ----------------------
    project_name: The name of the project.
    project_description: The description of the project.
    project_arn: The Amazon Resource Name (ARN) of the project.
    project_id: The ID of the project.
    creation_time: The time that the project was created.
    project_status: The status of the project.
    """

    project_name: Union[StrPipeVar, object]
    project_arn: StrPipeVar
    project_id: StrPipeVar
    creation_time: datetime.datetime
    project_status: StrPipeVar
    project_description: Optional[StrPipeVar] = Unassigned()


class QuotaAllocationSummary(Base):
    """
    QuotaAllocationSummary

    Attributes
    ----------------------
    quota_allocation_arn
    quota_id
    quota_allocation_name
    cluster_arn
    quota_resources
    creation_time
    last_modified_time
    quota_allocation_status
    quota_allocation_target
    activation_state
    preemption_config
    over_quota
    """

    quota_allocation_arn: Optional[StrPipeVar] = Unassigned()
    quota_id: Optional[StrPipeVar] = Unassigned()
    quota_allocation_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    cluster_arn: Optional[StrPipeVar] = Unassigned()
    quota_resources: Optional[List[QuotaResourceConfig]] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    quota_allocation_status: Optional[StrPipeVar] = Unassigned()
    quota_allocation_target: Optional[QuotaAllocationTarget] = Unassigned()
    activation_state: Optional[ActivationStateV1] = Unassigned()
    preemption_config: Optional[PreemptionConfig] = Unassigned()
    over_quota: Optional[OverQuota] = Unassigned()


class ResourceCatalog(Base):
    """
    ResourceCatalog
       A resource catalog containing all of the resources of a specific resource type within a resource owner account. For an example on sharing the Amazon SageMaker Feature Store DefaultFeatureGroupCatalog, see Share Amazon SageMaker Catalog resource type in the Amazon SageMaker Developer Guide.

    Attributes
    ----------------------
    resource_catalog_arn:  The Amazon Resource Name (ARN) of the ResourceCatalog.
    resource_catalog_name:  The name of the ResourceCatalog.
    description:  A free form description of the ResourceCatalog.
    creation_time:  The time the ResourceCatalog was created.
    """

    resource_catalog_arn: StrPipeVar
    resource_catalog_name: Union[StrPipeVar, object]
    description: StrPipeVar
    creation_time: datetime.datetime


class SharedModelVersionListEntity(Base):
    """
    SharedModelVersionListEntity

    Attributes
    ----------------------
    shared_model_version
    creator
    model_type
    problem_type
    description
    model_identifier
    creation_time
    last_modified_time
    """

    shared_model_version: Optional[StrPipeVar] = Unassigned()
    creator: Optional[StrPipeVar] = Unassigned()
    model_type: Optional[StrPipeVar] = Unassigned()
    problem_type: Optional[StrPipeVar] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    model_identifier: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class SharedModelListEntity(Base):
    """
    SharedModelListEntity

    Attributes
    ----------------------
    shared_model_id
    shared_model_version
    owner
    model_name
    model_type
    problem_type
    description
    shares
    model_identifier
    creation_time
    last_modified_time
    """

    shared_model_id: Optional[StrPipeVar] = Unassigned()
    shared_model_version: Optional[StrPipeVar] = Unassigned()
    owner: Optional[StrPipeVar] = Unassigned()
    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_type: Optional[StrPipeVar] = Unassigned()
    problem_type: Optional[StrPipeVar] = Unassigned()
    description: Optional[StrPipeVar] = Unassigned()
    shares: Optional[int] = Unassigned()
    model_identifier: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class SpaceSettingsSummary(Base):
    """
    SpaceSettingsSummary
      Specifies summary information about the space settings.

    Attributes
    ----------------------
    app_type: The type of app created within the space.
    remote_access: A setting that enables or disables remote access for a SageMaker space. When enabled, this allows you to connect to the remote space from your local IDE.
    space_storage_settings: The storage settings for a space.
    """

    app_type: Optional[StrPipeVar] = Unassigned()
    remote_access: Optional[StrPipeVar] = Unassigned()
    space_storage_settings: Optional[SpaceStorageSettings] = Unassigned()


class SpaceSharingSettingsSummary(Base):
    """
    SpaceSharingSettingsSummary
      Specifies summary information about the space sharing settings.

    Attributes
    ----------------------
    sharing_type: Specifies the sharing type of the space.
    """

    sharing_type: Optional[StrPipeVar] = Unassigned()


class OwnershipSettingsSummary(Base):
    """
    OwnershipSettingsSummary
      Specifies summary information about the ownership settings.

    Attributes
    ----------------------
    owner_user_profile_name: The user profile who is the owner of the space.
    """

    owner_user_profile_name: Optional[StrPipeVar] = Unassigned()


class SpaceDetails(Base):
    """
    SpaceDetails
      The space's details.

    Attributes
    ----------------------
    domain_id: The ID of the associated domain.
    space_name: The name of the space.
    status: The status.
    creation_time: The creation time.
    last_modified_time: The last modified time.
    space_settings_summary: Specifies summary information about the space settings.
    space_sharing_settings_summary: Specifies summary information about the space sharing settings.
    ownership_settings_summary: Specifies summary information about the ownership settings.
    space_display_name: The name of the space that appears in the Studio UI.
    """

    domain_id: Optional[StrPipeVar] = Unassigned()
    space_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    space_settings_summary: Optional[SpaceSettingsSummary] = Unassigned()
    space_sharing_settings_summary: Optional[SpaceSharingSettingsSummary] = Unassigned()
    ownership_settings_summary: Optional[OwnershipSettingsSummary] = Unassigned()
    space_display_name: Optional[StrPipeVar] = Unassigned()


class StudioLifecycleConfigDetails(Base):
    """
    StudioLifecycleConfigDetails
      Details of the Amazon SageMaker AI Studio Lifecycle Configuration.

    Attributes
    ----------------------
    studio_lifecycle_config_arn:  The Amazon Resource Name (ARN) of the Lifecycle Configuration.
    studio_lifecycle_config_name: The name of the Amazon SageMaker AI Studio Lifecycle Configuration.
    creation_time: The creation time of the Amazon SageMaker AI Studio Lifecycle Configuration.
    last_modified_time: This value is equivalent to CreationTime because Amazon SageMaker AI Studio Lifecycle Configurations are immutable.
    studio_lifecycle_config_app_type: The App type to which the Lifecycle Configuration is attached.
    """

    studio_lifecycle_config_arn: Optional[StrPipeVar] = Unassigned()
    studio_lifecycle_config_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    studio_lifecycle_config_app_type: Optional[StrPipeVar] = Unassigned()


class TrainingJobSummary(Base):
    """
    TrainingJobSummary
      Provides summary information about a training job.

    Attributes
    ----------------------
    training_job_name: The name of the training job that you want a summary for.
    training_job_arn: The Amazon Resource Name (ARN) of the training job.
    creation_time: A timestamp that shows when the training job was created.
    training_end_time: A timestamp that shows when the training job ended. This field is set only if the training job has one of the terminal statuses (Completed, Failed, or Stopped).
    last_modified_time:  Timestamp when the training job was last modified.
    training_job_status: The status of the training job.
    secondary_status: The secondary status of the training job.
    warm_pool_status: The status of the warm pool associated with the training job.
    keep_alive_period_in_seconds
    training_plan_arn: The Amazon Resource Name (ARN); of the training plan associated with this training job. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .
    """

    training_job_name: Union[StrPipeVar, object]
    training_job_arn: StrPipeVar
    creation_time: datetime.datetime
    training_job_status: StrPipeVar
    training_end_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    secondary_status: Optional[StrPipeVar] = Unassigned()
    warm_pool_status: Optional[WarmPoolStatus] = Unassigned()
    keep_alive_period_in_seconds: Optional[int] = Unassigned()
    training_plan_arn: Optional[StrPipeVar] = Unassigned()


class TrainingPlanFilter(Base):
    """
    TrainingPlanFilter
      A filter to apply when listing or searching for training plans. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .

    Attributes
    ----------------------
    name: The name of the filter field (e.g., Status, InstanceType).
    value: The value to filter by for the specified field.
    """

    name: StrPipeVar
    value: StrPipeVar


class TrainingPlanSummary(Base):
    """
    TrainingPlanSummary
      Details of the training plan. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .

    Attributes
    ----------------------
    training_plan_arn: The Amazon Resource Name (ARN); of the training plan.
    training_plan_name: The name of the training plan.
    status: The current status of the training plan (e.g., Pending, Active, Expired). To see the complete list of status values available for a training plan, refer to the Status attribute within the  TrainingPlanSummary  object.
    status_message: A message providing additional information about the current status of the training plan.
    duration_hours: The number of whole hours in the total duration for this training plan.
    duration_minutes: The additional minutes beyond whole hours in the total duration for this training plan.
    start_time: The start time of the training plan.
    end_time: The end time of the training plan.
    upfront_fee: The upfront fee for the training plan.
    currency_code: The currency code for the upfront fee (e.g., USD).
    total_instance_count: The total number of instances reserved in this training plan.
    available_instance_count: The number of instances currently available for use in this training plan.
    in_use_instance_count: The number of instances currently in use from this training plan.
    unhealthy_instance_count
    available_spare_instance_count
    total_ultra_server_count: The total number of UltraServers allocated to this training plan.
    target_resources: The target resources (e.g., training jobs, HyperPod clusters) that can use this training plan. Training plans are specific to their target resource.   A training plan designed for SageMaker training jobs can only be used to schedule and run training jobs.   A training plan for HyperPod clusters can be used exclusively to provide compute resources to a cluster's instance group.
    reserved_capacity_summaries: A list of reserved capacities associated with this training plan, including details such as instance types, counts, and availability zones.
    training_plan_status_transitions
    """

    training_plan_arn: StrPipeVar
    training_plan_name: Union[StrPipeVar, object]
    status: StrPipeVar
    status_message: Optional[StrPipeVar] = Unassigned()
    duration_hours: Optional[int] = Unassigned()
    duration_minutes: Optional[int] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()
    upfront_fee: Optional[StrPipeVar] = Unassigned()
    currency_code: Optional[StrPipeVar] = Unassigned()
    total_instance_count: Optional[int] = Unassigned()
    available_instance_count: Optional[int] = Unassigned()
    in_use_instance_count: Optional[int] = Unassigned()
    unhealthy_instance_count: Optional[int] = Unassigned()
    available_spare_instance_count: Optional[int] = Unassigned()
    total_ultra_server_count: Optional[int] = Unassigned()
    target_resources: Optional[List[StrPipeVar]] = Unassigned()
    reserved_capacity_summaries: Optional[List[ReservedCapacitySummary]] = Unassigned()
    training_plan_status_transitions: Optional[List[TrainingPlanStatusTransition]] = Unassigned()


class TransformJobSummary(Base):
    """
    TransformJobSummary
      Provides a summary of a transform job. Multiple TransformJobSummary objects are returned as a list after in response to a ListTransformJobs call.

    Attributes
    ----------------------
    transform_job_name: The name of the transform job.
    transform_job_arn: The Amazon Resource Name (ARN) of the transform job.
    creation_time: A timestamp that shows when the transform Job was created.
    transform_end_time: Indicates when the transform job ends on compute instances. For successful jobs and stopped jobs, this is the exact time recorded after the results are uploaded. For failed jobs, this is when Amazon SageMaker detected that the job failed.
    last_modified_time: Indicates when the transform job was last modified.
    transform_job_status: The status of the transform job.
    failure_reason: If the transform job failed, the reason it failed.
    """

    transform_job_name: Union[StrPipeVar, object]
    transform_job_arn: StrPipeVar
    creation_time: datetime.datetime
    transform_job_status: StrPipeVar
    transform_end_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()


class TrialComponentSummary(Base):
    """
    TrialComponentSummary
      A summary of the properties of a trial component. To get all the properties, call the DescribeTrialComponent API and provide the TrialComponentName.

    Attributes
    ----------------------
    trial_component_name: The name of the trial component.
    trial_component_arn: The Amazon Resource Name (ARN) of the trial component.
    display_name: The name of the component as displayed. If DisplayName isn't specified, TrialComponentName is displayed.
    trial_component_source
    status: The status of the component. States include:   InProgress   Completed   Failed
    start_time: When the component started.
    end_time: When the component ended.
    creation_time: When the component was created.
    created_by: Who created the trial component.
    last_modified_time: When the component was last modified.
    last_modified_by: Who last modified the component.
    """

    trial_component_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    trial_component_arn: Optional[StrPipeVar] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    trial_component_source: Optional[TrialComponentSource] = Unassigned()
    status: Optional[TrialComponentStatus] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()


class TrialSummary(Base):
    """
    TrialSummary
      A summary of the properties of a trial. To get the complete set of properties, call the DescribeTrial API and provide the TrialName.

    Attributes
    ----------------------
    trial_arn: The Amazon Resource Name (ARN) of the trial.
    trial_name: The name of the trial.
    display_name: The name of the trial as displayed. If DisplayName isn't specified, TrialName is displayed.
    trial_source
    creation_time: When the trial was created.
    last_modified_time: When the trial was last modified.
    """

    trial_arn: Optional[StrPipeVar] = Unassigned()
    trial_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    trial_source: Optional[TrialSource] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class UltraServer(Base):
    """
    UltraServer
      Represents a high-performance compute server used for distributed training in SageMaker AI. An UltraServer consists of multiple instances within a shared NVLink interconnect domain.

    Attributes
    ----------------------
    ultra_server_id: The unique identifier for the UltraServer.
    ultra_server_type: The type of UltraServer, such as ml.u-p6e-gb200x72.
    availability_zone: The name of the Availability Zone where the UltraServer is provisioned.
    instance_type: The Amazon EC2 instance type used in the UltraServer.
    total_instance_count: The total number of instances in this UltraServer.
    configured_spare_instance_count: The number of spare instances configured for this UltraServer to provide enhanced resiliency.
    available_instance_count: The number of instances currently available for use in this UltraServer.
    in_use_instance_count: The number of instances currently in use in this UltraServer.
    available_spare_instance_count: The number of available spare instances in the UltraServer.
    unhealthy_instance_count: The number of instances in this UltraServer that are currently in an unhealthy state.
    health_status: The overall health status of the UltraServer.
    """

    ultra_server_id: StrPipeVar
    ultra_server_type: StrPipeVar
    availability_zone: StrPipeVar
    instance_type: StrPipeVar
    total_instance_count: int
    configured_spare_instance_count: Optional[int] = Unassigned()
    available_instance_count: Optional[int] = Unassigned()
    in_use_instance_count: Optional[int] = Unassigned()
    available_spare_instance_count: Optional[int] = Unassigned()
    unhealthy_instance_count: Optional[int] = Unassigned()
    health_status: Optional[StrPipeVar] = Unassigned()


class UserProfileDetails(Base):
    """
    UserProfileDetails
      The user profile details.

    Attributes
    ----------------------
    domain_id: The domain ID.
    user_profile_name: The user profile name.
    status: The status.
    creation_time: The creation time.
    last_modified_time: The last modified time.
    """

    domain_id: Optional[StrPipeVar] = Unassigned()
    user_profile_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()


class Model(Base):
    """
    Model
      The properties of a model as returned by the Search API.

    Attributes
    ----------------------
    model_name: The name of the model.
    primary_container
    containers: The containers in the inference pipeline.
    inference_execution_config
    execution_role_arn: The Amazon Resource Name (ARN) of the IAM role that you specified for the model.
    vpc_config
    creation_time: A timestamp that indicates when the model was created.
    model_arn: The Amazon Resource Name (ARN) of the model.
    enable_network_isolation: Isolates the model container. No inbound or outbound network calls can be made to or from the model container.
    tags: A list of key-value pairs associated with the model. For more information, see Tagging Amazon Web Services resources in the Amazon Web Services General Reference Guide.
    deployment_recommendation: A set of recommended deployment configurations for the model.
    """

    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    primary_container: Optional[ContainerDefinition] = Unassigned()
    containers: Optional[List[ContainerDefinition]] = Unassigned()
    inference_execution_config: Optional[InferenceExecutionConfig] = Unassigned()
    execution_role_arn: Optional[StrPipeVar] = Unassigned()
    vpc_config: Optional[VpcConfig] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    model_arn: Optional[StrPipeVar] = Unassigned()
    enable_network_isolation: Optional[bool] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    deployment_recommendation: Optional[DeploymentRecommendation] = Unassigned()


class ModelCard(Base):
    """
    ModelCard
      An Amazon SageMaker Model Card.

    Attributes
    ----------------------
    model_card_arn: The Amazon Resource Name (ARN) of the model card.
    model_card_name: The unique name of the model card.
    model_card_version: The version of the model card.
    content: The content of the model card. Content uses the model card JSON schema and provided as a string.
    model_card_status: The approval status of the model card within your organization. Different organizations might have different criteria for model card review and approval.    Draft: The model card is a work in progress.    PendingReview: The model card is pending review.    Approved: The model card is approved.    Archived: The model card is archived. No more updates should be made to the model card, but it can still be exported.
    security_config: The security configuration used to protect model card data.
    creation_time: The date and time that the model card was created.
    created_by
    last_modified_time: The date and time that the model card was last modified.
    last_modified_by
    tags: Key-value pairs used to manage metadata for the model card.
    model_id: The unique name (ID) of the model.
    risk_rating: The risk rating of the model. Different organizations might have different criteria for model card risk ratings. For more information, see Risk ratings.
    model_package_group_name: The model package group that contains the model package. Only relevant for model cards created for model packages in the Amazon SageMaker Model Registry.
    """

    model_card_arn: Optional[StrPipeVar] = Unassigned()
    model_card_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_card_version: Optional[int] = Unassigned()
    content: Optional[StrPipeVar] = Unassigned()
    model_card_status: Optional[StrPipeVar] = Unassigned()
    security_config: Optional[ModelCardSecurityConfig] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    model_id: Optional[StrPipeVar] = Unassigned()
    risk_rating: Optional[StrPipeVar] = Unassigned()
    model_package_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()


class ModelDashboardEndpoint(Base):
    """
    ModelDashboardEndpoint
      An endpoint that hosts a model displayed in the Amazon SageMaker Model Dashboard.

    Attributes
    ----------------------
    endpoint_name: The endpoint name.
    endpoint_arn: The Amazon Resource Name (ARN) of the endpoint.
    creation_time: A timestamp that indicates when the endpoint was created.
    last_modified_time: The last time the endpoint was modified.
    endpoint_status: The endpoint status.
    """

    endpoint_name: Union[StrPipeVar, object]
    endpoint_arn: StrPipeVar
    creation_time: datetime.datetime
    last_modified_time: datetime.datetime
    endpoint_status: StrPipeVar


class TransformJob(Base):
    """
    TransformJob
      A batch transform job. For information about SageMaker batch transform, see Use Batch Transform.

    Attributes
    ----------------------
    transform_job_name: The name of the transform job.
    transform_job_arn: The Amazon Resource Name (ARN) of the transform job.
    transform_job_status: The status of the transform job. Transform job statuses are:    InProgress - The job is in progress.    Completed - The job has completed.    Failed - The transform job has failed. To see the reason for the failure, see the FailureReason field in the response to a DescribeTransformJob call.    Stopping - The transform job is stopping.    Stopped - The transform job has stopped.
    failure_reason: If the transform job failed, the reason it failed.
    model_name: The name of the model associated with the transform job.
    max_concurrent_transforms: The maximum number of parallel requests that can be sent to each instance in a transform job. If MaxConcurrentTransforms is set to 0 or left unset, SageMaker checks the optional execution-parameters to determine the settings for your chosen algorithm. If the execution-parameters endpoint is not enabled, the default value is 1. For built-in algorithms, you don't need to set a value for MaxConcurrentTransforms.
    model_client_config
    max_payload_in_mb: The maximum allowed size of the payload, in MB. A payload is the data portion of a record (without metadata). The value in MaxPayloadInMB must be greater than, or equal to, the size of a single record. To estimate the size of a record in MB, divide the size of your dataset by the number of records. To ensure that the records fit within the maximum payload size, we recommend using a slightly larger value. The default value is 6 MB. For cases where the payload might be arbitrarily large and is transmitted using HTTP chunked encoding, set the value to 0. This feature works only in supported algorithms. Currently, SageMaker built-in algorithms do not support HTTP chunked encoding.
    batch_strategy: Specifies the number of records to include in a mini-batch for an HTTP inference request. A record is a single unit of input data that inference can be made on. For example, a single line in a CSV file is a record.
    environment: The environment variables to set in the Docker container. We support up to 16 key and values entries in the map.
    transform_input
    transform_output
    data_capture_config
    transform_resources
    creation_time: A timestamp that shows when the transform Job was created.
    transform_start_time: Indicates when the transform job starts on ML instances. You are billed for the time interval between this time and the value of TransformEndTime.
    transform_end_time: Indicates when the transform job has been completed, or has stopped or failed. You are billed for the time interval between this time and the value of TransformStartTime.
    labeling_job_arn: The Amazon Resource Name (ARN) of the labeling job that created the transform job.
    auto_ml_job_arn: The Amazon Resource Name (ARN) of the AutoML job that created the transform job.
    transform_job_progress
    data_processing
    experiment_config
    last_modified_by
    created_by
    tags: A list of tags associated with the transform job.
    """

    transform_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    transform_job_arn: Optional[StrPipeVar] = Unassigned()
    transform_job_status: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    model_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    max_concurrent_transforms: Optional[int] = Unassigned()
    model_client_config: Optional[ModelClientConfig] = Unassigned()
    max_payload_in_mb: Optional[int] = Unassigned()
    batch_strategy: Optional[StrPipeVar] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    transform_input: Optional[TransformInput] = Unassigned()
    transform_output: Optional[TransformOutput] = Unassigned()
    data_capture_config: Optional[BatchDataCaptureConfig] = Unassigned()
    transform_resources: Optional[TransformResources] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    transform_start_time: Optional[datetime.datetime] = Unassigned()
    transform_end_time: Optional[datetime.datetime] = Unassigned()
    labeling_job_arn: Optional[StrPipeVar] = Unassigned()
    auto_ml_job_arn: Optional[StrPipeVar] = Unassigned()
    transform_job_progress: Optional[TransformJobProgress] = Unassigned()
    data_processing: Optional[DataProcessing] = Unassigned()
    experiment_config: Optional[ExperimentConfig] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class ModelDashboardMonitoringSchedule(Base):
    """
    ModelDashboardMonitoringSchedule
      A monitoring schedule for a model displayed in the Amazon SageMaker Model Dashboard.

    Attributes
    ----------------------
    monitoring_schedule_arn: The Amazon Resource Name (ARN) of a monitoring schedule.
    monitoring_schedule_name: The name of a monitoring schedule.
    monitoring_schedule_status: The status of the monitoring schedule.
    monitoring_type: The monitor type of a model monitor.
    failure_reason: If a monitoring job failed, provides the reason.
    creation_time: A timestamp that indicates when the monitoring schedule was created.
    last_modified_time: A timestamp that indicates when the monitoring schedule was last updated.
    monitoring_schedule_config
    endpoint_name: The endpoint which is monitored.
    monitoring_alert_summaries: A JSON array where each element is a summary for a monitoring alert.
    last_monitoring_execution_summary
    custom_monitoring_job_definition
    data_quality_job_definition
    model_quality_job_definition
    model_bias_job_definition
    model_explainability_job_definition
    batch_transform_input
    """

    monitoring_schedule_arn: Optional[StrPipeVar] = Unassigned()
    monitoring_schedule_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    monitoring_schedule_status: Optional[StrPipeVar] = Unassigned()
    monitoring_type: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    monitoring_schedule_config: Optional[MonitoringScheduleConfig] = Unassigned()
    endpoint_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    monitoring_alert_summaries: Optional[List[MonitoringAlertSummary]] = Unassigned()
    last_monitoring_execution_summary: Optional[MonitoringExecutionSummary] = Unassigned()
    custom_monitoring_job_definition: Optional[CustomMonitoringJobDefinition] = Unassigned()
    data_quality_job_definition: Optional[DataQualityJobDefinition] = Unassigned()
    model_quality_job_definition: Optional[ModelQualityJobDefinition] = Unassigned()
    model_bias_job_definition: Optional[ModelBiasJobDefinition] = Unassigned()
    model_explainability_job_definition: Optional[ModelExplainabilityJobDefinition] = Unassigned()
    batch_transform_input: Optional[BatchTransformInput] = Unassigned()


class ModelDashboardModelCard(Base):
    """
    ModelDashboardModelCard
      The model card for a model displayed in the Amazon SageMaker Model Dashboard.

    Attributes
    ----------------------
    model_card_arn: The Amazon Resource Name (ARN) for a model card.
    model_card_name: The name of a model card.
    model_card_version: The model card version.
    model_card_status: The model card status.
    security_config: The KMS Key ID (KMSKeyId) for encryption of model card information.
    creation_time: A timestamp that indicates when the model card was created.
    created_by
    last_modified_time: A timestamp that indicates when the model card was last updated.
    last_modified_by
    tags: The tags associated with a model card.
    model_id: For models created in SageMaker, this is the model ARN. For models created outside of SageMaker, this is a user-customized string.
    risk_rating: A model card's risk rating. Can be low, medium, or high.
    """

    model_card_arn: Optional[StrPipeVar] = Unassigned()
    model_card_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_card_version: Optional[int] = Unassigned()
    model_card_status: Optional[StrPipeVar] = Unassigned()
    security_config: Optional[ModelCardSecurityConfig] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    model_id: Optional[StrPipeVar] = Unassigned()
    risk_rating: Optional[StrPipeVar] = Unassigned()


class ModelDashboardModel(Base):
    """
    ModelDashboardModel
      A model displayed in the Amazon SageMaker Model Dashboard.

    Attributes
    ----------------------
    model: A model displayed in the Model Dashboard.
    endpoints: The endpoints that host a model.
    last_batch_transform_job
    monitoring_schedules: The monitoring schedules for a model.
    model_card: The model card for a model.
    """

    model: Optional[Model] = Unassigned()
    endpoints: Optional[List[ModelDashboardEndpoint]] = Unassigned()
    last_batch_transform_job: Optional[TransformJob] = Unassigned()
    monitoring_schedules: Optional[List[ModelDashboardMonitoringSchedule]] = Unassigned()
    model_card: Optional[ModelDashboardModelCard] = Unassigned()


class ModelPackage(Base):
    """
    ModelPackage
      A container for your trained model that can be deployed for SageMaker inference. This can include inference code, artifacts, and metadata. The model package type can be one of the following.   Versioned model: A part of a model package group in Model Registry.   Unversioned model: Not part of a model package group and used in Amazon Web Services Marketplace.   For more information, see  CreateModelPackage .

    Attributes
    ----------------------
    model_package_name: The name of the model package. The name can be as follows:   For a versioned model, the name is automatically generated by SageMaker Model Registry and follows the format 'ModelPackageGroupName/ModelPackageVersion'.   For an unversioned model, you must provide the name.
    model_package_group_name: The model group to which the model belongs.
    model_package_version: The version number of a versioned model.
    model_package_registration_type
    model_package_arn: The Amazon Resource Name (ARN) of the model package.
    model_package_description: The description of the model package.
    creation_time: The time that the model package was created.
    inference_specification: Defines how to perform inference generation after a training job is run.
    source_algorithm_specification: A list of algorithms that were used to create a model package.
    validation_specification: Specifies batch transform jobs that SageMaker runs to validate your model package.
    model_package_status: The status of the model package. This can be one of the following values.    PENDING - The model package is pending being created.    IN_PROGRESS - The model package is in the process of being created.    COMPLETED - The model package was successfully created.    FAILED - The model package failed.    DELETING - The model package is in the process of being deleted.
    model_package_status_details: Specifies the validation and image scan statuses of the model package.
    certify_for_marketplace: Whether the model package is to be certified to be listed on Amazon Web Services Marketplace. For information about listing model packages on Amazon Web Services Marketplace, see List Your Algorithm or Model Package on Amazon Web Services Marketplace.
    model_approval_status: The approval status of the model. This can be one of the following values.    APPROVED - The model is approved    REJECTED - The model is rejected.    PENDING_MANUAL_APPROVAL - The model is waiting for manual approval.
    created_by: Information about the user who created or modified an experiment, trial, trial component, lineage group, or project.
    metadata_properties: Metadata properties of the tracking entity, trial, or trial component.
    model_metrics: Metrics for the model.
    deployment_specification
    last_modified_time: The last time the model package was modified.
    last_modified_by: Information about the user who created or modified an experiment, trial, trial component, lineage group, or project.
    approval_description: A description provided when the model approval is set.
    domain: The machine learning domain of your model package and its components. Common machine learning domains include computer vision and natural language processing.
    task: The machine learning task your model package accomplishes. Common machine learning tasks include object detection and image classification.
    sample_payload_url: The Amazon Simple Storage Service path where the sample payload are stored. This path must point to a single gzip compressed tar archive (.tar.gz suffix).
    additional_inference_specifications: An array of additional Inference Specification objects.
    source_uri: The URI of the source for the model package.
    security_config
    model_card
    model_life_cycle:  A structure describing the current state of the model in its life cycle.
    tags: A list of the tags associated with the model package. For more information, see Tagging Amazon Web Services resources in the Amazon Web Services General Reference Guide.
    customer_metadata_properties: The metadata properties for the model package.
    drift_check_baselines: Represents the drift check baselines that can be used when the model monitor is set using the model package.
    skip_model_validation: Indicates if you want to skip model validation.
    """

    model_package_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_package_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_package_version: Optional[int] = Unassigned()
    model_package_registration_type: Optional[StrPipeVar] = Unassigned()
    model_package_arn: Optional[StrPipeVar] = Unassigned()
    model_package_description: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    inference_specification: Optional[InferenceSpecification] = Unassigned()
    source_algorithm_specification: Optional[SourceAlgorithmSpecification] = Unassigned()
    validation_specification: Optional[ModelPackageValidationSpecification] = Unassigned()
    model_package_status: Optional[StrPipeVar] = Unassigned()
    model_package_status_details: Optional[ModelPackageStatusDetails] = Unassigned()
    certify_for_marketplace: Optional[bool] = Unassigned()
    model_approval_status: Optional[StrPipeVar] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    metadata_properties: Optional[MetadataProperties] = Unassigned()
    model_metrics: Optional[ModelMetrics] = Unassigned()
    deployment_specification: Optional[DeploymentSpecification] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    approval_description: Optional[StrPipeVar] = Unassigned()
    domain: Optional[StrPipeVar] = Unassigned()
    task: Optional[StrPipeVar] = Unassigned()
    sample_payload_url: Optional[StrPipeVar] = Unassigned()
    additional_inference_specifications: Optional[
        List[AdditionalInferenceSpecificationDefinition]
    ] = Unassigned()
    source_uri: Optional[StrPipeVar] = Unassigned()
    security_config: Optional[ModelPackageSecurityConfig] = Unassigned()
    model_card: Optional[ModelPackageModelCard] = Unassigned()
    model_life_cycle: Optional[ModelLifeCycle] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    customer_metadata_properties: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    drift_check_baselines: Optional[DriftCheckBaselines] = Unassigned()
    skip_model_validation: Optional[StrPipeVar] = Unassigned()


class ModelPackageGroup(Base):
    """
    ModelPackageGroup
      A group of versioned models in the Model Registry.

    Attributes
    ----------------------
    model_package_group_name: The name of the model group.
    model_package_group_arn: The Amazon Resource Name (ARN) of the model group.
    model_package_group_description: The description for the model group.
    creation_time: The time that the model group was created.
    created_by
    model_package_group_status: The status of the model group. This can be one of the following values.    PENDING - The model group is pending being created.    IN_PROGRESS - The model group is in the process of being created.    COMPLETED - The model group was successfully created.    FAILED - The model group failed.    DELETING - The model group is in the process of being deleted.    DELETE_FAILED - SageMaker failed to delete the model group.
    tags: A list of the tags associated with the model group. For more information, see Tagging Amazon Web Services resources in the Amazon Web Services General Reference Guide.
    """

    model_package_group_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    model_package_group_arn: Optional[StrPipeVar] = Unassigned()
    model_package_group_description: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    model_package_group_status: Optional[StrPipeVar] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class NestedFilters(Base):
    """
    NestedFilters
      A list of nested Filter objects. A resource must satisfy the conditions of all filters to be included in the results returned from the Search API. For example, to filter on a training job's InputDataConfig property with a specific channel name and S3Uri prefix, define the following filters:    '{Name:"InputDataConfig.ChannelName", "Operator":"Equals", "Value":"train"}',     '{Name:"InputDataConfig.DataSource.S3DataSource.S3Uri", "Operator":"Contains", "Value":"mybucket/catdata"}'

    Attributes
    ----------------------
    nested_property_name: The name of the property to use in the nested filters. The value must match a listed property name, such as InputDataConfig.
    filters: A list of filters. Each filter acts on a property. Filters must contain at least one Filters value. For example, a NestedFilters call might include a filter on the PropertyName parameter of the InputDataConfig property: InputDataConfig.DataSource.S3DataSource.S3Uri.
    """

    nested_property_name: StrPipeVar
    filters: List[Filter]


class OnlineStoreConfigUpdate(Base):
    """
    OnlineStoreConfigUpdate
      Updates the feature group online store configuration.

    Attributes
    ----------------------
    ttl_duration: Time to live duration, where the record is hard deleted after the expiration time is reached; ExpiresAt = EventTime + TtlDuration. For information on HardDelete, see the DeleteRecord API in the Amazon SageMaker API Reference guide.
    """

    ttl_duration: Optional[TtlDuration] = Unassigned()


class Parent(Base):
    """
    Parent
      The trial that a trial component is associated with and the experiment the trial is part of. A component might not be associated with a trial. A component can be associated with multiple trials.

    Attributes
    ----------------------
    trial_name: The name of the trial.
    experiment_name: The name of the experiment.
    """

    trial_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    experiment_name: Optional[Union[StrPipeVar, object]] = Unassigned()


class Pipeline(Base):
    """
    Pipeline
      A SageMaker Model Building Pipeline instance.

    Attributes
    ----------------------
    pipeline_arn: The Amazon Resource Name (ARN) of the pipeline.
    pipeline_name: The name of the pipeline.
    pipeline_display_name: The display name of the pipeline.
    pipeline_description: The description of the pipeline.
    role_arn: The Amazon Resource Name (ARN) of the role that created the pipeline.
    pipeline_status: The status of the pipeline.
    creation_time: The creation time of the pipeline.
    last_modified_time: The time that the pipeline was last modified.
    last_run_time: The time when the pipeline was last run.
    created_by
    last_modified_by
    parallelism_configuration: The parallelism configuration applied to the pipeline.
    tags: A list of tags that apply to the pipeline.
    """

    pipeline_arn: Optional[StrPipeVar] = Unassigned()
    pipeline_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    pipeline_display_name: Optional[StrPipeVar] = Unassigned()
    pipeline_description: Optional[StrPipeVar] = Unassigned()
    role_arn: Optional[StrPipeVar] = Unassigned()
    pipeline_status: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_run_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    parallelism_configuration: Optional[ParallelismConfiguration] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class PipelineExecution(Base):
    """
    PipelineExecution
      An execution of a pipeline.

    Attributes
    ----------------------
    pipeline_arn: The Amazon Resource Name (ARN) of the pipeline that was executed.
    pipeline_execution_arn: The Amazon Resource Name (ARN) of the pipeline execution.
    pipeline_execution_display_name: The display name of the pipeline execution.
    pipeline_execution_status: The status of the pipeline status.
    pipeline_execution_description: The description of the pipeline execution.
    pipeline_experiment_config
    failure_reason: If the execution failed, a message describing why.
    creation_time: The creation time of the pipeline execution.
    last_modified_time: The time that the pipeline execution was last modified.
    created_by
    last_modified_by
    parallelism_configuration: The parallelism configuration applied to the pipeline execution.
    selective_execution_config: The selective execution configuration applied to the pipeline run.
    pipeline_parameters: Contains a list of pipeline parameters. This list can be empty.
    pipeline_version_id: The ID of the pipeline version that started this execution.
    pipeline_version_display_name: The display name of the pipeline version that started this execution.
    tags
    """

    pipeline_arn: Optional[StrPipeVar] = Unassigned()
    pipeline_execution_arn: Optional[StrPipeVar] = Unassigned()
    pipeline_execution_display_name: Optional[StrPipeVar] = Unassigned()
    pipeline_execution_status: Optional[StrPipeVar] = Unassigned()
    pipeline_execution_description: Optional[StrPipeVar] = Unassigned()
    pipeline_experiment_config: Optional[PipelineExperimentConfig] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    parallelism_configuration: Optional[ParallelismConfiguration] = Unassigned()
    selective_execution_config: Optional[SelectiveExecutionConfig] = Unassigned()
    pipeline_parameters: Optional[List[Parameter]] = Unassigned()
    pipeline_version_id: Optional[int] = Unassigned()
    pipeline_version_display_name: Optional[StrPipeVar] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class PipelineVersion(Base):
    """
    PipelineVersion
      The version of the pipeline.

    Attributes
    ----------------------
    pipeline_arn: The Amazon Resource Name (ARN) of the pipeline.
    pipeline_version_id: The ID of the pipeline version.
    pipeline_version_arn
    pipeline_version_display_name: The display name of the pipeline version.
    pipeline_version_description: The description of the pipeline version.
    creation_time: The creation time of the pipeline version.
    last_modified_time: The time when the pipeline version was last modified.
    created_by
    last_modified_by
    last_executed_pipeline_execution_arn: The Amazon Resource Name (ARN) of the most recent pipeline execution created from this pipeline version.
    last_executed_pipeline_execution_display_name: The display name of the most recent pipeline execution created from this pipeline version.
    last_executed_pipeline_execution_status: The status of the most recent pipeline execution created from this pipeline version.
    """

    pipeline_arn: Optional[StrPipeVar] = Unassigned()
    pipeline_version_id: Optional[int] = Unassigned()
    pipeline_version_arn: Optional[StrPipeVar] = Unassigned()
    pipeline_version_display_name: Optional[StrPipeVar] = Unassigned()
    pipeline_version_description: Optional[StrPipeVar] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    last_executed_pipeline_execution_arn: Optional[StrPipeVar] = Unassigned()
    last_executed_pipeline_execution_display_name: Optional[StrPipeVar] = Unassigned()
    last_executed_pipeline_execution_status: Optional[StrPipeVar] = Unassigned()


class ProcessingJob(Base):
    """
    ProcessingJob
      An Amazon SageMaker processing job that is used to analyze data and evaluate models. For more information, see Process Data and Evaluate Models.

    Attributes
    ----------------------
    processing_inputs: List of input configurations for the processing job.
    processing_output_config
    processing_job_name: The name of the processing job.
    processing_resources
    stopping_condition
    app_specification
    environment: Sets the environment variables in the Docker container.
    network_config
    role_arn: The ARN of the role used to create the processing job.
    experiment_config
    processing_job_arn: The ARN of the processing job.
    processing_job_status: The status of the processing job.
    exit_message: A string, up to one KB in size, that contains metadata from the processing container when the processing job exits.
    failure_reason: A string, up to one KB in size, that contains the reason a processing job failed, if it failed.
    processing_end_time: The time that the processing job ended.
    processing_start_time: The time that the processing job started.
    last_modified_time: The time the processing job was last modified.
    creation_time: The time the processing job was created.
    last_modified_by
    created_by
    monitoring_schedule_arn: The ARN of a monitoring schedule for an endpoint associated with this processing job.
    auto_ml_job_arn: The Amazon Resource Name (ARN) of the AutoML job associated with this processing job.
    training_job_arn: The ARN of the training job associated with this processing job.
    tags: An array of key-value pairs. For more information, see Using Cost Allocation Tags in the Amazon Web Services Billing and Cost Management User Guide.
    """

    processing_inputs: Optional[List[ProcessingInput]] = Unassigned()
    processing_output_config: Optional[ProcessingOutputConfig] = Unassigned()
    processing_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    processing_resources: Optional[ProcessingResources] = Unassigned()
    stopping_condition: Optional[ProcessingStoppingCondition] = Unassigned()
    app_specification: Optional[AppSpecification] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    network_config: Optional[NetworkConfig] = Unassigned()
    role_arn: Optional[StrPipeVar] = Unassigned()
    experiment_config: Optional[ExperimentConfig] = Unassigned()
    processing_job_arn: Optional[StrPipeVar] = Unassigned()
    processing_job_status: Optional[StrPipeVar] = Unassigned()
    exit_message: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    processing_end_time: Optional[datetime.datetime] = Unassigned()
    processing_start_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    monitoring_schedule_arn: Optional[StrPipeVar] = Unassigned()
    auto_ml_job_arn: Optional[StrPipeVar] = Unassigned()
    training_job_arn: Optional[StrPipeVar] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class ProfilerConfigForUpdate(Base):
    """
    ProfilerConfigForUpdate
      Configuration information for updating the Amazon SageMaker Debugger profile parameters, system and framework metrics configurations, and storage paths.

    Attributes
    ----------------------
    s3_output_path: Path to Amazon S3 storage location for system and framework metrics.
    profiling_interval_in_milliseconds: A time interval for capturing system metrics in milliseconds. Available values are 100, 200, 500, 1000 (1 second), 5000 (5 seconds), and 60000 (1 minute) milliseconds. The default value is 500 milliseconds.
    profiling_parameters: Configuration information for capturing framework metrics. Available key strings for different profiling options are DetailedProfilingConfig, PythonProfilingConfig, and DataLoaderProfilingConfig. The following codes are configuration structures for the ProfilingParameters parameter. To learn more about how to configure the ProfilingParameters parameter, see Use the SageMaker and Debugger Configuration API Operations to Create, Update, and Debug Your Training Job.
    disable_profiler: To turn off Amazon SageMaker Debugger monitoring and profiling while a training job is in progress, set to True.
    """

    s3_output_path: Optional[StrPipeVar] = Unassigned()
    profiling_interval_in_milliseconds: Optional[int] = Unassigned()
    profiling_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    disable_profiler: Optional[bool] = Unassigned()


class Project(Base):
    """
    Project
      The properties of a project as returned by the Search API.

    Attributes
    ----------------------
    project_arn: The Amazon Resource Name (ARN) of the project.
    project_name: The name of the project.
    project_id: The ID of the project.
    project_description: The description of the project.
    service_catalog_provisioning_details
    service_catalog_provisioned_product_details
    project_status: The status of the project.
    created_by: Who created the project.
    creation_time: A timestamp specifying when the project was created.
    template_provider_details:  An array of template providers associated with the project.
    tags: An array of key-value pairs. You can use tags to categorize your Amazon Web Services resources in different ways, for example, by purpose, owner, or environment. For more information, see Tagging Amazon Web Services Resources.
    last_modified_time: A timestamp container for when the project was last modified.
    last_modified_by
    """

    project_arn: Optional[StrPipeVar] = Unassigned()
    project_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    project_id: Optional[StrPipeVar] = Unassigned()
    project_description: Optional[StrPipeVar] = Unassigned()
    service_catalog_provisioning_details: Optional[ServiceCatalogProvisioningDetails] = Unassigned()
    service_catalog_provisioned_product_details: Optional[
        ServiceCatalogProvisionedProductDetails
    ] = Unassigned()
    project_status: Optional[StrPipeVar] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    template_provider_details: Optional[List[TemplateProviderDetail]] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()


class QueryFilters(Base):
    """
    QueryFilters
      A set of filters to narrow the set of lineage entities connected to the StartArn(s) returned by the QueryLineage API action.

    Attributes
    ----------------------
    types: Filter the lineage entities connected to the StartArn by type. For example: DataSet, Model, Endpoint, or ModelDeployment.
    lineage_types: Filter the lineage entities connected to the StartArn(s) by the type of the lineage entity.
    created_before: Filter the lineage entities connected to the StartArn(s) by created date.
    created_after: Filter the lineage entities connected to the StartArn(s) after the create date.
    modified_before: Filter the lineage entities connected to the StartArn(s) before the last modified date.
    modified_after: Filter the lineage entities connected to the StartArn(s) after the last modified date.
    properties: Filter the lineage entities connected to the StartArn(s) by a set if property key value pairs. If multiple pairs are provided, an entity is included in the results if it matches any of the provided pairs.
    """

    types: Optional[List[StrPipeVar]] = Unassigned()
    lineage_types: Optional[List[StrPipeVar]] = Unassigned()
    created_before: Optional[datetime.datetime] = Unassigned()
    created_after: Optional[datetime.datetime] = Unassigned()
    modified_before: Optional[datetime.datetime] = Unassigned()
    modified_after: Optional[datetime.datetime] = Unassigned()
    properties: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class Vertex(Base):
    """
    Vertex
      A lineage entity connected to the starting entity(ies).

    Attributes
    ----------------------
    arn: The Amazon Resource Name (ARN) of the lineage entity resource.
    type: The type of the lineage entity resource. For example: DataSet, Model, Endpoint, etc...
    lineage_type: The type of resource of the lineage entity.
    """

    arn: Optional[StrPipeVar] = Unassigned()
    type: Optional[StrPipeVar] = Unassigned()
    lineage_type: Optional[StrPipeVar] = Unassigned()


class RemoteDebugConfigForUpdate(Base):
    """
    RemoteDebugConfigForUpdate
      Configuration for remote debugging for the UpdateTrainingJob API. To learn more about the remote debugging functionality of SageMaker, see Access a training container through Amazon Web Services Systems Manager (SSM) for remote debugging.

    Attributes
    ----------------------
    enable_remote_debug: If set to True, enables remote debugging.
    """

    enable_remote_debug: Optional[bool] = Unassigned()


class RenderableTask(Base):
    """
    RenderableTask
      Contains input values for a task.

    Attributes
    ----------------------
    input: A JSON object that contains values for the variables defined in the template. It is made available to the template under the substitution variable task.input. For example, if you define a variable task.input.text in your template, you can supply the variable in the JSON object as "text": "sample text".
    """

    input: StrPipeVar


class RenderingError(Base):
    """
    RenderingError
      A description of an error that occurred while rendering the template.

    Attributes
    ----------------------
    code: A unique identifier for a specific class of errors.
    message: A human-readable message describing the error.
    """

    code: StrPipeVar
    message: StrPipeVar


class ReservedCapacityOffering(Base):
    """
    ReservedCapacityOffering
      Details about a reserved capacity offering for a training plan offering. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .

    Attributes
    ----------------------
    reserved_capacity_type: The type of reserved capacity offering.
    ultra_server_type: The type of UltraServer included in this reserved capacity offering, such as ml.u-p6e-gb200x72.
    ultra_server_count: The number of UltraServers included in this reserved capacity offering.
    instance_type: The instance type for the reserved capacity offering.
    instance_count: The number of instances in the reserved capacity offering.
    availability_zone: The availability zone for the reserved capacity offering.
    duration_hours: The number of whole hours in the total duration for this reserved capacity offering.
    duration_minutes: The additional minutes beyond whole hours in the total duration for this reserved capacity offering.
    start_time: The start time of the reserved capacity offering.
    end_time: The end time of the reserved capacity offering.
    """

    instance_type: StrPipeVar
    instance_count: int
    reserved_capacity_type: Optional[StrPipeVar] = Unassigned()
    ultra_server_type: Optional[StrPipeVar] = Unassigned()
    ultra_server_count: Optional[int] = Unassigned()
    availability_zone: Optional[StrPipeVar] = Unassigned()
    duration_hours: Optional[int] = Unassigned()
    duration_minutes: Optional[int] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()


class ResourceAlreadyExists(Base):
    """
    ResourceAlreadyExists

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class ResourceConfigForUpdate(Base):
    """
    ResourceConfigForUpdate
      The ResourceConfig to update KeepAlivePeriodInSeconds. Other fields in the ResourceConfig cannot be updated.

    Attributes
    ----------------------
    keep_alive_period_in_seconds: The KeepAlivePeriodInSeconds value specified in the ResourceConfig to update.
    """

    keep_alive_period_in_seconds: int


class ResourceInUse(Base):
    """
    ResourceInUse
      Resource being accessed is in use.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class ResourceLimitExceeded(Base):
    """
    ResourceLimitExceeded
       You have exceeded an SageMaker resource limit. For example, you might have too many training jobs created.

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class SearchExpression(Base):
    """
    SearchExpression
      A multi-expression that searches for the specified resource or resources in a search. All resource objects that satisfy the expression's condition are included in the search results. You must specify at least one subexpression, filter, or nested filter. A SearchExpression can contain up to twenty elements. A SearchExpression contains the following components:   A list of Filter objects. Each filter defines a simple Boolean expression comprised of a resource property name, Boolean operator, and value.   A list of NestedFilter objects. Each nested filter defines a list of Boolean expressions using a list of resource properties. A nested filter is satisfied if a single object in the list satisfies all Boolean expressions.   A list of SearchExpression objects. A search expression object can be nested in a list of search expression objects.   A Boolean operator: And or Or.

    Attributes
    ----------------------
    filters: A list of filter objects.
    nested_filters: A list of nested filter objects.
    sub_expressions: A list of search expression objects.
    operator: A Boolean operator used to evaluate the search expression. If you want every conditional statement in all lists to be satisfied for the entire search expression to be true, specify And. If only a single conditional statement needs to be true for the entire search expression to be true, specify Or. The default value is And.
    """

    filters: Optional[List[Filter]] = Unassigned()
    nested_filters: Optional[List[NestedFilters]] = Unassigned()
    sub_expressions: Optional[List["SearchExpression"]] = Unassigned()
    operator: Optional[StrPipeVar] = Unassigned()


class TrainingJob(Base):
    """
    TrainingJob
      Contains information about a training job.

    Attributes
    ----------------------
    training_job_name: The name of the training job.
    training_job_arn: The Amazon Resource Name (ARN) of the training job.
    tuning_job_arn: The Amazon Resource Name (ARN) of the associated hyperparameter tuning job if the training job was launched by a hyperparameter tuning job.
    labeling_job_arn: The Amazon Resource Name (ARN) of the labeling job.
    auto_ml_job_arn: The Amazon Resource Name (ARN) of the job.
    model_artifacts: Information about the Amazon S3 location that is configured for storing model artifacts.
    training_job_output
    training_job_status: The status of the training job. Training job statuses are:    InProgress - The training is in progress.    Completed - The training job has completed.    Failed - The training job has failed. To see the reason for the failure, see the FailureReason field in the response to a DescribeTrainingJobResponse call.    Stopping - The training job is stopping.    Stopped - The training job has stopped.   For more detailed information, see SecondaryStatus.
    secondary_status:  Provides detailed information about the state of the training job. For detailed information about the secondary status of the training job, see StatusMessage under SecondaryStatusTransition. SageMaker provides primary statuses and secondary statuses that apply to each of them:  InProgress     Starting - Starting the training job.    Downloading - An optional stage for algorithms that support File training input mode. It indicates that data is being downloaded to the ML storage volumes.    Training - Training is in progress.    Uploading - Training is complete and the model artifacts are being uploaded to the S3 location.    Completed     Completed - The training job has completed.    Failed     Failed - The training job has failed. The reason for the failure is returned in the FailureReason field of DescribeTrainingJobResponse.    Stopped     MaxRuntimeExceeded - The job stopped because it exceeded the maximum allowed runtime.    Stopped - The training job has stopped.    Stopping     Stopping - Stopping the training job.      Valid values for SecondaryStatus are subject to change.   We no longer support the following secondary statuses:    LaunchingMLInstances     PreparingTrainingStack     DownloadingTrainingImage
    failure_reason: If the training job failed, the reason it failed.
    hyper_parameters: Algorithm-specific parameters.
    algorithm_specification: Information about the algorithm used for training, and algorithm metadata.
    role_arn: The Amazon Web Services Identity and Access Management (IAM) role configured for the training job.
    input_data_config: An array of Channel objects that describes each data input channel. Your input must be in the same Amazon Web Services region as your training job.
    output_data_config: The S3 path where model artifacts that you configured when creating the job are stored. SageMaker creates subfolders for model artifacts.
    resource_config: Resources, including ML compute instances and ML storage volumes, that are configured for model training.
    vpc_config: A VpcConfig object that specifies the VPC that this training job has access to. For more information, see Protect Training Jobs by Using an Amazon Virtual Private Cloud.
    stopping_condition: Specifies a limit to how long a model training job can run. It also specifies how long a managed Spot training job has to complete. When the job reaches the time limit, SageMaker ends the training job. Use this API to cap model training costs. To stop a job, SageMaker sends the algorithm the SIGTERM signal, which delays job termination for 120 seconds. Algorithms can use this 120-second window to save the model artifacts, so the results of training are not lost.
    creation_time: A timestamp that indicates when the training job was created.
    training_start_time: Indicates the time when the training job starts on training instances. You are billed for the time interval between this time and the value of TrainingEndTime. The start time in CloudWatch Logs might be later than this time. The difference is due to the time it takes to download the training data and to the size of the training container.
    training_end_time: Indicates the time when the training job ends on training instances. You are billed for the time interval between the value of TrainingStartTime and this time. For successful jobs and stopped jobs, this is the time after model artifacts are uploaded. For failed jobs, this is the time when SageMaker detects a job failure.
    last_modified_time: A timestamp that indicates when the status of the training job was last modified.
    secondary_status_transitions: A history of all of the secondary statuses that the training job has transitioned through.
    final_metric_data_list: A list of final metric values that are set when the training job completes. Used only if the training job was configured to use metrics.
    enable_network_isolation: If the TrainingJob was created with network isolation, the value is set to true. If network isolation is enabled, nodes can't communicate beyond the VPC they run in.
    enable_inter_container_traffic_encryption: To encrypt all communications between ML compute instances in distributed training, choose True. Encryption provides greater security for distributed training, but training might take longer. How long it takes depends on the amount of communication between compute instances, especially if you use a deep learning algorithm in distributed training.
    enable_managed_spot_training: When true, enables managed spot training using Amazon EC2 Spot instances to run training jobs instead of on-demand instances. For more information, see Managed Spot Training.
    checkpoint_config
    training_time_in_seconds: The training time in seconds.
    billable_time_in_seconds: The billable time in seconds.
    debug_hook_config
    experiment_config
    debug_rule_configurations: Information about the debug rule configuration.
    tensor_board_output_config
    debug_rule_evaluation_statuses: Information about the evaluation status of the rules for the training job.
    output_model_package_arn
    model_package_config
    upstream_platform_config
    profiler_config
    disable_efa
    environment: The environment variables to set in the Docker container.
    retry_strategy: The number of times to retry the job when the job fails due to an InternalServerError.
    last_modified_by
    created_by
    tags: An array of key-value pairs. You can use tags to categorize your Amazon Web Services resources in different ways, for example, by purpose, owner, or environment. For more information, see Tagging Amazon Web Services Resources.
    """

    training_job_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    training_job_arn: Optional[StrPipeVar] = Unassigned()
    tuning_job_arn: Optional[StrPipeVar] = Unassigned()
    labeling_job_arn: Optional[StrPipeVar] = Unassigned()
    auto_ml_job_arn: Optional[StrPipeVar] = Unassigned()
    model_artifacts: Optional[ModelArtifacts] = Unassigned()
    training_job_output: Optional[TrainingJobOutput] = Unassigned()
    training_job_status: Optional[StrPipeVar] = Unassigned()
    secondary_status: Optional[StrPipeVar] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    hyper_parameters: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    algorithm_specification: Optional[AlgorithmSpecification] = Unassigned()
    role_arn: Optional[StrPipeVar] = Unassigned()
    input_data_config: Optional[List[Channel]] = Unassigned()
    output_data_config: Optional[OutputDataConfig] = Unassigned()
    resource_config: Optional[ResourceConfig] = Unassigned()
    vpc_config: Optional[VpcConfig] = Unassigned()
    stopping_condition: Optional[StoppingCondition] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    training_start_time: Optional[datetime.datetime] = Unassigned()
    training_end_time: Optional[datetime.datetime] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    secondary_status_transitions: Optional[List[SecondaryStatusTransition]] = Unassigned()
    final_metric_data_list: Optional[List[MetricData]] = Unassigned()
    enable_network_isolation: Optional[bool] = Unassigned()
    enable_inter_container_traffic_encryption: Optional[bool] = Unassigned()
    enable_managed_spot_training: Optional[bool] = Unassigned()
    checkpoint_config: Optional[CheckpointConfig] = Unassigned()
    training_time_in_seconds: Optional[int] = Unassigned()
    billable_time_in_seconds: Optional[int] = Unassigned()
    debug_hook_config: Optional[DebugHookConfig] = Unassigned()
    experiment_config: Optional[ExperimentConfig] = Unassigned()
    debug_rule_configurations: Optional[List[DebugRuleConfiguration]] = Unassigned()
    tensor_board_output_config: Optional[TensorBoardOutputConfig] = Unassigned()
    debug_rule_evaluation_statuses: Optional[List[DebugRuleEvaluationStatus]] = Unassigned()
    output_model_package_arn: Optional[StrPipeVar] = Unassigned()
    model_package_config: Optional[ModelPackageConfig] = Unassigned()
    upstream_platform_config: Optional[UpstreamPlatformConfig] = Unassigned()
    profiler_config: Optional[ProfilerConfig] = Unassigned()
    disable_efa: Optional[bool] = Unassigned()
    environment: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()
    retry_strategy: Optional[RetryStrategy] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class TrialComponentSimpleSummary(Base):
    """
    TrialComponentSimpleSummary
      A short summary of a trial component.

    Attributes
    ----------------------
    trial_component_name: The name of the trial component.
    trial_component_arn: The Amazon Resource Name (ARN) of the trial component.
    trial_component_source
    creation_time: When the component was created.
    created_by
    """

    trial_component_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    trial_component_arn: Optional[StrPipeVar] = Unassigned()
    trial_component_source: Optional[TrialComponentSource] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()


class Trial(Base):
    """
    Trial
      The properties of a trial as returned by the Search API.

    Attributes
    ----------------------
    trial_name: The name of the trial.
    trial_arn: The Amazon Resource Name (ARN) of the trial.
    display_name: The name of the trial as displayed. If DisplayName isn't specified, TrialName is displayed.
    experiment_name: The name of the experiment the trial is part of.
    source
    creation_time: When the trial was created.
    created_by: Who created the trial.
    last_modified_time: Who last modified the trial.
    last_modified_by
    metadata_properties
    tags: The list of tags that are associated with the trial. You can use Search API to search on the tags.
    trial_component_summaries: A list of the components associated with the trial. For each component, a summary of the component's properties is included.
    """

    trial_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    trial_arn: Optional[StrPipeVar] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    experiment_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    source: Optional[TrialSource] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    metadata_properties: Optional[MetadataProperties] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    trial_component_summaries: Optional[List[TrialComponentSimpleSummary]] = Unassigned()


class TrialComponentSourceDetail(Base):
    """
    TrialComponentSourceDetail
      Detailed information about the source of a trial component. Either ProcessingJob or TrainingJob is returned.

    Attributes
    ----------------------
    source_arn: The Amazon Resource Name (ARN) of the source.
    training_job: Information about a training job that's the source of a trial component.
    processing_job: Information about a processing job that's the source of a trial component.
    transform_job: Information about a transform job that's the source of a trial component.
    """

    source_arn: Optional[StrPipeVar] = Unassigned()
    training_job: Optional[TrainingJob] = Unassigned()
    processing_job: Optional[ProcessingJob] = Unassigned()
    transform_job: Optional[TransformJob] = Unassigned()


class TrialComponent(Base):
    """
    TrialComponent
      The properties of a trial component as returned by the Search API.

    Attributes
    ----------------------
    trial_component_name: The name of the trial component.
    display_name: The name of the component as displayed. If DisplayName isn't specified, TrialComponentName is displayed.
    trial_component_arn: The Amazon Resource Name (ARN) of the trial component.
    source: The Amazon Resource Name (ARN) and job type of the source of the component.
    status
    start_time: When the component started.
    end_time: When the component ended.
    creation_time: When the component was created.
    created_by: Who created the trial component.
    last_modified_time: When the component was last modified.
    last_modified_by
    parameters: The hyperparameters of the component.
    input_artifacts: The input artifacts of the component.
    output_artifacts: The output artifacts of the component.
    metrics: The metrics for the component.
    metadata_properties
    source_detail: Details of the source of the component.
    lineage_group_arn: The Amazon Resource Name (ARN) of the lineage group resource.
    tags: The list of tags that are associated with the component. You can use Search API to search on the tags.
    parents: An array of the parents of the component. A parent is a trial the component is associated with and the experiment the trial is part of. A component might not have any parents.
    run_name: The name of the experiment run.
    """

    trial_component_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    display_name: Optional[StrPipeVar] = Unassigned()
    trial_component_arn: Optional[StrPipeVar] = Unassigned()
    source: Optional[TrialComponentSource] = Unassigned()
    status: Optional[TrialComponentStatus] = Unassigned()
    start_time: Optional[datetime.datetime] = Unassigned()
    end_time: Optional[datetime.datetime] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    created_by: Optional[UserContext] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    last_modified_by: Optional[UserContext] = Unassigned()
    parameters: Optional[Dict[StrPipeVar, TrialComponentParameterValue]] = Unassigned()
    input_artifacts: Optional[Dict[StrPipeVar, TrialComponentArtifact]] = Unassigned()
    output_artifacts: Optional[Dict[StrPipeVar, TrialComponentArtifact]] = Unassigned()
    metrics: Optional[List[TrialComponentMetricSummary]] = Unassigned()
    metadata_properties: Optional[MetadataProperties] = Unassigned()
    source_detail: Optional[TrialComponentSourceDetail] = Unassigned()
    lineage_group_arn: Optional[StrPipeVar] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()
    parents: Optional[List[Parent]] = Unassigned()
    run_name: Optional[StrPipeVar] = Unassigned()


class UserProfile(Base):
    """
    UserProfile

    Attributes
    ----------------------
    domain_id
    user_profile_arn
    user_profile_name
    home_efs_file_system_uid
    status
    last_modified_time
    creation_time
    failure_reason
    single_sign_on_user_identifier
    single_sign_on_user_value
    user_policy
    user_settings
    tags
    """

    domain_id: Optional[StrPipeVar] = Unassigned()
    user_profile_arn: Optional[StrPipeVar] = Unassigned()
    user_profile_name: Optional[Union[StrPipeVar, object]] = Unassigned()
    home_efs_file_system_uid: Optional[StrPipeVar] = Unassigned()
    status: Optional[StrPipeVar] = Unassigned()
    last_modified_time: Optional[datetime.datetime] = Unassigned()
    creation_time: Optional[datetime.datetime] = Unassigned()
    failure_reason: Optional[StrPipeVar] = Unassigned()
    single_sign_on_user_identifier: Optional[StrPipeVar] = Unassigned()
    single_sign_on_user_value: Optional[StrPipeVar] = Unassigned()
    user_policy: Optional[StrPipeVar] = Unassigned()
    user_settings: Optional[UserSettings] = Unassigned()
    tags: Optional[List[Tag]] = Unassigned()


class SearchRecord(Base):
    """
    SearchRecord
      A single resource returned as part of the Search API response.

    Attributes
    ----------------------
    training_job: The properties of a training job.
    experiment: The properties of an experiment.
    trial: The properties of a trial.
    trial_component: The properties of a trial component.
    transform_job
    endpoint
    model_package
    model_package_group
    pipeline
    pipeline_execution
    pipeline_version: The version of the pipeline.
    feature_group
    feature_metadata: The feature metadata used to search through the features.
    image
    image_version
    project: The properties of a project.
    hyper_parameter_tuning_job: The properties of a hyperparameter tuning job.
    model_card: An Amazon SageMaker Model Card that documents details about a machine learning model.
    model
    app
    user_profile
    domain
    """

    training_job: Optional[TrainingJob] = Unassigned()
    experiment: Optional[Experiment] = Unassigned()
    trial: Optional[Trial] = Unassigned()
    trial_component: Optional[TrialComponent] = Unassigned()
    transform_job: Optional[TransformJob] = Unassigned()
    endpoint: Optional[Endpoint] = Unassigned()
    model_package: Optional[ModelPackage] = Unassigned()
    model_package_group: Optional[ModelPackageGroup] = Unassigned()
    pipeline: Optional[Pipeline] = Unassigned()
    pipeline_execution: Optional[PipelineExecution] = Unassigned()
    pipeline_version: Optional[PipelineVersion] = Unassigned()
    feature_group: Optional[FeatureGroup] = Unassigned()
    feature_metadata: Optional[FeatureMetadata] = Unassigned()
    image: Optional[ImageSearchShape] = Unassigned()
    image_version: Optional[ImageVersionSearchShape] = Unassigned()
    project: Optional[Project] = Unassigned()
    hyper_parameter_tuning_job: Optional[HyperParameterTuningJobSearchEntity] = Unassigned()
    model_card: Optional[ModelCard] = Unassigned()
    model: Optional[ModelDashboardModel] = Unassigned()
    app: Optional[App] = Unassigned()
    user_profile: Optional[UserProfile] = Unassigned()
    domain: Optional[Domain] = Unassigned()


class VisibilityConditions(Base):
    """
    VisibilityConditions
      The list of key-value pairs used to filter your search results. If a search result contains a key from your list, it is included in the final search response if the value associated with the key in the result matches the value you specified. If the value doesn't match, the result is excluded from the search response. Any resources that don't have a key from the list that you've provided will also be included in the search response.

    Attributes
    ----------------------
    key: The key that specifies the tag that you're using to filter the search results. It must be in the following format: Tags.&lt;key&gt;.
    value: The value for the tag that you're using to filter the search results.
    """

    key: Optional[StrPipeVar] = Unassigned()
    value: Optional[StrPipeVar] = Unassigned()


class TotalHits(Base):
    """
    TotalHits
      Represents the total number of matching results and indicates how accurate that count is. The Value field provides the count, which may be exact or estimated. The Relation field indicates whether it's an exact figure or a lower bound. This helps understand the full scope of search results, especially when dealing with large result sets.

    Attributes
    ----------------------
    value: The total number of matching results. This value may be exact or an estimate, depending on the Relation field.
    relation: Indicates the relationship between the returned Value and the actual total number of matching results. Possible values are:    EqualTo: The Value is the exact count of matching results.    GreaterThanOrEqualTo: The Value is a lower bound of the actual count of matching results.
    """

    value: Optional[int] = Unassigned()
    relation: Optional[StrPipeVar] = Unassigned()


class TrainingPlanOffering(Base):
    """
    TrainingPlanOffering
      Details about a training plan offering. For more information about how to reserve GPU capacity for your SageMaker HyperPod clusters using Amazon SageMaker Training Plan, see  CreateTrainingPlan .

    Attributes
    ----------------------
    training_plan_offering_id: The unique identifier for this training plan offering.
    target_resources: The target resources (e.g., SageMaker Training Jobs, SageMaker HyperPod) for this training plan offering. Training plans are specific to their target resource.   A training plan designed for SageMaker training jobs can only be used to schedule and run training jobs.   A training plan for HyperPod clusters can be used exclusively to provide compute resources to a cluster's instance group.
    requested_start_time_after: The requested start time that the user specified when searching for the training plan offering.
    requested_end_time_before: The requested end time that the user specified when searching for the training plan offering.
    duration_hours: The number of whole hours in the total duration for this training plan offering.
    duration_minutes: The additional minutes beyond whole hours in the total duration for this training plan offering.
    upfront_fee: The upfront fee for this training plan offering.
    currency_code: The currency code for the upfront fee (e.g., USD).
    reserved_capacity_offerings: A list of reserved capacity offerings associated with this training plan offering.
    """

    training_plan_offering_id: StrPipeVar
    target_resources: List[StrPipeVar]
    requested_start_time_after: Optional[datetime.datetime] = Unassigned()
    requested_end_time_before: Optional[datetime.datetime] = Unassigned()
    duration_hours: Optional[int] = Unassigned()
    duration_minutes: Optional[int] = Unassigned()
    upfront_fee: Optional[StrPipeVar] = Unassigned()
    currency_code: Optional[StrPipeVar] = Unassigned()
    reserved_capacity_offerings: Optional[List[ReservedCapacityOffering]] = Unassigned()


class ServiceCatalogProvisioningUpdateDetails(Base):
    """
    ServiceCatalogProvisioningUpdateDetails
      Details that you specify to provision a service catalog product. For information about service catalog, see What is Amazon Web Services Service Catalog.

    Attributes
    ----------------------
    provisioning_artifact_id: The ID of the provisioning artifact.
    provisioning_parameters: A list of key value pairs that you specify when you provision a product.
    """

    provisioning_artifact_id: Optional[StrPipeVar] = Unassigned()
    provisioning_parameters: Optional[List[ProvisioningParameter]] = Unassigned()


class StudioUserSettings(Base):
    """
    StudioUserSettings

    Attributes
    ----------------------
    space_storage_settings
    default_landing_uri
    """

    space_storage_settings: Optional[SpaceStorageSettings] = Unassigned()
    default_landing_uri: Optional[StrPipeVar] = Unassigned()


class TagrisAccessDeniedException(Base):
    """
    TagrisAccessDeniedException

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class TagrisInternalServiceException(Base):
    """
    TagrisInternalServiceException

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class TagrisSweepListItem(Base):
    """
    TagrisSweepListItem

    Attributes
    ----------------------
    tagris_account_id
    tagris_amazon_resource_name
    tagris_internal_id
    tagris_version
    """

    tagris_account_id: Optional[StrPipeVar] = Unassigned()
    tagris_amazon_resource_name: Optional[StrPipeVar] = Unassigned()
    tagris_internal_id: Optional[StrPipeVar] = Unassigned()
    tagris_version: Optional[int] = Unassigned()


class TagrisInvalidArnException(Base):
    """
    TagrisInvalidArnException

    Attributes
    ----------------------
    message
    sweep_list_item
    """

    message: Optional[StrPipeVar] = Unassigned()
    sweep_list_item: Optional[TagrisSweepListItem] = Unassigned()


class TagrisInvalidParameterException(Base):
    """
    TagrisInvalidParameterException

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class TagrisPartialResourcesExistResultsException(Base):
    """
    TagrisPartialResourcesExistResultsException

    Attributes
    ----------------------
    message
    resource_existence_information
    """

    message: Optional[StrPipeVar] = Unassigned()
    resource_existence_information: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class TagrisThrottledException(Base):
    """
    TagrisThrottledException

    Attributes
    ----------------------
    message
    """

    message: Optional[StrPipeVar] = Unassigned()


class ThroughputConfigUpdate(Base):
    """
    ThroughputConfigUpdate
      The new throughput configuration for the feature group. You can switch between on-demand and provisioned modes or update the read / write capacity of provisioned feature groups. You can switch a feature group to on-demand only once in a 24 hour period.

    Attributes
    ----------------------
    throughput_mode: Target throughput mode of the feature group. Throughput update is an asynchronous operation, and the outcome should be monitored by polling LastUpdateStatus field in DescribeFeatureGroup response. You cannot update a feature group's throughput while another update is in progress.
    provisioned_read_capacity_units: For provisioned feature groups with online store enabled, this indicates the read throughput you are billed for and can consume without throttling.
    provisioned_write_capacity_units: For provisioned feature groups, this indicates the write throughput you are billed for and can consume without throttling.
    """

    throughput_mode: Optional[StrPipeVar] = Unassigned()
    provisioned_read_capacity_units: Optional[int] = Unassigned()
    provisioned_write_capacity_units: Optional[int] = Unassigned()


class UpdateClusterSoftwareInstanceGroupSpecification(Base):
    """
    UpdateClusterSoftwareInstanceGroupSpecification
      The configuration that describes specifications of the instance groups to update.

    Attributes
    ----------------------
    instance_group_name: The name of the instance group to update.
    custom_metadata
    """

    instance_group_name: StrPipeVar
    custom_metadata: Optional[Dict[StrPipeVar, StrPipeVar]] = Unassigned()


class VariantProperty(Base):
    """
    VariantProperty
      Specifies a production variant property type for an Endpoint. If you are updating an endpoint with the RetainAllVariantProperties option of UpdateEndpointInput set to true, the VariantProperty objects listed in the ExcludeRetainedVariantProperties parameter of UpdateEndpointInput override the existing variant properties of the endpoint.

    Attributes
    ----------------------
    variant_property_type: The type of variant property. The supported values are:    DesiredInstanceCount: Overrides the existing variant instance counts using the InitialInstanceCount values in the ProductionVariants of CreateEndpointConfig.    DesiredWeight: Overrides the existing variant weights using the InitialVariantWeight values in the ProductionVariants of CreateEndpointConfig.    DataCaptureConfig: (Not currently supported.)
    """

    variant_property_type: StrPipeVar


class UpdateTemplateProvider(Base):
    """
    UpdateTemplateProvider
       Contains configuration details for updating an existing template provider in the project.

    Attributes
    ----------------------
    cfn_template_provider:  The CloudFormation template provider configuration to update.
    """

    cfn_template_provider: Optional[CfnUpdateTemplateProvider] = Unassigned()
