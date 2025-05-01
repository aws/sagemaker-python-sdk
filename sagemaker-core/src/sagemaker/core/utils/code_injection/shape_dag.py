SHAPE_DAG = {
    "AccessForbidden": {
        "members": [{"name": "Message", "shape": "Message", "type": "string"}],
        "type": "structure",
    },
    "ActionSource": {
        "members": [
            {"name": "SourceUri", "shape": "SourceUri", "type": "string"},
            {"name": "SourceType", "shape": "String256", "type": "string"},
            {"name": "SourceId", "shape": "String256", "type": "string"},
        ],
        "type": "structure",
    },
    "ActionSummaries": {
        "member_shape": "ActionSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ActionSummary": {
        "members": [
            {"name": "ActionArn", "shape": "ActionArn", "type": "string"},
            {"name": "ActionName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "ActionSource", "type": "structure"},
            {"name": "ActionType", "shape": "String64", "type": "string"},
            {"name": "Status", "shape": "ActionStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "AddAssociationRequest": {
        "members": [
            {"name": "SourceArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "DestinationArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "AssociationType", "shape": "AssociationEdgeType", "type": "string"},
        ],
        "type": "structure",
    },
    "AddAssociationResponse": {
        "members": [
            {"name": "SourceArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "DestinationArn", "shape": "AssociationEntityArn", "type": "string"},
        ],
        "type": "structure",
    },
    "AddTagsInput": {
        "members": [
            {"name": "ResourceArn", "shape": "ResourceArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "AddTagsOutput": {
        "members": [{"name": "Tags", "shape": "TagList", "type": "list"}],
        "type": "structure",
    },
    "AdditionalCodeRepositoryNamesOrUrls": {
        "member_shape": "CodeRepositoryNameOrUrl",
        "member_type": "string",
        "type": "list",
    },
    "AdditionalInferenceSpecificationDefinition": {
        "members": [
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
            {"name": "Containers", "shape": "ModelPackageContainerDefinitionList", "type": "list"},
            {
                "name": "SupportedTransformInstanceTypes",
                "shape": "TransformInstanceTypes",
                "type": "list",
            },
            {
                "name": "SupportedRealtimeInferenceInstanceTypes",
                "shape": "RealtimeInferenceInstanceTypes",
                "type": "list",
            },
            {"name": "SupportedContentTypes", "shape": "ContentTypes", "type": "list"},
            {"name": "SupportedResponseMIMETypes", "shape": "ResponseMIMETypes", "type": "list"},
        ],
        "type": "structure",
    },
    "AdditionalInferenceSpecifications": {
        "member_shape": "AdditionalInferenceSpecificationDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "AdditionalModelDataSource": {
        "members": [
            {"name": "ChannelName", "shape": "AdditionalModelChannelName", "type": "string"},
            {"name": "S3DataSource", "shape": "S3ModelDataSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "AdditionalModelDataSources": {
        "member_shape": "AdditionalModelDataSource",
        "member_type": "structure",
        "type": "list",
    },
    "AdditionalS3DataSource": {
        "members": [
            {"name": "S3DataType", "shape": "AdditionalS3DataSourceDataType", "type": "string"},
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "CompressionType", "shape": "CompressionType", "type": "string"},
            {"name": "ETag", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "AgentVersion": {
        "members": [
            {"name": "Version", "shape": "EdgeVersion", "type": "string"},
            {"name": "AgentCount", "shape": "Long", "type": "long"},
        ],
        "type": "structure",
    },
    "AgentVersions": {"member_shape": "AgentVersion", "member_type": "structure", "type": "list"},
    "AggregationTransformations": {
        "key_shape": "TransformationAttributeName",
        "key_type": "string",
        "type": "map",
        "value_shape": "AggregationTransformationValue",
        "value_type": "string",
    },
    "Alarm": {
        "members": [{"name": "AlarmName", "shape": "AlarmName", "type": "string"}],
        "type": "structure",
    },
    "AlarmList": {"member_shape": "Alarm", "member_type": "structure", "type": "list"},
    "AlgorithmSpecification": {
        "members": [
            {"name": "TrainingImage", "shape": "AlgorithmImage", "type": "string"},
            {"name": "AlgorithmName", "shape": "ArnOrName", "type": "string"},
            {"name": "TrainingInputMode", "shape": "TrainingInputMode", "type": "string"},
            {"name": "MetricDefinitions", "shape": "MetricDefinitionList", "type": "list"},
            {"name": "EnableSageMakerMetricsTimeSeries", "shape": "Boolean", "type": "boolean"},
            {"name": "ContainerEntrypoint", "shape": "TrainingContainerEntrypoint", "type": "list"},
            {"name": "ContainerArguments", "shape": "TrainingContainerArguments", "type": "list"},
            {"name": "TrainingImageConfig", "shape": "TrainingImageConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "AlgorithmStatusDetails": {
        "members": [
            {"name": "ValidationStatuses", "shape": "AlgorithmStatusItemList", "type": "list"},
            {"name": "ImageScanStatuses", "shape": "AlgorithmStatusItemList", "type": "list"},
        ],
        "type": "structure",
    },
    "AlgorithmStatusItem": {
        "members": [
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "Status", "shape": "DetailedAlgorithmStatus", "type": "string"},
            {"name": "FailureReason", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "AlgorithmStatusItemList": {
        "member_shape": "AlgorithmStatusItem",
        "member_type": "structure",
        "type": "list",
    },
    "AlgorithmSummary": {
        "members": [
            {"name": "AlgorithmName", "shape": "EntityName", "type": "string"},
            {"name": "AlgorithmArn", "shape": "AlgorithmArn", "type": "string"},
            {"name": "AlgorithmDescription", "shape": "EntityDescription", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "AlgorithmStatus", "shape": "AlgorithmStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "AlgorithmSummaryList": {
        "member_shape": "AlgorithmSummary",
        "member_type": "structure",
        "type": "list",
    },
    "AlgorithmValidationProfile": {
        "members": [
            {"name": "ProfileName", "shape": "EntityName", "type": "string"},
            {
                "name": "TrainingJobDefinition",
                "shape": "TrainingJobDefinition",
                "type": "structure",
            },
            {
                "name": "TransformJobDefinition",
                "shape": "TransformJobDefinition",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "AlgorithmValidationProfiles": {
        "member_shape": "AlgorithmValidationProfile",
        "member_type": "structure",
        "type": "list",
    },
    "AlgorithmValidationSpecification": {
        "members": [
            {"name": "ValidationRole", "shape": "RoleArn", "type": "string"},
            {"name": "ValidationProfiles", "shape": "AlgorithmValidationProfiles", "type": "list"},
        ],
        "type": "structure",
    },
    "AmazonQSettings": {
        "members": [
            {"name": "Status", "shape": "FeatureStatus", "type": "string"},
            {"name": "QProfileArn", "shape": "QProfileArn", "type": "string"},
        ],
        "type": "structure",
    },
    "AnnotationConsolidationConfig": {
        "members": [
            {
                "name": "AnnotationConsolidationLambdaArn",
                "shape": "LambdaFunctionArn",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "AppDetails": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "AppType", "shape": "AppType", "type": "string"},
            {"name": "AppName", "shape": "AppName", "type": "string"},
            {"name": "Status", "shape": "AppStatus", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "ResourceSpec", "shape": "ResourceSpec", "type": "structure"},
        ],
        "type": "structure",
    },
    "AppImageConfigDetails": {
        "members": [
            {"name": "AppImageConfigArn", "shape": "AppImageConfigArn", "type": "string"},
            {"name": "AppImageConfigName", "shape": "AppImageConfigName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "KernelGatewayImageConfig",
                "shape": "KernelGatewayImageConfig",
                "type": "structure",
            },
            {
                "name": "JupyterLabAppImageConfig",
                "shape": "JupyterLabAppImageConfig",
                "type": "structure",
            },
            {
                "name": "CodeEditorAppImageConfig",
                "shape": "CodeEditorAppImageConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "AppImageConfigList": {
        "member_shape": "AppImageConfigDetails",
        "member_type": "structure",
        "type": "list",
    },
    "AppLifecycleManagement": {
        "members": [{"name": "IdleSettings", "shape": "IdleSettings", "type": "structure"}],
        "type": "structure",
    },
    "AppList": {"member_shape": "AppDetails", "member_type": "structure", "type": "list"},
    "AppSpecification": {
        "members": [
            {"name": "ImageUri", "shape": "ImageUri", "type": "string"},
            {"name": "ContainerEntrypoint", "shape": "ContainerEntrypoint", "type": "list"},
            {"name": "ContainerArguments", "shape": "ContainerArguments", "type": "list"},
        ],
        "type": "structure",
    },
    "ArtifactProperties": {
        "key_shape": "StringParameterValue",
        "key_type": "string",
        "type": "map",
        "value_shape": "ArtifactPropertyValue",
        "value_type": "string",
    },
    "ArtifactSource": {
        "members": [
            {"name": "SourceUri", "shape": "SourceUri", "type": "string"},
            {"name": "SourceTypes", "shape": "ArtifactSourceTypes", "type": "list"},
        ],
        "type": "structure",
    },
    "ArtifactSourceType": {
        "members": [
            {"name": "SourceIdType", "shape": "ArtifactSourceIdType", "type": "string"},
            {"name": "Value", "shape": "String256", "type": "string"},
        ],
        "type": "structure",
    },
    "ArtifactSourceTypes": {
        "member_shape": "ArtifactSourceType",
        "member_type": "structure",
        "type": "list",
    },
    "ArtifactSummaries": {
        "member_shape": "ArtifactSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ArtifactSummary": {
        "members": [
            {"name": "ArtifactArn", "shape": "ArtifactArn", "type": "string"},
            {"name": "ArtifactName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "ArtifactSource", "type": "structure"},
            {"name": "ArtifactType", "shape": "String256", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "AssociateTrialComponentRequest": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "AssociateTrialComponentResponse": {
        "members": [
            {"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"},
            {"name": "TrialArn", "shape": "TrialArn", "type": "string"},
        ],
        "type": "structure",
    },
    "AssociationSummaries": {
        "member_shape": "AssociationSummary",
        "member_type": "structure",
        "type": "list",
    },
    "AssociationSummary": {
        "members": [
            {"name": "SourceArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "DestinationArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "SourceType", "shape": "String256", "type": "string"},
            {"name": "DestinationType", "shape": "String256", "type": "string"},
            {"name": "AssociationType", "shape": "AssociationEdgeType", "type": "string"},
            {"name": "SourceName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DestinationName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "AssumableRoleArns": {"member_shape": "RoleArn", "member_type": "string", "type": "list"},
    "AsyncInferenceClientConfig": {
        "members": [
            {
                "name": "MaxConcurrentInvocationsPerInstance",
                "shape": "MaxConcurrentInvocationsPerInstance",
                "type": "integer",
            }
        ],
        "type": "structure",
    },
    "AsyncInferenceConfig": {
        "members": [
            {"name": "ClientConfig", "shape": "AsyncInferenceClientConfig", "type": "structure"},
            {"name": "OutputConfig", "shape": "AsyncInferenceOutputConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "AsyncInferenceNotificationConfig": {
        "members": [
            {"name": "SuccessTopic", "shape": "SnsTopicArn", "type": "string"},
            {"name": "ErrorTopic", "shape": "SnsTopicArn", "type": "string"},
            {
                "name": "IncludeInferenceResponseIn",
                "shape": "AsyncNotificationTopicTypeList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "AsyncInferenceOutputConfig": {
        "members": [
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "S3OutputPath", "shape": "DestinationS3Uri", "type": "string"},
            {
                "name": "NotificationConfig",
                "shape": "AsyncInferenceNotificationConfig",
                "type": "structure",
            },
            {"name": "S3FailurePath", "shape": "DestinationS3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "AsyncNotificationTopicTypeList": {
        "member_shape": "AsyncNotificationTopicTypes",
        "member_type": "string",
        "type": "list",
    },
    "AthenaDatasetDefinition": {
        "members": [
            {"name": "Catalog", "shape": "AthenaCatalog", "type": "string"},
            {"name": "Database", "shape": "AthenaDatabase", "type": "string"},
            {"name": "QueryString", "shape": "AthenaQueryString", "type": "string"},
            {"name": "WorkGroup", "shape": "AthenaWorkGroup", "type": "string"},
            {"name": "OutputS3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "OutputFormat", "shape": "AthenaResultFormat", "type": "string"},
            {"name": "OutputCompression", "shape": "AthenaResultCompressionType", "type": "string"},
        ],
        "type": "structure",
    },
    "AttributeNames": {"member_shape": "AttributeName", "member_type": "string", "type": "list"},
    "AuthenticationRequestExtraParams": {
        "key_shape": "AuthenticationRequestExtraParamsKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "AuthenticationRequestExtraParamsValue",
        "value_type": "string",
    },
    "AutoMLAlgorithmConfig": {
        "members": [{"name": "AutoMLAlgorithms", "shape": "AutoMLAlgorithms", "type": "list"}],
        "type": "structure",
    },
    "AutoMLAlgorithms": {
        "member_shape": "AutoMLAlgorithm",
        "member_type": "string",
        "type": "list",
    },
    "AutoMLAlgorithmsConfig": {
        "member_shape": "AutoMLAlgorithmConfig",
        "member_type": "structure",
        "type": "list",
    },
    "AutoMLCandidate": {
        "members": [
            {"name": "CandidateName", "shape": "CandidateName", "type": "string"},
            {
                "name": "FinalAutoMLJobObjectiveMetric",
                "shape": "FinalAutoMLJobObjectiveMetric",
                "type": "structure",
            },
            {"name": "ObjectiveStatus", "shape": "ObjectiveStatus", "type": "string"},
            {"name": "CandidateSteps", "shape": "CandidateSteps", "type": "list"},
            {"name": "CandidateStatus", "shape": "CandidateStatus", "type": "string"},
            {"name": "InferenceContainers", "shape": "AutoMLContainerDefinitions", "type": "list"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "AutoMLFailureReason", "type": "string"},
            {"name": "CandidateProperties", "shape": "CandidateProperties", "type": "structure"},
            {
                "name": "InferenceContainerDefinitions",
                "shape": "AutoMLInferenceContainerDefinitions",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "AutoMLCandidateGenerationConfig": {
        "members": [
            {"name": "FeatureSpecificationS3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "AlgorithmsConfig", "shape": "AutoMLAlgorithmsConfig", "type": "list"},
        ],
        "type": "structure",
    },
    "AutoMLCandidateStep": {
        "members": [
            {"name": "CandidateStepType", "shape": "CandidateStepType", "type": "string"},
            {"name": "CandidateStepArn", "shape": "CandidateStepArn", "type": "string"},
            {"name": "CandidateStepName", "shape": "CandidateStepName", "type": "string"},
        ],
        "type": "structure",
    },
    "AutoMLCandidates": {
        "member_shape": "AutoMLCandidate",
        "member_type": "structure",
        "type": "list",
    },
    "AutoMLChannel": {
        "members": [
            {"name": "DataSource", "shape": "AutoMLDataSource", "type": "structure"},
            {"name": "CompressionType", "shape": "CompressionType", "type": "string"},
            {"name": "TargetAttributeName", "shape": "TargetAttributeName", "type": "string"},
            {"name": "ContentType", "shape": "ContentType", "type": "string"},
            {"name": "ChannelType", "shape": "AutoMLChannelType", "type": "string"},
            {
                "name": "SampleWeightAttributeName",
                "shape": "SampleWeightAttributeName",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "AutoMLComputeConfig": {
        "members": [
            {
                "name": "EmrServerlessComputeConfig",
                "shape": "EmrServerlessComputeConfig",
                "type": "structure",
            }
        ],
        "type": "structure",
    },
    "AutoMLContainerDefinition": {
        "members": [
            {"name": "Image", "shape": "ContainerImage", "type": "string"},
            {"name": "ModelDataUrl", "shape": "Url", "type": "string"},
            {"name": "Environment", "shape": "EnvironmentMap", "type": "map"},
        ],
        "type": "structure",
    },
    "AutoMLContainerDefinitions": {
        "member_shape": "AutoMLContainerDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "AutoMLDataSource": {
        "members": [{"name": "S3DataSource", "shape": "AutoMLS3DataSource", "type": "structure"}],
        "type": "structure",
    },
    "AutoMLDataSplitConfig": {
        "members": [{"name": "ValidationFraction", "shape": "ValidationFraction", "type": "float"}],
        "type": "structure",
    },
    "AutoMLInferenceContainerDefinitions": {
        "key_shape": "AutoMLProcessingUnit",
        "key_type": "string",
        "type": "map",
        "value_shape": "AutoMLContainerDefinitions",
        "value_type": "list",
    },
    "AutoMLInputDataConfig": {
        "member_shape": "AutoMLChannel",
        "member_type": "structure",
        "type": "list",
    },
    "AutoMLJobArtifacts": {
        "members": [
            {
                "name": "CandidateDefinitionNotebookLocation",
                "shape": "CandidateDefinitionNotebookLocation",
                "type": "string",
            },
            {
                "name": "DataExplorationNotebookLocation",
                "shape": "DataExplorationNotebookLocation",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "AutoMLJobChannel": {
        "members": [
            {"name": "ChannelType", "shape": "AutoMLChannelType", "type": "string"},
            {"name": "ContentType", "shape": "ContentType", "type": "string"},
            {"name": "CompressionType", "shape": "CompressionType", "type": "string"},
            {"name": "DataSource", "shape": "AutoMLDataSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "AutoMLJobCompletionCriteria": {
        "members": [
            {"name": "MaxCandidates", "shape": "MaxCandidates", "type": "integer"},
            {
                "name": "MaxRuntimePerTrainingJobInSeconds",
                "shape": "MaxRuntimePerTrainingJobInSeconds",
                "type": "integer",
            },
            {
                "name": "MaxAutoMLJobRuntimeInSeconds",
                "shape": "MaxAutoMLJobRuntimeInSeconds",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "AutoMLJobConfig": {
        "members": [
            {
                "name": "CompletionCriteria",
                "shape": "AutoMLJobCompletionCriteria",
                "type": "structure",
            },
            {"name": "SecurityConfig", "shape": "AutoMLSecurityConfig", "type": "structure"},
            {
                "name": "CandidateGenerationConfig",
                "shape": "AutoMLCandidateGenerationConfig",
                "type": "structure",
            },
            {"name": "DataSplitConfig", "shape": "AutoMLDataSplitConfig", "type": "structure"},
            {"name": "Mode", "shape": "AutoMLMode", "type": "string"},
        ],
        "type": "structure",
    },
    "AutoMLJobInputDataConfig": {
        "member_shape": "AutoMLJobChannel",
        "member_type": "structure",
        "type": "list",
    },
    "AutoMLJobObjective": {
        "members": [{"name": "MetricName", "shape": "AutoMLMetricEnum", "type": "string"}],
        "type": "structure",
    },
    "AutoMLJobStepMetadata": {
        "members": [{"name": "Arn", "shape": "AutoMLJobArn", "type": "string"}],
        "type": "structure",
    },
    "AutoMLJobSummaries": {
        "member_shape": "AutoMLJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "AutoMLJobSummary": {
        "members": [
            {"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {"name": "AutoMLJobStatus", "shape": "AutoMLJobStatus", "type": "string"},
            {
                "name": "AutoMLJobSecondaryStatus",
                "shape": "AutoMLJobSecondaryStatus",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "AutoMLFailureReason", "type": "string"},
            {
                "name": "PartialFailureReasons",
                "shape": "AutoMLPartialFailureReasons",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "AutoMLOutputDataConfig": {
        "members": [
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "AutoMLPartialFailureReason": {
        "members": [
            {"name": "PartialFailureMessage", "shape": "AutoMLFailureReason", "type": "string"}
        ],
        "type": "structure",
    },
    "AutoMLPartialFailureReasons": {
        "member_shape": "AutoMLPartialFailureReason",
        "member_type": "structure",
        "type": "list",
    },
    "AutoMLProblemTypeConfig": {
        "members": [
            {
                "name": "ImageClassificationJobConfig",
                "shape": "ImageClassificationJobConfig",
                "type": "structure",
            },
            {
                "name": "TextClassificationJobConfig",
                "shape": "TextClassificationJobConfig",
                "type": "structure",
            },
            {
                "name": "TimeSeriesForecastingJobConfig",
                "shape": "TimeSeriesForecastingJobConfig",
                "type": "structure",
            },
            {"name": "TabularJobConfig", "shape": "TabularJobConfig", "type": "structure"},
            {
                "name": "TextGenerationJobConfig",
                "shape": "TextGenerationJobConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "AutoMLProblemTypeResolvedAttributes": {
        "members": [
            {
                "name": "TabularResolvedAttributes",
                "shape": "TabularResolvedAttributes",
                "type": "structure",
            },
            {
                "name": "TextGenerationResolvedAttributes",
                "shape": "TextGenerationResolvedAttributes",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "AutoMLResolvedAttributes": {
        "members": [
            {"name": "AutoMLJobObjective", "shape": "AutoMLJobObjective", "type": "structure"},
            {
                "name": "CompletionCriteria",
                "shape": "AutoMLJobCompletionCriteria",
                "type": "structure",
            },
            {
                "name": "AutoMLProblemTypeResolvedAttributes",
                "shape": "AutoMLProblemTypeResolvedAttributes",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "AutoMLS3DataSource": {
        "members": [
            {"name": "S3DataType", "shape": "AutoMLS3DataType", "type": "string"},
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "AutoMLSecurityConfig": {
        "members": [
            {"name": "VolumeKmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {
                "name": "EnableInterContainerTrafficEncryption",
                "shape": "Boolean",
                "type": "boolean",
            },
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "AutoParameter": {
        "members": [
            {"name": "Name", "shape": "ParameterKey", "type": "string"},
            {"name": "ValueHint", "shape": "ParameterValue", "type": "string"},
        ],
        "type": "structure",
    },
    "AutoParameters": {"member_shape": "AutoParameter", "member_type": "structure", "type": "list"},
    "AutoRollbackConfig": {
        "members": [{"name": "Alarms", "shape": "AlarmList", "type": "list"}],
        "type": "structure",
    },
    "Autotune": {
        "members": [{"name": "Mode", "shape": "AutotuneMode", "type": "string"}],
        "type": "structure",
    },
    "BatchDataCaptureConfig": {
        "members": [
            {"name": "DestinationS3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "GenerateInferenceId", "shape": "Boolean", "type": "boolean"},
        ],
        "type": "structure",
    },
    "BatchDeleteClusterNodesError": {
        "members": [
            {"name": "Code", "shape": "BatchDeleteClusterNodesErrorCode", "type": "string"},
            {"name": "Message", "shape": "String", "type": "string"},
            {"name": "NodeId", "shape": "ClusterNodeId", "type": "string"},
        ],
        "type": "structure",
    },
    "BatchDeleteClusterNodesErrorList": {
        "member_shape": "BatchDeleteClusterNodesError",
        "member_type": "structure",
        "type": "list",
    },
    "BatchDeleteClusterNodesRequest": {
        "members": [
            {"name": "ClusterName", "shape": "ClusterNameOrArn", "type": "string"},
            {"name": "NodeIds", "shape": "ClusterNodeIds", "type": "list"},
        ],
        "type": "structure",
    },
    "BatchDeleteClusterNodesResponse": {
        "members": [
            {"name": "Failed", "shape": "BatchDeleteClusterNodesErrorList", "type": "list"},
            {"name": "Successful", "shape": "ClusterNodeIds", "type": "list"},
        ],
        "type": "structure",
    },
    "BatchDescribeModelPackageError": {
        "members": [
            {"name": "ErrorCode", "shape": "String", "type": "string"},
            {"name": "ErrorResponse", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "BatchDescribeModelPackageErrorMap": {
        "key_shape": "ModelPackageArn",
        "key_type": "string",
        "type": "map",
        "value_shape": "BatchDescribeModelPackageError",
        "value_type": "structure",
    },
    "BatchDescribeModelPackageInput": {
        "members": [
            {"name": "ModelPackageArnList", "shape": "ModelPackageArnList", "type": "list"}
        ],
        "type": "structure",
    },
    "BatchDescribeModelPackageOutput": {
        "members": [
            {"name": "ModelPackageSummaries", "shape": "ModelPackageSummaries", "type": "map"},
            {
                "name": "BatchDescribeModelPackageErrorMap",
                "shape": "BatchDescribeModelPackageErrorMap",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "BatchDescribeModelPackageSummary": {
        "members": [
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageVersion", "shape": "ModelPackageVersion", "type": "integer"},
            {"name": "ModelPackageArn", "shape": "ModelPackageArn", "type": "string"},
            {"name": "ModelPackageDescription", "shape": "EntityDescription", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {
                "name": "InferenceSpecification",
                "shape": "InferenceSpecification",
                "type": "structure",
            },
            {"name": "ModelPackageStatus", "shape": "ModelPackageStatus", "type": "string"},
            {"name": "ModelApprovalStatus", "shape": "ModelApprovalStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "BatchGetMetricsRequest": {
        "members": [{"name": "MetricQueries", "shape": "MetricQueryList", "type": "list"}],
        "type": "structure",
    },
    "BatchGetMetricsResponse": {
        "members": [
            {"name": "MetricQueryResults", "shape": "MetricQueryResultList", "type": "list"}
        ],
        "type": "structure",
    },
    "BatchGetRecordError": {
        "members": [
            {"name": "FeatureGroupName", "shape": "ValueAsString", "type": "string"},
            {"name": "RecordIdentifierValueAsString", "shape": "ValueAsString", "type": "string"},
            {"name": "ErrorCode", "shape": "ValueAsString", "type": "string"},
            {"name": "ErrorMessage", "shape": "Message", "type": "string"},
        ],
        "type": "structure",
    },
    "BatchGetRecordErrors": {
        "member_shape": "BatchGetRecordError",
        "member_type": "structure",
        "type": "list",
    },
    "BatchGetRecordIdentifier": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupNameOrArn", "type": "string"},
            {
                "name": "RecordIdentifiersValueAsString",
                "shape": "RecordIdentifiers",
                "type": "list",
            },
            {"name": "FeatureNames", "shape": "FeatureNames", "type": "list"},
        ],
        "type": "structure",
    },
    "BatchGetRecordIdentifiers": {
        "member_shape": "BatchGetRecordIdentifier",
        "member_type": "structure",
        "type": "list",
    },
    "BatchGetRecordRequest": {
        "members": [
            {"name": "Identifiers", "shape": "BatchGetRecordIdentifiers", "type": "list"},
            {"name": "ExpirationTimeResponse", "shape": "ExpirationTimeResponse", "type": "string"},
        ],
        "type": "structure",
    },
    "BatchGetRecordResponse": {
        "members": [
            {"name": "Records", "shape": "BatchGetRecordResultDetails", "type": "list"},
            {"name": "Errors", "shape": "BatchGetRecordErrors", "type": "list"},
            {"name": "UnprocessedIdentifiers", "shape": "UnprocessedIdentifiers", "type": "list"},
        ],
        "type": "structure",
    },
    "BatchGetRecordResultDetail": {
        "members": [
            {"name": "FeatureGroupName", "shape": "ValueAsString", "type": "string"},
            {"name": "RecordIdentifierValueAsString", "shape": "ValueAsString", "type": "string"},
            {"name": "Record", "shape": "Record", "type": "list"},
            {"name": "ExpiresAt", "shape": "ExpiresAt", "type": "string"},
        ],
        "type": "structure",
    },
    "BatchGetRecordResultDetails": {
        "member_shape": "BatchGetRecordResultDetail",
        "member_type": "structure",
        "type": "list",
    },
    "BatchPutMetricsError": {
        "members": [
            {"name": "Code", "shape": "PutMetricsErrorCode", "type": "string"},
            {"name": "MetricIndex", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "BatchPutMetricsErrorList": {
        "member_shape": "BatchPutMetricsError",
        "member_type": "structure",
        "type": "list",
    },
    "BatchPutMetricsRequest": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "MetricData", "shape": "RawMetricDataList", "type": "list"},
        ],
        "type": "structure",
    },
    "BatchPutMetricsResponse": {
        "members": [{"name": "Errors", "shape": "BatchPutMetricsErrorList", "type": "list"}],
        "type": "structure",
    },
    "BatchTransformInput": {
        "members": [
            {"name": "DataCapturedDestinationS3Uri", "shape": "DestinationS3Uri", "type": "string"},
            {"name": "DatasetFormat", "shape": "MonitoringDatasetFormat", "type": "structure"},
            {"name": "LocalPath", "shape": "ProcessingLocalPath", "type": "string"},
            {"name": "S3InputMode", "shape": "ProcessingS3InputMode", "type": "string"},
            {
                "name": "S3DataDistributionType",
                "shape": "ProcessingS3DataDistributionType",
                "type": "string",
            },
            {"name": "FeaturesAttribute", "shape": "String", "type": "string"},
            {"name": "InferenceAttribute", "shape": "String", "type": "string"},
            {"name": "ProbabilityAttribute", "shape": "String", "type": "string"},
            {
                "name": "ProbabilityThresholdAttribute",
                "shape": "ProbabilityThresholdAttribute",
                "type": "double",
            },
            {"name": "StartTimeOffset", "shape": "MonitoringTimeOffsetString", "type": "string"},
            {"name": "EndTimeOffset", "shape": "MonitoringTimeOffsetString", "type": "string"},
            {
                "name": "ExcludeFeaturesAttribute",
                "shape": "ExcludeFeaturesAttribute",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "BestObjectiveNotImproving": {
        "members": [
            {
                "name": "MaxNumberOfTrainingJobsNotImproving",
                "shape": "MaxNumberOfTrainingJobsNotImproving",
                "type": "integer",
            }
        ],
        "type": "structure",
    },
    "Bias": {
        "members": [
            {"name": "Report", "shape": "MetricsSource", "type": "structure"},
            {"name": "PreTrainingReport", "shape": "MetricsSource", "type": "structure"},
            {"name": "PostTrainingReport", "shape": "MetricsSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "BlueGreenUpdatePolicy": {
        "members": [
            {
                "name": "TrafficRoutingConfiguration",
                "shape": "TrafficRoutingConfig",
                "type": "structure",
            },
            {
                "name": "TerminationWaitInSeconds",
                "shape": "TerminationWaitInSeconds",
                "type": "integer",
            },
            {
                "name": "MaximumExecutionTimeoutInSeconds",
                "shape": "MaximumExecutionTimeoutInSeconds",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "CacheHitResult": {
        "members": [
            {
                "name": "SourcePipelineExecutionArn",
                "shape": "PipelineExecutionArn",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "CallbackStepMetadata": {
        "members": [
            {"name": "CallbackToken", "shape": "CallbackToken", "type": "string"},
            {"name": "SqsQueueUrl", "shape": "String256", "type": "string"},
            {"name": "OutputParameters", "shape": "OutputParameterList", "type": "list"},
        ],
        "type": "structure",
    },
    "CandidateArtifactLocations": {
        "members": [
            {"name": "Explainability", "shape": "ExplainabilityLocation", "type": "string"},
            {"name": "ModelInsights", "shape": "ModelInsightsLocation", "type": "string"},
            {"name": "BacktestResults", "shape": "BacktestResultsLocation", "type": "string"},
        ],
        "type": "structure",
    },
    "CandidateGenerationConfig": {
        "members": [
            {"name": "AlgorithmsConfig", "shape": "AutoMLAlgorithmsConfig", "type": "list"}
        ],
        "type": "structure",
    },
    "CandidateProperties": {
        "members": [
            {
                "name": "CandidateArtifactLocations",
                "shape": "CandidateArtifactLocations",
                "type": "structure",
            },
            {"name": "CandidateMetrics", "shape": "MetricDataList", "type": "list"},
        ],
        "type": "structure",
    },
    "CandidateSteps": {
        "member_shape": "AutoMLCandidateStep",
        "member_type": "structure",
        "type": "list",
    },
    "CanvasAppSettings": {
        "members": [
            {
                "name": "TimeSeriesForecastingSettings",
                "shape": "TimeSeriesForecastingSettings",
                "type": "structure",
            },
            {
                "name": "ModelRegisterSettings",
                "shape": "ModelRegisterSettings",
                "type": "structure",
            },
            {"name": "WorkspaceSettings", "shape": "WorkspaceSettings", "type": "structure"},
            {
                "name": "IdentityProviderOAuthSettings",
                "shape": "IdentityProviderOAuthSettings",
                "type": "list",
            },
            {"name": "DirectDeploySettings", "shape": "DirectDeploySettings", "type": "structure"},
            {"name": "KendraSettings", "shape": "KendraSettings", "type": "structure"},
            {"name": "GenerativeAiSettings", "shape": "GenerativeAiSettings", "type": "structure"},
            {
                "name": "EmrServerlessSettings",
                "shape": "EmrServerlessSettings",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "CapacitySize": {
        "members": [
            {"name": "Type", "shape": "CapacitySizeType", "type": "string"},
            {"name": "Value", "shape": "CapacitySizeValue", "type": "integer"},
        ],
        "type": "structure",
    },
    "CaptureContentTypeHeader": {
        "members": [
            {"name": "CsvContentTypes", "shape": "CsvContentTypes", "type": "list"},
            {"name": "JsonContentTypes", "shape": "JsonContentTypes", "type": "list"},
        ],
        "type": "structure",
    },
    "CaptureOption": {
        "members": [{"name": "CaptureMode", "shape": "CaptureMode", "type": "string"}],
        "type": "structure",
    },
    "CaptureOptionList": {
        "member_shape": "CaptureOption",
        "member_type": "structure",
        "type": "list",
    },
    "CategoricalParameter": {
        "members": [
            {"name": "Name", "shape": "String64", "type": "string"},
            {"name": "Value", "shape": "CategoricalParameterRangeValues", "type": "list"},
        ],
        "type": "structure",
    },
    "CategoricalParameterRange": {
        "members": [
            {"name": "Name", "shape": "ParameterKey", "type": "string"},
            {"name": "Values", "shape": "ParameterValues", "type": "list"},
        ],
        "type": "structure",
    },
    "CategoricalParameterRangeSpecification": {
        "members": [{"name": "Values", "shape": "ParameterValues", "type": "list"}],
        "type": "structure",
    },
    "CategoricalParameterRangeValues": {
        "member_shape": "String128",
        "member_type": "string",
        "type": "list",
    },
    "CategoricalParameterRanges": {
        "member_shape": "CategoricalParameterRange",
        "member_type": "structure",
        "type": "list",
    },
    "CategoricalParameters": {
        "member_shape": "CategoricalParameter",
        "member_type": "structure",
        "type": "list",
    },
    "Channel": {
        "members": [
            {"name": "ChannelName", "shape": "ChannelName", "type": "string"},
            {"name": "DataSource", "shape": "DataSource", "type": "structure"},
            {"name": "ContentType", "shape": "ContentType", "type": "string"},
            {"name": "CompressionType", "shape": "CompressionType", "type": "string"},
            {"name": "RecordWrapperType", "shape": "RecordWrapper", "type": "string"},
            {"name": "InputMode", "shape": "TrainingInputMode", "type": "string"},
            {"name": "ShuffleConfig", "shape": "ShuffleConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "ChannelSpecification": {
        "members": [
            {"name": "Name", "shape": "ChannelName", "type": "string"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
            {"name": "IsRequired", "shape": "Boolean", "type": "boolean"},
            {"name": "SupportedContentTypes", "shape": "ContentTypes", "type": "list"},
            {"name": "SupportedCompressionTypes", "shape": "CompressionTypes", "type": "list"},
            {"name": "SupportedInputModes", "shape": "InputModes", "type": "list"},
        ],
        "type": "structure",
    },
    "ChannelSpecifications": {
        "member_shape": "ChannelSpecification",
        "member_type": "structure",
        "type": "list",
    },
    "CheckpointConfig": {
        "members": [
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "LocalPath", "shape": "DirectoryPath", "type": "string"},
        ],
        "type": "structure",
    },
    "Cidrs": {"member_shape": "Cidr", "member_type": "string", "type": "list"},
    "ClarifyCheckStepMetadata": {
        "members": [
            {"name": "CheckType", "shape": "String256", "type": "string"},
            {
                "name": "BaselineUsedForDriftCheckConstraints",
                "shape": "String1024",
                "type": "string",
            },
            {"name": "CalculatedBaselineConstraints", "shape": "String1024", "type": "string"},
            {"name": "ModelPackageGroupName", "shape": "String256", "type": "string"},
            {"name": "ViolationReport", "shape": "String1024", "type": "string"},
            {"name": "CheckJobArn", "shape": "String256", "type": "string"},
            {"name": "SkipCheck", "shape": "Boolean", "type": "boolean"},
            {"name": "RegisterNewBaseline", "shape": "Boolean", "type": "boolean"},
        ],
        "type": "structure",
    },
    "ClarifyExplainerConfig": {
        "members": [
            {"name": "EnableExplanations", "shape": "ClarifyEnableExplanations", "type": "string"},
            {"name": "InferenceConfig", "shape": "ClarifyInferenceConfig", "type": "structure"},
            {"name": "ShapConfig", "shape": "ClarifyShapConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "ClarifyFeatureHeaders": {
        "member_shape": "ClarifyHeader",
        "member_type": "string",
        "type": "list",
    },
    "ClarifyFeatureTypes": {
        "member_shape": "ClarifyFeatureType",
        "member_type": "string",
        "type": "list",
    },
    "ClarifyInferenceConfig": {
        "members": [
            {"name": "FeaturesAttribute", "shape": "ClarifyFeaturesAttribute", "type": "string"},
            {"name": "ContentTemplate", "shape": "ClarifyContentTemplate", "type": "string"},
            {"name": "MaxRecordCount", "shape": "ClarifyMaxRecordCount", "type": "integer"},
            {"name": "MaxPayloadInMB", "shape": "ClarifyMaxPayloadInMB", "type": "integer"},
            {"name": "ProbabilityIndex", "shape": "ClarifyProbabilityIndex", "type": "integer"},
            {"name": "LabelIndex", "shape": "ClarifyLabelIndex", "type": "integer"},
            {
                "name": "ProbabilityAttribute",
                "shape": "ClarifyProbabilityAttribute",
                "type": "string",
            },
            {"name": "LabelAttribute", "shape": "ClarifyLabelAttribute", "type": "string"},
            {"name": "LabelHeaders", "shape": "ClarifyLabelHeaders", "type": "list"},
            {"name": "FeatureHeaders", "shape": "ClarifyFeatureHeaders", "type": "list"},
            {"name": "FeatureTypes", "shape": "ClarifyFeatureTypes", "type": "list"},
        ],
        "type": "structure",
    },
    "ClarifyLabelHeaders": {
        "member_shape": "ClarifyHeader",
        "member_type": "string",
        "type": "list",
    },
    "ClarifyShapBaselineConfig": {
        "members": [
            {"name": "MimeType", "shape": "ClarifyMimeType", "type": "string"},
            {"name": "ShapBaseline", "shape": "ClarifyShapBaseline", "type": "string"},
            {"name": "ShapBaselineUri", "shape": "Url", "type": "string"},
        ],
        "type": "structure",
    },
    "ClarifyShapConfig": {
        "members": [
            {
                "name": "ShapBaselineConfig",
                "shape": "ClarifyShapBaselineConfig",
                "type": "structure",
            },
            {"name": "NumberOfSamples", "shape": "ClarifyShapNumberOfSamples", "type": "integer"},
            {"name": "UseLogit", "shape": "ClarifyShapUseLogit", "type": "boolean"},
            {"name": "Seed", "shape": "ClarifyShapSeed", "type": "integer"},
            {"name": "TextConfig", "shape": "ClarifyTextConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "ClarifyTextConfig": {
        "members": [
            {"name": "Language", "shape": "ClarifyTextLanguage", "type": "string"},
            {"name": "Granularity", "shape": "ClarifyTextGranularity", "type": "string"},
        ],
        "type": "structure",
    },
    "ClusterEbsVolumeConfig": {
        "members": [
            {"name": "VolumeSizeInGB", "shape": "ClusterEbsVolumeSizeInGB", "type": "integer"}
        ],
        "type": "structure",
    },
    "ClusterInstanceGroupDetails": {
        "members": [
            {"name": "CurrentCount", "shape": "ClusterNonNegativeInstanceCount", "type": "integer"},
            {"name": "TargetCount", "shape": "ClusterInstanceCount", "type": "integer"},
            {"name": "InstanceGroupName", "shape": "ClusterInstanceGroupName", "type": "string"},
            {"name": "InstanceType", "shape": "ClusterInstanceType", "type": "string"},
            {"name": "LifeCycleConfig", "shape": "ClusterLifeCycleConfig", "type": "structure"},
            {"name": "ExecutionRole", "shape": "RoleArn", "type": "string"},
            {"name": "ThreadsPerCore", "shape": "ClusterThreadsPerCore", "type": "integer"},
            {
                "name": "InstanceStorageConfigs",
                "shape": "ClusterInstanceStorageConfigs",
                "type": "list",
            },
            {"name": "OnStartDeepHealthChecks", "shape": "OnStartDeepHealthChecks", "type": "list"},
            {"name": "Status", "shape": "InstanceGroupStatus", "type": "string"},
            {"name": "TrainingPlanArn", "shape": "TrainingPlanArn", "type": "string"},
            {
                "name": "TrainingPlanStatus",
                "shape": "InstanceGroupTrainingPlanStatus",
                "type": "string",
            },
            {"name": "OverrideVpcConfig", "shape": "VpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "ClusterInstanceGroupDetailsList": {
        "member_shape": "ClusterInstanceGroupDetails",
        "member_type": "structure",
        "type": "list",
    },
    "ClusterInstanceGroupSpecification": {
        "members": [
            {"name": "InstanceCount", "shape": "ClusterInstanceCount", "type": "integer"},
            {"name": "InstanceGroupName", "shape": "ClusterInstanceGroupName", "type": "string"},
            {"name": "InstanceType", "shape": "ClusterInstanceType", "type": "string"},
            {"name": "LifeCycleConfig", "shape": "ClusterLifeCycleConfig", "type": "structure"},
            {"name": "ExecutionRole", "shape": "RoleArn", "type": "string"},
            {"name": "ThreadsPerCore", "shape": "ClusterThreadsPerCore", "type": "integer"},
            {
                "name": "InstanceStorageConfigs",
                "shape": "ClusterInstanceStorageConfigs",
                "type": "list",
            },
            {"name": "OnStartDeepHealthChecks", "shape": "OnStartDeepHealthChecks", "type": "list"},
            {"name": "TrainingPlanArn", "shape": "TrainingPlanArn", "type": "string"},
            {"name": "OverrideVpcConfig", "shape": "VpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "ClusterInstanceGroupSpecifications": {
        "member_shape": "ClusterInstanceGroupSpecification",
        "member_type": "structure",
        "type": "list",
    },
    "ClusterInstanceGroupsToDelete": {
        "member_shape": "ClusterInstanceGroupName",
        "member_type": "string",
        "type": "list",
    },
    "ClusterInstancePlacement": {
        "members": [
            {"name": "AvailabilityZone", "shape": "ClusterAvailabilityZone", "type": "string"},
            {"name": "AvailabilityZoneId", "shape": "ClusterAvailabilityZoneId", "type": "string"},
        ],
        "type": "structure",
    },
    "ClusterInstanceStatusDetails": {
        "members": [
            {"name": "Status", "shape": "ClusterInstanceStatus", "type": "string"},
            {"name": "Message", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "ClusterInstanceStorageConfig": {
        "members": [
            {"name": "EbsVolumeConfig", "shape": "ClusterEbsVolumeConfig", "type": "structure"}
        ],
        "type": "structure",
    },
    "ClusterInstanceStorageConfigs": {
        "member_shape": "ClusterInstanceStorageConfig",
        "member_type": "structure",
        "type": "list",
    },
    "ClusterLifeCycleConfig": {
        "members": [
            {"name": "SourceS3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "OnCreate", "shape": "ClusterLifeCycleConfigFileName", "type": "string"},
        ],
        "type": "structure",
    },
    "ClusterNodeDetails": {
        "members": [
            {"name": "InstanceGroupName", "shape": "ClusterInstanceGroupName", "type": "string"},
            {"name": "InstanceId", "shape": "String", "type": "string"},
            {
                "name": "InstanceStatus",
                "shape": "ClusterInstanceStatusDetails",
                "type": "structure",
            },
            {"name": "InstanceType", "shape": "ClusterInstanceType", "type": "string"},
            {"name": "LaunchTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LifeCycleConfig", "shape": "ClusterLifeCycleConfig", "type": "structure"},
            {"name": "OverrideVpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "ThreadsPerCore", "shape": "ClusterThreadsPerCore", "type": "integer"},
            {
                "name": "InstanceStorageConfigs",
                "shape": "ClusterInstanceStorageConfigs",
                "type": "list",
            },
            {"name": "PrivatePrimaryIp", "shape": "ClusterPrivatePrimaryIp", "type": "string"},
            {"name": "PrivatePrimaryIpv6", "shape": "ClusterPrivatePrimaryIpv6", "type": "string"},
            {"name": "PrivateDnsHostname", "shape": "ClusterPrivateDnsHostname", "type": "string"},
            {"name": "Placement", "shape": "ClusterInstancePlacement", "type": "structure"},
        ],
        "type": "structure",
    },
    "ClusterNodeIds": {"member_shape": "ClusterNodeId", "member_type": "string", "type": "list"},
    "ClusterNodeSummaries": {
        "member_shape": "ClusterNodeSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ClusterNodeSummary": {
        "members": [
            {"name": "InstanceGroupName", "shape": "ClusterInstanceGroupName", "type": "string"},
            {"name": "InstanceId", "shape": "String", "type": "string"},
            {"name": "InstanceType", "shape": "ClusterInstanceType", "type": "string"},
            {"name": "LaunchTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "InstanceStatus",
                "shape": "ClusterInstanceStatusDetails",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ClusterOrchestrator": {
        "members": [{"name": "Eks", "shape": "ClusterOrchestratorEksConfig", "type": "structure"}],
        "type": "structure",
    },
    "ClusterOrchestratorEksConfig": {
        "members": [{"name": "ClusterArn", "shape": "EksClusterArn", "type": "string"}],
        "type": "structure",
    },
    "ClusterSchedulerConfigSummary": {
        "members": [
            {
                "name": "ClusterSchedulerConfigArn",
                "shape": "ClusterSchedulerConfigArn",
                "type": "string",
            },
            {
                "name": "ClusterSchedulerConfigId",
                "shape": "ClusterSchedulerConfigId",
                "type": "string",
            },
            {"name": "ClusterSchedulerConfigVersion", "shape": "Integer", "type": "integer"},
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Status", "shape": "SchedulerResourceStatus", "type": "string"},
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
        ],
        "type": "structure",
    },
    "ClusterSchedulerConfigSummaryList": {
        "member_shape": "ClusterSchedulerConfigSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ClusterSummaries": {
        "member_shape": "ClusterSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ClusterSummary": {
        "members": [
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "ClusterName", "shape": "ClusterName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ClusterStatus", "shape": "ClusterStatus", "type": "string"},
            {"name": "TrainingPlanArns", "shape": "TrainingPlanArns", "type": "list"},
        ],
        "type": "structure",
    },
    "CodeEditorAppImageConfig": {
        "members": [
            {"name": "FileSystemConfig", "shape": "FileSystemConfig", "type": "structure"},
            {"name": "ContainerConfig", "shape": "ContainerConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "CodeEditorAppSettings": {
        "members": [
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {"name": "CustomImages", "shape": "CustomImages", "type": "list"},
            {"name": "LifecycleConfigArns", "shape": "LifecycleConfigArns", "type": "list"},
            {
                "name": "AppLifecycleManagement",
                "shape": "AppLifecycleManagement",
                "type": "structure",
            },
            {
                "name": "BuiltInLifecycleConfigArn",
                "shape": "StudioLifecycleConfigArn",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "CodeRepositories": {
        "member_shape": "CodeRepository",
        "member_type": "structure",
        "type": "list",
    },
    "CodeRepository": {
        "members": [{"name": "RepositoryUrl", "shape": "RepositoryUrl", "type": "string"}],
        "type": "structure",
    },
    "CodeRepositorySummary": {
        "members": [
            {"name": "CodeRepositoryName", "shape": "EntityName", "type": "string"},
            {"name": "CodeRepositoryArn", "shape": "CodeRepositoryArn", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "GitConfig", "shape": "GitConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "CodeRepositorySummaryList": {
        "member_shape": "CodeRepositorySummary",
        "member_type": "structure",
        "type": "list",
    },
    "CognitoConfig": {
        "members": [
            {"name": "UserPool", "shape": "CognitoUserPool", "type": "string"},
            {"name": "ClientId", "shape": "ClientId", "type": "string"},
        ],
        "type": "structure",
    },
    "CognitoMemberDefinition": {
        "members": [
            {"name": "UserPool", "shape": "CognitoUserPool", "type": "string"},
            {"name": "UserGroup", "shape": "CognitoUserGroup", "type": "string"},
            {"name": "ClientId", "shape": "ClientId", "type": "string"},
        ],
        "type": "structure",
    },
    "CollectionConfig": {
        "members": [{"name": "VectorConfig", "shape": "VectorConfig", "type": "structure"}],
        "type": "structure",
    },
    "CollectionConfiguration": {
        "members": [
            {"name": "CollectionName", "shape": "CollectionName", "type": "string"},
            {"name": "CollectionParameters", "shape": "CollectionParameters", "type": "map"},
        ],
        "type": "structure",
    },
    "CollectionConfigurations": {
        "member_shape": "CollectionConfiguration",
        "member_type": "structure",
        "type": "list",
    },
    "CollectionParameters": {
        "key_shape": "ConfigKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "ConfigValue",
        "value_type": "string",
    },
    "CompilationJobSummaries": {
        "member_shape": "CompilationJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "CompilationJobSummary": {
        "members": [
            {"name": "CompilationJobName", "shape": "EntityName", "type": "string"},
            {"name": "CompilationJobArn", "shape": "CompilationJobArn", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CompilationStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CompilationEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CompilationTargetDevice", "shape": "TargetDevice", "type": "string"},
            {"name": "CompilationTargetPlatformOs", "shape": "TargetPlatformOs", "type": "string"},
            {
                "name": "CompilationTargetPlatformArch",
                "shape": "TargetPlatformArch",
                "type": "string",
            },
            {
                "name": "CompilationTargetPlatformAccelerator",
                "shape": "TargetPlatformAccelerator",
                "type": "string",
            },
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "CompilationJobStatus", "shape": "CompilationJobStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "CompressionTypes": {
        "member_shape": "CompressionType",
        "member_type": "string",
        "type": "list",
    },
    "ComputeQuotaConfig": {
        "members": [
            {
                "name": "ComputeQuotaResources",
                "shape": "ComputeQuotaResourceConfigList",
                "type": "list",
            },
            {
                "name": "ResourceSharingConfig",
                "shape": "ResourceSharingConfig",
                "type": "structure",
            },
            {"name": "PreemptTeamTasks", "shape": "PreemptTeamTasks", "type": "string"},
        ],
        "type": "structure",
    },
    "ComputeQuotaResourceConfig": {
        "members": [
            {"name": "InstanceType", "shape": "ClusterInstanceType", "type": "string"},
            {"name": "Count", "shape": "InstanceCount", "type": "integer"},
        ],
        "type": "structure",
    },
    "ComputeQuotaResourceConfigList": {
        "member_shape": "ComputeQuotaResourceConfig",
        "member_type": "structure",
        "type": "list",
    },
    "ComputeQuotaSummary": {
        "members": [
            {"name": "ComputeQuotaArn", "shape": "ComputeQuotaArn", "type": "string"},
            {"name": "ComputeQuotaId", "shape": "ComputeQuotaId", "type": "string"},
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "ComputeQuotaVersion", "shape": "Integer", "type": "integer"},
            {"name": "Status", "shape": "SchedulerResourceStatus", "type": "string"},
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "ComputeQuotaConfig", "shape": "ComputeQuotaConfig", "type": "structure"},
            {"name": "ComputeQuotaTarget", "shape": "ComputeQuotaTarget", "type": "structure"},
            {"name": "ActivationState", "shape": "ActivationState", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ComputeQuotaSummaryList": {
        "member_shape": "ComputeQuotaSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ComputeQuotaTarget": {
        "members": [
            {"name": "TeamName", "shape": "ComputeQuotaTargetTeamName", "type": "string"},
            {"name": "FairShareWeight", "shape": "FairShareWeight", "type": "integer"},
        ],
        "type": "structure",
    },
    "ConditionStepMetadata": {
        "members": [{"name": "Outcome", "shape": "ConditionOutcome", "type": "string"}],
        "type": "structure",
    },
    "ConflictException": {
        "members": [{"name": "Message", "shape": "FailureReason", "type": "string"}],
        "type": "structure",
    },
    "ContainerArguments": {
        "member_shape": "ContainerArgument",
        "member_type": "string",
        "type": "list",
    },
    "ContainerConfig": {
        "members": [
            {
                "name": "ContainerArguments",
                "shape": "CustomImageContainerArguments",
                "type": "list",
            },
            {
                "name": "ContainerEntrypoint",
                "shape": "CustomImageContainerEntrypoint",
                "type": "list",
            },
            {
                "name": "ContainerEnvironmentVariables",
                "shape": "CustomImageContainerEnvironmentVariables",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "ContainerDefinition": {
        "members": [
            {"name": "ContainerHostname", "shape": "ContainerHostname", "type": "string"},
            {"name": "Image", "shape": "ContainerImage", "type": "string"},
            {"name": "ImageConfig", "shape": "ImageConfig", "type": "structure"},
            {"name": "Mode", "shape": "ContainerMode", "type": "string"},
            {"name": "ModelDataUrl", "shape": "Url", "type": "string"},
            {"name": "ModelDataSource", "shape": "ModelDataSource", "type": "structure"},
            {
                "name": "AdditionalModelDataSources",
                "shape": "AdditionalModelDataSources",
                "type": "list",
            },
            {"name": "Environment", "shape": "EnvironmentMap", "type": "map"},
            {"name": "ModelPackageName", "shape": "VersionedArnOrName", "type": "string"},
            {
                "name": "InferenceSpecificationName",
                "shape": "InferenceSpecificationName",
                "type": "string",
            },
            {"name": "MultiModelConfig", "shape": "MultiModelConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "ContainerDefinitionList": {
        "member_shape": "ContainerDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "ContainerEntrypoint": {
        "member_shape": "ContainerEntrypointString",
        "member_type": "string",
        "type": "list",
    },
    "ContentClassifiers": {
        "member_shape": "ContentClassifier",
        "member_type": "string",
        "type": "list",
    },
    "ContentTypes": {"member_shape": "ContentType", "member_type": "string", "type": "list"},
    "ContextSource": {
        "members": [
            {"name": "SourceUri", "shape": "SourceUri", "type": "string"},
            {"name": "SourceType", "shape": "String256", "type": "string"},
            {"name": "SourceId", "shape": "String256", "type": "string"},
        ],
        "type": "structure",
    },
    "ContextSummaries": {
        "member_shape": "ContextSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ContextSummary": {
        "members": [
            {"name": "ContextArn", "shape": "ContextArn", "type": "string"},
            {"name": "ContextName", "shape": "ContextName", "type": "string"},
            {"name": "Source", "shape": "ContextSource", "type": "structure"},
            {"name": "ContextType", "shape": "String256", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ContinuousParameterRange": {
        "members": [
            {"name": "Name", "shape": "ParameterKey", "type": "string"},
            {"name": "MinValue", "shape": "ParameterValue", "type": "string"},
            {"name": "MaxValue", "shape": "ParameterValue", "type": "string"},
            {"name": "ScalingType", "shape": "HyperParameterScalingType", "type": "string"},
        ],
        "type": "structure",
    },
    "ContinuousParameterRangeSpecification": {
        "members": [
            {"name": "MinValue", "shape": "ParameterValue", "type": "string"},
            {"name": "MaxValue", "shape": "ParameterValue", "type": "string"},
        ],
        "type": "structure",
    },
    "ContinuousParameterRanges": {
        "member_shape": "ContinuousParameterRange",
        "member_type": "structure",
        "type": "list",
    },
    "ConvergenceDetected": {
        "members": [
            {"name": "CompleteOnConvergence", "shape": "CompleteOnConvergence", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateActionRequest": {
        "members": [
            {"name": "ActionName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "ActionSource", "type": "structure"},
            {"name": "ActionType", "shape": "String256", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "Status", "shape": "ActionStatus", "type": "string"},
            {"name": "Properties", "shape": "LineageEntityParameters", "type": "map"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateActionResponse": {
        "members": [{"name": "ActionArn", "shape": "ActionArn", "type": "string"}],
        "type": "structure",
    },
    "CreateAlgorithmInput": {
        "members": [
            {"name": "AlgorithmName", "shape": "EntityName", "type": "string"},
            {"name": "AlgorithmDescription", "shape": "EntityDescription", "type": "string"},
            {
                "name": "TrainingSpecification",
                "shape": "TrainingSpecification",
                "type": "structure",
            },
            {
                "name": "InferenceSpecification",
                "shape": "InferenceSpecification",
                "type": "structure",
            },
            {
                "name": "ValidationSpecification",
                "shape": "AlgorithmValidationSpecification",
                "type": "structure",
            },
            {"name": "CertifyForMarketplace", "shape": "CertifyForMarketplace", "type": "boolean"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateAlgorithmOutput": {
        "members": [{"name": "AlgorithmArn", "shape": "AlgorithmArn", "type": "string"}],
        "type": "structure",
    },
    "CreateAppImageConfigRequest": {
        "members": [
            {"name": "AppImageConfigName", "shape": "AppImageConfigName", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {
                "name": "KernelGatewayImageConfig",
                "shape": "KernelGatewayImageConfig",
                "type": "structure",
            },
            {
                "name": "JupyterLabAppImageConfig",
                "shape": "JupyterLabAppImageConfig",
                "type": "structure",
            },
            {
                "name": "CodeEditorAppImageConfig",
                "shape": "CodeEditorAppImageConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "CreateAppImageConfigResponse": {
        "members": [{"name": "AppImageConfigArn", "shape": "AppImageConfigArn", "type": "string"}],
        "type": "structure",
    },
    "CreateAppRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "AppType", "shape": "AppType", "type": "string"},
            {"name": "AppName", "shape": "AppName", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "ResourceSpec", "shape": "ResourceSpec", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateAppResponse": {
        "members": [{"name": "AppArn", "shape": "AppArn", "type": "string"}],
        "type": "structure",
    },
    "CreateArtifactRequest": {
        "members": [
            {"name": "ArtifactName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "ArtifactSource", "type": "structure"},
            {"name": "ArtifactType", "shape": "String256", "type": "string"},
            {"name": "Properties", "shape": "ArtifactProperties", "type": "map"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateArtifactResponse": {
        "members": [{"name": "ArtifactArn", "shape": "ArtifactArn", "type": "string"}],
        "type": "structure",
    },
    "CreateAutoMLJobRequest": {
        "members": [
            {"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"},
            {"name": "InputDataConfig", "shape": "AutoMLInputDataConfig", "type": "list"},
            {"name": "OutputDataConfig", "shape": "AutoMLOutputDataConfig", "type": "structure"},
            {"name": "ProblemType", "shape": "ProblemType", "type": "string"},
            {"name": "AutoMLJobObjective", "shape": "AutoMLJobObjective", "type": "structure"},
            {"name": "AutoMLJobConfig", "shape": "AutoMLJobConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "GenerateCandidateDefinitionsOnly",
                "shape": "GenerateCandidateDefinitionsOnly",
                "type": "boolean",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "ModelDeployConfig", "shape": "ModelDeployConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateAutoMLJobResponse": {
        "members": [{"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"}],
        "type": "structure",
    },
    "CreateAutoMLJobV2Request": {
        "members": [
            {"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"},
            {
                "name": "AutoMLJobInputDataConfig",
                "shape": "AutoMLJobInputDataConfig",
                "type": "list",
            },
            {"name": "OutputDataConfig", "shape": "AutoMLOutputDataConfig", "type": "structure"},
            {
                "name": "AutoMLProblemTypeConfig",
                "shape": "AutoMLProblemTypeConfig",
                "type": "structure",
            },
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "SecurityConfig", "shape": "AutoMLSecurityConfig", "type": "structure"},
            {"name": "AutoMLJobObjective", "shape": "AutoMLJobObjective", "type": "structure"},
            {"name": "ModelDeployConfig", "shape": "ModelDeployConfig", "type": "structure"},
            {"name": "DataSplitConfig", "shape": "AutoMLDataSplitConfig", "type": "structure"},
            {"name": "AutoMLComputeConfig", "shape": "AutoMLComputeConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateAutoMLJobV2Response": {
        "members": [{"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"}],
        "type": "structure",
    },
    "CreateClusterRequest": {
        "members": [
            {"name": "ClusterName", "shape": "ClusterName", "type": "string"},
            {
                "name": "InstanceGroups",
                "shape": "ClusterInstanceGroupSpecifications",
                "type": "list",
            },
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "Orchestrator", "shape": "ClusterOrchestrator", "type": "structure"},
            {"name": "NodeRecovery", "shape": "ClusterNodeRecovery", "type": "string"},
        ],
        "type": "structure",
    },
    "CreateClusterResponse": {
        "members": [{"name": "ClusterArn", "shape": "ClusterArn", "type": "string"}],
        "type": "structure",
    },
    "CreateClusterSchedulerConfigRequest": {
        "members": [
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "SchedulerConfig", "shape": "SchedulerConfig", "type": "structure"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateClusterSchedulerConfigResponse": {
        "members": [
            {
                "name": "ClusterSchedulerConfigArn",
                "shape": "ClusterSchedulerConfigArn",
                "type": "string",
            },
            {
                "name": "ClusterSchedulerConfigId",
                "shape": "ClusterSchedulerConfigId",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "CreateCodeRepositoryInput": {
        "members": [
            {"name": "CodeRepositoryName", "shape": "EntityName", "type": "string"},
            {"name": "GitConfig", "shape": "GitConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateCodeRepositoryOutput": {
        "members": [{"name": "CodeRepositoryArn", "shape": "CodeRepositoryArn", "type": "string"}],
        "type": "structure",
    },
    "CreateCompilationJobRequest": {
        "members": [
            {"name": "CompilationJobName", "shape": "EntityName", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "ModelPackageVersionArn", "shape": "ModelPackageArn", "type": "string"},
            {"name": "InputConfig", "shape": "InputConfig", "type": "structure"},
            {"name": "OutputConfig", "shape": "OutputConfig", "type": "structure"},
            {"name": "VpcConfig", "shape": "NeoVpcConfig", "type": "structure"},
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateCompilationJobResponse": {
        "members": [{"name": "CompilationJobArn", "shape": "CompilationJobArn", "type": "string"}],
        "type": "structure",
    },
    "CreateComputeQuotaRequest": {
        "members": [
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "ComputeQuotaConfig", "shape": "ComputeQuotaConfig", "type": "structure"},
            {"name": "ComputeQuotaTarget", "shape": "ComputeQuotaTarget", "type": "structure"},
            {"name": "ActivationState", "shape": "ActivationState", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateComputeQuotaResponse": {
        "members": [
            {"name": "ComputeQuotaArn", "shape": "ComputeQuotaArn", "type": "string"},
            {"name": "ComputeQuotaId", "shape": "ComputeQuotaId", "type": "string"},
        ],
        "type": "structure",
    },
    "CreateContextRequest": {
        "members": [
            {"name": "ContextName", "shape": "ContextName", "type": "string"},
            {"name": "Source", "shape": "ContextSource", "type": "structure"},
            {"name": "ContextType", "shape": "String256", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "Properties", "shape": "LineageEntityParameters", "type": "map"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateContextResponse": {
        "members": [{"name": "ContextArn", "shape": "ContextArn", "type": "string"}],
        "type": "structure",
    },
    "CreateDataQualityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"},
            {
                "name": "DataQualityBaselineConfig",
                "shape": "DataQualityBaselineConfig",
                "type": "structure",
            },
            {
                "name": "DataQualityAppSpecification",
                "shape": "DataQualityAppSpecification",
                "type": "structure",
            },
            {"name": "DataQualityJobInput", "shape": "DataQualityJobInput", "type": "structure"},
            {
                "name": "DataQualityJobOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "JobResources", "shape": "MonitoringResources", "type": "structure"},
            {"name": "NetworkConfig", "shape": "MonitoringNetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateDataQualityJobDefinitionResponse": {
        "members": [
            {"name": "JobDefinitionArn", "shape": "MonitoringJobDefinitionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateDeviceFleetRequest": {
        "members": [
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Description", "shape": "DeviceFleetDescription", "type": "string"},
            {"name": "OutputConfig", "shape": "EdgeOutputConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "EnableIotRoleAlias", "shape": "EnableIotRoleAlias", "type": "boolean"},
        ],
        "type": "structure",
    },
    "CreateDomainRequest": {
        "members": [
            {"name": "DomainName", "shape": "DomainName", "type": "string"},
            {"name": "AuthMode", "shape": "AuthMode", "type": "string"},
            {"name": "DefaultUserSettings", "shape": "UserSettings", "type": "structure"},
            {"name": "DomainSettings", "shape": "DomainSettings", "type": "structure"},
            {"name": "SubnetIds", "shape": "Subnets", "type": "list"},
            {"name": "VpcId", "shape": "VpcId", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "AppNetworkAccessType", "shape": "AppNetworkAccessType", "type": "string"},
            {"name": "HomeEfsFileSystemKmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {
                "name": "AppSecurityGroupManagement",
                "shape": "AppSecurityGroupManagement",
                "type": "string",
            },
            {"name": "TagPropagation", "shape": "TagPropagation", "type": "string"},
            {"name": "DefaultSpaceSettings", "shape": "DefaultSpaceSettings", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateDomainResponse": {
        "members": [
            {"name": "DomainArn", "shape": "DomainArn", "type": "string"},
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "Url", "shape": "String1024", "type": "string"},
        ],
        "type": "structure",
    },
    "CreateEdgeDeploymentPlanRequest": {
        "members": [
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "ModelConfigs", "shape": "EdgeDeploymentModelConfigs", "type": "list"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "Stages", "shape": "DeploymentStages", "type": "list"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateEdgeDeploymentPlanResponse": {
        "members": [
            {"name": "EdgeDeploymentPlanArn", "shape": "EdgeDeploymentPlanArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateEdgeDeploymentStageRequest": {
        "members": [
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "Stages", "shape": "DeploymentStages", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateEdgePackagingJobRequest": {
        "members": [
            {"name": "EdgePackagingJobName", "shape": "EntityName", "type": "string"},
            {"name": "CompilationJobName", "shape": "EntityName", "type": "string"},
            {"name": "ModelName", "shape": "EntityName", "type": "string"},
            {"name": "ModelVersion", "shape": "EdgeVersion", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "OutputConfig", "shape": "EdgeOutputConfig", "type": "structure"},
            {"name": "ResourceKey", "shape": "KmsKeyId", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateEndpointConfigInput": {
        "members": [
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {"name": "ProductionVariants", "shape": "ProductionVariantList", "type": "list"},
            {"name": "DataCaptureConfig", "shape": "DataCaptureConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "AsyncInferenceConfig", "shape": "AsyncInferenceConfig", "type": "structure"},
            {"name": "ExplainerConfig", "shape": "ExplainerConfig", "type": "structure"},
            {"name": "ShadowProductionVariants", "shape": "ProductionVariantList", "type": "list"},
            {"name": "ExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
        ],
        "type": "structure",
    },
    "CreateEndpointConfigOutput": {
        "members": [{"name": "EndpointConfigArn", "shape": "EndpointConfigArn", "type": "string"}],
        "type": "structure",
    },
    "CreateEndpointInput": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {"name": "DeploymentConfig", "shape": "DeploymentConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateEndpointOutput": {
        "members": [{"name": "EndpointArn", "shape": "EndpointArn", "type": "string"}],
        "type": "structure",
    },
    "CreateExperimentRequest": {
        "members": [
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateExperimentResponse": {
        "members": [{"name": "ExperimentArn", "shape": "ExperimentArn", "type": "string"}],
        "type": "structure",
    },
    "CreateFeatureGroupRequest": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupName", "type": "string"},
            {"name": "RecordIdentifierFeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "EventTimeFeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "FeatureDefinitions", "shape": "FeatureDefinitions", "type": "list"},
            {"name": "OnlineStoreConfig", "shape": "OnlineStoreConfig", "type": "structure"},
            {"name": "OfflineStoreConfig", "shape": "OfflineStoreConfig", "type": "structure"},
            {"name": "ThroughputConfig", "shape": "ThroughputConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Description", "shape": "Description", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateFeatureGroupResponse": {
        "members": [{"name": "FeatureGroupArn", "shape": "FeatureGroupArn", "type": "string"}],
        "type": "structure",
    },
    "CreateFlowDefinitionRequest": {
        "members": [
            {"name": "FlowDefinitionName", "shape": "FlowDefinitionName", "type": "string"},
            {
                "name": "HumanLoopRequestSource",
                "shape": "HumanLoopRequestSource",
                "type": "structure",
            },
            {
                "name": "HumanLoopActivationConfig",
                "shape": "HumanLoopActivationConfig",
                "type": "structure",
            },
            {"name": "HumanLoopConfig", "shape": "HumanLoopConfig", "type": "structure"},
            {"name": "OutputConfig", "shape": "FlowDefinitionOutputConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateFlowDefinitionResponse": {
        "members": [{"name": "FlowDefinitionArn", "shape": "FlowDefinitionArn", "type": "string"}],
        "type": "structure",
    },
    "CreateHubContentReferenceRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {
                "name": "SageMakerPublicHubContentArn",
                "shape": "SageMakerPublicHubContentArn",
                "type": "string",
            },
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "MinVersion", "shape": "HubContentVersion", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateHubContentReferenceResponse": {
        "members": [
            {"name": "HubArn", "shape": "HubArn", "type": "string"},
            {"name": "HubContentArn", "shape": "HubContentArn", "type": "string"},
        ],
        "type": "structure",
    },
    "CreateHubRequest": {
        "members": [
            {"name": "HubName", "shape": "HubName", "type": "string"},
            {"name": "HubDescription", "shape": "HubDescription", "type": "string"},
            {"name": "HubDisplayName", "shape": "HubDisplayName", "type": "string"},
            {"name": "HubSearchKeywords", "shape": "HubSearchKeywordList", "type": "list"},
            {"name": "S3StorageConfig", "shape": "HubS3StorageConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateHubResponse": {
        "members": [{"name": "HubArn", "shape": "HubArn", "type": "string"}],
        "type": "structure",
    },
    "CreateHumanTaskUiRequest": {
        "members": [
            {"name": "HumanTaskUiName", "shape": "HumanTaskUiName", "type": "string"},
            {"name": "UiTemplate", "shape": "UiTemplate", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateHumanTaskUiResponse": {
        "members": [{"name": "HumanTaskUiArn", "shape": "HumanTaskUiArn", "type": "string"}],
        "type": "structure",
    },
    "CreateHyperParameterTuningJobRequest": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            },
            {
                "name": "HyperParameterTuningJobConfig",
                "shape": "HyperParameterTuningJobConfig",
                "type": "structure",
            },
            {
                "name": "TrainingJobDefinition",
                "shape": "HyperParameterTrainingJobDefinition",
                "type": "structure",
            },
            {
                "name": "TrainingJobDefinitions",
                "shape": "HyperParameterTrainingJobDefinitions",
                "type": "list",
            },
            {
                "name": "WarmStartConfig",
                "shape": "HyperParameterTuningJobWarmStartConfig",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "Autotune", "shape": "Autotune", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateHyperParameterTuningJobResponse": {
        "members": [
            {
                "name": "HyperParameterTuningJobArn",
                "shape": "HyperParameterTuningJobArn",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "CreateImageRequest": {
        "members": [
            {"name": "Description", "shape": "ImageDescription", "type": "string"},
            {"name": "DisplayName", "shape": "ImageDisplayName", "type": "string"},
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateImageResponse": {
        "members": [{"name": "ImageArn", "shape": "ImageArn", "type": "string"}],
        "type": "structure",
    },
    "CreateImageVersionRequest": {
        "members": [
            {"name": "BaseImage", "shape": "ImageBaseImage", "type": "string"},
            {"name": "ClientToken", "shape": "ClientToken", "type": "string"},
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "Aliases", "shape": "SageMakerImageVersionAliases", "type": "list"},
            {"name": "VendorGuidance", "shape": "VendorGuidance", "type": "string"},
            {"name": "JobType", "shape": "JobType", "type": "string"},
            {"name": "MLFramework", "shape": "MLFramework", "type": "string"},
            {"name": "ProgrammingLang", "shape": "ProgrammingLang", "type": "string"},
            {"name": "Processor", "shape": "Processor", "type": "string"},
            {"name": "Horovod", "shape": "Horovod", "type": "boolean"},
            {"name": "ReleaseNotes", "shape": "ReleaseNotes", "type": "string"},
        ],
        "type": "structure",
    },
    "CreateImageVersionResponse": {
        "members": [{"name": "ImageVersionArn", "shape": "ImageVersionArn", "type": "string"}],
        "type": "structure",
    },
    "CreateInferenceComponentInput": {
        "members": [
            {"name": "InferenceComponentName", "shape": "InferenceComponentName", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "VariantName", "shape": "VariantName", "type": "string"},
            {
                "name": "Specification",
                "shape": "InferenceComponentSpecification",
                "type": "structure",
            },
            {
                "name": "RuntimeConfig",
                "shape": "InferenceComponentRuntimeConfig",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateInferenceComponentOutput": {
        "members": [
            {"name": "InferenceComponentArn", "shape": "InferenceComponentArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateInferenceExperimentRequest": {
        "members": [
            {"name": "Name", "shape": "InferenceExperimentName", "type": "string"},
            {"name": "Type", "shape": "InferenceExperimentType", "type": "string"},
            {"name": "Schedule", "shape": "InferenceExperimentSchedule", "type": "structure"},
            {"name": "Description", "shape": "InferenceExperimentDescription", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "ModelVariants", "shape": "ModelVariantConfigList", "type": "list"},
            {
                "name": "DataStorageConfig",
                "shape": "InferenceExperimentDataStorageConfig",
                "type": "structure",
            },
            {"name": "ShadowModeConfig", "shape": "ShadowModeConfig", "type": "structure"},
            {"name": "KmsKey", "shape": "KmsKeyId", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateInferenceExperimentResponse": {
        "members": [
            {"name": "InferenceExperimentArn", "shape": "InferenceExperimentArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateInferenceRecommendationsJobRequest": {
        "members": [
            {"name": "JobName", "shape": "RecommendationJobName", "type": "string"},
            {"name": "JobType", "shape": "RecommendationJobType", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "InputConfig", "shape": "RecommendationJobInputConfig", "type": "structure"},
            {"name": "JobDescription", "shape": "RecommendationJobDescription", "type": "string"},
            {
                "name": "StoppingConditions",
                "shape": "RecommendationJobStoppingConditions",
                "type": "structure",
            },
            {"name": "OutputConfig", "shape": "RecommendationJobOutputConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateInferenceRecommendationsJobResponse": {
        "members": [{"name": "JobArn", "shape": "RecommendationJobArn", "type": "string"}],
        "type": "structure",
    },
    "CreateLabelingJobRequest": {
        "members": [
            {"name": "LabelingJobName", "shape": "LabelingJobName", "type": "string"},
            {"name": "LabelAttributeName", "shape": "LabelAttributeName", "type": "string"},
            {"name": "InputConfig", "shape": "LabelingJobInputConfig", "type": "structure"},
            {"name": "OutputConfig", "shape": "LabelingJobOutputConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "LabelCategoryConfigS3Uri", "shape": "S3Uri", "type": "string"},
            {
                "name": "StoppingConditions",
                "shape": "LabelingJobStoppingConditions",
                "type": "structure",
            },
            {
                "name": "LabelingJobAlgorithmsConfig",
                "shape": "LabelingJobAlgorithmsConfig",
                "type": "structure",
            },
            {"name": "HumanTaskConfig", "shape": "HumanTaskConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateLabelingJobResponse": {
        "members": [{"name": "LabelingJobArn", "shape": "LabelingJobArn", "type": "string"}],
        "type": "structure",
    },
    "CreateMlflowTrackingServerRequest": {
        "members": [
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"},
            {"name": "ArtifactStoreUri", "shape": "S3Uri", "type": "string"},
            {"name": "TrackingServerSize", "shape": "TrackingServerSize", "type": "string"},
            {"name": "MlflowVersion", "shape": "MlflowVersion", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "AutomaticModelRegistration", "shape": "Boolean", "type": "boolean"},
            {
                "name": "WeeklyMaintenanceWindowStart",
                "shape": "WeeklyMaintenanceWindowStart",
                "type": "string",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateMlflowTrackingServerResponse": {
        "members": [{"name": "TrackingServerArn", "shape": "TrackingServerArn", "type": "string"}],
        "type": "structure",
    },
    "CreateModelBiasJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"},
            {
                "name": "ModelBiasBaselineConfig",
                "shape": "ModelBiasBaselineConfig",
                "type": "structure",
            },
            {
                "name": "ModelBiasAppSpecification",
                "shape": "ModelBiasAppSpecification",
                "type": "structure",
            },
            {"name": "ModelBiasJobInput", "shape": "ModelBiasJobInput", "type": "structure"},
            {
                "name": "ModelBiasJobOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "JobResources", "shape": "MonitoringResources", "type": "structure"},
            {"name": "NetworkConfig", "shape": "MonitoringNetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateModelBiasJobDefinitionResponse": {
        "members": [
            {"name": "JobDefinitionArn", "shape": "MonitoringJobDefinitionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateModelCardExportJobRequest": {
        "members": [
            {"name": "ModelCardName", "shape": "ModelCardNameOrArn", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
            {"name": "ModelCardExportJobName", "shape": "EntityName", "type": "string"},
            {"name": "OutputConfig", "shape": "ModelCardExportOutputConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateModelCardExportJobResponse": {
        "members": [
            {"name": "ModelCardExportJobArn", "shape": "ModelCardExportJobArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateModelCardRequest": {
        "members": [
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "SecurityConfig", "shape": "ModelCardSecurityConfig", "type": "structure"},
            {"name": "Content", "shape": "ModelCardContent", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateModelCardResponse": {
        "members": [{"name": "ModelCardArn", "shape": "ModelCardArn", "type": "string"}],
        "type": "structure",
    },
    "CreateModelExplainabilityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"},
            {
                "name": "ModelExplainabilityBaselineConfig",
                "shape": "ModelExplainabilityBaselineConfig",
                "type": "structure",
            },
            {
                "name": "ModelExplainabilityAppSpecification",
                "shape": "ModelExplainabilityAppSpecification",
                "type": "structure",
            },
            {
                "name": "ModelExplainabilityJobInput",
                "shape": "ModelExplainabilityJobInput",
                "type": "structure",
            },
            {
                "name": "ModelExplainabilityJobOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "JobResources", "shape": "MonitoringResources", "type": "structure"},
            {"name": "NetworkConfig", "shape": "MonitoringNetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateModelExplainabilityJobDefinitionResponse": {
        "members": [
            {"name": "JobDefinitionArn", "shape": "MonitoringJobDefinitionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateModelInput": {
        "members": [
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "PrimaryContainer", "shape": "ContainerDefinition", "type": "structure"},
            {"name": "Containers", "shape": "ContainerDefinitionList", "type": "list"},
            {
                "name": "InferenceExecutionConfig",
                "shape": "InferenceExecutionConfig",
                "type": "structure",
            },
            {"name": "ExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
        ],
        "type": "structure",
    },
    "CreateModelOutput": {
        "members": [{"name": "ModelArn", "shape": "ModelArn", "type": "string"}],
        "type": "structure",
    },
    "CreateModelPackageGroupInput": {
        "members": [
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {
                "name": "ModelPackageGroupDescription",
                "shape": "EntityDescription",
                "type": "string",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateModelPackageGroupOutput": {
        "members": [
            {"name": "ModelPackageGroupArn", "shape": "ModelPackageGroupArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateModelPackageInput": {
        "members": [
            {"name": "ModelPackageName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageGroupName", "shape": "ArnOrName", "type": "string"},
            {"name": "ModelPackageDescription", "shape": "EntityDescription", "type": "string"},
            {
                "name": "InferenceSpecification",
                "shape": "InferenceSpecification",
                "type": "structure",
            },
            {
                "name": "ValidationSpecification",
                "shape": "ModelPackageValidationSpecification",
                "type": "structure",
            },
            {
                "name": "SourceAlgorithmSpecification",
                "shape": "SourceAlgorithmSpecification",
                "type": "structure",
            },
            {"name": "CertifyForMarketplace", "shape": "CertifyForMarketplace", "type": "boolean"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "ModelApprovalStatus", "shape": "ModelApprovalStatus", "type": "string"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "ModelMetrics", "shape": "ModelMetrics", "type": "structure"},
            {"name": "ClientToken", "shape": "ClientToken", "type": "string"},
            {"name": "Domain", "shape": "String", "type": "string"},
            {"name": "Task", "shape": "String", "type": "string"},
            {"name": "SamplePayloadUrl", "shape": "S3Uri", "type": "string"},
            {"name": "CustomerMetadataProperties", "shape": "CustomerMetadataMap", "type": "map"},
            {"name": "DriftCheckBaselines", "shape": "DriftCheckBaselines", "type": "structure"},
            {
                "name": "AdditionalInferenceSpecifications",
                "shape": "AdditionalInferenceSpecifications",
                "type": "list",
            },
            {"name": "SkipModelValidation", "shape": "SkipModelValidation", "type": "string"},
            {"name": "SourceUri", "shape": "ModelPackageSourceUri", "type": "string"},
            {"name": "SecurityConfig", "shape": "ModelPackageSecurityConfig", "type": "structure"},
            {"name": "ModelCard", "shape": "ModelPackageModelCard", "type": "structure"},
            {"name": "ModelLifeCycle", "shape": "ModelLifeCycle", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateModelPackageOutput": {
        "members": [{"name": "ModelPackageArn", "shape": "ModelPackageArn", "type": "string"}],
        "type": "structure",
    },
    "CreateModelQualityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"},
            {
                "name": "ModelQualityBaselineConfig",
                "shape": "ModelQualityBaselineConfig",
                "type": "structure",
            },
            {
                "name": "ModelQualityAppSpecification",
                "shape": "ModelQualityAppSpecification",
                "type": "structure",
            },
            {"name": "ModelQualityJobInput", "shape": "ModelQualityJobInput", "type": "structure"},
            {
                "name": "ModelQualityJobOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "JobResources", "shape": "MonitoringResources", "type": "structure"},
            {"name": "NetworkConfig", "shape": "MonitoringNetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateModelQualityJobDefinitionResponse": {
        "members": [
            {"name": "JobDefinitionArn", "shape": "MonitoringJobDefinitionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateMonitoringScheduleRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {
                "name": "MonitoringScheduleConfig",
                "shape": "MonitoringScheduleConfig",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateMonitoringScheduleResponse": {
        "members": [
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateNotebookInstanceInput": {
        "members": [
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"},
            {"name": "InstanceType", "shape": "InstanceType", "type": "string"},
            {"name": "SubnetId", "shape": "SubnetId", "type": "string"},
            {"name": "SecurityGroupIds", "shape": "SecurityGroupIds", "type": "list"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {
                "name": "LifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {"name": "DirectInternetAccess", "shape": "DirectInternetAccess", "type": "string"},
            {
                "name": "VolumeSizeInGB",
                "shape": "NotebookInstanceVolumeSizeInGB",
                "type": "integer",
            },
            {
                "name": "AcceleratorTypes",
                "shape": "NotebookInstanceAcceleratorTypes",
                "type": "list",
            },
            {"name": "DefaultCodeRepository", "shape": "CodeRepositoryNameOrUrl", "type": "string"},
            {
                "name": "AdditionalCodeRepositories",
                "shape": "AdditionalCodeRepositoryNamesOrUrls",
                "type": "list",
            },
            {"name": "RootAccess", "shape": "RootAccess", "type": "string"},
            {"name": "PlatformIdentifier", "shape": "PlatformIdentifier", "type": "string"},
            {
                "name": "InstanceMetadataServiceConfiguration",
                "shape": "InstanceMetadataServiceConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "CreateNotebookInstanceLifecycleConfigInput": {
        "members": [
            {
                "name": "NotebookInstanceLifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {"name": "OnCreate", "shape": "NotebookInstanceLifecycleConfigList", "type": "list"},
            {"name": "OnStart", "shape": "NotebookInstanceLifecycleConfigList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateNotebookInstanceLifecycleConfigOutput": {
        "members": [
            {
                "name": "NotebookInstanceLifecycleConfigArn",
                "shape": "NotebookInstanceLifecycleConfigArn",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "CreateNotebookInstanceOutput": {
        "members": [
            {"name": "NotebookInstanceArn", "shape": "NotebookInstanceArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreateOptimizationJobRequest": {
        "members": [
            {"name": "OptimizationJobName", "shape": "EntityName", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "ModelSource", "shape": "OptimizationJobModelSource", "type": "structure"},
            {
                "name": "DeploymentInstanceType",
                "shape": "OptimizationJobDeploymentInstanceType",
                "type": "string",
            },
            {
                "name": "OptimizationEnvironment",
                "shape": "OptimizationJobEnvironmentVariables",
                "type": "map",
            },
            {"name": "OptimizationConfigs", "shape": "OptimizationConfigs", "type": "list"},
            {"name": "OutputConfig", "shape": "OptimizationJobOutputConfig", "type": "structure"},
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "VpcConfig", "shape": "OptimizationVpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateOptimizationJobResponse": {
        "members": [
            {"name": "OptimizationJobArn", "shape": "OptimizationJobArn", "type": "string"}
        ],
        "type": "structure",
    },
    "CreatePartnerAppPresignedUrlRequest": {
        "members": [
            {"name": "Arn", "shape": "PartnerAppArn", "type": "string"},
            {"name": "ExpiresInSeconds", "shape": "ExpiresInSeconds", "type": "integer"},
            {
                "name": "SessionExpirationDurationInSeconds",
                "shape": "SessionExpirationDurationInSeconds",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "CreatePartnerAppPresignedUrlResponse": {
        "members": [{"name": "Url", "shape": "String2048", "type": "string"}],
        "type": "structure",
    },
    "CreatePartnerAppRequest": {
        "members": [
            {"name": "Name", "shape": "PartnerAppName", "type": "string"},
            {"name": "Type", "shape": "PartnerAppType", "type": "string"},
            {"name": "ExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "MaintenanceConfig",
                "shape": "PartnerAppMaintenanceConfig",
                "type": "structure",
            },
            {"name": "Tier", "shape": "NonEmptyString64", "type": "string"},
            {"name": "ApplicationConfig", "shape": "PartnerAppConfig", "type": "structure"},
            {"name": "AuthType", "shape": "PartnerAppAuthType", "type": "string"},
            {"name": "EnableIamSessionBasedIdentity", "shape": "Boolean", "type": "boolean"},
            {"name": "ClientToken", "shape": "ClientToken", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreatePartnerAppResponse": {
        "members": [{"name": "Arn", "shape": "PartnerAppArn", "type": "string"}],
        "type": "structure",
    },
    "CreatePipelineRequest": {
        "members": [
            {"name": "PipelineName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDisplayName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDefinition", "shape": "PipelineDefinition", "type": "string"},
            {
                "name": "PipelineDefinitionS3Location",
                "shape": "PipelineDefinitionS3Location",
                "type": "structure",
            },
            {"name": "PipelineDescription", "shape": "PipelineDescription", "type": "string"},
            {"name": "ClientRequestToken", "shape": "IdempotencyToken", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "CreatePipelineResponse": {
        "members": [{"name": "PipelineArn", "shape": "PipelineArn", "type": "string"}],
        "type": "structure",
    },
    "CreatePresignedDomainUrlRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {
                "name": "SessionExpirationDurationInSeconds",
                "shape": "SessionExpirationDurationInSeconds",
                "type": "integer",
            },
            {"name": "ExpiresInSeconds", "shape": "ExpiresInSeconds", "type": "integer"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "LandingUri", "shape": "LandingUri", "type": "string"},
        ],
        "type": "structure",
    },
    "CreatePresignedDomainUrlResponse": {
        "members": [{"name": "AuthorizedUrl", "shape": "PresignedDomainUrl", "type": "string"}],
        "type": "structure",
    },
    "CreatePresignedMlflowTrackingServerUrlRequest": {
        "members": [
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"},
            {"name": "ExpiresInSeconds", "shape": "ExpiresInSeconds", "type": "integer"},
            {
                "name": "SessionExpirationDurationInSeconds",
                "shape": "SessionExpirationDurationInSeconds",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "CreatePresignedMlflowTrackingServerUrlResponse": {
        "members": [{"name": "AuthorizedUrl", "shape": "TrackingServerUrl", "type": "string"}],
        "type": "structure",
    },
    "CreatePresignedNotebookInstanceUrlInput": {
        "members": [
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"},
            {
                "name": "SessionExpirationDurationInSeconds",
                "shape": "SessionExpirationDurationInSeconds",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "CreatePresignedNotebookInstanceUrlOutput": {
        "members": [{"name": "AuthorizedUrl", "shape": "NotebookInstanceUrl", "type": "string"}],
        "type": "structure",
    },
    "CreateProcessingJobRequest": {
        "members": [
            {"name": "ProcessingInputs", "shape": "ProcessingInputs", "type": "list"},
            {
                "name": "ProcessingOutputConfig",
                "shape": "ProcessingOutputConfig",
                "type": "structure",
            },
            {"name": "ProcessingJobName", "shape": "ProcessingJobName", "type": "string"},
            {"name": "ProcessingResources", "shape": "ProcessingResources", "type": "structure"},
            {
                "name": "StoppingCondition",
                "shape": "ProcessingStoppingCondition",
                "type": "structure",
            },
            {"name": "AppSpecification", "shape": "AppSpecification", "type": "structure"},
            {"name": "Environment", "shape": "ProcessingEnvironmentMap", "type": "map"},
            {"name": "NetworkConfig", "shape": "NetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateProcessingJobResponse": {
        "members": [{"name": "ProcessingJobArn", "shape": "ProcessingJobArn", "type": "string"}],
        "type": "structure",
    },
    "CreateProjectInput": {
        "members": [
            {"name": "ProjectName", "shape": "ProjectEntityName", "type": "string"},
            {"name": "ProjectDescription", "shape": "EntityDescription", "type": "string"},
            {
                "name": "ServiceCatalogProvisioningDetails",
                "shape": "ServiceCatalogProvisioningDetails",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateProjectOutput": {
        "members": [
            {"name": "ProjectArn", "shape": "ProjectArn", "type": "string"},
            {"name": "ProjectId", "shape": "ProjectId", "type": "string"},
        ],
        "type": "structure",
    },
    "CreateSpaceRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "SpaceSettings", "shape": "SpaceSettings", "type": "structure"},
            {"name": "OwnershipSettings", "shape": "OwnershipSettings", "type": "structure"},
            {"name": "SpaceSharingSettings", "shape": "SpaceSharingSettings", "type": "structure"},
            {"name": "SpaceDisplayName", "shape": "NonEmptyString64", "type": "string"},
        ],
        "type": "structure",
    },
    "CreateSpaceResponse": {
        "members": [{"name": "SpaceArn", "shape": "SpaceArn", "type": "string"}],
        "type": "structure",
    },
    "CreateStudioLifecycleConfigRequest": {
        "members": [
            {
                "name": "StudioLifecycleConfigName",
                "shape": "StudioLifecycleConfigName",
                "type": "string",
            },
            {
                "name": "StudioLifecycleConfigContent",
                "shape": "StudioLifecycleConfigContent",
                "type": "string",
            },
            {
                "name": "StudioLifecycleConfigAppType",
                "shape": "StudioLifecycleConfigAppType",
                "type": "string",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateStudioLifecycleConfigResponse": {
        "members": [
            {
                "name": "StudioLifecycleConfigArn",
                "shape": "StudioLifecycleConfigArn",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "CreateTrainingJobRequest": {
        "members": [
            {"name": "TrainingJobName", "shape": "TrainingJobName", "type": "string"},
            {"name": "HyperParameters", "shape": "HyperParameters", "type": "map"},
            {
                "name": "AlgorithmSpecification",
                "shape": "AlgorithmSpecification",
                "type": "structure",
            },
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "InputDataConfig", "shape": "InputDataConfig", "type": "list"},
            {"name": "OutputDataConfig", "shape": "OutputDataConfig", "type": "structure"},
            {"name": "ResourceConfig", "shape": "ResourceConfig", "type": "structure"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
            {
                "name": "EnableInterContainerTrafficEncryption",
                "shape": "Boolean",
                "type": "boolean",
            },
            {"name": "EnableManagedSpotTraining", "shape": "Boolean", "type": "boolean"},
            {"name": "CheckpointConfig", "shape": "CheckpointConfig", "type": "structure"},
            {"name": "DebugHookConfig", "shape": "DebugHookConfig", "type": "structure"},
            {"name": "DebugRuleConfigurations", "shape": "DebugRuleConfigurations", "type": "list"},
            {
                "name": "TensorBoardOutputConfig",
                "shape": "TensorBoardOutputConfig",
                "type": "structure",
            },
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
            {"name": "ProfilerConfig", "shape": "ProfilerConfig", "type": "structure"},
            {
                "name": "ProfilerRuleConfigurations",
                "shape": "ProfilerRuleConfigurations",
                "type": "list",
            },
            {"name": "Environment", "shape": "TrainingEnvironmentMap", "type": "map"},
            {"name": "RetryStrategy", "shape": "RetryStrategy", "type": "structure"},
            {"name": "RemoteDebugConfig", "shape": "RemoteDebugConfig", "type": "structure"},
            {"name": "InfraCheckConfig", "shape": "InfraCheckConfig", "type": "structure"},
            {
                "name": "SessionChainingConfig",
                "shape": "SessionChainingConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "CreateTrainingJobResponse": {
        "members": [{"name": "TrainingJobArn", "shape": "TrainingJobArn", "type": "string"}],
        "type": "structure",
    },
    "CreateTrainingPlanRequest": {
        "members": [
            {"name": "TrainingPlanName", "shape": "TrainingPlanName", "type": "string"},
            {"name": "TrainingPlanOfferingId", "shape": "TrainingPlanOfferingId", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateTrainingPlanResponse": {
        "members": [{"name": "TrainingPlanArn", "shape": "TrainingPlanArn", "type": "string"}],
        "type": "structure",
    },
    "CreateTransformJobRequest": {
        "members": [
            {"name": "TransformJobName", "shape": "TransformJobName", "type": "string"},
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {
                "name": "MaxConcurrentTransforms",
                "shape": "MaxConcurrentTransforms",
                "type": "integer",
            },
            {"name": "ModelClientConfig", "shape": "ModelClientConfig", "type": "structure"},
            {"name": "MaxPayloadInMB", "shape": "MaxPayloadInMB", "type": "integer"},
            {"name": "BatchStrategy", "shape": "BatchStrategy", "type": "string"},
            {"name": "Environment", "shape": "TransformEnvironmentMap", "type": "map"},
            {"name": "TransformInput", "shape": "TransformInput", "type": "structure"},
            {"name": "TransformOutput", "shape": "TransformOutput", "type": "structure"},
            {"name": "DataCaptureConfig", "shape": "BatchDataCaptureConfig", "type": "structure"},
            {"name": "TransformResources", "shape": "TransformResources", "type": "structure"},
            {"name": "DataProcessing", "shape": "DataProcessing", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateTransformJobResponse": {
        "members": [{"name": "TransformJobArn", "shape": "TransformJobArn", "type": "string"}],
        "type": "structure",
    },
    "CreateTrialComponentRequest": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Status", "shape": "TrialComponentStatus", "type": "structure"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Parameters", "shape": "TrialComponentParameters", "type": "map"},
            {"name": "InputArtifacts", "shape": "TrialComponentArtifacts", "type": "map"},
            {"name": "OutputArtifacts", "shape": "TrialComponentArtifacts", "type": "map"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateTrialComponentResponse": {
        "members": [{"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"}],
        "type": "structure",
    },
    "CreateTrialRequest": {
        "members": [
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateTrialResponse": {
        "members": [{"name": "TrialArn", "shape": "TrialArn", "type": "string"}],
        "type": "structure",
    },
    "CreateUserProfileRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {
                "name": "SingleSignOnUserIdentifier",
                "shape": "SingleSignOnUserIdentifier",
                "type": "string",
            },
            {"name": "SingleSignOnUserValue", "shape": "String256", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "UserSettings", "shape": "UserSettings", "type": "structure"},
        ],
        "type": "structure",
    },
    "CreateUserProfileResponse": {
        "members": [{"name": "UserProfileArn", "shape": "UserProfileArn", "type": "string"}],
        "type": "structure",
    },
    "CreateWorkforceRequest": {
        "members": [
            {"name": "CognitoConfig", "shape": "CognitoConfig", "type": "structure"},
            {"name": "OidcConfig", "shape": "OidcConfig", "type": "structure"},
            {"name": "SourceIpConfig", "shape": "SourceIpConfig", "type": "structure"},
            {"name": "WorkforceName", "shape": "WorkforceName", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {
                "name": "WorkforceVpcConfig",
                "shape": "WorkforceVpcConfigRequest",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "CreateWorkforceResponse": {
        "members": [{"name": "WorkforceArn", "shape": "WorkforceArn", "type": "string"}],
        "type": "structure",
    },
    "CreateWorkteamRequest": {
        "members": [
            {"name": "WorkteamName", "shape": "WorkteamName", "type": "string"},
            {"name": "WorkforceName", "shape": "WorkforceName", "type": "string"},
            {"name": "MemberDefinitions", "shape": "MemberDefinitions", "type": "list"},
            {"name": "Description", "shape": "String200", "type": "string"},
            {
                "name": "NotificationConfiguration",
                "shape": "NotificationConfiguration",
                "type": "structure",
            },
            {
                "name": "WorkerAccessConfiguration",
                "shape": "WorkerAccessConfiguration",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "CreateWorkteamResponse": {
        "members": [{"name": "WorkteamArn", "shape": "WorkteamArn", "type": "string"}],
        "type": "structure",
    },
    "CsvContentTypes": {"member_shape": "CsvContentType", "member_type": "string", "type": "list"},
    "CustomFileSystem": {
        "members": [
            {"name": "EFSFileSystem", "shape": "EFSFileSystem", "type": "structure"},
            {"name": "FSxLustreFileSystem", "shape": "FSxLustreFileSystem", "type": "structure"},
        ],
        "type": "structure",
    },
    "CustomFileSystemConfig": {
        "members": [
            {"name": "EFSFileSystemConfig", "shape": "EFSFileSystemConfig", "type": "structure"},
            {
                "name": "FSxLustreFileSystemConfig",
                "shape": "FSxLustreFileSystemConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "CustomFileSystemConfigs": {
        "member_shape": "CustomFileSystemConfig",
        "member_type": "structure",
        "type": "list",
    },
    "CustomFileSystems": {
        "member_shape": "CustomFileSystem",
        "member_type": "structure",
        "type": "list",
    },
    "CustomImage": {
        "members": [
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "ImageVersionNumber", "shape": "ImageVersionNumber", "type": "integer"},
            {"name": "AppImageConfigName", "shape": "AppImageConfigName", "type": "string"},
        ],
        "type": "structure",
    },
    "CustomImageContainerArguments": {
        "member_shape": "NonEmptyString64",
        "member_type": "string",
        "type": "list",
    },
    "CustomImageContainerEntrypoint": {
        "member_shape": "NonEmptyString256",
        "member_type": "string",
        "type": "list",
    },
    "CustomImageContainerEnvironmentVariables": {
        "key_shape": "NonEmptyString256",
        "key_type": "string",
        "type": "map",
        "value_shape": "String256",
        "value_type": "string",
    },
    "CustomImages": {"member_shape": "CustomImage", "member_type": "structure", "type": "list"},
    "CustomPosixUserConfig": {
        "members": [
            {"name": "Uid", "shape": "Uid", "type": "long"},
            {"name": "Gid", "shape": "Gid", "type": "long"},
        ],
        "type": "structure",
    },
    "CustomerMetadataKeyList": {
        "member_shape": "CustomerMetadataKey",
        "member_type": "string",
        "type": "list",
    },
    "CustomerMetadataMap": {
        "key_shape": "CustomerMetadataKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "CustomerMetadataValue",
        "value_type": "string",
    },
    "CustomizedMetricSpecification": {
        "members": [
            {"name": "MetricName", "shape": "String", "type": "string"},
            {"name": "Namespace", "shape": "String", "type": "string"},
            {"name": "Statistic", "shape": "Statistic", "type": "string"},
        ],
        "type": "structure",
    },
    "DataCaptureConfig": {
        "members": [
            {"name": "EnableCapture", "shape": "EnableCapture", "type": "boolean"},
            {"name": "InitialSamplingPercentage", "shape": "SamplingPercentage", "type": "integer"},
            {"name": "DestinationS3Uri", "shape": "DestinationS3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "CaptureOptions", "shape": "CaptureOptionList", "type": "list"},
            {
                "name": "CaptureContentTypeHeader",
                "shape": "CaptureContentTypeHeader",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DataCaptureConfigSummary": {
        "members": [
            {"name": "EnableCapture", "shape": "EnableCapture", "type": "boolean"},
            {"name": "CaptureStatus", "shape": "CaptureStatus", "type": "string"},
            {"name": "CurrentSamplingPercentage", "shape": "SamplingPercentage", "type": "integer"},
            {"name": "DestinationS3Uri", "shape": "DestinationS3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "DataCatalogConfig": {
        "members": [
            {"name": "TableName", "shape": "TableName", "type": "string"},
            {"name": "Catalog", "shape": "Catalog", "type": "string"},
            {"name": "Database", "shape": "Database", "type": "string"},
        ],
        "type": "structure",
    },
    "DataProcessing": {
        "members": [
            {"name": "InputFilter", "shape": "JsonPath", "type": "string"},
            {"name": "OutputFilter", "shape": "JsonPath", "type": "string"},
            {"name": "JoinSource", "shape": "JoinSource", "type": "string"},
        ],
        "type": "structure",
    },
    "DataQualityAppSpecification": {
        "members": [
            {"name": "ImageUri", "shape": "ImageUri", "type": "string"},
            {"name": "ContainerEntrypoint", "shape": "ContainerEntrypoint", "type": "list"},
            {"name": "ContainerArguments", "shape": "MonitoringContainerArguments", "type": "list"},
            {"name": "RecordPreprocessorSourceUri", "shape": "S3Uri", "type": "string"},
            {"name": "PostAnalyticsProcessorSourceUri", "shape": "S3Uri", "type": "string"},
            {"name": "Environment", "shape": "MonitoringEnvironmentMap", "type": "map"},
        ],
        "type": "structure",
    },
    "DataQualityBaselineConfig": {
        "members": [
            {"name": "BaseliningJobName", "shape": "ProcessingJobName", "type": "string"},
            {
                "name": "ConstraintsResource",
                "shape": "MonitoringConstraintsResource",
                "type": "structure",
            },
            {
                "name": "StatisticsResource",
                "shape": "MonitoringStatisticsResource",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DataQualityJobInput": {
        "members": [
            {"name": "EndpointInput", "shape": "EndpointInput", "type": "structure"},
            {"name": "BatchTransformInput", "shape": "BatchTransformInput", "type": "structure"},
        ],
        "type": "structure",
    },
    "DataSource": {
        "members": [
            {"name": "S3DataSource", "shape": "S3DataSource", "type": "structure"},
            {"name": "FileSystemDataSource", "shape": "FileSystemDataSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "DatasetDefinition": {
        "members": [
            {
                "name": "AthenaDatasetDefinition",
                "shape": "AthenaDatasetDefinition",
                "type": "structure",
            },
            {
                "name": "RedshiftDatasetDefinition",
                "shape": "RedshiftDatasetDefinition",
                "type": "structure",
            },
            {"name": "LocalPath", "shape": "ProcessingLocalPath", "type": "string"},
            {"name": "DataDistributionType", "shape": "DataDistributionType", "type": "string"},
            {"name": "InputMode", "shape": "InputMode", "type": "string"},
        ],
        "type": "structure",
    },
    "DebugHookConfig": {
        "members": [
            {"name": "LocalPath", "shape": "DirectoryPath", "type": "string"},
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {"name": "HookParameters", "shape": "HookParameters", "type": "map"},
            {
                "name": "CollectionConfigurations",
                "shape": "CollectionConfigurations",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "DebugRuleConfiguration": {
        "members": [
            {"name": "RuleConfigurationName", "shape": "RuleConfigurationName", "type": "string"},
            {"name": "LocalPath", "shape": "DirectoryPath", "type": "string"},
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {"name": "RuleEvaluatorImage", "shape": "AlgorithmImage", "type": "string"},
            {"name": "InstanceType", "shape": "ProcessingInstanceType", "type": "string"},
            {"name": "VolumeSizeInGB", "shape": "OptionalVolumeSizeInGB", "type": "integer"},
            {"name": "RuleParameters", "shape": "RuleParameters", "type": "map"},
        ],
        "type": "structure",
    },
    "DebugRuleConfigurations": {
        "member_shape": "DebugRuleConfiguration",
        "member_type": "structure",
        "type": "list",
    },
    "DebugRuleEvaluationStatus": {
        "members": [
            {"name": "RuleConfigurationName", "shape": "RuleConfigurationName", "type": "string"},
            {"name": "RuleEvaluationJobArn", "shape": "ProcessingJobArn", "type": "string"},
            {"name": "RuleEvaluationStatus", "shape": "RuleEvaluationStatus", "type": "string"},
            {"name": "StatusDetails", "shape": "StatusDetails", "type": "string"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DebugRuleEvaluationStatuses": {
        "member_shape": "DebugRuleEvaluationStatus",
        "member_type": "structure",
        "type": "list",
    },
    "DefaultEbsStorageSettings": {
        "members": [
            {
                "name": "DefaultEbsVolumeSizeInGb",
                "shape": "SpaceEbsVolumeSizeInGb",
                "type": "integer",
            },
            {
                "name": "MaximumEbsVolumeSizeInGb",
                "shape": "SpaceEbsVolumeSizeInGb",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "DefaultSpaceSettings": {
        "members": [
            {"name": "ExecutionRole", "shape": "RoleArn", "type": "string"},
            {"name": "SecurityGroups", "shape": "SecurityGroupIds", "type": "list"},
            {
                "name": "JupyterServerAppSettings",
                "shape": "JupyterServerAppSettings",
                "type": "structure",
            },
            {
                "name": "KernelGatewayAppSettings",
                "shape": "KernelGatewayAppSettings",
                "type": "structure",
            },
            {
                "name": "JupyterLabAppSettings",
                "shape": "JupyterLabAppSettings",
                "type": "structure",
            },
            {
                "name": "SpaceStorageSettings",
                "shape": "DefaultSpaceStorageSettings",
                "type": "structure",
            },
            {
                "name": "CustomPosixUserConfig",
                "shape": "CustomPosixUserConfig",
                "type": "structure",
            },
            {"name": "CustomFileSystemConfigs", "shape": "CustomFileSystemConfigs", "type": "list"},
        ],
        "type": "structure",
    },
    "DefaultSpaceStorageSettings": {
        "members": [
            {
                "name": "DefaultEbsStorageSettings",
                "shape": "DefaultEbsStorageSettings",
                "type": "structure",
            }
        ],
        "type": "structure",
    },
    "DeleteActionRequest": {
        "members": [{"name": "ActionName", "shape": "ExperimentEntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteActionResponse": {
        "members": [{"name": "ActionArn", "shape": "ActionArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteAlgorithmInput": {
        "members": [{"name": "AlgorithmName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteAppImageConfigRequest": {
        "members": [
            {"name": "AppImageConfigName", "shape": "AppImageConfigName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteAppRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "AppType", "shape": "AppType", "type": "string"},
            {"name": "AppName", "shape": "AppName", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteArtifactRequest": {
        "members": [
            {"name": "ArtifactArn", "shape": "ArtifactArn", "type": "string"},
            {"name": "Source", "shape": "ArtifactSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "DeleteArtifactResponse": {
        "members": [{"name": "ArtifactArn", "shape": "ArtifactArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteAssociationRequest": {
        "members": [
            {"name": "SourceArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "DestinationArn", "shape": "AssociationEntityArn", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteAssociationResponse": {
        "members": [
            {"name": "SourceArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "DestinationArn", "shape": "AssociationEntityArn", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteClusterRequest": {
        "members": [{"name": "ClusterName", "shape": "ClusterNameOrArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteClusterResponse": {
        "members": [{"name": "ClusterArn", "shape": "ClusterArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteClusterSchedulerConfigRequest": {
        "members": [
            {
                "name": "ClusterSchedulerConfigId",
                "shape": "ClusterSchedulerConfigId",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "DeleteCodeRepositoryInput": {
        "members": [{"name": "CodeRepositoryName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteCompilationJobRequest": {
        "members": [{"name": "CompilationJobName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteComputeQuotaRequest": {
        "members": [{"name": "ComputeQuotaId", "shape": "ComputeQuotaId", "type": "string"}],
        "type": "structure",
    },
    "DeleteContextRequest": {
        "members": [{"name": "ContextName", "shape": "ContextName", "type": "string"}],
        "type": "structure",
    },
    "DeleteContextResponse": {
        "members": [{"name": "ContextArn", "shape": "ContextArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteDataQualityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteDeviceFleetRequest": {
        "members": [{"name": "DeviceFleetName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteDomainRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "RetentionPolicy", "shape": "RetentionPolicy", "type": "structure"},
        ],
        "type": "structure",
    },
    "DeleteEdgeDeploymentPlanRequest": {
        "members": [{"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteEdgeDeploymentStageRequest": {
        "members": [
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "StageName", "shape": "EntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteEndpointConfigInput": {
        "members": [
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteEndpointInput": {
        "members": [{"name": "EndpointName", "shape": "EndpointName", "type": "string"}],
        "type": "structure",
    },
    "DeleteExperimentRequest": {
        "members": [{"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteExperimentResponse": {
        "members": [{"name": "ExperimentArn", "shape": "ExperimentArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteFeatureGroupRequest": {
        "members": [{"name": "FeatureGroupName", "shape": "FeatureGroupName", "type": "string"}],
        "type": "structure",
    },
    "DeleteFlowDefinitionRequest": {
        "members": [
            {"name": "FlowDefinitionName", "shape": "FlowDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteFlowDefinitionResponse": {"members": [], "type": "structure"},
    "DeleteHubContentReferenceRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteHubContentRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "HubContentVersion", "shape": "HubContentVersion", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteHubRequest": {
        "members": [{"name": "HubName", "shape": "HubNameOrArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteHumanTaskUiRequest": {
        "members": [{"name": "HumanTaskUiName", "shape": "HumanTaskUiName", "type": "string"}],
        "type": "structure",
    },
    "DeleteHumanTaskUiResponse": {"members": [], "type": "structure"},
    "DeleteHyperParameterTuningJobRequest": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "DeleteImageRequest": {
        "members": [{"name": "ImageName", "shape": "ImageName", "type": "string"}],
        "type": "structure",
    },
    "DeleteImageResponse": {"members": [], "type": "structure"},
    "DeleteImageVersionRequest": {
        "members": [
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "Version", "shape": "ImageVersionNumber", "type": "integer"},
            {"name": "Alias", "shape": "SageMakerImageVersionAlias", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteImageVersionResponse": {"members": [], "type": "structure"},
    "DeleteInferenceComponentInput": {
        "members": [
            {"name": "InferenceComponentName", "shape": "InferenceComponentName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteInferenceExperimentRequest": {
        "members": [{"name": "Name", "shape": "InferenceExperimentName", "type": "string"}],
        "type": "structure",
    },
    "DeleteInferenceExperimentResponse": {
        "members": [
            {"name": "InferenceExperimentArn", "shape": "InferenceExperimentArn", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteMlflowTrackingServerRequest": {
        "members": [
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteMlflowTrackingServerResponse": {
        "members": [{"name": "TrackingServerArn", "shape": "TrackingServerArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteModelBiasJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteModelCardRequest": {
        "members": [{"name": "ModelCardName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteModelExplainabilityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteModelInput": {
        "members": [{"name": "ModelName", "shape": "ModelName", "type": "string"}],
        "type": "structure",
    },
    "DeleteModelPackageGroupInput": {
        "members": [{"name": "ModelPackageGroupName", "shape": "ArnOrName", "type": "string"}],
        "type": "structure",
    },
    "DeleteModelPackageGroupPolicyInput": {
        "members": [{"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteModelPackageInput": {
        "members": [{"name": "ModelPackageName", "shape": "VersionedArnOrName", "type": "string"}],
        "type": "structure",
    },
    "DeleteModelQualityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteMonitoringScheduleRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteNotebookInstanceInput": {
        "members": [
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteNotebookInstanceLifecycleConfigInput": {
        "members": [
            {
                "name": "NotebookInstanceLifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "DeleteOptimizationJobRequest": {
        "members": [{"name": "OptimizationJobName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DeletePartnerAppRequest": {
        "members": [
            {"name": "Arn", "shape": "PartnerAppArn", "type": "string"},
            {"name": "ClientToken", "shape": "ClientToken", "type": "string"},
        ],
        "type": "structure",
    },
    "DeletePartnerAppResponse": {
        "members": [{"name": "Arn", "shape": "PartnerAppArn", "type": "string"}],
        "type": "structure",
    },
    "DeletePipelineRequest": {
        "members": [
            {"name": "PipelineName", "shape": "PipelineName", "type": "string"},
            {"name": "ClientRequestToken", "shape": "IdempotencyToken", "type": "string"},
        ],
        "type": "structure",
    },
    "DeletePipelineResponse": {
        "members": [{"name": "PipelineArn", "shape": "PipelineArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteProjectInput": {
        "members": [{"name": "ProjectName", "shape": "ProjectEntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteRecordRequest": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupNameOrArn", "type": "string"},
            {"name": "RecordIdentifierValueAsString", "shape": "ValueAsString", "type": "string"},
            {"name": "EventTime", "shape": "ValueAsString", "type": "string"},
            {"name": "TargetStores", "shape": "TargetStores", "type": "list"},
            {"name": "DeletionMode", "shape": "DeletionMode", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteSpaceRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteStudioLifecycleConfigRequest": {
        "members": [
            {
                "name": "StudioLifecycleConfigName",
                "shape": "StudioLifecycleConfigName",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "DeleteTagsInput": {
        "members": [
            {"name": "ResourceArn", "shape": "ResourceArn", "type": "string"},
            {"name": "TagKeys", "shape": "TagKeyList", "type": "list"},
        ],
        "type": "structure",
    },
    "DeleteTagsOutput": {"members": [], "type": "structure"},
    "DeleteTrialComponentRequest": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"}
        ],
        "type": "structure",
    },
    "DeleteTrialComponentResponse": {
        "members": [{"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteTrialRequest": {
        "members": [{"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"}],
        "type": "structure",
    },
    "DeleteTrialResponse": {
        "members": [{"name": "TrialArn", "shape": "TrialArn", "type": "string"}],
        "type": "structure",
    },
    "DeleteUserProfileRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
        ],
        "type": "structure",
    },
    "DeleteWorkforceRequest": {
        "members": [{"name": "WorkforceName", "shape": "WorkforceName", "type": "string"}],
        "type": "structure",
    },
    "DeleteWorkforceResponse": {"members": [], "type": "structure"},
    "DeleteWorkteamRequest": {
        "members": [{"name": "WorkteamName", "shape": "WorkteamName", "type": "string"}],
        "type": "structure",
    },
    "DeleteWorkteamResponse": {
        "members": [{"name": "Success", "shape": "Success", "type": "boolean"}],
        "type": "structure",
    },
    "DeployedImage": {
        "members": [
            {"name": "SpecifiedImage", "shape": "ContainerImage", "type": "string"},
            {"name": "ResolvedImage", "shape": "ContainerImage", "type": "string"},
            {"name": "ResolutionTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DeployedImages": {"member_shape": "DeployedImage", "member_type": "structure", "type": "list"},
    "DeploymentConfig": {
        "members": [
            {
                "name": "BlueGreenUpdatePolicy",
                "shape": "BlueGreenUpdatePolicy",
                "type": "structure",
            },
            {"name": "RollingUpdatePolicy", "shape": "RollingUpdatePolicy", "type": "structure"},
            {
                "name": "AutoRollbackConfiguration",
                "shape": "AutoRollbackConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DeploymentRecommendation": {
        "members": [
            {"name": "RecommendationStatus", "shape": "RecommendationStatus", "type": "string"},
            {
                "name": "RealTimeInferenceRecommendations",
                "shape": "RealTimeInferenceRecommendations",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "DeploymentStage": {
        "members": [
            {"name": "StageName", "shape": "EntityName", "type": "string"},
            {
                "name": "DeviceSelectionConfig",
                "shape": "DeviceSelectionConfig",
                "type": "structure",
            },
            {"name": "DeploymentConfig", "shape": "EdgeDeploymentConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "DeploymentStageStatusSummaries": {
        "member_shape": "DeploymentStageStatusSummary",
        "member_type": "structure",
        "type": "list",
    },
    "DeploymentStageStatusSummary": {
        "members": [
            {"name": "StageName", "shape": "EntityName", "type": "string"},
            {
                "name": "DeviceSelectionConfig",
                "shape": "DeviceSelectionConfig",
                "type": "structure",
            },
            {"name": "DeploymentConfig", "shape": "EdgeDeploymentConfig", "type": "structure"},
            {"name": "DeploymentStatus", "shape": "EdgeDeploymentStatus", "type": "structure"},
        ],
        "type": "structure",
    },
    "DeploymentStages": {
        "member_shape": "DeploymentStage",
        "member_type": "structure",
        "type": "list",
    },
    "DeregisterDevicesRequest": {
        "members": [
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "DeviceNames", "shape": "DeviceNames", "type": "list"},
        ],
        "type": "structure",
    },
    "DerivedInformation": {
        "members": [
            {"name": "DerivedDataInputConfig", "shape": "DataInputConfig", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeActionRequest": {
        "members": [{"name": "ActionName", "shape": "ExperimentEntityNameOrArn", "type": "string"}],
        "type": "structure",
    },
    "DescribeActionResponse": {
        "members": [
            {"name": "ActionName", "shape": "ExperimentEntityNameOrArn", "type": "string"},
            {"name": "ActionArn", "shape": "ActionArn", "type": "string"},
            {"name": "Source", "shape": "ActionSource", "type": "structure"},
            {"name": "ActionType", "shape": "String256", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "Status", "shape": "ActionStatus", "type": "string"},
            {"name": "Properties", "shape": "LineageEntityParameters", "type": "map"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "LineageGroupArn", "shape": "LineageGroupArn", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeAlgorithmInput": {
        "members": [{"name": "AlgorithmName", "shape": "ArnOrName", "type": "string"}],
        "type": "structure",
    },
    "DescribeAlgorithmOutput": {
        "members": [
            {"name": "AlgorithmName", "shape": "EntityName", "type": "string"},
            {"name": "AlgorithmArn", "shape": "AlgorithmArn", "type": "string"},
            {"name": "AlgorithmDescription", "shape": "EntityDescription", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {
                "name": "TrainingSpecification",
                "shape": "TrainingSpecification",
                "type": "structure",
            },
            {
                "name": "InferenceSpecification",
                "shape": "InferenceSpecification",
                "type": "structure",
            },
            {
                "name": "ValidationSpecification",
                "shape": "AlgorithmValidationSpecification",
                "type": "structure",
            },
            {"name": "AlgorithmStatus", "shape": "AlgorithmStatus", "type": "string"},
            {
                "name": "AlgorithmStatusDetails",
                "shape": "AlgorithmStatusDetails",
                "type": "structure",
            },
            {"name": "ProductId", "shape": "ProductId", "type": "string"},
            {"name": "CertifyForMarketplace", "shape": "CertifyForMarketplace", "type": "boolean"},
        ],
        "type": "structure",
    },
    "DescribeAppImageConfigRequest": {
        "members": [
            {"name": "AppImageConfigName", "shape": "AppImageConfigName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeAppImageConfigResponse": {
        "members": [
            {"name": "AppImageConfigArn", "shape": "AppImageConfigArn", "type": "string"},
            {"name": "AppImageConfigName", "shape": "AppImageConfigName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "KernelGatewayImageConfig",
                "shape": "KernelGatewayImageConfig",
                "type": "structure",
            },
            {
                "name": "JupyterLabAppImageConfig",
                "shape": "JupyterLabAppImageConfig",
                "type": "structure",
            },
            {
                "name": "CodeEditorAppImageConfig",
                "shape": "CodeEditorAppImageConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeAppRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "AppType", "shape": "AppType", "type": "string"},
            {"name": "AppName", "shape": "AppName", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeAppResponse": {
        "members": [
            {"name": "AppArn", "shape": "AppArn", "type": "string"},
            {"name": "AppType", "shape": "AppType", "type": "string"},
            {"name": "AppName", "shape": "AppName", "type": "string"},
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "Status", "shape": "AppStatus", "type": "string"},
            {"name": "LastHealthCheckTimestamp", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastUserActivityTimestamp", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {
                "name": "BuiltInLifecycleConfigArn",
                "shape": "StudioLifecycleConfigArn",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "DescribeArtifactRequest": {
        "members": [{"name": "ArtifactArn", "shape": "ArtifactArn", "type": "string"}],
        "type": "structure",
    },
    "DescribeArtifactResponse": {
        "members": [
            {"name": "ArtifactName", "shape": "ExperimentEntityNameOrArn", "type": "string"},
            {"name": "ArtifactArn", "shape": "ArtifactArn", "type": "string"},
            {"name": "Source", "shape": "ArtifactSource", "type": "structure"},
            {"name": "ArtifactType", "shape": "String256", "type": "string"},
            {"name": "Properties", "shape": "LineageEntityParameters", "type": "map"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "LineageGroupArn", "shape": "LineageGroupArn", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeAutoMLJobRequest": {
        "members": [{"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"}],
        "type": "structure",
    },
    "DescribeAutoMLJobResponse": {
        "members": [
            {"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {"name": "InputDataConfig", "shape": "AutoMLInputDataConfig", "type": "list"},
            {"name": "OutputDataConfig", "shape": "AutoMLOutputDataConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "AutoMLJobObjective", "shape": "AutoMLJobObjective", "type": "structure"},
            {"name": "ProblemType", "shape": "ProblemType", "type": "string"},
            {"name": "AutoMLJobConfig", "shape": "AutoMLJobConfig", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "AutoMLFailureReason", "type": "string"},
            {
                "name": "PartialFailureReasons",
                "shape": "AutoMLPartialFailureReasons",
                "type": "list",
            },
            {"name": "BestCandidate", "shape": "AutoMLCandidate", "type": "structure"},
            {"name": "AutoMLJobStatus", "shape": "AutoMLJobStatus", "type": "string"},
            {
                "name": "AutoMLJobSecondaryStatus",
                "shape": "AutoMLJobSecondaryStatus",
                "type": "string",
            },
            {
                "name": "GenerateCandidateDefinitionsOnly",
                "shape": "GenerateCandidateDefinitionsOnly",
                "type": "boolean",
            },
            {"name": "AutoMLJobArtifacts", "shape": "AutoMLJobArtifacts", "type": "structure"},
            {"name": "ResolvedAttributes", "shape": "ResolvedAttributes", "type": "structure"},
            {"name": "ModelDeployConfig", "shape": "ModelDeployConfig", "type": "structure"},
            {"name": "ModelDeployResult", "shape": "ModelDeployResult", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeAutoMLJobV2Request": {
        "members": [{"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"}],
        "type": "structure",
    },
    "DescribeAutoMLJobV2Response": {
        "members": [
            {"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {
                "name": "AutoMLJobInputDataConfig",
                "shape": "AutoMLJobInputDataConfig",
                "type": "list",
            },
            {"name": "OutputDataConfig", "shape": "AutoMLOutputDataConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "AutoMLJobObjective", "shape": "AutoMLJobObjective", "type": "structure"},
            {
                "name": "AutoMLProblemTypeConfig",
                "shape": "AutoMLProblemTypeConfig",
                "type": "structure",
            },
            {
                "name": "AutoMLProblemTypeConfigName",
                "shape": "AutoMLProblemTypeConfigName",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "AutoMLFailureReason", "type": "string"},
            {
                "name": "PartialFailureReasons",
                "shape": "AutoMLPartialFailureReasons",
                "type": "list",
            },
            {"name": "BestCandidate", "shape": "AutoMLCandidate", "type": "structure"},
            {"name": "AutoMLJobStatus", "shape": "AutoMLJobStatus", "type": "string"},
            {
                "name": "AutoMLJobSecondaryStatus",
                "shape": "AutoMLJobSecondaryStatus",
                "type": "string",
            },
            {"name": "AutoMLJobArtifacts", "shape": "AutoMLJobArtifacts", "type": "structure"},
            {
                "name": "ResolvedAttributes",
                "shape": "AutoMLResolvedAttributes",
                "type": "structure",
            },
            {"name": "ModelDeployConfig", "shape": "ModelDeployConfig", "type": "structure"},
            {"name": "ModelDeployResult", "shape": "ModelDeployResult", "type": "structure"},
            {"name": "DataSplitConfig", "shape": "AutoMLDataSplitConfig", "type": "structure"},
            {"name": "SecurityConfig", "shape": "AutoMLSecurityConfig", "type": "structure"},
            {"name": "AutoMLComputeConfig", "shape": "AutoMLComputeConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeClusterNodeRequest": {
        "members": [
            {"name": "ClusterName", "shape": "ClusterNameOrArn", "type": "string"},
            {"name": "NodeId", "shape": "ClusterNodeId", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeClusterNodeResponse": {
        "members": [{"name": "NodeDetails", "shape": "ClusterNodeDetails", "type": "structure"}],
        "type": "structure",
    },
    "DescribeClusterRequest": {
        "members": [{"name": "ClusterName", "shape": "ClusterNameOrArn", "type": "string"}],
        "type": "structure",
    },
    "DescribeClusterResponse": {
        "members": [
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "ClusterName", "shape": "ClusterName", "type": "string"},
            {"name": "ClusterStatus", "shape": "ClusterStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureMessage", "shape": "String", "type": "string"},
            {"name": "InstanceGroups", "shape": "ClusterInstanceGroupDetailsList", "type": "list"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "Orchestrator", "shape": "ClusterOrchestrator", "type": "structure"},
            {"name": "NodeRecovery", "shape": "ClusterNodeRecovery", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeClusterSchedulerConfigRequest": {
        "members": [
            {
                "name": "ClusterSchedulerConfigId",
                "shape": "ClusterSchedulerConfigId",
                "type": "string",
            },
            {"name": "ClusterSchedulerConfigVersion", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "DescribeClusterSchedulerConfigResponse": {
        "members": [
            {
                "name": "ClusterSchedulerConfigArn",
                "shape": "ClusterSchedulerConfigArn",
                "type": "string",
            },
            {
                "name": "ClusterSchedulerConfigId",
                "shape": "ClusterSchedulerConfigId",
                "type": "string",
            },
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "ClusterSchedulerConfigVersion", "shape": "Integer", "type": "integer"},
            {"name": "Status", "shape": "SchedulerResourceStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "SchedulerConfig", "shape": "SchedulerConfig", "type": "structure"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeCodeRepositoryInput": {
        "members": [{"name": "CodeRepositoryName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DescribeCodeRepositoryOutput": {
        "members": [
            {"name": "CodeRepositoryName", "shape": "EntityName", "type": "string"},
            {"name": "CodeRepositoryArn", "shape": "CodeRepositoryArn", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "GitConfig", "shape": "GitConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeCompilationJobRequest": {
        "members": [{"name": "CompilationJobName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DescribeCompilationJobResponse": {
        "members": [
            {"name": "CompilationJobName", "shape": "EntityName", "type": "string"},
            {"name": "CompilationJobArn", "shape": "CompilationJobArn", "type": "string"},
            {"name": "CompilationJobStatus", "shape": "CompilationJobStatus", "type": "string"},
            {"name": "CompilationStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CompilationEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
            {"name": "InferenceImage", "shape": "InferenceImage", "type": "string"},
            {"name": "ModelPackageVersionArn", "shape": "ModelPackageArn", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ModelArtifacts", "shape": "ModelArtifacts", "type": "structure"},
            {"name": "ModelDigests", "shape": "ModelDigests", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "InputConfig", "shape": "InputConfig", "type": "structure"},
            {"name": "OutputConfig", "shape": "OutputConfig", "type": "structure"},
            {"name": "VpcConfig", "shape": "NeoVpcConfig", "type": "structure"},
            {"name": "DerivedInformation", "shape": "DerivedInformation", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeComputeQuotaRequest": {
        "members": [
            {"name": "ComputeQuotaId", "shape": "ComputeQuotaId", "type": "string"},
            {"name": "ComputeQuotaVersion", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "DescribeComputeQuotaResponse": {
        "members": [
            {"name": "ComputeQuotaArn", "shape": "ComputeQuotaArn", "type": "string"},
            {"name": "ComputeQuotaId", "shape": "ComputeQuotaId", "type": "string"},
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
            {"name": "ComputeQuotaVersion", "shape": "Integer", "type": "integer"},
            {"name": "Status", "shape": "SchedulerResourceStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "ComputeQuotaConfig", "shape": "ComputeQuotaConfig", "type": "structure"},
            {"name": "ComputeQuotaTarget", "shape": "ComputeQuotaTarget", "type": "structure"},
            {"name": "ActivationState", "shape": "ActivationState", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeContextRequest": {
        "members": [{"name": "ContextName", "shape": "ContextNameOrArn", "type": "string"}],
        "type": "structure",
    },
    "DescribeContextResponse": {
        "members": [
            {"name": "ContextName", "shape": "ContextName", "type": "string"},
            {"name": "ContextArn", "shape": "ContextArn", "type": "string"},
            {"name": "Source", "shape": "ContextSource", "type": "structure"},
            {"name": "ContextType", "shape": "String256", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "Properties", "shape": "LineageEntityParameters", "type": "map"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LineageGroupArn", "shape": "LineageGroupArn", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeDataQualityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeDataQualityJobDefinitionResponse": {
        "members": [
            {"name": "JobDefinitionArn", "shape": "MonitoringJobDefinitionArn", "type": "string"},
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "DataQualityBaselineConfig",
                "shape": "DataQualityBaselineConfig",
                "type": "structure",
            },
            {
                "name": "DataQualityAppSpecification",
                "shape": "DataQualityAppSpecification",
                "type": "structure",
            },
            {"name": "DataQualityJobInput", "shape": "DataQualityJobInput", "type": "structure"},
            {
                "name": "DataQualityJobOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "JobResources", "shape": "MonitoringResources", "type": "structure"},
            {"name": "NetworkConfig", "shape": "MonitoringNetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeDeviceFleetRequest": {
        "members": [{"name": "DeviceFleetName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DescribeDeviceFleetResponse": {
        "members": [
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "DeviceFleetArn", "shape": "DeviceFleetArn", "type": "string"},
            {"name": "OutputConfig", "shape": "EdgeOutputConfig", "type": "structure"},
            {"name": "Description", "shape": "DeviceFleetDescription", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "IotRoleAlias", "shape": "IotRoleAlias", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeDeviceRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "DeviceName", "shape": "EntityName", "type": "string"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeDeviceResponse": {
        "members": [
            {"name": "DeviceArn", "shape": "DeviceArn", "type": "string"},
            {"name": "DeviceName", "shape": "EntityName", "type": "string"},
            {"name": "Description", "shape": "DeviceDescription", "type": "string"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "IotThingName", "shape": "ThingName", "type": "string"},
            {"name": "RegistrationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LatestHeartbeat", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Models", "shape": "EdgeModels", "type": "list"},
            {"name": "MaxModels", "shape": "Integer", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "AgentVersion", "shape": "EdgeVersion", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeDomainRequest": {
        "members": [{"name": "DomainId", "shape": "DomainId", "type": "string"}],
        "type": "structure",
    },
    "DescribeDomainResponse": {
        "members": [
            {"name": "DomainArn", "shape": "DomainArn", "type": "string"},
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "DomainName", "shape": "DomainName", "type": "string"},
            {"name": "HomeEfsFileSystemId", "shape": "ResourceId", "type": "string"},
            {
                "name": "SingleSignOnManagedApplicationInstanceId",
                "shape": "String256",
                "type": "string",
            },
            {
                "name": "SingleSignOnApplicationArn",
                "shape": "SingleSignOnApplicationArn",
                "type": "string",
            },
            {"name": "Status", "shape": "DomainStatus", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {
                "name": "SecurityGroupIdForDomainBoundary",
                "shape": "SecurityGroupId",
                "type": "string",
            },
            {"name": "AuthMode", "shape": "AuthMode", "type": "string"},
            {"name": "DefaultUserSettings", "shape": "UserSettings", "type": "structure"},
            {"name": "DomainSettings", "shape": "DomainSettings", "type": "structure"},
            {"name": "AppNetworkAccessType", "shape": "AppNetworkAccessType", "type": "string"},
            {"name": "HomeEfsFileSystemKmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "SubnetIds", "shape": "Subnets", "type": "list"},
            {"name": "Url", "shape": "String1024", "type": "string"},
            {"name": "VpcId", "shape": "VpcId", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {
                "name": "AppSecurityGroupManagement",
                "shape": "AppSecurityGroupManagement",
                "type": "string",
            },
            {"name": "TagPropagation", "shape": "TagPropagation", "type": "string"},
            {"name": "DefaultSpaceSettings", "shape": "DefaultSpaceSettings", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeEdgeDeploymentPlanRequest": {
        "members": [
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "DeploymentStageMaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "DescribeEdgeDeploymentPlanResponse": {
        "members": [
            {"name": "EdgeDeploymentPlanArn", "shape": "EdgeDeploymentPlanArn", "type": "string"},
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "ModelConfigs", "shape": "EdgeDeploymentModelConfigs", "type": "list"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "EdgeDeploymentSuccess", "shape": "Integer", "type": "integer"},
            {"name": "EdgeDeploymentPending", "shape": "Integer", "type": "integer"},
            {"name": "EdgeDeploymentFailed", "shape": "Integer", "type": "integer"},
            {"name": "Stages", "shape": "DeploymentStageStatusSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DescribeEdgePackagingJobRequest": {
        "members": [{"name": "EdgePackagingJobName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DescribeEdgePackagingJobResponse": {
        "members": [
            {"name": "EdgePackagingJobArn", "shape": "EdgePackagingJobArn", "type": "string"},
            {"name": "EdgePackagingJobName", "shape": "EntityName", "type": "string"},
            {"name": "CompilationJobName", "shape": "EntityName", "type": "string"},
            {"name": "ModelName", "shape": "EntityName", "type": "string"},
            {"name": "ModelVersion", "shape": "EdgeVersion", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "OutputConfig", "shape": "EdgeOutputConfig", "type": "structure"},
            {"name": "ResourceKey", "shape": "KmsKeyId", "type": "string"},
            {"name": "EdgePackagingJobStatus", "shape": "EdgePackagingJobStatus", "type": "string"},
            {"name": "EdgePackagingJobStatusMessage", "shape": "String", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModelArtifact", "shape": "S3Uri", "type": "string"},
            {"name": "ModelSignature", "shape": "String", "type": "string"},
            {
                "name": "PresetDeploymentOutput",
                "shape": "EdgePresetDeploymentOutput",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeEndpointConfigInput": {
        "members": [
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeEndpointConfigOutput": {
        "members": [
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {"name": "EndpointConfigArn", "shape": "EndpointConfigArn", "type": "string"},
            {"name": "ProductionVariants", "shape": "ProductionVariantList", "type": "list"},
            {"name": "DataCaptureConfig", "shape": "DataCaptureConfig", "type": "structure"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "AsyncInferenceConfig", "shape": "AsyncInferenceConfig", "type": "structure"},
            {"name": "ExplainerConfig", "shape": "ExplainerConfig", "type": "structure"},
            {"name": "ShadowProductionVariants", "shape": "ProductionVariantList", "type": "list"},
            {"name": "ExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
        ],
        "type": "structure",
    },
    "DescribeEndpointInput": {
        "members": [{"name": "EndpointName", "shape": "EndpointName", "type": "string"}],
        "type": "structure",
    },
    "DescribeEndpointOutput": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "EndpointArn", "shape": "EndpointArn", "type": "string"},
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {"name": "ProductionVariants", "shape": "ProductionVariantSummaryList", "type": "list"},
            {"name": "DataCaptureConfig", "shape": "DataCaptureConfigSummary", "type": "structure"},
            {"name": "EndpointStatus", "shape": "EndpointStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastDeploymentConfig", "shape": "DeploymentConfig", "type": "structure"},
            {"name": "AsyncInferenceConfig", "shape": "AsyncInferenceConfig", "type": "structure"},
            {
                "name": "PendingDeploymentSummary",
                "shape": "PendingDeploymentSummary",
                "type": "structure",
            },
            {"name": "ExplainerConfig", "shape": "ExplainerConfig", "type": "structure"},
            {
                "name": "ShadowProductionVariants",
                "shape": "ProductionVariantSummaryList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "DescribeExperimentRequest": {
        "members": [{"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"}],
        "type": "structure",
    },
    "DescribeExperimentResponse": {
        "members": [
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "ExperimentArn", "shape": "ExperimentArn", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "ExperimentSource", "type": "structure"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeFeatureGroupRequest": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupNameOrArn", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeFeatureGroupResponse": {
        "members": [
            {"name": "FeatureGroupArn", "shape": "FeatureGroupArn", "type": "string"},
            {"name": "FeatureGroupName", "shape": "FeatureGroupName", "type": "string"},
            {"name": "RecordIdentifierFeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "EventTimeFeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "FeatureDefinitions", "shape": "FeatureDefinitions", "type": "list"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "OnlineStoreConfig", "shape": "OnlineStoreConfig", "type": "structure"},
            {"name": "OfflineStoreConfig", "shape": "OfflineStoreConfig", "type": "structure"},
            {
                "name": "ThroughputConfig",
                "shape": "ThroughputConfigDescription",
                "type": "structure",
            },
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "FeatureGroupStatus", "shape": "FeatureGroupStatus", "type": "string"},
            {"name": "OfflineStoreStatus", "shape": "OfflineStoreStatus", "type": "structure"},
            {"name": "LastUpdateStatus", "shape": "LastUpdateStatus", "type": "structure"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "Description", "shape": "Description", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {
                "name": "OnlineStoreTotalSizeBytes",
                "shape": "OnlineStoreTotalSizeBytes",
                "type": "long",
            },
        ],
        "type": "structure",
    },
    "DescribeFeatureMetadataRequest": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupNameOrArn", "type": "string"},
            {"name": "FeatureName", "shape": "FeatureName", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeFeatureMetadataResponse": {
        "members": [
            {"name": "FeatureGroupArn", "shape": "FeatureGroupArn", "type": "string"},
            {"name": "FeatureGroupName", "shape": "FeatureGroupName", "type": "string"},
            {"name": "FeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "FeatureType", "shape": "FeatureType", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "Description", "shape": "FeatureDescription", "type": "string"},
            {"name": "Parameters", "shape": "FeatureParameters", "type": "list"},
        ],
        "type": "structure",
    },
    "DescribeFlowDefinitionRequest": {
        "members": [
            {"name": "FlowDefinitionName", "shape": "FlowDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeFlowDefinitionResponse": {
        "members": [
            {"name": "FlowDefinitionArn", "shape": "FlowDefinitionArn", "type": "string"},
            {"name": "FlowDefinitionName", "shape": "FlowDefinitionName", "type": "string"},
            {"name": "FlowDefinitionStatus", "shape": "FlowDefinitionStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "HumanLoopRequestSource",
                "shape": "HumanLoopRequestSource",
                "type": "structure",
            },
            {
                "name": "HumanLoopActivationConfig",
                "shape": "HumanLoopActivationConfig",
                "type": "structure",
            },
            {"name": "HumanLoopConfig", "shape": "HumanLoopConfig", "type": "structure"},
            {"name": "OutputConfig", "shape": "FlowDefinitionOutputConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeHubContentRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "HubContentVersion", "shape": "HubContentVersion", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeHubContentResponse": {
        "members": [
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "HubContentArn", "shape": "HubContentArn", "type": "string"},
            {"name": "HubContentVersion", "shape": "HubContentVersion", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "DocumentSchemaVersion", "shape": "DocumentSchemaVersion", "type": "string"},
            {"name": "HubName", "shape": "HubName", "type": "string"},
            {"name": "HubArn", "shape": "HubArn", "type": "string"},
            {"name": "HubContentDisplayName", "shape": "HubContentDisplayName", "type": "string"},
            {"name": "HubContentDescription", "shape": "HubContentDescription", "type": "string"},
            {"name": "HubContentMarkdown", "shape": "HubContentMarkdown", "type": "string"},
            {"name": "HubContentDocument", "shape": "HubContentDocument", "type": "string"},
            {
                "name": "SageMakerPublicHubContentArn",
                "shape": "SageMakerPublicHubContentArn",
                "type": "string",
            },
            {"name": "ReferenceMinVersion", "shape": "ReferenceMinVersion", "type": "string"},
            {"name": "SupportStatus", "shape": "HubContentSupportStatus", "type": "string"},
            {
                "name": "HubContentSearchKeywords",
                "shape": "HubContentSearchKeywordList",
                "type": "list",
            },
            {"name": "HubContentDependencies", "shape": "HubContentDependencyList", "type": "list"},
            {"name": "HubContentStatus", "shape": "HubContentStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DescribeHubRequest": {
        "members": [{"name": "HubName", "shape": "HubNameOrArn", "type": "string"}],
        "type": "structure",
    },
    "DescribeHubResponse": {
        "members": [
            {"name": "HubName", "shape": "HubName", "type": "string"},
            {"name": "HubArn", "shape": "HubArn", "type": "string"},
            {"name": "HubDisplayName", "shape": "HubDisplayName", "type": "string"},
            {"name": "HubDescription", "shape": "HubDescription", "type": "string"},
            {"name": "HubSearchKeywords", "shape": "HubSearchKeywordList", "type": "list"},
            {"name": "S3StorageConfig", "shape": "HubS3StorageConfig", "type": "structure"},
            {"name": "HubStatus", "shape": "HubStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DescribeHumanTaskUiRequest": {
        "members": [{"name": "HumanTaskUiName", "shape": "HumanTaskUiName", "type": "string"}],
        "type": "structure",
    },
    "DescribeHumanTaskUiResponse": {
        "members": [
            {"name": "HumanTaskUiArn", "shape": "HumanTaskUiArn", "type": "string"},
            {"name": "HumanTaskUiName", "shape": "HumanTaskUiName", "type": "string"},
            {"name": "HumanTaskUiStatus", "shape": "HumanTaskUiStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "UiTemplate", "shape": "UiTemplateInfo", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeHyperParameterTuningJobRequest": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "DescribeHyperParameterTuningJobResponse": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            },
            {
                "name": "HyperParameterTuningJobArn",
                "shape": "HyperParameterTuningJobArn",
                "type": "string",
            },
            {
                "name": "HyperParameterTuningJobConfig",
                "shape": "HyperParameterTuningJobConfig",
                "type": "structure",
            },
            {
                "name": "TrainingJobDefinition",
                "shape": "HyperParameterTrainingJobDefinition",
                "type": "structure",
            },
            {
                "name": "TrainingJobDefinitions",
                "shape": "HyperParameterTrainingJobDefinitions",
                "type": "list",
            },
            {
                "name": "HyperParameterTuningJobStatus",
                "shape": "HyperParameterTuningJobStatus",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "HyperParameterTuningEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "TrainingJobStatusCounters",
                "shape": "TrainingJobStatusCounters",
                "type": "structure",
            },
            {
                "name": "ObjectiveStatusCounters",
                "shape": "ObjectiveStatusCounters",
                "type": "structure",
            },
            {
                "name": "BestTrainingJob",
                "shape": "HyperParameterTrainingJobSummary",
                "type": "structure",
            },
            {
                "name": "OverallBestTrainingJob",
                "shape": "HyperParameterTrainingJobSummary",
                "type": "structure",
            },
            {
                "name": "WarmStartConfig",
                "shape": "HyperParameterTuningJobWarmStartConfig",
                "type": "structure",
            },
            {"name": "Autotune", "shape": "Autotune", "type": "structure"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {
                "name": "TuningJobCompletionDetails",
                "shape": "HyperParameterTuningJobCompletionDetails",
                "type": "structure",
            },
            {
                "name": "ConsumedResources",
                "shape": "HyperParameterTuningJobConsumedResources",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeImageRequest": {
        "members": [{"name": "ImageName", "shape": "ImageName", "type": "string"}],
        "type": "structure",
    },
    "DescribeImageResponse": {
        "members": [
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Description", "shape": "ImageDescription", "type": "string"},
            {"name": "DisplayName", "shape": "ImageDisplayName", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ImageArn", "shape": "ImageArn", "type": "string"},
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "ImageStatus", "shape": "ImageStatus", "type": "string"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeImageVersionRequest": {
        "members": [
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "Version", "shape": "ImageVersionNumber", "type": "integer"},
            {"name": "Alias", "shape": "SageMakerImageVersionAlias", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeImageVersionResponse": {
        "members": [
            {"name": "BaseImage", "shape": "ImageBaseImage", "type": "string"},
            {"name": "ContainerImage", "shape": "ImageContainerImage", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ImageArn", "shape": "ImageArn", "type": "string"},
            {"name": "ImageVersionArn", "shape": "ImageVersionArn", "type": "string"},
            {"name": "ImageVersionStatus", "shape": "ImageVersionStatus", "type": "string"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Version", "shape": "ImageVersionNumber", "type": "integer"},
            {"name": "VendorGuidance", "shape": "VendorGuidance", "type": "string"},
            {"name": "JobType", "shape": "JobType", "type": "string"},
            {"name": "MLFramework", "shape": "MLFramework", "type": "string"},
            {"name": "ProgrammingLang", "shape": "ProgrammingLang", "type": "string"},
            {"name": "Processor", "shape": "Processor", "type": "string"},
            {"name": "Horovod", "shape": "Horovod", "type": "boolean"},
            {"name": "ReleaseNotes", "shape": "ReleaseNotes", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeInferenceComponentInput": {
        "members": [
            {"name": "InferenceComponentName", "shape": "InferenceComponentName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeInferenceComponentOutput": {
        "members": [
            {"name": "InferenceComponentName", "shape": "InferenceComponentName", "type": "string"},
            {"name": "InferenceComponentArn", "shape": "InferenceComponentArn", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "EndpointArn", "shape": "EndpointArn", "type": "string"},
            {"name": "VariantName", "shape": "VariantName", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {
                "name": "Specification",
                "shape": "InferenceComponentSpecificationSummary",
                "type": "structure",
            },
            {
                "name": "RuntimeConfig",
                "shape": "InferenceComponentRuntimeConfigSummary",
                "type": "structure",
            },
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "InferenceComponentStatus",
                "shape": "InferenceComponentStatus",
                "type": "string",
            },
            {
                "name": "LastDeploymentConfig",
                "shape": "InferenceComponentDeploymentConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeInferenceExperimentRequest": {
        "members": [{"name": "Name", "shape": "InferenceExperimentName", "type": "string"}],
        "type": "structure",
    },
    "DescribeInferenceExperimentResponse": {
        "members": [
            {"name": "Arn", "shape": "InferenceExperimentArn", "type": "string"},
            {"name": "Name", "shape": "InferenceExperimentName", "type": "string"},
            {"name": "Type", "shape": "InferenceExperimentType", "type": "string"},
            {"name": "Schedule", "shape": "InferenceExperimentSchedule", "type": "structure"},
            {"name": "Status", "shape": "InferenceExperimentStatus", "type": "string"},
            {"name": "StatusReason", "shape": "InferenceExperimentStatusReason", "type": "string"},
            {"name": "Description", "shape": "InferenceExperimentDescription", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CompletionTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "EndpointMetadata", "shape": "EndpointMetadata", "type": "structure"},
            {"name": "ModelVariants", "shape": "ModelVariantConfigSummaryList", "type": "list"},
            {
                "name": "DataStorageConfig",
                "shape": "InferenceExperimentDataStorageConfig",
                "type": "structure",
            },
            {"name": "ShadowModeConfig", "shape": "ShadowModeConfig", "type": "structure"},
            {"name": "KmsKey", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeInferenceRecommendationsJobRequest": {
        "members": [{"name": "JobName", "shape": "RecommendationJobName", "type": "string"}],
        "type": "structure",
    },
    "DescribeInferenceRecommendationsJobResponse": {
        "members": [
            {"name": "JobName", "shape": "RecommendationJobName", "type": "string"},
            {"name": "JobDescription", "shape": "RecommendationJobDescription", "type": "string"},
            {"name": "JobType", "shape": "RecommendationJobType", "type": "string"},
            {"name": "JobArn", "shape": "RecommendationJobArn", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Status", "shape": "RecommendationJobStatus", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CompletionTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "InputConfig", "shape": "RecommendationJobInputConfig", "type": "structure"},
            {
                "name": "StoppingConditions",
                "shape": "RecommendationJobStoppingConditions",
                "type": "structure",
            },
            {
                "name": "InferenceRecommendations",
                "shape": "InferenceRecommendations",
                "type": "list",
            },
            {"name": "EndpointPerformances", "shape": "EndpointPerformances", "type": "list"},
        ],
        "type": "structure",
    },
    "DescribeLabelingJobRequest": {
        "members": [{"name": "LabelingJobName", "shape": "LabelingJobName", "type": "string"}],
        "type": "structure",
    },
    "DescribeLabelingJobResponse": {
        "members": [
            {"name": "LabelingJobStatus", "shape": "LabelingJobStatus", "type": "string"},
            {"name": "LabelCounters", "shape": "LabelCounters", "type": "structure"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "JobReferenceCode", "shape": "JobReferenceCode", "type": "string"},
            {"name": "LabelingJobName", "shape": "LabelingJobName", "type": "string"},
            {"name": "LabelingJobArn", "shape": "LabelingJobArn", "type": "string"},
            {"name": "LabelAttributeName", "shape": "LabelAttributeName", "type": "string"},
            {"name": "InputConfig", "shape": "LabelingJobInputConfig", "type": "structure"},
            {"name": "OutputConfig", "shape": "LabelingJobOutputConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "LabelCategoryConfigS3Uri", "shape": "S3Uri", "type": "string"},
            {
                "name": "StoppingConditions",
                "shape": "LabelingJobStoppingConditions",
                "type": "structure",
            },
            {
                "name": "LabelingJobAlgorithmsConfig",
                "shape": "LabelingJobAlgorithmsConfig",
                "type": "structure",
            },
            {"name": "HumanTaskConfig", "shape": "HumanTaskConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "LabelingJobOutput", "shape": "LabelingJobOutput", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeLineageGroupRequest": {
        "members": [
            {"name": "LineageGroupName", "shape": "ExperimentEntityName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeLineageGroupResponse": {
        "members": [
            {"name": "LineageGroupName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "LineageGroupArn", "shape": "LineageGroupArn", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeMlflowTrackingServerRequest": {
        "members": [
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeMlflowTrackingServerResponse": {
        "members": [
            {"name": "TrackingServerArn", "shape": "TrackingServerArn", "type": "string"},
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"},
            {"name": "ArtifactStoreUri", "shape": "S3Uri", "type": "string"},
            {"name": "TrackingServerSize", "shape": "TrackingServerSize", "type": "string"},
            {"name": "MlflowVersion", "shape": "MlflowVersion", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "TrackingServerStatus", "shape": "TrackingServerStatus", "type": "string"},
            {"name": "IsActive", "shape": "IsTrackingServerActive", "type": "string"},
            {"name": "TrackingServerUrl", "shape": "TrackingServerUrl", "type": "string"},
            {
                "name": "WeeklyMaintenanceWindowStart",
                "shape": "WeeklyMaintenanceWindowStart",
                "type": "string",
            },
            {"name": "AutomaticModelRegistration", "shape": "Boolean", "type": "boolean"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeModelBiasJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeModelBiasJobDefinitionResponse": {
        "members": [
            {"name": "JobDefinitionArn", "shape": "MonitoringJobDefinitionArn", "type": "string"},
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "ModelBiasBaselineConfig",
                "shape": "ModelBiasBaselineConfig",
                "type": "structure",
            },
            {
                "name": "ModelBiasAppSpecification",
                "shape": "ModelBiasAppSpecification",
                "type": "structure",
            },
            {"name": "ModelBiasJobInput", "shape": "ModelBiasJobInput", "type": "structure"},
            {
                "name": "ModelBiasJobOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "JobResources", "shape": "MonitoringResources", "type": "structure"},
            {"name": "NetworkConfig", "shape": "MonitoringNetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeModelCardExportJobRequest": {
        "members": [
            {"name": "ModelCardExportJobArn", "shape": "ModelCardExportJobArn", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeModelCardExportJobResponse": {
        "members": [
            {"name": "ModelCardExportJobName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardExportJobArn", "shape": "ModelCardExportJobArn", "type": "string"},
            {"name": "Status", "shape": "ModelCardExportJobStatus", "type": "string"},
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
            {"name": "OutputConfig", "shape": "ModelCardExportOutputConfig", "type": "structure"},
            {"name": "CreatedAt", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedAt", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ExportArtifacts", "shape": "ModelCardExportArtifacts", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeModelCardRequest": {
        "members": [
            {"name": "ModelCardName", "shape": "ModelCardNameOrArn", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "DescribeModelCardResponse": {
        "members": [
            {"name": "ModelCardArn", "shape": "ModelCardArn", "type": "string"},
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
            {"name": "Content", "shape": "ModelCardContent", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
            {"name": "SecurityConfig", "shape": "ModelCardSecurityConfig", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {
                "name": "ModelCardProcessingStatus",
                "shape": "ModelCardProcessingStatus",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "DescribeModelExplainabilityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeModelExplainabilityJobDefinitionResponse": {
        "members": [
            {"name": "JobDefinitionArn", "shape": "MonitoringJobDefinitionArn", "type": "string"},
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "ModelExplainabilityBaselineConfig",
                "shape": "ModelExplainabilityBaselineConfig",
                "type": "structure",
            },
            {
                "name": "ModelExplainabilityAppSpecification",
                "shape": "ModelExplainabilityAppSpecification",
                "type": "structure",
            },
            {
                "name": "ModelExplainabilityJobInput",
                "shape": "ModelExplainabilityJobInput",
                "type": "structure",
            },
            {
                "name": "ModelExplainabilityJobOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "JobResources", "shape": "MonitoringResources", "type": "structure"},
            {"name": "NetworkConfig", "shape": "MonitoringNetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeModelInput": {
        "members": [{"name": "ModelName", "shape": "ModelName", "type": "string"}],
        "type": "structure",
    },
    "DescribeModelOutput": {
        "members": [
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "PrimaryContainer", "shape": "ContainerDefinition", "type": "structure"},
            {"name": "Containers", "shape": "ContainerDefinitionList", "type": "list"},
            {
                "name": "InferenceExecutionConfig",
                "shape": "InferenceExecutionConfig",
                "type": "structure",
            },
            {"name": "ExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModelArn", "shape": "ModelArn", "type": "string"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
            {
                "name": "DeploymentRecommendation",
                "shape": "DeploymentRecommendation",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeModelPackageGroupInput": {
        "members": [{"name": "ModelPackageGroupName", "shape": "ArnOrName", "type": "string"}],
        "type": "structure",
    },
    "DescribeModelPackageGroupOutput": {
        "members": [
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageGroupArn", "shape": "ModelPackageGroupArn", "type": "string"},
            {
                "name": "ModelPackageGroupDescription",
                "shape": "EntityDescription",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {
                "name": "ModelPackageGroupStatus",
                "shape": "ModelPackageGroupStatus",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "DescribeModelPackageInput": {
        "members": [{"name": "ModelPackageName", "shape": "VersionedArnOrName", "type": "string"}],
        "type": "structure",
    },
    "DescribeModelPackageOutput": {
        "members": [
            {"name": "ModelPackageName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageVersion", "shape": "ModelPackageVersion", "type": "integer"},
            {"name": "ModelPackageArn", "shape": "ModelPackageArn", "type": "string"},
            {"name": "ModelPackageDescription", "shape": "EntityDescription", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {
                "name": "InferenceSpecification",
                "shape": "InferenceSpecification",
                "type": "structure",
            },
            {
                "name": "SourceAlgorithmSpecification",
                "shape": "SourceAlgorithmSpecification",
                "type": "structure",
            },
            {
                "name": "ValidationSpecification",
                "shape": "ModelPackageValidationSpecification",
                "type": "structure",
            },
            {"name": "ModelPackageStatus", "shape": "ModelPackageStatus", "type": "string"},
            {
                "name": "ModelPackageStatusDetails",
                "shape": "ModelPackageStatusDetails",
                "type": "structure",
            },
            {"name": "CertifyForMarketplace", "shape": "CertifyForMarketplace", "type": "boolean"},
            {"name": "ModelApprovalStatus", "shape": "ModelApprovalStatus", "type": "string"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "ModelMetrics", "shape": "ModelMetrics", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "ApprovalDescription", "shape": "ApprovalDescription", "type": "string"},
            {"name": "Domain", "shape": "String", "type": "string"},
            {"name": "Task", "shape": "String", "type": "string"},
            {"name": "SamplePayloadUrl", "shape": "String", "type": "string"},
            {"name": "CustomerMetadataProperties", "shape": "CustomerMetadataMap", "type": "map"},
            {"name": "DriftCheckBaselines", "shape": "DriftCheckBaselines", "type": "structure"},
            {
                "name": "AdditionalInferenceSpecifications",
                "shape": "AdditionalInferenceSpecifications",
                "type": "list",
            },
            {"name": "SkipModelValidation", "shape": "SkipModelValidation", "type": "string"},
            {"name": "SourceUri", "shape": "ModelPackageSourceUri", "type": "string"},
            {"name": "SecurityConfig", "shape": "ModelPackageSecurityConfig", "type": "structure"},
            {"name": "ModelCard", "shape": "ModelPackageModelCard", "type": "structure"},
            {"name": "ModelLifeCycle", "shape": "ModelLifeCycle", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeModelQualityJobDefinitionRequest": {
        "members": [
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeModelQualityJobDefinitionResponse": {
        "members": [
            {"name": "JobDefinitionArn", "shape": "MonitoringJobDefinitionArn", "type": "string"},
            {"name": "JobDefinitionName", "shape": "MonitoringJobDefinitionName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "ModelQualityBaselineConfig",
                "shape": "ModelQualityBaselineConfig",
                "type": "structure",
            },
            {
                "name": "ModelQualityAppSpecification",
                "shape": "ModelQualityAppSpecification",
                "type": "structure",
            },
            {"name": "ModelQualityJobInput", "shape": "ModelQualityJobInput", "type": "structure"},
            {
                "name": "ModelQualityJobOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "JobResources", "shape": "MonitoringResources", "type": "structure"},
            {"name": "NetworkConfig", "shape": "MonitoringNetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeMonitoringScheduleRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeMonitoringScheduleResponse": {
        "members": [
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"},
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "MonitoringScheduleStatus", "shape": "ScheduleStatus", "type": "string"},
            {"name": "MonitoringType", "shape": "MonitoringType", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "MonitoringScheduleConfig",
                "shape": "MonitoringScheduleConfig",
                "type": "structure",
            },
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {
                "name": "LastMonitoringExecutionSummary",
                "shape": "MonitoringExecutionSummary",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeNotebookInstanceInput": {
        "members": [
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeNotebookInstanceLifecycleConfigInput": {
        "members": [
            {
                "name": "NotebookInstanceLifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "DescribeNotebookInstanceLifecycleConfigOutput": {
        "members": [
            {
                "name": "NotebookInstanceLifecycleConfigArn",
                "shape": "NotebookInstanceLifecycleConfigArn",
                "type": "string",
            },
            {
                "name": "NotebookInstanceLifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {"name": "OnCreate", "shape": "NotebookInstanceLifecycleConfigList", "type": "list"},
            {"name": "OnStart", "shape": "NotebookInstanceLifecycleConfigList", "type": "list"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DescribeNotebookInstanceOutput": {
        "members": [
            {"name": "NotebookInstanceArn", "shape": "NotebookInstanceArn", "type": "string"},
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"},
            {"name": "NotebookInstanceStatus", "shape": "NotebookInstanceStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "Url", "shape": "NotebookInstanceUrl", "type": "string"},
            {"name": "InstanceType", "shape": "InstanceType", "type": "string"},
            {"name": "SubnetId", "shape": "SubnetId", "type": "string"},
            {"name": "SecurityGroups", "shape": "SecurityGroupIds", "type": "list"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "NetworkInterfaceId", "shape": "NetworkInterfaceId", "type": "string"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {
                "name": "NotebookInstanceLifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {"name": "DirectInternetAccess", "shape": "DirectInternetAccess", "type": "string"},
            {
                "name": "VolumeSizeInGB",
                "shape": "NotebookInstanceVolumeSizeInGB",
                "type": "integer",
            },
            {
                "name": "AcceleratorTypes",
                "shape": "NotebookInstanceAcceleratorTypes",
                "type": "list",
            },
            {"name": "DefaultCodeRepository", "shape": "CodeRepositoryNameOrUrl", "type": "string"},
            {
                "name": "AdditionalCodeRepositories",
                "shape": "AdditionalCodeRepositoryNamesOrUrls",
                "type": "list",
            },
            {"name": "RootAccess", "shape": "RootAccess", "type": "string"},
            {"name": "PlatformIdentifier", "shape": "PlatformIdentifier", "type": "string"},
            {
                "name": "InstanceMetadataServiceConfiguration",
                "shape": "InstanceMetadataServiceConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeOptimizationJobRequest": {
        "members": [{"name": "OptimizationJobName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "DescribeOptimizationJobResponse": {
        "members": [
            {"name": "OptimizationJobArn", "shape": "OptimizationJobArn", "type": "string"},
            {"name": "OptimizationJobStatus", "shape": "OptimizationJobStatus", "type": "string"},
            {"name": "OptimizationStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "OptimizationEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "OptimizationJobName", "shape": "EntityName", "type": "string"},
            {"name": "ModelSource", "shape": "OptimizationJobModelSource", "type": "structure"},
            {
                "name": "OptimizationEnvironment",
                "shape": "OptimizationJobEnvironmentVariables",
                "type": "map",
            },
            {
                "name": "DeploymentInstanceType",
                "shape": "OptimizationJobDeploymentInstanceType",
                "type": "string",
            },
            {"name": "OptimizationConfigs", "shape": "OptimizationConfigs", "type": "list"},
            {"name": "OutputConfig", "shape": "OptimizationJobOutputConfig", "type": "structure"},
            {"name": "OptimizationOutput", "shape": "OptimizationOutput", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
            {"name": "VpcConfig", "shape": "OptimizationVpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribePartnerAppRequest": {
        "members": [{"name": "Arn", "shape": "PartnerAppArn", "type": "string"}],
        "type": "structure",
    },
    "DescribePartnerAppResponse": {
        "members": [
            {"name": "Arn", "shape": "PartnerAppArn", "type": "string"},
            {"name": "Name", "shape": "PartnerAppName", "type": "string"},
            {"name": "Type", "shape": "PartnerAppType", "type": "string"},
            {"name": "Status", "shape": "PartnerAppStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "BaseUrl", "shape": "String2048", "type": "string"},
            {
                "name": "MaintenanceConfig",
                "shape": "PartnerAppMaintenanceConfig",
                "type": "structure",
            },
            {"name": "Tier", "shape": "NonEmptyString64", "type": "string"},
            {"name": "Version", "shape": "NonEmptyString64", "type": "string"},
            {"name": "ApplicationConfig", "shape": "PartnerAppConfig", "type": "structure"},
            {"name": "AuthType", "shape": "PartnerAppAuthType", "type": "string"},
            {"name": "EnableIamSessionBasedIdentity", "shape": "Boolean", "type": "boolean"},
            {"name": "Error", "shape": "ErrorInfo", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribePipelineDefinitionForExecutionRequest": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribePipelineDefinitionForExecutionResponse": {
        "members": [
            {"name": "PipelineDefinition", "shape": "PipelineDefinition", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DescribePipelineExecutionRequest": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribePipelineExecutionResponse": {
        "members": [
            {"name": "PipelineArn", "shape": "PipelineArn", "type": "string"},
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"},
            {
                "name": "PipelineExecutionDisplayName",
                "shape": "PipelineExecutionName",
                "type": "string",
            },
            {
                "name": "PipelineExecutionStatus",
                "shape": "PipelineExecutionStatus",
                "type": "string",
            },
            {
                "name": "PipelineExecutionDescription",
                "shape": "PipelineExecutionDescription",
                "type": "string",
            },
            {
                "name": "PipelineExperimentConfig",
                "shape": "PipelineExperimentConfig",
                "type": "structure",
            },
            {"name": "FailureReason", "shape": "PipelineExecutionFailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
            {
                "name": "SelectiveExecutionConfig",
                "shape": "SelectiveExecutionConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribePipelineRequest": {
        "members": [{"name": "PipelineName", "shape": "PipelineNameOrArn", "type": "string"}],
        "type": "structure",
    },
    "DescribePipelineResponse": {
        "members": [
            {"name": "PipelineArn", "shape": "PipelineArn", "type": "string"},
            {"name": "PipelineName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDisplayName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDefinition", "shape": "PipelineDefinition", "type": "string"},
            {"name": "PipelineDescription", "shape": "PipelineDescription", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "PipelineStatus", "shape": "PipelineStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastRunTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DescribeProcessingJobRequest": {
        "members": [{"name": "ProcessingJobName", "shape": "ProcessingJobName", "type": "string"}],
        "type": "structure",
    },
    "DescribeProcessingJobResponse": {
        "members": [
            {"name": "ProcessingInputs", "shape": "ProcessingInputs", "type": "list"},
            {
                "name": "ProcessingOutputConfig",
                "shape": "ProcessingOutputConfig",
                "type": "structure",
            },
            {"name": "ProcessingJobName", "shape": "ProcessingJobName", "type": "string"},
            {"name": "ProcessingResources", "shape": "ProcessingResources", "type": "structure"},
            {
                "name": "StoppingCondition",
                "shape": "ProcessingStoppingCondition",
                "type": "structure",
            },
            {"name": "AppSpecification", "shape": "AppSpecification", "type": "structure"},
            {"name": "Environment", "shape": "ProcessingEnvironmentMap", "type": "map"},
            {"name": "NetworkConfig", "shape": "NetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
            {"name": "ProcessingJobArn", "shape": "ProcessingJobArn", "type": "string"},
            {"name": "ProcessingJobStatus", "shape": "ProcessingJobStatus", "type": "string"},
            {"name": "ExitMessage", "shape": "ExitMessage", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ProcessingEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ProcessingStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {"name": "TrainingJobArn", "shape": "TrainingJobArn", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeProjectInput": {
        "members": [{"name": "ProjectName", "shape": "ProjectEntityName", "type": "string"}],
        "type": "structure",
    },
    "DescribeProjectOutput": {
        "members": [
            {"name": "ProjectArn", "shape": "ProjectArn", "type": "string"},
            {"name": "ProjectName", "shape": "ProjectEntityName", "type": "string"},
            {"name": "ProjectId", "shape": "ProjectId", "type": "string"},
            {"name": "ProjectDescription", "shape": "EntityDescription", "type": "string"},
            {
                "name": "ServiceCatalogProvisioningDetails",
                "shape": "ServiceCatalogProvisioningDetails",
                "type": "structure",
            },
            {
                "name": "ServiceCatalogProvisionedProductDetails",
                "shape": "ServiceCatalogProvisionedProductDetails",
                "type": "structure",
            },
            {"name": "ProjectStatus", "shape": "ProjectStatus", "type": "string"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeSpaceRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeSpaceResponse": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "SpaceArn", "shape": "SpaceArn", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "HomeEfsFileSystemUid", "shape": "EfsUid", "type": "string"},
            {"name": "Status", "shape": "SpaceStatus", "type": "string"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "SpaceSettings", "shape": "SpaceSettings", "type": "structure"},
            {"name": "OwnershipSettings", "shape": "OwnershipSettings", "type": "structure"},
            {"name": "SpaceSharingSettings", "shape": "SpaceSharingSettings", "type": "structure"},
            {"name": "SpaceDisplayName", "shape": "NonEmptyString64", "type": "string"},
            {"name": "Url", "shape": "String1024", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeStudioLifecycleConfigRequest": {
        "members": [
            {
                "name": "StudioLifecycleConfigName",
                "shape": "StudioLifecycleConfigName",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "DescribeStudioLifecycleConfigResponse": {
        "members": [
            {
                "name": "StudioLifecycleConfigArn",
                "shape": "StudioLifecycleConfigArn",
                "type": "string",
            },
            {
                "name": "StudioLifecycleConfigName",
                "shape": "StudioLifecycleConfigName",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "StudioLifecycleConfigContent",
                "shape": "StudioLifecycleConfigContent",
                "type": "string",
            },
            {
                "name": "StudioLifecycleConfigAppType",
                "shape": "StudioLifecycleConfigAppType",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "DescribeSubscribedWorkteamRequest": {
        "members": [{"name": "WorkteamArn", "shape": "WorkteamArn", "type": "string"}],
        "type": "structure",
    },
    "DescribeSubscribedWorkteamResponse": {
        "members": [
            {"name": "SubscribedWorkteam", "shape": "SubscribedWorkteam", "type": "structure"}
        ],
        "type": "structure",
    },
    "DescribeTrainingJobRequest": {
        "members": [{"name": "TrainingJobName", "shape": "TrainingJobName", "type": "string"}],
        "type": "structure",
    },
    "DescribeTrainingJobResponse": {
        "members": [
            {"name": "TrainingJobName", "shape": "TrainingJobName", "type": "string"},
            {"name": "TrainingJobArn", "shape": "TrainingJobArn", "type": "string"},
            {"name": "TuningJobArn", "shape": "HyperParameterTuningJobArn", "type": "string"},
            {"name": "LabelingJobArn", "shape": "LabelingJobArn", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {"name": "ModelArtifacts", "shape": "ModelArtifacts", "type": "structure"},
            {"name": "TrainingJobStatus", "shape": "TrainingJobStatus", "type": "string"},
            {"name": "SecondaryStatus", "shape": "SecondaryStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "HyperParameters", "shape": "HyperParameters", "type": "map"},
            {
                "name": "AlgorithmSpecification",
                "shape": "AlgorithmSpecification",
                "type": "structure",
            },
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "InputDataConfig", "shape": "InputDataConfig", "type": "list"},
            {"name": "OutputDataConfig", "shape": "OutputDataConfig", "type": "structure"},
            {"name": "ResourceConfig", "shape": "ResourceConfig", "type": "structure"},
            {"name": "WarmPoolStatus", "shape": "WarmPoolStatus", "type": "structure"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "SecondaryStatusTransitions",
                "shape": "SecondaryStatusTransitions",
                "type": "list",
            },
            {"name": "FinalMetricDataList", "shape": "FinalMetricDataList", "type": "list"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
            {
                "name": "EnableInterContainerTrafficEncryption",
                "shape": "Boolean",
                "type": "boolean",
            },
            {"name": "EnableManagedSpotTraining", "shape": "Boolean", "type": "boolean"},
            {"name": "CheckpointConfig", "shape": "CheckpointConfig", "type": "structure"},
            {"name": "TrainingTimeInSeconds", "shape": "TrainingTimeInSeconds", "type": "integer"},
            {"name": "BillableTimeInSeconds", "shape": "BillableTimeInSeconds", "type": "integer"},
            {"name": "DebugHookConfig", "shape": "DebugHookConfig", "type": "structure"},
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
            {"name": "DebugRuleConfigurations", "shape": "DebugRuleConfigurations", "type": "list"},
            {
                "name": "TensorBoardOutputConfig",
                "shape": "TensorBoardOutputConfig",
                "type": "structure",
            },
            {
                "name": "DebugRuleEvaluationStatuses",
                "shape": "DebugRuleEvaluationStatuses",
                "type": "list",
            },
            {"name": "ProfilerConfig", "shape": "ProfilerConfig", "type": "structure"},
            {
                "name": "ProfilerRuleConfigurations",
                "shape": "ProfilerRuleConfigurations",
                "type": "list",
            },
            {
                "name": "ProfilerRuleEvaluationStatuses",
                "shape": "ProfilerRuleEvaluationStatuses",
                "type": "list",
            },
            {"name": "ProfilingStatus", "shape": "ProfilingStatus", "type": "string"},
            {"name": "Environment", "shape": "TrainingEnvironmentMap", "type": "map"},
            {"name": "RetryStrategy", "shape": "RetryStrategy", "type": "structure"},
            {"name": "RemoteDebugConfig", "shape": "RemoteDebugConfig", "type": "structure"},
            {"name": "InfraCheckConfig", "shape": "InfraCheckConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeTrainingPlanRequest": {
        "members": [{"name": "TrainingPlanName", "shape": "TrainingPlanName", "type": "string"}],
        "type": "structure",
    },
    "DescribeTrainingPlanResponse": {
        "members": [
            {"name": "TrainingPlanArn", "shape": "TrainingPlanArn", "type": "string"},
            {"name": "TrainingPlanName", "shape": "TrainingPlanName", "type": "string"},
            {"name": "Status", "shape": "TrainingPlanStatus", "type": "string"},
            {"name": "StatusMessage", "shape": "TrainingPlanStatusMessage", "type": "string"},
            {"name": "DurationHours", "shape": "TrainingPlanDurationHours", "type": "long"},
            {"name": "DurationMinutes", "shape": "TrainingPlanDurationMinutes", "type": "long"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "UpfrontFee", "shape": "String256", "type": "string"},
            {"name": "CurrencyCode", "shape": "CurrencyCode", "type": "string"},
            {"name": "TotalInstanceCount", "shape": "TotalInstanceCount", "type": "integer"},
            {
                "name": "AvailableInstanceCount",
                "shape": "AvailableInstanceCount",
                "type": "integer",
            },
            {"name": "InUseInstanceCount", "shape": "InUseInstanceCount", "type": "integer"},
            {"name": "TargetResources", "shape": "SageMakerResourceNames", "type": "list"},
            {
                "name": "ReservedCapacitySummaries",
                "shape": "ReservedCapacitySummaries",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "DescribeTransformJobRequest": {
        "members": [{"name": "TransformJobName", "shape": "TransformJobName", "type": "string"}],
        "type": "structure",
    },
    "DescribeTransformJobResponse": {
        "members": [
            {"name": "TransformJobName", "shape": "TransformJobName", "type": "string"},
            {"name": "TransformJobArn", "shape": "TransformJobArn", "type": "string"},
            {"name": "TransformJobStatus", "shape": "TransformJobStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {
                "name": "MaxConcurrentTransforms",
                "shape": "MaxConcurrentTransforms",
                "type": "integer",
            },
            {"name": "ModelClientConfig", "shape": "ModelClientConfig", "type": "structure"},
            {"name": "MaxPayloadInMB", "shape": "MaxPayloadInMB", "type": "integer"},
            {"name": "BatchStrategy", "shape": "BatchStrategy", "type": "string"},
            {"name": "Environment", "shape": "TransformEnvironmentMap", "type": "map"},
            {"name": "TransformInput", "shape": "TransformInput", "type": "structure"},
            {"name": "TransformOutput", "shape": "TransformOutput", "type": "structure"},
            {"name": "DataCaptureConfig", "shape": "BatchDataCaptureConfig", "type": "structure"},
            {"name": "TransformResources", "shape": "TransformResources", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TransformStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TransformEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LabelingJobArn", "shape": "LabelingJobArn", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {"name": "DataProcessing", "shape": "DataProcessing", "type": "structure"},
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeTrialComponentRequest": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityNameOrArn", "type": "string"}
        ],
        "type": "structure",
    },
    "DescribeTrialComponentResponse": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "TrialComponentSource", "type": "structure"},
            {"name": "Status", "shape": "TrialComponentStatus", "type": "structure"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "Parameters", "shape": "TrialComponentParameters", "type": "map"},
            {"name": "InputArtifacts", "shape": "TrialComponentArtifacts", "type": "map"},
            {"name": "OutputArtifacts", "shape": "TrialComponentArtifacts", "type": "map"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "Metrics", "shape": "TrialComponentMetricSummaries", "type": "list"},
            {"name": "LineageGroupArn", "shape": "LineageGroupArn", "type": "string"},
            {"name": "Sources", "shape": "TrialComponentSources", "type": "list"},
        ],
        "type": "structure",
    },
    "DescribeTrialRequest": {
        "members": [{"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"}],
        "type": "structure",
    },
    "DescribeTrialResponse": {
        "members": [
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialArn", "shape": "TrialArn", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "TrialSource", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeUserProfileRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
        ],
        "type": "structure",
    },
    "DescribeUserProfileResponse": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileArn", "shape": "UserProfileArn", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {"name": "HomeEfsFileSystemUid", "shape": "EfsUid", "type": "string"},
            {"name": "Status", "shape": "UserProfileStatus", "type": "string"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {
                "name": "SingleSignOnUserIdentifier",
                "shape": "SingleSignOnUserIdentifier",
                "type": "string",
            },
            {"name": "SingleSignOnUserValue", "shape": "String256", "type": "string"},
            {"name": "UserSettings", "shape": "UserSettings", "type": "structure"},
        ],
        "type": "structure",
    },
    "DescribeWorkforceRequest": {
        "members": [{"name": "WorkforceName", "shape": "WorkforceName", "type": "string"}],
        "type": "structure",
    },
    "DescribeWorkforceResponse": {
        "members": [{"name": "Workforce", "shape": "Workforce", "type": "structure"}],
        "type": "structure",
    },
    "DescribeWorkteamRequest": {
        "members": [{"name": "WorkteamName", "shape": "WorkteamName", "type": "string"}],
        "type": "structure",
    },
    "DescribeWorkteamResponse": {
        "members": [{"name": "Workteam", "shape": "Workteam", "type": "structure"}],
        "type": "structure",
    },
    "DesiredWeightAndCapacity": {
        "members": [
            {"name": "VariantName", "shape": "VariantName", "type": "string"},
            {"name": "DesiredWeight", "shape": "VariantWeight", "type": "float"},
            {"name": "DesiredInstanceCount", "shape": "TaskCount", "type": "integer"},
            {
                "name": "ServerlessUpdateConfig",
                "shape": "ProductionVariantServerlessUpdateConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DesiredWeightAndCapacityList": {
        "member_shape": "DesiredWeightAndCapacity",
        "member_type": "structure",
        "type": "list",
    },
    "Device": {
        "members": [
            {"name": "DeviceName", "shape": "DeviceName", "type": "string"},
            {"name": "Description", "shape": "DeviceDescription", "type": "string"},
            {"name": "IotThingName", "shape": "ThingName", "type": "string"},
        ],
        "type": "structure",
    },
    "DeviceDeploymentSummaries": {
        "member_shape": "DeviceDeploymentSummary",
        "member_type": "structure",
        "type": "list",
    },
    "DeviceDeploymentSummary": {
        "members": [
            {"name": "EdgeDeploymentPlanArn", "shape": "EdgeDeploymentPlanArn", "type": "string"},
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "StageName", "shape": "EntityName", "type": "string"},
            {"name": "DeployedStageName", "shape": "EntityName", "type": "string"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "DeviceName", "shape": "DeviceName", "type": "string"},
            {"name": "DeviceArn", "shape": "DeviceArn", "type": "string"},
            {"name": "DeviceDeploymentStatus", "shape": "DeviceDeploymentStatus", "type": "string"},
            {"name": "DeviceDeploymentStatusMessage", "shape": "String", "type": "string"},
            {"name": "Description", "shape": "DeviceDescription", "type": "string"},
            {"name": "DeploymentStartTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DeviceFleetSummaries": {
        "member_shape": "DeviceFleetSummary",
        "member_type": "structure",
        "type": "list",
    },
    "DeviceFleetSummary": {
        "members": [
            {"name": "DeviceFleetArn", "shape": "DeviceFleetArn", "type": "string"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "DeviceNames": {"member_shape": "DeviceName", "member_type": "string", "type": "list"},
    "DeviceSelectionConfig": {
        "members": [
            {"name": "DeviceSubsetType", "shape": "DeviceSubsetType", "type": "string"},
            {"name": "Percentage", "shape": "Percentage", "type": "integer"},
            {"name": "DeviceNames", "shape": "DeviceNames", "type": "list"},
            {"name": "DeviceNameContains", "shape": "DeviceName", "type": "string"},
        ],
        "type": "structure",
    },
    "DeviceStats": {
        "members": [
            {"name": "ConnectedDeviceCount", "shape": "Long", "type": "long"},
            {"name": "RegisteredDeviceCount", "shape": "Long", "type": "long"},
        ],
        "type": "structure",
    },
    "DeviceSummaries": {
        "member_shape": "DeviceSummary",
        "member_type": "structure",
        "type": "list",
    },
    "DeviceSummary": {
        "members": [
            {"name": "DeviceName", "shape": "EntityName", "type": "string"},
            {"name": "DeviceArn", "shape": "DeviceArn", "type": "string"},
            {"name": "Description", "shape": "DeviceDescription", "type": "string"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "IotThingName", "shape": "ThingName", "type": "string"},
            {"name": "RegistrationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LatestHeartbeat", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Models", "shape": "EdgeModelSummaries", "type": "list"},
            {"name": "AgentVersion", "shape": "EdgeVersion", "type": "string"},
        ],
        "type": "structure",
    },
    "Devices": {"member_shape": "Device", "member_type": "structure", "type": "list"},
    "DirectDeploySettings": {
        "members": [{"name": "Status", "shape": "FeatureStatus", "type": "string"}],
        "type": "structure",
    },
    "DisableSagemakerServicecatalogPortfolioInput": {"members": [], "type": "structure"},
    "DisableSagemakerServicecatalogPortfolioOutput": {"members": [], "type": "structure"},
    "DisassociateTrialComponentRequest": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "DisassociateTrialComponentResponse": {
        "members": [
            {"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"},
            {"name": "TrialArn", "shape": "TrialArn", "type": "string"},
        ],
        "type": "structure",
    },
    "DockerSettings": {
        "members": [
            {"name": "EnableDockerAccess", "shape": "FeatureStatus", "type": "string"},
            {"name": "VpcOnlyTrustedAccounts", "shape": "VpcOnlyTrustedAccounts", "type": "list"},
        ],
        "type": "structure",
    },
    "DomainDetails": {
        "members": [
            {"name": "DomainArn", "shape": "DomainArn", "type": "string"},
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "DomainName", "shape": "DomainName", "type": "string"},
            {"name": "Status", "shape": "DomainStatus", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "Url", "shape": "String1024", "type": "string"},
        ],
        "type": "structure",
    },
    "DomainList": {"member_shape": "DomainDetails", "member_type": "structure", "type": "list"},
    "DomainSecurityGroupIds": {
        "member_shape": "SecurityGroupId",
        "member_type": "string",
        "type": "list",
    },
    "DomainSettings": {
        "members": [
            {"name": "SecurityGroupIds", "shape": "DomainSecurityGroupIds", "type": "list"},
            {
                "name": "RStudioServerProDomainSettings",
                "shape": "RStudioServerProDomainSettings",
                "type": "structure",
            },
            {
                "name": "ExecutionRoleIdentityConfig",
                "shape": "ExecutionRoleIdentityConfig",
                "type": "string",
            },
            {"name": "DockerSettings", "shape": "DockerSettings", "type": "structure"},
            {"name": "AmazonQSettings", "shape": "AmazonQSettings", "type": "structure"},
        ],
        "type": "structure",
    },
    "DomainSettingsForUpdate": {
        "members": [
            {
                "name": "RStudioServerProDomainSettingsForUpdate",
                "shape": "RStudioServerProDomainSettingsForUpdate",
                "type": "structure",
            },
            {
                "name": "ExecutionRoleIdentityConfig",
                "shape": "ExecutionRoleIdentityConfig",
                "type": "string",
            },
            {"name": "SecurityGroupIds", "shape": "DomainSecurityGroupIds", "type": "list"},
            {"name": "DockerSettings", "shape": "DockerSettings", "type": "structure"},
            {"name": "AmazonQSettings", "shape": "AmazonQSettings", "type": "structure"},
        ],
        "type": "structure",
    },
    "DriftCheckBaselines": {
        "members": [
            {"name": "Bias", "shape": "DriftCheckBias", "type": "structure"},
            {"name": "Explainability", "shape": "DriftCheckExplainability", "type": "structure"},
            {"name": "ModelQuality", "shape": "DriftCheckModelQuality", "type": "structure"},
            {
                "name": "ModelDataQuality",
                "shape": "DriftCheckModelDataQuality",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "DriftCheckBias": {
        "members": [
            {"name": "ConfigFile", "shape": "FileSource", "type": "structure"},
            {"name": "PreTrainingConstraints", "shape": "MetricsSource", "type": "structure"},
            {"name": "PostTrainingConstraints", "shape": "MetricsSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "DriftCheckExplainability": {
        "members": [
            {"name": "Constraints", "shape": "MetricsSource", "type": "structure"},
            {"name": "ConfigFile", "shape": "FileSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "DriftCheckModelDataQuality": {
        "members": [
            {"name": "Statistics", "shape": "MetricsSource", "type": "structure"},
            {"name": "Constraints", "shape": "MetricsSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "DriftCheckModelQuality": {
        "members": [
            {"name": "Statistics", "shape": "MetricsSource", "type": "structure"},
            {"name": "Constraints", "shape": "MetricsSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "DynamicScalingConfiguration": {
        "members": [
            {"name": "MinCapacity", "shape": "Integer", "type": "integer"},
            {"name": "MaxCapacity", "shape": "Integer", "type": "integer"},
            {"name": "ScaleInCooldown", "shape": "Integer", "type": "integer"},
            {"name": "ScaleOutCooldown", "shape": "Integer", "type": "integer"},
            {"name": "ScalingPolicies", "shape": "ScalingPolicies", "type": "list"},
        ],
        "type": "structure",
    },
    "EFSFileSystem": {
        "members": [{"name": "FileSystemId", "shape": "FileSystemId", "type": "string"}],
        "type": "structure",
    },
    "EFSFileSystemConfig": {
        "members": [
            {"name": "FileSystemId", "shape": "FileSystemId", "type": "string"},
            {"name": "FileSystemPath", "shape": "FileSystemPath", "type": "string"},
        ],
        "type": "structure",
    },
    "EMRStepMetadata": {
        "members": [
            {"name": "ClusterId", "shape": "String256", "type": "string"},
            {"name": "StepId", "shape": "String256", "type": "string"},
            {"name": "StepName", "shape": "String256", "type": "string"},
            {"name": "LogFilePath", "shape": "String1024", "type": "string"},
        ],
        "type": "structure",
    },
    "EbsStorageSettings": {
        "members": [
            {"name": "EbsVolumeSizeInGb", "shape": "SpaceEbsVolumeSizeInGb", "type": "integer"}
        ],
        "type": "structure",
    },
    "Edge": {
        "members": [
            {"name": "SourceArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "DestinationArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "AssociationType", "shape": "AssociationEdgeType", "type": "string"},
        ],
        "type": "structure",
    },
    "EdgeDeploymentConfig": {
        "members": [
            {"name": "FailureHandlingPolicy", "shape": "FailureHandlingPolicy", "type": "string"}
        ],
        "type": "structure",
    },
    "EdgeDeploymentModelConfig": {
        "members": [
            {"name": "ModelHandle", "shape": "EntityName", "type": "string"},
            {"name": "EdgePackagingJobName", "shape": "EntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "EdgeDeploymentModelConfigs": {
        "member_shape": "EdgeDeploymentModelConfig",
        "member_type": "structure",
        "type": "list",
    },
    "EdgeDeploymentPlanSummaries": {
        "member_shape": "EdgeDeploymentPlanSummary",
        "member_type": "structure",
        "type": "list",
    },
    "EdgeDeploymentPlanSummary": {
        "members": [
            {"name": "EdgeDeploymentPlanArn", "shape": "EdgeDeploymentPlanArn", "type": "string"},
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "EdgeDeploymentSuccess", "shape": "Integer", "type": "integer"},
            {"name": "EdgeDeploymentPending", "shape": "Integer", "type": "integer"},
            {"name": "EdgeDeploymentFailed", "shape": "Integer", "type": "integer"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "EdgeDeploymentStatus": {
        "members": [
            {"name": "StageStatus", "shape": "StageStatus", "type": "string"},
            {"name": "EdgeDeploymentSuccessInStage", "shape": "Integer", "type": "integer"},
            {"name": "EdgeDeploymentPendingInStage", "shape": "Integer", "type": "integer"},
            {"name": "EdgeDeploymentFailedInStage", "shape": "Integer", "type": "integer"},
            {"name": "EdgeDeploymentStatusMessage", "shape": "String", "type": "string"},
            {"name": "EdgeDeploymentStageStartTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "EdgeModel": {
        "members": [
            {"name": "ModelName", "shape": "EntityName", "type": "string"},
            {"name": "ModelVersion", "shape": "EdgeVersion", "type": "string"},
            {"name": "LatestSampleTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LatestInference", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "EdgeModelStat": {
        "members": [
            {"name": "ModelName", "shape": "EntityName", "type": "string"},
            {"name": "ModelVersion", "shape": "EdgeVersion", "type": "string"},
            {"name": "OfflineDeviceCount", "shape": "Long", "type": "long"},
            {"name": "ConnectedDeviceCount", "shape": "Long", "type": "long"},
            {"name": "ActiveDeviceCount", "shape": "Long", "type": "long"},
            {"name": "SamplingDeviceCount", "shape": "Long", "type": "long"},
        ],
        "type": "structure",
    },
    "EdgeModelStats": {"member_shape": "EdgeModelStat", "member_type": "structure", "type": "list"},
    "EdgeModelSummaries": {
        "member_shape": "EdgeModelSummary",
        "member_type": "structure",
        "type": "list",
    },
    "EdgeModelSummary": {
        "members": [
            {"name": "ModelName", "shape": "EntityName", "type": "string"},
            {"name": "ModelVersion", "shape": "EdgeVersion", "type": "string"},
        ],
        "type": "structure",
    },
    "EdgeModels": {"member_shape": "EdgeModel", "member_type": "structure", "type": "list"},
    "EdgeOutputConfig": {
        "members": [
            {"name": "S3OutputLocation", "shape": "S3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "PresetDeploymentType", "shape": "EdgePresetDeploymentType", "type": "string"},
            {"name": "PresetDeploymentConfig", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "EdgePackagingJobSummaries": {
        "member_shape": "EdgePackagingJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "EdgePackagingJobSummary": {
        "members": [
            {"name": "EdgePackagingJobArn", "shape": "EdgePackagingJobArn", "type": "string"},
            {"name": "EdgePackagingJobName", "shape": "EntityName", "type": "string"},
            {"name": "EdgePackagingJobStatus", "shape": "EdgePackagingJobStatus", "type": "string"},
            {"name": "CompilationJobName", "shape": "EntityName", "type": "string"},
            {"name": "ModelName", "shape": "EntityName", "type": "string"},
            {"name": "ModelVersion", "shape": "EdgeVersion", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "EdgePresetDeploymentOutput": {
        "members": [
            {"name": "Type", "shape": "EdgePresetDeploymentType", "type": "string"},
            {"name": "Artifact", "shape": "EdgePresetDeploymentArtifact", "type": "string"},
            {"name": "Status", "shape": "EdgePresetDeploymentStatus", "type": "string"},
            {"name": "StatusMessage", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "Edges": {"member_shape": "Edge", "member_type": "structure", "type": "list"},
    "EmrServerlessComputeConfig": {
        "members": [{"name": "ExecutionRoleARN", "shape": "RoleArn", "type": "string"}],
        "type": "structure",
    },
    "EmrServerlessSettings": {
        "members": [
            {"name": "ExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Status", "shape": "FeatureStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "EmrSettings": {
        "members": [
            {"name": "AssumableRoleArns", "shape": "AssumableRoleArns", "type": "list"},
            {"name": "ExecutionRoleArns", "shape": "ExecutionRoleArns", "type": "list"},
        ],
        "type": "structure",
    },
    "EnableSagemakerServicecatalogPortfolioInput": {"members": [], "type": "structure"},
    "EnableSagemakerServicecatalogPortfolioOutput": {"members": [], "type": "structure"},
    "Endpoint": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "EndpointArn", "shape": "EndpointArn", "type": "string"},
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {"name": "ProductionVariants", "shape": "ProductionVariantSummaryList", "type": "list"},
            {"name": "DataCaptureConfig", "shape": "DataCaptureConfigSummary", "type": "structure"},
            {"name": "EndpointStatus", "shape": "EndpointStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MonitoringSchedules", "shape": "MonitoringScheduleList", "type": "list"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {
                "name": "ShadowProductionVariants",
                "shape": "ProductionVariantSummaryList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "EndpointConfigStepMetadata": {
        "members": [{"name": "Arn", "shape": "EndpointConfigArn", "type": "string"}],
        "type": "structure",
    },
    "EndpointConfigSummary": {
        "members": [
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {"name": "EndpointConfigArn", "shape": "EndpointConfigArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "EndpointConfigSummaryList": {
        "member_shape": "EndpointConfigSummary",
        "member_type": "structure",
        "type": "list",
    },
    "EndpointInfo": {
        "members": [{"name": "EndpointName", "shape": "EndpointName", "type": "string"}],
        "type": "structure",
    },
    "EndpointInput": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "LocalPath", "shape": "ProcessingLocalPath", "type": "string"},
            {"name": "S3InputMode", "shape": "ProcessingS3InputMode", "type": "string"},
            {
                "name": "S3DataDistributionType",
                "shape": "ProcessingS3DataDistributionType",
                "type": "string",
            },
            {"name": "FeaturesAttribute", "shape": "String", "type": "string"},
            {"name": "InferenceAttribute", "shape": "String", "type": "string"},
            {"name": "ProbabilityAttribute", "shape": "String", "type": "string"},
            {
                "name": "ProbabilityThresholdAttribute",
                "shape": "ProbabilityThresholdAttribute",
                "type": "double",
            },
            {"name": "StartTimeOffset", "shape": "MonitoringTimeOffsetString", "type": "string"},
            {"name": "EndTimeOffset", "shape": "MonitoringTimeOffsetString", "type": "string"},
            {
                "name": "ExcludeFeaturesAttribute",
                "shape": "ExcludeFeaturesAttribute",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "EndpointInputConfiguration": {
        "members": [
            {"name": "InstanceType", "shape": "ProductionVariantInstanceType", "type": "string"},
            {
                "name": "ServerlessConfig",
                "shape": "ProductionVariantServerlessConfig",
                "type": "structure",
            },
            {
                "name": "InferenceSpecificationName",
                "shape": "InferenceSpecificationName",
                "type": "string",
            },
            {
                "name": "EnvironmentParameterRanges",
                "shape": "EnvironmentParameterRanges",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "EndpointInputConfigurations": {
        "member_shape": "EndpointInputConfiguration",
        "member_type": "structure",
        "type": "list",
    },
    "EndpointMetadata": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {"name": "EndpointStatus", "shape": "EndpointStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
        ],
        "type": "structure",
    },
    "EndpointOutputConfiguration": {
        "members": [
            {"name": "EndpointName", "shape": "String", "type": "string"},
            {"name": "VariantName", "shape": "String", "type": "string"},
            {"name": "InstanceType", "shape": "ProductionVariantInstanceType", "type": "string"},
            {"name": "InitialInstanceCount", "shape": "InitialInstanceCount", "type": "integer"},
            {
                "name": "ServerlessConfig",
                "shape": "ProductionVariantServerlessConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "EndpointPerformance": {
        "members": [
            {"name": "Metrics", "shape": "InferenceMetrics", "type": "structure"},
            {"name": "EndpointInfo", "shape": "EndpointInfo", "type": "structure"},
        ],
        "type": "structure",
    },
    "EndpointPerformances": {
        "member_shape": "EndpointPerformance",
        "member_type": "structure",
        "type": "list",
    },
    "EndpointStepMetadata": {
        "members": [{"name": "Arn", "shape": "EndpointArn", "type": "string"}],
        "type": "structure",
    },
    "EndpointSummary": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "EndpointArn", "shape": "EndpointArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndpointStatus", "shape": "EndpointStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "EndpointSummaryList": {
        "member_shape": "EndpointSummary",
        "member_type": "structure",
        "type": "list",
    },
    "Endpoints": {"member_shape": "EndpointInfo", "member_type": "structure", "type": "list"},
    "EnvironmentMap": {
        "key_shape": "EnvironmentKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "EnvironmentValue",
        "value_type": "string",
    },
    "EnvironmentParameter": {
        "members": [
            {"name": "Key", "shape": "String", "type": "string"},
            {"name": "ValueType", "shape": "String", "type": "string"},
            {"name": "Value", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "EnvironmentParameterRanges": {
        "members": [
            {"name": "CategoricalParameterRanges", "shape": "CategoricalParameters", "type": "list"}
        ],
        "type": "structure",
    },
    "EnvironmentParameters": {
        "member_shape": "EnvironmentParameter",
        "member_type": "structure",
        "type": "list",
    },
    "ErrorInfo": {
        "members": [
            {"name": "Code", "shape": "NonEmptyString64", "type": "string"},
            {"name": "Reason", "shape": "NonEmptyString256", "type": "string"},
        ],
        "type": "structure",
    },
    "ExecutionRoleArns": {"member_shape": "RoleArn", "member_type": "string", "type": "list"},
    "Experiment": {
        "members": [
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "ExperimentArn", "shape": "ExperimentArn", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "ExperimentSource", "type": "structure"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "ExperimentConfig": {
        "members": [
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
            {
                "name": "TrialComponentDisplayName",
                "shape": "ExperimentEntityName",
                "type": "string",
            },
            {"name": "RunName", "shape": "ExperimentEntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "ExperimentSource": {
        "members": [
            {"name": "SourceArn", "shape": "ExperimentSourceArn", "type": "string"},
            {"name": "SourceType", "shape": "SourceType", "type": "string"},
        ],
        "type": "structure",
    },
    "ExperimentSummaries": {
        "member_shape": "ExperimentSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ExperimentSummary": {
        "members": [
            {"name": "ExperimentArn", "shape": "ExperimentArn", "type": "string"},
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "ExperimentSource", "shape": "ExperimentSource", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "Explainability": {
        "members": [{"name": "Report", "shape": "MetricsSource", "type": "structure"}],
        "type": "structure",
    },
    "ExplainerConfig": {
        "members": [
            {
                "name": "ClarifyExplainerConfig",
                "shape": "ClarifyExplainerConfig",
                "type": "structure",
            }
        ],
        "type": "structure",
    },
    "FSxLustreFileSystem": {
        "members": [{"name": "FileSystemId", "shape": "FileSystemId", "type": "string"}],
        "type": "structure",
    },
    "FSxLustreFileSystemConfig": {
        "members": [
            {"name": "FileSystemId", "shape": "FileSystemId", "type": "string"},
            {"name": "FileSystemPath", "shape": "FileSystemPath", "type": "string"},
        ],
        "type": "structure",
    },
    "FailStepMetadata": {
        "members": [{"name": "ErrorMessage", "shape": "String3072", "type": "string"}],
        "type": "structure",
    },
    "FeatureAdditions": {
        "member_shape": "FeatureDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "FeatureDefinition": {
        "members": [
            {"name": "FeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "FeatureType", "shape": "FeatureType", "type": "string"},
            {"name": "CollectionType", "shape": "CollectionType", "type": "string"},
            {"name": "CollectionConfig", "shape": "CollectionConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "FeatureDefinitions": {
        "member_shape": "FeatureDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "FeatureGroup": {
        "members": [
            {"name": "FeatureGroupArn", "shape": "FeatureGroupArn", "type": "string"},
            {"name": "FeatureGroupName", "shape": "FeatureGroupName", "type": "string"},
            {"name": "RecordIdentifierFeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "EventTimeFeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "FeatureDefinitions", "shape": "FeatureDefinitions", "type": "list"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "OnlineStoreConfig", "shape": "OnlineStoreConfig", "type": "structure"},
            {"name": "OfflineStoreConfig", "shape": "OfflineStoreConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "FeatureGroupStatus", "shape": "FeatureGroupStatus", "type": "string"},
            {"name": "OfflineStoreStatus", "shape": "OfflineStoreStatus", "type": "structure"},
            {"name": "LastUpdateStatus", "shape": "LastUpdateStatus", "type": "structure"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "Description", "shape": "Description", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "FeatureGroupSummaries": {
        "member_shape": "FeatureGroupSummary",
        "member_type": "structure",
        "type": "list",
    },
    "FeatureGroupSummary": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupName", "type": "string"},
            {"name": "FeatureGroupArn", "shape": "FeatureGroupArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FeatureGroupStatus", "shape": "FeatureGroupStatus", "type": "string"},
            {"name": "OfflineStoreStatus", "shape": "OfflineStoreStatus", "type": "structure"},
        ],
        "type": "structure",
    },
    "FeatureMetadata": {
        "members": [
            {"name": "FeatureGroupArn", "shape": "FeatureGroupArn", "type": "string"},
            {"name": "FeatureGroupName", "shape": "FeatureGroupName", "type": "string"},
            {"name": "FeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "FeatureType", "shape": "FeatureType", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "Description", "shape": "FeatureDescription", "type": "string"},
            {"name": "Parameters", "shape": "FeatureParameters", "type": "list"},
        ],
        "type": "structure",
    },
    "FeatureNames": {"member_shape": "FeatureName", "member_type": "string", "type": "list"},
    "FeatureParameter": {
        "members": [
            {"name": "Key", "shape": "FeatureParameterKey", "type": "string"},
            {"name": "Value", "shape": "FeatureParameterValue", "type": "string"},
        ],
        "type": "structure",
    },
    "FeatureParameterAdditions": {
        "member_shape": "FeatureParameter",
        "member_type": "structure",
        "type": "list",
    },
    "FeatureParameterRemovals": {
        "member_shape": "FeatureParameterKey",
        "member_type": "string",
        "type": "list",
    },
    "FeatureParameters": {
        "member_shape": "FeatureParameter",
        "member_type": "structure",
        "type": "list",
    },
    "FeatureValue": {
        "members": [
            {"name": "FeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "ValueAsString", "shape": "ValueAsString", "type": "string"},
            {"name": "ValueAsStringList", "shape": "ValueAsStringList", "type": "list"},
        ],
        "type": "structure",
    },
    "FileSource": {
        "members": [
            {"name": "ContentType", "shape": "ContentType", "type": "string"},
            {"name": "ContentDigest", "shape": "ContentDigest", "type": "string"},
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "FileSystemConfig": {
        "members": [
            {"name": "MountPath", "shape": "MountPath", "type": "string"},
            {"name": "DefaultUid", "shape": "DefaultUid", "type": "integer"},
            {"name": "DefaultGid", "shape": "DefaultGid", "type": "integer"},
        ],
        "type": "structure",
    },
    "FileSystemDataSource": {
        "members": [
            {"name": "FileSystemId", "shape": "FileSystemId", "type": "string"},
            {"name": "FileSystemAccessMode", "shape": "FileSystemAccessMode", "type": "string"},
            {"name": "FileSystemType", "shape": "FileSystemType", "type": "string"},
            {"name": "DirectoryPath", "shape": "DirectoryPath", "type": "string"},
        ],
        "type": "structure",
    },
    "FillingTransformationMap": {
        "key_shape": "FillingType",
        "key_type": "string",
        "type": "map",
        "value_shape": "FillingTransformationValue",
        "value_type": "string",
    },
    "FillingTransformations": {
        "key_shape": "TransformationAttributeName",
        "key_type": "string",
        "type": "map",
        "value_shape": "FillingTransformationMap",
        "value_type": "map",
    },
    "Filter": {
        "members": [
            {"name": "Name", "shape": "ResourcePropertyName", "type": "string"},
            {"name": "Operator", "shape": "Operator", "type": "string"},
            {"name": "Value", "shape": "FilterValue", "type": "string"},
        ],
        "type": "structure",
    },
    "FilterList": {"member_shape": "Filter", "member_type": "structure", "type": "list"},
    "FinalAutoMLJobObjectiveMetric": {
        "members": [
            {"name": "Type", "shape": "AutoMLJobObjectiveType", "type": "string"},
            {"name": "MetricName", "shape": "AutoMLMetricEnum", "type": "string"},
            {"name": "Value", "shape": "MetricValue", "type": "float"},
            {"name": "StandardMetricName", "shape": "AutoMLMetricEnum", "type": "string"},
        ],
        "type": "structure",
    },
    "FinalHyperParameterTuningJobObjectiveMetric": {
        "members": [
            {"name": "Type", "shape": "HyperParameterTuningJobObjectiveType", "type": "string"},
            {"name": "MetricName", "shape": "MetricName", "type": "string"},
            {"name": "Value", "shape": "MetricValue", "type": "float"},
        ],
        "type": "structure",
    },
    "FinalMetricDataList": {
        "member_shape": "MetricData",
        "member_type": "structure",
        "type": "list",
    },
    "FlowDefinitionOutputConfig": {
        "members": [
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "FlowDefinitionSummaries": {
        "member_shape": "FlowDefinitionSummary",
        "member_type": "structure",
        "type": "list",
    },
    "FlowDefinitionSummary": {
        "members": [
            {"name": "FlowDefinitionName", "shape": "FlowDefinitionName", "type": "string"},
            {"name": "FlowDefinitionArn", "shape": "FlowDefinitionArn", "type": "string"},
            {"name": "FlowDefinitionStatus", "shape": "FlowDefinitionStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
        ],
        "type": "structure",
    },
    "FlowDefinitionTaskKeywords": {
        "member_shape": "FlowDefinitionTaskKeyword",
        "member_type": "string",
        "type": "list",
    },
    "ForecastQuantiles": {
        "member_shape": "ForecastQuantile",
        "member_type": "string",
        "type": "list",
    },
    "GenerativeAiSettings": {
        "members": [{"name": "AmazonBedrockRoleArn", "shape": "RoleArn", "type": "string"}],
        "type": "structure",
    },
    "GetDeviceFleetReportRequest": {
        "members": [{"name": "DeviceFleetName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "GetDeviceFleetReportResponse": {
        "members": [
            {"name": "DeviceFleetArn", "shape": "DeviceFleetArn", "type": "string"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "OutputConfig", "shape": "EdgeOutputConfig", "type": "structure"},
            {"name": "Description", "shape": "DeviceFleetDescription", "type": "string"},
            {"name": "ReportGenerated", "shape": "Timestamp", "type": "timestamp"},
            {"name": "DeviceStats", "shape": "DeviceStats", "type": "structure"},
            {"name": "AgentVersions", "shape": "AgentVersions", "type": "list"},
            {"name": "ModelStats", "shape": "EdgeModelStats", "type": "list"},
        ],
        "type": "structure",
    },
    "GetLineageGroupPolicyRequest": {
        "members": [
            {"name": "LineageGroupName", "shape": "LineageGroupNameOrArn", "type": "string"}
        ],
        "type": "structure",
    },
    "GetLineageGroupPolicyResponse": {
        "members": [
            {"name": "LineageGroupArn", "shape": "LineageGroupArn", "type": "string"},
            {"name": "ResourcePolicy", "shape": "ResourcePolicyString", "type": "string"},
        ],
        "type": "structure",
    },
    "GetModelPackageGroupPolicyInput": {
        "members": [{"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "GetModelPackageGroupPolicyOutput": {
        "members": [{"name": "ResourcePolicy", "shape": "PolicyString", "type": "string"}],
        "type": "structure",
    },
    "GetRecordRequest": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupNameOrArn", "type": "string"},
            {"name": "RecordIdentifierValueAsString", "shape": "ValueAsString", "type": "string"},
            {"name": "FeatureNames", "shape": "FeatureNames", "type": "list"},
            {"name": "ExpirationTimeResponse", "shape": "ExpirationTimeResponse", "type": "string"},
        ],
        "type": "structure",
    },
    "GetRecordResponse": {
        "members": [
            {"name": "Record", "shape": "Record", "type": "list"},
            {"name": "ExpiresAt", "shape": "ExpiresAt", "type": "string"},
        ],
        "type": "structure",
    },
    "GetSagemakerServicecatalogPortfolioStatusInput": {"members": [], "type": "structure"},
    "GetSagemakerServicecatalogPortfolioStatusOutput": {
        "members": [{"name": "Status", "shape": "SagemakerServicecatalogStatus", "type": "string"}],
        "type": "structure",
    },
    "GetScalingConfigurationRecommendationRequest": {
        "members": [
            {
                "name": "InferenceRecommendationsJobName",
                "shape": "RecommendationJobName",
                "type": "string",
            },
            {"name": "RecommendationId", "shape": "String", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {
                "name": "TargetCpuUtilizationPerCore",
                "shape": "UtilizationPercentagePerCore",
                "type": "integer",
            },
            {
                "name": "ScalingPolicyObjective",
                "shape": "ScalingPolicyObjective",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "GetScalingConfigurationRecommendationResponse": {
        "members": [
            {
                "name": "InferenceRecommendationsJobName",
                "shape": "RecommendationJobName",
                "type": "string",
            },
            {"name": "RecommendationId", "shape": "String", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {
                "name": "TargetCpuUtilizationPerCore",
                "shape": "UtilizationPercentagePerCore",
                "type": "integer",
            },
            {
                "name": "ScalingPolicyObjective",
                "shape": "ScalingPolicyObjective",
                "type": "structure",
            },
            {"name": "Metric", "shape": "ScalingPolicyMetric", "type": "structure"},
            {
                "name": "DynamicScalingConfiguration",
                "shape": "DynamicScalingConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "GetSearchSuggestionsRequest": {
        "members": [
            {"name": "Resource", "shape": "ResourceType", "type": "string"},
            {"name": "SuggestionQuery", "shape": "SuggestionQuery", "type": "structure"},
        ],
        "type": "structure",
    },
    "GetSearchSuggestionsResponse": {
        "members": [
            {
                "name": "PropertyNameSuggestions",
                "shape": "PropertyNameSuggestionList",
                "type": "list",
            }
        ],
        "type": "structure",
    },
    "GitConfig": {
        "members": [
            {"name": "RepositoryUrl", "shape": "GitConfigUrl", "type": "string"},
            {"name": "Branch", "shape": "Branch", "type": "string"},
            {"name": "SecretArn", "shape": "SecretArn", "type": "string"},
        ],
        "type": "structure",
    },
    "GitConfigForUpdate": {
        "members": [{"name": "SecretArn", "shape": "SecretArn", "type": "string"}],
        "type": "structure",
    },
    "GroupingAttributeNames": {
        "member_shape": "GroupingAttributeName",
        "member_type": "string",
        "type": "list",
    },
    "Groups": {"member_shape": "Group", "member_type": "string", "type": "list"},
    "HiddenAppTypesList": {"member_shape": "AppType", "member_type": "string", "type": "list"},
    "HiddenInstanceTypesList": {
        "member_shape": "AppInstanceType",
        "member_type": "string",
        "type": "list",
    },
    "HiddenMlToolsList": {"member_shape": "MlTools", "member_type": "string", "type": "list"},
    "HiddenSageMakerImage": {
        "members": [
            {"name": "SageMakerImageName", "shape": "SageMakerImageName", "type": "string"},
            {"name": "VersionAliases", "shape": "VersionAliasesList", "type": "list"},
        ],
        "type": "structure",
    },
    "HiddenSageMakerImageVersionAliasesList": {
        "member_shape": "HiddenSageMakerImage",
        "member_type": "structure",
        "type": "list",
    },
    "HolidayConfig": {
        "member_shape": "HolidayConfigAttributes",
        "member_type": "structure",
        "type": "list",
    },
    "HolidayConfigAttributes": {
        "members": [{"name": "CountryCode", "shape": "CountryCode", "type": "string"}],
        "type": "structure",
    },
    "HookParameters": {
        "key_shape": "ConfigKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "ConfigValue",
        "value_type": "string",
    },
    "HubAccessConfig": {
        "members": [{"name": "HubContentArn", "shape": "HubContentArn", "type": "string"}],
        "type": "structure",
    },
    "HubContentDependency": {
        "members": [
            {"name": "DependencyOriginPath", "shape": "DependencyOriginPath", "type": "string"},
            {"name": "DependencyCopyPath", "shape": "DependencyCopyPath", "type": "string"},
        ],
        "type": "structure",
    },
    "HubContentDependencyList": {
        "member_shape": "HubContentDependency",
        "member_type": "structure",
        "type": "list",
    },
    "HubContentInfo": {
        "members": [
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "HubContentArn", "shape": "HubContentArn", "type": "string"},
            {
                "name": "SageMakerPublicHubContentArn",
                "shape": "SageMakerPublicHubContentArn",
                "type": "string",
            },
            {"name": "HubContentVersion", "shape": "HubContentVersion", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "DocumentSchemaVersion", "shape": "DocumentSchemaVersion", "type": "string"},
            {"name": "HubContentDisplayName", "shape": "HubContentDisplayName", "type": "string"},
            {"name": "HubContentDescription", "shape": "HubContentDescription", "type": "string"},
            {"name": "SupportStatus", "shape": "HubContentSupportStatus", "type": "string"},
            {
                "name": "HubContentSearchKeywords",
                "shape": "HubContentSearchKeywordList",
                "type": "list",
            },
            {"name": "HubContentStatus", "shape": "HubContentStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "OriginalCreationTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "HubContentInfoList": {
        "member_shape": "HubContentInfo",
        "member_type": "structure",
        "type": "list",
    },
    "HubContentSearchKeywordList": {
        "member_shape": "HubSearchKeyword",
        "member_type": "string",
        "type": "list",
    },
    "HubInfo": {
        "members": [
            {"name": "HubName", "shape": "HubName", "type": "string"},
            {"name": "HubArn", "shape": "HubArn", "type": "string"},
            {"name": "HubDisplayName", "shape": "HubDisplayName", "type": "string"},
            {"name": "HubDescription", "shape": "HubDescription", "type": "string"},
            {"name": "HubSearchKeywords", "shape": "HubSearchKeywordList", "type": "list"},
            {"name": "HubStatus", "shape": "HubStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "HubInfoList": {"member_shape": "HubInfo", "member_type": "structure", "type": "list"},
    "HubS3StorageConfig": {
        "members": [{"name": "S3OutputPath", "shape": "S3OutputPath", "type": "string"}],
        "type": "structure",
    },
    "HubSearchKeywordList": {
        "member_shape": "HubSearchKeyword",
        "member_type": "string",
        "type": "list",
    },
    "HumanLoopActivationConditionsConfig": {
        "members": [
            {
                "name": "HumanLoopActivationConditions",
                "shape": "HumanLoopActivationConditions",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "HumanLoopActivationConfig": {
        "members": [
            {
                "name": "HumanLoopActivationConditionsConfig",
                "shape": "HumanLoopActivationConditionsConfig",
                "type": "structure",
            }
        ],
        "type": "structure",
    },
    "HumanLoopConfig": {
        "members": [
            {"name": "WorkteamArn", "shape": "WorkteamArn", "type": "string"},
            {"name": "HumanTaskUiArn", "shape": "HumanTaskUiArn", "type": "string"},
            {"name": "TaskTitle", "shape": "FlowDefinitionTaskTitle", "type": "string"},
            {"name": "TaskDescription", "shape": "FlowDefinitionTaskDescription", "type": "string"},
            {"name": "TaskCount", "shape": "FlowDefinitionTaskCount", "type": "integer"},
            {
                "name": "TaskAvailabilityLifetimeInSeconds",
                "shape": "FlowDefinitionTaskAvailabilityLifetimeInSeconds",
                "type": "integer",
            },
            {
                "name": "TaskTimeLimitInSeconds",
                "shape": "FlowDefinitionTaskTimeLimitInSeconds",
                "type": "integer",
            },
            {"name": "TaskKeywords", "shape": "FlowDefinitionTaskKeywords", "type": "list"},
            {
                "name": "PublicWorkforceTaskPrice",
                "shape": "PublicWorkforceTaskPrice",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "HumanLoopRequestSource": {
        "members": [
            {
                "name": "AwsManagedHumanLoopRequestSource",
                "shape": "AwsManagedHumanLoopRequestSource",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "HumanTaskConfig": {
        "members": [
            {"name": "WorkteamArn", "shape": "WorkteamArn", "type": "string"},
            {"name": "UiConfig", "shape": "UiConfig", "type": "structure"},
            {"name": "PreHumanTaskLambdaArn", "shape": "LambdaFunctionArn", "type": "string"},
            {"name": "TaskKeywords", "shape": "TaskKeywords", "type": "list"},
            {"name": "TaskTitle", "shape": "TaskTitle", "type": "string"},
            {"name": "TaskDescription", "shape": "TaskDescription", "type": "string"},
            {
                "name": "NumberOfHumanWorkersPerDataObject",
                "shape": "NumberOfHumanWorkersPerDataObject",
                "type": "integer",
            },
            {
                "name": "TaskTimeLimitInSeconds",
                "shape": "TaskTimeLimitInSeconds",
                "type": "integer",
            },
            {
                "name": "TaskAvailabilityLifetimeInSeconds",
                "shape": "TaskAvailabilityLifetimeInSeconds",
                "type": "integer",
            },
            {
                "name": "MaxConcurrentTaskCount",
                "shape": "MaxConcurrentTaskCount",
                "type": "integer",
            },
            {
                "name": "AnnotationConsolidationConfig",
                "shape": "AnnotationConsolidationConfig",
                "type": "structure",
            },
            {
                "name": "PublicWorkforceTaskPrice",
                "shape": "PublicWorkforceTaskPrice",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "HumanTaskUiSummaries": {
        "member_shape": "HumanTaskUiSummary",
        "member_type": "structure",
        "type": "list",
    },
    "HumanTaskUiSummary": {
        "members": [
            {"name": "HumanTaskUiName", "shape": "HumanTaskUiName", "type": "string"},
            {"name": "HumanTaskUiArn", "shape": "HumanTaskUiArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "HyperParameterAlgorithmSpecification": {
        "members": [
            {"name": "TrainingImage", "shape": "AlgorithmImage", "type": "string"},
            {"name": "TrainingInputMode", "shape": "TrainingInputMode", "type": "string"},
            {"name": "AlgorithmName", "shape": "ArnOrName", "type": "string"},
            {"name": "MetricDefinitions", "shape": "MetricDefinitionList", "type": "list"},
        ],
        "type": "structure",
    },
    "HyperParameterSpecification": {
        "members": [
            {"name": "Name", "shape": "ParameterName", "type": "string"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
            {"name": "Type", "shape": "ParameterType", "type": "string"},
            {"name": "Range", "shape": "ParameterRange", "type": "structure"},
            {"name": "IsTunable", "shape": "Boolean", "type": "boolean"},
            {"name": "IsRequired", "shape": "Boolean", "type": "boolean"},
            {"name": "DefaultValue", "shape": "HyperParameterValue", "type": "string"},
        ],
        "type": "structure",
    },
    "HyperParameterSpecifications": {
        "member_shape": "HyperParameterSpecification",
        "member_type": "structure",
        "type": "list",
    },
    "HyperParameterTrainingJobDefinition": {
        "members": [
            {
                "name": "DefinitionName",
                "shape": "HyperParameterTrainingJobDefinitionName",
                "type": "string",
            },
            {
                "name": "TuningObjective",
                "shape": "HyperParameterTuningJobObjective",
                "type": "structure",
            },
            {"name": "HyperParameterRanges", "shape": "ParameterRanges", "type": "structure"},
            {"name": "StaticHyperParameters", "shape": "HyperParameters", "type": "map"},
            {
                "name": "AlgorithmSpecification",
                "shape": "HyperParameterAlgorithmSpecification",
                "type": "structure",
            },
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "InputDataConfig", "shape": "InputDataConfig", "type": "list"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "OutputDataConfig", "shape": "OutputDataConfig", "type": "structure"},
            {"name": "ResourceConfig", "shape": "ResourceConfig", "type": "structure"},
            {
                "name": "HyperParameterTuningResourceConfig",
                "shape": "HyperParameterTuningResourceConfig",
                "type": "structure",
            },
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
            {
                "name": "EnableInterContainerTrafficEncryption",
                "shape": "Boolean",
                "type": "boolean",
            },
            {"name": "EnableManagedSpotTraining", "shape": "Boolean", "type": "boolean"},
            {"name": "CheckpointConfig", "shape": "CheckpointConfig", "type": "structure"},
            {"name": "RetryStrategy", "shape": "RetryStrategy", "type": "structure"},
            {
                "name": "Environment",
                "shape": "HyperParameterTrainingJobEnvironmentMap",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "HyperParameterTrainingJobDefinitions": {
        "member_shape": "HyperParameterTrainingJobDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "HyperParameterTrainingJobEnvironmentMap": {
        "key_shape": "HyperParameterTrainingJobEnvironmentKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "HyperParameterTrainingJobEnvironmentValue",
        "value_type": "string",
    },
    "HyperParameterTrainingJobSummaries": {
        "member_shape": "HyperParameterTrainingJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "HyperParameterTrainingJobSummary": {
        "members": [
            {
                "name": "TrainingJobDefinitionName",
                "shape": "HyperParameterTrainingJobDefinitionName",
                "type": "string",
            },
            {"name": "TrainingJobName", "shape": "TrainingJobName", "type": "string"},
            {"name": "TrainingJobArn", "shape": "TrainingJobArn", "type": "string"},
            {"name": "TuningJobName", "shape": "HyperParameterTuningJobName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingJobStatus", "shape": "TrainingJobStatus", "type": "string"},
            {"name": "TunedHyperParameters", "shape": "HyperParameters", "type": "map"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {
                "name": "FinalHyperParameterTuningJobObjectiveMetric",
                "shape": "FinalHyperParameterTuningJobObjectiveMetric",
                "type": "structure",
            },
            {"name": "ObjectiveStatus", "shape": "ObjectiveStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "HyperParameterTuningInstanceConfig": {
        "members": [
            {"name": "InstanceType", "shape": "TrainingInstanceType", "type": "string"},
            {"name": "InstanceCount", "shape": "TrainingInstanceCount", "type": "integer"},
            {"name": "VolumeSizeInGB", "shape": "VolumeSizeInGB", "type": "integer"},
        ],
        "type": "structure",
    },
    "HyperParameterTuningInstanceConfigs": {
        "member_shape": "HyperParameterTuningInstanceConfig",
        "member_type": "structure",
        "type": "list",
    },
    "HyperParameterTuningJobCompletionDetails": {
        "members": [
            {
                "name": "NumberOfTrainingJobsObjectiveNotImproving",
                "shape": "Integer",
                "type": "integer",
            },
            {"name": "ConvergenceDetectedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "HyperParameterTuningJobConfig": {
        "members": [
            {"name": "Strategy", "shape": "HyperParameterTuningJobStrategyType", "type": "string"},
            {
                "name": "StrategyConfig",
                "shape": "HyperParameterTuningJobStrategyConfig",
                "type": "structure",
            },
            {
                "name": "HyperParameterTuningJobObjective",
                "shape": "HyperParameterTuningJobObjective",
                "type": "structure",
            },
            {"name": "ResourceLimits", "shape": "ResourceLimits", "type": "structure"},
            {"name": "ParameterRanges", "shape": "ParameterRanges", "type": "structure"},
            {
                "name": "TrainingJobEarlyStoppingType",
                "shape": "TrainingJobEarlyStoppingType",
                "type": "string",
            },
            {
                "name": "TuningJobCompletionCriteria",
                "shape": "TuningJobCompletionCriteria",
                "type": "structure",
            },
            {"name": "RandomSeed", "shape": "RandomSeed", "type": "integer"},
        ],
        "type": "structure",
    },
    "HyperParameterTuningJobConsumedResources": {
        "members": [{"name": "RuntimeInSeconds", "shape": "Integer", "type": "integer"}],
        "type": "structure",
    },
    "HyperParameterTuningJobObjective": {
        "members": [
            {"name": "Type", "shape": "HyperParameterTuningJobObjectiveType", "type": "string"},
            {"name": "MetricName", "shape": "MetricName", "type": "string"},
        ],
        "type": "structure",
    },
    "HyperParameterTuningJobObjectives": {
        "member_shape": "HyperParameterTuningJobObjective",
        "member_type": "structure",
        "type": "list",
    },
    "HyperParameterTuningJobSearchEntity": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            },
            {
                "name": "HyperParameterTuningJobArn",
                "shape": "HyperParameterTuningJobArn",
                "type": "string",
            },
            {
                "name": "HyperParameterTuningJobConfig",
                "shape": "HyperParameterTuningJobConfig",
                "type": "structure",
            },
            {
                "name": "TrainingJobDefinition",
                "shape": "HyperParameterTrainingJobDefinition",
                "type": "structure",
            },
            {
                "name": "TrainingJobDefinitions",
                "shape": "HyperParameterTrainingJobDefinitions",
                "type": "list",
            },
            {
                "name": "HyperParameterTuningJobStatus",
                "shape": "HyperParameterTuningJobStatus",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "HyperParameterTuningEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "TrainingJobStatusCounters",
                "shape": "TrainingJobStatusCounters",
                "type": "structure",
            },
            {
                "name": "ObjectiveStatusCounters",
                "shape": "ObjectiveStatusCounters",
                "type": "structure",
            },
            {
                "name": "BestTrainingJob",
                "shape": "HyperParameterTrainingJobSummary",
                "type": "structure",
            },
            {
                "name": "OverallBestTrainingJob",
                "shape": "HyperParameterTrainingJobSummary",
                "type": "structure",
            },
            {
                "name": "WarmStartConfig",
                "shape": "HyperParameterTuningJobWarmStartConfig",
                "type": "structure",
            },
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {
                "name": "TuningJobCompletionDetails",
                "shape": "HyperParameterTuningJobCompletionDetails",
                "type": "structure",
            },
            {
                "name": "ConsumedResources",
                "shape": "HyperParameterTuningJobConsumedResources",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "HyperParameterTuningJobStrategyConfig": {
        "members": [
            {
                "name": "HyperbandStrategyConfig",
                "shape": "HyperbandStrategyConfig",
                "type": "structure",
            }
        ],
        "type": "structure",
    },
    "HyperParameterTuningJobSummaries": {
        "member_shape": "HyperParameterTuningJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "HyperParameterTuningJobSummary": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            },
            {
                "name": "HyperParameterTuningJobArn",
                "shape": "HyperParameterTuningJobArn",
                "type": "string",
            },
            {
                "name": "HyperParameterTuningJobStatus",
                "shape": "HyperParameterTuningJobStatus",
                "type": "string",
            },
            {"name": "Strategy", "shape": "HyperParameterTuningJobStrategyType", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "HyperParameterTuningEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "TrainingJobStatusCounters",
                "shape": "TrainingJobStatusCounters",
                "type": "structure",
            },
            {
                "name": "ObjectiveStatusCounters",
                "shape": "ObjectiveStatusCounters",
                "type": "structure",
            },
            {"name": "ResourceLimits", "shape": "ResourceLimits", "type": "structure"},
        ],
        "type": "structure",
    },
    "HyperParameterTuningJobWarmStartConfig": {
        "members": [
            {
                "name": "ParentHyperParameterTuningJobs",
                "shape": "ParentHyperParameterTuningJobs",
                "type": "list",
            },
            {
                "name": "WarmStartType",
                "shape": "HyperParameterTuningJobWarmStartType",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "HyperParameterTuningResourceConfig": {
        "members": [
            {"name": "InstanceType", "shape": "TrainingInstanceType", "type": "string"},
            {"name": "InstanceCount", "shape": "TrainingInstanceCount", "type": "integer"},
            {"name": "VolumeSizeInGB", "shape": "OptionalVolumeSizeInGB", "type": "integer"},
            {"name": "VolumeKmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {
                "name": "AllocationStrategy",
                "shape": "HyperParameterTuningAllocationStrategy",
                "type": "string",
            },
            {
                "name": "InstanceConfigs",
                "shape": "HyperParameterTuningInstanceConfigs",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "HyperParameters": {
        "key_shape": "HyperParameterKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "HyperParameterValue",
        "value_type": "string",
    },
    "HyperbandStrategyConfig": {
        "members": [
            {"name": "MinResource", "shape": "HyperbandStrategyMinResource", "type": "integer"},
            {"name": "MaxResource", "shape": "HyperbandStrategyMaxResource", "type": "integer"},
        ],
        "type": "structure",
    },
    "IamIdentity": {
        "members": [
            {"name": "Arn", "shape": "String", "type": "string"},
            {"name": "PrincipalId", "shape": "String", "type": "string"},
            {"name": "SourceIdentity", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "IamPolicyConstraints": {
        "members": [
            {"name": "SourceIp", "shape": "EnabledOrDisabled", "type": "string"},
            {"name": "VpcSourceIp", "shape": "EnabledOrDisabled", "type": "string"},
        ],
        "type": "structure",
    },
    "IdentityProviderOAuthSetting": {
        "members": [
            {"name": "DataSourceName", "shape": "DataSourceName", "type": "string"},
            {"name": "Status", "shape": "FeatureStatus", "type": "string"},
            {"name": "SecretArn", "shape": "SecretArn", "type": "string"},
        ],
        "type": "structure",
    },
    "IdentityProviderOAuthSettings": {
        "member_shape": "IdentityProviderOAuthSetting",
        "member_type": "structure",
        "type": "list",
    },
    "IdleSettings": {
        "members": [
            {"name": "LifecycleManagement", "shape": "LifecycleManagement", "type": "string"},
            {"name": "IdleTimeoutInMinutes", "shape": "IdleTimeoutInMinutes", "type": "integer"},
            {"name": "MinIdleTimeoutInMinutes", "shape": "IdleTimeoutInMinutes", "type": "integer"},
            {"name": "MaxIdleTimeoutInMinutes", "shape": "IdleTimeoutInMinutes", "type": "integer"},
        ],
        "type": "structure",
    },
    "Image": {
        "members": [
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Description", "shape": "ImageDescription", "type": "string"},
            {"name": "DisplayName", "shape": "ImageDisplayName", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ImageArn", "shape": "ImageArn", "type": "string"},
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "ImageStatus", "shape": "ImageStatus", "type": "string"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ImageClassificationJobConfig": {
        "members": [
            {
                "name": "CompletionCriteria",
                "shape": "AutoMLJobCompletionCriteria",
                "type": "structure",
            }
        ],
        "type": "structure",
    },
    "ImageConfig": {
        "members": [
            {"name": "RepositoryAccessMode", "shape": "RepositoryAccessMode", "type": "string"},
            {"name": "RepositoryAuthConfig", "shape": "RepositoryAuthConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "ImageDeletePropertyList": {
        "member_shape": "ImageDeleteProperty",
        "member_type": "string",
        "type": "list",
    },
    "ImageVersion": {
        "members": [
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ImageArn", "shape": "ImageArn", "type": "string"},
            {"name": "ImageVersionArn", "shape": "ImageVersionArn", "type": "string"},
            {"name": "ImageVersionStatus", "shape": "ImageVersionStatus", "type": "string"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Version", "shape": "ImageVersionNumber", "type": "integer"},
        ],
        "type": "structure",
    },
    "ImageVersions": {"member_shape": "ImageVersion", "member_type": "structure", "type": "list"},
    "Images": {"member_shape": "Image", "member_type": "structure", "type": "list"},
    "ImportHubContentRequest": {
        "members": [
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "HubContentVersion", "shape": "HubContentVersion", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "DocumentSchemaVersion", "shape": "DocumentSchemaVersion", "type": "string"},
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubContentDisplayName", "shape": "HubContentDisplayName", "type": "string"},
            {"name": "HubContentDescription", "shape": "HubContentDescription", "type": "string"},
            {"name": "HubContentMarkdown", "shape": "HubContentMarkdown", "type": "string"},
            {"name": "HubContentDocument", "shape": "HubContentDocument", "type": "string"},
            {"name": "SupportStatus", "shape": "HubContentSupportStatus", "type": "string"},
            {
                "name": "HubContentSearchKeywords",
                "shape": "HubContentSearchKeywordList",
                "type": "list",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "ImportHubContentResponse": {
        "members": [
            {"name": "HubArn", "shape": "HubArn", "type": "string"},
            {"name": "HubContentArn", "shape": "HubContentArn", "type": "string"},
        ],
        "type": "structure",
    },
    "InferenceComponentCapacitySize": {
        "members": [
            {"name": "Type", "shape": "InferenceComponentCapacitySizeType", "type": "string"},
            {"name": "Value", "shape": "CapacitySizeValue", "type": "integer"},
        ],
        "type": "structure",
    },
    "InferenceComponentComputeResourceRequirements": {
        "members": [
            {"name": "NumberOfCpuCoresRequired", "shape": "NumberOfCpuCores", "type": "float"},
            {
                "name": "NumberOfAcceleratorDevicesRequired",
                "shape": "NumberOfAcceleratorDevices",
                "type": "float",
            },
            {"name": "MinMemoryRequiredInMb", "shape": "MemoryInMb", "type": "integer"},
            {"name": "MaxMemoryRequiredInMb", "shape": "MemoryInMb", "type": "integer"},
        ],
        "type": "structure",
    },
    "InferenceComponentContainerSpecification": {
        "members": [
            {"name": "Image", "shape": "ContainerImage", "type": "string"},
            {"name": "ArtifactUrl", "shape": "Url", "type": "string"},
            {"name": "Environment", "shape": "EnvironmentMap", "type": "map"},
        ],
        "type": "structure",
    },
    "InferenceComponentContainerSpecificationSummary": {
        "members": [
            {"name": "DeployedImage", "shape": "DeployedImage", "type": "structure"},
            {"name": "ArtifactUrl", "shape": "Url", "type": "string"},
            {"name": "Environment", "shape": "EnvironmentMap", "type": "map"},
        ],
        "type": "structure",
    },
    "InferenceComponentDeploymentConfig": {
        "members": [
            {
                "name": "RollingUpdatePolicy",
                "shape": "InferenceComponentRollingUpdatePolicy",
                "type": "structure",
            },
            {
                "name": "AutoRollbackConfiguration",
                "shape": "AutoRollbackConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "InferenceComponentRollingUpdatePolicy": {
        "members": [
            {
                "name": "MaximumBatchSize",
                "shape": "InferenceComponentCapacitySize",
                "type": "structure",
            },
            {"name": "WaitIntervalInSeconds", "shape": "WaitIntervalInSeconds", "type": "integer"},
            {
                "name": "MaximumExecutionTimeoutInSeconds",
                "shape": "MaximumExecutionTimeoutInSeconds",
                "type": "integer",
            },
            {
                "name": "RollbackMaximumBatchSize",
                "shape": "InferenceComponentCapacitySize",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "InferenceComponentRuntimeConfig": {
        "members": [
            {"name": "CopyCount", "shape": "InferenceComponentCopyCount", "type": "integer"}
        ],
        "type": "structure",
    },
    "InferenceComponentRuntimeConfigSummary": {
        "members": [
            {"name": "DesiredCopyCount", "shape": "InferenceComponentCopyCount", "type": "integer"},
            {"name": "CurrentCopyCount", "shape": "InferenceComponentCopyCount", "type": "integer"},
        ],
        "type": "structure",
    },
    "InferenceComponentSpecification": {
        "members": [
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {
                "name": "Container",
                "shape": "InferenceComponentContainerSpecification",
                "type": "structure",
            },
            {
                "name": "StartupParameters",
                "shape": "InferenceComponentStartupParameters",
                "type": "structure",
            },
            {
                "name": "ComputeResourceRequirements",
                "shape": "InferenceComponentComputeResourceRequirements",
                "type": "structure",
            },
            {
                "name": "BaseInferenceComponentName",
                "shape": "InferenceComponentName",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "InferenceComponentSpecificationSummary": {
        "members": [
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {
                "name": "Container",
                "shape": "InferenceComponentContainerSpecificationSummary",
                "type": "structure",
            },
            {
                "name": "StartupParameters",
                "shape": "InferenceComponentStartupParameters",
                "type": "structure",
            },
            {
                "name": "ComputeResourceRequirements",
                "shape": "InferenceComponentComputeResourceRequirements",
                "type": "structure",
            },
            {
                "name": "BaseInferenceComponentName",
                "shape": "InferenceComponentName",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "InferenceComponentStartupParameters": {
        "members": [
            {
                "name": "ModelDataDownloadTimeoutInSeconds",
                "shape": "ProductionVariantModelDataDownloadTimeoutInSeconds",
                "type": "integer",
            },
            {
                "name": "ContainerStartupHealthCheckTimeoutInSeconds",
                "shape": "ProductionVariantContainerStartupHealthCheckTimeoutInSeconds",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "InferenceComponentSummary": {
        "members": [
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "InferenceComponentArn", "shape": "InferenceComponentArn", "type": "string"},
            {"name": "InferenceComponentName", "shape": "InferenceComponentName", "type": "string"},
            {"name": "EndpointArn", "shape": "EndpointArn", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "VariantName", "shape": "VariantName", "type": "string"},
            {
                "name": "InferenceComponentStatus",
                "shape": "InferenceComponentStatus",
                "type": "string",
            },
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "InferenceComponentSummaryList": {
        "member_shape": "InferenceComponentSummary",
        "member_type": "structure",
        "type": "list",
    },
    "InferenceExecutionConfig": {
        "members": [{"name": "Mode", "shape": "InferenceExecutionMode", "type": "string"}],
        "type": "structure",
    },
    "InferenceExperimentDataStorageConfig": {
        "members": [
            {"name": "Destination", "shape": "DestinationS3Uri", "type": "string"},
            {"name": "KmsKey", "shape": "KmsKeyId", "type": "string"},
            {"name": "ContentType", "shape": "CaptureContentTypeHeader", "type": "structure"},
        ],
        "type": "structure",
    },
    "InferenceExperimentList": {
        "member_shape": "InferenceExperimentSummary",
        "member_type": "structure",
        "type": "list",
    },
    "InferenceExperimentSchedule": {
        "members": [
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "InferenceExperimentSummary": {
        "members": [
            {"name": "Name", "shape": "InferenceExperimentName", "type": "string"},
            {"name": "Type", "shape": "InferenceExperimentType", "type": "string"},
            {"name": "Schedule", "shape": "InferenceExperimentSchedule", "type": "structure"},
            {"name": "Status", "shape": "InferenceExperimentStatus", "type": "string"},
            {"name": "StatusReason", "shape": "InferenceExperimentStatusReason", "type": "string"},
            {"name": "Description", "shape": "InferenceExperimentDescription", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CompletionTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
        ],
        "type": "structure",
    },
    "InferenceHubAccessConfig": {
        "members": [{"name": "HubContentArn", "shape": "HubContentArn", "type": "string"}],
        "type": "structure",
    },
    "InferenceMetrics": {
        "members": [
            {"name": "MaxInvocations", "shape": "Integer", "type": "integer"},
            {"name": "ModelLatency", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "InferenceRecommendation": {
        "members": [
            {"name": "RecommendationId", "shape": "String", "type": "string"},
            {"name": "Metrics", "shape": "RecommendationMetrics", "type": "structure"},
            {
                "name": "EndpointConfiguration",
                "shape": "EndpointOutputConfiguration",
                "type": "structure",
            },
            {"name": "ModelConfiguration", "shape": "ModelConfiguration", "type": "structure"},
            {"name": "InvocationEndTime", "shape": "InvocationEndTime", "type": "timestamp"},
            {"name": "InvocationStartTime", "shape": "InvocationStartTime", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "InferenceRecommendations": {
        "member_shape": "InferenceRecommendation",
        "member_type": "structure",
        "type": "list",
    },
    "InferenceRecommendationsJob": {
        "members": [
            {"name": "JobName", "shape": "RecommendationJobName", "type": "string"},
            {"name": "JobDescription", "shape": "RecommendationJobDescription", "type": "string"},
            {"name": "JobType", "shape": "RecommendationJobType", "type": "string"},
            {"name": "JobArn", "shape": "RecommendationJobArn", "type": "string"},
            {"name": "Status", "shape": "RecommendationJobStatus", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CompletionTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "SamplePayloadUrl", "shape": "S3Uri", "type": "string"},
            {"name": "ModelPackageVersionArn", "shape": "ModelPackageArn", "type": "string"},
        ],
        "type": "structure",
    },
    "InferenceRecommendationsJobStep": {
        "members": [
            {"name": "StepType", "shape": "RecommendationStepType", "type": "string"},
            {"name": "JobName", "shape": "RecommendationJobName", "type": "string"},
            {"name": "Status", "shape": "RecommendationJobStatus", "type": "string"},
            {
                "name": "InferenceBenchmark",
                "shape": "RecommendationJobInferenceBenchmark",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "InferenceRecommendationsJobSteps": {
        "member_shape": "InferenceRecommendationsJobStep",
        "member_type": "structure",
        "type": "list",
    },
    "InferenceRecommendationsJobs": {
        "member_shape": "InferenceRecommendationsJob",
        "member_type": "structure",
        "type": "list",
    },
    "InferenceSpecification": {
        "members": [
            {"name": "Containers", "shape": "ModelPackageContainerDefinitionList", "type": "list"},
            {
                "name": "SupportedTransformInstanceTypes",
                "shape": "TransformInstanceTypes",
                "type": "list",
            },
            {
                "name": "SupportedRealtimeInferenceInstanceTypes",
                "shape": "RealtimeInferenceInstanceTypes",
                "type": "list",
            },
            {"name": "SupportedContentTypes", "shape": "ContentTypes", "type": "list"},
            {"name": "SupportedResponseMIMETypes", "shape": "ResponseMIMETypes", "type": "list"},
        ],
        "type": "structure",
    },
    "InfraCheckConfig": {
        "members": [{"name": "EnableInfraCheck", "shape": "EnableInfraCheck", "type": "boolean"}],
        "type": "structure",
    },
    "InputConfig": {
        "members": [
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "DataInputConfig", "shape": "DataInputConfig", "type": "string"},
            {"name": "Framework", "shape": "Framework", "type": "string"},
            {"name": "FrameworkVersion", "shape": "FrameworkVersion", "type": "string"},
        ],
        "type": "structure",
    },
    "InputDataConfig": {"member_shape": "Channel", "member_type": "structure", "type": "list"},
    "InputModes": {"member_shape": "TrainingInputMode", "member_type": "string", "type": "list"},
    "InstanceGroup": {
        "members": [
            {"name": "InstanceType", "shape": "TrainingInstanceType", "type": "string"},
            {"name": "InstanceCount", "shape": "TrainingInstanceCount", "type": "integer"},
            {"name": "InstanceGroupName", "shape": "InstanceGroupName", "type": "string"},
        ],
        "type": "structure",
    },
    "InstanceGroupNames": {
        "member_shape": "InstanceGroupName",
        "member_type": "string",
        "type": "list",
    },
    "InstanceGroups": {"member_shape": "InstanceGroup", "member_type": "structure", "type": "list"},
    "InstanceMetadataServiceConfiguration": {
        "members": [
            {
                "name": "MinimumInstanceMetadataServiceVersion",
                "shape": "MinimumInstanceMetadataServiceVersion",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "IntegerParameterRange": {
        "members": [
            {"name": "Name", "shape": "ParameterKey", "type": "string"},
            {"name": "MinValue", "shape": "ParameterValue", "type": "string"},
            {"name": "MaxValue", "shape": "ParameterValue", "type": "string"},
            {"name": "ScalingType", "shape": "HyperParameterScalingType", "type": "string"},
        ],
        "type": "structure",
    },
    "IntegerParameterRangeSpecification": {
        "members": [
            {"name": "MinValue", "shape": "ParameterValue", "type": "string"},
            {"name": "MaxValue", "shape": "ParameterValue", "type": "string"},
        ],
        "type": "structure",
    },
    "IntegerParameterRanges": {
        "member_shape": "IntegerParameterRange",
        "member_type": "structure",
        "type": "list",
    },
    "InternalDependencyException": {
        "members": [{"name": "Message", "shape": "Message", "type": "string"}],
        "type": "structure",
    },
    "InternalFailure": {
        "members": [{"name": "Message", "shape": "Message", "type": "string"}],
        "type": "structure",
    },
    "InternalStreamFailure": {
        "members": [{"name": "Message", "shape": "Message", "type": "string"}],
        "type": "structure",
    },
    "InvokeEndpointAsyncInput": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "ContentType", "shape": "Header", "type": "string"},
            {"name": "Accept", "shape": "Header", "type": "string"},
            {"name": "CustomAttributes", "shape": "CustomAttributesHeader", "type": "string"},
            {"name": "InferenceId", "shape": "InferenceId", "type": "string"},
            {"name": "InputLocation", "shape": "InputLocationHeader", "type": "string"},
            {"name": "RequestTTLSeconds", "shape": "RequestTTLSecondsHeader", "type": "integer"},
            {
                "name": "InvocationTimeoutSeconds",
                "shape": "InvocationTimeoutSecondsHeader",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "InvokeEndpointAsyncOutput": {
        "members": [
            {"name": "InferenceId", "shape": "Header", "type": "string"},
            {"name": "OutputLocation", "shape": "Header", "type": "string"},
            {"name": "FailureLocation", "shape": "Header", "type": "string"},
        ],
        "type": "structure",
    },
    "InvokeEndpointInput": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "Body", "shape": "BodyBlob", "type": "blob"},
            {"name": "ContentType", "shape": "Header", "type": "string"},
            {"name": "Accept", "shape": "Header", "type": "string"},
            {"name": "CustomAttributes", "shape": "CustomAttributesHeader", "type": "string"},
            {"name": "TargetModel", "shape": "TargetModelHeader", "type": "string"},
            {"name": "TargetVariant", "shape": "TargetVariantHeader", "type": "string"},
            {
                "name": "TargetContainerHostname",
                "shape": "TargetContainerHostnameHeader",
                "type": "string",
            },
            {"name": "InferenceId", "shape": "InferenceId", "type": "string"},
            {"name": "EnableExplanations", "shape": "EnableExplanationsHeader", "type": "string"},
            {
                "name": "InferenceComponentName",
                "shape": "InferenceComponentHeader",
                "type": "string",
            },
            {"name": "SessionId", "shape": "SessionIdOrNewSessionConstantHeader", "type": "string"},
        ],
        "type": "structure",
    },
    "InvokeEndpointOutput": {
        "members": [
            {"name": "Body", "shape": "BodyBlob", "type": "blob"},
            {"name": "ContentType", "shape": "Header", "type": "string"},
            {"name": "InvokedProductionVariant", "shape": "Header", "type": "string"},
            {"name": "CustomAttributes", "shape": "CustomAttributesHeader", "type": "string"},
            {"name": "NewSessionId", "shape": "NewSessionResponseHeader", "type": "string"},
            {"name": "ClosedSessionId", "shape": "SessionIdHeader", "type": "string"},
        ],
        "type": "structure",
    },
    "InvokeEndpointWithResponseStreamInput": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "Body", "shape": "BodyBlob", "type": "blob"},
            {"name": "ContentType", "shape": "Header", "type": "string"},
            {"name": "Accept", "shape": "Header", "type": "string"},
            {"name": "CustomAttributes", "shape": "CustomAttributesHeader", "type": "string"},
            {"name": "TargetVariant", "shape": "TargetVariantHeader", "type": "string"},
            {
                "name": "TargetContainerHostname",
                "shape": "TargetContainerHostnameHeader",
                "type": "string",
            },
            {"name": "InferenceId", "shape": "InferenceId", "type": "string"},
            {
                "name": "InferenceComponentName",
                "shape": "InferenceComponentHeader",
                "type": "string",
            },
            {"name": "SessionId", "shape": "SessionIdHeader", "type": "string"},
        ],
        "type": "structure",
    },
    "InvokeEndpointWithResponseStreamOutput": {
        "members": [
            {"name": "Body", "shape": "ResponseStream", "type": "structure"},
            {"name": "ContentType", "shape": "Header", "type": "string"},
            {"name": "InvokedProductionVariant", "shape": "Header", "type": "string"},
            {"name": "CustomAttributes", "shape": "CustomAttributesHeader", "type": "string"},
        ],
        "type": "structure",
    },
    "JsonContentTypes": {
        "member_shape": "JsonContentType",
        "member_type": "string",
        "type": "list",
    },
    "JupyterLabAppImageConfig": {
        "members": [
            {"name": "FileSystemConfig", "shape": "FileSystemConfig", "type": "structure"},
            {"name": "ContainerConfig", "shape": "ContainerConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "JupyterLabAppSettings": {
        "members": [
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {"name": "CustomImages", "shape": "CustomImages", "type": "list"},
            {"name": "LifecycleConfigArns", "shape": "LifecycleConfigArns", "type": "list"},
            {"name": "CodeRepositories", "shape": "CodeRepositories", "type": "list"},
            {
                "name": "AppLifecycleManagement",
                "shape": "AppLifecycleManagement",
                "type": "structure",
            },
            {"name": "EmrSettings", "shape": "EmrSettings", "type": "structure"},
            {
                "name": "BuiltInLifecycleConfigArn",
                "shape": "StudioLifecycleConfigArn",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "JupyterServerAppSettings": {
        "members": [
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {"name": "LifecycleConfigArns", "shape": "LifecycleConfigArns", "type": "list"},
            {"name": "CodeRepositories", "shape": "CodeRepositories", "type": "list"},
        ],
        "type": "structure",
    },
    "KendraSettings": {
        "members": [{"name": "Status", "shape": "FeatureStatus", "type": "string"}],
        "type": "structure",
    },
    "KernelGatewayAppSettings": {
        "members": [
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {"name": "CustomImages", "shape": "CustomImages", "type": "list"},
            {"name": "LifecycleConfigArns", "shape": "LifecycleConfigArns", "type": "list"},
        ],
        "type": "structure",
    },
    "KernelGatewayImageConfig": {
        "members": [
            {"name": "KernelSpecs", "shape": "KernelSpecs", "type": "list"},
            {"name": "FileSystemConfig", "shape": "FileSystemConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "KernelSpec": {
        "members": [
            {"name": "Name", "shape": "KernelName", "type": "string"},
            {"name": "DisplayName", "shape": "KernelDisplayName", "type": "string"},
        ],
        "type": "structure",
    },
    "KernelSpecs": {"member_shape": "KernelSpec", "member_type": "structure", "type": "list"},
    "LabelCounters": {
        "members": [
            {"name": "TotalLabeled", "shape": "LabelCounter", "type": "integer"},
            {"name": "HumanLabeled", "shape": "LabelCounter", "type": "integer"},
            {"name": "MachineLabeled", "shape": "LabelCounter", "type": "integer"},
            {"name": "FailedNonRetryableError", "shape": "LabelCounter", "type": "integer"},
            {"name": "Unlabeled", "shape": "LabelCounter", "type": "integer"},
        ],
        "type": "structure",
    },
    "LabelCountersForWorkteam": {
        "members": [
            {"name": "HumanLabeled", "shape": "LabelCounter", "type": "integer"},
            {"name": "PendingHuman", "shape": "LabelCounter", "type": "integer"},
            {"name": "Total", "shape": "LabelCounter", "type": "integer"},
        ],
        "type": "structure",
    },
    "LabelingJobAlgorithmsConfig": {
        "members": [
            {
                "name": "LabelingJobAlgorithmSpecificationArn",
                "shape": "LabelingJobAlgorithmSpecificationArn",
                "type": "string",
            },
            {"name": "InitialActiveLearningModelArn", "shape": "ModelArn", "type": "string"},
            {
                "name": "LabelingJobResourceConfig",
                "shape": "LabelingJobResourceConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "LabelingJobDataAttributes": {
        "members": [{"name": "ContentClassifiers", "shape": "ContentClassifiers", "type": "list"}],
        "type": "structure",
    },
    "LabelingJobDataSource": {
        "members": [
            {"name": "S3DataSource", "shape": "LabelingJobS3DataSource", "type": "structure"},
            {"name": "SnsDataSource", "shape": "LabelingJobSnsDataSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "LabelingJobForWorkteamSummary": {
        "members": [
            {"name": "LabelingJobName", "shape": "LabelingJobName", "type": "string"},
            {"name": "JobReferenceCode", "shape": "JobReferenceCode", "type": "string"},
            {"name": "WorkRequesterAccountId", "shape": "AccountId", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LabelCounters", "shape": "LabelCountersForWorkteam", "type": "structure"},
            {
                "name": "NumberOfHumanWorkersPerDataObject",
                "shape": "NumberOfHumanWorkersPerDataObject",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "LabelingJobForWorkteamSummaryList": {
        "member_shape": "LabelingJobForWorkteamSummary",
        "member_type": "structure",
        "type": "list",
    },
    "LabelingJobInputConfig": {
        "members": [
            {"name": "DataSource", "shape": "LabelingJobDataSource", "type": "structure"},
            {"name": "DataAttributes", "shape": "LabelingJobDataAttributes", "type": "structure"},
        ],
        "type": "structure",
    },
    "LabelingJobOutput": {
        "members": [
            {"name": "OutputDatasetS3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "FinalActiveLearningModelArn", "shape": "ModelArn", "type": "string"},
        ],
        "type": "structure",
    },
    "LabelingJobOutputConfig": {
        "members": [
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "SnsTopicArn", "shape": "SnsTopicArn", "type": "string"},
        ],
        "type": "structure",
    },
    "LabelingJobResourceConfig": {
        "members": [
            {"name": "VolumeKmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "LabelingJobS3DataSource": {
        "members": [{"name": "ManifestS3Uri", "shape": "S3Uri", "type": "string"}],
        "type": "structure",
    },
    "LabelingJobSnsDataSource": {
        "members": [{"name": "SnsTopicArn", "shape": "SnsTopicArn", "type": "string"}],
        "type": "structure",
    },
    "LabelingJobStoppingConditions": {
        "members": [
            {
                "name": "MaxHumanLabeledObjectCount",
                "shape": "MaxHumanLabeledObjectCount",
                "type": "integer",
            },
            {
                "name": "MaxPercentageOfInputDatasetLabeled",
                "shape": "MaxPercentageOfInputDatasetLabeled",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "LabelingJobSummary": {
        "members": [
            {"name": "LabelingJobName", "shape": "LabelingJobName", "type": "string"},
            {"name": "LabelingJobArn", "shape": "LabelingJobArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LabelingJobStatus", "shape": "LabelingJobStatus", "type": "string"},
            {"name": "LabelCounters", "shape": "LabelCounters", "type": "structure"},
            {"name": "WorkteamArn", "shape": "WorkteamArn", "type": "string"},
            {"name": "PreHumanTaskLambdaArn", "shape": "LambdaFunctionArn", "type": "string"},
            {
                "name": "AnnotationConsolidationLambdaArn",
                "shape": "LambdaFunctionArn",
                "type": "string",
            },
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "LabelingJobOutput", "shape": "LabelingJobOutput", "type": "structure"},
            {"name": "InputConfig", "shape": "LabelingJobInputConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "LabelingJobSummaryList": {
        "member_shape": "LabelingJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "LambdaStepMetadata": {
        "members": [
            {"name": "Arn", "shape": "String256", "type": "string"},
            {"name": "OutputParameters", "shape": "OutputParameterList", "type": "list"},
        ],
        "type": "structure",
    },
    "LastUpdateStatus": {
        "members": [
            {"name": "Status", "shape": "LastUpdateStatusValue", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
        ],
        "type": "structure",
    },
    "LifecycleConfigArns": {
        "member_shape": "StudioLifecycleConfigArn",
        "member_type": "string",
        "type": "list",
    },
    "LineageEntityParameters": {
        "key_shape": "StringParameterValue",
        "key_type": "string",
        "type": "map",
        "value_shape": "StringParameterValue",
        "value_type": "string",
    },
    "LineageGroupSummaries": {
        "member_shape": "LineageGroupSummary",
        "member_type": "structure",
        "type": "list",
    },
    "LineageGroupSummary": {
        "members": [
            {"name": "LineageGroupArn", "shape": "LineageGroupArn", "type": "string"},
            {"name": "LineageGroupName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ListActionsRequest": {
        "members": [
            {"name": "SourceUri", "shape": "SourceUri", "type": "string"},
            {"name": "ActionType", "shape": "String256", "type": "string"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortActionsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListActionsResponse": {
        "members": [
            {"name": "ActionSummaries", "shape": "ActionSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAlgorithmsInput": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "AlgorithmSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAlgorithmsOutput": {
        "members": [
            {"name": "AlgorithmSummaryList", "shape": "AlgorithmSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAliasesRequest": {
        "members": [
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "Alias", "shape": "SageMakerImageVersionAlias", "type": "string"},
            {"name": "Version", "shape": "ImageVersionNumber", "type": "integer"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAliasesResponse": {
        "members": [
            {
                "name": "SageMakerImageVersionAliases",
                "shape": "SageMakerImageVersionAliases",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAppImageConfigsRequest": {
        "members": [
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "NameContains", "shape": "AppImageConfigName", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "AppImageConfigSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAppImageConfigsResponse": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "AppImageConfigs", "shape": "AppImageConfigList", "type": "list"},
        ],
        "type": "structure",
    },
    "ListAppsRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "SortBy", "shape": "AppSortKey", "type": "string"},
            {"name": "DomainIdEquals", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileNameEquals", "shape": "UserProfileName", "type": "string"},
            {"name": "SpaceNameEquals", "shape": "SpaceName", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAppsResponse": {
        "members": [
            {"name": "Apps", "shape": "AppList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListArtifactsRequest": {
        "members": [
            {"name": "SourceUri", "shape": "SourceUri", "type": "string"},
            {"name": "ArtifactType", "shape": "String256", "type": "string"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortArtifactsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListArtifactsResponse": {
        "members": [
            {"name": "ArtifactSummaries", "shape": "ArtifactSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAssociationsRequest": {
        "members": [
            {"name": "SourceArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "DestinationArn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "SourceType", "shape": "String256", "type": "string"},
            {"name": "DestinationType", "shape": "String256", "type": "string"},
            {"name": "AssociationType", "shape": "AssociationEdgeType", "type": "string"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortAssociationsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListAssociationsResponse": {
        "members": [
            {"name": "AssociationSummaries", "shape": "AssociationSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAutoMLJobsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "AutoMLNameContains", "type": "string"},
            {"name": "StatusEquals", "shape": "AutoMLJobStatus", "type": "string"},
            {"name": "SortOrder", "shape": "AutoMLSortOrder", "type": "string"},
            {"name": "SortBy", "shape": "AutoMLSortBy", "type": "string"},
            {"name": "MaxResults", "shape": "AutoMLMaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListAutoMLJobsResponse": {
        "members": [
            {"name": "AutoMLJobSummaries", "shape": "AutoMLJobSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListCandidatesForAutoMLJobRequest": {
        "members": [
            {"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"},
            {"name": "StatusEquals", "shape": "CandidateStatus", "type": "string"},
            {"name": "CandidateNameEquals", "shape": "CandidateName", "type": "string"},
            {"name": "SortOrder", "shape": "AutoMLSortOrder", "type": "string"},
            {"name": "SortBy", "shape": "CandidateSortBy", "type": "string"},
            {"name": "MaxResults", "shape": "AutoMLMaxResultsForTrials", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListCandidatesForAutoMLJobResponse": {
        "members": [
            {"name": "Candidates", "shape": "AutoMLCandidates", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListClusterNodesRequest": {
        "members": [
            {"name": "ClusterName", "shape": "ClusterNameOrArn", "type": "string"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "InstanceGroupNameContains",
                "shape": "ClusterInstanceGroupName",
                "type": "string",
            },
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ClusterSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListClusterNodesResponse": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "ClusterNodeSummaries", "shape": "ClusterNodeSummaries", "type": "list"},
        ],
        "type": "structure",
    },
    "ListClusterSchedulerConfigsRequest": {
        "members": [
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "EntityName", "type": "string"},
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "Status", "shape": "SchedulerResourceStatus", "type": "string"},
            {"name": "SortBy", "shape": "SortClusterSchedulerConfigBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListClusterSchedulerConfigsResponse": {
        "members": [
            {
                "name": "ClusterSchedulerConfigSummaries",
                "shape": "ClusterSchedulerConfigSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListClustersRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ClusterSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "TrainingPlanArn", "shape": "TrainingPlanArn", "type": "string"},
        ],
        "type": "structure",
    },
    "ListClustersResponse": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "ClusterSummaries", "shape": "ClusterSummaries", "type": "list"},
        ],
        "type": "structure",
    },
    "ListCodeRepositoriesInput": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "CodeRepositoryNameContains", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "CodeRepositorySortBy", "type": "string"},
            {"name": "SortOrder", "shape": "CodeRepositorySortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListCodeRepositoriesOutput": {
        "members": [
            {
                "name": "CodeRepositorySummaryList",
                "shape": "CodeRepositorySummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListCompilationJobsRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "StatusEquals", "shape": "CompilationJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "ListCompilationJobsSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListCompilationJobsResponse": {
        "members": [
            {"name": "CompilationJobSummaries", "shape": "CompilationJobSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListComputeQuotasRequest": {
        "members": [
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "EntityName", "type": "string"},
            {"name": "Status", "shape": "SchedulerResourceStatus", "type": "string"},
            {"name": "ClusterArn", "shape": "ClusterArn", "type": "string"},
            {"name": "SortBy", "shape": "SortQuotaBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListComputeQuotasResponse": {
        "members": [
            {"name": "ComputeQuotaSummaries", "shape": "ComputeQuotaSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListContextsRequest": {
        "members": [
            {"name": "SourceUri", "shape": "SourceUri", "type": "string"},
            {"name": "ContextType", "shape": "String256", "type": "string"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortContextsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListContextsResponse": {
        "members": [
            {"name": "ContextSummaries", "shape": "ContextSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListDataQualityJobDefinitionsRequest": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "SortBy", "shape": "MonitoringJobDefinitionSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ListDataQualityJobDefinitionsResponse": {
        "members": [
            {
                "name": "JobDefinitionSummaries",
                "shape": "MonitoringJobDefinitionSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListDeviceFleetsRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "ListMaxResults", "type": "integer"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "SortBy", "shape": "ListDeviceFleetsSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListDeviceFleetsResponse": {
        "members": [
            {"name": "DeviceFleetSummaries", "shape": "DeviceFleetSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListDevicesRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "ListMaxResults", "type": "integer"},
            {"name": "LatestHeartbeatAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModelName", "shape": "EntityName", "type": "string"},
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "ListDevicesResponse": {
        "members": [
            {"name": "DeviceSummaries", "shape": "DeviceSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListDomainsRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListDomainsResponse": {
        "members": [
            {"name": "Domains", "shape": "DomainList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListEdgeDeploymentPlansRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "ListMaxResults", "type": "integer"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "DeviceFleetNameContains", "shape": "NameContains", "type": "string"},
            {"name": "SortBy", "shape": "ListEdgeDeploymentPlansSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListEdgeDeploymentPlansResponse": {
        "members": [
            {
                "name": "EdgeDeploymentPlanSummaries",
                "shape": "EdgeDeploymentPlanSummaries",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListEdgePackagingJobsRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "ListMaxResults", "type": "integer"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "ModelNameContains", "shape": "NameContains", "type": "string"},
            {"name": "StatusEquals", "shape": "EdgePackagingJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "ListEdgePackagingJobsSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListEdgePackagingJobsResponse": {
        "members": [
            {
                "name": "EdgePackagingJobSummaries",
                "shape": "EdgePackagingJobSummaries",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListEndpointConfigsInput": {
        "members": [
            {"name": "SortBy", "shape": "EndpointConfigSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "OrderKey", "type": "string"},
            {"name": "NextToken", "shape": "PaginationToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "EndpointConfigNameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ListEndpointConfigsOutput": {
        "members": [
            {"name": "EndpointConfigs", "shape": "EndpointConfigSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "PaginationToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListEndpointsInput": {
        "members": [
            {"name": "SortBy", "shape": "EndpointSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "OrderKey", "type": "string"},
            {"name": "NextToken", "shape": "PaginationToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "EndpointNameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StatusEquals", "shape": "EndpointStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ListEndpointsOutput": {
        "members": [
            {"name": "Endpoints", "shape": "EndpointSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "PaginationToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListExperimentsRequest": {
        "members": [
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortExperimentsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListExperimentsResponse": {
        "members": [
            {"name": "ExperimentSummaries", "shape": "ExperimentSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListFeatureGroupsRequest": {
        "members": [
            {"name": "NameContains", "shape": "FeatureGroupNameContains", "type": "string"},
            {"name": "FeatureGroupStatusEquals", "shape": "FeatureGroupStatus", "type": "string"},
            {
                "name": "OfflineStoreStatusEquals",
                "shape": "OfflineStoreStatusValue",
                "type": "string",
            },
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "SortOrder", "shape": "FeatureGroupSortOrder", "type": "string"},
            {"name": "SortBy", "shape": "FeatureGroupSortBy", "type": "string"},
            {"name": "MaxResults", "shape": "FeatureGroupMaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListFeatureGroupsResponse": {
        "members": [
            {"name": "FeatureGroupSummaries", "shape": "FeatureGroupSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListFlowDefinitionsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListFlowDefinitionsResponse": {
        "members": [
            {"name": "FlowDefinitionSummaries", "shape": "FlowDefinitionSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHubContentVersionsRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "MinVersion", "shape": "HubContentVersion", "type": "string"},
            {"name": "MaxSchemaVersion", "shape": "DocumentSchemaVersion", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "HubContentSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHubContentVersionsResponse": {
        "members": [
            {"name": "HubContentSummaries", "shape": "HubContentInfoList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHubContentsRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "MaxSchemaVersion", "shape": "DocumentSchemaVersion", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "HubContentSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHubContentsResponse": {
        "members": [
            {"name": "HubContentSummaries", "shape": "HubContentInfoList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHubsRequest": {
        "members": [
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "HubSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHubsResponse": {
        "members": [
            {"name": "HubSummaries", "shape": "HubInfoList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHumanTaskUisRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListHumanTaskUisResponse": {
        "members": [
            {"name": "HumanTaskUiSummaries", "shape": "HumanTaskUiSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHyperParameterTuningJobsRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "SortBy", "shape": "HyperParameterTuningJobSortByOptions", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StatusEquals", "shape": "HyperParameterTuningJobStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ListHyperParameterTuningJobsResponse": {
        "members": [
            {
                "name": "HyperParameterTuningJobSummaries",
                "shape": "HyperParameterTuningJobSummaries",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListImageVersionsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ImageVersionSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "ImageVersionSortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListImageVersionsResponse": {
        "members": [
            {"name": "ImageVersions", "shape": "ImageVersions", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListImagesRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "ImageNameContains", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ImageSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "ImageSortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListImagesResponse": {
        "members": [
            {"name": "Images", "shape": "Images", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListInferenceComponentsInput": {
        "members": [
            {"name": "SortBy", "shape": "InferenceComponentSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "OrderKey", "type": "string"},
            {"name": "NextToken", "shape": "PaginationToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "InferenceComponentNameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StatusEquals", "shape": "InferenceComponentStatus", "type": "string"},
            {"name": "EndpointNameEquals", "shape": "EndpointName", "type": "string"},
            {"name": "VariantNameEquals", "shape": "VariantName", "type": "string"},
        ],
        "type": "structure",
    },
    "ListInferenceComponentsOutput": {
        "members": [
            {
                "name": "InferenceComponents",
                "shape": "InferenceComponentSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "PaginationToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListInferenceExperimentsRequest": {
        "members": [
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "Type", "shape": "InferenceExperimentType", "type": "string"},
            {"name": "StatusEquals", "shape": "InferenceExperimentStatus", "type": "string"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortInferenceExperimentsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListInferenceExperimentsResponse": {
        "members": [
            {"name": "InferenceExperiments", "shape": "InferenceExperimentList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListInferenceRecommendationsJobStepsRequest": {
        "members": [
            {"name": "JobName", "shape": "RecommendationJobName", "type": "string"},
            {"name": "Status", "shape": "RecommendationJobStatus", "type": "string"},
            {"name": "StepType", "shape": "RecommendationStepType", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListInferenceRecommendationsJobStepsResponse": {
        "members": [
            {"name": "Steps", "shape": "InferenceRecommendationsJobSteps", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListInferenceRecommendationsJobsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "StatusEquals", "shape": "RecommendationJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "ListInferenceRecommendationsJobsSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "ModelNameEquals", "shape": "ModelName", "type": "string"},
            {"name": "ModelPackageVersionArnEquals", "shape": "ModelPackageArn", "type": "string"},
        ],
        "type": "structure",
    },
    "ListInferenceRecommendationsJobsResponse": {
        "members": [
            {
                "name": "InferenceRecommendationsJobs",
                "shape": "InferenceRecommendationsJobs",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListLabelingJobsForWorkteamRequest": {
        "members": [
            {"name": "WorkteamArn", "shape": "WorkteamArn", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "JobReferenceCodeContains",
                "shape": "JobReferenceCodeContains",
                "type": "string",
            },
            {
                "name": "SortBy",
                "shape": "ListLabelingJobsForWorkteamSortByOptions",
                "type": "string",
            },
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListLabelingJobsForWorkteamResponse": {
        "members": [
            {
                "name": "LabelingJobSummaryList",
                "shape": "LabelingJobForWorkteamSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListLabelingJobsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "SortBy", "shape": "SortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "StatusEquals", "shape": "LabelingJobStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ListLabelingJobsResponse": {
        "members": [
            {"name": "LabelingJobSummaryList", "shape": "LabelingJobSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListLineageEntityParameterKey": {
        "member_shape": "StringParameterValue",
        "member_type": "string",
        "type": "list",
    },
    "ListLineageGroupsRequest": {
        "members": [
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortLineageGroupsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListLineageGroupsResponse": {
        "members": [
            {"name": "LineageGroupSummaries", "shape": "LineageGroupSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListMlflowTrackingServersRequest": {
        "members": [
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrackingServerStatus", "shape": "TrackingServerStatus", "type": "string"},
            {"name": "MlflowVersion", "shape": "MlflowVersion", "type": "string"},
            {"name": "SortBy", "shape": "SortTrackingServerBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListMlflowTrackingServersResponse": {
        "members": [
            {
                "name": "TrackingServerSummaries",
                "shape": "TrackingServerSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelBiasJobDefinitionsRequest": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "SortBy", "shape": "MonitoringJobDefinitionSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ListModelBiasJobDefinitionsResponse": {
        "members": [
            {
                "name": "JobDefinitionSummaries",
                "shape": "MonitoringJobDefinitionSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelCardExportJobsRequest": {
        "members": [
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModelCardExportJobNameContains", "shape": "EntityName", "type": "string"},
            {"name": "StatusEquals", "shape": "ModelCardExportJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "ModelCardExportJobSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "ModelCardExportJobSortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListModelCardExportJobsResponse": {
        "members": [
            {
                "name": "ModelCardExportJobSummaries",
                "shape": "ModelCardExportJobSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelCardVersionsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "ModelCardName", "shape": "ModelCardNameOrArn", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ModelCardVersionSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "ModelCardSortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelCardVersionsResponse": {
        "members": [
            {
                "name": "ModelCardVersionSummaryList",
                "shape": "ModelCardVersionSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelCardsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ModelCardSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "ModelCardSortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelCardsResponse": {
        "members": [
            {"name": "ModelCardSummaries", "shape": "ModelCardSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelExplainabilityJobDefinitionsRequest": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "SortBy", "shape": "MonitoringJobDefinitionSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ListModelExplainabilityJobDefinitionsResponse": {
        "members": [
            {
                "name": "JobDefinitionSummaries",
                "shape": "MonitoringJobDefinitionSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelMetadataRequest": {
        "members": [
            {
                "name": "SearchExpression",
                "shape": "ModelMetadataSearchExpression",
                "type": "structure",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListModelMetadataResponse": {
        "members": [
            {"name": "ModelMetadataSummaries", "shape": "ModelMetadataSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelPackageGroupsInput": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ModelPackageGroupSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {
                "name": "CrossAccountFilterOption",
                "shape": "CrossAccountFilterOption",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "ListModelPackageGroupsOutput": {
        "members": [
            {
                "name": "ModelPackageGroupSummaryList",
                "shape": "ModelPackageGroupSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelPackagesInput": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "ModelApprovalStatus", "shape": "ModelApprovalStatus", "type": "string"},
            {"name": "ModelPackageGroupName", "shape": "ArnOrName", "type": "string"},
            {"name": "ModelPackageType", "shape": "ModelPackageType", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ModelPackageSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelPackagesOutput": {
        "members": [
            {"name": "ModelPackageSummaryList", "shape": "ModelPackageSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelQualityJobDefinitionsRequest": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "SortBy", "shape": "MonitoringJobDefinitionSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ListModelQualityJobDefinitionsResponse": {
        "members": [
            {
                "name": "JobDefinitionSummaries",
                "shape": "MonitoringJobDefinitionSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListModelsInput": {
        "members": [
            {"name": "SortBy", "shape": "ModelSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "OrderKey", "type": "string"},
            {"name": "NextToken", "shape": "PaginationToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "ModelNameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ListModelsOutput": {
        "members": [
            {"name": "Models", "shape": "ModelSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "PaginationToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListMonitoringAlertHistoryRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "MonitoringAlertName", "shape": "MonitoringAlertName", "type": "string"},
            {"name": "SortBy", "shape": "MonitoringAlertHistorySortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StatusEquals", "shape": "MonitoringAlertStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ListMonitoringAlertHistoryResponse": {
        "members": [
            {
                "name": "MonitoringAlertHistory",
                "shape": "MonitoringAlertHistoryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListMonitoringAlertsRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListMonitoringAlertsResponse": {
        "members": [
            {
                "name": "MonitoringAlertSummaries",
                "shape": "MonitoringAlertSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListMonitoringExecutionsRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "SortBy", "shape": "MonitoringExecutionSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "ScheduledTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ScheduledTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StatusEquals", "shape": "ExecutionStatus", "type": "string"},
            {
                "name": "MonitoringJobDefinitionName",
                "shape": "MonitoringJobDefinitionName",
                "type": "string",
            },
            {"name": "MonitoringTypeEquals", "shape": "MonitoringType", "type": "string"},
        ],
        "type": "structure",
    },
    "ListMonitoringExecutionsResponse": {
        "members": [
            {
                "name": "MonitoringExecutionSummaries",
                "shape": "MonitoringExecutionSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListMonitoringSchedulesRequest": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "SortBy", "shape": "MonitoringScheduleSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StatusEquals", "shape": "ScheduleStatus", "type": "string"},
            {
                "name": "MonitoringJobDefinitionName",
                "shape": "MonitoringJobDefinitionName",
                "type": "string",
            },
            {"name": "MonitoringTypeEquals", "shape": "MonitoringType", "type": "string"},
        ],
        "type": "structure",
    },
    "ListMonitoringSchedulesResponse": {
        "members": [
            {
                "name": "MonitoringScheduleSummaries",
                "shape": "MonitoringScheduleSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListNotebookInstanceLifecycleConfigsInput": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "SortBy", "shape": "NotebookInstanceLifecycleConfigSortKey", "type": "string"},
            {
                "name": "SortOrder",
                "shape": "NotebookInstanceLifecycleConfigSortOrder",
                "type": "string",
            },
            {
                "name": "NameContains",
                "shape": "NotebookInstanceLifecycleConfigNameContains",
                "type": "string",
            },
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "LastModifiedTime", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ListNotebookInstanceLifecycleConfigsOutput": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {
                "name": "NotebookInstanceLifecycleConfigs",
                "shape": "NotebookInstanceLifecycleConfigSummaryList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "ListNotebookInstancesInput": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "SortBy", "shape": "NotebookInstanceSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "NotebookInstanceSortOrder", "type": "string"},
            {"name": "NameContains", "shape": "NotebookInstanceNameContains", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "StatusEquals", "shape": "NotebookInstanceStatus", "type": "string"},
            {
                "name": "NotebookInstanceLifecycleConfigNameContains",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {
                "name": "DefaultCodeRepositoryContains",
                "shape": "CodeRepositoryContains",
                "type": "string",
            },
            {
                "name": "AdditionalCodeRepositoryEquals",
                "shape": "CodeRepositoryNameOrUrl",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "ListNotebookInstancesOutput": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "NotebookInstances", "shape": "NotebookInstanceSummaryList", "type": "list"},
        ],
        "type": "structure",
    },
    "ListOptimizationJobsRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "CreationTimeAfter", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "OptimizationContains", "shape": "NameContains", "type": "string"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "StatusEquals", "shape": "OptimizationJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "ListOptimizationJobsSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListOptimizationJobsResponse": {
        "members": [
            {
                "name": "OptimizationJobSummaries",
                "shape": "OptimizationJobSummaries",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListPartnerAppsRequest": {
        "members": [
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListPartnerAppsResponse": {
        "members": [
            {"name": "Summaries", "shape": "PartnerAppSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListPipelineExecutionStepsRequest": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListPipelineExecutionStepsResponse": {
        "members": [
            {
                "name": "PipelineExecutionSteps",
                "shape": "PipelineExecutionStepList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListPipelineExecutionsRequest": {
        "members": [
            {"name": "PipelineName", "shape": "PipelineNameOrArn", "type": "string"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortPipelineExecutionsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListPipelineExecutionsResponse": {
        "members": [
            {
                "name": "PipelineExecutionSummaries",
                "shape": "PipelineExecutionSummaryList",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListPipelineParametersForExecutionRequest": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListPipelineParametersForExecutionResponse": {
        "members": [
            {"name": "PipelineParameters", "shape": "ParameterList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListPipelinesRequest": {
        "members": [
            {"name": "PipelineNamePrefix", "shape": "PipelineName", "type": "string"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortPipelinesBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListPipelinesResponse": {
        "members": [
            {"name": "PipelineSummaries", "shape": "PipelineSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListProcessingJobsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "String", "type": "string"},
            {"name": "StatusEquals", "shape": "ProcessingJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "SortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListProcessingJobsResponse": {
        "members": [
            {"name": "ProcessingJobSummaries", "shape": "ProcessingJobSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListProjectsInput": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NameContains", "shape": "ProjectEntityName", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "SortBy", "shape": "ProjectSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "ProjectSortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListProjectsOutput": {
        "members": [
            {"name": "ProjectSummaryList", "shape": "ProjectSummaryList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListResourceCatalogsRequest": {
        "members": [
            {"name": "NameContains", "shape": "ResourceCatalogName", "type": "string"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortOrder", "shape": "ResourceCatalogSortOrder", "type": "string"},
            {"name": "SortBy", "shape": "ResourceCatalogSortBy", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListResourceCatalogsResponse": {
        "members": [
            {"name": "ResourceCatalogs", "shape": "ResourceCatalogList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListSpacesRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "SortBy", "shape": "SpaceSortKey", "type": "string"},
            {"name": "DomainIdEquals", "shape": "DomainId", "type": "string"},
            {"name": "SpaceNameContains", "shape": "SpaceName", "type": "string"},
        ],
        "type": "structure",
    },
    "ListSpacesResponse": {
        "members": [
            {"name": "Spaces", "shape": "SpaceList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListStageDevicesRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "ListMaxResults", "type": "integer"},
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "ExcludeDevicesDeployedInOtherStage", "shape": "Boolean", "type": "boolean"},
            {"name": "StageName", "shape": "EntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "ListStageDevicesResponse": {
        "members": [
            {
                "name": "DeviceDeploymentSummaries",
                "shape": "DeviceDeploymentSummaries",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListStudioLifecycleConfigsRequest": {
        "members": [
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "NameContains", "shape": "StudioLifecycleConfigName", "type": "string"},
            {"name": "AppTypeEquals", "shape": "StudioLifecycleConfigAppType", "type": "string"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "StudioLifecycleConfigSortKey", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListStudioLifecycleConfigsResponse": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {
                "name": "StudioLifecycleConfigs",
                "shape": "StudioLifecycleConfigsList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "ListSubscribedWorkteamsRequest": {
        "members": [
            {"name": "NameContains", "shape": "WorkteamName", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListSubscribedWorkteamsResponse": {
        "members": [
            {"name": "SubscribedWorkteams", "shape": "SubscribedWorkteams", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTagsInput": {
        "members": [
            {"name": "ResourceArn", "shape": "ResourceArn", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "ListTagsMaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListTagsOutput": {
        "members": [
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrainingJobsForHyperParameterTuningJobRequest": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "StatusEquals", "shape": "TrainingJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "TrainingJobSortByOptions", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrainingJobsForHyperParameterTuningJobResponse": {
        "members": [
            {
                "name": "TrainingJobSummaries",
                "shape": "HyperParameterTrainingJobSummaries",
                "type": "list",
            },
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrainingJobsRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "StatusEquals", "shape": "TrainingJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "SortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "WarmPoolStatusEquals", "shape": "WarmPoolResourceStatus", "type": "string"},
            {"name": "TrainingPlanArnEquals", "shape": "TrainingPlanArn", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrainingJobsResponse": {
        "members": [
            {"name": "TrainingJobSummaries", "shape": "TrainingJobSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrainingPlansRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "StartTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StartTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "TrainingPlanSortBy", "type": "string"},
            {"name": "SortOrder", "shape": "TrainingPlanSortOrder", "type": "string"},
            {"name": "Filters", "shape": "TrainingPlanFilters", "type": "list"},
        ],
        "type": "structure",
    },
    "ListTrainingPlansResponse": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "TrainingPlanSummaries", "shape": "TrainingPlanSummaries", "type": "list"},
        ],
        "type": "structure",
    },
    "ListTransformJobsRequest": {
        "members": [
            {"name": "CreationTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "NameContains", "shape": "NameContains", "type": "string"},
            {"name": "StatusEquals", "shape": "TransformJobStatus", "type": "string"},
            {"name": "SortBy", "shape": "SortBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListTransformJobsResponse": {
        "members": [
            {"name": "TransformJobSummaries", "shape": "TransformJobSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrialComponentKey256": {
        "member_shape": "TrialComponentKey256",
        "member_type": "string",
        "type": "list",
    },
    "ListTrialComponentsRequest": {
        "members": [
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "SourceArn", "shape": "String256", "type": "string"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortTrialComponentsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrialComponentsResponse": {
        "members": [
            {"name": "TrialComponentSummaries", "shape": "TrialComponentSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrialsRequest": {
        "members": [
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SortBy", "shape": "SortTrialsBy", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListTrialsResponse": {
        "members": [
            {"name": "TrialSummaries", "shape": "TrialSummaries", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListUserProfilesRequest": {
        "members": [
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "SortBy", "shape": "UserProfileSortKey", "type": "string"},
            {"name": "DomainIdEquals", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileNameContains", "shape": "UserProfileName", "type": "string"},
        ],
        "type": "structure",
    },
    "ListUserProfilesResponse": {
        "members": [
            {"name": "UserProfiles", "shape": "UserProfileList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListWorkforcesRequest": {
        "members": [
            {"name": "SortBy", "shape": "ListWorkforcesSortByOptions", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NameContains", "shape": "WorkforceName", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListWorkforcesResponse": {
        "members": [
            {"name": "Workforces", "shape": "Workforces", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "ListWorkteamsRequest": {
        "members": [
            {"name": "SortBy", "shape": "ListWorkteamsSortByOptions", "type": "string"},
            {"name": "SortOrder", "shape": "SortOrder", "type": "string"},
            {"name": "NameContains", "shape": "WorkteamName", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
        ],
        "type": "structure",
    },
    "ListWorkteamsResponse": {
        "members": [
            {"name": "Workteams", "shape": "Workteams", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "MemberDefinition": {
        "members": [
            {
                "name": "CognitoMemberDefinition",
                "shape": "CognitoMemberDefinition",
                "type": "structure",
            },
            {"name": "OidcMemberDefinition", "shape": "OidcMemberDefinition", "type": "structure"},
        ],
        "type": "structure",
    },
    "MemberDefinitions": {
        "member_shape": "MemberDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "MetadataProperties": {
        "members": [
            {"name": "CommitId", "shape": "MetadataPropertyValue", "type": "string"},
            {"name": "Repository", "shape": "MetadataPropertyValue", "type": "string"},
            {"name": "GeneratedBy", "shape": "MetadataPropertyValue", "type": "string"},
            {"name": "ProjectId", "shape": "MetadataPropertyValue", "type": "string"},
        ],
        "type": "structure",
    },
    "MetricData": {
        "members": [
            {"name": "MetricName", "shape": "MetricName", "type": "string"},
            {"name": "Value", "shape": "Float", "type": "float"},
            {"name": "Timestamp", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "MetricDataList": {"member_shape": "MetricDatum", "member_type": "structure", "type": "list"},
    "MetricDatum": {
        "members": [
            {"name": "MetricName", "shape": "AutoMLMetricEnum", "type": "string"},
            {"name": "Value", "shape": "Float", "type": "float"},
            {"name": "Set", "shape": "MetricSetSource", "type": "string"},
            {"name": "StandardMetricName", "shape": "AutoMLMetricExtendedEnum", "type": "string"},
        ],
        "type": "structure",
    },
    "MetricDefinition": {
        "members": [
            {"name": "Name", "shape": "MetricName", "type": "string"},
            {"name": "Regex", "shape": "MetricRegex", "type": "string"},
        ],
        "type": "structure",
    },
    "MetricDefinitionList": {
        "member_shape": "MetricDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "MetricQuery": {
        "members": [
            {"name": "MetricName", "shape": "MetricName", "type": "string"},
            {"name": "ResourceArn", "shape": "SageMakerResourceArn", "type": "string"},
            {"name": "MetricStat", "shape": "MetricStatistic", "type": "string"},
            {"name": "Period", "shape": "Period", "type": "string"},
            {"name": "XAxisType", "shape": "XAxisType", "type": "string"},
            {"name": "Start", "shape": "Long", "type": "long"},
            {"name": "End", "shape": "Long", "type": "long"},
        ],
        "type": "structure",
    },
    "MetricQueryList": {"member_shape": "MetricQuery", "member_type": "structure", "type": "list"},
    "MetricQueryResult": {
        "members": [
            {"name": "Status", "shape": "MetricQueryResultStatus", "type": "string"},
            {"name": "Message", "shape": "Message", "type": "string"},
            {"name": "XAxisValues", "shape": "XAxisValues", "type": "list"},
            {"name": "MetricValues", "shape": "MetricValues", "type": "list"},
        ],
        "type": "structure",
    },
    "MetricQueryResultList": {
        "member_shape": "MetricQueryResult",
        "member_type": "structure",
        "type": "list",
    },
    "MetricSpecification": {
        "members": [
            {"name": "Predefined", "shape": "PredefinedMetricSpecification", "type": "structure"},
            {"name": "Customized", "shape": "CustomizedMetricSpecification", "type": "structure"},
        ],
        "type": "structure",
    },
    "MetricValues": {"member_shape": "Double", "member_type": "double", "type": "list"},
    "MetricsSource": {
        "members": [
            {"name": "ContentType", "shape": "ContentType", "type": "string"},
            {"name": "ContentDigest", "shape": "ContentDigest", "type": "string"},
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "Model": {
        "members": [
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "PrimaryContainer", "shape": "ContainerDefinition", "type": "structure"},
            {"name": "Containers", "shape": "ContainerDefinitionList", "type": "list"},
            {
                "name": "InferenceExecutionConfig",
                "shape": "InferenceExecutionConfig",
                "type": "structure",
            },
            {"name": "ExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModelArn", "shape": "ModelArn", "type": "string"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {
                "name": "DeploymentRecommendation",
                "shape": "DeploymentRecommendation",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelAccessConfig": {
        "members": [{"name": "AcceptEula", "shape": "AcceptEula", "type": "boolean"}],
        "type": "structure",
    },
    "ModelArtifacts": {
        "members": [{"name": "S3ModelArtifacts", "shape": "S3Uri", "type": "string"}],
        "type": "structure",
    },
    "ModelBiasAppSpecification": {
        "members": [
            {"name": "ImageUri", "shape": "ImageUri", "type": "string"},
            {"name": "ConfigUri", "shape": "S3Uri", "type": "string"},
            {"name": "Environment", "shape": "MonitoringEnvironmentMap", "type": "map"},
        ],
        "type": "structure",
    },
    "ModelBiasBaselineConfig": {
        "members": [
            {"name": "BaseliningJobName", "shape": "ProcessingJobName", "type": "string"},
            {
                "name": "ConstraintsResource",
                "shape": "MonitoringConstraintsResource",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelBiasJobInput": {
        "members": [
            {"name": "EndpointInput", "shape": "EndpointInput", "type": "structure"},
            {"name": "BatchTransformInput", "shape": "BatchTransformInput", "type": "structure"},
            {
                "name": "GroundTruthS3Input",
                "shape": "MonitoringGroundTruthS3Input",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelCard": {
        "members": [
            {"name": "ModelCardArn", "shape": "ModelCardArn", "type": "string"},
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
            {"name": "Content", "shape": "ModelCardContent", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
            {"name": "SecurityConfig", "shape": "ModelCardSecurityConfig", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "ModelId", "shape": "String", "type": "string"},
            {"name": "RiskRating", "shape": "String", "type": "string"},
            {"name": "ModelPackageGroupName", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelCardExportArtifacts": {
        "members": [{"name": "S3ExportArtifacts", "shape": "S3Uri", "type": "string"}],
        "type": "structure",
    },
    "ModelCardExportJobSummary": {
        "members": [
            {"name": "ModelCardExportJobName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardExportJobArn", "shape": "ModelCardExportJobArn", "type": "string"},
            {"name": "Status", "shape": "ModelCardExportJobStatus", "type": "string"},
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
            {"name": "CreatedAt", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedAt", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ModelCardExportJobSummaryList": {
        "member_shape": "ModelCardExportJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ModelCardExportOutputConfig": {
        "members": [{"name": "S3OutputPath", "shape": "S3Uri", "type": "string"}],
        "type": "structure",
    },
    "ModelCardSecurityConfig": {
        "members": [{"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"}],
        "type": "structure",
    },
    "ModelCardSummary": {
        "members": [
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardArn", "shape": "ModelCardArn", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ModelCardSummaryList": {
        "member_shape": "ModelCardSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ModelCardVersionSummary": {
        "members": [
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardArn", "shape": "ModelCardArn", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ModelCardVersionSummaryList": {
        "member_shape": "ModelCardVersionSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ModelClientConfig": {
        "members": [
            {
                "name": "InvocationsTimeoutInSeconds",
                "shape": "InvocationsTimeoutInSeconds",
                "type": "integer",
            },
            {"name": "InvocationsMaxRetries", "shape": "InvocationsMaxRetries", "type": "integer"},
        ],
        "type": "structure",
    },
    "ModelCompilationConfig": {
        "members": [
            {"name": "Image", "shape": "OptimizationContainerImage", "type": "string"},
            {
                "name": "OverrideEnvironment",
                "shape": "OptimizationJobEnvironmentVariables",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "ModelConfiguration": {
        "members": [
            {
                "name": "InferenceSpecificationName",
                "shape": "InferenceSpecificationName",
                "type": "string",
            },
            {"name": "EnvironmentParameters", "shape": "EnvironmentParameters", "type": "list"},
            {
                "name": "CompilationJobName",
                "shape": "RecommendationJobCompilationJobName",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "ModelDashboardEndpoint": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "EndpointArn", "shape": "EndpointArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndpointStatus", "shape": "EndpointStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelDashboardEndpoints": {
        "member_shape": "ModelDashboardEndpoint",
        "member_type": "structure",
        "type": "list",
    },
    "ModelDashboardIndicatorAction": {
        "members": [{"name": "Enabled", "shape": "Boolean", "type": "boolean"}],
        "type": "structure",
    },
    "ModelDashboardModel": {
        "members": [
            {"name": "Model", "shape": "Model", "type": "structure"},
            {"name": "Endpoints", "shape": "ModelDashboardEndpoints", "type": "list"},
            {"name": "LastBatchTransformJob", "shape": "TransformJob", "type": "structure"},
            {
                "name": "MonitoringSchedules",
                "shape": "ModelDashboardMonitoringSchedules",
                "type": "list",
            },
            {"name": "ModelCard", "shape": "ModelDashboardModelCard", "type": "structure"},
        ],
        "type": "structure",
    },
    "ModelDashboardModelCard": {
        "members": [
            {"name": "ModelCardArn", "shape": "ModelCardArn", "type": "string"},
            {"name": "ModelCardName", "shape": "EntityName", "type": "string"},
            {"name": "ModelCardVersion", "shape": "Integer", "type": "integer"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
            {"name": "SecurityConfig", "shape": "ModelCardSecurityConfig", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "ModelId", "shape": "String", "type": "string"},
            {"name": "RiskRating", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelDashboardMonitoringSchedule": {
        "members": [
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"},
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "MonitoringScheduleStatus", "shape": "ScheduleStatus", "type": "string"},
            {"name": "MonitoringType", "shape": "MonitoringType", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "MonitoringScheduleConfig",
                "shape": "MonitoringScheduleConfig",
                "type": "structure",
            },
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {
                "name": "MonitoringAlertSummaries",
                "shape": "MonitoringAlertSummaryList",
                "type": "list",
            },
            {
                "name": "LastMonitoringExecutionSummary",
                "shape": "MonitoringExecutionSummary",
                "type": "structure",
            },
            {"name": "BatchTransformInput", "shape": "BatchTransformInput", "type": "structure"},
        ],
        "type": "structure",
    },
    "ModelDashboardMonitoringSchedules": {
        "member_shape": "ModelDashboardMonitoringSchedule",
        "member_type": "structure",
        "type": "list",
    },
    "ModelDataQuality": {
        "members": [
            {"name": "Statistics", "shape": "MetricsSource", "type": "structure"},
            {"name": "Constraints", "shape": "MetricsSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "ModelDataSource": {
        "members": [{"name": "S3DataSource", "shape": "S3ModelDataSource", "type": "structure"}],
        "type": "structure",
    },
    "ModelDeployConfig": {
        "members": [
            {
                "name": "AutoGenerateEndpointName",
                "shape": "AutoGenerateEndpointName",
                "type": "boolean",
            },
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelDeployResult": {
        "members": [{"name": "EndpointName", "shape": "EndpointName", "type": "string"}],
        "type": "structure",
    },
    "ModelDigests": {
        "members": [{"name": "ArtifactDigest", "shape": "ArtifactDigest", "type": "string"}],
        "type": "structure",
    },
    "ModelError": {
        "members": [
            {"name": "Message", "shape": "Message", "type": "string"},
            {"name": "OriginalStatusCode", "shape": "StatusCode", "type": "integer"},
            {"name": "OriginalMessage", "shape": "Message", "type": "string"},
            {"name": "LogStreamArn", "shape": "LogStreamArn", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelExplainabilityAppSpecification": {
        "members": [
            {"name": "ImageUri", "shape": "ImageUri", "type": "string"},
            {"name": "ConfigUri", "shape": "S3Uri", "type": "string"},
            {"name": "Environment", "shape": "MonitoringEnvironmentMap", "type": "map"},
        ],
        "type": "structure",
    },
    "ModelExplainabilityBaselineConfig": {
        "members": [
            {"name": "BaseliningJobName", "shape": "ProcessingJobName", "type": "string"},
            {
                "name": "ConstraintsResource",
                "shape": "MonitoringConstraintsResource",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelExplainabilityJobInput": {
        "members": [
            {"name": "EndpointInput", "shape": "EndpointInput", "type": "structure"},
            {"name": "BatchTransformInput", "shape": "BatchTransformInput", "type": "structure"},
        ],
        "type": "structure",
    },
    "ModelInfrastructureConfig": {
        "members": [
            {"name": "InfrastructureType", "shape": "ModelInfrastructureType", "type": "string"},
            {
                "name": "RealTimeInferenceConfig",
                "shape": "RealTimeInferenceConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelInput": {
        "members": [{"name": "DataInputConfig", "shape": "DataInputConfig", "type": "string"}],
        "type": "structure",
    },
    "ModelLatencyThreshold": {
        "members": [
            {"name": "Percentile", "shape": "String64", "type": "string"},
            {"name": "ValueInMilliseconds", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "ModelLatencyThresholds": {
        "member_shape": "ModelLatencyThreshold",
        "member_type": "structure",
        "type": "list",
    },
    "ModelLifeCycle": {
        "members": [
            {"name": "Stage", "shape": "EntityName", "type": "string"},
            {"name": "StageStatus", "shape": "EntityName", "type": "string"},
            {"name": "StageDescription", "shape": "StageDescription", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelMetadataFilter": {
        "members": [
            {"name": "Name", "shape": "ModelMetadataFilterType", "type": "string"},
            {"name": "Value", "shape": "String256", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelMetadataFilters": {
        "member_shape": "ModelMetadataFilter",
        "member_type": "structure",
        "type": "list",
    },
    "ModelMetadataSearchExpression": {
        "members": [{"name": "Filters", "shape": "ModelMetadataFilters", "type": "list"}],
        "type": "structure",
    },
    "ModelMetadataSummaries": {
        "member_shape": "ModelMetadataSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ModelMetadataSummary": {
        "members": [
            {"name": "Domain", "shape": "String", "type": "string"},
            {"name": "Framework", "shape": "String", "type": "string"},
            {"name": "Task", "shape": "String", "type": "string"},
            {"name": "Model", "shape": "String", "type": "string"},
            {"name": "FrameworkVersion", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelMetrics": {
        "members": [
            {"name": "ModelQuality", "shape": "ModelQuality", "type": "structure"},
            {"name": "ModelDataQuality", "shape": "ModelDataQuality", "type": "structure"},
            {"name": "Bias", "shape": "Bias", "type": "structure"},
            {"name": "Explainability", "shape": "Explainability", "type": "structure"},
        ],
        "type": "structure",
    },
    "ModelNotReadyException": {
        "members": [{"name": "Message", "shape": "Message", "type": "string"}],
        "type": "structure",
    },
    "ModelPackage": {
        "members": [
            {"name": "ModelPackageName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageVersion", "shape": "ModelPackageVersion", "type": "integer"},
            {"name": "ModelPackageArn", "shape": "ModelPackageArn", "type": "string"},
            {"name": "ModelPackageDescription", "shape": "EntityDescription", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {
                "name": "InferenceSpecification",
                "shape": "InferenceSpecification",
                "type": "structure",
            },
            {
                "name": "SourceAlgorithmSpecification",
                "shape": "SourceAlgorithmSpecification",
                "type": "structure",
            },
            {
                "name": "ValidationSpecification",
                "shape": "ModelPackageValidationSpecification",
                "type": "structure",
            },
            {"name": "ModelPackageStatus", "shape": "ModelPackageStatus", "type": "string"},
            {
                "name": "ModelPackageStatusDetails",
                "shape": "ModelPackageStatusDetails",
                "type": "structure",
            },
            {"name": "CertifyForMarketplace", "shape": "CertifyForMarketplace", "type": "boolean"},
            {"name": "ModelApprovalStatus", "shape": "ModelApprovalStatus", "type": "string"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "ModelMetrics", "shape": "ModelMetrics", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "ApprovalDescription", "shape": "ApprovalDescription", "type": "string"},
            {"name": "Domain", "shape": "String", "type": "string"},
            {"name": "Task", "shape": "String", "type": "string"},
            {"name": "SamplePayloadUrl", "shape": "String", "type": "string"},
            {
                "name": "AdditionalInferenceSpecifications",
                "shape": "AdditionalInferenceSpecifications",
                "type": "list",
            },
            {"name": "SourceUri", "shape": "ModelPackageSourceUri", "type": "string"},
            {"name": "SecurityConfig", "shape": "ModelPackageSecurityConfig", "type": "structure"},
            {"name": "ModelCard", "shape": "ModelPackageModelCard", "type": "structure"},
            {"name": "ModelLifeCycle", "shape": "ModelLifeCycle", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "CustomerMetadataProperties", "shape": "CustomerMetadataMap", "type": "map"},
            {"name": "DriftCheckBaselines", "shape": "DriftCheckBaselines", "type": "structure"},
            {"name": "SkipModelValidation", "shape": "SkipModelValidation", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelPackageArnList": {
        "member_shape": "ModelPackageArn",
        "member_type": "string",
        "type": "list",
    },
    "ModelPackageContainerDefinition": {
        "members": [
            {"name": "ContainerHostname", "shape": "ContainerHostname", "type": "string"},
            {"name": "Image", "shape": "ContainerImage", "type": "string"},
            {"name": "ImageDigest", "shape": "ImageDigest", "type": "string"},
            {"name": "ModelDataUrl", "shape": "Url", "type": "string"},
            {"name": "ModelDataSource", "shape": "ModelDataSource", "type": "structure"},
            {"name": "ProductId", "shape": "ProductId", "type": "string"},
            {"name": "Environment", "shape": "EnvironmentMap", "type": "map"},
            {"name": "ModelInput", "shape": "ModelInput", "type": "structure"},
            {"name": "Framework", "shape": "String", "type": "string"},
            {"name": "FrameworkVersion", "shape": "ModelPackageFrameworkVersion", "type": "string"},
            {"name": "NearestModelName", "shape": "String", "type": "string"},
            {
                "name": "AdditionalS3DataSource",
                "shape": "AdditionalS3DataSource",
                "type": "structure",
            },
            {"name": "ModelDataETag", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelPackageContainerDefinitionList": {
        "member_shape": "ModelPackageContainerDefinition",
        "member_type": "structure",
        "type": "list",
    },
    "ModelPackageGroup": {
        "members": [
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageGroupArn", "shape": "ModelPackageGroupArn", "type": "string"},
            {
                "name": "ModelPackageGroupDescription",
                "shape": "EntityDescription",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {
                "name": "ModelPackageGroupStatus",
                "shape": "ModelPackageGroupStatus",
                "type": "string",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "ModelPackageGroupSummary": {
        "members": [
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageGroupArn", "shape": "ModelPackageGroupArn", "type": "string"},
            {
                "name": "ModelPackageGroupDescription",
                "shape": "EntityDescription",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {
                "name": "ModelPackageGroupStatus",
                "shape": "ModelPackageGroupStatus",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "ModelPackageGroupSummaryList": {
        "member_shape": "ModelPackageGroupSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ModelPackageModelCard": {
        "members": [
            {"name": "ModelCardContent", "shape": "ModelCardContent", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelPackageSecurityConfig": {
        "members": [{"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"}],
        "type": "structure",
    },
    "ModelPackageStatusDetails": {
        "members": [
            {"name": "ValidationStatuses", "shape": "ModelPackageStatusItemList", "type": "list"},
            {"name": "ImageScanStatuses", "shape": "ModelPackageStatusItemList", "type": "list"},
        ],
        "type": "structure",
    },
    "ModelPackageStatusItem": {
        "members": [
            {"name": "Name", "shape": "EntityName", "type": "string"},
            {"name": "Status", "shape": "DetailedModelPackageStatus", "type": "string"},
            {"name": "FailureReason", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelPackageStatusItemList": {
        "member_shape": "ModelPackageStatusItem",
        "member_type": "structure",
        "type": "list",
    },
    "ModelPackageSummaries": {
        "key_shape": "ModelPackageArn",
        "key_type": "string",
        "type": "map",
        "value_shape": "BatchDescribeModelPackageSummary",
        "value_type": "structure",
    },
    "ModelPackageSummary": {
        "members": [
            {"name": "ModelPackageName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {"name": "ModelPackageVersion", "shape": "ModelPackageVersion", "type": "integer"},
            {"name": "ModelPackageArn", "shape": "ModelPackageArn", "type": "string"},
            {"name": "ModelPackageDescription", "shape": "EntityDescription", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "ModelPackageStatus", "shape": "ModelPackageStatus", "type": "string"},
            {"name": "ModelApprovalStatus", "shape": "ModelApprovalStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelPackageSummaryList": {
        "member_shape": "ModelPackageSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ModelPackageValidationProfile": {
        "members": [
            {"name": "ProfileName", "shape": "EntityName", "type": "string"},
            {
                "name": "TransformJobDefinition",
                "shape": "TransformJobDefinition",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelPackageValidationProfiles": {
        "member_shape": "ModelPackageValidationProfile",
        "member_type": "structure",
        "type": "list",
    },
    "ModelPackageValidationSpecification": {
        "members": [
            {"name": "ValidationRole", "shape": "RoleArn", "type": "string"},
            {
                "name": "ValidationProfiles",
                "shape": "ModelPackageValidationProfiles",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "ModelQuality": {
        "members": [
            {"name": "Statistics", "shape": "MetricsSource", "type": "structure"},
            {"name": "Constraints", "shape": "MetricsSource", "type": "structure"},
        ],
        "type": "structure",
    },
    "ModelQualityAppSpecification": {
        "members": [
            {"name": "ImageUri", "shape": "ImageUri", "type": "string"},
            {"name": "ContainerEntrypoint", "shape": "ContainerEntrypoint", "type": "list"},
            {"name": "ContainerArguments", "shape": "MonitoringContainerArguments", "type": "list"},
            {"name": "RecordPreprocessorSourceUri", "shape": "S3Uri", "type": "string"},
            {"name": "PostAnalyticsProcessorSourceUri", "shape": "S3Uri", "type": "string"},
            {"name": "ProblemType", "shape": "MonitoringProblemType", "type": "string"},
            {"name": "Environment", "shape": "MonitoringEnvironmentMap", "type": "map"},
        ],
        "type": "structure",
    },
    "ModelQualityBaselineConfig": {
        "members": [
            {"name": "BaseliningJobName", "shape": "ProcessingJobName", "type": "string"},
            {
                "name": "ConstraintsResource",
                "shape": "MonitoringConstraintsResource",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelQualityJobInput": {
        "members": [
            {"name": "EndpointInput", "shape": "EndpointInput", "type": "structure"},
            {"name": "BatchTransformInput", "shape": "BatchTransformInput", "type": "structure"},
            {
                "name": "GroundTruthS3Input",
                "shape": "MonitoringGroundTruthS3Input",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelQuantizationConfig": {
        "members": [
            {"name": "Image", "shape": "OptimizationContainerImage", "type": "string"},
            {
                "name": "OverrideEnvironment",
                "shape": "OptimizationJobEnvironmentVariables",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "ModelRegisterSettings": {
        "members": [
            {"name": "Status", "shape": "FeatureStatus", "type": "string"},
            {"name": "CrossAccountModelRegisterRoleArn", "shape": "RoleArn", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelShardingConfig": {
        "members": [
            {"name": "Image", "shape": "OptimizationContainerImage", "type": "string"},
            {
                "name": "OverrideEnvironment",
                "shape": "OptimizationJobEnvironmentVariables",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "ModelStepMetadata": {
        "members": [{"name": "Arn", "shape": "String256", "type": "string"}],
        "type": "structure",
    },
    "ModelStreamError": {
        "members": [
            {"name": "Message", "shape": "Message", "type": "string"},
            {"name": "ErrorCode", "shape": "ErrorCode", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelSummary": {
        "members": [
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "ModelArn", "shape": "ModelArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ModelSummaryList": {
        "member_shape": "ModelSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ModelVariantActionMap": {
        "key_shape": "ModelVariantName",
        "key_type": "string",
        "type": "map",
        "value_shape": "ModelVariantAction",
        "value_type": "string",
    },
    "ModelVariantConfig": {
        "members": [
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "VariantName", "shape": "ModelVariantName", "type": "string"},
            {
                "name": "InfrastructureConfig",
                "shape": "ModelInfrastructureConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ModelVariantConfigList": {
        "member_shape": "ModelVariantConfig",
        "member_type": "structure",
        "type": "list",
    },
    "ModelVariantConfigSummary": {
        "members": [
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "VariantName", "shape": "ModelVariantName", "type": "string"},
            {
                "name": "InfrastructureConfig",
                "shape": "ModelInfrastructureConfig",
                "type": "structure",
            },
            {"name": "Status", "shape": "ModelVariantStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ModelVariantConfigSummaryList": {
        "member_shape": "ModelVariantConfigSummary",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringAlertActions": {
        "members": [
            {
                "name": "ModelDashboardIndicator",
                "shape": "ModelDashboardIndicatorAction",
                "type": "structure",
            }
        ],
        "type": "structure",
    },
    "MonitoringAlertHistoryList": {
        "member_shape": "MonitoringAlertHistorySummary",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringAlertHistorySummary": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "MonitoringAlertName", "shape": "MonitoringAlertName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "AlertStatus", "shape": "MonitoringAlertStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringAlertSummary": {
        "members": [
            {"name": "MonitoringAlertName", "shape": "MonitoringAlertName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "AlertStatus", "shape": "MonitoringAlertStatus", "type": "string"},
            {
                "name": "DatapointsToAlert",
                "shape": "MonitoringDatapointsToAlert",
                "type": "integer",
            },
            {"name": "EvaluationPeriod", "shape": "MonitoringEvaluationPeriod", "type": "integer"},
            {"name": "Actions", "shape": "MonitoringAlertActions", "type": "structure"},
        ],
        "type": "structure",
    },
    "MonitoringAlertSummaryList": {
        "member_shape": "MonitoringAlertSummary",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringAppSpecification": {
        "members": [
            {"name": "ImageUri", "shape": "ImageUri", "type": "string"},
            {"name": "ContainerEntrypoint", "shape": "ContainerEntrypoint", "type": "list"},
            {"name": "ContainerArguments", "shape": "MonitoringContainerArguments", "type": "list"},
            {"name": "RecordPreprocessorSourceUri", "shape": "S3Uri", "type": "string"},
            {"name": "PostAnalyticsProcessorSourceUri", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringBaselineConfig": {
        "members": [
            {"name": "BaseliningJobName", "shape": "ProcessingJobName", "type": "string"},
            {
                "name": "ConstraintsResource",
                "shape": "MonitoringConstraintsResource",
                "type": "structure",
            },
            {
                "name": "StatisticsResource",
                "shape": "MonitoringStatisticsResource",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "MonitoringClusterConfig": {
        "members": [
            {"name": "InstanceCount", "shape": "ProcessingInstanceCount", "type": "integer"},
            {"name": "InstanceType", "shape": "ProcessingInstanceType", "type": "string"},
            {"name": "VolumeSizeInGB", "shape": "ProcessingVolumeSizeInGB", "type": "integer"},
            {"name": "VolumeKmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringConstraintsResource": {
        "members": [{"name": "S3Uri", "shape": "S3Uri", "type": "string"}],
        "type": "structure",
    },
    "MonitoringContainerArguments": {
        "member_shape": "ContainerArgument",
        "member_type": "string",
        "type": "list",
    },
    "MonitoringCsvDatasetFormat": {
        "members": [{"name": "Header", "shape": "Boolean", "type": "boolean"}],
        "type": "structure",
    },
    "MonitoringDatasetFormat": {
        "members": [
            {"name": "Csv", "shape": "MonitoringCsvDatasetFormat", "type": "structure"},
            {"name": "Json", "shape": "MonitoringJsonDatasetFormat", "type": "structure"},
            {"name": "Parquet", "shape": "MonitoringParquetDatasetFormat", "type": "structure"},
        ],
        "type": "structure",
    },
    "MonitoringEnvironmentMap": {
        "key_shape": "ProcessingEnvironmentKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "ProcessingEnvironmentValue",
        "value_type": "string",
    },
    "MonitoringExecutionSummary": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "ScheduledTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MonitoringExecutionStatus", "shape": "ExecutionStatus", "type": "string"},
            {"name": "ProcessingJobArn", "shape": "ProcessingJobArn", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {
                "name": "MonitoringJobDefinitionName",
                "shape": "MonitoringJobDefinitionName",
                "type": "string",
            },
            {"name": "MonitoringType", "shape": "MonitoringType", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringExecutionSummaryList": {
        "member_shape": "MonitoringExecutionSummary",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringGroundTruthS3Input": {
        "members": [{"name": "S3Uri", "shape": "MonitoringS3Uri", "type": "string"}],
        "type": "structure",
    },
    "MonitoringInput": {
        "members": [
            {"name": "EndpointInput", "shape": "EndpointInput", "type": "structure"},
            {"name": "BatchTransformInput", "shape": "BatchTransformInput", "type": "structure"},
        ],
        "type": "structure",
    },
    "MonitoringInputs": {
        "member_shape": "MonitoringInput",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringJobDefinition": {
        "members": [
            {"name": "BaselineConfig", "shape": "MonitoringBaselineConfig", "type": "structure"},
            {"name": "MonitoringInputs", "shape": "MonitoringInputs", "type": "list"},
            {
                "name": "MonitoringOutputConfig",
                "shape": "MonitoringOutputConfig",
                "type": "structure",
            },
            {"name": "MonitoringResources", "shape": "MonitoringResources", "type": "structure"},
            {
                "name": "MonitoringAppSpecification",
                "shape": "MonitoringAppSpecification",
                "type": "structure",
            },
            {
                "name": "StoppingCondition",
                "shape": "MonitoringStoppingCondition",
                "type": "structure",
            },
            {"name": "Environment", "shape": "MonitoringEnvironmentMap", "type": "map"},
            {"name": "NetworkConfig", "shape": "NetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringJobDefinitionSummary": {
        "members": [
            {
                "name": "MonitoringJobDefinitionName",
                "shape": "MonitoringJobDefinitionName",
                "type": "string",
            },
            {
                "name": "MonitoringJobDefinitionArn",
                "shape": "MonitoringJobDefinitionArn",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringJobDefinitionSummaryList": {
        "member_shape": "MonitoringJobDefinitionSummary",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringJsonDatasetFormat": {
        "members": [{"name": "Line", "shape": "Boolean", "type": "boolean"}],
        "type": "structure",
    },
    "MonitoringNetworkConfig": {
        "members": [
            {
                "name": "EnableInterContainerTrafficEncryption",
                "shape": "Boolean",
                "type": "boolean",
            },
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "MonitoringOutput": {
        "members": [{"name": "S3Output", "shape": "MonitoringS3Output", "type": "structure"}],
        "type": "structure",
    },
    "MonitoringOutputConfig": {
        "members": [
            {"name": "MonitoringOutputs", "shape": "MonitoringOutputs", "type": "list"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringOutputs": {
        "member_shape": "MonitoringOutput",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringParquetDatasetFormat": {"members": [], "type": "structure"},
    "MonitoringResources": {
        "members": [
            {"name": "ClusterConfig", "shape": "MonitoringClusterConfig", "type": "structure"}
        ],
        "type": "structure",
    },
    "MonitoringS3Output": {
        "members": [
            {"name": "S3Uri", "shape": "MonitoringS3Uri", "type": "string"},
            {"name": "LocalPath", "shape": "ProcessingLocalPath", "type": "string"},
            {"name": "S3UploadMode", "shape": "ProcessingS3UploadMode", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringSchedule": {
        "members": [
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"},
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "MonitoringScheduleStatus", "shape": "ScheduleStatus", "type": "string"},
            {"name": "MonitoringType", "shape": "MonitoringType", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "MonitoringScheduleConfig",
                "shape": "MonitoringScheduleConfig",
                "type": "structure",
            },
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {
                "name": "LastMonitoringExecutionSummary",
                "shape": "MonitoringExecutionSummary",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "MonitoringScheduleConfig": {
        "members": [
            {"name": "ScheduleConfig", "shape": "ScheduleConfig", "type": "structure"},
            {
                "name": "MonitoringJobDefinition",
                "shape": "MonitoringJobDefinition",
                "type": "structure",
            },
            {
                "name": "MonitoringJobDefinitionName",
                "shape": "MonitoringJobDefinitionName",
                "type": "string",
            },
            {"name": "MonitoringType", "shape": "MonitoringType", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringScheduleList": {
        "member_shape": "MonitoringSchedule",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringScheduleSummary": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MonitoringScheduleStatus", "shape": "ScheduleStatus", "type": "string"},
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {
                "name": "MonitoringJobDefinitionName",
                "shape": "MonitoringJobDefinitionName",
                "type": "string",
            },
            {"name": "MonitoringType", "shape": "MonitoringType", "type": "string"},
        ],
        "type": "structure",
    },
    "MonitoringScheduleSummaryList": {
        "member_shape": "MonitoringScheduleSummary",
        "member_type": "structure",
        "type": "list",
    },
    "MonitoringStatisticsResource": {
        "members": [{"name": "S3Uri", "shape": "S3Uri", "type": "string"}],
        "type": "structure",
    },
    "MonitoringStoppingCondition": {
        "members": [
            {
                "name": "MaxRuntimeInSeconds",
                "shape": "MonitoringMaxRuntimeInSeconds",
                "type": "integer",
            }
        ],
        "type": "structure",
    },
    "MultiModelConfig": {
        "members": [{"name": "ModelCacheSetting", "shape": "ModelCacheSetting", "type": "string"}],
        "type": "structure",
    },
    "NeoVpcConfig": {
        "members": [
            {"name": "SecurityGroupIds", "shape": "NeoVpcSecurityGroupIds", "type": "list"},
            {"name": "Subnets", "shape": "NeoVpcSubnets", "type": "list"},
        ],
        "type": "structure",
    },
    "NeoVpcSecurityGroupIds": {
        "member_shape": "NeoVpcSecurityGroupId",
        "member_type": "string",
        "type": "list",
    },
    "NeoVpcSubnets": {"member_shape": "NeoVpcSubnetId", "member_type": "string", "type": "list"},
    "NestedFilters": {
        "members": [
            {"name": "NestedPropertyName", "shape": "ResourcePropertyName", "type": "string"},
            {"name": "Filters", "shape": "FilterList", "type": "list"},
        ],
        "type": "structure",
    },
    "NestedFiltersList": {
        "member_shape": "NestedFilters",
        "member_type": "structure",
        "type": "list",
    },
    "NetworkConfig": {
        "members": [
            {
                "name": "EnableInterContainerTrafficEncryption",
                "shape": "Boolean",
                "type": "boolean",
            },
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "NotebookInstanceAcceleratorTypes": {
        "member_shape": "NotebookInstanceAcceleratorType",
        "member_type": "string",
        "type": "list",
    },
    "NotebookInstanceLifecycleConfigList": {
        "member_shape": "NotebookInstanceLifecycleHook",
        "member_type": "structure",
        "type": "list",
    },
    "NotebookInstanceLifecycleConfigSummary": {
        "members": [
            {
                "name": "NotebookInstanceLifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {
                "name": "NotebookInstanceLifecycleConfigArn",
                "shape": "NotebookInstanceLifecycleConfigArn",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "NotebookInstanceLifecycleConfigSummaryList": {
        "member_shape": "NotebookInstanceLifecycleConfigSummary",
        "member_type": "structure",
        "type": "list",
    },
    "NotebookInstanceLifecycleHook": {
        "members": [
            {"name": "Content", "shape": "NotebookInstanceLifecycleConfigContent", "type": "string"}
        ],
        "type": "structure",
    },
    "NotebookInstanceSummary": {
        "members": [
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"},
            {"name": "NotebookInstanceArn", "shape": "NotebookInstanceArn", "type": "string"},
            {"name": "NotebookInstanceStatus", "shape": "NotebookInstanceStatus", "type": "string"},
            {"name": "Url", "shape": "NotebookInstanceUrl", "type": "string"},
            {"name": "InstanceType", "shape": "InstanceType", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {
                "name": "NotebookInstanceLifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {"name": "DefaultCodeRepository", "shape": "CodeRepositoryNameOrUrl", "type": "string"},
            {
                "name": "AdditionalCodeRepositories",
                "shape": "AdditionalCodeRepositoryNamesOrUrls",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "NotebookInstanceSummaryList": {
        "member_shape": "NotebookInstanceSummary",
        "member_type": "structure",
        "type": "list",
    },
    "NotificationConfiguration": {
        "members": [
            {"name": "NotificationTopicArn", "shape": "NotificationTopicArn", "type": "string"}
        ],
        "type": "structure",
    },
    "ObjectiveStatusCounters": {
        "members": [
            {"name": "Succeeded", "shape": "ObjectiveStatusCounter", "type": "integer"},
            {"name": "Pending", "shape": "ObjectiveStatusCounter", "type": "integer"},
            {"name": "Failed", "shape": "ObjectiveStatusCounter", "type": "integer"},
        ],
        "type": "structure",
    },
    "OfflineStoreConfig": {
        "members": [
            {"name": "S3StorageConfig", "shape": "S3StorageConfig", "type": "structure"},
            {"name": "DisableGlueTableCreation", "shape": "Boolean", "type": "boolean"},
            {"name": "DataCatalogConfig", "shape": "DataCatalogConfig", "type": "structure"},
            {"name": "TableFormat", "shape": "TableFormat", "type": "string"},
        ],
        "type": "structure",
    },
    "OfflineStoreStatus": {
        "members": [
            {"name": "Status", "shape": "OfflineStoreStatusValue", "type": "string"},
            {"name": "BlockedReason", "shape": "BlockedReason", "type": "string"},
        ],
        "type": "structure",
    },
    "OidcConfig": {
        "members": [
            {"name": "ClientId", "shape": "ClientId", "type": "string"},
            {"name": "ClientSecret", "shape": "ClientSecret", "type": "string"},
            {"name": "Issuer", "shape": "OidcEndpoint", "type": "string"},
            {"name": "AuthorizationEndpoint", "shape": "OidcEndpoint", "type": "string"},
            {"name": "TokenEndpoint", "shape": "OidcEndpoint", "type": "string"},
            {"name": "UserInfoEndpoint", "shape": "OidcEndpoint", "type": "string"},
            {"name": "LogoutEndpoint", "shape": "OidcEndpoint", "type": "string"},
            {"name": "JwksUri", "shape": "OidcEndpoint", "type": "string"},
            {"name": "Scope", "shape": "Scope", "type": "string"},
            {
                "name": "AuthenticationRequestExtraParams",
                "shape": "AuthenticationRequestExtraParams",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "OidcConfigForResponse": {
        "members": [
            {"name": "ClientId", "shape": "ClientId", "type": "string"},
            {"name": "Issuer", "shape": "OidcEndpoint", "type": "string"},
            {"name": "AuthorizationEndpoint", "shape": "OidcEndpoint", "type": "string"},
            {"name": "TokenEndpoint", "shape": "OidcEndpoint", "type": "string"},
            {"name": "UserInfoEndpoint", "shape": "OidcEndpoint", "type": "string"},
            {"name": "LogoutEndpoint", "shape": "OidcEndpoint", "type": "string"},
            {"name": "JwksUri", "shape": "OidcEndpoint", "type": "string"},
            {"name": "Scope", "shape": "Scope", "type": "string"},
            {
                "name": "AuthenticationRequestExtraParams",
                "shape": "AuthenticationRequestExtraParams",
                "type": "map",
            },
        ],
        "type": "structure",
    },
    "OidcMemberDefinition": {
        "members": [{"name": "Groups", "shape": "Groups", "type": "list"}],
        "type": "structure",
    },
    "OnStartDeepHealthChecks": {
        "member_shape": "DeepHealthCheckType",
        "member_type": "string",
        "type": "list",
    },
    "OnlineStoreConfig": {
        "members": [
            {"name": "SecurityConfig", "shape": "OnlineStoreSecurityConfig", "type": "structure"},
            {"name": "EnableOnlineStore", "shape": "Boolean", "type": "boolean"},
            {"name": "TtlDuration", "shape": "TtlDuration", "type": "structure"},
            {"name": "StorageType", "shape": "StorageType", "type": "string"},
        ],
        "type": "structure",
    },
    "OnlineStoreConfigUpdate": {
        "members": [{"name": "TtlDuration", "shape": "TtlDuration", "type": "structure"}],
        "type": "structure",
    },
    "OnlineStoreSecurityConfig": {
        "members": [{"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"}],
        "type": "structure",
    },
    "OptimizationConfig": {
        "members": [
            {
                "name": "ModelQuantizationConfig",
                "shape": "ModelQuantizationConfig",
                "type": "structure",
            },
            {
                "name": "ModelCompilationConfig",
                "shape": "ModelCompilationConfig",
                "type": "structure",
            },
            {"name": "ModelShardingConfig", "shape": "ModelShardingConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "OptimizationConfigs": {
        "member_shape": "OptimizationConfig",
        "member_type": "structure",
        "type": "list",
    },
    "OptimizationJobEnvironmentVariables": {
        "key_shape": "NonEmptyString256",
        "key_type": "string",
        "type": "map",
        "value_shape": "String256",
        "value_type": "string",
    },
    "OptimizationJobModelSource": {
        "members": [{"name": "S3", "shape": "OptimizationJobModelSourceS3", "type": "structure"}],
        "type": "structure",
    },
    "OptimizationJobModelSourceS3": {
        "members": [
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
            {
                "name": "ModelAccessConfig",
                "shape": "OptimizationModelAccessConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "OptimizationJobOutputConfig": {
        "members": [
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "S3OutputLocation", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "OptimizationJobSummaries": {
        "member_shape": "OptimizationJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "OptimizationJobSummary": {
        "members": [
            {"name": "OptimizationJobName", "shape": "EntityName", "type": "string"},
            {"name": "OptimizationJobArn", "shape": "OptimizationJobArn", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "OptimizationJobStatus", "shape": "OptimizationJobStatus", "type": "string"},
            {"name": "OptimizationStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "OptimizationEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {
                "name": "DeploymentInstanceType",
                "shape": "OptimizationJobDeploymentInstanceType",
                "type": "string",
            },
            {"name": "OptimizationTypes", "shape": "OptimizationTypes", "type": "list"},
        ],
        "type": "structure",
    },
    "OptimizationModelAccessConfig": {
        "members": [
            {"name": "AcceptEula", "shape": "OptimizationModelAcceptEula", "type": "boolean"}
        ],
        "type": "structure",
    },
    "OptimizationOutput": {
        "members": [
            {
                "name": "RecommendedInferenceImage",
                "shape": "OptimizationContainerImage",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "OptimizationTypes": {
        "member_shape": "OptimizationType",
        "member_type": "string",
        "type": "list",
    },
    "OptimizationVpcConfig": {
        "members": [
            {
                "name": "SecurityGroupIds",
                "shape": "OptimizationVpcSecurityGroupIds",
                "type": "list",
            },
            {"name": "Subnets", "shape": "OptimizationVpcSubnets", "type": "list"},
        ],
        "type": "structure",
    },
    "OptimizationVpcSecurityGroupIds": {
        "member_shape": "OptimizationVpcSecurityGroupId",
        "member_type": "string",
        "type": "list",
    },
    "OptimizationVpcSubnets": {
        "member_shape": "OptimizationVpcSubnetId",
        "member_type": "string",
        "type": "list",
    },
    "OutputConfig": {
        "members": [
            {"name": "S3OutputLocation", "shape": "S3Uri", "type": "string"},
            {"name": "TargetDevice", "shape": "TargetDevice", "type": "string"},
            {"name": "TargetPlatform", "shape": "TargetPlatform", "type": "structure"},
            {"name": "CompilerOptions", "shape": "CompilerOptions", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "OutputDataConfig": {
        "members": [
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {"name": "CompressionType", "shape": "OutputCompressionType", "type": "string"},
        ],
        "type": "structure",
    },
    "OutputParameter": {
        "members": [
            {"name": "Name", "shape": "String256", "type": "string"},
            {"name": "Value", "shape": "String1024", "type": "string"},
        ],
        "type": "structure",
    },
    "OutputParameterList": {
        "member_shape": "OutputParameter",
        "member_type": "structure",
        "type": "list",
    },
    "OwnershipSettings": {
        "members": [{"name": "OwnerUserProfileName", "shape": "UserProfileName", "type": "string"}],
        "type": "structure",
    },
    "OwnershipSettingsSummary": {
        "members": [{"name": "OwnerUserProfileName", "shape": "UserProfileName", "type": "string"}],
        "type": "structure",
    },
    "ParallelismConfiguration": {
        "members": [
            {
                "name": "MaxParallelExecutionSteps",
                "shape": "MaxParallelExecutionSteps",
                "type": "integer",
            }
        ],
        "type": "structure",
    },
    "Parameter": {
        "members": [
            {"name": "Name", "shape": "PipelineParameterName", "type": "string"},
            {"name": "Value", "shape": "String1024", "type": "string"},
        ],
        "type": "structure",
    },
    "ParameterList": {"member_shape": "Parameter", "member_type": "structure", "type": "list"},
    "ParameterRange": {
        "members": [
            {
                "name": "IntegerParameterRangeSpecification",
                "shape": "IntegerParameterRangeSpecification",
                "type": "structure",
            },
            {
                "name": "ContinuousParameterRangeSpecification",
                "shape": "ContinuousParameterRangeSpecification",
                "type": "structure",
            },
            {
                "name": "CategoricalParameterRangeSpecification",
                "shape": "CategoricalParameterRangeSpecification",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ParameterRanges": {
        "members": [
            {"name": "IntegerParameterRanges", "shape": "IntegerParameterRanges", "type": "list"},
            {
                "name": "ContinuousParameterRanges",
                "shape": "ContinuousParameterRanges",
                "type": "list",
            },
            {
                "name": "CategoricalParameterRanges",
                "shape": "CategoricalParameterRanges",
                "type": "list",
            },
            {"name": "AutoParameters", "shape": "AutoParameters", "type": "list"},
        ],
        "type": "structure",
    },
    "ParameterValues": {"member_shape": "ParameterValue", "member_type": "string", "type": "list"},
    "Parent": {
        "members": [
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "ParentHyperParameterTuningJob": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "ParentHyperParameterTuningJobs": {
        "member_shape": "ParentHyperParameterTuningJob",
        "member_type": "structure",
        "type": "list",
    },
    "Parents": {"member_shape": "Parent", "member_type": "structure", "type": "list"},
    "PartnerAppAdminUserList": {
        "member_shape": "NonEmptyString256",
        "member_type": "string",
        "type": "list",
    },
    "PartnerAppArguments": {
        "key_shape": "NonEmptyString256",
        "key_type": "string",
        "type": "map",
        "value_shape": "String1024",
        "value_type": "string",
    },
    "PartnerAppConfig": {
        "members": [
            {"name": "AdminUsers", "shape": "PartnerAppAdminUserList", "type": "list"},
            {"name": "Arguments", "shape": "PartnerAppArguments", "type": "map"},
        ],
        "type": "structure",
    },
    "PartnerAppMaintenanceConfig": {
        "members": [
            {
                "name": "MaintenanceWindowStart",
                "shape": "WeeklyScheduleTimeFormat",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "PartnerAppSummaries": {
        "member_shape": "PartnerAppSummary",
        "member_type": "structure",
        "type": "list",
    },
    "PartnerAppSummary": {
        "members": [
            {"name": "Arn", "shape": "PartnerAppArn", "type": "string"},
            {"name": "Name", "shape": "PartnerAppName", "type": "string"},
            {"name": "Type", "shape": "PartnerAppType", "type": "string"},
            {"name": "Status", "shape": "PartnerAppStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "PayloadPart": {
        "members": [{"name": "Bytes", "shape": "PartBlob", "type": "blob"}],
        "type": "structure",
    },
    "PendingDeploymentSummary": {
        "members": [
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {
                "name": "ProductionVariants",
                "shape": "PendingProductionVariantSummaryList",
                "type": "list",
            },
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "ShadowProductionVariants",
                "shape": "PendingProductionVariantSummaryList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "PendingProductionVariantSummary": {
        "members": [
            {"name": "VariantName", "shape": "VariantName", "type": "string"},
            {"name": "DeployedImages", "shape": "DeployedImages", "type": "list"},
            {"name": "CurrentWeight", "shape": "VariantWeight", "type": "float"},
            {"name": "DesiredWeight", "shape": "VariantWeight", "type": "float"},
            {"name": "CurrentInstanceCount", "shape": "TaskCount", "type": "integer"},
            {"name": "DesiredInstanceCount", "shape": "TaskCount", "type": "integer"},
            {"name": "InstanceType", "shape": "ProductionVariantInstanceType", "type": "string"},
            {
                "name": "AcceleratorType",
                "shape": "ProductionVariantAcceleratorType",
                "type": "string",
            },
            {"name": "VariantStatus", "shape": "ProductionVariantStatusList", "type": "list"},
            {
                "name": "CurrentServerlessConfig",
                "shape": "ProductionVariantServerlessConfig",
                "type": "structure",
            },
            {
                "name": "DesiredServerlessConfig",
                "shape": "ProductionVariantServerlessConfig",
                "type": "structure",
            },
            {
                "name": "ManagedInstanceScaling",
                "shape": "ProductionVariantManagedInstanceScaling",
                "type": "structure",
            },
            {
                "name": "RoutingConfig",
                "shape": "ProductionVariantRoutingConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "PendingProductionVariantSummaryList": {
        "member_shape": "PendingProductionVariantSummary",
        "member_type": "structure",
        "type": "list",
    },
    "Phase": {
        "members": [
            {"name": "InitialNumberOfUsers", "shape": "InitialNumberOfUsers", "type": "integer"},
            {"name": "SpawnRate", "shape": "SpawnRate", "type": "integer"},
            {"name": "DurationInSeconds", "shape": "TrafficDurationInSeconds", "type": "integer"},
        ],
        "type": "structure",
    },
    "Phases": {"member_shape": "Phase", "member_type": "structure", "type": "list"},
    "Pipeline": {
        "members": [
            {"name": "PipelineArn", "shape": "PipelineArn", "type": "string"},
            {"name": "PipelineName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDisplayName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDescription", "shape": "PipelineDescription", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "PipelineStatus", "shape": "PipelineStatus", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastRunTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "PipelineDefinitionS3Location": {
        "members": [
            {"name": "Bucket", "shape": "BucketName", "type": "string"},
            {"name": "ObjectKey", "shape": "Key", "type": "string"},
            {"name": "VersionId", "shape": "VersionId", "type": "string"},
        ],
        "type": "structure",
    },
    "PipelineExecution": {
        "members": [
            {"name": "PipelineArn", "shape": "PipelineArn", "type": "string"},
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"},
            {
                "name": "PipelineExecutionDisplayName",
                "shape": "PipelineExecutionName",
                "type": "string",
            },
            {
                "name": "PipelineExecutionStatus",
                "shape": "PipelineExecutionStatus",
                "type": "string",
            },
            {
                "name": "PipelineExecutionDescription",
                "shape": "PipelineExecutionDescription",
                "type": "string",
            },
            {
                "name": "PipelineExperimentConfig",
                "shape": "PipelineExperimentConfig",
                "type": "structure",
            },
            {"name": "FailureReason", "shape": "PipelineExecutionFailureReason", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
            {
                "name": "SelectiveExecutionConfig",
                "shape": "SelectiveExecutionConfig",
                "type": "structure",
            },
            {"name": "PipelineParameters", "shape": "ParameterList", "type": "list"},
        ],
        "type": "structure",
    },
    "PipelineExecutionStep": {
        "members": [
            {"name": "StepName", "shape": "StepName", "type": "string"},
            {"name": "StepDisplayName", "shape": "StepDisplayName", "type": "string"},
            {"name": "StepDescription", "shape": "StepDescription", "type": "string"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StepStatus", "shape": "StepStatus", "type": "string"},
            {"name": "CacheHitResult", "shape": "CacheHitResult", "type": "structure"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "Metadata", "shape": "PipelineExecutionStepMetadata", "type": "structure"},
            {"name": "AttemptCount", "shape": "Integer", "type": "integer"},
            {
                "name": "SelectiveExecutionResult",
                "shape": "SelectiveExecutionResult",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "PipelineExecutionStepList": {
        "member_shape": "PipelineExecutionStep",
        "member_type": "structure",
        "type": "list",
    },
    "PipelineExecutionStepMetadata": {
        "members": [
            {"name": "TrainingJob", "shape": "TrainingJobStepMetadata", "type": "structure"},
            {"name": "ProcessingJob", "shape": "ProcessingJobStepMetadata", "type": "structure"},
            {"name": "TransformJob", "shape": "TransformJobStepMetadata", "type": "structure"},
            {"name": "TuningJob", "shape": "TuningJobStepMetaData", "type": "structure"},
            {"name": "Model", "shape": "ModelStepMetadata", "type": "structure"},
            {"name": "RegisterModel", "shape": "RegisterModelStepMetadata", "type": "structure"},
            {"name": "Condition", "shape": "ConditionStepMetadata", "type": "structure"},
            {"name": "Callback", "shape": "CallbackStepMetadata", "type": "structure"},
            {"name": "Lambda", "shape": "LambdaStepMetadata", "type": "structure"},
            {"name": "EMR", "shape": "EMRStepMetadata", "type": "structure"},
            {"name": "QualityCheck", "shape": "QualityCheckStepMetadata", "type": "structure"},
            {"name": "ClarifyCheck", "shape": "ClarifyCheckStepMetadata", "type": "structure"},
            {"name": "Fail", "shape": "FailStepMetadata", "type": "structure"},
            {"name": "AutoMLJob", "shape": "AutoMLJobStepMetadata", "type": "structure"},
            {"name": "Endpoint", "shape": "EndpointStepMetadata", "type": "structure"},
            {"name": "EndpointConfig", "shape": "EndpointConfigStepMetadata", "type": "structure"},
        ],
        "type": "structure",
    },
    "PipelineExecutionSummary": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "PipelineExecutionStatus",
                "shape": "PipelineExecutionStatus",
                "type": "string",
            },
            {
                "name": "PipelineExecutionDescription",
                "shape": "PipelineExecutionDescription",
                "type": "string",
            },
            {
                "name": "PipelineExecutionDisplayName",
                "shape": "PipelineExecutionName",
                "type": "string",
            },
            {"name": "PipelineExecutionFailureReason", "shape": "String3072", "type": "string"},
        ],
        "type": "structure",
    },
    "PipelineExecutionSummaryList": {
        "member_shape": "PipelineExecutionSummary",
        "member_type": "structure",
        "type": "list",
    },
    "PipelineExperimentConfig": {
        "members": [
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "PipelineSummary": {
        "members": [
            {"name": "PipelineArn", "shape": "PipelineArn", "type": "string"},
            {"name": "PipelineName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDisplayName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDescription", "shape": "PipelineDescription", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastExecutionTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "PipelineSummaryList": {
        "member_shape": "PipelineSummary",
        "member_type": "structure",
        "type": "list",
    },
    "PredefinedMetricSpecification": {
        "members": [{"name": "PredefinedMetricType", "shape": "String", "type": "string"}],
        "type": "structure",
    },
    "PriorityClass": {
        "members": [
            {"name": "Name", "shape": "ClusterSchedulerPriorityClassName", "type": "string"},
            {"name": "Weight", "shape": "PriorityWeight", "type": "integer"},
        ],
        "type": "structure",
    },
    "PriorityClassList": {
        "member_shape": "PriorityClass",
        "member_type": "structure",
        "type": "list",
    },
    "ProcessingClusterConfig": {
        "members": [
            {"name": "InstanceCount", "shape": "ProcessingInstanceCount", "type": "integer"},
            {"name": "InstanceType", "shape": "ProcessingInstanceType", "type": "string"},
            {"name": "VolumeSizeInGB", "shape": "ProcessingVolumeSizeInGB", "type": "integer"},
            {"name": "VolumeKmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "ProcessingEnvironmentMap": {
        "key_shape": "ProcessingEnvironmentKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "ProcessingEnvironmentValue",
        "value_type": "string",
    },
    "ProcessingFeatureStoreOutput": {
        "members": [{"name": "FeatureGroupName", "shape": "FeatureGroupName", "type": "string"}],
        "type": "structure",
    },
    "ProcessingInput": {
        "members": [
            {"name": "InputName", "shape": "String", "type": "string"},
            {"name": "AppManaged", "shape": "AppManaged", "type": "boolean"},
            {"name": "S3Input", "shape": "ProcessingS3Input", "type": "structure"},
            {"name": "DatasetDefinition", "shape": "DatasetDefinition", "type": "structure"},
        ],
        "type": "structure",
    },
    "ProcessingInputs": {
        "member_shape": "ProcessingInput",
        "member_type": "structure",
        "type": "list",
    },
    "ProcessingJob": {
        "members": [
            {"name": "ProcessingInputs", "shape": "ProcessingInputs", "type": "list"},
            {
                "name": "ProcessingOutputConfig",
                "shape": "ProcessingOutputConfig",
                "type": "structure",
            },
            {"name": "ProcessingJobName", "shape": "ProcessingJobName", "type": "string"},
            {"name": "ProcessingResources", "shape": "ProcessingResources", "type": "structure"},
            {
                "name": "StoppingCondition",
                "shape": "ProcessingStoppingCondition",
                "type": "structure",
            },
            {"name": "AppSpecification", "shape": "AppSpecification", "type": "structure"},
            {"name": "Environment", "shape": "ProcessingEnvironmentMap", "type": "map"},
            {"name": "NetworkConfig", "shape": "NetworkConfig", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
            {"name": "ProcessingJobArn", "shape": "ProcessingJobArn", "type": "string"},
            {"name": "ProcessingJobStatus", "shape": "ProcessingJobStatus", "type": "string"},
            {"name": "ExitMessage", "shape": "ExitMessage", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ProcessingEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ProcessingStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {"name": "TrainingJobArn", "shape": "TrainingJobArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "ProcessingJobStepMetadata": {
        "members": [{"name": "Arn", "shape": "ProcessingJobArn", "type": "string"}],
        "type": "structure",
    },
    "ProcessingJobSummaries": {
        "member_shape": "ProcessingJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ProcessingJobSummary": {
        "members": [
            {"name": "ProcessingJobName", "shape": "ProcessingJobName", "type": "string"},
            {"name": "ProcessingJobArn", "shape": "ProcessingJobArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ProcessingEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ProcessingJobStatus", "shape": "ProcessingJobStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ExitMessage", "shape": "ExitMessage", "type": "string"},
        ],
        "type": "structure",
    },
    "ProcessingOutput": {
        "members": [
            {"name": "OutputName", "shape": "String", "type": "string"},
            {"name": "S3Output", "shape": "ProcessingS3Output", "type": "structure"},
            {
                "name": "FeatureStoreOutput",
                "shape": "ProcessingFeatureStoreOutput",
                "type": "structure",
            },
            {"name": "AppManaged", "shape": "AppManaged", "type": "boolean"},
        ],
        "type": "structure",
    },
    "ProcessingOutputConfig": {
        "members": [
            {"name": "Outputs", "shape": "ProcessingOutputs", "type": "list"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "ProcessingOutputs": {
        "member_shape": "ProcessingOutput",
        "member_type": "structure",
        "type": "list",
    },
    "ProcessingResources": {
        "members": [
            {"name": "ClusterConfig", "shape": "ProcessingClusterConfig", "type": "structure"}
        ],
        "type": "structure",
    },
    "ProcessingS3Input": {
        "members": [
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "LocalPath", "shape": "ProcessingLocalPath", "type": "string"},
            {"name": "S3DataType", "shape": "ProcessingS3DataType", "type": "string"},
            {"name": "S3InputMode", "shape": "ProcessingS3InputMode", "type": "string"},
            {
                "name": "S3DataDistributionType",
                "shape": "ProcessingS3DataDistributionType",
                "type": "string",
            },
            {"name": "S3CompressionType", "shape": "ProcessingS3CompressionType", "type": "string"},
        ],
        "type": "structure",
    },
    "ProcessingS3Output": {
        "members": [
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "LocalPath", "shape": "ProcessingLocalPath", "type": "string"},
            {"name": "S3UploadMode", "shape": "ProcessingS3UploadMode", "type": "string"},
        ],
        "type": "structure",
    },
    "ProcessingStoppingCondition": {
        "members": [
            {
                "name": "MaxRuntimeInSeconds",
                "shape": "ProcessingMaxRuntimeInSeconds",
                "type": "integer",
            }
        ],
        "type": "structure",
    },
    "ProductListings": {"member_shape": "String", "member_type": "string", "type": "list"},
    "ProductionVariant": {
        "members": [
            {"name": "VariantName", "shape": "VariantName", "type": "string"},
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "InitialInstanceCount", "shape": "InitialTaskCount", "type": "integer"},
            {"name": "InstanceType", "shape": "ProductionVariantInstanceType", "type": "string"},
            {"name": "InitialVariantWeight", "shape": "VariantWeight", "type": "float"},
            {
                "name": "AcceleratorType",
                "shape": "ProductionVariantAcceleratorType",
                "type": "string",
            },
            {
                "name": "CoreDumpConfig",
                "shape": "ProductionVariantCoreDumpConfig",
                "type": "structure",
            },
            {
                "name": "ServerlessConfig",
                "shape": "ProductionVariantServerlessConfig",
                "type": "structure",
            },
            {
                "name": "VolumeSizeInGB",
                "shape": "ProductionVariantVolumeSizeInGB",
                "type": "integer",
            },
            {
                "name": "ModelDataDownloadTimeoutInSeconds",
                "shape": "ProductionVariantModelDataDownloadTimeoutInSeconds",
                "type": "integer",
            },
            {
                "name": "ContainerStartupHealthCheckTimeoutInSeconds",
                "shape": "ProductionVariantContainerStartupHealthCheckTimeoutInSeconds",
                "type": "integer",
            },
            {"name": "EnableSSMAccess", "shape": "ProductionVariantSSMAccess", "type": "boolean"},
            {
                "name": "ManagedInstanceScaling",
                "shape": "ProductionVariantManagedInstanceScaling",
                "type": "structure",
            },
            {
                "name": "RoutingConfig",
                "shape": "ProductionVariantRoutingConfig",
                "type": "structure",
            },
            {
                "name": "InferenceAmiVersion",
                "shape": "ProductionVariantInferenceAmiVersion",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "ProductionVariantCoreDumpConfig": {
        "members": [
            {"name": "DestinationS3Uri", "shape": "DestinationS3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "ProductionVariantList": {
        "member_shape": "ProductionVariant",
        "member_type": "structure",
        "type": "list",
    },
    "ProductionVariantManagedInstanceScaling": {
        "members": [
            {"name": "Status", "shape": "ManagedInstanceScalingStatus", "type": "string"},
            {
                "name": "MinInstanceCount",
                "shape": "ManagedInstanceScalingMinInstanceCount",
                "type": "integer",
            },
            {
                "name": "MaxInstanceCount",
                "shape": "ManagedInstanceScalingMaxInstanceCount",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "ProductionVariantRoutingConfig": {
        "members": [{"name": "RoutingStrategy", "shape": "RoutingStrategy", "type": "string"}],
        "type": "structure",
    },
    "ProductionVariantServerlessConfig": {
        "members": [
            {"name": "MemorySizeInMB", "shape": "ServerlessMemorySizeInMB", "type": "integer"},
            {"name": "MaxConcurrency", "shape": "ServerlessMaxConcurrency", "type": "integer"},
            {
                "name": "ProvisionedConcurrency",
                "shape": "ServerlessProvisionedConcurrency",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "ProductionVariantServerlessUpdateConfig": {
        "members": [
            {"name": "MaxConcurrency", "shape": "ServerlessMaxConcurrency", "type": "integer"},
            {
                "name": "ProvisionedConcurrency",
                "shape": "ServerlessProvisionedConcurrency",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "ProductionVariantStatus": {
        "members": [
            {"name": "Status", "shape": "VariantStatus", "type": "string"},
            {"name": "StatusMessage", "shape": "VariantStatusMessage", "type": "string"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ProductionVariantStatusList": {
        "member_shape": "ProductionVariantStatus",
        "member_type": "structure",
        "type": "list",
    },
    "ProductionVariantSummary": {
        "members": [
            {"name": "VariantName", "shape": "VariantName", "type": "string"},
            {"name": "DeployedImages", "shape": "DeployedImages", "type": "list"},
            {"name": "CurrentWeight", "shape": "VariantWeight", "type": "float"},
            {"name": "DesiredWeight", "shape": "VariantWeight", "type": "float"},
            {"name": "CurrentInstanceCount", "shape": "TaskCount", "type": "integer"},
            {"name": "DesiredInstanceCount", "shape": "TaskCount", "type": "integer"},
            {"name": "VariantStatus", "shape": "ProductionVariantStatusList", "type": "list"},
            {
                "name": "CurrentServerlessConfig",
                "shape": "ProductionVariantServerlessConfig",
                "type": "structure",
            },
            {
                "name": "DesiredServerlessConfig",
                "shape": "ProductionVariantServerlessConfig",
                "type": "structure",
            },
            {
                "name": "ManagedInstanceScaling",
                "shape": "ProductionVariantManagedInstanceScaling",
                "type": "structure",
            },
            {
                "name": "RoutingConfig",
                "shape": "ProductionVariantRoutingConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ProductionVariantSummaryList": {
        "member_shape": "ProductionVariantSummary",
        "member_type": "structure",
        "type": "list",
    },
    "ProfilerConfig": {
        "members": [
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {
                "name": "ProfilingIntervalInMilliseconds",
                "shape": "ProfilingIntervalInMilliseconds",
                "type": "long",
            },
            {"name": "ProfilingParameters", "shape": "ProfilingParameters", "type": "map"},
            {"name": "DisableProfiler", "shape": "DisableProfiler", "type": "boolean"},
        ],
        "type": "structure",
    },
    "ProfilerConfigForUpdate": {
        "members": [
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {
                "name": "ProfilingIntervalInMilliseconds",
                "shape": "ProfilingIntervalInMilliseconds",
                "type": "long",
            },
            {"name": "ProfilingParameters", "shape": "ProfilingParameters", "type": "map"},
            {"name": "DisableProfiler", "shape": "DisableProfiler", "type": "boolean"},
        ],
        "type": "structure",
    },
    "ProfilerRuleConfiguration": {
        "members": [
            {"name": "RuleConfigurationName", "shape": "RuleConfigurationName", "type": "string"},
            {"name": "LocalPath", "shape": "DirectoryPath", "type": "string"},
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {"name": "RuleEvaluatorImage", "shape": "AlgorithmImage", "type": "string"},
            {"name": "InstanceType", "shape": "ProcessingInstanceType", "type": "string"},
            {"name": "VolumeSizeInGB", "shape": "OptionalVolumeSizeInGB", "type": "integer"},
            {"name": "RuleParameters", "shape": "RuleParameters", "type": "map"},
        ],
        "type": "structure",
    },
    "ProfilerRuleConfigurations": {
        "member_shape": "ProfilerRuleConfiguration",
        "member_type": "structure",
        "type": "list",
    },
    "ProfilerRuleEvaluationStatus": {
        "members": [
            {"name": "RuleConfigurationName", "shape": "RuleConfigurationName", "type": "string"},
            {"name": "RuleEvaluationJobArn", "shape": "ProcessingJobArn", "type": "string"},
            {"name": "RuleEvaluationStatus", "shape": "RuleEvaluationStatus", "type": "string"},
            {"name": "StatusDetails", "shape": "StatusDetails", "type": "string"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ProfilerRuleEvaluationStatuses": {
        "member_shape": "ProfilerRuleEvaluationStatus",
        "member_type": "structure",
        "type": "list",
    },
    "ProfilingParameters": {
        "key_shape": "ConfigKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "ConfigValue",
        "value_type": "string",
    },
    "Project": {
        "members": [
            {"name": "ProjectArn", "shape": "ProjectArn", "type": "string"},
            {"name": "ProjectName", "shape": "ProjectEntityName", "type": "string"},
            {"name": "ProjectId", "shape": "ProjectId", "type": "string"},
            {"name": "ProjectDescription", "shape": "EntityDescription", "type": "string"},
            {
                "name": "ServiceCatalogProvisioningDetails",
                "shape": "ServiceCatalogProvisioningDetails",
                "type": "structure",
            },
            {
                "name": "ServiceCatalogProvisionedProductDetails",
                "shape": "ServiceCatalogProvisionedProductDetails",
                "type": "structure",
            },
            {"name": "ProjectStatus", "shape": "ProjectStatus", "type": "string"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "ProjectSummary": {
        "members": [
            {"name": "ProjectName", "shape": "ProjectEntityName", "type": "string"},
            {"name": "ProjectDescription", "shape": "EntityDescription", "type": "string"},
            {"name": "ProjectArn", "shape": "ProjectArn", "type": "string"},
            {"name": "ProjectId", "shape": "ProjectId", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ProjectStatus", "shape": "ProjectStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "ProjectSummaryList": {
        "member_shape": "ProjectSummary",
        "member_type": "structure",
        "type": "list",
    },
    "PropertyNameQuery": {
        "members": [{"name": "PropertyNameHint", "shape": "PropertyNameHint", "type": "string"}],
        "type": "structure",
    },
    "PropertyNameSuggestion": {
        "members": [{"name": "PropertyName", "shape": "ResourcePropertyName", "type": "string"}],
        "type": "structure",
    },
    "PropertyNameSuggestionList": {
        "member_shape": "PropertyNameSuggestion",
        "member_type": "structure",
        "type": "list",
    },
    "ProvisioningParameter": {
        "members": [
            {"name": "Key", "shape": "ProvisioningParameterKey", "type": "string"},
            {"name": "Value", "shape": "ProvisioningParameterValue", "type": "string"},
        ],
        "type": "structure",
    },
    "ProvisioningParameters": {
        "member_shape": "ProvisioningParameter",
        "member_type": "structure",
        "type": "list",
    },
    "PublicWorkforceTaskPrice": {
        "members": [{"name": "AmountInUsd", "shape": "USD", "type": "structure"}],
        "type": "structure",
    },
    "PutModelPackageGroupPolicyInput": {
        "members": [
            {"name": "ModelPackageGroupName", "shape": "EntityName", "type": "string"},
            {"name": "ResourcePolicy", "shape": "PolicyString", "type": "string"},
        ],
        "type": "structure",
    },
    "PutModelPackageGroupPolicyOutput": {
        "members": [
            {"name": "ModelPackageGroupArn", "shape": "ModelPackageGroupArn", "type": "string"}
        ],
        "type": "structure",
    },
    "PutRecordRequest": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupNameOrArn", "type": "string"},
            {"name": "Record", "shape": "Record", "type": "list"},
            {"name": "TargetStores", "shape": "TargetStores", "type": "list"},
            {"name": "TtlDuration", "shape": "TtlDuration", "type": "structure"},
        ],
        "type": "structure",
    },
    "QualityCheckStepMetadata": {
        "members": [
            {"name": "CheckType", "shape": "String256", "type": "string"},
            {
                "name": "BaselineUsedForDriftCheckStatistics",
                "shape": "String1024",
                "type": "string",
            },
            {
                "name": "BaselineUsedForDriftCheckConstraints",
                "shape": "String1024",
                "type": "string",
            },
            {"name": "CalculatedBaselineStatistics", "shape": "String1024", "type": "string"},
            {"name": "CalculatedBaselineConstraints", "shape": "String1024", "type": "string"},
            {"name": "ModelPackageGroupName", "shape": "String256", "type": "string"},
            {"name": "ViolationReport", "shape": "String1024", "type": "string"},
            {"name": "CheckJobArn", "shape": "String256", "type": "string"},
            {"name": "SkipCheck", "shape": "Boolean", "type": "boolean"},
            {"name": "RegisterNewBaseline", "shape": "Boolean", "type": "boolean"},
        ],
        "type": "structure",
    },
    "QueryFilters": {
        "members": [
            {"name": "Types", "shape": "QueryTypes", "type": "list"},
            {"name": "LineageTypes", "shape": "QueryLineageTypes", "type": "list"},
            {"name": "CreatedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModifiedBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "ModifiedAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Properties", "shape": "QueryProperties", "type": "map"},
        ],
        "type": "structure",
    },
    "QueryLineageRequest": {
        "members": [
            {"name": "StartArns", "shape": "QueryLineageStartArns", "type": "list"},
            {"name": "Direction", "shape": "Direction", "type": "string"},
            {"name": "IncludeEdges", "shape": "Boolean", "type": "boolean"},
            {"name": "Filters", "shape": "QueryFilters", "type": "structure"},
            {"name": "MaxDepth", "shape": "QueryLineageMaxDepth", "type": "integer"},
            {"name": "MaxResults", "shape": "QueryLineageMaxResults", "type": "integer"},
            {"name": "NextToken", "shape": "String8192", "type": "string"},
        ],
        "type": "structure",
    },
    "QueryLineageResponse": {
        "members": [
            {"name": "Vertices", "shape": "Vertices", "type": "list"},
            {"name": "Edges", "shape": "Edges", "type": "list"},
            {"name": "NextToken", "shape": "String8192", "type": "string"},
        ],
        "type": "structure",
    },
    "QueryLineageStartArns": {
        "member_shape": "AssociationEntityArn",
        "member_type": "string",
        "type": "list",
    },
    "QueryLineageTypes": {"member_shape": "LineageType", "member_type": "string", "type": "list"},
    "QueryProperties": {
        "key_shape": "String256",
        "key_type": "string",
        "type": "map",
        "value_shape": "String256",
        "value_type": "string",
    },
    "QueryTypes": {"member_shape": "String40", "member_type": "string", "type": "list"},
    "RSessionAppSettings": {
        "members": [
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {"name": "CustomImages", "shape": "CustomImages", "type": "list"},
        ],
        "type": "structure",
    },
    "RStudioServerProAppSettings": {
        "members": [
            {"name": "AccessStatus", "shape": "RStudioServerProAccessStatus", "type": "string"},
            {"name": "UserGroup", "shape": "RStudioServerProUserGroup", "type": "string"},
        ],
        "type": "structure",
    },
    "RStudioServerProDomainSettings": {
        "members": [
            {"name": "DomainExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "RStudioConnectUrl", "shape": "String", "type": "string"},
            {"name": "RStudioPackageManagerUrl", "shape": "String", "type": "string"},
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
        ],
        "type": "structure",
    },
    "RStudioServerProDomainSettingsForUpdate": {
        "members": [
            {"name": "DomainExecutionRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {"name": "RStudioConnectUrl", "shape": "String", "type": "string"},
            {"name": "RStudioPackageManagerUrl", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "RawMetricData": {
        "members": [
            {"name": "MetricName", "shape": "MetricName", "type": "string"},
            {"name": "Timestamp", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Step", "shape": "Step", "type": "integer"},
            {"name": "Value", "shape": "Double", "type": "double"},
        ],
        "type": "structure",
    },
    "RawMetricDataList": {
        "member_shape": "RawMetricData",
        "member_type": "structure",
        "type": "list",
    },
    "RealTimeInferenceConfig": {
        "members": [
            {"name": "InstanceType", "shape": "InstanceType", "type": "string"},
            {"name": "InstanceCount", "shape": "TaskCount", "type": "integer"},
        ],
        "type": "structure",
    },
    "RealTimeInferenceRecommendation": {
        "members": [
            {"name": "RecommendationId", "shape": "String", "type": "string"},
            {"name": "InstanceType", "shape": "ProductionVariantInstanceType", "type": "string"},
            {"name": "Environment", "shape": "EnvironmentMap", "type": "map"},
        ],
        "type": "structure",
    },
    "RealTimeInferenceRecommendations": {
        "member_shape": "RealTimeInferenceRecommendation",
        "member_type": "structure",
        "type": "list",
    },
    "RealtimeInferenceInstanceTypes": {
        "member_shape": "ProductionVariantInstanceType",
        "member_type": "string",
        "type": "list",
    },
    "RecommendationJobCompiledOutputConfig": {
        "members": [{"name": "S3OutputUri", "shape": "S3Uri", "type": "string"}],
        "type": "structure",
    },
    "RecommendationJobContainerConfig": {
        "members": [
            {"name": "Domain", "shape": "String", "type": "string"},
            {"name": "Task", "shape": "String", "type": "string"},
            {"name": "Framework", "shape": "String", "type": "string"},
            {
                "name": "FrameworkVersion",
                "shape": "RecommendationJobFrameworkVersion",
                "type": "string",
            },
            {
                "name": "PayloadConfig",
                "shape": "RecommendationJobPayloadConfig",
                "type": "structure",
            },
            {"name": "NearestModelName", "shape": "String", "type": "string"},
            {
                "name": "SupportedInstanceTypes",
                "shape": "RecommendationJobSupportedInstanceTypes",
                "type": "list",
            },
            {
                "name": "SupportedEndpointType",
                "shape": "RecommendationJobSupportedEndpointType",
                "type": "string",
            },
            {
                "name": "DataInputConfig",
                "shape": "RecommendationJobDataInputConfig",
                "type": "string",
            },
            {
                "name": "SupportedResponseMIMETypes",
                "shape": "RecommendationJobSupportedResponseMIMETypes",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "RecommendationJobInferenceBenchmark": {
        "members": [
            {"name": "Metrics", "shape": "RecommendationMetrics", "type": "structure"},
            {"name": "EndpointMetrics", "shape": "InferenceMetrics", "type": "structure"},
            {
                "name": "EndpointConfiguration",
                "shape": "EndpointOutputConfiguration",
                "type": "structure",
            },
            {"name": "ModelConfiguration", "shape": "ModelConfiguration", "type": "structure"},
            {"name": "FailureReason", "shape": "RecommendationFailureReason", "type": "string"},
            {"name": "InvocationEndTime", "shape": "InvocationEndTime", "type": "timestamp"},
            {"name": "InvocationStartTime", "shape": "InvocationStartTime", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "RecommendationJobInputConfig": {
        "members": [
            {"name": "ModelPackageVersionArn", "shape": "ModelPackageArn", "type": "string"},
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {"name": "JobDurationInSeconds", "shape": "JobDurationInSeconds", "type": "integer"},
            {"name": "TrafficPattern", "shape": "TrafficPattern", "type": "structure"},
            {
                "name": "ResourceLimit",
                "shape": "RecommendationJobResourceLimit",
                "type": "structure",
            },
            {
                "name": "EndpointConfigurations",
                "shape": "EndpointInputConfigurations",
                "type": "list",
            },
            {"name": "VolumeKmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {
                "name": "ContainerConfig",
                "shape": "RecommendationJobContainerConfig",
                "type": "structure",
            },
            {"name": "Endpoints", "shape": "Endpoints", "type": "list"},
            {"name": "VpcConfig", "shape": "RecommendationJobVpcConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "RecommendationJobOutputConfig": {
        "members": [
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {
                "name": "CompiledOutputConfig",
                "shape": "RecommendationJobCompiledOutputConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "RecommendationJobPayloadConfig": {
        "members": [
            {"name": "SamplePayloadUrl", "shape": "S3Uri", "type": "string"},
            {
                "name": "SupportedContentTypes",
                "shape": "RecommendationJobSupportedContentTypes",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "RecommendationJobResourceLimit": {
        "members": [
            {"name": "MaxNumberOfTests", "shape": "MaxNumberOfTests", "type": "integer"},
            {"name": "MaxParallelOfTests", "shape": "MaxParallelOfTests", "type": "integer"},
        ],
        "type": "structure",
    },
    "RecommendationJobStoppingConditions": {
        "members": [
            {"name": "MaxInvocations", "shape": "Integer", "type": "integer"},
            {"name": "ModelLatencyThresholds", "shape": "ModelLatencyThresholds", "type": "list"},
            {"name": "FlatInvocations", "shape": "FlatInvocations", "type": "string"},
        ],
        "type": "structure",
    },
    "RecommendationJobSupportedContentTypes": {
        "member_shape": "RecommendationJobSupportedContentType",
        "member_type": "string",
        "type": "list",
    },
    "RecommendationJobSupportedInstanceTypes": {
        "member_shape": "String",
        "member_type": "string",
        "type": "list",
    },
    "RecommendationJobSupportedResponseMIMETypes": {
        "member_shape": "RecommendationJobSupportedResponseMIMEType",
        "member_type": "string",
        "type": "list",
    },
    "RecommendationJobVpcConfig": {
        "members": [
            {
                "name": "SecurityGroupIds",
                "shape": "RecommendationJobVpcSecurityGroupIds",
                "type": "list",
            },
            {"name": "Subnets", "shape": "RecommendationJobVpcSubnets", "type": "list"},
        ],
        "type": "structure",
    },
    "RecommendationJobVpcSecurityGroupIds": {
        "member_shape": "RecommendationJobVpcSecurityGroupId",
        "member_type": "string",
        "type": "list",
    },
    "RecommendationJobVpcSubnets": {
        "member_shape": "RecommendationJobVpcSubnetId",
        "member_type": "string",
        "type": "list",
    },
    "RecommendationMetrics": {
        "members": [
            {"name": "CostPerHour", "shape": "Float", "type": "float"},
            {"name": "CostPerInference", "shape": "Float", "type": "float"},
            {"name": "MaxInvocations", "shape": "Integer", "type": "integer"},
            {"name": "ModelLatency", "shape": "Integer", "type": "integer"},
            {"name": "CpuUtilization", "shape": "UtilizationMetric", "type": "float"},
            {"name": "MemoryUtilization", "shape": "UtilizationMetric", "type": "float"},
            {"name": "ModelSetupTime", "shape": "ModelSetupTime", "type": "integer"},
        ],
        "type": "structure",
    },
    "Record": {"member_shape": "FeatureValue", "member_type": "structure", "type": "list"},
    "RecordIdentifiers": {"member_shape": "ValueAsString", "member_type": "string", "type": "list"},
    "RedshiftDatasetDefinition": {
        "members": [
            {"name": "ClusterId", "shape": "RedshiftClusterId", "type": "string"},
            {"name": "Database", "shape": "RedshiftDatabase", "type": "string"},
            {"name": "DbUser", "shape": "RedshiftUserName", "type": "string"},
            {"name": "QueryString", "shape": "RedshiftQueryString", "type": "string"},
            {"name": "ClusterRoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "OutputS3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "OutputFormat", "shape": "RedshiftResultFormat", "type": "string"},
            {
                "name": "OutputCompression",
                "shape": "RedshiftResultCompressionType",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "RegisterDevicesRequest": {
        "members": [
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "Devices", "shape": "Devices", "type": "list"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "RegisterModelStepMetadata": {
        "members": [{"name": "Arn", "shape": "String256", "type": "string"}],
        "type": "structure",
    },
    "RemoteDebugConfig": {
        "members": [{"name": "EnableRemoteDebug", "shape": "EnableRemoteDebug", "type": "boolean"}],
        "type": "structure",
    },
    "RemoteDebugConfigForUpdate": {
        "members": [{"name": "EnableRemoteDebug", "shape": "EnableRemoteDebug", "type": "boolean"}],
        "type": "structure",
    },
    "RenderUiTemplateRequest": {
        "members": [
            {"name": "UiTemplate", "shape": "UiTemplate", "type": "structure"},
            {"name": "Task", "shape": "RenderableTask", "type": "structure"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "HumanTaskUiArn", "shape": "HumanTaskUiArn", "type": "string"},
        ],
        "type": "structure",
    },
    "RenderUiTemplateResponse": {
        "members": [
            {"name": "RenderedContent", "shape": "String", "type": "string"},
            {"name": "Errors", "shape": "RenderingErrorList", "type": "list"},
        ],
        "type": "structure",
    },
    "RenderableTask": {
        "members": [{"name": "Input", "shape": "TaskInput", "type": "string"}],
        "type": "structure",
    },
    "RenderingError": {
        "members": [
            {"name": "Code", "shape": "String", "type": "string"},
            {"name": "Message", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "RenderingErrorList": {
        "member_shape": "RenderingError",
        "member_type": "structure",
        "type": "list",
    },
    "RepositoryAuthConfig": {
        "members": [
            {
                "name": "RepositoryCredentialsProviderArn",
                "shape": "RepositoryCredentialsProviderArn",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "ReservedCapacityOffering": {
        "members": [
            {"name": "InstanceType", "shape": "ReservedCapacityInstanceType", "type": "string"},
            {"name": "InstanceCount", "shape": "ReservedCapacityInstanceCount", "type": "integer"},
            {"name": "AvailabilityZone", "shape": "AvailabilityZone", "type": "string"},
            {"name": "DurationHours", "shape": "ReservedCapacityDurationHours", "type": "long"},
            {"name": "DurationMinutes", "shape": "ReservedCapacityDurationMinutes", "type": "long"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ReservedCapacityOfferings": {
        "member_shape": "ReservedCapacityOffering",
        "member_type": "structure",
        "type": "list",
    },
    "ReservedCapacitySummaries": {
        "member_shape": "ReservedCapacitySummary",
        "member_type": "structure",
        "type": "list",
    },
    "ReservedCapacitySummary": {
        "members": [
            {"name": "ReservedCapacityArn", "shape": "ReservedCapacityArn", "type": "string"},
            {"name": "InstanceType", "shape": "ReservedCapacityInstanceType", "type": "string"},
            {"name": "TotalInstanceCount", "shape": "TotalInstanceCount", "type": "integer"},
            {"name": "Status", "shape": "ReservedCapacityStatus", "type": "string"},
            {"name": "AvailabilityZone", "shape": "AvailabilityZone", "type": "string"},
            {"name": "DurationHours", "shape": "ReservedCapacityDurationHours", "type": "long"},
            {"name": "DurationMinutes", "shape": "ReservedCapacityDurationMinutes", "type": "long"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ResolvedAttributes": {
        "members": [
            {"name": "AutoMLJobObjective", "shape": "AutoMLJobObjective", "type": "structure"},
            {"name": "ProblemType", "shape": "ProblemType", "type": "string"},
            {
                "name": "CompletionCriteria",
                "shape": "AutoMLJobCompletionCriteria",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "ResourceCatalog": {
        "members": [
            {"name": "ResourceCatalogArn", "shape": "ResourceCatalogArn", "type": "string"},
            {"name": "ResourceCatalogName", "shape": "ResourceCatalogName", "type": "string"},
            {"name": "Description", "shape": "ResourceCatalogDescription", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "ResourceCatalogList": {
        "member_shape": "ResourceCatalog",
        "member_type": "structure",
        "type": "list",
    },
    "ResourceConfig": {
        "members": [
            {"name": "InstanceType", "shape": "TrainingInstanceType", "type": "string"},
            {"name": "InstanceCount", "shape": "TrainingInstanceCount", "type": "integer"},
            {"name": "VolumeSizeInGB", "shape": "VolumeSizeInGB", "type": "integer"},
            {"name": "VolumeKmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {
                "name": "KeepAlivePeriodInSeconds",
                "shape": "KeepAlivePeriodInSeconds",
                "type": "integer",
            },
            {"name": "InstanceGroups", "shape": "InstanceGroups", "type": "list"},
            {"name": "TrainingPlanArn", "shape": "TrainingPlanArn", "type": "string"},
        ],
        "type": "structure",
    },
    "ResourceConfigForUpdate": {
        "members": [
            {
                "name": "KeepAlivePeriodInSeconds",
                "shape": "KeepAlivePeriodInSeconds",
                "type": "integer",
            }
        ],
        "type": "structure",
    },
    "ResourceInUse": {
        "members": [{"name": "Message", "shape": "FailureReason", "type": "string"}],
        "type": "structure",
    },
    "ResourceLimitExceeded": {
        "members": [{"name": "Message", "shape": "FailureReason", "type": "string"}],
        "type": "structure",
    },
    "ResourceLimits": {
        "members": [
            {
                "name": "MaxNumberOfTrainingJobs",
                "shape": "MaxNumberOfTrainingJobs",
                "type": "integer",
            },
            {
                "name": "MaxParallelTrainingJobs",
                "shape": "MaxParallelTrainingJobs",
                "type": "integer",
            },
            {
                "name": "MaxRuntimeInSeconds",
                "shape": "HyperParameterTuningMaxRuntimeInSeconds",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "ResourceNotFound": {
        "members": [{"name": "Message", "shape": "FailureReason", "type": "string"}],
        "type": "structure",
    },
    "ResourceSharingConfig": {
        "members": [
            {"name": "Strategy", "shape": "ResourceSharingStrategy", "type": "string"},
            {"name": "BorrowLimit", "shape": "BorrowLimit", "type": "integer"},
        ],
        "type": "structure",
    },
    "ResourceSpec": {
        "members": [
            {"name": "SageMakerImageArn", "shape": "ImageArn", "type": "string"},
            {"name": "SageMakerImageVersionArn", "shape": "ImageVersionArn", "type": "string"},
            {"name": "SageMakerImageVersionAlias", "shape": "ImageVersionAlias", "type": "string"},
            {"name": "InstanceType", "shape": "AppInstanceType", "type": "string"},
            {"name": "LifecycleConfigArn", "shape": "StudioLifecycleConfigArn", "type": "string"},
        ],
        "type": "structure",
    },
    "ResponseMIMETypes": {
        "member_shape": "ResponseMIMEType",
        "member_type": "string",
        "type": "list",
    },
    "ResponseStream": {
        "members": [
            {"name": "PayloadPart", "shape": "PayloadPart", "type": "structure"},
            {"name": "ModelStreamError", "shape": "ModelStreamError", "type": "structure"},
            {
                "name": "InternalStreamFailure",
                "shape": "InternalStreamFailure",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "RetentionPolicy": {
        "members": [{"name": "HomeEfsFileSystem", "shape": "RetentionType", "type": "string"}],
        "type": "structure",
    },
    "RetryPipelineExecutionRequest": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"},
            {"name": "ClientRequestToken", "shape": "IdempotencyToken", "type": "string"},
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "RetryPipelineExecutionResponse": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "RetryStrategy": {
        "members": [
            {"name": "MaximumRetryAttempts", "shape": "MaximumRetryAttempts", "type": "integer"}
        ],
        "type": "structure",
    },
    "RollingUpdatePolicy": {
        "members": [
            {"name": "MaximumBatchSize", "shape": "CapacitySize", "type": "structure"},
            {"name": "WaitIntervalInSeconds", "shape": "WaitIntervalInSeconds", "type": "integer"},
            {
                "name": "MaximumExecutionTimeoutInSeconds",
                "shape": "MaximumExecutionTimeoutInSeconds",
                "type": "integer",
            },
            {"name": "RollbackMaximumBatchSize", "shape": "CapacitySize", "type": "structure"},
        ],
        "type": "structure",
    },
    "RuleParameters": {
        "key_shape": "ConfigKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "ConfigValue",
        "value_type": "string",
    },
    "S3DataSource": {
        "members": [
            {"name": "S3DataType", "shape": "S3DataType", "type": "string"},
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "S3DataDistributionType", "shape": "S3DataDistribution", "type": "string"},
            {"name": "AttributeNames", "shape": "AttributeNames", "type": "list"},
            {"name": "InstanceGroupNames", "shape": "InstanceGroupNames", "type": "list"},
            {"name": "ModelAccessConfig", "shape": "ModelAccessConfig", "type": "structure"},
            {"name": "HubAccessConfig", "shape": "HubAccessConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "S3ModelDataSource": {
        "members": [
            {"name": "S3Uri", "shape": "S3ModelUri", "type": "string"},
            {"name": "S3DataType", "shape": "S3ModelDataType", "type": "string"},
            {"name": "CompressionType", "shape": "ModelCompressionType", "type": "string"},
            {"name": "ModelAccessConfig", "shape": "ModelAccessConfig", "type": "structure"},
            {"name": "HubAccessConfig", "shape": "InferenceHubAccessConfig", "type": "structure"},
            {"name": "ManifestS3Uri", "shape": "S3ModelUri", "type": "string"},
            {"name": "ETag", "shape": "String", "type": "string"},
            {"name": "ManifestEtag", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "S3Presign": {
        "members": [
            {"name": "IamPolicyConstraints", "shape": "IamPolicyConstraints", "type": "structure"}
        ],
        "type": "structure",
    },
    "S3StorageConfig": {
        "members": [
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
            {"name": "ResolvedOutputS3Uri", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "SageMakerImageVersionAliases": {
        "member_shape": "SageMakerImageVersionAlias",
        "member_type": "string",
        "type": "list",
    },
    "SageMakerResourceNames": {
        "member_shape": "SageMakerResourceName",
        "member_type": "string",
        "type": "list",
    },
    "ScalingPolicies": {
        "member_shape": "ScalingPolicy",
        "member_type": "structure",
        "type": "list",
    },
    "ScalingPolicy": {
        "members": [
            {
                "name": "TargetTracking",
                "shape": "TargetTrackingScalingPolicyConfiguration",
                "type": "structure",
            }
        ],
        "type": "structure",
    },
    "ScalingPolicyMetric": {
        "members": [
            {"name": "InvocationsPerInstance", "shape": "Integer", "type": "integer"},
            {"name": "ModelLatency", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "ScalingPolicyObjective": {
        "members": [
            {"name": "MinInvocationsPerMinute", "shape": "Integer", "type": "integer"},
            {"name": "MaxInvocationsPerMinute", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "ScheduleConfig": {
        "members": [
            {"name": "ScheduleExpression", "shape": "ScheduleExpression", "type": "string"},
            {"name": "DataAnalysisStartTime", "shape": "String", "type": "string"},
            {"name": "DataAnalysisEndTime", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "SchedulerConfig": {
        "members": [
            {"name": "PriorityClasses", "shape": "PriorityClassList", "type": "list"},
            {"name": "FairShare", "shape": "FairShare", "type": "string"},
        ],
        "type": "structure",
    },
    "SearchExpression": {
        "members": [
            {"name": "Filters", "shape": "FilterList", "type": "list"},
            {"name": "NestedFilters", "shape": "NestedFiltersList", "type": "list"},
            {"name": "SubExpressions", "shape": "SearchExpressionList", "type": "list"},
            {"name": "Operator", "shape": "BooleanOperator", "type": "string"},
        ],
        "type": "structure",
    },
    "SearchExpressionList": {
        "member_shape": "SearchExpression",
        "member_type": "structure",
        "type": "list",
    },
    "SearchRecord": {
        "members": [
            {"name": "TrainingJob", "shape": "TrainingJob", "type": "structure"},
            {"name": "Experiment", "shape": "Experiment", "type": "structure"},
            {"name": "Trial", "shape": "Trial", "type": "structure"},
            {"name": "TrialComponent", "shape": "TrialComponent", "type": "structure"},
            {"name": "Endpoint", "shape": "Endpoint", "type": "structure"},
            {"name": "ModelPackage", "shape": "ModelPackage", "type": "structure"},
            {"name": "ModelPackageGroup", "shape": "ModelPackageGroup", "type": "structure"},
            {"name": "Pipeline", "shape": "Pipeline", "type": "structure"},
            {"name": "PipelineExecution", "shape": "PipelineExecution", "type": "structure"},
            {"name": "FeatureGroup", "shape": "FeatureGroup", "type": "structure"},
            {"name": "FeatureMetadata", "shape": "FeatureMetadata", "type": "structure"},
            {"name": "Project", "shape": "Project", "type": "structure"},
            {
                "name": "HyperParameterTuningJob",
                "shape": "HyperParameterTuningJobSearchEntity",
                "type": "structure",
            },
            {"name": "ModelCard", "shape": "ModelCard", "type": "structure"},
            {"name": "Model", "shape": "ModelDashboardModel", "type": "structure"},
        ],
        "type": "structure",
    },
    "SearchRequest": {
        "members": [
            {"name": "Resource", "shape": "ResourceType", "type": "string"},
            {"name": "SearchExpression", "shape": "SearchExpression", "type": "structure"},
            {"name": "SortBy", "shape": "ResourcePropertyName", "type": "string"},
            {"name": "SortOrder", "shape": "SearchSortOrder", "type": "string"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
            {"name": "MaxResults", "shape": "MaxResults", "type": "integer"},
            {
                "name": "CrossAccountFilterOption",
                "shape": "CrossAccountFilterOption",
                "type": "string",
            },
            {"name": "VisibilityConditions", "shape": "VisibilityConditionsList", "type": "list"},
        ],
        "type": "structure",
    },
    "SearchResponse": {
        "members": [
            {"name": "Results", "shape": "SearchResultsList", "type": "list"},
            {"name": "NextToken", "shape": "NextToken", "type": "string"},
        ],
        "type": "structure",
    },
    "SearchResultsList": {
        "member_shape": "SearchRecord",
        "member_type": "structure",
        "type": "list",
    },
    "SearchTrainingPlanOfferingsRequest": {
        "members": [
            {"name": "InstanceType", "shape": "ReservedCapacityInstanceType", "type": "string"},
            {"name": "InstanceCount", "shape": "ReservedCapacityInstanceCount", "type": "integer"},
            {"name": "StartTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "DurationHours", "shape": "TrainingPlanDurationHoursInput", "type": "long"},
            {"name": "TargetResources", "shape": "SageMakerResourceNames", "type": "list"},
        ],
        "type": "structure",
    },
    "SearchTrainingPlanOfferingsResponse": {
        "members": [
            {"name": "TrainingPlanOfferings", "shape": "TrainingPlanOfferings", "type": "list"}
        ],
        "type": "structure",
    },
    "SecondaryStatusTransition": {
        "members": [
            {"name": "Status", "shape": "SecondaryStatus", "type": "string"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "StatusMessage", "shape": "StatusMessage", "type": "string"},
        ],
        "type": "structure",
    },
    "SecondaryStatusTransitions": {
        "member_shape": "SecondaryStatusTransition",
        "member_type": "structure",
        "type": "list",
    },
    "SecurityGroupIds": {
        "member_shape": "SecurityGroupId",
        "member_type": "string",
        "type": "list",
    },
    "SelectedStep": {
        "members": [{"name": "StepName", "shape": "String256", "type": "string"}],
        "type": "structure",
    },
    "SelectedStepList": {
        "member_shape": "SelectedStep",
        "member_type": "structure",
        "type": "list",
    },
    "SelectiveExecutionConfig": {
        "members": [
            {
                "name": "SourcePipelineExecutionArn",
                "shape": "PipelineExecutionArn",
                "type": "string",
            },
            {"name": "SelectedSteps", "shape": "SelectedStepList", "type": "list"},
        ],
        "type": "structure",
    },
    "SelectiveExecutionResult": {
        "members": [
            {
                "name": "SourcePipelineExecutionArn",
                "shape": "PipelineExecutionArn",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "SendPipelineExecutionStepFailureRequest": {
        "members": [
            {"name": "CallbackToken", "shape": "CallbackToken", "type": "string"},
            {"name": "FailureReason", "shape": "String256", "type": "string"},
            {"name": "ClientRequestToken", "shape": "IdempotencyToken", "type": "string"},
        ],
        "type": "structure",
    },
    "SendPipelineExecutionStepFailureResponse": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "SendPipelineExecutionStepSuccessRequest": {
        "members": [
            {"name": "CallbackToken", "shape": "CallbackToken", "type": "string"},
            {"name": "OutputParameters", "shape": "OutputParameterList", "type": "list"},
            {"name": "ClientRequestToken", "shape": "IdempotencyToken", "type": "string"},
        ],
        "type": "structure",
    },
    "SendPipelineExecutionStepSuccessResponse": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "ServiceCatalogProvisionedProductDetails": {
        "members": [
            {"name": "ProvisionedProductId", "shape": "ServiceCatalogEntityId", "type": "string"},
            {
                "name": "ProvisionedProductStatusMessage",
                "shape": "ProvisionedProductStatusMessage",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "ServiceCatalogProvisioningDetails": {
        "members": [
            {"name": "ProductId", "shape": "ServiceCatalogEntityId", "type": "string"},
            {"name": "ProvisioningArtifactId", "shape": "ServiceCatalogEntityId", "type": "string"},
            {"name": "PathId", "shape": "ServiceCatalogEntityId", "type": "string"},
            {"name": "ProvisioningParameters", "shape": "ProvisioningParameters", "type": "list"},
        ],
        "type": "structure",
    },
    "ServiceCatalogProvisioningUpdateDetails": {
        "members": [
            {"name": "ProvisioningArtifactId", "shape": "ServiceCatalogEntityId", "type": "string"},
            {"name": "ProvisioningParameters", "shape": "ProvisioningParameters", "type": "list"},
        ],
        "type": "structure",
    },
    "ServiceUnavailable": {
        "members": [{"name": "Message", "shape": "Message", "type": "string"}],
        "type": "structure",
    },
    "SessionChainingConfig": {
        "members": [
            {
                "name": "EnableSessionTagChaining",
                "shape": "EnableSessionTagChaining",
                "type": "boolean",
            }
        ],
        "type": "structure",
    },
    "ShadowModeConfig": {
        "members": [
            {"name": "SourceModelVariantName", "shape": "ModelVariantName", "type": "string"},
            {
                "name": "ShadowModelVariants",
                "shape": "ShadowModelVariantConfigList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "ShadowModelVariantConfig": {
        "members": [
            {"name": "ShadowModelVariantName", "shape": "ModelVariantName", "type": "string"},
            {"name": "SamplingPercentage", "shape": "Percentage", "type": "integer"},
        ],
        "type": "structure",
    },
    "ShadowModelVariantConfigList": {
        "member_shape": "ShadowModelVariantConfig",
        "member_type": "structure",
        "type": "list",
    },
    "SharingSettings": {
        "members": [
            {"name": "NotebookOutputOption", "shape": "NotebookOutputOption", "type": "string"},
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {"name": "S3KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "ShuffleConfig": {
        "members": [{"name": "Seed", "shape": "Seed", "type": "long"}],
        "type": "structure",
    },
    "SourceAlgorithm": {
        "members": [
            {"name": "ModelDataUrl", "shape": "Url", "type": "string"},
            {"name": "ModelDataSource", "shape": "ModelDataSource", "type": "structure"},
            {"name": "ModelDataETag", "shape": "String", "type": "string"},
            {"name": "AlgorithmName", "shape": "ArnOrName", "type": "string"},
        ],
        "type": "structure",
    },
    "SourceAlgorithmList": {
        "member_shape": "SourceAlgorithm",
        "member_type": "structure",
        "type": "list",
    },
    "SourceAlgorithmSpecification": {
        "members": [{"name": "SourceAlgorithms", "shape": "SourceAlgorithmList", "type": "list"}],
        "type": "structure",
    },
    "SourceIpConfig": {
        "members": [{"name": "Cidrs", "shape": "Cidrs", "type": "list"}],
        "type": "structure",
    },
    "SpaceAppLifecycleManagement": {
        "members": [{"name": "IdleSettings", "shape": "SpaceIdleSettings", "type": "structure"}],
        "type": "structure",
    },
    "SpaceCodeEditorAppSettings": {
        "members": [
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {
                "name": "AppLifecycleManagement",
                "shape": "SpaceAppLifecycleManagement",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "SpaceDetails": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "Status", "shape": "SpaceStatus", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
            {"name": "SpaceSettingsSummary", "shape": "SpaceSettingsSummary", "type": "structure"},
            {
                "name": "SpaceSharingSettingsSummary",
                "shape": "SpaceSharingSettingsSummary",
                "type": "structure",
            },
            {
                "name": "OwnershipSettingsSummary",
                "shape": "OwnershipSettingsSummary",
                "type": "structure",
            },
            {"name": "SpaceDisplayName", "shape": "NonEmptyString64", "type": "string"},
        ],
        "type": "structure",
    },
    "SpaceIdleSettings": {
        "members": [
            {"name": "IdleTimeoutInMinutes", "shape": "IdleTimeoutInMinutes", "type": "integer"}
        ],
        "type": "structure",
    },
    "SpaceJupyterLabAppSettings": {
        "members": [
            {"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"},
            {"name": "CodeRepositories", "shape": "CodeRepositories", "type": "list"},
            {
                "name": "AppLifecycleManagement",
                "shape": "SpaceAppLifecycleManagement",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "SpaceList": {"member_shape": "SpaceDetails", "member_type": "structure", "type": "list"},
    "SpaceSettings": {
        "members": [
            {
                "name": "JupyterServerAppSettings",
                "shape": "JupyterServerAppSettings",
                "type": "structure",
            },
            {
                "name": "KernelGatewayAppSettings",
                "shape": "KernelGatewayAppSettings",
                "type": "structure",
            },
            {
                "name": "CodeEditorAppSettings",
                "shape": "SpaceCodeEditorAppSettings",
                "type": "structure",
            },
            {
                "name": "JupyterLabAppSettings",
                "shape": "SpaceJupyterLabAppSettings",
                "type": "structure",
            },
            {"name": "AppType", "shape": "AppType", "type": "string"},
            {"name": "SpaceStorageSettings", "shape": "SpaceStorageSettings", "type": "structure"},
            {"name": "CustomFileSystems", "shape": "CustomFileSystems", "type": "list"},
        ],
        "type": "structure",
    },
    "SpaceSettingsSummary": {
        "members": [
            {"name": "AppType", "shape": "AppType", "type": "string"},
            {"name": "SpaceStorageSettings", "shape": "SpaceStorageSettings", "type": "structure"},
        ],
        "type": "structure",
    },
    "SpaceSharingSettings": {
        "members": [{"name": "SharingType", "shape": "SharingType", "type": "string"}],
        "type": "structure",
    },
    "SpaceSharingSettingsSummary": {
        "members": [{"name": "SharingType", "shape": "SharingType", "type": "string"}],
        "type": "structure",
    },
    "SpaceStorageSettings": {
        "members": [
            {"name": "EbsStorageSettings", "shape": "EbsStorageSettings", "type": "structure"}
        ],
        "type": "structure",
    },
    "Stairs": {
        "members": [
            {"name": "DurationInSeconds", "shape": "TrafficDurationInSeconds", "type": "integer"},
            {"name": "NumberOfSteps", "shape": "NumberOfSteps", "type": "integer"},
            {"name": "UsersPerStep", "shape": "UsersPerStep", "type": "integer"},
        ],
        "type": "structure",
    },
    "StartEdgeDeploymentStageRequest": {
        "members": [
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "StageName", "shape": "EntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "StartInferenceExperimentRequest": {
        "members": [{"name": "Name", "shape": "InferenceExperimentName", "type": "string"}],
        "type": "structure",
    },
    "StartInferenceExperimentResponse": {
        "members": [
            {"name": "InferenceExperimentArn", "shape": "InferenceExperimentArn", "type": "string"}
        ],
        "type": "structure",
    },
    "StartMlflowTrackingServerRequest": {
        "members": [
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"}
        ],
        "type": "structure",
    },
    "StartMlflowTrackingServerResponse": {
        "members": [{"name": "TrackingServerArn", "shape": "TrackingServerArn", "type": "string"}],
        "type": "structure",
    },
    "StartMonitoringScheduleRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"}
        ],
        "type": "structure",
    },
    "StartNotebookInstanceInput": {
        "members": [
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"}
        ],
        "type": "structure",
    },
    "StartPipelineExecutionRequest": {
        "members": [
            {"name": "PipelineName", "shape": "PipelineNameOrArn", "type": "string"},
            {
                "name": "PipelineExecutionDisplayName",
                "shape": "PipelineExecutionName",
                "type": "string",
            },
            {"name": "PipelineParameters", "shape": "ParameterList", "type": "list"},
            {
                "name": "PipelineExecutionDescription",
                "shape": "PipelineExecutionDescription",
                "type": "string",
            },
            {"name": "ClientRequestToken", "shape": "IdempotencyToken", "type": "string"},
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
            {
                "name": "SelectiveExecutionConfig",
                "shape": "SelectiveExecutionConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "StartPipelineExecutionResponse": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "StopAutoMLJobRequest": {
        "members": [{"name": "AutoMLJobName", "shape": "AutoMLJobName", "type": "string"}],
        "type": "structure",
    },
    "StopCompilationJobRequest": {
        "members": [{"name": "CompilationJobName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "StopEdgeDeploymentStageRequest": {
        "members": [
            {"name": "EdgeDeploymentPlanName", "shape": "EntityName", "type": "string"},
            {"name": "StageName", "shape": "EntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "StopEdgePackagingJobRequest": {
        "members": [{"name": "EdgePackagingJobName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "StopHyperParameterTuningJobRequest": {
        "members": [
            {
                "name": "HyperParameterTuningJobName",
                "shape": "HyperParameterTuningJobName",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "StopInferenceExperimentRequest": {
        "members": [
            {"name": "Name", "shape": "InferenceExperimentName", "type": "string"},
            {"name": "ModelVariantActions", "shape": "ModelVariantActionMap", "type": "map"},
            {"name": "DesiredModelVariants", "shape": "ModelVariantConfigList", "type": "list"},
            {
                "name": "DesiredState",
                "shape": "InferenceExperimentStopDesiredState",
                "type": "string",
            },
            {"name": "Reason", "shape": "InferenceExperimentStatusReason", "type": "string"},
        ],
        "type": "structure",
    },
    "StopInferenceExperimentResponse": {
        "members": [
            {"name": "InferenceExperimentArn", "shape": "InferenceExperimentArn", "type": "string"}
        ],
        "type": "structure",
    },
    "StopInferenceRecommendationsJobRequest": {
        "members": [{"name": "JobName", "shape": "RecommendationJobName", "type": "string"}],
        "type": "structure",
    },
    "StopLabelingJobRequest": {
        "members": [{"name": "LabelingJobName", "shape": "LabelingJobName", "type": "string"}],
        "type": "structure",
    },
    "StopMlflowTrackingServerRequest": {
        "members": [
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"}
        ],
        "type": "structure",
    },
    "StopMlflowTrackingServerResponse": {
        "members": [{"name": "TrackingServerArn", "shape": "TrackingServerArn", "type": "string"}],
        "type": "structure",
    },
    "StopMonitoringScheduleRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"}
        ],
        "type": "structure",
    },
    "StopNotebookInstanceInput": {
        "members": [
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"}
        ],
        "type": "structure",
    },
    "StopOptimizationJobRequest": {
        "members": [{"name": "OptimizationJobName", "shape": "EntityName", "type": "string"}],
        "type": "structure",
    },
    "StopPipelineExecutionRequest": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"},
            {"name": "ClientRequestToken", "shape": "IdempotencyToken", "type": "string"},
        ],
        "type": "structure",
    },
    "StopPipelineExecutionResponse": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "StopProcessingJobRequest": {
        "members": [{"name": "ProcessingJobName", "shape": "ProcessingJobName", "type": "string"}],
        "type": "structure",
    },
    "StopTrainingJobRequest": {
        "members": [{"name": "TrainingJobName", "shape": "TrainingJobName", "type": "string"}],
        "type": "structure",
    },
    "StopTransformJobRequest": {
        "members": [{"name": "TransformJobName", "shape": "TransformJobName", "type": "string"}],
        "type": "structure",
    },
    "StoppingCondition": {
        "members": [
            {"name": "MaxRuntimeInSeconds", "shape": "MaxRuntimeInSeconds", "type": "integer"},
            {"name": "MaxWaitTimeInSeconds", "shape": "MaxWaitTimeInSeconds", "type": "integer"},
            {
                "name": "MaxPendingTimeInSeconds",
                "shape": "MaxPendingTimeInSeconds",
                "type": "integer",
            },
        ],
        "type": "structure",
    },
    "StudioLifecycleConfigDetails": {
        "members": [
            {
                "name": "StudioLifecycleConfigArn",
                "shape": "StudioLifecycleConfigArn",
                "type": "string",
            },
            {
                "name": "StudioLifecycleConfigName",
                "shape": "StudioLifecycleConfigName",
                "type": "string",
            },
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "StudioLifecycleConfigAppType",
                "shape": "StudioLifecycleConfigAppType",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "StudioLifecycleConfigsList": {
        "member_shape": "StudioLifecycleConfigDetails",
        "member_type": "structure",
        "type": "list",
    },
    "StudioWebPortalSettings": {
        "members": [
            {"name": "HiddenMlTools", "shape": "HiddenMlToolsList", "type": "list"},
            {"name": "HiddenAppTypes", "shape": "HiddenAppTypesList", "type": "list"},
            {"name": "HiddenInstanceTypes", "shape": "HiddenInstanceTypesList", "type": "list"},
            {
                "name": "HiddenSageMakerImageVersionAliases",
                "shape": "HiddenSageMakerImageVersionAliasesList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "Subnets": {"member_shape": "SubnetId", "member_type": "string", "type": "list"},
    "SubscribedWorkteam": {
        "members": [
            {"name": "WorkteamArn", "shape": "WorkteamArn", "type": "string"},
            {"name": "MarketplaceTitle", "shape": "String200", "type": "string"},
            {"name": "SellerName", "shape": "String", "type": "string"},
            {"name": "MarketplaceDescription", "shape": "String200", "type": "string"},
            {"name": "ListingId", "shape": "String", "type": "string"},
        ],
        "type": "structure",
    },
    "SubscribedWorkteams": {
        "member_shape": "SubscribedWorkteam",
        "member_type": "structure",
        "type": "list",
    },
    "SuggestionQuery": {
        "members": [
            {"name": "PropertyNameQuery", "shape": "PropertyNameQuery", "type": "structure"}
        ],
        "type": "structure",
    },
    "TabularJobConfig": {
        "members": [
            {
                "name": "CandidateGenerationConfig",
                "shape": "CandidateGenerationConfig",
                "type": "structure",
            },
            {
                "name": "CompletionCriteria",
                "shape": "AutoMLJobCompletionCriteria",
                "type": "structure",
            },
            {"name": "FeatureSpecificationS3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "Mode", "shape": "AutoMLMode", "type": "string"},
            {
                "name": "GenerateCandidateDefinitionsOnly",
                "shape": "GenerateCandidateDefinitionsOnly",
                "type": "boolean",
            },
            {"name": "ProblemType", "shape": "ProblemType", "type": "string"},
            {"name": "TargetAttributeName", "shape": "TargetAttributeName", "type": "string"},
            {
                "name": "SampleWeightAttributeName",
                "shape": "SampleWeightAttributeName",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "TabularResolvedAttributes": {
        "members": [{"name": "ProblemType", "shape": "ProblemType", "type": "string"}],
        "type": "structure",
    },
    "Tag": {
        "members": [
            {"name": "Key", "shape": "TagKey", "type": "string"},
            {"name": "Value", "shape": "TagValue", "type": "string"},
        ],
        "type": "structure",
    },
    "TagKeyList": {"member_shape": "TagKey", "member_type": "string", "type": "list"},
    "TagList": {"member_shape": "Tag", "member_type": "structure", "type": "list"},
    "TargetPlatform": {
        "members": [
            {"name": "Os", "shape": "TargetPlatformOs", "type": "string"},
            {"name": "Arch", "shape": "TargetPlatformArch", "type": "string"},
            {"name": "Accelerator", "shape": "TargetPlatformAccelerator", "type": "string"},
        ],
        "type": "structure",
    },
    "TargetStores": {"member_shape": "TargetStore", "member_type": "string", "type": "list"},
    "TargetTrackingScalingPolicyConfiguration": {
        "members": [
            {"name": "MetricSpecification", "shape": "MetricSpecification", "type": "structure"},
            {"name": "TargetValue", "shape": "Double", "type": "double"},
        ],
        "type": "structure",
    },
    "TaskKeywords": {"member_shape": "TaskKeyword", "member_type": "string", "type": "list"},
    "TensorBoardAppSettings": {
        "members": [{"name": "DefaultResourceSpec", "shape": "ResourceSpec", "type": "structure"}],
        "type": "structure",
    },
    "TensorBoardOutputConfig": {
        "members": [
            {"name": "LocalPath", "shape": "DirectoryPath", "type": "string"},
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "TextClassificationJobConfig": {
        "members": [
            {
                "name": "CompletionCriteria",
                "shape": "AutoMLJobCompletionCriteria",
                "type": "structure",
            },
            {"name": "ContentColumn", "shape": "ContentColumn", "type": "string"},
            {"name": "TargetLabelColumn", "shape": "TargetLabelColumn", "type": "string"},
        ],
        "type": "structure",
    },
    "TextGenerationHyperParameters": {
        "key_shape": "TextGenerationHyperParameterKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "TextGenerationHyperParameterValue",
        "value_type": "string",
    },
    "TextGenerationJobConfig": {
        "members": [
            {
                "name": "CompletionCriteria",
                "shape": "AutoMLJobCompletionCriteria",
                "type": "structure",
            },
            {"name": "BaseModelName", "shape": "BaseModelName", "type": "string"},
            {
                "name": "TextGenerationHyperParameters",
                "shape": "TextGenerationHyperParameters",
                "type": "map",
            },
            {"name": "ModelAccessConfig", "shape": "ModelAccessConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "TextGenerationResolvedAttributes": {
        "members": [{"name": "BaseModelName", "shape": "BaseModelName", "type": "string"}],
        "type": "structure",
    },
    "ThroughputConfig": {
        "members": [
            {"name": "ThroughputMode", "shape": "ThroughputMode", "type": "string"},
            {"name": "ProvisionedReadCapacityUnits", "shape": "CapacityUnit", "type": "integer"},
            {"name": "ProvisionedWriteCapacityUnits", "shape": "CapacityUnit", "type": "integer"},
        ],
        "type": "structure",
    },
    "ThroughputConfigDescription": {
        "members": [
            {"name": "ThroughputMode", "shape": "ThroughputMode", "type": "string"},
            {"name": "ProvisionedReadCapacityUnits", "shape": "CapacityUnit", "type": "integer"},
            {"name": "ProvisionedWriteCapacityUnits", "shape": "CapacityUnit", "type": "integer"},
        ],
        "type": "structure",
    },
    "ThroughputConfigUpdate": {
        "members": [
            {"name": "ThroughputMode", "shape": "ThroughputMode", "type": "string"},
            {"name": "ProvisionedReadCapacityUnits", "shape": "CapacityUnit", "type": "integer"},
            {"name": "ProvisionedWriteCapacityUnits", "shape": "CapacityUnit", "type": "integer"},
        ],
        "type": "structure",
    },
    "TimeSeriesConfig": {
        "members": [
            {"name": "TargetAttributeName", "shape": "TargetAttributeName", "type": "string"},
            {"name": "TimestampAttributeName", "shape": "TimestampAttributeName", "type": "string"},
            {
                "name": "ItemIdentifierAttributeName",
                "shape": "ItemIdentifierAttributeName",
                "type": "string",
            },
            {"name": "GroupingAttributeNames", "shape": "GroupingAttributeNames", "type": "list"},
        ],
        "type": "structure",
    },
    "TimeSeriesForecastingJobConfig": {
        "members": [
            {"name": "FeatureSpecificationS3Uri", "shape": "S3Uri", "type": "string"},
            {
                "name": "CompletionCriteria",
                "shape": "AutoMLJobCompletionCriteria",
                "type": "structure",
            },
            {"name": "ForecastFrequency", "shape": "ForecastFrequency", "type": "string"},
            {"name": "ForecastHorizon", "shape": "ForecastHorizon", "type": "integer"},
            {"name": "ForecastQuantiles", "shape": "ForecastQuantiles", "type": "list"},
            {"name": "Transformations", "shape": "TimeSeriesTransformations", "type": "structure"},
            {"name": "TimeSeriesConfig", "shape": "TimeSeriesConfig", "type": "structure"},
            {"name": "HolidayConfig", "shape": "HolidayConfig", "type": "list"},
            {
                "name": "CandidateGenerationConfig",
                "shape": "CandidateGenerationConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "TimeSeriesForecastingSettings": {
        "members": [
            {"name": "Status", "shape": "FeatureStatus", "type": "string"},
            {"name": "AmazonForecastRoleArn", "shape": "RoleArn", "type": "string"},
        ],
        "type": "structure",
    },
    "TimeSeriesTransformations": {
        "members": [
            {"name": "Filling", "shape": "FillingTransformations", "type": "map"},
            {"name": "Aggregation", "shape": "AggregationTransformations", "type": "map"},
        ],
        "type": "structure",
    },
    "TrackingServerSummary": {
        "members": [
            {"name": "TrackingServerArn", "shape": "TrackingServerArn", "type": "string"},
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrackingServerStatus", "shape": "TrackingServerStatus", "type": "string"},
            {"name": "IsActive", "shape": "IsTrackingServerActive", "type": "string"},
            {"name": "MlflowVersion", "shape": "MlflowVersion", "type": "string"},
        ],
        "type": "structure",
    },
    "TrackingServerSummaryList": {
        "member_shape": "TrackingServerSummary",
        "member_type": "structure",
        "type": "list",
    },
    "TrafficPattern": {
        "members": [
            {"name": "TrafficType", "shape": "TrafficType", "type": "string"},
            {"name": "Phases", "shape": "Phases", "type": "list"},
            {"name": "Stairs", "shape": "Stairs", "type": "structure"},
        ],
        "type": "structure",
    },
    "TrafficRoutingConfig": {
        "members": [
            {"name": "Type", "shape": "TrafficRoutingConfigType", "type": "string"},
            {"name": "WaitIntervalInSeconds", "shape": "WaitIntervalInSeconds", "type": "integer"},
            {"name": "CanarySize", "shape": "CapacitySize", "type": "structure"},
            {"name": "LinearStepSize", "shape": "CapacitySize", "type": "structure"},
        ],
        "type": "structure",
    },
    "TrainingContainerArguments": {
        "member_shape": "TrainingContainerArgument",
        "member_type": "string",
        "type": "list",
    },
    "TrainingContainerEntrypoint": {
        "member_shape": "TrainingContainerEntrypointString",
        "member_type": "string",
        "type": "list",
    },
    "TrainingEnvironmentMap": {
        "key_shape": "TrainingEnvironmentKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "TrainingEnvironmentValue",
        "value_type": "string",
    },
    "TrainingImageConfig": {
        "members": [
            {
                "name": "TrainingRepositoryAccessMode",
                "shape": "TrainingRepositoryAccessMode",
                "type": "string",
            },
            {
                "name": "TrainingRepositoryAuthConfig",
                "shape": "TrainingRepositoryAuthConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "TrainingInstanceTypes": {
        "member_shape": "TrainingInstanceType",
        "member_type": "string",
        "type": "list",
    },
    "TrainingJob": {
        "members": [
            {"name": "TrainingJobName", "shape": "TrainingJobName", "type": "string"},
            {"name": "TrainingJobArn", "shape": "TrainingJobArn", "type": "string"},
            {"name": "TuningJobArn", "shape": "HyperParameterTuningJobArn", "type": "string"},
            {"name": "LabelingJobArn", "shape": "LabelingJobArn", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {"name": "ModelArtifacts", "shape": "ModelArtifacts", "type": "structure"},
            {"name": "TrainingJobStatus", "shape": "TrainingJobStatus", "type": "string"},
            {"name": "SecondaryStatus", "shape": "SecondaryStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "HyperParameters", "shape": "HyperParameters", "type": "map"},
            {
                "name": "AlgorithmSpecification",
                "shape": "AlgorithmSpecification",
                "type": "structure",
            },
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "InputDataConfig", "shape": "InputDataConfig", "type": "list"},
            {"name": "OutputDataConfig", "shape": "OutputDataConfig", "type": "structure"},
            {"name": "ResourceConfig", "shape": "ResourceConfig", "type": "structure"},
            {"name": "VpcConfig", "shape": "VpcConfig", "type": "structure"},
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "SecondaryStatusTransitions",
                "shape": "SecondaryStatusTransitions",
                "type": "list",
            },
            {"name": "FinalMetricDataList", "shape": "FinalMetricDataList", "type": "list"},
            {"name": "EnableNetworkIsolation", "shape": "Boolean", "type": "boolean"},
            {
                "name": "EnableInterContainerTrafficEncryption",
                "shape": "Boolean",
                "type": "boolean",
            },
            {"name": "EnableManagedSpotTraining", "shape": "Boolean", "type": "boolean"},
            {"name": "CheckpointConfig", "shape": "CheckpointConfig", "type": "structure"},
            {"name": "TrainingTimeInSeconds", "shape": "TrainingTimeInSeconds", "type": "integer"},
            {"name": "BillableTimeInSeconds", "shape": "BillableTimeInSeconds", "type": "integer"},
            {"name": "DebugHookConfig", "shape": "DebugHookConfig", "type": "structure"},
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
            {"name": "DebugRuleConfigurations", "shape": "DebugRuleConfigurations", "type": "list"},
            {
                "name": "TensorBoardOutputConfig",
                "shape": "TensorBoardOutputConfig",
                "type": "structure",
            },
            {
                "name": "DebugRuleEvaluationStatuses",
                "shape": "DebugRuleEvaluationStatuses",
                "type": "list",
            },
            {"name": "ProfilerConfig", "shape": "ProfilerConfig", "type": "structure"},
            {"name": "Environment", "shape": "TrainingEnvironmentMap", "type": "map"},
            {"name": "RetryStrategy", "shape": "RetryStrategy", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "TrainingJobDefinition": {
        "members": [
            {"name": "TrainingInputMode", "shape": "TrainingInputMode", "type": "string"},
            {"name": "HyperParameters", "shape": "HyperParameters", "type": "map"},
            {"name": "InputDataConfig", "shape": "InputDataConfig", "type": "list"},
            {"name": "OutputDataConfig", "shape": "OutputDataConfig", "type": "structure"},
            {"name": "ResourceConfig", "shape": "ResourceConfig", "type": "structure"},
            {"name": "StoppingCondition", "shape": "StoppingCondition", "type": "structure"},
        ],
        "type": "structure",
    },
    "TrainingJobStatusCounters": {
        "members": [
            {"name": "Completed", "shape": "TrainingJobStatusCounter", "type": "integer"},
            {"name": "InProgress", "shape": "TrainingJobStatusCounter", "type": "integer"},
            {"name": "RetryableError", "shape": "TrainingJobStatusCounter", "type": "integer"},
            {"name": "NonRetryableError", "shape": "TrainingJobStatusCounter", "type": "integer"},
            {"name": "Stopped", "shape": "TrainingJobStatusCounter", "type": "integer"},
        ],
        "type": "structure",
    },
    "TrainingJobStepMetadata": {
        "members": [{"name": "Arn", "shape": "TrainingJobArn", "type": "string"}],
        "type": "structure",
    },
    "TrainingJobSummaries": {
        "member_shape": "TrainingJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "TrainingJobSummary": {
        "members": [
            {"name": "TrainingJobName", "shape": "TrainingJobName", "type": "string"},
            {"name": "TrainingJobArn", "shape": "TrainingJobArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TrainingJobStatus", "shape": "TrainingJobStatus", "type": "string"},
            {"name": "SecondaryStatus", "shape": "SecondaryStatus", "type": "string"},
            {"name": "WarmPoolStatus", "shape": "WarmPoolStatus", "type": "structure"},
            {"name": "TrainingPlanArn", "shape": "TrainingPlanArn", "type": "string"},
        ],
        "type": "structure",
    },
    "TrainingPlanArns": {
        "member_shape": "TrainingPlanArn",
        "member_type": "string",
        "type": "list",
    },
    "TrainingPlanFilter": {
        "members": [
            {"name": "Name", "shape": "TrainingPlanFilterName", "type": "string"},
            {"name": "Value", "shape": "String64", "type": "string"},
        ],
        "type": "structure",
    },
    "TrainingPlanFilters": {
        "member_shape": "TrainingPlanFilter",
        "member_type": "structure",
        "type": "list",
    },
    "TrainingPlanOffering": {
        "members": [
            {"name": "TrainingPlanOfferingId", "shape": "TrainingPlanOfferingId", "type": "string"},
            {"name": "TargetResources", "shape": "SageMakerResourceNames", "type": "list"},
            {"name": "RequestedStartTimeAfter", "shape": "Timestamp", "type": "timestamp"},
            {"name": "RequestedEndTimeBefore", "shape": "Timestamp", "type": "timestamp"},
            {"name": "DurationHours", "shape": "TrainingPlanDurationHours", "type": "long"},
            {"name": "DurationMinutes", "shape": "TrainingPlanDurationMinutes", "type": "long"},
            {"name": "UpfrontFee", "shape": "String256", "type": "string"},
            {"name": "CurrencyCode", "shape": "CurrencyCode", "type": "string"},
            {
                "name": "ReservedCapacityOfferings",
                "shape": "ReservedCapacityOfferings",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "TrainingPlanOfferings": {
        "member_shape": "TrainingPlanOffering",
        "member_type": "structure",
        "type": "list",
    },
    "TrainingPlanSummaries": {
        "member_shape": "TrainingPlanSummary",
        "member_type": "structure",
        "type": "list",
    },
    "TrainingPlanSummary": {
        "members": [
            {"name": "TrainingPlanArn", "shape": "TrainingPlanArn", "type": "string"},
            {"name": "TrainingPlanName", "shape": "TrainingPlanName", "type": "string"},
            {"name": "Status", "shape": "TrainingPlanStatus", "type": "string"},
            {"name": "StatusMessage", "shape": "TrainingPlanStatusMessage", "type": "string"},
            {"name": "DurationHours", "shape": "TrainingPlanDurationHours", "type": "long"},
            {"name": "DurationMinutes", "shape": "TrainingPlanDurationMinutes", "type": "long"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "UpfrontFee", "shape": "String256", "type": "string"},
            {"name": "CurrencyCode", "shape": "CurrencyCode", "type": "string"},
            {"name": "TotalInstanceCount", "shape": "TotalInstanceCount", "type": "integer"},
            {
                "name": "AvailableInstanceCount",
                "shape": "AvailableInstanceCount",
                "type": "integer",
            },
            {"name": "InUseInstanceCount", "shape": "InUseInstanceCount", "type": "integer"},
            {"name": "TargetResources", "shape": "SageMakerResourceNames", "type": "list"},
            {
                "name": "ReservedCapacitySummaries",
                "shape": "ReservedCapacitySummaries",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "TrainingRepositoryAuthConfig": {
        "members": [
            {
                "name": "TrainingRepositoryCredentialsProviderArn",
                "shape": "TrainingRepositoryCredentialsProviderArn",
                "type": "string",
            }
        ],
        "type": "structure",
    },
    "TrainingSpecification": {
        "members": [
            {"name": "TrainingImage", "shape": "ContainerImage", "type": "string"},
            {"name": "TrainingImageDigest", "shape": "ImageDigest", "type": "string"},
            {
                "name": "SupportedHyperParameters",
                "shape": "HyperParameterSpecifications",
                "type": "list",
            },
            {
                "name": "SupportedTrainingInstanceTypes",
                "shape": "TrainingInstanceTypes",
                "type": "list",
            },
            {"name": "SupportsDistributedTraining", "shape": "Boolean", "type": "boolean"},
            {"name": "MetricDefinitions", "shape": "MetricDefinitionList", "type": "list"},
            {"name": "TrainingChannels", "shape": "ChannelSpecifications", "type": "list"},
            {
                "name": "SupportedTuningJobObjectiveMetrics",
                "shape": "HyperParameterTuningJobObjectives",
                "type": "list",
            },
            {
                "name": "AdditionalS3DataSource",
                "shape": "AdditionalS3DataSource",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "TransformDataSource": {
        "members": [
            {"name": "S3DataSource", "shape": "TransformS3DataSource", "type": "structure"}
        ],
        "type": "structure",
    },
    "TransformEnvironmentMap": {
        "key_shape": "TransformEnvironmentKey",
        "key_type": "string",
        "type": "map",
        "value_shape": "TransformEnvironmentValue",
        "value_type": "string",
    },
    "TransformInput": {
        "members": [
            {"name": "DataSource", "shape": "TransformDataSource", "type": "structure"},
            {"name": "ContentType", "shape": "ContentType", "type": "string"},
            {"name": "CompressionType", "shape": "CompressionType", "type": "string"},
            {"name": "SplitType", "shape": "SplitType", "type": "string"},
        ],
        "type": "structure",
    },
    "TransformInstanceTypes": {
        "member_shape": "TransformInstanceType",
        "member_type": "string",
        "type": "list",
    },
    "TransformJob": {
        "members": [
            {"name": "TransformJobName", "shape": "TransformJobName", "type": "string"},
            {"name": "TransformJobArn", "shape": "TransformJobArn", "type": "string"},
            {"name": "TransformJobStatus", "shape": "TransformJobStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
            {"name": "ModelName", "shape": "ModelName", "type": "string"},
            {
                "name": "MaxConcurrentTransforms",
                "shape": "MaxConcurrentTransforms",
                "type": "integer",
            },
            {"name": "ModelClientConfig", "shape": "ModelClientConfig", "type": "structure"},
            {"name": "MaxPayloadInMB", "shape": "MaxPayloadInMB", "type": "integer"},
            {"name": "BatchStrategy", "shape": "BatchStrategy", "type": "string"},
            {"name": "Environment", "shape": "TransformEnvironmentMap", "type": "map"},
            {"name": "TransformInput", "shape": "TransformInput", "type": "structure"},
            {"name": "TransformOutput", "shape": "TransformOutput", "type": "structure"},
            {"name": "DataCaptureConfig", "shape": "BatchDataCaptureConfig", "type": "structure"},
            {"name": "TransformResources", "shape": "TransformResources", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TransformStartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TransformEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LabelingJobArn", "shape": "LabelingJobArn", "type": "string"},
            {"name": "AutoMLJobArn", "shape": "AutoMLJobArn", "type": "string"},
            {"name": "DataProcessing", "shape": "DataProcessing", "type": "structure"},
            {"name": "ExperimentConfig", "shape": "ExperimentConfig", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "TransformJobDefinition": {
        "members": [
            {
                "name": "MaxConcurrentTransforms",
                "shape": "MaxConcurrentTransforms",
                "type": "integer",
            },
            {"name": "MaxPayloadInMB", "shape": "MaxPayloadInMB", "type": "integer"},
            {"name": "BatchStrategy", "shape": "BatchStrategy", "type": "string"},
            {"name": "Environment", "shape": "TransformEnvironmentMap", "type": "map"},
            {"name": "TransformInput", "shape": "TransformInput", "type": "structure"},
            {"name": "TransformOutput", "shape": "TransformOutput", "type": "structure"},
            {"name": "TransformResources", "shape": "TransformResources", "type": "structure"},
        ],
        "type": "structure",
    },
    "TransformJobStepMetadata": {
        "members": [{"name": "Arn", "shape": "TransformJobArn", "type": "string"}],
        "type": "structure",
    },
    "TransformJobSummaries": {
        "member_shape": "TransformJobSummary",
        "member_type": "structure",
        "type": "list",
    },
    "TransformJobSummary": {
        "members": [
            {"name": "TransformJobName", "shape": "TransformJobName", "type": "string"},
            {"name": "TransformJobArn", "shape": "TransformJobArn", "type": "string"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TransformEndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "TransformJobStatus", "shape": "TransformJobStatus", "type": "string"},
            {"name": "FailureReason", "shape": "FailureReason", "type": "string"},
        ],
        "type": "structure",
    },
    "TransformOutput": {
        "members": [
            {"name": "S3OutputPath", "shape": "S3Uri", "type": "string"},
            {"name": "Accept", "shape": "Accept", "type": "string"},
            {"name": "AssembleWith", "shape": "AssemblyType", "type": "string"},
            {"name": "KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "TransformResources": {
        "members": [
            {"name": "InstanceType", "shape": "TransformInstanceType", "type": "string"},
            {"name": "InstanceCount", "shape": "TransformInstanceCount", "type": "integer"},
            {"name": "VolumeKmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "TransformS3DataSource": {
        "members": [
            {"name": "S3DataType", "shape": "S3DataType", "type": "string"},
            {"name": "S3Uri", "shape": "S3Uri", "type": "string"},
        ],
        "type": "structure",
    },
    "Trial": {
        "members": [
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialArn", "shape": "TrialArn", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Source", "shape": "TrialSource", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {
                "name": "TrialComponentSummaries",
                "shape": "TrialComponentSimpleSummaries",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "TrialComponent": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"},
            {"name": "Source", "shape": "TrialComponentSource", "type": "structure"},
            {"name": "Status", "shape": "TrialComponentStatus", "type": "structure"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
            {"name": "Parameters", "shape": "TrialComponentParameters", "type": "map"},
            {"name": "InputArtifacts", "shape": "TrialComponentArtifacts", "type": "map"},
            {"name": "OutputArtifacts", "shape": "TrialComponentArtifacts", "type": "map"},
            {"name": "Metrics", "shape": "TrialComponentMetricSummaries", "type": "list"},
            {"name": "MetadataProperties", "shape": "MetadataProperties", "type": "structure"},
            {"name": "SourceDetail", "shape": "TrialComponentSourceDetail", "type": "structure"},
            {"name": "LineageGroupArn", "shape": "LineageGroupArn", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
            {"name": "Parents", "shape": "Parents", "type": "list"},
            {"name": "RunName", "shape": "ExperimentEntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "TrialComponentArtifact": {
        "members": [
            {"name": "MediaType", "shape": "MediaType", "type": "string"},
            {"name": "Value", "shape": "TrialComponentArtifactValue", "type": "string"},
        ],
        "type": "structure",
    },
    "TrialComponentArtifacts": {
        "key_shape": "TrialComponentKey128",
        "key_type": "string",
        "type": "map",
        "value_shape": "TrialComponentArtifact",
        "value_type": "structure",
    },
    "TrialComponentMetricSummaries": {
        "member_shape": "TrialComponentMetricSummary",
        "member_type": "structure",
        "type": "list",
    },
    "TrialComponentMetricSummary": {
        "members": [
            {"name": "MetricName", "shape": "MetricName", "type": "string"},
            {"name": "SourceArn", "shape": "TrialComponentSourceArn", "type": "string"},
            {"name": "TimeStamp", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Max", "shape": "OptionalDouble", "type": "double"},
            {"name": "Min", "shape": "OptionalDouble", "type": "double"},
            {"name": "Last", "shape": "OptionalDouble", "type": "double"},
            {"name": "Count", "shape": "OptionalInteger", "type": "integer"},
            {"name": "Avg", "shape": "OptionalDouble", "type": "double"},
            {"name": "StdDev", "shape": "OptionalDouble", "type": "double"},
        ],
        "type": "structure",
    },
    "TrialComponentParameterValue": {
        "members": [
            {"name": "StringValue", "shape": "StringParameterValue", "type": "string"},
            {"name": "NumberValue", "shape": "DoubleParameterValue", "type": "double"},
        ],
        "type": "structure",
    },
    "TrialComponentParameters": {
        "key_shape": "TrialComponentKey320",
        "key_type": "string",
        "type": "map",
        "value_shape": "TrialComponentParameterValue",
        "value_type": "structure",
    },
    "TrialComponentSimpleSummaries": {
        "member_shape": "TrialComponentSimpleSummary",
        "member_type": "structure",
        "type": "list",
    },
    "TrialComponentSimpleSummary": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"},
            {"name": "TrialComponentSource", "shape": "TrialComponentSource", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "TrialComponentSource": {
        "members": [
            {"name": "SourceArn", "shape": "TrialComponentSourceArn", "type": "string"},
            {"name": "SourceType", "shape": "SourceType", "type": "string"},
        ],
        "type": "structure",
    },
    "TrialComponentSourceDetail": {
        "members": [
            {"name": "SourceArn", "shape": "TrialComponentSourceArn", "type": "string"},
            {"name": "TrainingJob", "shape": "TrainingJob", "type": "structure"},
            {"name": "ProcessingJob", "shape": "ProcessingJob", "type": "structure"},
            {"name": "TransformJob", "shape": "TransformJob", "type": "structure"},
        ],
        "type": "structure",
    },
    "TrialComponentSources": {
        "member_shape": "TrialComponentSource",
        "member_type": "structure",
        "type": "list",
    },
    "TrialComponentStatus": {
        "members": [
            {"name": "PrimaryStatus", "shape": "TrialComponentPrimaryStatus", "type": "string"},
            {"name": "Message", "shape": "TrialComponentStatusMessage", "type": "string"},
        ],
        "type": "structure",
    },
    "TrialComponentSummaries": {
        "member_shape": "TrialComponentSummary",
        "member_type": "structure",
        "type": "list",
    },
    "TrialComponentSummary": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialComponentSource", "shape": "TrialComponentSource", "type": "structure"},
            {"name": "Status", "shape": "TrialComponentStatus", "type": "structure"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "CreatedBy", "shape": "UserContext", "type": "structure"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedBy", "shape": "UserContext", "type": "structure"},
        ],
        "type": "structure",
    },
    "TrialSource": {
        "members": [
            {"name": "SourceArn", "shape": "TrialSourceArn", "type": "string"},
            {"name": "SourceType", "shape": "SourceType", "type": "string"},
        ],
        "type": "structure",
    },
    "TrialSummaries": {"member_shape": "TrialSummary", "member_type": "structure", "type": "list"},
    "TrialSummary": {
        "members": [
            {"name": "TrialArn", "shape": "TrialArn", "type": "string"},
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "TrialSource", "shape": "TrialSource", "type": "structure"},
            {"name": "CreationTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "Timestamp", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "TtlDuration": {
        "members": [
            {"name": "Unit", "shape": "TtlDurationUnit", "type": "string"},
            {"name": "Value", "shape": "TtlDurationValue", "type": "integer"},
        ],
        "type": "structure",
    },
    "TuningJobCompletionCriteria": {
        "members": [
            {
                "name": "TargetObjectiveMetricValue",
                "shape": "TargetObjectiveMetricValue",
                "type": "float",
            },
            {
                "name": "BestObjectiveNotImproving",
                "shape": "BestObjectiveNotImproving",
                "type": "structure",
            },
            {"name": "ConvergenceDetected", "shape": "ConvergenceDetected", "type": "structure"},
        ],
        "type": "structure",
    },
    "TuningJobStepMetaData": {
        "members": [{"name": "Arn", "shape": "HyperParameterTuningJobArn", "type": "string"}],
        "type": "structure",
    },
    "USD": {
        "members": [
            {"name": "Dollars", "shape": "Dollars", "type": "integer"},
            {"name": "Cents", "shape": "Cents", "type": "integer"},
            {"name": "TenthFractionsOfACent", "shape": "TenthFractionsOfACent", "type": "integer"},
        ],
        "type": "structure",
    },
    "UiConfig": {
        "members": [
            {"name": "UiTemplateS3Uri", "shape": "S3Uri", "type": "string"},
            {"name": "HumanTaskUiArn", "shape": "HumanTaskUiArn", "type": "string"},
        ],
        "type": "structure",
    },
    "UiTemplate": {
        "members": [{"name": "Content", "shape": "TemplateContent", "type": "string"}],
        "type": "structure",
    },
    "UiTemplateInfo": {
        "members": [
            {"name": "Url", "shape": "TemplateUrl", "type": "string"},
            {"name": "ContentSha256", "shape": "TemplateContentSha256", "type": "string"},
        ],
        "type": "structure",
    },
    "UnprocessedIdentifiers": {
        "member_shape": "BatchGetRecordIdentifier",
        "member_type": "structure",
        "type": "list",
    },
    "UpdateActionRequest": {
        "members": [
            {"name": "ActionName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "Status", "shape": "ActionStatus", "type": "string"},
            {"name": "Properties", "shape": "LineageEntityParameters", "type": "map"},
            {
                "name": "PropertiesToRemove",
                "shape": "ListLineageEntityParameterKey",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "UpdateActionResponse": {
        "members": [{"name": "ActionArn", "shape": "ActionArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateAppImageConfigRequest": {
        "members": [
            {"name": "AppImageConfigName", "shape": "AppImageConfigName", "type": "string"},
            {
                "name": "KernelGatewayImageConfig",
                "shape": "KernelGatewayImageConfig",
                "type": "structure",
            },
            {
                "name": "JupyterLabAppImageConfig",
                "shape": "JupyterLabAppImageConfig",
                "type": "structure",
            },
            {
                "name": "CodeEditorAppImageConfig",
                "shape": "CodeEditorAppImageConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdateAppImageConfigResponse": {
        "members": [{"name": "AppImageConfigArn", "shape": "AppImageConfigArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateArtifactRequest": {
        "members": [
            {"name": "ArtifactArn", "shape": "ArtifactArn", "type": "string"},
            {"name": "ArtifactName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Properties", "shape": "ArtifactProperties", "type": "map"},
            {
                "name": "PropertiesToRemove",
                "shape": "ListLineageEntityParameterKey",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "UpdateArtifactResponse": {
        "members": [{"name": "ArtifactArn", "shape": "ArtifactArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateClusterRequest": {
        "members": [
            {"name": "ClusterName", "shape": "ClusterNameOrArn", "type": "string"},
            {
                "name": "InstanceGroups",
                "shape": "ClusterInstanceGroupSpecifications",
                "type": "list",
            },
            {"name": "NodeRecovery", "shape": "ClusterNodeRecovery", "type": "string"},
            {
                "name": "InstanceGroupsToDelete",
                "shape": "ClusterInstanceGroupsToDelete",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "UpdateClusterResponse": {
        "members": [{"name": "ClusterArn", "shape": "ClusterArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateClusterSchedulerConfigRequest": {
        "members": [
            {
                "name": "ClusterSchedulerConfigId",
                "shape": "ClusterSchedulerConfigId",
                "type": "string",
            },
            {"name": "TargetVersion", "shape": "Integer", "type": "integer"},
            {"name": "SchedulerConfig", "shape": "SchedulerConfig", "type": "structure"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateClusterSchedulerConfigResponse": {
        "members": [
            {
                "name": "ClusterSchedulerConfigArn",
                "shape": "ClusterSchedulerConfigArn",
                "type": "string",
            },
            {"name": "ClusterSchedulerConfigVersion", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "UpdateClusterSoftwareRequest": {
        "members": [{"name": "ClusterName", "shape": "ClusterNameOrArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateClusterSoftwareResponse": {
        "members": [{"name": "ClusterArn", "shape": "ClusterArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateCodeRepositoryInput": {
        "members": [
            {"name": "CodeRepositoryName", "shape": "EntityName", "type": "string"},
            {"name": "GitConfig", "shape": "GitConfigForUpdate", "type": "structure"},
        ],
        "type": "structure",
    },
    "UpdateCodeRepositoryOutput": {
        "members": [{"name": "CodeRepositoryArn", "shape": "CodeRepositoryArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateComputeQuotaRequest": {
        "members": [
            {"name": "ComputeQuotaId", "shape": "ComputeQuotaId", "type": "string"},
            {"name": "TargetVersion", "shape": "Integer", "type": "integer"},
            {"name": "ComputeQuotaConfig", "shape": "ComputeQuotaConfig", "type": "structure"},
            {"name": "ComputeQuotaTarget", "shape": "ComputeQuotaTarget", "type": "structure"},
            {"name": "ActivationState", "shape": "ActivationState", "type": "string"},
            {"name": "Description", "shape": "EntityDescription", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateComputeQuotaResponse": {
        "members": [
            {"name": "ComputeQuotaArn", "shape": "ComputeQuotaArn", "type": "string"},
            {"name": "ComputeQuotaVersion", "shape": "Integer", "type": "integer"},
        ],
        "type": "structure",
    },
    "UpdateContextRequest": {
        "members": [
            {"name": "ContextName", "shape": "ContextName", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
            {"name": "Properties", "shape": "LineageEntityParameters", "type": "map"},
            {
                "name": "PropertiesToRemove",
                "shape": "ListLineageEntityParameterKey",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "UpdateContextResponse": {
        "members": [{"name": "ContextArn", "shape": "ContextArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateDeviceFleetRequest": {
        "members": [
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {"name": "Description", "shape": "DeviceFleetDescription", "type": "string"},
            {"name": "OutputConfig", "shape": "EdgeOutputConfig", "type": "structure"},
            {"name": "EnableIotRoleAlias", "shape": "EnableIotRoleAlias", "type": "boolean"},
        ],
        "type": "structure",
    },
    "UpdateDevicesRequest": {
        "members": [
            {"name": "DeviceFleetName", "shape": "EntityName", "type": "string"},
            {"name": "Devices", "shape": "Devices", "type": "list"},
        ],
        "type": "structure",
    },
    "UpdateDomainRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "DefaultUserSettings", "shape": "UserSettings", "type": "structure"},
            {
                "name": "DomainSettingsForUpdate",
                "shape": "DomainSettingsForUpdate",
                "type": "structure",
            },
            {
                "name": "AppSecurityGroupManagement",
                "shape": "AppSecurityGroupManagement",
                "type": "string",
            },
            {"name": "DefaultSpaceSettings", "shape": "DefaultSpaceSettings", "type": "structure"},
            {"name": "SubnetIds", "shape": "Subnets", "type": "list"},
            {"name": "AppNetworkAccessType", "shape": "AppNetworkAccessType", "type": "string"},
            {"name": "TagPropagation", "shape": "TagPropagation", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateDomainResponse": {
        "members": [{"name": "DomainArn", "shape": "DomainArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateEndpointInput": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {"name": "EndpointConfigName", "shape": "EndpointConfigName", "type": "string"},
            {"name": "RetainAllVariantProperties", "shape": "Boolean", "type": "boolean"},
            {
                "name": "ExcludeRetainedVariantProperties",
                "shape": "VariantPropertyList",
                "type": "list",
            },
            {"name": "DeploymentConfig", "shape": "DeploymentConfig", "type": "structure"},
            {"name": "RetainDeploymentConfig", "shape": "Boolean", "type": "boolean"},
        ],
        "type": "structure",
    },
    "UpdateEndpointOutput": {
        "members": [{"name": "EndpointArn", "shape": "EndpointArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateEndpointWeightsAndCapacitiesInput": {
        "members": [
            {"name": "EndpointName", "shape": "EndpointName", "type": "string"},
            {
                "name": "DesiredWeightsAndCapacities",
                "shape": "DesiredWeightAndCapacityList",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "UpdateEndpointWeightsAndCapacitiesOutput": {
        "members": [{"name": "EndpointArn", "shape": "EndpointArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateExperimentRequest": {
        "members": [
            {"name": "ExperimentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Description", "shape": "ExperimentDescription", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateExperimentResponse": {
        "members": [{"name": "ExperimentArn", "shape": "ExperimentArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateFeatureGroupRequest": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupNameOrArn", "type": "string"},
            {"name": "FeatureAdditions", "shape": "FeatureAdditions", "type": "list"},
            {"name": "OnlineStoreConfig", "shape": "OnlineStoreConfigUpdate", "type": "structure"},
            {"name": "ThroughputConfig", "shape": "ThroughputConfigUpdate", "type": "structure"},
        ],
        "type": "structure",
    },
    "UpdateFeatureGroupResponse": {
        "members": [{"name": "FeatureGroupArn", "shape": "FeatureGroupArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateFeatureMetadataRequest": {
        "members": [
            {"name": "FeatureGroupName", "shape": "FeatureGroupNameOrArn", "type": "string"},
            {"name": "FeatureName", "shape": "FeatureName", "type": "string"},
            {"name": "Description", "shape": "FeatureDescription", "type": "string"},
            {"name": "ParameterAdditions", "shape": "FeatureParameterAdditions", "type": "list"},
            {"name": "ParameterRemovals", "shape": "FeatureParameterRemovals", "type": "list"},
        ],
        "type": "structure",
    },
    "UpdateHubContentReferenceRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "MinVersion", "shape": "HubContentVersion", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateHubContentReferenceResponse": {
        "members": [
            {"name": "HubArn", "shape": "HubArn", "type": "string"},
            {"name": "HubContentArn", "shape": "HubContentArn", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateHubContentRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubContentName", "shape": "HubContentName", "type": "string"},
            {"name": "HubContentType", "shape": "HubContentType", "type": "string"},
            {"name": "HubContentVersion", "shape": "HubContentVersion", "type": "string"},
            {"name": "HubContentDisplayName", "shape": "HubContentDisplayName", "type": "string"},
            {"name": "HubContentDescription", "shape": "HubContentDescription", "type": "string"},
            {"name": "HubContentMarkdown", "shape": "HubContentMarkdown", "type": "string"},
            {
                "name": "HubContentSearchKeywords",
                "shape": "HubContentSearchKeywordList",
                "type": "list",
            },
            {"name": "SupportStatus", "shape": "HubContentSupportStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateHubContentResponse": {
        "members": [
            {"name": "HubArn", "shape": "HubArn", "type": "string"},
            {"name": "HubContentArn", "shape": "HubContentArn", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateHubRequest": {
        "members": [
            {"name": "HubName", "shape": "HubNameOrArn", "type": "string"},
            {"name": "HubDescription", "shape": "HubDescription", "type": "string"},
            {"name": "HubDisplayName", "shape": "HubDisplayName", "type": "string"},
            {"name": "HubSearchKeywords", "shape": "HubSearchKeywordList", "type": "list"},
        ],
        "type": "structure",
    },
    "UpdateHubResponse": {
        "members": [{"name": "HubArn", "shape": "HubArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateImageRequest": {
        "members": [
            {"name": "DeleteProperties", "shape": "ImageDeletePropertyList", "type": "list"},
            {"name": "Description", "shape": "ImageDescription", "type": "string"},
            {"name": "DisplayName", "shape": "ImageDisplayName", "type": "string"},
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateImageResponse": {
        "members": [{"name": "ImageArn", "shape": "ImageArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateImageVersionRequest": {
        "members": [
            {"name": "ImageName", "shape": "ImageName", "type": "string"},
            {"name": "Alias", "shape": "SageMakerImageVersionAlias", "type": "string"},
            {"name": "Version", "shape": "ImageVersionNumber", "type": "integer"},
            {"name": "AliasesToAdd", "shape": "SageMakerImageVersionAliases", "type": "list"},
            {"name": "AliasesToDelete", "shape": "SageMakerImageVersionAliases", "type": "list"},
            {"name": "VendorGuidance", "shape": "VendorGuidance", "type": "string"},
            {"name": "JobType", "shape": "JobType", "type": "string"},
            {"name": "MLFramework", "shape": "MLFramework", "type": "string"},
            {"name": "ProgrammingLang", "shape": "ProgrammingLang", "type": "string"},
            {"name": "Processor", "shape": "Processor", "type": "string"},
            {"name": "Horovod", "shape": "Horovod", "type": "boolean"},
            {"name": "ReleaseNotes", "shape": "ReleaseNotes", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateImageVersionResponse": {
        "members": [{"name": "ImageVersionArn", "shape": "ImageVersionArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateInferenceComponentInput": {
        "members": [
            {"name": "InferenceComponentName", "shape": "InferenceComponentName", "type": "string"},
            {
                "name": "Specification",
                "shape": "InferenceComponentSpecification",
                "type": "structure",
            },
            {
                "name": "RuntimeConfig",
                "shape": "InferenceComponentRuntimeConfig",
                "type": "structure",
            },
            {
                "name": "DeploymentConfig",
                "shape": "InferenceComponentDeploymentConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdateInferenceComponentOutput": {
        "members": [
            {"name": "InferenceComponentArn", "shape": "InferenceComponentArn", "type": "string"}
        ],
        "type": "structure",
    },
    "UpdateInferenceComponentRuntimeConfigInput": {
        "members": [
            {"name": "InferenceComponentName", "shape": "InferenceComponentName", "type": "string"},
            {
                "name": "DesiredRuntimeConfig",
                "shape": "InferenceComponentRuntimeConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdateInferenceComponentRuntimeConfigOutput": {
        "members": [
            {"name": "InferenceComponentArn", "shape": "InferenceComponentArn", "type": "string"}
        ],
        "type": "structure",
    },
    "UpdateInferenceExperimentRequest": {
        "members": [
            {"name": "Name", "shape": "InferenceExperimentName", "type": "string"},
            {"name": "Schedule", "shape": "InferenceExperimentSchedule", "type": "structure"},
            {"name": "Description", "shape": "InferenceExperimentDescription", "type": "string"},
            {"name": "ModelVariants", "shape": "ModelVariantConfigList", "type": "list"},
            {
                "name": "DataStorageConfig",
                "shape": "InferenceExperimentDataStorageConfig",
                "type": "structure",
            },
            {"name": "ShadowModeConfig", "shape": "ShadowModeConfig", "type": "structure"},
        ],
        "type": "structure",
    },
    "UpdateInferenceExperimentResponse": {
        "members": [
            {"name": "InferenceExperimentArn", "shape": "InferenceExperimentArn", "type": "string"}
        ],
        "type": "structure",
    },
    "UpdateMlflowTrackingServerRequest": {
        "members": [
            {"name": "TrackingServerName", "shape": "TrackingServerName", "type": "string"},
            {"name": "ArtifactStoreUri", "shape": "S3Uri", "type": "string"},
            {"name": "TrackingServerSize", "shape": "TrackingServerSize", "type": "string"},
            {"name": "AutomaticModelRegistration", "shape": "Boolean", "type": "boolean"},
            {
                "name": "WeeklyMaintenanceWindowStart",
                "shape": "WeeklyMaintenanceWindowStart",
                "type": "string",
            },
        ],
        "type": "structure",
    },
    "UpdateMlflowTrackingServerResponse": {
        "members": [{"name": "TrackingServerArn", "shape": "TrackingServerArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateModelCardRequest": {
        "members": [
            {"name": "ModelCardName", "shape": "ModelCardNameOrArn", "type": "string"},
            {"name": "Content", "shape": "ModelCardContent", "type": "string"},
            {"name": "ModelCardStatus", "shape": "ModelCardStatus", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateModelCardResponse": {
        "members": [{"name": "ModelCardArn", "shape": "ModelCardArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateModelPackageInput": {
        "members": [
            {"name": "ModelPackageArn", "shape": "ModelPackageArn", "type": "string"},
            {"name": "ModelApprovalStatus", "shape": "ModelApprovalStatus", "type": "string"},
            {"name": "ApprovalDescription", "shape": "ApprovalDescription", "type": "string"},
            {"name": "CustomerMetadataProperties", "shape": "CustomerMetadataMap", "type": "map"},
            {
                "name": "CustomerMetadataPropertiesToRemove",
                "shape": "CustomerMetadataKeyList",
                "type": "list",
            },
            {
                "name": "AdditionalInferenceSpecificationsToAdd",
                "shape": "AdditionalInferenceSpecifications",
                "type": "list",
            },
            {
                "name": "InferenceSpecification",
                "shape": "InferenceSpecification",
                "type": "structure",
            },
            {"name": "SourceUri", "shape": "ModelPackageSourceUri", "type": "string"},
            {"name": "ModelCard", "shape": "ModelPackageModelCard", "type": "structure"},
            {"name": "ModelLifeCycle", "shape": "ModelLifeCycle", "type": "structure"},
            {"name": "ClientToken", "shape": "ClientToken", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateModelPackageOutput": {
        "members": [{"name": "ModelPackageArn", "shape": "ModelPackageArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateMonitoringAlertRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {"name": "MonitoringAlertName", "shape": "MonitoringAlertName", "type": "string"},
            {
                "name": "DatapointsToAlert",
                "shape": "MonitoringDatapointsToAlert",
                "type": "integer",
            },
            {"name": "EvaluationPeriod", "shape": "MonitoringEvaluationPeriod", "type": "integer"},
        ],
        "type": "structure",
    },
    "UpdateMonitoringAlertResponse": {
        "members": [
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"},
            {"name": "MonitoringAlertName", "shape": "MonitoringAlertName", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateMonitoringScheduleRequest": {
        "members": [
            {"name": "MonitoringScheduleName", "shape": "MonitoringScheduleName", "type": "string"},
            {
                "name": "MonitoringScheduleConfig",
                "shape": "MonitoringScheduleConfig",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdateMonitoringScheduleResponse": {
        "members": [
            {"name": "MonitoringScheduleArn", "shape": "MonitoringScheduleArn", "type": "string"}
        ],
        "type": "structure",
    },
    "UpdateNotebookInstanceInput": {
        "members": [
            {"name": "NotebookInstanceName", "shape": "NotebookInstanceName", "type": "string"},
            {"name": "InstanceType", "shape": "InstanceType", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "LifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {
                "name": "DisassociateLifecycleConfig",
                "shape": "DisassociateNotebookInstanceLifecycleConfig",
                "type": "boolean",
            },
            {
                "name": "VolumeSizeInGB",
                "shape": "NotebookInstanceVolumeSizeInGB",
                "type": "integer",
            },
            {"name": "DefaultCodeRepository", "shape": "CodeRepositoryNameOrUrl", "type": "string"},
            {
                "name": "AdditionalCodeRepositories",
                "shape": "AdditionalCodeRepositoryNamesOrUrls",
                "type": "list",
            },
            {
                "name": "AcceleratorTypes",
                "shape": "NotebookInstanceAcceleratorTypes",
                "type": "list",
            },
            {
                "name": "DisassociateAcceleratorTypes",
                "shape": "DisassociateNotebookInstanceAcceleratorTypes",
                "type": "boolean",
            },
            {
                "name": "DisassociateDefaultCodeRepository",
                "shape": "DisassociateDefaultCodeRepository",
                "type": "boolean",
            },
            {
                "name": "DisassociateAdditionalCodeRepositories",
                "shape": "DisassociateAdditionalCodeRepositories",
                "type": "boolean",
            },
            {"name": "RootAccess", "shape": "RootAccess", "type": "string"},
            {
                "name": "InstanceMetadataServiceConfiguration",
                "shape": "InstanceMetadataServiceConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdateNotebookInstanceLifecycleConfigInput": {
        "members": [
            {
                "name": "NotebookInstanceLifecycleConfigName",
                "shape": "NotebookInstanceLifecycleConfigName",
                "type": "string",
            },
            {"name": "OnCreate", "shape": "NotebookInstanceLifecycleConfigList", "type": "list"},
            {"name": "OnStart", "shape": "NotebookInstanceLifecycleConfigList", "type": "list"},
        ],
        "type": "structure",
    },
    "UpdateNotebookInstanceLifecycleConfigOutput": {"members": [], "type": "structure"},
    "UpdateNotebookInstanceOutput": {"members": [], "type": "structure"},
    "UpdatePartnerAppRequest": {
        "members": [
            {"name": "Arn", "shape": "PartnerAppArn", "type": "string"},
            {
                "name": "MaintenanceConfig",
                "shape": "PartnerAppMaintenanceConfig",
                "type": "structure",
            },
            {"name": "Tier", "shape": "NonEmptyString64", "type": "string"},
            {"name": "ApplicationConfig", "shape": "PartnerAppConfig", "type": "structure"},
            {"name": "EnableIamSessionBasedIdentity", "shape": "Boolean", "type": "boolean"},
            {"name": "ClientToken", "shape": "ClientToken", "type": "string"},
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "UpdatePartnerAppResponse": {
        "members": [{"name": "Arn", "shape": "PartnerAppArn", "type": "string"}],
        "type": "structure",
    },
    "UpdatePipelineExecutionRequest": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"},
            {
                "name": "PipelineExecutionDescription",
                "shape": "PipelineExecutionDescription",
                "type": "string",
            },
            {
                "name": "PipelineExecutionDisplayName",
                "shape": "PipelineExecutionName",
                "type": "string",
            },
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdatePipelineExecutionResponse": {
        "members": [
            {"name": "PipelineExecutionArn", "shape": "PipelineExecutionArn", "type": "string"}
        ],
        "type": "structure",
    },
    "UpdatePipelineRequest": {
        "members": [
            {"name": "PipelineName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDisplayName", "shape": "PipelineName", "type": "string"},
            {"name": "PipelineDefinition", "shape": "PipelineDefinition", "type": "string"},
            {
                "name": "PipelineDefinitionS3Location",
                "shape": "PipelineDefinitionS3Location",
                "type": "structure",
            },
            {"name": "PipelineDescription", "shape": "PipelineDescription", "type": "string"},
            {"name": "RoleArn", "shape": "RoleArn", "type": "string"},
            {
                "name": "ParallelismConfiguration",
                "shape": "ParallelismConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdatePipelineResponse": {
        "members": [{"name": "PipelineArn", "shape": "PipelineArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateProjectInput": {
        "members": [
            {"name": "ProjectName", "shape": "ProjectEntityName", "type": "string"},
            {"name": "ProjectDescription", "shape": "EntityDescription", "type": "string"},
            {
                "name": "ServiceCatalogProvisioningUpdateDetails",
                "shape": "ServiceCatalogProvisioningUpdateDetails",
                "type": "structure",
            },
            {"name": "Tags", "shape": "TagList", "type": "list"},
        ],
        "type": "structure",
    },
    "UpdateProjectOutput": {
        "members": [{"name": "ProjectArn", "shape": "ProjectArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateSpaceRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "SpaceName", "shape": "SpaceName", "type": "string"},
            {"name": "SpaceSettings", "shape": "SpaceSettings", "type": "structure"},
            {"name": "SpaceDisplayName", "shape": "NonEmptyString64", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateSpaceResponse": {
        "members": [{"name": "SpaceArn", "shape": "SpaceArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateTrainingJobRequest": {
        "members": [
            {"name": "TrainingJobName", "shape": "TrainingJobName", "type": "string"},
            {"name": "ProfilerConfig", "shape": "ProfilerConfigForUpdate", "type": "structure"},
            {
                "name": "ProfilerRuleConfigurations",
                "shape": "ProfilerRuleConfigurations",
                "type": "list",
            },
            {"name": "ResourceConfig", "shape": "ResourceConfigForUpdate", "type": "structure"},
            {
                "name": "RemoteDebugConfig",
                "shape": "RemoteDebugConfigForUpdate",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdateTrainingJobResponse": {
        "members": [{"name": "TrainingJobArn", "shape": "TrainingJobArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateTrialComponentRequest": {
        "members": [
            {"name": "TrialComponentName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "Status", "shape": "TrialComponentStatus", "type": "structure"},
            {"name": "StartTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "EndTime", "shape": "Timestamp", "type": "timestamp"},
            {"name": "Parameters", "shape": "TrialComponentParameters", "type": "map"},
            {"name": "ParametersToRemove", "shape": "ListTrialComponentKey256", "type": "list"},
            {"name": "InputArtifacts", "shape": "TrialComponentArtifacts", "type": "map"},
            {"name": "InputArtifactsToRemove", "shape": "ListTrialComponentKey256", "type": "list"},
            {"name": "OutputArtifacts", "shape": "TrialComponentArtifacts", "type": "map"},
            {
                "name": "OutputArtifactsToRemove",
                "shape": "ListTrialComponentKey256",
                "type": "list",
            },
        ],
        "type": "structure",
    },
    "UpdateTrialComponentResponse": {
        "members": [{"name": "TrialComponentArn", "shape": "TrialComponentArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateTrialRequest": {
        "members": [
            {"name": "TrialName", "shape": "ExperimentEntityName", "type": "string"},
            {"name": "DisplayName", "shape": "ExperimentEntityName", "type": "string"},
        ],
        "type": "structure",
    },
    "UpdateTrialResponse": {
        "members": [{"name": "TrialArn", "shape": "TrialArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateUserProfileRequest": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {"name": "UserSettings", "shape": "UserSettings", "type": "structure"},
        ],
        "type": "structure",
    },
    "UpdateUserProfileResponse": {
        "members": [{"name": "UserProfileArn", "shape": "UserProfileArn", "type": "string"}],
        "type": "structure",
    },
    "UpdateWorkforceRequest": {
        "members": [
            {"name": "WorkforceName", "shape": "WorkforceName", "type": "string"},
            {"name": "SourceIpConfig", "shape": "SourceIpConfig", "type": "structure"},
            {"name": "OidcConfig", "shape": "OidcConfig", "type": "structure"},
            {
                "name": "WorkforceVpcConfig",
                "shape": "WorkforceVpcConfigRequest",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdateWorkforceResponse": {
        "members": [{"name": "Workforce", "shape": "Workforce", "type": "structure"}],
        "type": "structure",
    },
    "UpdateWorkteamRequest": {
        "members": [
            {"name": "WorkteamName", "shape": "WorkteamName", "type": "string"},
            {"name": "MemberDefinitions", "shape": "MemberDefinitions", "type": "list"},
            {"name": "Description", "shape": "String200", "type": "string"},
            {
                "name": "NotificationConfiguration",
                "shape": "NotificationConfiguration",
                "type": "structure",
            },
            {
                "name": "WorkerAccessConfiguration",
                "shape": "WorkerAccessConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "UpdateWorkteamResponse": {
        "members": [{"name": "Workteam", "shape": "Workteam", "type": "structure"}],
        "type": "structure",
    },
    "UserContext": {
        "members": [
            {"name": "UserProfileArn", "shape": "String", "type": "string"},
            {"name": "UserProfileName", "shape": "String", "type": "string"},
            {"name": "DomainId", "shape": "String", "type": "string"},
            {"name": "IamIdentity", "shape": "IamIdentity", "type": "structure"},
        ],
        "type": "structure",
    },
    "UserProfileDetails": {
        "members": [
            {"name": "DomainId", "shape": "DomainId", "type": "string"},
            {"name": "UserProfileName", "shape": "UserProfileName", "type": "string"},
            {"name": "Status", "shape": "UserProfileStatus", "type": "string"},
            {"name": "CreationTime", "shape": "CreationTime", "type": "timestamp"},
            {"name": "LastModifiedTime", "shape": "LastModifiedTime", "type": "timestamp"},
        ],
        "type": "structure",
    },
    "UserProfileList": {
        "member_shape": "UserProfileDetails",
        "member_type": "structure",
        "type": "list",
    },
    "UserSettings": {
        "members": [
            {"name": "ExecutionRole", "shape": "RoleArn", "type": "string"},
            {"name": "SecurityGroups", "shape": "SecurityGroupIds", "type": "list"},
            {"name": "SharingSettings", "shape": "SharingSettings", "type": "structure"},
            {
                "name": "JupyterServerAppSettings",
                "shape": "JupyterServerAppSettings",
                "type": "structure",
            },
            {
                "name": "KernelGatewayAppSettings",
                "shape": "KernelGatewayAppSettings",
                "type": "structure",
            },
            {
                "name": "TensorBoardAppSettings",
                "shape": "TensorBoardAppSettings",
                "type": "structure",
            },
            {
                "name": "RStudioServerProAppSettings",
                "shape": "RStudioServerProAppSettings",
                "type": "structure",
            },
            {"name": "RSessionAppSettings", "shape": "RSessionAppSettings", "type": "structure"},
            {"name": "CanvasAppSettings", "shape": "CanvasAppSettings", "type": "structure"},
            {
                "name": "CodeEditorAppSettings",
                "shape": "CodeEditorAppSettings",
                "type": "structure",
            },
            {
                "name": "JupyterLabAppSettings",
                "shape": "JupyterLabAppSettings",
                "type": "structure",
            },
            {
                "name": "SpaceStorageSettings",
                "shape": "DefaultSpaceStorageSettings",
                "type": "structure",
            },
            {"name": "DefaultLandingUri", "shape": "LandingUri", "type": "string"},
            {"name": "StudioWebPortal", "shape": "StudioWebPortal", "type": "string"},
            {
                "name": "CustomPosixUserConfig",
                "shape": "CustomPosixUserConfig",
                "type": "structure",
            },
            {"name": "CustomFileSystemConfigs", "shape": "CustomFileSystemConfigs", "type": "list"},
            {
                "name": "StudioWebPortalSettings",
                "shape": "StudioWebPortalSettings",
                "type": "structure",
            },
            {"name": "AutoMountHomeEFS", "shape": "AutoMountHomeEFS", "type": "string"},
        ],
        "type": "structure",
    },
    "ValidationError": {
        "members": [{"name": "Message", "shape": "Message", "type": "string"}],
        "type": "structure",
    },
    "ValueAsStringList": {"member_shape": "ValueAsString", "member_type": "string", "type": "list"},
    "VariantProperty": {
        "members": [
            {"name": "VariantPropertyType", "shape": "VariantPropertyType", "type": "string"}
        ],
        "type": "structure",
    },
    "VariantPropertyList": {
        "member_shape": "VariantProperty",
        "member_type": "structure",
        "type": "list",
    },
    "VectorConfig": {
        "members": [{"name": "Dimension", "shape": "Dimension", "type": "integer"}],
        "type": "structure",
    },
    "VersionAliasesList": {
        "member_shape": "ImageVersionAliasPattern",
        "member_type": "string",
        "type": "list",
    },
    "Vertex": {
        "members": [
            {"name": "Arn", "shape": "AssociationEntityArn", "type": "string"},
            {"name": "Type", "shape": "String40", "type": "string"},
            {"name": "LineageType", "shape": "LineageType", "type": "string"},
        ],
        "type": "structure",
    },
    "Vertices": {"member_shape": "Vertex", "member_type": "structure", "type": "list"},
    "VisibilityConditions": {
        "members": [
            {"name": "Key", "shape": "VisibilityConditionsKey", "type": "string"},
            {"name": "Value", "shape": "VisibilityConditionsValue", "type": "string"},
        ],
        "type": "structure",
    },
    "VisibilityConditionsList": {
        "member_shape": "VisibilityConditions",
        "member_type": "structure",
        "type": "list",
    },
    "VpcConfig": {
        "members": [
            {"name": "SecurityGroupIds", "shape": "VpcSecurityGroupIds", "type": "list"},
            {"name": "Subnets", "shape": "Subnets", "type": "list"},
        ],
        "type": "structure",
    },
    "VpcOnlyTrustedAccounts": {
        "member_shape": "AccountId",
        "member_type": "string",
        "type": "list",
    },
    "VpcSecurityGroupIds": {
        "member_shape": "SecurityGroupId",
        "member_type": "string",
        "type": "list",
    },
    "WarmPoolStatus": {
        "members": [
            {"name": "Status", "shape": "WarmPoolResourceStatus", "type": "string"},
            {
                "name": "ResourceRetainedBillableTimeInSeconds",
                "shape": "ResourceRetainedBillableTimeInSeconds",
                "type": "integer",
            },
            {"name": "ReusedByJob", "shape": "TrainingJobName", "type": "string"},
        ],
        "type": "structure",
    },
    "WorkerAccessConfiguration": {
        "members": [{"name": "S3Presign", "shape": "S3Presign", "type": "structure"}],
        "type": "structure",
    },
    "Workforce": {
        "members": [
            {"name": "WorkforceName", "shape": "WorkforceName", "type": "string"},
            {"name": "WorkforceArn", "shape": "WorkforceArn", "type": "string"},
            {"name": "LastUpdatedDate", "shape": "Timestamp", "type": "timestamp"},
            {"name": "SourceIpConfig", "shape": "SourceIpConfig", "type": "structure"},
            {"name": "SubDomain", "shape": "String", "type": "string"},
            {"name": "CognitoConfig", "shape": "CognitoConfig", "type": "structure"},
            {"name": "OidcConfig", "shape": "OidcConfigForResponse", "type": "structure"},
            {"name": "CreateDate", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "WorkforceVpcConfig",
                "shape": "WorkforceVpcConfigResponse",
                "type": "structure",
            },
            {"name": "Status", "shape": "WorkforceStatus", "type": "string"},
            {"name": "FailureReason", "shape": "WorkforceFailureReason", "type": "string"},
        ],
        "type": "structure",
    },
    "WorkforceSecurityGroupIds": {
        "member_shape": "WorkforceSecurityGroupId",
        "member_type": "string",
        "type": "list",
    },
    "WorkforceSubnets": {
        "member_shape": "WorkforceSubnetId",
        "member_type": "string",
        "type": "list",
    },
    "WorkforceVpcConfigRequest": {
        "members": [
            {"name": "VpcId", "shape": "WorkforceVpcId", "type": "string"},
            {"name": "SecurityGroupIds", "shape": "WorkforceSecurityGroupIds", "type": "list"},
            {"name": "Subnets", "shape": "WorkforceSubnets", "type": "list"},
        ],
        "type": "structure",
    },
    "WorkforceVpcConfigResponse": {
        "members": [
            {"name": "VpcId", "shape": "WorkforceVpcId", "type": "string"},
            {"name": "SecurityGroupIds", "shape": "WorkforceSecurityGroupIds", "type": "list"},
            {"name": "Subnets", "shape": "WorkforceSubnets", "type": "list"},
            {"name": "VpcEndpointId", "shape": "WorkforceVpcEndpointId", "type": "string"},
        ],
        "type": "structure",
    },
    "Workforces": {"member_shape": "Workforce", "member_type": "structure", "type": "list"},
    "WorkspaceSettings": {
        "members": [
            {"name": "S3ArtifactPath", "shape": "S3Uri", "type": "string"},
            {"name": "S3KmsKeyId", "shape": "KmsKeyId", "type": "string"},
        ],
        "type": "structure",
    },
    "Workteam": {
        "members": [
            {"name": "WorkteamName", "shape": "WorkteamName", "type": "string"},
            {"name": "MemberDefinitions", "shape": "MemberDefinitions", "type": "list"},
            {"name": "WorkteamArn", "shape": "WorkteamArn", "type": "string"},
            {"name": "WorkforceArn", "shape": "WorkforceArn", "type": "string"},
            {"name": "ProductListingIds", "shape": "ProductListings", "type": "list"},
            {"name": "Description", "shape": "String200", "type": "string"},
            {"name": "SubDomain", "shape": "String", "type": "string"},
            {"name": "CreateDate", "shape": "Timestamp", "type": "timestamp"},
            {"name": "LastUpdatedDate", "shape": "Timestamp", "type": "timestamp"},
            {
                "name": "NotificationConfiguration",
                "shape": "NotificationConfiguration",
                "type": "structure",
            },
            {
                "name": "WorkerAccessConfiguration",
                "shape": "WorkerAccessConfiguration",
                "type": "structure",
            },
        ],
        "type": "structure",
    },
    "Workteams": {"member_shape": "Workteam", "member_type": "structure", "type": "list"},
    "XAxisValues": {"member_shape": "Long", "member_type": "long", "type": "list"},
}
