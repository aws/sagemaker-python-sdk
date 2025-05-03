import datetime
from dateutil.tz import tzlocal
from pprint import pprint
import unittest
from sagemaker.core.utils.code_injection.codec import pascal_to_snake
from sagemaker.core.utils.code_injection.codec import transform
from sagemaker.core.resources import Model, TrialComponent, AutoMLJobV2


class TestConversion(unittest.TestCase):
    def test_pascal_to_snake(self):
        self.assertEqual(pascal_to_snake("PascalCase"), "pascal_case")
        self.assertEqual(pascal_to_snake("AnotherExample"), "another_example")
        self.assertEqual(pascal_to_snake("test"), "test")
        self.assertEqual(pascal_to_snake("AutoMLJob"), "auto_ml_job")


class DummyResourceClass:
    pass


def test_deserializer_for_structure_type():
    """Validate deserializer() - for structure type"""
    # The test validates the following relations
    # StructA → basic_type_member
    # StructA → StructB -> basic_type_member
    describe_model_response = {
        "CreationTime": datetime.datetime(2024, 3, 13, 15, 7, 44, 459000, tzinfo=tzlocal()),
        "DeploymentRecommendation": {
            "RealTimeInferenceRecommendations": [],
            "RecommendationStatus": "COMPLETED",
        },
        "EnableNetworkIsolation": False,
        "ExecutionRoleArn": "arn:aws:iam::616250812882:role/SageMakerRole",
        "ModelArn": "arn:aws:sagemaker:us-west-2:616250812882:model/lmi-model-falcon-7b-1710367662-a49c",
        "ModelName": "lmi-model-falcon-7b-1710367662-a49c",
        "PrimaryContainer": {
            "Environment": {},
            "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118",
            "Mode": "SingleModel",
            "ModelDataSource": {
                "S3DataSource": {
                    "CompressionType": "Gzip",
                    "S3DataType": "S3Object",
                    "S3Uri": "s3://sagemaker-us-west-2-616250812882/session-default-prefix/large-model-lmi/code/mymodel-7B.tar.gz",
                }
            },
            "ModelDataUrl": "s3://sagemaker-us-west-2-616250812882/session-default-prefix/large-model-lmi/code/mymodel-7B.tar.gz",
        },
    }
    transformed_data = transform(describe_model_response, "DescribeModelOutput")
    pprint(transformed_data)
    instance = Model(**transformed_data)
    assert instance.execution_role_arn == "arn:aws:iam::616250812882:role/SageMakerRole"
    assert not instance.enable_network_isolation
    assert instance.primary_container.model_data_source.s3_data_source.s3_data_type == "S3Object"


def test_deserializer_for_list_type():
    """Validate deserializer() - for list type"""
    # The test validates the following relations
    # StructA → StructB -> list(structure)
    # ToDo: Struct -> list(basic types)
    # StructA → StructB -> list(structure) -> map(string, string)
    describe_model_response = {
        "CreationTime": datetime.datetime(2024, 3, 13, 15, 7, 44, 459000, tzinfo=tzlocal()),
        "DeploymentRecommendation": {
            "RealTimeInferenceRecommendations": [
                {
                    "RecommendationId": "dummy-recomm-id-1",
                    "InstanceType": "mlt4",
                    "Environment": {"ENV_VAR_1": "ENV_VAR_1_VALUE"},
                },
                {
                    "RecommendationId": "dummy-recomm-id-2",
                    "InstanceType": "mlm4",
                    "Environment": {"ENV_VAR_2": "ENV_VAR_2_VALUE"},
                },
            ],
            "RecommendationStatus": "COMPLETED",
        },
        "ModelArn": "arn:aws:sagemaker:us-west-2:616250812882:model/lmi-model-falcon-7b-1710367662-a49c",
        "ModelName": "lmi-model-falcon-7b-1710367662-a49c",
    }
    transformed_data = transform(describe_model_response, "DescribeModelOutput")
    pprint(transformed_data)
    instance = Model(**transformed_data)
    real_time_inference_recommendations = (
        instance.deployment_recommendation.real_time_inference_recommendations
    )
    assert type(real_time_inference_recommendations) == list
    assert real_time_inference_recommendations[0].recommendation_id == "dummy-recomm-id-1"
    assert real_time_inference_recommendations[1].instance_type == "mlm4"
    assert real_time_inference_recommendations[1].environment == {"ENV_VAR_2": "ENV_VAR_2_VALUE"}


def test_deserializer_for_map_type():
    """Validate deserializer() - for map type"""
    # The test validates the following relations
    # StructA → map(string, structure)
    describe_trial_component_response = {
        "CreatedBy": {},
        "DisplayName": "huggingface-pytorch-training-2024-01-10-02-32-59-730-aws-training-job",
        "OutputArtifacts": {
            "SageMaker.DebugHookOutput": {
                "Value": "s3://sagemaker-us-west-2-616250812882/session-default-prefix/"
            },
            "SageMaker.ModelArtifact": {
                "Value": "s3://sagemaker-us-west-2-616250812882/session-default-prefix/huggingface-pytorch-training-2024-01-10-02-32-59-730/output/model.tar.gz"
            },
        },
        "Parameters": {
            "SageMaker.ImageUri": {
                "StringValue": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
            },
            "SageMaker.InstanceCount": {"NumberValue": 1.0},
            "SageMaker.InstanceType": {"StringValue": "ml.g5.4xlarge"},
        },
        "TrialComponentArn": "arn:aws:sagemaker:us-west-2:616250812882:experiment-trial-component/huggingface-pytorch-training-2024-01-10-02-32-59-730-aws-training-job",
        "TrialComponentName": "huggingface-pytorch-training-2024-01-10-02-32-59-730-aws-training-job",
    }
    transformed_data = transform(
        describe_trial_component_response, "DescribeTrialComponentResponse"
    )
    pprint(transformed_data)
    instance = TrialComponent(**transformed_data)
    parameters = instance.parameters
    assert type(parameters) == dict
    assert parameters["SageMaker.InstanceType"].string_value == "ml.g5.4xlarge"
    assert parameters["SageMaker.InstanceCount"].number_value == 1.0
    output_artifacts = instance.output_artifacts
    assert type(output_artifacts) == dict
    assert (
        output_artifacts["SageMaker.DebugHookOutput"].value
        == "s3://sagemaker-us-west-2-616250812882/session-default-prefix/"
    )

    # StructA -> map(string, list) -> list(structure) -> map(string, string)
    describe_auto_ml_job_v2_response = {
        "AutoMLJobArn": "arn:aws:sagemaker:us-west-2:616250812882:automl-job/python-sdk-integ-test-base-job",
        "AutoMLJobInputDataConfig": [
            {
                "ChannelType": "training",
                "ContentType": "text/csv;header=present",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://sagemaker-us-west-2-616250812882/sagemaker/beta-automl-xgboost/input/iris_training.csv",
                    }
                },
            }
        ],
        "AutoMLJobName": "python-sdk-integ-test-base-job",
        "AutoMLJobSecondaryStatus": "Completed",
        "AutoMLJobStatus": "Completed",
        "AutoMLProblemTypeConfig": {
            "TabularJobConfig": {
                "CompletionCriteria": {"MaxCandidates": 3},
                "GenerateCandidateDefinitionsOnly": False,
                "TargetAttributeName": "virginica",
            },
            "TimeSeriesForecastingJobConfig": {
                "Transformations": {"Filling": {"map1_key": {"map2_key": "map2_val"}}},
                "ForecastFrequency": "dummy",
                "ForecastHorizon": 20,
                "TimeSeriesConfig": {
                    "TargetAttributeName": "dummy",
                    "TimestampAttributeName": "dummy",
                    "ItemIdentifierAttributeName": "dummy",
                    "GroupingAttributeNames": ["dummy"],
                },
            },
        },
        "BestCandidate": {
            "CandidateName": "python-sdk-integ-test-base-jobTA-001-143b672d",
            "CandidateStatus": "Completed",
            "CandidateSteps": [
                {
                    "CandidateStepArn": "arn:aws:sagemaker:us-west-2:616250812882:processing-job/python-sdk-integ-test-base-job-db-1-0661642ca7be48d280cb7fe6197",
                    "CandidateStepName": "python-sdk-integ-test-base-job-db-1-0661642ca7be48d280cb7fe6197",
                    "CandidateStepType": "AWS::SageMaker::ProcessingJob",
                },
                {
                    "CandidateStepArn": "arn:aws:sagemaker:us-west-2:616250812882:training-job/python-sdk-integ-test-base-job-dpp1-1-e49c814570994bd98293d0087",
                    "CandidateStepName": "python-sdk-integ-test-base-job-dpp1-1-e49c814570994bd98293d0087",
                    "CandidateStepType": "AWS::SageMaker::TrainingJob",
                },
                {
                    "CandidateStepArn": "arn:aws:sagemaker:us-west-2:616250812882:transform-job/python-sdk-integ-test-base-job-dpp1-csv-1-73af2590ca7a4719988c3",
                    "CandidateStepName": "python-sdk-integ-test-base-job-dpp1-csv-1-73af2590ca7a4719988c3",
                    "CandidateStepType": "AWS::SageMaker::TransformJob",
                },
                {
                    "CandidateStepArn": "arn:aws:sagemaker:us-west-2:616250812882:training-job/python-sdk-integ-test-base-jobta-001-143b672d",
                    "CandidateStepName": "python-sdk-integ-test-base-jobTA-001-143b672d",
                    "CandidateStepType": "AWS::SageMaker::TrainingJob",
                },
            ],
            "CreationTime": datetime.datetime(2021, 10, 4, 11, 5, 38, tzinfo=tzlocal()),
            "InferenceContainerDefinitions": {
                "def1": [
                    {
                        "Image": "dummy-image-1",
                        "ModelDataUrl": "dummy-model-data-url-1",
                        "Environment": {"ENV_VAR_1": "ENV_VAR_1_VALUE"},
                    },
                    {
                        "Image": "dummy-image-2",
                        "ModelDataUrl": "dummy-model-data-url-2",
                        "Environment": {"ENV_VAR_2": "ENV_VAR_2_VALUE"},
                    },
                ]
            },
            "LastModifiedTime": datetime.datetime(2021, 10, 4, 11, 8, 9, 941000, tzinfo=tzlocal()),
            "ObjectiveStatus": "Succeeded",
        },
        "OutputDataConfig": {"S3OutputPath": "s3://sagemaker-us-west-2-616250812882/"},
        "RoleArn": "arn:aws:iam::616250812882:role/SageMakerRole",
        "CreationTime": datetime.datetime(2024, 3, 13, 15, 7, 44, 459000, tzinfo=tzlocal()),
        "LastModifiedTime": datetime.datetime(2021, 10, 4, 11, 8, 9, 941000, tzinfo=tzlocal()),
    }
    transformed_data = transform(describe_auto_ml_job_v2_response, "DescribeAutoMLJobV2Response")
    instance = AutoMLJobV2(**transformed_data)
    best_candidate = instance.best_candidate
    inference_container_definitions = best_candidate.inference_container_definitions
    assert type(inference_container_definitions) == dict
    assert best_candidate.candidate_name == "python-sdk-integ-test-base-jobTA-001-143b672d"
    inference_container_definitions_def1 = inference_container_definitions["def1"]
    assert type(inference_container_definitions_def1) == list
    assert inference_container_definitions_def1[0].image == "dummy-image-1"
    assert inference_container_definitions_def1[1].environment == {"ENV_VAR_2": "ENV_VAR_2_VALUE"}
    # StructA -> map(string, map)
    assert (
        instance.auto_ml_problem_type_config.time_series_forecasting_job_config.transformations.filling
        == {"map1_key": {"map2_key": "map2_val"}}
    )


if __name__ == "__main__":
    unittest.main()
