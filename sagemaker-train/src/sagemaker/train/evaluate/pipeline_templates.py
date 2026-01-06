"""Pipeline templates for SageMaker Model Evaluation.

This module contains Jinja2 template strings for generating SageMaker Pipeline
definitions for different evaluation types (benchmark, custom scorer, LLM-as-judge).
"""

from .constants import EvalType

DETERMINISTIC_TEMPLATE = """{
    "Version": "2020-12-01",
    "Metadata": {},
    "MlflowConfig": {
        "MlflowResourceArn": "{{ mlflow_resource_arn }}"{% if mlflow_experiment_name %},
        "MlflowExperimentName": "{{ mlflow_experiment_name }}"{% endif %}{% if mlflow_run_name %},
        "MlflowRunName": "{{ mlflow_run_name }}"{% endif %}
    },
    "Parameters": [],
    "Steps": [
        {
            "Name": "CreateEvaluationAction",
            "Type": "Lineage",
            "Arguments": {
                "Actions": [
                    {
                        "ActionName": {
                            "Get": "Execution.PipelineExecutionId"
                        },
                        "ActionType": "Evaluation",
                        "Source": {
                            "SourceUri": "{{ source_model_package_arn }}",
                            "SourceType": "ModelPackage"
                        },
                        "Properties": {
                            "PipelineExecutionArn": {
                                "Get": "Execution.PipelineExecutionArn"
                            },
                            "PipelineName": "{{ pipeline_name }}"
                        }
                    }
                ],
                "Contexts": [
                    {
                        "ContextName": {
                            "Get": "Execution.PipelineExecutionId"
                        },
                        "ContextType": "PipelineExecution",
                        "Source": {
                            "SourceUri": {
                                "Get": "Execution.PipelineExecutionArn"
                            }
                        }
                    }
                ],
                "Associations": [
                    {
                        "Source": {
                            "Name": {
                                "Get": "Execution.PipelineExecutionId"
                            },
                            "Type": "Action"
                        },
                        "Destination": {
                            "Name": {
                                "Get": "Execution.PipelineExecutionId"
                            },
                            "Type": "Context"
                        },
                        "AssociationType": "ContributedTo"
                    },
                    {
                        "Source": {
                            "Arn": "{{ dataset_artifact_arn }}"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    }
                ]
            }
        },{% if evaluate_base_model %}
        {
            "Name": "EvaluateBaseModel",
            "Type": "Training",
            "Arguments": {
                "RoleArn": "{{ role_arn }}",
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": "{{ model_package_group_arn }}",
                    "SourceModelPackageArn": "{{ source_model_package_arn }}"
                },
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "BenchmarkEvaluation"{% if evaluator_arn %},
                    "EvaluatorArn": "{{ evaluator_arn }}"{% endif %}
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "task": "{{ task }}",
                    "strategy": "{{ strategy }}"{% if metric is defined %},
                    "metric": "{{ metric }}"{% elif evaluation_metric is defined %},
                    "evaluation_metric": "{{ evaluation_metric }}"{% endif %}{% if max_new_tokens is defined %},
                    "max_new_tokens": "{{ max_new_tokens }}"{% endif %}{% if temperature is defined %},
                    "temperature": "{{ temperature }}"{% endif %}{% if top_k is defined %},
                    "top_k": "{{ top_k }}"{% endif %}{% if top_p is defined %},
                    "top_p": "{{ top_p }}"{% endif %}{% if max_model_len is defined %},
                    "max_model_len": "{{ max_model_len }}"{% endif %}{% if aggregation is defined %},
                    "aggregation": "{{ aggregation }}"{% endif %}{% if postprocessing is defined %},
                    "postprocessing": "{{ postprocessing }}"{% endif %}{% if preset_reward_function is defined %},
                    "preset_reward_function": "{{ preset_reward_function }}"{% endif %}{% if subtask %},
                    "subtask": "{{ subtask }}"{% endif %}
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
                "VpcConfig": {
                    "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                    "Subnets": {{ vpc_subnets | tojson }}
                }{% endif %}
            }
        },{% endif %}
        {
            "Name": "EvaluateCustomModel",
            "Type": "Training",
            "Arguments": {
                "RoleArn": "{{ role_arn }}",
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": "{{ model_package_group_arn }}",
                    "SourceModelPackageArn": "{{ source_model_package_arn }}"
                },
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "BenchmarkEvaluation"{% if evaluator_arn %},
                    "EvaluatorArn": "{{ evaluator_arn }}"{% endif %}
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "task": "{{ task }}",
                    "strategy": "{{ strategy }}"{% if metric is defined %},
                    "metric": "{{ metric }}"{% elif evaluation_metric is defined %},
                    "evaluation_metric": "{{ evaluation_metric }}"{% endif %}{% if max_new_tokens is defined %},
                    "max_new_tokens": "{{ max_new_tokens }}"{% endif %}{% if temperature is defined %},
                    "temperature": "{{ temperature }}"{% endif %}{% if top_k is defined %},
                    "top_k": "{{ top_k }}"{% endif %}{% if top_p is defined %},
                    "top_p": "{{ top_p }}"{% endif %}{% if max_model_len is defined %},
                    "max_model_len": "{{ max_model_len }}"{% endif %}{% if aggregation is defined %},
                    "aggregation": "{{ aggregation }}"{% endif %}{% if postprocessing is defined %},
                    "postprocessing": "{{ postprocessing }}"{% endif %}{% if preset_reward_function is defined %},
                    "preset_reward_function": "{{ preset_reward_function }}"{% endif %}{% if subtask %},
                    "subtask": "{{ subtask }}"{% endif %}
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
                "VpcConfig": {
                    "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                    "Subnets": {{ vpc_subnets | tojson }}
                }{% endif %}
            }
        },
        {
            "Name": "AssociateLineage",
            "Type": "Lineage",
            "DependsOn": [
                "CreateEvaluationAction"
            ],
            "Arguments": {
                "Artifacts": [{% if evaluate_base_model %}
                    {
                        "ArtifactName": {
                            "Std:Join": {
                                "On": "-",
                                "Values": [
                                    {
                                        "Get": "Execution.PipelineExecutionId"
                                    },
                                    "base-eval-report"
                                ]
                            }
                        },
                        "ArtifactType": "EvaluationReport",
                        "Source": {
                            "SourceUri": {
                                "Get": "Steps.EvaluateBaseModel.OutputDataConfig.S3OutputPath"
                            }
                        }
                    },{% endif %}
                    {
                        "ArtifactName": {
                            "Std:Join": {
                                "On": "-",
                                "Values": [
                                    {
                                        "Get": "Execution.PipelineExecutionId"
                                    },
                                    "custom-eval-report"
                                ]
                            }
                        },
                        "ArtifactType": "EvaluationReport",
                        "Source": {
                            "SourceUri": {
                                "Get": "Steps.EvaluateCustomModel.OutputDataConfig.S3OutputPath"
                            }
                        }
                    }
                ],
                "Associations": [{% if evaluate_base_model %}
                    {
                        "Source": {
                            "Name": {
                                "Std:Join": {
                                    "On": "-",
                                    "Values": [
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        },
                                        "base-eval-report"
                                    ]
                                }
                            },
                            "Type": "Artifact"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    },{% endif %}
                    {
                        "Source": {
                            "Name": {
                                "Std:Join": {
                                    "On": "-",
                                    "Values": [
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        },
                                        "custom-eval-report"
                                    ]
                                }
                            },
                            "Type": "Artifact"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    }
                ]
            }
        }
    ]
}"""

# LLM-as-a-Judge Template for Base Model Only - 2-Phase Evaluation with optional ModelPackageConfig
LLMAJ_TEMPLATE_BASE_MODEL_ONLY = """{
    "Version": "2020-12-01",
    "Metadata": {},
    "Parameters": [],
    "Steps": [
        {
            "Name": "EvaluateBaseInferenceModel",
            "Type": "Training",
            "Arguments": {
                "TrainingJobName": "BaseInference",
                "RoleArn": "{{ role_arn }}",
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "BenchmarkEvaluation"
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "name": "BaseInference",
                    "task": "inference_only"
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
            "VpcConfig": {
                "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                "Subnets": {{ vpc_subnets | tojson }}
            }{% endif %}
            }
        },
        {
            "Name": "EvaluateBaseModelMetrics",
            "Type": "Training",
            "DependsOn": [
                "EvaluateBaseInferenceModel"
            ],
            "Arguments": {
                "TrainingJobName": {
                    "Std:Join": {
                        "On": "-",
                        "Values": [
                            "base-llmaj-eval",
                            {
                                "Get": "Execution.PipelineExecutionId"
                            }
                        ]
                    }
                },
                "RoleArn": "{{ role_arn }}",
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "LLMAJEvaluation"
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "name": {
                        "Std:Join": {
                            "On": "-",
                            "Values": [
                                "base-llmaj-eval",
                                {
                                    "Get": "Execution.PipelineExecutionId"
                                }
                            ]
                        }
                    },
                    "judge_model_id": "{{ judge_model_id }}",
                    "inference_data_s3_path": {
                        "Std:Join": {
                            "On": "",
                            "Values": [
                                {
                                    "Get": "Steps.EvaluateBaseInferenceModel.OutputDataConfig.S3OutputPath"
                                },
                                "/",
                                {
                                    "Get": "Steps.EvaluateBaseInferenceModel.TrainingJobName"
                                },
                                "/output/output/",
                                "BaseInference",
                                "/eval_results/inference_output.jsonl"
                            ]
                        }
                    },
                    "output_path": "{{ s3_output_path }}",
                    "llmaj_metrics": {{ llmaj_metrics | tojson }},{% if custom_metrics %}
                    "custom_metrics": "{{ custom_metrics }}",{% endif %}
                    "max_new_tokens": "{{ max_new_tokens }}",
                    "temperature": "{{ temperature }}",
                    "top_k": "{{ top_k }}",
                    "top_p": "{{ top_p }}"
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                }
            }
        }
    ]
}"""

# Deterministic Template for Base Model Only - Single step evaluation with optional ModelPackageConfig
DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY = """{
    "Version": "2020-12-01",
    "Metadata": {},
    "Parameters": [],
    "Steps": [
        {
            "Name": "EvaluateBaseModel",
            "Type": "Training",
            "Arguments": {
                "RoleArn": "{{ role_arn }}",
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "BenchmarkEvaluation"{% if evaluator_arn %},
                    "EvaluatorArn": "{{ evaluator_arn }}"{% endif %}
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "task": "{{ task }}",
                    "strategy": "{{ strategy }}"{% if metric is defined %},
                    "metric": "{{ metric }}"{% elif evaluation_metric is defined %},
                    "evaluation_metric": "{{ evaluation_metric }}"{% endif %}{% if max_new_tokens is defined %},
                    "max_new_tokens": "{{ max_new_tokens }}"{% endif %}{% if temperature is defined %},
                    "temperature": "{{ temperature }}"{% endif %}{% if top_k is defined %},
                    "top_k": "{{ top_k }}"{% endif %}{% if top_p is defined %},
                    "top_p": "{{ top_p }}"{% endif %}{% if max_model_len is defined %},
                    "max_model_len": "{{ max_model_len }}"{% endif %}{% if aggregation is defined %},
                    "aggregation": "{{ aggregation }}"{% endif %}{% if postprocessing is defined %},
                    "postprocessing": "{{ postprocessing }}"{% endif %}{% if preset_reward_function is defined %},
                    "preset_reward_function": "{{ preset_reward_function }}"{% endif %}{% if subtask %},
                    "subtask": "{{ subtask }}"{% endif %}
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
            "VpcConfig": {
                "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                "Subnets": {{ vpc_subnets | tojson }}
            }{% endif %}
            }
        }
    ]
}"""

# Custom Scorer Template with Jinja2 Placeholders - Similar to Deterministic but uses CustomScorerEvaluation
CUSTOM_SCORER_TEMPLATE = """{
    "Version": "2020-12-01",
    "Metadata": {},
    "MlflowConfig": {
        "MlflowResourceArn": "{{ mlflow_resource_arn }}"{% if mlflow_experiment_name %},
        "MlflowExperimentName": "{{ mlflow_experiment_name }}"{% endif %}{% if mlflow_run_name %},
        "MlflowRunName": "{{ mlflow_run_name }}"{% endif %}
    },
    "Parameters": [],
    "Steps": [
        {
            "Name": "CreateEvaluationAction",
            "Type": "Lineage",
            "Arguments": {
                "Actions": [
                    {
                        "ActionName": {
                            "Get": "Execution.PipelineExecutionId"
                        },
                        "ActionType": "Evaluation",
                        "Source": {
                            "SourceUri": "{{ source_model_package_arn }}",
                            "SourceType": "ModelPackage"
                        },
                        "Properties": {
                            "PipelineExecutionArn": {
                                "Get": "Execution.PipelineExecutionArn"
                            },
                            "PipelineName": "{{ pipeline_name }}"
                        }
                    }
                ],
                "Contexts": [
                    {
                        "ContextName": {
                            "Get": "Execution.PipelineExecutionId"
                        },
                        "ContextType": "PipelineExecution",
                        "Source": {
                            "SourceUri": {
                                "Get": "Execution.PipelineExecutionArn"
                            }
                        }
                    }
                ],
                "Associations": [
                    {
                        "Source": {
                            "Name": {
                                "Get": "Execution.PipelineExecutionId"
                            },
                            "Type": "Action"
                        },
                        "Destination": {
                            "Name": {
                                "Get": "Execution.PipelineExecutionId"
                            },
                            "Type": "Context"
                        },
                        "AssociationType": "ContributedTo"
                    },
                    {
                        "Source": {
                            "Arn": "{{ dataset_artifact_arn }}"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    }
                ]
            }
        },{% if evaluate_base_model %}
        {
            "Name": "EvaluateBaseModel",
            "Type": "Training",
            "Arguments": {
                "RoleArn": "{{ role_arn }}",
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": "{{ model_package_group_arn }}",
                    "SourceModelPackageArn": "{{ source_model_package_arn }}"
                },
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "CustomScorerEvaluation"{% if evaluator_arn %},
                    "EvaluatorArn": "{{ evaluator_arn }}"{% endif %}
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "task": "{{ task }}",
                    "strategy": "{{ strategy }}"{% if metric is defined %},
                    "metric": "{{ metric }}"{% elif evaluation_metric is defined %},
                    "evaluation_metric": "{{ evaluation_metric }}"{% endif %}{% if lambda_type is defined %},
                    "lambda_type": "{{ lambda_type }}"{% endif %}{% if max_new_tokens is defined %},
                    "max_new_tokens": "{{ max_new_tokens }}"{% endif %}{% if temperature is defined %},
                    "temperature": "{{ temperature }}"{% endif %}{% if top_k is defined %},
                    "top_k": "{{ top_k }}"{% endif %}{% if top_p is defined %},
                    "top_p": "{{ top_p }}"{% endif %}{% if max_model_len is defined %},
                    "max_model_len": "{{ max_model_len }}"{% endif %}{% if aggregation is defined %},
                    "aggregation": "{{ aggregation }}"{% endif %}{% if postprocessing is defined %},
                    "postprocessing": "{{ postprocessing }}"{% endif %}{% if preset_reward_function is defined %},
                    "preset_reward_function": "{{ preset_reward_function }}"{% endif %}{% if subtask %},
                    "subtask": "{{ subtask }}"{% endif %}
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
            "VpcConfig": {
                "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                "Subnets": {{ vpc_subnets | tojson }}
            }{% endif %}
            }
        },{% endif %}
        {
            "Name": "EvaluateCustomModel",
            "Type": "Training",
            "Arguments": {
                "RoleArn": "{{ role_arn }}",
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": "{{ model_package_group_arn }}",
                    "SourceModelPackageArn": "{{ source_model_package_arn }}"
                },
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "CustomScorerEvaluation"{% if evaluator_arn %},
                    "EvaluatorArn": "{{ evaluator_arn }}"{% endif %}
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "task": "{{ task }}",
                    "strategy": "{{ strategy }}"{% if metric is defined %},
                    "metric": "{{ metric }}"{% elif evaluation_metric is defined %},
                    "evaluation_metric": "{{ evaluation_metric }}"{% endif %}{% if lambda_type is defined %},
                    "lambda_type": "{{ lambda_type }}"{% endif %}{% if max_new_tokens is defined %},
                    "max_new_tokens": "{{ max_new_tokens }}"{% endif %}{% if temperature is defined %},
                    "temperature": "{{ temperature }}"{% endif %}{% if top_k is defined %},
                    "top_k": "{{ top_k }}"{% endif %}{% if top_p is defined %},
                    "top_p": "{{ top_p }}"{% endif %}{% if max_model_len is defined %},
                    "max_model_len": "{{ max_model_len }}"{% endif %}{% if aggregation is defined %},
                    "aggregation": "{{ aggregation }}"{% endif %}{% if postprocessing is defined %},
                    "postprocessing": "{{ postprocessing }}"{% endif %}{% if preset_reward_function is defined %},
                    "preset_reward_function": "{{ preset_reward_function }}"{% endif %}{% if subtask %},
                    "subtask": "{{ subtask }}"{% endif %}
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
            "VpcConfig": {
                "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                "Subnets": {{ vpc_subnets | tojson }}
            }{% endif %}
            }
        },
        {
            "Name": "AssociateLineage",
            "Type": "Lineage",
            "DependsOn": [
                "CreateEvaluationAction"
            ],
            "Arguments": {
                "Artifacts": [{% if evaluate_base_model %}
                    {
                        "ArtifactName": {
                            "Std:Join": {
                                "On": "-",
                                "Values": [
                                    {
                                        "Get": "Execution.PipelineExecutionId"
                                    },
                                    "base-eval-report"
                                ]
                            }
                        },
                        "ArtifactType": "EvaluationReport",
                        "Source": {
                            "SourceUri": {
                                "Get": "Steps.EvaluateBaseModel.OutputDataConfig.S3OutputPath"
                            }
                        }
                    },{% endif %}
                    {
                        "ArtifactName": {
                            "Std:Join": {
                                "On": "-",
                                "Values": [
                                    {
                                        "Get": "Execution.PipelineExecutionId"
                                    },
                                    "custom-eval-report"
                                ]
                            }
                        },
                        "ArtifactType": "EvaluationReport",
                        "Source": {
                            "SourceUri": {
                                "Get": "Steps.EvaluateCustomModel.OutputDataConfig.S3OutputPath"
                            }
                        }
                    }
                ],
                "Associations": [{% if evaluate_base_model %}
                    {
                        "Source": {
                            "Name": {
                                "Std:Join": {
                                    "On": "-",
                                    "Values": [
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        },
                                        "base-eval-report"
                                    ]
                                }
                            },
                            "Type": "Artifact"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    },{% endif %}
                    {
                        "Source": {
                            "Name": {
                                "Std:Join": {
                                    "On": "-",
                                    "Values": [
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        },
                                        "custom-eval-report"
                                    ]
                                }
                            },
                            "Type": "Artifact"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    }
                ]
            }
        }
    ]
}"""

# Custom Scorer Template for Base Model Only - Single step evaluation with optional ModelPackageConfig
CUSTOM_SCORER_TEMPLATE_BASE_MODEL_ONLY = """{
    "Version": "2020-12-01",
    "Metadata": {},
    "Parameters": [],
    "Steps": [
        {
            "Name": "EvaluateBaseModel",
            "Type": "Training",
            "Arguments": {
                "RoleArn": "{{ role_arn }}",
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "CustomScorerEvaluation"{% if evaluator_arn %},
                    "EvaluatorArn": "{{ evaluator_arn }}"{% endif %}
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "task": "{{ task }}",
                    "strategy": "{{ strategy }}"{% if metric is defined %},
                    "metric": "{{ metric }}"{% elif evaluation_metric is defined %},
                    "evaluation_metric": "{{ evaluation_metric }}"{% endif %}{% if lambda_type is defined %},
                    "lambda_type": "{{ lambda_type }}"{% endif %}{% if max_new_tokens is defined %},
                    "max_new_tokens": "{{ max_new_tokens }}"{% endif %}{% if temperature is defined %},
                    "temperature": "{{ temperature }}"{% endif %}{% if top_k is defined %},
                    "top_k": "{{ top_k }}"{% endif %}{% if top_p is defined %},
                    "top_p": "{{ top_p }}"{% endif %}{% if max_model_len is defined %},
                    "max_model_len": "{{ max_model_len }}"{% endif %}{% if aggregation is defined %},
                    "aggregation": "{{ aggregation }}"{% endif %}{% if postprocessing is defined %},
                    "postprocessing": "{{ postprocessing }}"{% endif %}{% if preset_reward_function is defined %},
                    "preset_reward_function": "{{ preset_reward_function }}"{% endif %}{% if subtask %},
                    "subtask": "{{ subtask }}"{% endif %}
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
            "VpcConfig": {
                "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                "Subnets": {{ vpc_subnets | tojson }}
            }{% endif %}
            }
        }
    ]
}"""

# LLM-as-a-Judge Template with Jinja2 Placeholders - 2-Phase Evaluation Pipeline (Type 2)
# Phase 1: Generate inference responses from base and custom models  
# Phase 2: Use judge model to evaluate responses with built-in and custom metrics
LLMAJ_TEMPLATE = """{
    "Version": "2020-12-01",
    "Metadata": {},
    "MlflowConfig": {
        "MlflowResourceArn": "{{ mlflow_resource_arn }}"{% if mlflow_experiment_name %},
        "MlflowExperimentName": "{{ mlflow_experiment_name }}"{% endif %}{% if mlflow_run_name %},
        "MlflowRunName": "{{ mlflow_run_name }}"{% endif %}
    },
    "Parameters": [],
    "Steps": [
        {
            "Name": "CreateEvaluationAction",
            "Type": "Lineage",
            "Arguments": {
                "Actions": [
                    {
                        "ActionName": {
                            "Get": "Execution.PipelineExecutionId"
                        },
                        "ActionType": "Evaluation",
                        "Source": {
                            "SourceUri": "{{ source_model_package_arn }}",
                            "SourceType": "ModelPackage"
                        },
                        "Properties": {
                            "PipelineExecutionArn": {
                                "Get": "Execution.PipelineExecutionArn"
                            },
                            "PipelineName": "{{ pipeline_name }}"
                        }
                    }
                ],
                "Contexts": [
                    {
                        "ContextName": {
                            "Get": "Execution.PipelineExecutionId"
                        },
                        "ContextType": "PipelineExecution",
                        "Source": {
                            "SourceUri": {
                                "Get": "Execution.PipelineExecutionArn"
                            }
                        }
                    }
                ],
                "Associations": [
                    {
                        "Source": {
                            "Name": {
                                "Get": "Execution.PipelineExecutionId"
                            },
                            "Type": "Action"
                        },
                        "Destination": {
                            "Name": {
                                "Get": "Execution.PipelineExecutionId"
                            },
                            "Type": "Context"
                        },
                        "AssociationType": "ContributedTo"
                    },
                    {
                        "Source": {
                            "Arn": "{{ dataset_artifact_arn }}"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    }
                ]
            }
        },{% if evaluate_base_model %}
        {
            "Name": "EvaluateBaseInferenceModel",
            "Type": "Training",
            "Arguments": {
                "TrainingJobName": "BaseInference",
                "RoleArn": "{{ role_arn }}",
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "BenchmarkEvaluation"
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "name": "BaseInference",
                    "task": "inference_only"
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                },
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": "{{ model_package_group_arn }}",
                    "SourceModelPackageArn": "{{ source_model_package_arn }}"
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
            "VpcConfig": {
                "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                "Subnets": {{ vpc_subnets | tojson }}
            }{% endif %}
            },
            "DependsOn": [
                "CreateEvaluationAction"
            ]
        },{% endif %}
        {
            "Name": "EvaluateCustomInferenceModel",
            "Type": "Training",
            "Arguments": {
                "TrainingJobName": "CustomInference",
                "RoleArn": "{{ role_arn }}",
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "BenchmarkEvaluation"
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "name": "CustomInference",
                    "task": "inference_only"
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                },
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": "{{ model_package_group_arn }}",
                    "SourceModelPackageArn": "{{ source_model_package_arn }}"
                }{% if dataset_uri %},
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {% if dataset_uri.startswith('arn:') and 'hub-content' in dataset_uri and '/DataSet/' in dataset_uri %}{
                            "DatasetSource": {
                                "DatasetArn": "{{ dataset_uri }}"
                            }
                        }{% else %}{
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "{{ dataset_uri }}"
                            }
                        }{% endif %}
                    }
                ]{% endif %}{% if vpc_config %},
            "VpcConfig": {
                "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},
                "Subnets": {{ vpc_subnets | tojson }}
            }{% endif %}
            },
            "DependsOn": [
                "CreateEvaluationAction"
            ]
        },{% if evaluate_base_model %}
        {
            "Name": "EvaluateBaseModelMetrics",
            "Type": "Training",
            "DependsOn": [
                "EvaluateBaseInferenceModel"
            ],
            "Arguments": {
                "TrainingJobName": {
                    "Std:Join": {
                        "On": "-",
                        "Values": [
                            "base-llmaj-eval",
                            {
                                "Get": "Execution.PipelineExecutionId"
                            }
                        ]
                    }
                },
                "RoleArn": "{{ role_arn }}",
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "LLMAJEvaluation"
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "name": {
                        "Std:Join": {
                            "On": "-",
                            "Values": [
                                "base-llmaj-eval",
                                {
                                    "Get": "Execution.PipelineExecutionId"
                                }
                            ]
                        }
                    },
                    "judge_model_id": "{{ judge_model_id }}",
                    "inference_data_s3_path": {
                        "Std:Join": {
                            "On": "",
                            "Values": [
                                {
                                    "Get": "Steps.EvaluateBaseInferenceModel.OutputDataConfig.S3OutputPath"
                                },
                                "/",
                                {
                                    "Get": "Steps.EvaluateBaseInferenceModel.TrainingJobName"
                                },
                                "/output/output/",
                                "BaseInference",
                                "/eval_results/inference_output.jsonl"
                            ]
                        }
                    },
                    "output_path": "{{ s3_output_path }}",
                    "llmaj_metrics": {{ llmaj_metrics | tojson }},{% if custom_metrics %}
                    "custom_metrics": "{{ custom_metrics }}",{% endif %}
                    "max_new_tokens": "{{ max_new_tokens }}",
                    "temperature": "{{ temperature }}",
                    "top_k": "{{ top_k }}",
                    "top_p": "{{ top_p }}"
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                },
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": "{{ model_package_group_arn }}",
                    "SourceModelPackageArn": "{{ source_model_package_arn }}"
                }
            }
        },{% endif %}
        {
            "Name": "EvaluateCustomModelMetrics",
            "Type": "Training",
            "DependsOn": [
                "EvaluateCustomInferenceModel"
            ],
            "Arguments": {
                "TrainingJobName": {
                    "Std:Join": {
                        "On": "-",
                        "Values": [
                            "custom-llmaj-eval",
                            {
                                "Get": "Execution.PipelineExecutionId"
                            }
                        ]
                    }
                },
                "RoleArn": "{{ role_arn }}",
                "ServerlessJobConfig": {
                    "BaseModelArn": "{{ base_model_arn }}",
                    "AcceptEula": true,
                    "JobType": "Evaluation",
                    "EvaluationType": "LLMAJEvaluation"
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                },
                "HyperParameters": {
                    "name": {
                        "Std:Join": {
                            "On": "-",
                            "Values": [
                                "custom-llmaj-eval",
                                {
                                    "Get": "Execution.PipelineExecutionId"
                                }
                            ]
                        }
                    },
                    "judge_model_id": "{{ judge_model_id }}",
                    "inference_data_s3_path": {
                        "Std:Join": {
                            "On": "",
                            "Values": [
                                {
                                    "Get": "Steps.EvaluateCustomInferenceModel.OutputDataConfig.S3OutputPath"
                                },
                                "/",
                                {
                                    "Get": "Steps.EvaluateCustomInferenceModel.TrainingJobName"
                                },
                                "/output/output/",
                                "CustomInference",
                                "/eval_results/inference_output.jsonl"
                            ]
                        }
                    },
                    "output_path": "{{ s3_output_path }}",
                    "llmaj_metrics": {{ llmaj_metrics | tojson }},{% if custom_metrics %}
                    "custom_metrics": "{{ custom_metrics }}",{% endif %}
                    "max_new_tokens": "{{ max_new_tokens }}",
                    "temperature": "{{ temperature }}",
                    "top_k": "{{ top_k }}",
                    "top_p": "{{ top_p }}"
                },
                "OutputDataConfig": {
                    "S3OutputPath": "{{ s3_output_path }}",
                    "CompressionType": "NONE"
                {% if kms_key_id %},
                "KmsKeyId": "{{ kms_key_id }}"
                {% endif %}
                },
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": "{{ model_package_group_arn }}",
                    "SourceModelPackageArn": "{{ source_model_package_arn }}"
                }
            }
        },
        {
            "Name": "AssociateLineage",
            "Type": "Lineage",
            "DependsOn": [
                "CreateEvaluationAction"
            ],
            "Arguments": {
                "Artifacts": [{% if evaluate_base_model %}
                    {
                        "ArtifactName": {
                            "Std:Join": {
                                "On": "-",
                                "Values": [
                                    {
                                        "Get": "Execution.PipelineExecutionId"
                                    },
                                    "base-inference-results"
                                ]
                            }
                        },
                        "ArtifactType": "InferenceResults",
                        "Source": {
                            "SourceUri": {
                                "Get": "Steps.EvaluateBaseInferenceModel.OutputDataConfig.S3OutputPath"
                            }
                        }
                    },
                    {
                        "ArtifactName": {
                            "Std:Join": {
                                "On": "-",
                                "Values": [
                                    {
                                        "Get": "Execution.PipelineExecutionId"
                                    },
                                    "base-eval-report"
                                ]
                            }
                        },
                        "ArtifactType": "EvaluationReport",
                        "Source": {
                            "SourceUri": {
                                "Get": "Steps.EvaluateBaseModelMetrics.OutputDataConfig.S3OutputPath"
                            }
                        }
                    },{% endif %}
                    {
                        "ArtifactName": {
                            "Std:Join": {
                                "On": "-",
                                "Values": [
                                    {
                                        "Get": "Execution.PipelineExecutionId"
                                    },
                                    "custom-inference-results"
                                ]
                            }
                        },
                        "ArtifactType": "InferenceResults",
                        "Source": {
                            "SourceUri": {
                                "Get": "Steps.EvaluateCustomInferenceModel.OutputDataConfig.S3OutputPath"
                            }
                        }
                    },
                    {
                        "ArtifactName": {
                            "Std:Join": {
                                "On": "-",
                                "Values": [
                                    {
                                        "Get": "Execution.PipelineExecutionId"
                                    },
                                    "custom-eval-report"
                                ]
                            }
                        },
                        "ArtifactType": "EvaluationReport",
                        "Source": {
                            "SourceUri": {
                                "Get": "Steps.EvaluateCustomModelMetrics.OutputDataConfig.S3OutputPath"
                            }
                        }
                    }
                ],
                "Associations": [{% if evaluate_base_model %}
                    {
                        "Source": {
                            "Name": {
                                "Std:Join": {
                                    "On": "-",
                                    "Values": [
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        },
                                        "base-inference-results"
                                    ]
                                }
                            },
                            "Type": "Artifact"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    },
                    {
                        "Source": {
                            "Name": {
                                "Std:Join": {
                                    "On": "-",
                                    "Values": [
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        },
                                        "base-eval-report"
                                    ]
                                }
                            },
                            "Type": "Artifact"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    },{% endif %}
                    {
                        "Source": {
                            "Name": {
                                "Std:Join": {
                                    "On": "-",
                                    "Values": [
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        },
                                        "custom-inference-results"
                                    ]
                                }
                            },
                            "Type": "Artifact"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    },
                    {
                        "Source": {
                            "Name": {
                                "Std:Join": {
                                    "On": "-",
                                    "Values": [
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        },
                                        "custom-eval-report"
                                    ]
                                }
                            },
                            "Type": "Artifact"
                        },
                        "Destination": {
                            "Arn": {
                                "Std:Join": {
                                    "On": "/",
                                    "Values": [
                                        "{{ action_arn_prefix }}",
                                        {
                                            "Get": "Execution.PipelineExecutionId"
                                        }
                                    ]
                                }
                            }
                        },
                        "AssociationType": "ContributedTo"
                    }
                ]
            }
        }
    ]
}"""
