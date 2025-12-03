SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "SchemaVersion": {
            "type": "string",
            "enum": ["1.0"],
            "description": "The schema version of the document.",
        },
        "SageMaker": {
            "type": "object",
            "properties": {
                "PythonSDK": {
                    "type": "object",
                    "properties": {
                        "Resources": {
                            "type": "object",
                            "properties": {
                                "Algorithm": {
                                    "type": "object",
                                    "properties": {
                                        "training_specification": {
                                            "additional_s3_data_source": {
                                                "s3_data_type": {"type": "string"},
                                                "s3_uri": {"type": "string"},
                                                "manifest_s3_uri": {"type": "string"},
                                            }
                                        },
                                        "validation_specification": {
                                            "validation_role": {"type": "string"}
                                        },
                                    },
                                },
                                "AutoMLJob": {
                                    "type": "object",
                                    "properties": {
                                        "output_data_config": {
                                            "s3_output_path": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "role_arn": {"type": "string"},
                                        "auto_ml_job_config": {
                                            "security_config": {
                                                "volume_kms_key_id": {"type": "string"},
                                                "vpc_config": {
                                                    "security_group_ids": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "subnets": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                },
                                            },
                                            "candidate_generation_config": {
                                                "feature_specification_s3_uri": {"type": "string"}
                                            },
                                        },
                                    },
                                },
                                "AutoMLJobV2": {
                                    "type": "object",
                                    "properties": {
                                        "output_data_config": {
                                            "s3_output_path": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "role_arn": {"type": "string"},
                                        "auto_ml_problem_type_config": {
                                            "time_series_forecasting_job_config": {
                                                "feature_specification_s3_uri": {"type": "string"}
                                            },
                                            "tabular_job_config": {
                                                "feature_specification_s3_uri": {"type": "string"}
                                            },
                                        },
                                        "security_config": {
                                            "volume_kms_key_id": {"type": "string"},
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                        "auto_ml_compute_config": {
                                            "emr_serverless_compute_config": {
                                                "execution_role_arn": {"type": "string"}
                                            }
                                        },
                                    },
                                },
                                "Cluster": {
                                    "type": "object",
                                    "properties": {
                                        "vpc_config": {
                                            "security_group_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "subnets": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                        "cluster_role": {"type": "string"},
                                    },
                                },
                                "CompilationJob": {
                                    "type": "object",
                                    "properties": {
                                        "model_artifacts": {
                                            "s3_model_artifacts": {"type": "string"}
                                        },
                                        "role_arn": {"type": "string"},
                                        "input_config": {"s3_uri": {"type": "string"}},
                                        "output_config": {
                                            "s3_output_location": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "resource_config": {
                                            "volume_kms_key_id": {"type": "string"}
                                        },
                                        "vpc_config": {
                                            "security_group_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "subnets": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                                "CustomMonitoringJobDefinition": {
                                    "type": "object",
                                    "properties": {
                                        "custom_monitoring_job_input": {
                                            "endpoint_input": {
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                            "batch_transform_input": {
                                                "data_captured_destination_s3_uri": {
                                                    "type": "string"
                                                },
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                            "ground_truth_s3_input": {"s3_uri": {"type": "string"}},
                                        },
                                        "job_resources": {
                                            "cluster_config": {
                                                "volume_kms_key_id": {"type": "string"}
                                            }
                                        },
                                        "role_arn": {"type": "string"},
                                        "custom_monitoring_job_output_config": {
                                            "kms_key_id": {"type": "string"}
                                        },
                                        "network_config": {
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            }
                                        },
                                    },
                                },
                                "DataQualityJobDefinition": {
                                    "type": "object",
                                    "properties": {
                                        "data_quality_job_input": {
                                            "endpoint_input": {
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                            "batch_transform_input": {
                                                "data_captured_destination_s3_uri": {
                                                    "type": "string"
                                                },
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                        },
                                        "data_quality_job_output_config": {
                                            "kms_key_id": {"type": "string"}
                                        },
                                        "job_resources": {
                                            "cluster_config": {
                                                "volume_kms_key_id": {"type": "string"}
                                            }
                                        },
                                        "role_arn": {"type": "string"},
                                        "data_quality_baseline_config": {
                                            "constraints_resource": {"s3_uri": {"type": "string"}},
                                            "statistics_resource": {"s3_uri": {"type": "string"}},
                                        },
                                        "network_config": {
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            }
                                        },
                                    },
                                },
                                "DeviceFleet": {
                                    "type": "object",
                                    "properties": {
                                        "output_config": {
                                            "s3_output_location": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "role_arn": {"type": "string"},
                                        "iot_role_alias": {"type": "string"},
                                    },
                                },
                                "Domain": {
                                    "type": "object",
                                    "properties": {
                                        "security_group_id_for_domain_boundary": {"type": "string"},
                                        "default_user_settings": {
                                            "execution_role": {"type": "string"},
                                            "environment_settings": {
                                                "default_s3_artifact_path": {"type": "string"},
                                                "default_s3_kms_key_id": {"type": "string"},
                                            },
                                            "security_groups": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "sharing_settings": {
                                                "s3_output_path": {"type": "string"},
                                                "s3_kms_key_id": {"type": "string"},
                                            },
                                            "canvas_app_settings": {
                                                "time_series_forecasting_settings": {
                                                    "amazon_forecast_role_arn": {"type": "string"}
                                                },
                                                "model_register_settings": {
                                                    "cross_account_model_register_role_arn": {
                                                        "type": "string"
                                                    }
                                                },
                                                "workspace_settings": {
                                                    "s3_artifact_path": {"type": "string"},
                                                    "s3_kms_key_id": {"type": "string"},
                                                },
                                                "generative_ai_settings": {
                                                    "amazon_bedrock_role_arn": {"type": "string"}
                                                },
                                                "emr_serverless_settings": {
                                                    "execution_role_arn": {"type": "string"}
                                                },
                                            },
                                            "jupyter_lab_app_settings": {
                                                "emr_settings": {
                                                    "assumable_role_arns": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "execution_role_arns": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                }
                                            },
                                            "emr_settings": {
                                                "assumable_role_arns": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "execution_role_arns": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                        "domain_settings": {
                                            "security_group_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "r_studio_server_pro_domain_settings": {
                                                "domain_execution_role_arn": {"type": "string"}
                                            },
                                            "execution_role_identity_config": {"type": "string"},
                                            "unified_studio_settings": {
                                                "project_s3_path": {"type": "string"}
                                            },
                                        },
                                        "home_efs_file_system_kms_key_id": {"type": "string"},
                                        "subnet_ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "kms_key_id": {"type": "string"},
                                        "app_security_group_management": {"type": "string"},
                                        "default_space_settings": {
                                            "execution_role": {"type": "string"},
                                            "security_groups": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "jupyter_lab_app_settings": {
                                                "emr_settings": {
                                                    "assumable_role_arns": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "execution_role_arns": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                }
                                            },
                                        },
                                    },
                                },
                                "EdgePackagingJob": {
                                    "type": "object",
                                    "properties": {
                                        "role_arn": {"type": "string"},
                                        "output_config": {
                                            "s3_output_location": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                    },
                                },
                                "Endpoint": {
                                    "type": "object",
                                    "properties": {
                                        "data_capture_config": {
                                            "destination_s3_uri": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "async_inference_config": {
                                            "output_config": {
                                                "kms_key_id": {"type": "string"},
                                                "s3_output_path": {"type": "string"},
                                                "s3_failure_path": {"type": "string"},
                                            }
                                        },
                                    },
                                },
                                "EndpointConfig": {
                                    "type": "object",
                                    "properties": {
                                        "data_capture_config": {
                                            "destination_s3_uri": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "kms_key_id": {"type": "string"},
                                        "async_inference_config": {
                                            "output_config": {
                                                "kms_key_id": {"type": "string"},
                                                "s3_output_path": {"type": "string"},
                                                "s3_failure_path": {"type": "string"},
                                            }
                                        },
                                        "execution_role_arn": {"type": "string"},
                                        "vpc_config": {
                                            "security_group_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "subnets": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                                "EvaluationJob": {
                                    "type": "object",
                                    "properties": {
                                        "output_data_config": {
                                            "s3_uri": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "role_arn": {"type": "string"},
                                        "upstream_platform_config": {
                                            "upstream_platform_customer_output_data_config": {
                                                "s3_uri": {"type": "string"},
                                                "kms_key_id": {"type": "string"},
                                                "s3_kms_encryption_context": {"type": "string"},
                                            },
                                            "upstream_platform_customer_execution_role": {
                                                "type": "string"
                                            },
                                        },
                                    },
                                },
                                "FeatureGroup": {
                                    "type": "object",
                                    "properties": {
                                        "online_store_config": {
                                            "security_config": {"kms_key_id": {"type": "string"}}
                                        },
                                        "offline_store_config": {
                                            "s3_storage_config": {
                                                "s3_uri": {"type": "string"},
                                                "kms_key_id": {"type": "string"},
                                                "resolved_output_s3_uri": {"type": "string"},
                                            }
                                        },
                                        "role_arn": {"type": "string"},
                                    },
                                },
                                "FlowDefinition": {
                                    "type": "object",
                                    "properties": {
                                        "output_config": {
                                            "s3_output_path": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "role_arn": {"type": "string"},
                                        "task_rendering_role_arn": {"type": "string"},
                                        "kms_key_id": {"type": "string"},
                                    },
                                },
                                "GroundTruthJob": {
                                    "type": "object",
                                    "properties": {
                                        "input_config": {
                                            "data_source": {
                                                "s3_data_source": {"s3_uri": {"type": "string"}}
                                            }
                                        },
                                        "output_config": {"s3_output_path": {"type": "string"}},
                                    },
                                },
                                "GroundTruthWorkflow": {
                                    "type": "object",
                                    "properties": {"execution_role_arn": {"type": "string"}},
                                },
                                "Hub": {
                                    "type": "object",
                                    "properties": {
                                        "s3_storage_config": {"s3_output_path": {"type": "string"}}
                                    },
                                },
                                "HumanTaskUi": {
                                    "type": "object",
                                    "properties": {"kms_key_id": {"type": "string"}},
                                },
                                "HyperParameterTuningJob": {
                                    "type": "object",
                                    "properties": {
                                        "training_job_definition": {
                                            "role_arn": {"type": "string"},
                                            "output_data_config": {
                                                "s3_output_path": {"type": "string"},
                                                "kms_key_id": {"type": "string"},
                                                "remove_job_name_from_s3_output_path": {
                                                    "type": "boolean"
                                                },
                                            },
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                            "resource_config": {
                                                "volume_kms_key_id": {"type": "string"}
                                            },
                                            "hyper_parameter_tuning_resource_config": {
                                                "volume_kms_key_id": {"type": "string"}
                                            },
                                            "checkpoint_config": {"s3_uri": {"type": "string"}},
                                        }
                                    },
                                },
                                "Image": {
                                    "type": "object",
                                    "properties": {"role_arn": {"type": "string"}},
                                },
                                "InferenceExperiment": {
                                    "type": "object",
                                    "properties": {
                                        "role_arn": {"type": "string"},
                                        "data_storage_config": {"kms_key": {"type": "string"}},
                                        "kms_key": {"type": "string"},
                                    },
                                },
                                "InferenceRecommendationsJob": {
                                    "type": "object",
                                    "properties": {
                                        "role_arn": {"type": "string"},
                                        "input_config": {
                                            "volume_kms_key_id": {"type": "string"},
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                        "output_config": {
                                            "kms_key_id": {"type": "string"},
                                            "compiled_output_config": {
                                                "s3_output_uri": {"type": "string"}
                                            },
                                            "benchmark_results_output_config": {
                                                "s3_output_uri": {"type": "string"}
                                            },
                                        },
                                    },
                                },
                                "LabelingJob": {
                                    "type": "object",
                                    "properties": {
                                        "input_config": {
                                            "data_source": {
                                                "s3_data_source": {
                                                    "manifest_s3_uri": {"type": "string"}
                                                }
                                            }
                                        },
                                        "output_config": {
                                            "s3_output_path": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "role_arn": {"type": "string"},
                                        "human_task_config": {
                                            "ui_config": {"ui_template_s3_uri": {"type": "string"}}
                                        },
                                        "task_rendering_role_arn": {"type": "string"},
                                        "label_category_config_s3_uri": {"type": "string"},
                                        "labeling_job_algorithms_config": {
                                            "labeling_job_resource_config": {
                                                "volume_kms_key_id": {"type": "string"},
                                                "vpc_config": {
                                                    "security_group_ids": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "subnets": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                },
                                            }
                                        },
                                        "labeling_job_output": {
                                            "output_dataset_s3_uri": {"type": "string"}
                                        },
                                    },
                                },
                                "MlflowApp": {
                                    "type": "object",
                                    "properties": {"role_arn": {"type": "string"}},
                                },
                                "MlflowTrackingServer": {
                                    "type": "object",
                                    "properties": {"role_arn": {"type": "string"}},
                                },
                                "Model": {
                                    "type": "object",
                                    "properties": {
                                        "primary_container": {
                                            "model_data_source": {
                                                "s3_data_source": {
                                                    "s3_uri": {"type": "string"},
                                                    "s3_data_type": {"type": "string"},
                                                    "manifest_s3_uri": {"type": "string"},
                                                }
                                            }
                                        },
                                        "execution_role_arn": {"type": "string"},
                                        "vpc_config": {
                                            "security_group_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "subnets": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                                "ModelBiasJobDefinition": {
                                    "type": "object",
                                    "properties": {
                                        "model_bias_job_input": {
                                            "ground_truth_s3_input": {"s3_uri": {"type": "string"}},
                                            "endpoint_input": {
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                            "batch_transform_input": {
                                                "data_captured_destination_s3_uri": {
                                                    "type": "string"
                                                },
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                        },
                                        "model_bias_job_output_config": {
                                            "kms_key_id": {"type": "string"}
                                        },
                                        "job_resources": {
                                            "cluster_config": {
                                                "volume_kms_key_id": {"type": "string"}
                                            }
                                        },
                                        "role_arn": {"type": "string"},
                                        "model_bias_baseline_config": {
                                            "constraints_resource": {"s3_uri": {"type": "string"}}
                                        },
                                        "network_config": {
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            }
                                        },
                                    },
                                },
                                "ModelCard": {
                                    "type": "object",
                                    "properties": {
                                        "security_config": {"kms_key_id": {"type": "string"}}
                                    },
                                },
                                "ModelCardExportJob": {
                                    "type": "object",
                                    "properties": {
                                        "output_config": {"s3_output_path": {"type": "string"}},
                                        "export_artifacts": {
                                            "s3_export_artifacts": {"type": "string"}
                                        },
                                    },
                                },
                                "ModelExplainabilityJobDefinition": {
                                    "type": "object",
                                    "properties": {
                                        "model_explainability_job_input": {
                                            "endpoint_input": {
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                            "batch_transform_input": {
                                                "data_captured_destination_s3_uri": {
                                                    "type": "string"
                                                },
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                        },
                                        "model_explainability_job_output_config": {
                                            "kms_key_id": {"type": "string"}
                                        },
                                        "job_resources": {
                                            "cluster_config": {
                                                "volume_kms_key_id": {"type": "string"}
                                            }
                                        },
                                        "role_arn": {"type": "string"},
                                        "model_explainability_baseline_config": {
                                            "constraints_resource": {"s3_uri": {"type": "string"}}
                                        },
                                        "network_config": {
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            }
                                        },
                                    },
                                },
                                "ModelPackage": {
                                    "type": "object",
                                    "properties": {
                                        "validation_specification": {
                                            "validation_role": {"type": "string"}
                                        },
                                        "model_metrics": {
                                            "model_quality": {
                                                "statistics": {"s3_uri": {"type": "string"}},
                                                "constraints": {"s3_uri": {"type": "string"}},
                                            },
                                            "model_data_quality": {
                                                "statistics": {"s3_uri": {"type": "string"}},
                                                "constraints": {"s3_uri": {"type": "string"}},
                                            },
                                            "bias": {
                                                "report": {"s3_uri": {"type": "string"}},
                                                "pre_training_report": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                                "post_training_report": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                            },
                                            "explainability": {
                                                "report": {"s3_uri": {"type": "string"}}
                                            },
                                        },
                                        "deployment_specification": {
                                            "test_input": {
                                                "data_source": {
                                                    "s3_data_source": {
                                                        "s3_data_type": {"type": "string"},
                                                        "s3_uri": {"type": "string"},
                                                        "s3_data_distribution_type": {
                                                            "type": "string"
                                                        },
                                                    }
                                                }
                                            }
                                        },
                                        "drift_check_baselines": {
                                            "bias": {
                                                "config_file": {"s3_uri": {"type": "string"}},
                                                "pre_training_constraints": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                                "post_training_constraints": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                            },
                                            "explainability": {
                                                "constraints": {"s3_uri": {"type": "string"}},
                                                "config_file": {"s3_uri": {"type": "string"}},
                                            },
                                            "model_quality": {
                                                "statistics": {"s3_uri": {"type": "string"}},
                                                "constraints": {"s3_uri": {"type": "string"}},
                                            },
                                            "model_data_quality": {
                                                "statistics": {"s3_uri": {"type": "string"}},
                                                "constraints": {"s3_uri": {"type": "string"}},
                                            },
                                        },
                                        "security_config": {"kms_key_id": {"type": "string"}},
                                    },
                                },
                                "ModelQualityJobDefinition": {
                                    "type": "object",
                                    "properties": {
                                        "model_quality_job_input": {
                                            "ground_truth_s3_input": {"s3_uri": {"type": "string"}},
                                            "endpoint_input": {
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                            "batch_transform_input": {
                                                "data_captured_destination_s3_uri": {
                                                    "type": "string"
                                                },
                                                "s3_input_mode": {"type": "string"},
                                                "s3_data_distribution_type": {"type": "string"},
                                            },
                                        },
                                        "model_quality_job_output_config": {
                                            "kms_key_id": {"type": "string"}
                                        },
                                        "job_resources": {
                                            "cluster_config": {
                                                "volume_kms_key_id": {"type": "string"}
                                            }
                                        },
                                        "role_arn": {"type": "string"},
                                        "model_quality_baseline_config": {
                                            "constraints_resource": {"s3_uri": {"type": "string"}}
                                        },
                                        "network_config": {
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            }
                                        },
                                    },
                                },
                                "MonitoringSchedule": {
                                    "type": "object",
                                    "properties": {
                                        "monitoring_schedule_config": {
                                            "monitoring_job_definition": {
                                                "monitoring_output_config": {
                                                    "kms_key_id": {"type": "string"}
                                                },
                                                "monitoring_resources": {
                                                    "cluster_config": {
                                                        "volume_kms_key_id": {"type": "string"}
                                                    }
                                                },
                                                "role_arn": {"type": "string"},
                                                "baseline_config": {
                                                    "constraints_resource": {
                                                        "s3_uri": {"type": "string"}
                                                    },
                                                    "statistics_resource": {
                                                        "s3_uri": {"type": "string"}
                                                    },
                                                },
                                                "network_config": {
                                                    "vpc_config": {
                                                        "security_group_ids": {
                                                            "type": "array",
                                                            "items": {"type": "string"},
                                                        },
                                                        "subnets": {
                                                            "type": "array",
                                                            "items": {"type": "string"},
                                                        },
                                                    }
                                                },
                                            }
                                        },
                                        "custom_monitoring_job_definition": {
                                            "custom_monitoring_job_input": {
                                                "endpoint_input": {
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                                "batch_transform_input": {
                                                    "data_captured_destination_s3_uri": {
                                                        "type": "string"
                                                    },
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                                "ground_truth_s3_input": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                            },
                                            "custom_monitoring_job_output_config": {
                                                "kms_key_id": {"type": "string"}
                                            },
                                            "job_resources": {
                                                "cluster_config": {
                                                    "volume_kms_key_id": {"type": "string"}
                                                }
                                            },
                                            "role_arn": {"type": "string"},
                                            "network_config": {
                                                "vpc_config": {
                                                    "security_group_ids": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "subnets": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                }
                                            },
                                        },
                                        "data_quality_job_definition": {
                                            "data_quality_job_input": {
                                                "endpoint_input": {
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                                "batch_transform_input": {
                                                    "data_captured_destination_s3_uri": {
                                                        "type": "string"
                                                    },
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                            },
                                            "data_quality_job_output_config": {
                                                "kms_key_id": {"type": "string"}
                                            },
                                            "job_resources": {
                                                "cluster_config": {
                                                    "volume_kms_key_id": {"type": "string"}
                                                }
                                            },
                                            "role_arn": {"type": "string"},
                                            "data_quality_baseline_config": {
                                                "constraints_resource": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                                "statistics_resource": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                            },
                                            "network_config": {
                                                "vpc_config": {
                                                    "security_group_ids": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "subnets": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                }
                                            },
                                        },
                                        "model_quality_job_definition": {
                                            "model_quality_job_input": {
                                                "ground_truth_s3_input": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                                "endpoint_input": {
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                                "batch_transform_input": {
                                                    "data_captured_destination_s3_uri": {
                                                        "type": "string"
                                                    },
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                            },
                                            "model_quality_job_output_config": {
                                                "kms_key_id": {"type": "string"}
                                            },
                                            "job_resources": {
                                                "cluster_config": {
                                                    "volume_kms_key_id": {"type": "string"}
                                                }
                                            },
                                            "role_arn": {"type": "string"},
                                            "model_quality_baseline_config": {
                                                "constraints_resource": {
                                                    "s3_uri": {"type": "string"}
                                                }
                                            },
                                            "network_config": {
                                                "vpc_config": {
                                                    "security_group_ids": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "subnets": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                }
                                            },
                                        },
                                        "model_bias_job_definition": {
                                            "model_bias_job_input": {
                                                "ground_truth_s3_input": {
                                                    "s3_uri": {"type": "string"}
                                                },
                                                "endpoint_input": {
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                                "batch_transform_input": {
                                                    "data_captured_destination_s3_uri": {
                                                        "type": "string"
                                                    },
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                            },
                                            "model_bias_job_output_config": {
                                                "kms_key_id": {"type": "string"}
                                            },
                                            "job_resources": {
                                                "cluster_config": {
                                                    "volume_kms_key_id": {"type": "string"}
                                                }
                                            },
                                            "role_arn": {"type": "string"},
                                            "model_bias_baseline_config": {
                                                "constraints_resource": {
                                                    "s3_uri": {"type": "string"}
                                                }
                                            },
                                            "network_config": {
                                                "vpc_config": {
                                                    "security_group_ids": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "subnets": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                }
                                            },
                                        },
                                        "model_explainability_job_definition": {
                                            "model_explainability_job_input": {
                                                "endpoint_input": {
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                                "batch_transform_input": {
                                                    "data_captured_destination_s3_uri": {
                                                        "type": "string"
                                                    },
                                                    "s3_input_mode": {"type": "string"},
                                                    "s3_data_distribution_type": {"type": "string"},
                                                },
                                            },
                                            "model_explainability_job_output_config": {
                                                "kms_key_id": {"type": "string"}
                                            },
                                            "job_resources": {
                                                "cluster_config": {
                                                    "volume_kms_key_id": {"type": "string"}
                                                }
                                            },
                                            "role_arn": {"type": "string"},
                                            "model_explainability_baseline_config": {
                                                "constraints_resource": {
                                                    "s3_uri": {"type": "string"}
                                                }
                                            },
                                            "network_config": {
                                                "vpc_config": {
                                                    "security_group_ids": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "subnets": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                }
                                            },
                                        },
                                    },
                                },
                                "NotebookInstance": {
                                    "type": "object",
                                    "properties": {
                                        "subnet_id": {"type": "string"},
                                        "security_groups": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "role_arn": {"type": "string"},
                                        "kms_key_id": {"type": "string"},
                                    },
                                },
                                "OptimizationJob": {
                                    "type": "object",
                                    "properties": {
                                        "model_source": {"s3": {"s3_uri": {"type": "string"}}},
                                        "output_config": {
                                            "s3_output_location": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "role_arn": {"type": "string"},
                                        "vpc_config": {
                                            "security_group_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "subnets": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                                "PartnerApp": {
                                    "type": "object",
                                    "properties": {
                                        "execution_role_arn": {"type": "string"},
                                        "kms_key_id": {"type": "string"},
                                    },
                                },
                                "Pipeline": {
                                    "type": "object",
                                    "properties": {"role_arn": {"type": "string"}},
                                },
                                "ProcessingJob": {
                                    "type": "object",
                                    "properties": {
                                        "processing_resources": {
                                            "cluster_config": {
                                                "volume_kms_key_id": {"type": "string"}
                                            }
                                        },
                                        "processing_output_config": {
                                            "kms_key_id": {"type": "string"}
                                        },
                                        "network_config": {
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            }
                                        },
                                        "role_arn": {"type": "string"},
                                    },
                                },
                                "QuotaAllocation": {
                                    "type": "object",
                                    "properties": {
                                        "quota_allocation_target": {
                                            "roles": {"type": "array", "items": {"type": "string"}}
                                        }
                                    },
                                },
                                "TrainingJob": {
                                    "type": "object",
                                    "properties": {
                                        "model_artifacts": {
                                            "s3_model_artifacts": {"type": "string"}
                                        },
                                        "training_job_output": {
                                            "s3_training_job_output": {"type": "string"}
                                        },
                                        "role_arn": {"type": "string"},
                                        "output_data_config": {
                                            "s3_output_path": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                            "remove_job_name_from_s3_output_path": {
                                                "type": "boolean"
                                            },
                                        },
                                        "resource_config": {
                                            "volume_kms_key_id": {"type": "string"}
                                        },
                                        "vpc_config": {
                                            "security_group_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "subnets": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                        "checkpoint_config": {"s3_uri": {"type": "string"}},
                                        "debug_hook_config": {"s3_output_path": {"type": "string"}},
                                        "tensor_board_output_config": {
                                            "s3_output_path": {"type": "string"}
                                        },
                                        "upstream_platform_config": {
                                            "credential_proxy_config": {
                                                "customer_credential_provider_kms_key_id": {
                                                    "type": "string"
                                                },
                                                "platform_credential_provider_kms_key_id": {
                                                    "type": "string"
                                                },
                                            },
                                            "vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                            "output_data_config": {
                                                "kms_key_id": {"type": "string"}
                                            },
                                            "checkpoint_config": {"s3_uri": {"type": "string"}},
                                            "enable_s3_context_keys_on_input_data": {
                                                "type": "boolean"
                                            },
                                            "execution_role": {"type": "string"},
                                        },
                                        "profiler_config": {"s3_output_path": {"type": "string"}},
                                        "processing_job_config": {
                                            "processing_output_config": {
                                                "kms_key_id": {"type": "string"}
                                            },
                                            "upstream_processing_output_config": {
                                                "kms_key_id": {"type": "string"}
                                            },
                                        },
                                    },
                                },
                                "TransformJob": {
                                    "type": "object",
                                    "properties": {
                                        "transform_input": {
                                            "data_source": {
                                                "s3_data_source": {
                                                    "s3_data_type": {"type": "string"},
                                                    "s3_uri": {"type": "string"},
                                                }
                                            }
                                        },
                                        "transform_resources": {
                                            "volume_kms_key_id": {"type": "string"}
                                        },
                                        "transform_output": {
                                            "s3_output_path": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                        "data_capture_config": {
                                            "destination_s3_uri": {"type": "string"},
                                            "kms_key_id": {"type": "string"},
                                        },
                                    },
                                },
                                "UserProfile": {
                                    "type": "object",
                                    "properties": {
                                        "user_settings": {
                                            "execution_role": {"type": "string"},
                                            "environment_settings": {
                                                "default_s3_artifact_path": {"type": "string"},
                                                "default_s3_kms_key_id": {"type": "string"},
                                            },
                                            "security_groups": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "sharing_settings": {
                                                "s3_output_path": {"type": "string"},
                                                "s3_kms_key_id": {"type": "string"},
                                            },
                                            "canvas_app_settings": {
                                                "time_series_forecasting_settings": {
                                                    "amazon_forecast_role_arn": {"type": "string"}
                                                },
                                                "model_register_settings": {
                                                    "cross_account_model_register_role_arn": {
                                                        "type": "string"
                                                    }
                                                },
                                                "workspace_settings": {
                                                    "s3_artifact_path": {"type": "string"},
                                                    "s3_kms_key_id": {"type": "string"},
                                                },
                                                "generative_ai_settings": {
                                                    "amazon_bedrock_role_arn": {"type": "string"}
                                                },
                                                "emr_serverless_settings": {
                                                    "execution_role_arn": {"type": "string"}
                                                },
                                            },
                                            "jupyter_lab_app_settings": {
                                                "emr_settings": {
                                                    "assumable_role_arns": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    "execution_role_arns": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                }
                                            },
                                            "emr_settings": {
                                                "assumable_role_arns": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "execution_role_arns": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        }
                                    },
                                },
                                "Workforce": {
                                    "type": "object",
                                    "properties": {
                                        "workforce": {
                                            "workforce_vpc_config": {
                                                "security_group_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "subnets": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            }
                                        }
                                    },
                                },
                            },
                        }
                    },
                    "required": ["Resources"],
                }
            },
            "required": ["PythonSDK"],
        },
    },
    "required": ["SageMaker"],
}
