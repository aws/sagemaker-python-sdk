"""IAM policy configuration for SageMaker role auto-creation.

Defined as a Python constant (rather than a bundled JSON data file) so it is
always packaged with the module — no MANIFEST.in / package_data entry required.
Consumed by :mod:`sagemaker.core.helper.iam_role_resolver`.
"""
from __future__ import absolute_import

# Maps each role type to its trust policy and the least-privilege policies
# attached to the auto-created role. ``S3_PLACEHOLDER`` / ``KMS_PLACEHOLDER``
# resource values are substituted at runtime with the caller's actual resources.
IAM_POLICY_CONFIG = {
    "training": {
        "role_name": "SageMaker-AutoRole-Training",
        "trust_policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        },
        "policies": {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:ListBucket",
                            "s3:GetBucketLocation",
                        ],
                        "Resource": "S3_PLACEHOLDER",
                    }
                ],
            },
            "ecr_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ecr:GetDownloadUrlForLayer",
                            "ecr:BatchGetImage",
                            "ecr:BatchCheckLayerAvailability",
                            "ecr:GetAuthorizationToken",
                        ],
                        "Resource": "*",
                    }
                ],
            },
            "cloudwatch_logs_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                            "logs:DescribeLogStreams",
                        ],
                        "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/TrainingJobs*",
                    }
                ],
            },
            "cloudwatch_metric_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["cloudwatch:PutMetricData"],
                        "Resource": "*",
                    }
                ],
            },
            "ec2_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ec2:CreateNetworkInterface",
                            "ec2:CreateNetworkInterfacePermission",
                            "ec2:DeleteNetworkInterface",
                            "ec2:DeleteNetworkInterfacePermission",
                            "ec2:DescribeNetworkInterfaces",
                            "ec2:DescribeVpcs",
                            "ec2:DescribeDhcpOptions",
                            "ec2:DescribeSubnets",
                            "ec2:DescribeSecurityGroups",
                        ],
                        "Resource": "*",
                    }
                ],
            },
            "kms_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"],
                        "Resource": "KMS_PLACEHOLDER",
                    }
                ],
            },
        },
    },
    "serving": {
        "role_name": "SageMaker-AutoRole-Serving",
        "trust_policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        },
        "policies": {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
                        "Resource": "S3_PLACEHOLDER",
                    }
                ],
            },
            "ecr_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ecr:GetDownloadUrlForLayer",
                            "ecr:BatchGetImage",
                            "ecr:BatchCheckLayerAvailability",
                            "ecr:GetAuthorizationToken",
                        ],
                        "Resource": "*",
                    }
                ],
            },
            "cloudwatch_logs_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                        ],
                        "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
                    }
                ],
            },
            "model_package_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["sagemaker:AccessModelPackage"],
                        "Resource": "arn:aws:sagemaker:*:*:model-package/*",
                    }
                ],
            },
        },
    },
    "pipeline": {
        "role_name": "SageMaker-AutoRole-Pipeline",
        "trust_policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        },
        "policies": {
            "sagemaker_policy": {
                "Version": "2012-10-17",
                # Scoped to the specific SageMaker resource types these actions
                # operate on (rather than "*") to keep the role least-privilege.
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:CreateTrainingJob",
                            "sagemaker:DescribeTrainingJob",
                            "sagemaker:StopTrainingJob",
                            "sagemaker:CreateModel",
                            "sagemaker:CreateEndpoint",
                            "sagemaker:CreateEndpointConfig",
                            "sagemaker:CreateProcessingJob",
                            "sagemaker:CreateTransformJob",
                            "sagemaker:DescribeEndpoint",
                            "sagemaker:DescribeModel",
                            # Required dependent action for the Create* calls above,
                            # which tag the resources they create.
                            "sagemaker:AddTags",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:training-job/*",
                            "arn:aws:sagemaker:*:*:processing-job/*",
                            "arn:aws:sagemaker:*:*:transform-job/*",
                            "arn:aws:sagemaker:*:*:model/*",
                            "arn:aws:sagemaker:*:*:endpoint/*",
                            "arn:aws:sagemaker:*:*:endpoint-config/*",
                        ],
                    }
                ],
            },
            "iam_passrole_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["iam:PassRole"],
                        # Scoped at runtime to roles in the caller's account; the
                        # PassedToService condition further restricts to SageMaker.
                        "Resource": "IAM_PASSROLE_PLACEHOLDER",
                        "Condition": {
                            "StringEquals": {
                                "iam:PassedToService": "sagemaker.amazonaws.com"
                            }
                        },
                    }
                ],
            },
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                        "Resource": "S3_PLACEHOLDER",
                    }
                ],
            },
            "cloudwatch_logs_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                        ],
                        "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
                    }
                ],
            },
        },
    },
    "hyperpod": {
        "role_name": "SageMaker-AutoRole-HyperPod",
        # This is the *job execution role* that the HyperPod training job assumes
        # while running ON the cluster, so it is trusted by the SageMaker service
        # principal — like the "training" role. It carries only job-runtime
        # permissions (S3/ECR/CloudWatch/KMS). It intentionally does NOT carry the
        # cluster-connect permissions (sagemaker:DescribeCluster / eks:*): those are
        # needed by the *caller* who runs the HyperPod CLI locally, not by the job
        # on the cluster. See HYPERPOD_CLI_CONNECT_ACTIONS in iam_role_resolver and
        # verify_hyperpod_connect_permissions() for how the caller side is handled.
        # Compared to "training" it omits the EC2 ENI block, which is for VPC-mode
        # SageMaker training jobs and does not apply to HyperPod cluster jobs.
        "trust_policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        },
        "policies": {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:ListBucket",
                            "s3:GetBucketLocation",
                        ],
                        "Resource": "S3_PLACEHOLDER",
                    }
                ],
            },
            "ecr_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ecr:GetDownloadUrlForLayer",
                            "ecr:BatchGetImage",
                            "ecr:BatchCheckLayerAvailability",
                            "ecr:GetAuthorizationToken",
                        ],
                        "Resource": "*",
                    }
                ],
            },
            "cloudwatch_logs_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                            "logs:DescribeLogStreams",
                        ],
                        "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
                    }
                ],
            },
            "cloudwatch_metric_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["cloudwatch:PutMetricData"],
                        "Resource": "*",
                    }
                ],
            },
            "kms_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"],
                        "Resource": "KMS_PLACEHOLDER",
                    }
                ],
            },
        },
    },
    "bedrock": {
        "role_name": "SageMaker-AutoRole-Bedrock",
        "trust_policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        },
        "policies": {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket",
                            "s3:GetBucketLocation",
                        ],
                        "Resource": "S3_PLACEHOLDER",
                    }
                ],
            },
            "kms_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["kms:Decrypt", "kms:GenerateDataKey"],
                        "Resource": "KMS_PLACEHOLDER",
                    }
                ],
            },
            "model_package_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["sagemaker:AccessModelPackage"],
                        "Resource": "arn:aws:sagemaker:*:*:model-package/*",
                    }
                ],
            },
        },
    },
    "feature_store": {
        "role_name": "SageMaker-AutoRole-FeatureStore",
        "trust_policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": [
                            "sagemaker.amazonaws.com",
                            "scheduler.amazonaws.com",
                        ]
                    },
                    "Action": "sts:AssumeRole",
                }
            ],
        },
        "policies": {
            "featurestore_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:PutRecord",
                            "sagemaker:GetRecord",
                            "sagemaker:BatchGetRecord",
                            "sagemaker:DescribeFeatureGroup",
                        ],
                        "Resource": "arn:aws:sagemaker:*:*:feature-group/*",
                    }
                ],
            },
            "pipeline_execution_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["sagemaker:StartPipelineExecution"],
                        "Resource": "arn:aws:sagemaker:*:*:pipeline/*",
                    }
                ],
            },
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:GetBucketLocation"],
                        "Resource": "S3_PLACEHOLDER",
                    }
                ],
            },
            "glue_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "glue:GetTable",
                            "glue:GetDatabase",
                            "glue:GetPartitions",
                        ],
                        "Resource": "*",
                    }
                ],
            },
        },
    },
}
