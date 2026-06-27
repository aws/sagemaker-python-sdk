"""IAM policy configuration for SageMaker role auto-creation.

Defined as a Python constant (rather than a bundled JSON data file) so it is
always packaged with the module — no MANIFEST.in / package_data entry required.
Consumed by :mod:`sagemaker.core.helper.iam_role_resolver`.
"""
from __future__ import absolute_import

# Maps each role type to its trust policy and the least-privilege policies
# attached to the auto-created role. ``S3_PLACEHOLDER`` / ``KMS_PLACEHOLDER``
# resource values are substituted at runtime with the caller's actual resources.
# ``ACCOUNT_PLACEHOLDER`` in trust-policy ``aws:SourceAccount`` conditions is
# substituted with the caller's account ID to prevent the cross-service
# confused-deputy problem (the role can only be assumed on behalf of the
# caller's own account).
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
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": "ACCOUNT_PLACEHOLDER"}
                    },
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
            # Fine-tuning jobs reference a base model via a hub-content ARN
            # (e.g. SageMakerPublicHub/Model/<name>/<version>). The SageMaker
            # service reads that hub content as this execution role when it
            # creates the training job, so the role needs hub access. This
            # requires permissions on BOTH the hub-content resource
            # (DescribeHubContent) AND the parent hub resource itself
            # (DescribeHub / ListHubContents) — granting DescribeHubContent alone
            # still fails with "Access denied to hub content" because the service
            # resolves the hub before reading its content. The public hub lives in
            # the "aws" account, so the account segment is left wildcard to also
            # cover private hubs (SAGEMAKER_HUB_NAME).
            "hub_content_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:DescribeHubContent",
                            "sagemaker:ListHubContents",
                            "sagemaker:ListHubs",
                            "sagemaker:DescribeHub",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:hub/*",
                            "arn:aws:sagemaker:*:*:hub-content/*",
                        ],
                    }
                ],
            },
            # Fine-tuning registers the resulting model into a model package
            # group (the SDK's ``model_package_group`` argument), creating the
            # group if needed and adding a new model package version to it. The
            # service performs these as the execution role during/after the
            # training job, so the role needs model-package-group + model-package
            # CRUD. Scoped to model-package-group/* and model-package/* in the
            # caller's account; AddTags is the dependent action for the Create*
            # calls, which tag the resources they create.
            "model_package_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:CreateModelPackageGroup",
                            "sagemaker:DescribeModelPackageGroup",
                            "sagemaker:CreateModelPackage",
                            "sagemaker:DescribeModelPackage",
                            "sagemaker:ListModelPackages",
                            "sagemaker:UpdateModelPackage",
                            "sagemaker:AddTags",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:model-package-group/*",
                            "arn:aws:sagemaker:*:*:model-package/*",
                        ],
                    }
                ],
            },
            "lambda_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["lambda:InvokeFunction"],
                        "Resource": "arn:aws:lambda:*:*:function:*",
                    }
                ],
            },
            # When the trainer is given an ``mlflow_resource_arn`` (a SageMaker
            # managed MLflow tracking server / mlflow-app), the training/eval job
            # logs to it *as the execution role*. This policy has two statements:
            #
            #   1. Data-plane logging via the ``sagemaker-mlflow`` namespace. Those
            #      REST calls are SigV4-signed by the ``sagemaker-mlflow`` auth
            #      plugin. The action set mirrors the Nova Forge SDK's documented
            #      MLflowSageMaker policy (nova-customization-sdk/docs/iam_setup.md)
            #      so jobs can do the full experiment-tracking + model-registry
            #      workflow (experiments, runs, logged models, registered models,
            #      tags).
            #   2. Control-plane describe via the ``sagemaker`` namespace. The job
            #      resolves the tracking endpoint from the MLflow ARN before
            #      logging — DescribeMlflowApp for the newer mlflow-app and
            #      DescribeMlflowTrackingServer for the classic tracking server.
            #
            # Both statements are scoped to MLflow resources only (mlflow-app/* and
            # mlflow-tracking-server/*). No-op for jobs that do not use MLflow.
            "mlflow_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker-mlflow:AccessUI",
                            "sagemaker-mlflow:CreateExperiment",
                            "sagemaker-mlflow:CreateModelVersion",
                            "sagemaker-mlflow:CreateRegisteredModel",
                            "sagemaker-mlflow:CreateRun",
                            "sagemaker-mlflow:DeleteTag",
                            "sagemaker-mlflow:FinalizeLoggedModel",
                            "sagemaker-mlflow:Get*",
                            "sagemaker-mlflow:ListArtifacts",
                            "sagemaker-mlflow:ListLoggedModelArtifacts",
                            "sagemaker-mlflow:LogBatch",
                            "sagemaker-mlflow:LogInputs",
                            "sagemaker-mlflow:LogLoggedModelParams",
                            "sagemaker-mlflow:LogMetric",
                            "sagemaker-mlflow:LogModel",
                            "sagemaker-mlflow:LogOutputs",
                            "sagemaker-mlflow:LogParam",
                            "sagemaker-mlflow:RenameRegisteredModel",
                            "sagemaker-mlflow:RestoreExperiment",
                            "sagemaker-mlflow:RestoreRun",
                            "sagemaker-mlflow:Search*",
                            "sagemaker-mlflow:SetExperimentTag",
                            "sagemaker-mlflow:SetLoggedModelTags",
                            "sagemaker-mlflow:SetRegisteredModelAlias",
                            "sagemaker-mlflow:SetRegisteredModelTag",
                            "sagemaker-mlflow:SetTag",
                            "sagemaker-mlflow:TransitionModelVersionStage",
                            "sagemaker-mlflow:UpdateExperiment",
                            "sagemaker-mlflow:UpdateModelVersion",
                            "sagemaker-mlflow:UpdateRegisteredModel",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:mlflow-app/*",
                            "arn:aws:sagemaker:*:*:mlflow-tracking-server/*",
                        ],
                    },
                    {
                        # Control-plane describe to resolve the tracking endpoint
                        # from the provided MLflow ARN (app or classic server).
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:DescribeMlflowApp",
                            "sagemaker:DescribeMlflowTrackingServer",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:mlflow-app/*",
                            "arn:aws:sagemaker:*:*:mlflow-tracking-server/*",
                        ],
                    },
                ],
            },
            # Evaluation runs as a SageMaker Pipeline under this execution role and
            # does two things the plain fine-tuning grants above don't cover:
            #   1. Records lineage — it creates an evaluation Action and the
            #      input/output Artifacts (and the associations between them), so it
            #      needs the lineage Create/Describe/Add* actions on action/*,
            #      artifact/* and context/*.
            #   2. Launches the evaluation training job and tags it, so it needs
            #      CreateTrainingJob + AddTags on training-job/* (the existing
            #      model_package AddTags is scoped to model-package* resources only).
            # Scoped to the lineage + training-job resource types these steps touch.
            "evaluation_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:CreateAction",
                            "sagemaker:UpdateAction",
                            "sagemaker:DescribeAction",
                            "sagemaker:DeleteAction",
                            "sagemaker:ListActions",
                            "sagemaker:CreateArtifact",
                            "sagemaker:DescribeArtifact",
                            "sagemaker:UpdateArtifact",
                            "sagemaker:DeleteArtifact",
                            "sagemaker:ListArtifacts",
                            "sagemaker:AddAssociation",
                            "sagemaker:DeleteAssociation",
                            "sagemaker:ListAssociations",
                            "sagemaker:CreateContext",
                            "sagemaker:DescribeContext",
                            "sagemaker:AddTags",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:action/*",
                            "arn:aws:sagemaker:*:*:artifact/*",
                            "arn:aws:sagemaker:*:*:context/*",
                        ],
                    },
                    {
                        # The evaluation pipeline launches and tags a training job.
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:CreateTrainingJob",
                            "sagemaker:DescribeTrainingJob",
                            "sagemaker:StopTrainingJob",
                            "sagemaker:AddTags",
                        ],
                        "Resource": "arn:aws:sagemaker:*:*:training-job/*",
                    },
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
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": "ACCOUNT_PLACEHOLDER"}
                    },
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
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": "ACCOUNT_PLACEHOLDER"}
                    },
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
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": "ACCOUNT_PLACEHOLDER"}
                    },
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
            # Fine-tuning jobs reference a base model via a hub-content ARN
            # (e.g. SageMakerPublicHub/Model/<name>/<version>). The job needs to
            # resolve that base model, which requires access to BOTH the parent
            # hub resource (DescribeHub / ListHubContents) AND the hub-content
            # resource (DescribeHubContent) — DescribeHubContent alone still fails
            # with "Access denied to hub content". Account segment left wildcard to
            # cover the public hub ("aws") and private hubs (SAGEMAKER_HUB_NAME).
            "hub_content_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:DescribeHubContent",
                            "sagemaker:ListHubContents",
                            "sagemaker:ListHubs",
                            "sagemaker:DescribeHub",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:hub/*",
                            "arn:aws:sagemaker:*:*:hub-content/*",
                        ],
                    }
                ],
            },
            # HyperPod fine-tuning jobs log to managed MLflow as the execution role
            # just like SMTJ/serverless jobs: when an ``mlflow_resource_arn`` is
            # provided, the MLflow tracking URI is injected into the HyperPod recipe
            # (see base_trainer._train_hyperpod), and the job logs to it on the
            # cluster as this role. Mirrors the "training" role's mlflow_policy —
            # data-plane logging via ``sagemaker-mlflow`` plus control-plane describe
            # to resolve the tracking endpoint. Scoped to MLflow resources only.
            "mlflow_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker-mlflow:AccessUI",
                            "sagemaker-mlflow:CreateExperiment",
                            "sagemaker-mlflow:CreateModelVersion",
                            "sagemaker-mlflow:CreateRegisteredModel",
                            "sagemaker-mlflow:CreateRun",
                            "sagemaker-mlflow:DeleteTag",
                            "sagemaker-mlflow:FinalizeLoggedModel",
                            "sagemaker-mlflow:Get*",
                            "sagemaker-mlflow:ListArtifacts",
                            "sagemaker-mlflow:ListLoggedModelArtifacts",
                            "sagemaker-mlflow:LogBatch",
                            "sagemaker-mlflow:LogInputs",
                            "sagemaker-mlflow:LogLoggedModelParams",
                            "sagemaker-mlflow:LogMetric",
                            "sagemaker-mlflow:LogModel",
                            "sagemaker-mlflow:LogOutputs",
                            "sagemaker-mlflow:LogParam",
                            "sagemaker-mlflow:RenameRegisteredModel",
                            "sagemaker-mlflow:RestoreExperiment",
                            "sagemaker-mlflow:RestoreRun",
                            "sagemaker-mlflow:Search*",
                            "sagemaker-mlflow:SetExperimentTag",
                            "sagemaker-mlflow:SetLoggedModelTags",
                            "sagemaker-mlflow:SetRegisteredModelAlias",
                            "sagemaker-mlflow:SetRegisteredModelTag",
                            "sagemaker-mlflow:SetTag",
                            "sagemaker-mlflow:TransitionModelVersionStage",
                            "sagemaker-mlflow:UpdateExperiment",
                            "sagemaker-mlflow:UpdateModelVersion",
                            "sagemaker-mlflow:UpdateRegisteredModel",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:mlflow-app/*",
                            "arn:aws:sagemaker:*:*:mlflow-tracking-server/*",
                        ],
                    },
                    {
                        # Control-plane describe to resolve the tracking endpoint
                        # from the provided MLflow ARN (app or classic server).
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:DescribeMlflowApp",
                            "sagemaker:DescribeMlflowTrackingServer",
                        ],
                        "Resource": [
                            "arn:aws:sagemaker:*:*:mlflow-app/*",
                            "arn:aws:sagemaker:*:*:mlflow-tracking-server/*",
                        ],
                    },
                ],
            },
            # RLVR fine-tuning supports a Lambda ``custom_reward_function``, and
            # RLVR runs on HyperPod (HyperPodCompute -> _train_hyperpod). When the
            # reward function is a Lambda ARN, the training job invokes it *as the
            # execution role* during the RL loop, so the role needs
            # lambda:InvokeFunction (mirrors the "training" role). No-op for jobs
            # that do not use a Lambda reward function.
            "lambda_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["lambda:InvokeFunction"],
                        "Resource": "arn:aws:lambda:*:*:function:*",
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
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": "ACCOUNT_PLACEHOLDER"}
                    },
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
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": "ACCOUNT_PLACEHOLDER"}
                    },
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
