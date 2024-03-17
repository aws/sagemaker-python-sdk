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
from __future__ import absolute_import

import pytest

import time
import json
from boto3 import Session as BotoSession
from sagemaker import get_execution_role
from sagemaker.session import Session
from botocore.exceptions import ClientError
from unittest import SkipTest

@pytest.fixture(scope="module")
def sagemaker_session():
    return Session(boto_session=BotoSession(region_name="us-east-2"))


@pytest.fixture(scope="module")
def sagemaker_client(sagemaker_session):
    return sagemaker_session.sagemaker_client


@pytest.fixture(scope="module")
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture(scope="module")
def domain_exec_role_arn(sagemaker_session):
    iam_client = sagemaker_session.boto_session.client("iam")
    role_name='AmazonDataZoneDomainExecution'
    try:
        exec_role = iam_client.get_role(RoleName=role_name)
        return exec_role["Role"]["Arn"]
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print('Domain execution role not found, creating.')

    test_role_name='AmazonDataZoneDomainExecutionIntegTest'
    try:
        exec_role = iam_client.get_role(RoleName=test_role_name)
        return exec_role["Role"]["Arn"]
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print('Domain execution role not found, creating.')

        assume_role_policy = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                        "Service": [
                            "datazone.amazonaws.com",
                            "datazone.aws.internal"
                        ]},
                    "Action": [
                        "sts:AssumeRole",
                        "sts:TagSession"]
                    }
                ],
            })
        test_role_name='AmazonDataZoneDomainExecutionIntegTest'
        iam_client.create_role(RoleName=test_role_name, AssumeRolePolicyDocument=assume_role_policy)
        iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonDataZoneDomainExecutionRolePolicy",
            RoleName=test_role_name,
        )
        return iam_client.get_role(RoleName=test_role_name)["Role"]["Arn"]

@pytest.fixture(scope="module")
def domain_id(sagemaker_session, domain_exec_role_arn):
    dz_client = sagemaker_session.boto_session.client("datazone")
    iam_client = sagemaker_session.boto_session.client("sts")
    account_id = iam_client.get_caller_identity()["Account"]
    domains = dz_client.list_domains()["items"]
    domain_name = 'SDKIntegTestDomain'
    domain_id = None
    for domain in domains:
        if domain["name"] == domain_name:
            domain_id = domain["id"]
    if domain_id is None:
        create_resp = dz_client.create_domain(name=domain_name, domainExecutionRole=domain_exec_role_arn)
        domain_id = create_resp["id"]
    blueprints = dz_client.list_environment_blueprints(domainIdentifier=domain_id, managed=True)["items"]
    sm_blueprint = None
    for bp in blueprints:
        if bp["name"] == "DefaultSageMaker":
            sm_blueprint = bp
    if sm_blueprint is None:
        raise SkipTest('SageMaker blueprint not available')
    blueprint_id = sm_blueprint["id"]
    sm_bp_config = None
    configs = dz_client.list_environment_blueprint_configurations(domainIdentifier=domain_id)["items"]
    for config in configs:
        if config["environmentBlueprintId"] == blueprint_id:
            sm_bp_config = config
    if sm_bp_config is None:
        manageAccessRoleArn = create_role_if_not_exists(
            sagemaker_session,
            'AmazonDataZoneSageMakerManageAccessRole')
        provisioningRoleArn = create_role_if_not_exists(
            sagemaker_session,
            'AmazonDataZoneSageMakerProvisioningRole',
            get_provisioning_policy())
        dz_client.put_environment_blueprint_configuration(
            domainIdentifier=domain_id,
            environmentBlueprintIdentifier=blueprint_id,
            manageAccessRoleArn=manageAccessRoleArn,
            provisioningRoleArn=provisioningRoleArn,
            enabledRegions=[sagemaker_session.boto_session.region_name]
        )
    return domain_id


@pytest.fixture(scope="module")
def project_id(sagemaker_session, domain_id):
    dz_client = sagemaker_session.boto_session.client("datazone")
    projects = dz_client.list_projects(domainIdentifier=domain_id)["items"]
    if len(projects) == 0:
        project_name = 'SDKIntegTestProject'
        dz_client.create_project(name=project_name, domainIdentifier=domain_id)
        time.sleep(1)
        projects = dz_client.list_projects(domainIdentifier=domain_id)["items"]
        if len(projects) == 0:
            time.sleep(5)
        projects = dz_client.list_projects(domainIdentifier=domain_id)["items"]
    project = projects[0]
    return project["id"]


@pytest.fixture(scope="module")
def environment_id(sagemaker_session, domain_id, project_id):
    dz_client = sagemaker_session.boto_session.client("datazone")
    environments = dz_client.list_environments(domainIdentifier=domain_id, projectIdentifier=project_id)["items"]
    env_profile_id = create_environment_profile(sagemaker_session, domain_id, project_id)
    if len(environments) == 0:
        environment_name = 'SDKIntegTestDummyEnv'
        create_resp = dz_client.create_environment(
            name=environment_name,
            domainIdentifier=domain_id,
            projectIdentifier=project_id,
            environmentProfileIdentifier=env_profile_id)
        environment_id = create_resp["id"]
    else:
        environment_id = environments[0]["id"]

    environment = dz_client.get_environment(
        domainIdentifier=domain_id,
        identifier=environment_id)

    while True:
        environment = dz_client.get_environment(
            domainIdentifier=domain_id,
            identifier=environment_id)
        status = environment["status"]
        if status == "CREATING":
            print(f'Waiting for DataZone environment {environment["name"]} creation completion.')
            time.sleep(30)
        else:
            break

    if status != "ACTIVE":
        raise SkipTest(f"DataZone integration tests will fail due to environment not in ACTIVE status: {status}")

    return environment_id


def create_role_if_not_exists(sagemaker_session, role_name, inline_policy=None):
    iam_client = sagemaker_session.boto_session.client("iam")
    assume_role_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": [
                            "datazone.amazonaws.com",
                            "datazone.aws.internal"
                        ]},
                    "Action": [
                        "sts:AssumeRole",
                        "sts:TagSession"]
                }
            ],
        })
    try:
        exec_role = iam_client.get_role(RoleName=role_name)
        return exec_role["Role"]["Arn"]
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print('Role not found, creating.')
    create_reps = iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=assume_role_policy)
    arn = create_reps["Role"]["Arn"]
    if inline_policy:
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=role_name + '-Policy',
            PolicyDocument=inline_policy)
    return arn


def create_environment_profile(sagemaker_session, domain_id, project_id):
    dz_client = sagemaker_session.boto_session.client("datazone")
    sts_client = sagemaker_session.boto_session.client("sts")
    blueprints = dz_client.list_environment_blueprints(domainIdentifier=domain_id, managed=True)["items"]
    sm_blueprint = None
    for bp in blueprints:
        if bp["name"] == "DefaultSageMaker":
            sm_blueprint = bp
    identity = sts_client.get_caller_identity()
    account_id=identity["Account"]

    sm_profile_name = 'SDKTestSMEnvProfile'
    profiles = dz_client.list_environment_profiles(
        domainIdentifier=domain_id,
        environmentBlueprintIdentifier=sm_blueprint["id"],
        awsAccountId=account_id,
        awsAccountRegion=sagemaker_session.boto_session.region_name,
    )

    for profile in profiles["items"]:
        if profile["name"] == sm_profile_name:
            print(f'Profile {sm_profile_name} already exists.')
            return profile["id"]

    ec2_client = sagemaker_session.boto_session.client("ec2")
    vpcs = ec2_client.describe_vpcs()["Vpcs"]
    vpc = vpcs[0]
    subnets = ec2_client.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc["VpcId"]]}])["Subnets"]
    subnetIds = ','.join([subnets[0]["SubnetId"],subnets[1]["SubnetId"],subnets[2]["SubnetId"]])
    create_resp = dz_client.create_environment_profile(
        domainIdentifier=domain_id,
        projectIdentifier=project_id,
        name=sm_profile_name,
        environmentBlueprintIdentifier=sm_blueprint["id"],
        awsAccountId=account_id,
        awsAccountRegion=sagemaker_session.boto_session.region_name,
        userParameters=[
            {"name": "vpcId", "value": vpc["VpcId"]},
            {"name": "subnetIds", "value": subnetIds},
            {"name": "sagemakerDomainAuthMode", "value": "IAM"},
            {"name": "sagemakerDomainNetworkType", "value": "PublicInternetOnly"},
            {"name": "kmsKeyId", "value": "dummy-key-id"},
        ]
    )
    return create_resp["id"]

def get_provisioning_policy():
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CreateSagemakerStudio",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:CreateDomain",
                    "sagemaker:AddTags"
                ],
                "Resource": [
                    "*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:CalledViaFirst": [
                            "cloudformation.amazonaws.com"
                        ]
                    },
                    "ForAnyValue:StringLike": {
                        "aws:TagKeys": [
                            "AmazonDataZoneEnvironment"
                        ]
                    },
                    "Null": {
                        "aws:TagKeys": "false",
                        "aws:ResourceTag/AmazonDataZoneEnvironment": "false",
                        "aws:RequestTag/AmazonDataZoneEnvironment": "false"
                    }
                }
            },
            {
                "Sid": "DeleteSagemakerStudio",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:DeleteDomain"
                ],
                "Resource": [
                    "*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:CalledViaFirst": [
                            "cloudformation.amazonaws.com"
                        ]
                    },
                    "ForAnyValue:StringLike": {
                        "aws:TagKeys": [
                            "AmazonDataZoneEnvironment"
                        ]
                    },
                    "Null": {
                        "aws:TagKeys": "false",
                        "aws:ResourceTag/AmazonDataZoneEnvironment": "false"
                    }
                }
            },
            {
                "Sid": "AmazonDataZoneEnvironmentSageMakerDescribePermissions",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:DescribeDomain"
                ],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "aws:CalledViaFirst": [
                            "cloudformation.amazonaws.com"
                        ]
                    }
                }
            },
            {
                "Sid": "IamPassRolePermissions",
                "Effect": "Allow",
                "Action": [
                    "iam:PassRole"
                ],
                "Resource": [
                    "arn:aws:iam::*:role/sm-provisioning/datazone_usr*"
                ],
                "Condition": {
                    "StringEquals": {
                        "iam:PassedToService": [
                            "glue.amazonaws.com",
                            "lakeformation.amazonaws.com",
                            "sagemaker.amazonaws.com"
                        ],
                        "aws:CalledViaFirst": [
                            "cloudformation.amazonaws.com"
                        ]
                    }
                }
            },
            {
                "Sid": "AmazonDataZonePermissionsToCreateEnvironmentRole",
                "Effect": "Allow",
                "Action": [
                    "iam:CreateRole",
                    "iam:DetachRolePolicy",
                    "iam:DeleteRolePolicy",
                    "iam:AttachRolePolicy",
                    "iam:PutRolePolicy"
                ],
                "Resource": [
                    "arn:aws:iam::*:role/sm-provisioning/datazone_usr*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:CalledViaFirst": [
                            "cloudformation.amazonaws.com"
                        ],
                        "iam:PermissionsBoundary": "arn:aws:iam::aws:policy/AmazonDataZoneSageMakerEnvironmentRolePermissionsBoundary"
                    }
                }
            },
            {
                "Sid": "AmazonDataZonePermissionsToManageEnvironmentRole",
                "Effect": "Allow",
                "Action": [
                    "iam:GetRole",
                    "iam:GetRolePolicy",
                    "iam:DeleteRole"
                ],
                "Resource": [
                    "arn:aws:iam::*:role/sm-provisioning/datazone_usr*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:CalledViaFirst": [
                            "cloudformation.amazonaws.com"
                        ]
                    }
                }
            },
            {
                "Sid": "AmazonDataZonePermissionsToCreateSageMakerServiceRole",
                "Effect": "Allow",
                "Action": [
                    "iam:CreateServiceLinkedRole"
                ],
                "Resource": [
                    "arn:aws:iam::*:role/aws-service-role/sagemaker.amazonaws.com/AWSServiceRoleForAmazonSageMakerNotebooks"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:CalledViaFirst": [
                            "cloudformation.amazonaws.com"
                        ]
                    }
                }
            },
            {
                "Sid": "AmazonDataZoneEnvironmentParameterValidation",
                "Effect": "Allow",
                "Action": [
                    "ec2:DescribeVpcs",
                    "ec2:DescribeSubnets",
                    "sagemaker:ListDomains"
                ],
                "Resource": "*"
            },
            {
                "Sid": "AmazonDataZoneEnvironmentKMSKeyValidation",
                "Effect": "Allow",
                "Action": [
                    "kms:DescribeKey"
                ],
                "Resource": "*",
                "Condition": {
                    "Null":
                        {
                            "aws:ResourceTag/AmazonDataZoneEnvironment": "false"
                        }
                }
            },
            {
                "Sid": "AmazonDataZoneEnvironmentGluePermissions",
                "Effect": "Allow",
                "Action":
                    [
                        "glue:CreateConnection",
                        "glue:DeleteConnection"
                    ],
                "Resource": [
                    "arn:aws:glue:*:*:connection/dz-sm-athena-glue-connection-*",
                    "arn:aws:glue:*:*:connection/dz-sm-redshift-cluster-connection-*",
                    "arn:aws:glue:*:*:connection/dz-sm-redshift-serverless-connection-*",
                    "arn:aws:glue:*:*:catalog"
                ],
                "Condition":
                    {
                        "StringEquals":
                            {
                                "aws:CalledViaFirst":
                                    [
                                        "cloudformation.amazonaws.com"
                                    ]
                            }
                    }
            }
        ]
    })