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

import os
import tempfile

import tests.integ.lock as lock

VPC_NAME = "sagemaker-python-sdk-test-vpc"
LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_test_vpc_lock")
LOCK_PATH_EFS = os.path.join(tempfile.gettempdir(), "sagemaker_efs_fsx_vpc_lock")


def _get_subnet_ids_by_name(ec2_client, name):
    desc = ec2_client.describe_subnets(Filters=[{"Name": "tag-value", "Values": [name]}])
    if len(desc["Subnets"]) == 0:
        return None
    else:
        return [subnet["SubnetId"] for subnet in desc["Subnets"]]


def _get_security_id_by_name(ec2_client, name):
    desc = ec2_client.describe_security_groups(Filters=[{"Name": "tag-value", "Values": [name]}])
    if len(desc["SecurityGroups"]) == 0:
        return None
    else:
        return desc["SecurityGroups"][0]["GroupId"]


def _security_group_ids_by_vpc_id(sagemaker_session, vpc_id):
    ec2_resource = sagemaker_session.boto_session.resource("ec2")
    security_group_ids = []
    vpc = ec2_resource.Vpc(vpc_id)
    for sg in vpc.security_groups.all():
        security_group_ids.append(sg.id)
    return security_group_ids


def _vpc_exists(ec2_client, name):
    desc = ec2_client.describe_vpcs(Filters=[{"Name": "tag-value", "Values": [name]}])
    return len(desc["Vpcs"]) > 0


def _vpc_id_by_name(ec2_client, name):
    desc = ec2_client.describe_vpcs(Filters=[{"Name": "tag-value", "Values": [name]}])
    vpc_id = desc["Vpcs"][0]["VpcId"]
    return vpc_id


def _route_table_id(ec2_client, vpc_id):
    desc = ec2_client.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    return desc["RouteTables"][0]["RouteTableId"]


def check_or_create_vpc_resources_efs_fsx(sagemaker_session, name=VPC_NAME):
    # use lock to prevent race condition when tests are running concurrently
    with lock.lock(LOCK_PATH_EFS):
        ec2_client = sagemaker_session.boto_session.client("ec2")

        if _vpc_exists(ec2_client, name):
            vpc_id = _vpc_id_by_name(ec2_client, name)
            return (
                _get_subnet_ids_by_name(ec2_client, name),
                _security_group_ids_by_vpc_id(sagemaker_session, vpc_id),
            )
        else:
            return _create_vpc_with_name_efs_fsx(ec2_client, name)


def _create_vpc_with_name_efs_fsx(ec2_client, name):
    vpc_id, [subnet_id_a, subnet_id_b], security_group_id = _create_vpc_resources(ec2_client, name)
    ec2_client.modify_vpc_attribute(EnableDnsHostnames={"Value": True}, VpcId=vpc_id)

    ig = ec2_client.create_internet_gateway()
    internet_gateway_id = ig["InternetGateway"]["InternetGatewayId"]
    ec2_client.attach_internet_gateway(InternetGatewayId=internet_gateway_id, VpcId=vpc_id)

    route_table_id = _route_table_id(ec2_client, vpc_id)
    ec2_client.create_route(
        DestinationCidrBlock="0.0.0.0/0", GatewayId=internet_gateway_id, RouteTableId=route_table_id
    )
    ec2_client.associate_route_table(RouteTableId=route_table_id, SubnetId=subnet_id_a)
    ec2_client.associate_route_table(RouteTableId=route_table_id, SubnetId=subnet_id_b)

    ec2_client.authorize_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 988,
                "ToPort": 988,
                "UserIdGroupPairs": [{"GroupId": security_group_id}],
            },
            {
                "IpProtocol": "tcp",
                "FromPort": 2049,
                "ToPort": 2049,
                "UserIdGroupPairs": [{"GroupId": security_group_id}],
            },
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "For SSH to EC2"}],
            },
        ],
    )

    return [subnet_id_a], [security_group_id]


def _create_vpc_resources(ec2_client, name):
    vpc_id = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]["VpcId"]
    ec2_client.create_tags(Resources=[vpc_id], Tags=[{"Key": "Name", "Value": name}])

    availability_zone_name = ec2_client.describe_availability_zones()["AvailabilityZones"][0][
        "ZoneName"
    ]

    subnet_id_a = ec2_client.create_subnet(
        CidrBlock="10.0.0.0/24", VpcId=vpc_id, AvailabilityZone=availability_zone_name
    )["Subnet"]["SubnetId"]
    print("created subnet: {}".format(subnet_id_a))
    subnet_id_b = ec2_client.create_subnet(
        CidrBlock="10.0.1.0/24", VpcId=vpc_id, AvailabilityZone=availability_zone_name
    )["Subnet"]["SubnetId"]
    print("created subnet: {}".format(subnet_id_b))

    s3_service = [
        s for s in ec2_client.describe_vpc_endpoint_services()["ServiceNames"] if s.endswith("s3")
    ][0]
    ec2_client.create_vpc_endpoint(
        VpcId=vpc_id, ServiceName=s3_service, RouteTableIds=[_route_table_id(ec2_client, vpc_id)]
    )
    print("created s3 vpc endpoint")

    security_group_id = ec2_client.create_security_group(
        VpcId=vpc_id, GroupName=name, Description=name
    )["GroupId"]
    print("created security group: {}".format(security_group_id))

    # multi-host vpc jobs require communication among hosts
    ec2_client.authorize_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 0,
                "ToPort": 65535,
                "UserIdGroupPairs": [{"GroupId": security_group_id}],
            }
        ],
    )

    ec2_client.create_tags(
        Resources=[subnet_id_a, subnet_id_b, security_group_id],
        Tags=[{"Key": "Name", "Value": name}],
    )
    return vpc_id, [subnet_id_a, subnet_id_b], security_group_id


def _create_vpc_with_name(ec2_client, name):
    vpc_id, [subnet_id_a, subnet_id_b], security_group_id = _create_vpc_resources(ec2_client, name)
    return [subnet_id_a, subnet_id_b], security_group_id


def get_or_create_vpc_resources(ec2_client):
    # use lock to prevent race condition when tests are running concurrently
    with lock.lock(LOCK_PATH):
        if _vpc_exists(ec2_client, VPC_NAME):
            print("using existing vpc: {}".format(VPC_NAME))
            return (
                _get_subnet_ids_by_name(ec2_client, VPC_NAME),
                _get_security_id_by_name(ec2_client, VPC_NAME),
            )
        else:
            print("creating new vpc: {}".format(VPC_NAME))
            return _create_vpc_with_name(ec2_client, VPC_NAME)


def setup_security_group_for_encryption(ec2_client, security_group_id):
    sg_desc = ec2_client.describe_security_groups(GroupIds=[security_group_id])
    ingress_perms = sg_desc["SecurityGroups"][0]["IpPermissions"]
    if len(ingress_perms) == 1:
        ec2_client.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {"IpProtocol": "50", "UserIdGroupPairs": [{"GroupId": security_group_id}]},
                {
                    "IpProtocol": "udp",
                    "FromPort": 500,
                    "ToPort": 500,
                    "UserIdGroupPairs": [{"GroupId": security_group_id}],
                },
            ],
        )
