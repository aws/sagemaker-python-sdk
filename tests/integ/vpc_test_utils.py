# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


def _vpc_exists(ec2_client, name):
    desc = ec2_client.describe_vpcs(Filters=[{"Name": "tag-value", "Values": [name]}])
    return len(desc["Vpcs"]) > 0


def _get_route_table_id(ec2_client, vpc_id):
    desc = ec2_client.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    return desc["RouteTables"][0]["RouteTableId"]


def _create_vpc_with_name(ec2_client, region, name):
    vpc_id = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]["VpcId"]
    print("created vpc: {}".format(vpc_id))

    # sagemaker endpoints require subnets in at least 2 different AZs for vpc mode
    subnet_id_a = ec2_client.create_subnet(
        CidrBlock="10.0.0.0/24", VpcId=vpc_id, AvailabilityZone=(region + "a")
    )["Subnet"]["SubnetId"]
    print("created subnet: {}".format(subnet_id_a))
    subnet_id_b = ec2_client.create_subnet(
        CidrBlock="10.0.1.0/24", VpcId=vpc_id, AvailabilityZone=(region + "b")
    )["Subnet"]["SubnetId"]
    print("created subnet: {}".format(subnet_id_b))

    s3_service = [
        s for s in ec2_client.describe_vpc_endpoint_services()["ServiceNames"] if s.endswith("s3")
    ][0]
    ec2_client.create_vpc_endpoint(
        VpcId=vpc_id,
        ServiceName=s3_service,
        RouteTableIds=[_get_route_table_id(ec2_client, vpc_id)],
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
        Resources=[vpc_id, subnet_id_a, subnet_id_b, security_group_id],
        Tags=[{"Key": "Name", "Value": name}],
    )

    return [subnet_id_a, subnet_id_b], security_group_id


def get_or_create_vpc_resources(ec2_client, region, name=VPC_NAME):
    # use lock to prevent race condition when tests are running concurrently
    with lock.lock(LOCK_PATH):
        if _vpc_exists(ec2_client, name):
            print("using existing vpc: {}".format(name))
            return (
                _get_subnet_ids_by_name(ec2_client, name),
                _get_security_id_by_name(ec2_client, name),
            )
        else:
            print("creating new vpc: {}".format(name))
            return _create_vpc_with_name(ec2_client, region, name)


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
