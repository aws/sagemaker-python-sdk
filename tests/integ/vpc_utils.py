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

def _get_subnet_id_by_name(ec2_client, name):
    desc = ec2_client.describe_subnets(Filters=[
        {'Name': 'tag-value', 'Values': [name]}
    ])
    if len(desc['Subnets']) == 0:
        return None
    else:
        return desc['Subnets'][0]['SubnetId']


def _get_security_id_by_name(ec2_client, name):
    desc = ec2_client.describe_security_groups(Filters=[
        {'Name': 'tag-value', 'Values': [name]}
    ])
    if len(desc['SecurityGroups']) == 0:
        return None
    else:
        return desc['SecurityGroups'][0]['GroupId']


def _vpc_exists(ec2_client, name):
    desc = ec2_client.describe_vpcs(Filters=[
        {'Name': 'tag-value', 'Values': [name]}
    ])
    return len(desc['Vpcs']) > 0


def _get_route_table_id(ec2_client, vpc_id):
    desc = ec2_client.describe_route_tables(Filters=[
        {'Name': 'vpc-id', 'Values': [vpc_id]}
    ])
    return desc['RouteTables'][0]['RouteTableId']


def create_vpc_with_name(ec2_client, name):
    vpc_id = ec2_client.create_vpc(CidrBlock='10.0.0.0/16')['Vpc']['VpcId']

    subnet_id = ec2_client.create_subnet(CidrBlock='10.0.0.0/24', VpcId=vpc_id)['Subnet']['SubnetId']

    s3_service = [s for s in ec2_client.describe_vpc_endpoint_services()['ServiceNames'] if s.endswith('s3')][0]
    ec2_client.create_vpc_endpoint(VpcId=vpc_id, ServiceName=s3_service,
                                   RouteTableIds=[_get_route_table_id(ec2_client, vpc_id)])

    security_group_id = ec2_client.create_security_group(GroupName='TrainingJobTestGroup', Description='Testing',
                                                         VpcId=vpc_id)['GroupId']

    ec2_client.create_tags(Resources=[vpc_id, subnet_id, security_group_id], Tags=[{'Key': 'Name', 'Value': name}])

    return subnet_id, security_group_id


def get_or_create_subnet_and_security_group(ec2_client, name):
    if _vpc_exists(ec2_client, name):
        return _get_subnet_id_by_name(ec2_client, name), _get_security_id_by_name(ec2_client, name)
    else:
        return create_vpc_with_name(ec2_client, name)
