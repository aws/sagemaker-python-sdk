# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import collections
import logging
import os
from os import path
import stat
import tempfile
import time
import uuid

from botocore.exceptions import ClientError
from fabric import Connection

from tests.integ.vpc_test_utils import check_or_create_vpc_resources_efs_fsx

VPC_NAME = "sagemaker-efs-fsx-vpc"
EFS_CREATION_TOKEN = str(uuid.uuid4())
PREFIX = "ec2_fs_key_"
KEY_NAME = PREFIX + str(uuid.uuid4().hex.upper()[0:8])
ROLE_NAME = "SageMakerRole"
REGION = "us-west-2"
EC2_INSTANCE_TYPE = "t2.micro"
AMI_ID = "ami-082b5a644766e0e6f"
MIN_COUNT = 1
MAX_COUNT = 1
TIME_SLEEP_DURATION = 10

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
MNIST_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "tensorflow_mnist")
MNIST_LOCAL_DATA = os.path.join(MNIST_RESOURCE_PATH, "data")
ONE_P_RESOURCE_PATH = os.path.join(RESOURCE_PATH, "protobuf_data")
ONE_P_LOCAL_DATA = os.path.join(ONE_P_RESOURCE_PATH, "matrix_0.pbr")

SCRIPTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "scripts")
FS_MOUNT_SCRIPT = os.path.join(SCRIPTS_FOLDER, "fs_mount_setup.sh")
FILE_NAME = KEY_NAME + ".pem"
KEY_PATH = os.path.join(tempfile.gettempdir(), FILE_NAME)
STORAGE_CAPACITY_IN_BYTES = 3600

FsResources = collections.namedtuple(
    "FsResources",
    [
        "key_name",
        "key_path",
        "role_name",
        "subnet_id",
        "security_group_ids",
        "file_system_efs_id",
        "file_system_fsx_id",
        "ec2_instance_id",
        "mount_efs_target_id",
    ],
)


def set_up_efs_fsx(sagemaker_session):
    _check_or_create_key_pair(sagemaker_session)
    _check_or_create_iam_profile_and_attach_role(sagemaker_session)
    subnet_ids, security_group_ids = check_or_create_vpc_resources_efs_fsx(
        sagemaker_session, REGION, VPC_NAME
    )

    ec2_instance = _create_ec2_instance(
        sagemaker_session,
        AMI_ID,
        EC2_INSTANCE_TYPE,
        KEY_NAME,
        MIN_COUNT,
        MAX_COUNT,
        security_group_ids,
        subnet_ids[0],
    )

    file_system_efs_id = _check_or_create_efs(sagemaker_session)
    mount_efs_target_id = _create_efs_mount(sagemaker_session, file_system_efs_id)

    file_system_fsx_id = _check_or_create_fsx(sagemaker_session)

    fs_resources = FsResources(
        KEY_NAME,
        KEY_PATH,
        ROLE_NAME,
        subnet_ids[0],
        security_group_ids,
        file_system_efs_id,
        file_system_fsx_id,
        ec2_instance.id,
        mount_efs_target_id,
    )

    try:
        connected_instance = _connect_ec2_instance(ec2_instance)
        _upload_data_and_mount_fs(connected_instance, file_system_efs_id, file_system_fsx_id)
    except Exception:
        tear_down(sagemaker_session, fs_resources)
        raise

    return fs_resources


def _connect_ec2_instance(ec2_instance):
    public_ip_address = ec2_instance.public_ip_address
    connected_instance = Connection(
        host=public_ip_address, port=22, user="ec2-user", connect_kwargs={"key_filename": KEY_PATH}
    )
    return connected_instance


def _upload_data_and_mount_fs(connected_instance, file_system_efs_id, file_system_fsx_id):
    connected_instance.put(FS_MOUNT_SCRIPT, ".")
    connected_instance.run("mkdir temp_tf; mkdir temp_one_p", in_stream=False)
    for dir_name, subdir_list, file_list in os.walk(MNIST_LOCAL_DATA):
        for fname in file_list:
            local_file = os.path.join(MNIST_LOCAL_DATA, fname)
            connected_instance.put(local_file, "temp_tf/")
    connected_instance.put(ONE_P_LOCAL_DATA, "temp_one_p/")
    connected_instance.run(
        "sudo sh fs_mount_setup.sh {} {}".format(file_system_efs_id, file_system_fsx_id),
        in_stream=False,
    )


def _check_or_create_efs(sagemaker_session):
    efs_client = sagemaker_session.boto_session.client("efs")
    file_system_exists = False
    efs_id = ""
    try:
        create_response = efs_client.create_file_system(CreationToken=EFS_CREATION_TOKEN)
        efs_id = create_response["FileSystemId"]
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "FileSystemAlreadyExists":
            file_system_exists = True
            logging.warning(
                "File system with given creation token %s already exists", EFS_CREATION_TOKEN
            )
        else:
            raise

    if file_system_exists:
        desc = efs_client.describe_file_systems(CreationToken=EFS_CREATION_TOKEN)
        efs_id = desc["FileSystems"][0]["FileSystemId"]
        mount_target_id = efs_client.describe_mount_targets(FileSystemId=efs_id)["MountTargets"][0][
            "MountTargetId"
        ]
        return efs_id, mount_target_id

    for _ in retries(50, "Checking EFS creating status"):
        desc = efs_client.describe_file_systems(CreationToken=EFS_CREATION_TOKEN)
        status = desc["FileSystems"][0]["LifeCycleState"]
        if status == "available":
            break

    return efs_id


def _create_efs_mount(sagemaker_session, file_system_id):
    subnet_ids, security_group_ids = check_or_create_vpc_resources_efs_fsx(
        sagemaker_session, REGION, VPC_NAME
    )
    efs_client = sagemaker_session.boto_session.client("efs")
    mount_response = efs_client.create_mount_target(
        FileSystemId=file_system_id, SubnetId=subnet_ids[0], SecurityGroups=security_group_ids
    )
    mount_target_id = mount_response["MountTargetId"]

    for _ in retries(50, "Checking EFS mounting target status"):
        desc = efs_client.describe_mount_targets(MountTargetId=mount_target_id)
        status = desc["MountTargets"][0]["LifeCycleState"]
        if status == "available":
            break

    return mount_target_id


def _check_or_create_fsx(sagemaker_session):
    fsx_client = sagemaker_session.boto_session.client("fsx")
    subnet_ids, security_group_ids = check_or_create_vpc_resources_efs_fsx(
        sagemaker_session, REGION, VPC_NAME
    )
    create_response = fsx_client.create_file_system(
        FileSystemType="LUSTRE",
        StorageCapacity=STORAGE_CAPACITY_IN_BYTES,
        SubnetIds=[subnet_ids[0]],
        SecurityGroupIds=security_group_ids,
    )
    fsx_id = create_response["FileSystem"]["FileSystemId"]

    for _ in retries(50, "Checking FSX creating status"):
        desc = fsx_client.describe_file_systems(FileSystemIds=[fsx_id])
        status = desc["FileSystems"][0]["Lifecycle"]
        if status == "AVAILABLE":
            break

    return fsx_id


def _create_ec2_instance(
    sagemaker_session,
    image_id,
    instance_type,
    key_name,
    min_count,
    max_count,
    security_group_ids,
    subnet_id,
):
    ec2_resource = sagemaker_session.boto_session.resource("ec2")
    ec2_instances = ec2_resource.create_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        KeyName=key_name,
        MinCount=min_count,
        MaxCount=max_count,
        IamInstanceProfile={"Name": ROLE_NAME},
        DryRun=False,
        NetworkInterfaces=[
            {
                "SubnetId": subnet_id,
                "DeviceIndex": 0,
                "AssociatePublicIpAddress": True,
                "Groups": security_group_ids,
            }
        ],
    )

    ec2_instances[0].wait_until_running()
    ec2_instances[0].reload()
    ec2_client = sagemaker_session.boto_session.client("ec2")

    for _ in retries(30, "Checking EC2 creation status"):
        statuses = ec2_client.describe_instance_status(InstanceIds=[ec2_instances[0].id])
        status = statuses["InstanceStatuses"][0]
        if status["InstanceStatus"]["Status"] == "ok" and status["SystemStatus"]["Status"] == "ok":
            break
    return ec2_instances[0]


def _check_key_pair_and_cleanup_old_artifacts(sagemaker_session):
    ec2_client = sagemaker_session.boto_session.client("ec2")
    response = ec2_client.describe_key_pairs(Filters=[{"Name": "key-name", "Values": [KEY_NAME]}])
    if len(response["KeyPairs"]) > 0 and not path.exists(KEY_PATH):
        ec2_client.delete_key_pair(KeyName=KEY_NAME)
    if len(response["KeyPairs"]) == 0 and path.exists(KEY_PATH):
        os.remove(KEY_PATH)
    return len(response["KeyPairs"]) > 0 and path.exists(KEY_PATH)


def _check_or_create_key_pair(sagemaker_session):
    if _check_key_pair_and_cleanup_old_artifacts(sagemaker_session):
        return
    ec2_client = sagemaker_session.boto_session.client("ec2")
    key_pair = ec2_client.create_key_pair(KeyName=KEY_NAME)
    with open(KEY_PATH, "w") as file:
        file.write(key_pair["KeyMaterial"])
    fd = os.open(KEY_PATH, os.O_RDONLY)
    os.fchmod(fd, stat.S_IREAD)


def _delete_key_pair(sagemaker_session):
    ec2_client = sagemaker_session.boto_session.client("ec2")
    ec2_client.delete_key_pair(KeyName=KEY_NAME)
    os.remove(KEY_PATH)


def _terminate_instance(ec2_resource, instance_ids):
    ec2_resource.instances.filter(InstanceIds=instance_ids).terminate()


def _check_or_create_iam_profile_and_attach_role(sagemaker_session):
    if _instance_profile_exists(sagemaker_session):
        return
    iam_client = sagemaker_session.boto_session.client("iam")
    iam_client.create_instance_profile(InstanceProfileName=ROLE_NAME)
    iam_client.add_role_to_instance_profile(InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME)

    for _ in retries(30, "Checking EC2 instance profile creating status"):
        profile_info = iam_client.get_instance_profile(InstanceProfileName=ROLE_NAME)
        if profile_info["InstanceProfile"]["Roles"][0]["RoleName"] == ROLE_NAME:
            break


def _instance_profile_exists(sagemaker_session):
    iam = sagemaker_session.boto_session.client("iam")
    try:
        iam.get_instance_profile(InstanceProfileName=ROLE_NAME)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        message = e.response["Error"]["Message"]
        if error_code == "NoSuchEntity" and ROLE_NAME in message:
            return False
        else:
            raise
    return True


def retries(max_retry_count, exception_message_prefix):
    current_retry_count = 0
    while current_retry_count <= max_retry_count:
        yield current_retry_count

        current_retry_count += 1
        time.sleep(TIME_SLEEP_DURATION)

    raise Exception(
        "{} has reached the maximum retry count {}".format(
            exception_message_prefix, max_retry_count
        )
    )


def tear_down(sagemaker_session, fs_resources):
    fsx_client = sagemaker_session.boto_session.client("fsx")
    file_system_fsx_id = fs_resources.file_system_fsx_id
    fsx_client.delete_file_system(FileSystemId=file_system_fsx_id)

    efs_client = sagemaker_session.boto_session.client("efs")
    mount_efs_target_id = fs_resources.mount_efs_target_id
    efs_client.delete_mount_target(MountTargetId=mount_efs_target_id)

    file_system_efs_id = fs_resources.file_system_efs_id
    for _ in retries(30, "Checking mount target deleting status"):
        desc = efs_client.describe_mount_targets(FileSystemId=file_system_efs_id)
        if len(desc["MountTargets"]) > 0:
            status = desc["MountTargets"][0]["LifeCycleState"]
            if status == "deleted":
                break
        else:
            break

    efs_client.delete_file_system(FileSystemId=file_system_efs_id)

    ec2_resource = sagemaker_session.boto_session.resource("ec2")
    instance_id = fs_resources.ec2_instance_id
    _terminate_instance(ec2_resource, [instance_id])

    _delete_key_pair(sagemaker_session)
