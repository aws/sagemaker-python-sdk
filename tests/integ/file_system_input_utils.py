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

from operator import itemgetter
import os
from os import path
import stat
import tempfile
import uuid

from botocore.exceptions import ClientError
from fabric import Connection

from tests.integ.retry import retries
from tests.integ.vpc_test_utils import check_or_create_vpc_resources_efs_fsx

VPC_NAME = "sagemaker-efs-fsx-vpc"
ALINUX_AMI_NAME_FILTER = "amzn-ami-hvm-????.??.?.????????-x86_64-gp2"
EFS_CREATION_TOKEN = str(uuid.uuid4())
PREFIX = "ec2_fs_key_"
KEY_NAME = PREFIX + str(uuid.uuid4().hex.upper()[0:8])
ROLE_NAME = "SageMakerRole"
MIN_COUNT = 1
MAX_COUNT = 1

EFS_MOUNT_DIRECTORY = "efs"
FSX_MOUNT_DIRECTORY = "/mnt/fsx"

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

fs_resources = {"key_name": KEY_NAME, "key_path": KEY_PATH, "role_name": ROLE_NAME}


def set_up_efs_fsx(sagemaker_session, ec2_instance_type):
    try:
        _check_or_create_key_pair(sagemaker_session)
        _check_or_create_iam_profile_and_attach_role(sagemaker_session)

        subnet_ids, security_group_ids = check_or_create_vpc_resources_efs_fsx(
            sagemaker_session, VPC_NAME
        )
        fs_resources["subnet_id"] = subnet_ids[0]
        fs_resources["security_group_ids"] = security_group_ids

        ami_id = _ami_id_for_region(sagemaker_session)
        ec2_instance = _create_ec2_instance(
            sagemaker_session,
            ami_id,
            ec2_instance_type,
            KEY_NAME,
            MIN_COUNT,
            MAX_COUNT,
            security_group_ids,
            subnet_ids[0],
        )

        file_system_efs_id, mount_efs_target_id = _create_efs(sagemaker_session)
        file_system_fsx_id = _create_fsx(sagemaker_session)

        connected_instance = _connect_ec2_instance(ec2_instance)
        region = sagemaker_session.boto_region_name
        _upload_data_and_mount_fs(
            connected_instance, file_system_efs_id, file_system_fsx_id, region
        )
        return fs_resources
    except Exception:
        tear_down(sagemaker_session, fs_resources)
        raise


def _ami_id_for_region(sagemaker_session):
    ec2_client = sagemaker_session.boto_session.client("ec2")
    filters = [
        {"Name": "name", "Values": [ALINUX_AMI_NAME_FILTER]},
        {"Name": "state", "Values": ["available"]},
    ]
    response = ec2_client.describe_images(Filters=filters)
    image_details = sorted(response["Images"], key=itemgetter("CreationDate"), reverse=True)

    if len(image_details) == 0:
        raise Exception("AMI was not found based on current search criteria: {}".format(filters))

    return image_details[0]["ImageId"]


def _connect_ec2_instance(ec2_instance):
    public_ip_address = ec2_instance.public_ip_address
    connected_instance = Connection(
        host=public_ip_address,
        port=22,
        user="ec2-user",
        connect_kwargs={"key_filename": [KEY_PATH]},
    )
    return connected_instance


def _upload_data_and_mount_fs(connected_instance, file_system_efs_id, file_system_fsx_id, region):
    connected_instance.put(FS_MOUNT_SCRIPT, ".")
    connected_instance.run("mkdir temp_tf; mkdir temp_one_p", in_stream=False)
    for dir_name, subdir_list, file_list in os.walk(MNIST_LOCAL_DATA):
        for fname in file_list:
            local_file = os.path.join(MNIST_LOCAL_DATA, fname)
            connected_instance.put(local_file, "temp_tf/")
    connected_instance.put(ONE_P_LOCAL_DATA, "temp_one_p/")
    connected_instance.run(
        "sudo sh fs_mount_setup.sh {} {} {} {} {}".format(
            file_system_efs_id, file_system_fsx_id, region, EFS_MOUNT_DIRECTORY, FSX_MOUNT_DIRECTORY
        ),
        in_stream=False,
    )


def _create_efs(sagemaker_session):
    efs_client = sagemaker_session.boto_session.client("efs")
    create_response = efs_client.create_file_system(CreationToken=EFS_CREATION_TOKEN)
    efs_id = create_response["FileSystemId"]
    fs_resources["file_system_efs_id"] = efs_id
    for _ in retries(50, "Checking EFS creating status"):
        desc = efs_client.describe_file_systems(CreationToken=EFS_CREATION_TOKEN)
        status = desc["FileSystems"][0]["LifeCycleState"]
        if status == "available":
            break
    mount_target_id = _create_efs_mount(sagemaker_session, efs_id)

    return efs_id, mount_target_id


def _create_efs_mount(sagemaker_session, file_system_id):
    subnet_ids, security_group_ids = check_or_create_vpc_resources_efs_fsx(
        sagemaker_session, VPC_NAME
    )
    efs_client = sagemaker_session.boto_session.client("efs")
    mount_response = efs_client.create_mount_target(
        FileSystemId=file_system_id, SubnetId=subnet_ids[0], SecurityGroups=security_group_ids
    )
    mount_target_id = mount_response["MountTargetId"]
    fs_resources["mount_efs_target_id"] = mount_target_id

    for _ in retries(50, "Checking EFS mounting target status"):
        desc = efs_client.describe_mount_targets(MountTargetId=mount_target_id)
        status = desc["MountTargets"][0]["LifeCycleState"]
        if status == "available":
            break

    return mount_target_id


def _create_fsx(sagemaker_session):
    fsx_client = sagemaker_session.boto_session.client("fsx")
    subnet_ids, security_group_ids = check_or_create_vpc_resources_efs_fsx(
        sagemaker_session, VPC_NAME
    )
    create_response = fsx_client.create_file_system(
        FileSystemType="LUSTRE",
        StorageCapacity=STORAGE_CAPACITY_IN_BYTES,
        SubnetIds=[subnet_ids[0]],
        SecurityGroupIds=security_group_ids,
    )
    fsx_id = create_response["FileSystem"]["FileSystemId"]
    fs_resources["file_system_fsx_id"] = fsx_id

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
    fs_resources["ec2_instance_id"] = ec2_instances[0].id
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


def tear_down(sagemaker_session, fs_resources={}):
    try:
        if "file_system_fsx_id" in fs_resources:
            fsx_client = sagemaker_session.boto_session.client("fsx")
            fsx_client.delete_file_system(FileSystemId=fs_resources["file_system_fsx_id"])

        efs_client = sagemaker_session.boto_session.client("efs")
        if "mount_efs_target_id" in fs_resources:
            efs_client.delete_mount_target(MountTargetId=fs_resources["mount_efs_target_id"])

        if "file_system_efs_id" in fs_resources:
            for _ in retries(30, "Checking mount target deleting status"):
                desc = efs_client.describe_mount_targets(
                    FileSystemId=fs_resources["file_system_efs_id"]
                )
                if len(desc["MountTargets"]) > 0:
                    status = desc["MountTargets"][0]["LifeCycleState"]
                    if status == "deleted":
                        break
                else:
                    break

            efs_client.delete_file_system(FileSystemId=fs_resources["file_system_efs_id"])

        if "ec2_instance_id" in fs_resources:
            ec2_resource = sagemaker_session.boto_session.resource("ec2")
            _terminate_instance(ec2_resource, [fs_resources["ec2_instance_id"]])

        _delete_key_pair(sagemaker_session)

    except Exception:
        pass
