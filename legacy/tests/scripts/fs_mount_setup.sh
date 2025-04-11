#!/bin/bash
#
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
#
# Mounting EFS and FSx for Lustre file systems for integration Tests
FILE_SYSTEM_EFS_ID=$1
FILE_SYSTEM_FSX_ID=$2
REGION=$3
EFS_MOUNT_DIRECTORY=$4
FSX_MOUNT_DIRECTORY=$5

echo "Mounting EFS File Systems"
sudo yum install -y amazon-efs-utils
if [ ! -d $EFS_MOUNT_DIRECTORY ]
then
    echo "$EFS_MOUNT_DIRECTORY does not exist. Creating.."
    sudo mkdir "$EFS_MOUNT_DIRECTORY"
fi

if grep -qs $EFS_MOUNT_DIRECTORY /proc/mounts
then
    echo "$EFS_MOUNT_DIRECTORY already mounted."
else
    sudo mount -t efs "$FILE_SYSTEM_EFS_ID":/ efs
    if [ $? -eq 0 ]
    then
        echo "Successfully mounted $FILE_SYSTEM_EFS_ID to $EFS_MOUNT_DIRECTORY."
    else
        echo "Something went wrong. Could not mount $FILE_SYSTEM_EFS_ID to $EFS_MOUNT_DIRECTORY."
        exit -1
    fi
fi

echo "Creating subfolders for training data from EFS"
if [ ! -d $EFS_MOUNT_DIRECTORY/tensorflow ]
then
    echo "$EFS_MOUNT_DIRECTORY/tensorflow does not exist. Creating.."
    sudo mkdir $EFS_MOUNT_DIRECTORY/tensorflow
fi

if [ ! -d $EFS_MOUNT_DIRECTORY/one_p_mnist ]
then
    echo "$EFS_MOUNT_DIRECTORY/one_p_mnist does not exist. Creating.."
    sudo mkdir $EFS_MOUNT_DIRECTORY/one_p_mnist
fi

echo "Mounting FSx for Lustre File System"
sudo yum install -y lustre-client
if [ ! -d $FSX_MOUNT_DIRECTORY ]
then
    echo "$FSX_MOUNT_DIRECTORY does not exist. Creating.."
    sudo mkdir -p $FSX_MOUNT_DIRECTORY
fi

if grep -qs $FSX_MOUNT_DIRECTORY /proc/mounts;
then
    echo "$FSX_MOUNT_DIRECTORY already mounted."
else
    sudo mount -t lustre -o noatime,flock "$FILE_SYSTEM_FSX_ID".fsx."$REGION".amazonaws.com@tcp:/fsx $FSX_MOUNT_DIRECTORY
    if [ $? -eq 0 ]
    then
        echo "Successfully mounted $FILE_SYSTEM_FSX_ID to $FSX_MOUNT_DIRECTORY."
    else
        echo "Something went wrong. Could not mount $FILE_SYSTEM_FSX_ID to $FSX_MOUNT_DIRECTORY."
        exit -1
    fi
fi

echo "Creating subfolders for training data from Lustre file systems"
if [ ! -d $FSX_MOUNT_DIRECTORY/tensorflow ]
then
    echo "$FSX_MOUNT_DIRECTORY/tensorflow does not exist. Creating.."
    sudo mkdir $FSX_MOUNT_DIRECTORY/tensorflow
fi

if [ ! -d $FSX_MOUNT_DIRECTORY/one_p_mnist ]
then
    echo "$FSX_MOUNT_DIRECTORY/one_p_mnist does not exist. Creating.."
    sudo mkdir $FSX_MOUNT_DIRECTORY/one_p_mnist
fi

echo "Copying files to the mounted folders"
sudo cp temp_tf/* efs/tensorflow
sudo cp temp_tf/* /mnt/fsx/tensorflow
sudo cp temp_one_p/* efs/one_p_mnist/
sudo cp temp_one_p/* /mnt/fsx/one_p_mnist
