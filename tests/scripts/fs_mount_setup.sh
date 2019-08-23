#!/bin/bash
#
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
#
# Mounting EFS and FSx for Lustre file systems for integration Tests
FILE_SYSTEM_EFS_ID=$1
FILE_SYSTEM_FSX_ID=$2

echo "Mounting EFS File Systems"
sudo yum install -y amazon-efs-utils.noarch 0:1.10-1.amzn2
sudo mkdir efs
sudo mount -t efs "$FILE_SYSTEM_EFS_ID":/ efs
sudo mkdir efs/tensorflow
sudo mkdir efs/one_p_mnist

echo "Mounting FSx for Lustre File System"
sudo amazon-linux-extras install -y lustre2.10
sudo mkdir -p /mnt/fsx
sudo mount -t lustre -o noatime,flock "$FILE_SYSTEM_FSX_ID".fsx.us-west-2.amazonaws.com@tcp:/fsx /mnt/fsx
sudo mkdir /mnt/fsx/tensorflow
sudo mkdir /mnt/fsx/one_p_mnist

echo "Copying files to the mounted folders"
sudo cp temp_tf/* efs/tensorflow
sudo cp temp_tf/* /mnt/fsx/tensorflow
sudo cp temp_one_p/* efs/one_p_mnist/
sudo cp temp_one_p/* /mnt/fsx/one_p_mnist
