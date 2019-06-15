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

import contextlib
import json

from botocore import exceptions

PRINCIPAL_TEMPLATE = '["{account_id}", "{role_arn}", ' \
                     '"arn:aws:iam::{account_id}:role/{sagemaker_role}"] '

KEY_ALIAS = 'SageMakerTestKMSKey'
KMS_S3_ALIAS = 'SageMakerTestS3KMSKey'
POLICY_NAME = 'default'
KEY_POLICY = '''
{{
  "Version": "2012-10-17",
  "Id": "{id}",
  "Statement": [
    {{
      "Sid": "Enable IAM User Permissions",
      "Effect": "Allow",
      "Principal": {{
        "AWS": {principal}
      }},
      "Action": "kms:*",
      "Resource": "*"
    }}
  ]
}}
'''


def _get_kms_key_arn(kms_client, alias):
    try:
        response = kms_client.describe_key(KeyId='alias/' + alias)
        return response['KeyMetadata']['Arn']
    except kms_client.exceptions.NotFoundException:
        return None


def _get_kms_key_id(kms_client, alias):
    try:
        response = kms_client.describe_key(KeyId='alias/' + alias)
        return response['KeyMetadata']['KeyId']
    except kms_client.exceptions.NotFoundException:
        return None


def _create_kms_key(kms_client,
                    account_id,
                    role_arn=None,
                    sagemaker_role='SageMakerRole',
                    alias=KEY_ALIAS):
    if role_arn:
        principal = PRINCIPAL_TEMPLATE.format(account_id=account_id,
                                              role_arn=role_arn,
                                              sagemaker_role=sagemaker_role)
    else:
        principal = '"{account_id}"'.format(account_id=account_id)

    response = kms_client.create_key(
        Policy=KEY_POLICY.format(id=POLICY_NAME, principal=principal, sagemaker_role=sagemaker_role),
        Description='KMS key for SageMaker Python SDK integ tests',
    )
    key_arn = response['KeyMetadata']['Arn']

    if alias:
        kms_client.create_alias(AliasName='alias/' + alias, TargetKeyId=key_arn)
    return key_arn


def _add_role_to_policy(kms_client,
                        account_id,
                        role_arn,
                        alias=KEY_ALIAS,
                        sagemaker_role='SageMakerRole'):
    key_id = _get_kms_key_id(kms_client, alias)
    policy = kms_client.get_key_policy(KeyId=key_id, PolicyName=POLICY_NAME)
    policy = json.loads(policy['Policy'])
    principal = policy['Statement'][0]['Principal']['AWS']

    if role_arn not in principal or sagemaker_role not in principal:
        principal = PRINCIPAL_TEMPLATE.format(account_id=account_id,
                                              role_arn=role_arn,
                                              sagemaker_role=sagemaker_role)

        kms_client.put_key_policy(KeyId=key_id,
                                  PolicyName=POLICY_NAME,
                                  Policy=KEY_POLICY.format(id=POLICY_NAME, principal=principal))


def get_or_create_kms_key(sagemaker_session,
                          role_arn=None,
                          alias=KEY_ALIAS,
                          sagemaker_role='SageMakerRole'):
    kms_client = sagemaker_session.boto_session.client('kms')
    kms_key_arn = _get_kms_key_arn(kms_client, alias)

    sts_client = sagemaker_session.boto_session.client('sts')
    account_id = sts_client.get_caller_identity()['Account']

    if kms_key_arn is None:
        return _create_kms_key(kms_client, account_id, role_arn, sagemaker_role, alias)

    if role_arn:
        _add_role_to_policy(kms_client,
                            account_id,
                            role_arn,
                            alias,
                            sagemaker_role)

    return kms_key_arn


KMS_BUCKET_POLICY = """{
  "Version": "2012-10-17",
  "Id": "PutObjPolicy",
  "Statement": [
    {
      "Sid": "DenyIncorrectEncryptionHeader",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::%s/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms"
        }
      }
    },
    {
      "Sid": "DenyUnEncryptedObjectUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::%s/*",
      "Condition": {
        "Null": {
          "s3:x-amz-server-side-encryption": "true"
        }
      }
    }
  ]
}"""


@contextlib.contextmanager
def bucket_with_encryption(boto_session, sagemaker_role):
    account = boto_session.client('sts').get_caller_identity()['Account']
    role_arn = boto_session.client('sts').get_caller_identity()['Arn']

    kms_client = boto_session.client('kms')
    kms_key_arn = _create_kms_key(kms_client, account, role_arn, sagemaker_role, None)

    region = boto_session.region_name
    bucket_name = 'sagemaker-{}-{}-with-kms'.format(region, account)

    s3 = boto_session.client('s3')
    try:
        # 'us-east-1' cannot be specified because it is the default region:
        # https://github.com/boto/boto3/issues/125
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(Bucket=bucket_name,
                             CreateBucketConfiguration={'LocationConstraint': region})

    except exceptions.ClientError as e:
        if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
            raise

    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            'Rules': [
                {
                    'ApplyServerSideEncryptionByDefault': {
                        'SSEAlgorithm': 'aws:kms',
                        'KMSMasterKeyID': kms_key_arn
                    }
                },
            ]
        }
    )

    s3.put_bucket_policy(
        Bucket=bucket_name,
        Policy=KMS_BUCKET_POLICY % (bucket_name, bucket_name)
    )

    yield 's3://' + bucket_name, kms_key_arn

    kms_client.schedule_key_deletion(KeyId=kms_key_arn, PendingWindowInDays=7)
