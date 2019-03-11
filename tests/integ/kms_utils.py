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

from botocore import exceptions

KEY_ALIAS = "SageMakerKmsKey"
KEY_POLICY = '''
{{
  "Version": "2012-10-17",
  "Id": "sagemaker-kms-integ-test-policy",
  "Statement": [
    {{
      "Sid": "Enable IAM User Permissions",
      "Effect": "Allow",
      "Principal": {{
        "AWS": "{account_id}"
      }},
      "Action": "kms:*",
      "Resource": "*"
    }},
    {{
      "Sid": "Allow use of the key",
      "Effect": "Allow",
      "Principal": {{
        "AWS": "{account_id}"
      }},
      "Action": [
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:ReEncrypt*",
        "kms:GenerateDataKey*",
        "kms:DescribeKey"
      ],
      "Resource": "*"
    }},
    {{
      "Sid": "Allow attachment of persistent resources",
      "Effect": "Allow",
      "Principal": {{
        "AWS": "{account_id}"
      }},
      "Action": [
        "kms:CreateGrant",
        "kms:ListGrants",
        "kms:RevokeGrant"
      ],
      "Resource": "*",
      "Condition": {{
        "Bool": {{
          "kms:GrantIsForAWSResource": "true"
        }}
      }}
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


def _create_kms_key(kms_client, account_id):
    response = kms_client.create_key(
        Policy=KEY_POLICY.format(account_id=account_id),
        Description='KMS key for SageMaker Python SDK integ tests',
    )
    key_arn = response['KeyMetadata']['Arn']
    response = kms_client.create_alias(AliasName='alias/' + KEY_ALIAS, TargetKeyId=key_arn)
    return key_arn


def get_or_create_kms_key(kms_client, account_id):
    kms_key_arn = _get_kms_key_arn(kms_client, KEY_ALIAS)
    if kms_key_arn is not None:
        return kms_key_arn
    else:
        return _create_kms_key(kms_client, account_id)


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


def get_or_create_bucket_with_encryption(boto_session):
    account = boto_session.client('sts').get_caller_identity()['Account']
    kms_key_arn = get_or_create_kms_key(boto_session.client('kms'), account)

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

    return 's3://' + bucket_name, kms_key_arn
