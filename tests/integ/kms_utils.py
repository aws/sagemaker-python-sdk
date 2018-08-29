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
        "AWS": "arn:aws:iam::{account_id}:root"
      }},
      "Action": "kms:*",
      "Resource": "*"
    }},
    {{
      "Sid": "Allow use of the key",
      "Effect": "Allow",
      "Principal": {{
        "AWS": "arn:aws:iam::{account_id}:role/SageMakerRole"
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
        "AWS": "arn:aws:iam::{account_id}:role/SageMakerRole"
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
