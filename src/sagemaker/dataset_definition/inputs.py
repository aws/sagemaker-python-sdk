# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""The input configs for DatasetDefinition.

DatasetDefinition supports the data sources like S3 which can be queried via Athena
and Redshift. A mechanism has to be created for customers to generate datasets
from Athena/Redshift queries and to retrieve the data, using Processing jobs
so as to make it available for other downstream processes.
"""
from __future__ import absolute_import

from sagemaker.apiutils._base_types import ApiObject


class RedshiftDatasetDefinition(ApiObject):
    """DatasetDefinition for Redshift.

    With this input, SQL queries will be executed using Redshift to generate datasets to S3.

    Attributes:
        cluster_id (str): The Redshift cluster Identifier.
        database (str): The Redshift database created for your cluster.
        db_user (str): The user name of a user account that has permission to connect
            to the database.
        query_string (str): The SQL query statements to be executed.
        cluster_role_arn (str): Redshift cluster role arn.
        output_s3_uri (str): The path to a specific S3 object or a S3 prefix for output
        kms_key_id (str): KMS key id.
        output_format (str): the data storage format for Redshift query results.
            Valid options are "PARQUET", "CSV"
        output_compression (str): compression used for Redshift query results.
            Valid options are "None", "GZIP", "SNAPPY", "ZSTD", "BZIP2"
    """

    cluster_id = None
    database = None
    db_user = None
    query_string = None
    cluster_role_arn = None
    output_s3_uri = None
    kms_key_id = None
    output_format = None
    output_compression = None


class AthenaDatasetDefinition(ApiObject):
    """DatasetDefinition for Athena.

    With this input, SQL queries will be executed using Athena to generate datasets to S3.

    Attributes:
        catalog (str): The name of the data catalog used in query execution.
        database (str): The name of the database used in the query execution.
        query_string (str): The SQL query statements to be executed.
        output_s3_uri (str): the path to a specific S3 object or a S3 prefix for output
        work_group (str): The name of the workgroup in which the query is being started.
        kms_key_id (str): KMS key id.
        output_format (str): the data storage format for Athena query results.
            Valid options are "PARQUET", "ORC", "AVRO", "JSON", "TEXTFILE"
        output_compression (str): compression used for Athena query results.
            Valid options are "GZIP", "SNAPPY", "ZLIB"
    """

    catalog = None
    database = None
    query_string = None
    output_s3_uri = None
    work_group = None
    kms_key_id = None
    output_format = None
    output_compression = None


class DatasetDefinition(ApiObject):
    """DatasetDefinition input.

    Attributes:
        data_distribution_type (str): Valid options are "FullyReplicated" or "ShardedByS3Key".
        input_mode (str): Valid options are "Pipe" or "File".
        local_path (str): the path to a local directory. If not provided, skips data download by
            SageMaker platform.
        redshift_dataset_definition
            (:class:`~sagemaker.dataset_definition.RedshiftDatasetDefinition`): Redshift
            dataset definition.
        athena_dataset_definition (:class:`~sagemaker.dataset_definition.AthenaDatasetDefinition`):
            Athena dataset definition.
    """

    _custom_boto_types = {
        "redshift_dataset_definition": (RedshiftDatasetDefinition, True),
        "athena_dataset_definition": (AthenaDatasetDefinition, True),
    }

    data_distribution_type = "ShardedByS3Key"
    input_mode = "File"
    local_path = None
    redshift_dataset_definition = None
    athena_dataset_definition = None


class S3Input(ApiObject):
    """Metadata of data objects stored in S3.

    Two options are provided: specifying a S3 prefix or by explicitly listing the files
    in a manifest file and referencing the manifest file's S3 path.
    Note: Strong consistency is not guaranteed if S3Prefix is provided here.
    S3 list operations are not strongly consistent.
    Use ManifestFile if strong consistency is required.

    Attributes:
        s3_uri (str): the path to a specific S3 object or a S3 prefix
        local_path (str): the path to a local directory. If not provided, skips data download
            by SageMaker platform.
        s3_data_type (str): Valid options are "ManifestFile" or "S3Prefix".
        s3_input_mode (str): Valid options are "Pipe" or "File".
        s3_data_distribution_type (str): Valid options are "FullyReplicated" or "ShardedByS3Key".
        s3_compression_type (str): Valid options are "None" or "Gzip".
    """

    s3_uri = None
    local_path = None
    s3_data_type = "S3Prefix"
    s3_input_mode = "File"
    s3_data_distribution_type = "FullyReplicated"
    s3_compression_type = None
