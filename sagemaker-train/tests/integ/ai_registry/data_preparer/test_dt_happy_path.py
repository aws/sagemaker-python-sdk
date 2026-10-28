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
"""Integration tests: DT happy path on BYOD datasets (direct import and DataSet.create)."""
from __future__ import absolute_import

from urllib.parse import urlparse
import uuid

import boto3
import pytest

from sagemaker.ai_registry import DataPreparer, DataSet, DatasetJobConfig, JobCategory


@pytest.mark.integ
class TestDTOnByodDataset:
    """DT runs TextQualityFilter over a v4 BYOD DataSet and emits a new version.

    Registers the BYOD JSONL fixture as a v4 DataSet via ``import_dataset``.
    """

    def test_transform_filters_rows(
        self, sagemaker_session, dt_role, s3_fixtures, integ_bucket, import_dataset
    ):
        """Submit DT job and assert output DataSet is a new version with fewer rows."""
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])

        # byod_training_sft.jsonl has 5 rows with prompt lengths [178, 167, 167,
        # 159, 149]. MinLength=170 drops 4 rows and keeps 1.
        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={
                "S3OutputPath": f"s3://{integ_bucket}/data-preparer-integ-dt-output/",
            },
            transformation_config={
                "Operations": [{
                    "Operator": "TextQualityFilter",
                    "Config": {
                        "FilterType": "CharacterLength",
                        "TextField": "prompt",
                        "Action": "drop",
                        "MinLength": 170,
                    },
                }],
            },
        )
        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=dt_role,
        )
        result = preparer.transform(dataset)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].arn is not None
        assert result[0].arn != input_arn
        assert result[0].source is not None
        assert result[0].source.startswith("s3://")

        # Verify the filter dropped 4 of 5 rows.
        s3 = boto3.client("s3", region_name=sagemaker_session.boto_region_name)
        out = urlparse(result[0].source)
        output_body = s3.get_object(Bucket=out.netloc, Key=out.path.lstrip("/"))["Body"].read()
        output_rows = sum(1 for line in output_body.splitlines() if line.strip())
        assert output_rows == 1


@pytest.mark.integ
class TestDTOnDataSetCreate:
    """DT runs over a DataSet registered with ``DataSet.create()`` -- the path the
    DataTransformation tutorial documents for bringing your own data.

    Uses the ``Custom`` format and each document schema version DT reads, so the
    documented bring-your-own-data path stays covered end to end. ``create()``
    writes ``DatasetType`` and ``DatasetS3Uri``; a document with neither fails DT
    with "Unable to size inputs for compute routing".
    """

    @pytest.mark.parametrize("document_schema_version", ["4.0.0", "5.0.0"])
    def test_transform_on_dataset_from_create(
        self,
        sagemaker_session,
        dt_role,
        s3_fixtures,
        integ_bucket,
        integ_hub_name,
        document_schema_version,
    ):
        """create() a BYOD DataSet, run a filter on it, assert the expected rows survive."""
        source = DataSet.create(
            name=f"dt-integ-create-{uuid.uuid4().hex[:8]}",
            source=s3_fixtures["byod_training_sft.jsonl"],
            document_schema_version=document_schema_version,
            hub_name=integ_hub_name,
            sagemaker_session=sagemaker_session,
            wait=True,
        )
        assert source.arn is not None

        # Same fixture and filter as TestDTOnByodDataset: 1 of 5 rows survives.
        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": source.arn}],
            output_data_config={
                "S3OutputPath": f"s3://{integ_bucket}/data-preparer-integ-dt-output/",
            },
            transformation_config={
                "Operations": [{
                    "Operator": "TextQualityFilter",
                    "Config": {
                        "FilterType": "CharacterLength",
                        "TextField": "prompt",
                        "Action": "drop",
                        "MinLength": 170,
                    },
                }],
            },
        )
        result = DataPreparer(sagemaker_session=sagemaker_session, role=dt_role).transform(dataset)

        assert isinstance(result, list) and len(result) == 1
        assert result[0].arn not in (None, source.arn)
        s3 = boto3.client("s3", region_name=sagemaker_session.boto_region_name)
        out = urlparse(result[0].source)
        output_body = s3.get_object(Bucket=out.netloc, Key=out.path.lstrip("/"))["Body"].read()
        assert sum(1 for line in output_body.splitlines() if line.strip()) == 1
