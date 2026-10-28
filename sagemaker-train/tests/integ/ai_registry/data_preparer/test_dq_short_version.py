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
"""Integration tests: DataPreparer.attach() on a DQ job with schema version "2.0".

Pre-requisites
--------------
This module is READ-ONLY — it attaches to an ARN you supply; it never submits
a new job.

    DATA_PREPARER_INTEG_DQ_SHORT_VERSION_JOB_ARN
        ARN of a *completed* DataQualityEvaluation job whose
        ``JobConfigSchemaVersion`` is exactly ``"2.0"`` (two-part, no patch
        component).  All tests in this module are skipped when this variable
        is unset.

        Find a suitable ARN by searching completed DQ jobs in the us-west-2
        integ account and checking ``DescribeJob.JobConfigSchemaVersion``.

Background
----------
A7 (DQ version normalization in ``_api_shape.py``):

Before the fix, ``DqShape.for_version("2.0")`` used exact-string membership
against ``("0.2.0", "2.0.0")`` and fell through to the v1 shape.  The v1
shape reads a scalar ``DatasetArn`` key that does not exist in v2 documents
(which use ``InputDataConfig[0].DatasetArn``).  This made every attach to a
job submitted with schema version ``"2.0"`` raise::

    DataPreparerJobError: Completed job missing DatasetArn in JobConfigDocument.

After the fix, ``DqShape.for_version`` extracts the major-version component
and maps major == 2 to V2DqShape, which accepts ``"2"``, ``"2.0"``, and
``"2.0.0"`` interchangeably.
"""
from __future__ import absolute_import

import os

import boto3
import pytest

from sagemaker.ai_registry import DataPreparer
from sagemaker.core.helper.session_helper import Session

_ENV_VAR = "DATA_PREPARER_INTEG_DQ_SHORT_VERSION_JOB_ARN"


@pytest.mark.integ
class TestDQShortVersionAttach:
    """Attach to a completed DQ job that used schema version "2.0" (no patch).

    The test is READ-ONLY: it never submits a new job.  Supply the ARN of an
    existing completed DQ job via ``DATA_PREPARER_INTEG_DQ_SHORT_VERSION_JOB_ARN``.
    """

    def test_attach_dq_short_version_schema(self):
        """attach() on a v2.0 DQ job returns a DataSet, not DataPreparerJobError.

        Before the A7 fix, DqShape.for_version("2.0") fell through to V1DqShape
        because the lookup used exact-string membership against ("0.2.0", "2.0.0").
        The v1 extractor then read ``DatasetArn`` (absent in v2 documents) and
        raised ``DataPreparerJobError: Completed job missing DatasetArn``.

        After the fix, major-version dispatch maps "2.0" to V2DqShape, which
        reads ``InputDataConfig[0].DatasetArn`` correctly.
        """
        arn = os.environ.get(_ENV_VAR)
        if not arn:
            pytest.skip(
                f"Set {_ENV_VAR} to the ARN of a completed DataQualityEvaluation "
                f"job with JobConfigSchemaVersion='2.0' to run this test."
            )

        # Parse the region from the ARN so we can build the right session.
        # ARN format: arn:aws:sagemaker:{region}:{account}:job/{category}/{name}
        arn_parts = arn.split(":")
        if len(arn_parts) < 4 or not arn_parts[3]:
            pytest.fail(
                f"A7: could not parse region from ARN {arn!r}; "
                f"expected format arn:aws:sagemaker:{{region}}:{{account}}:job/..."
            )
        region = arn_parts[3]

        session = Session(boto_session=boto3.Session(region_name=region))

        # Before the A7 fix this would raise:
        #   DataPreparerJobError: Completed job missing DatasetArn in JobConfigDocument.
        # After the fix it should return a DataSet with a non-empty arn.
        result = DataPreparer.attach(arn, sagemaker_session=session)

        assert result is not None, f"A7: DataPreparer.attach returned None for ARN {arn!r}"
        dataset_arn = getattr(result, "arn", None)
        assert dataset_arn, (
            f"A7: attach returned a DataSet with empty arn for job ARN {arn!r}; "
            f"result={result!r}"
        )
