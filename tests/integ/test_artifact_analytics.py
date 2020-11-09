from __future__ import absolute_import

import time
import uuid
from contextlib import contextmanager

import pytest
import botocore

from sagemaker.analytics import ArtifactAnalytics
from sagemaker.lineage import artifact
import logging
from sagemaker.session import Session


@pytest.fixture()
def sagemaker_session(boto_session):
    return Session(boto_session=boto_session)


@contextmanager
def generate_artifacts(sagemaker_session):
    artifacts = []  # for resource cleanup

    try:
        # Search returns 10 results by default. Add 20 trials to verify pagination.
        for i in range(20):
            artifact_name = "experiment-" + str(uuid.uuid4())
            obj = artifact.Artifact.create(
                artifact_name="SDKAnalyticsIntegrationTest",
                artifact_type="SDKAnalyticsIntegrationTest",
                source_uri=artifact_name,
                properties={"k1": "v1"},
                sagemaker_session=sagemaker_session,
            )
            artifacts.append(obj)
            logging.info(f"Created {artifact_name}")
            time.sleep(1)

        # wait for search to get updated
        time.sleep(15)
        yield
    finally:
        _delete_resources(artifacts)


@pytest.mark.canary_quick
@pytest.mark.skip("Failing as restricted to the SageMaker/Pipeline runtimes")
def test_artifact_analytics(sagemaker_session):
    with generate_artifacts(sagemaker_session):
        analytics = ArtifactAnalytics(
            artifact_type="SDKAnalyticsIntegrationTest", sagemaker_session=sagemaker_session
        )

        df = analytics.dataframe()
        assert list(df.columns) == [
            "ArtifactName",
            "ArtifactArn",
            "ArtifactType",
            "ArtifactSourceUri",
            "CreationTime",
            "LastModifiedTime",
        ]

        assert len(df) > 10


def _delete_resources(artifacts):
    for art in artifacts:
        with _ignore_resource_not_found():
            art.delete()


@contextmanager
def _ignore_resource_not_found():
    try:
        yield
    except botocore.errorfactory.ResourceNotFoundException:
        pass
