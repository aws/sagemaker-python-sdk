from __future__ import absolute_import

import mock
import pandas as pd
from collections import OrderedDict
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage._api_types import ArtifactSource

from sagemaker.analytics import ArtifactAnalytics


def artifact(artifact_name):
    return Artifact(
        artifact_arn=artifact_name + "-arn",
        artifact_name=artifact_name,
        source=ArtifactSource(source_uri="some-source-uri", source_types=[]),
        artifact_type="UnitTestType",
        creation_time="creation-time",
        last_modified_time="last-modified-time",
    )


def test_analytics_dataframe():
    test_artifacts = [
        artifact("artifact-1"),
        artifact("artifact-2"),
    ]
    with mock.patch.object(Artifact, "list", return_value=test_artifacts):
        analytics = ArtifactAnalytics()
        actual_dataframe = analytics.dataframe()

    expected_dataframe = pd.DataFrame.from_dict(
        OrderedDict(
            [
                ("ArtifactName", ["artifact-1", "artifact-2"]),
                ("ArtifactArn", ["artifact-1-arn", "artifact-2-arn"]),
                ("ArtifactType", ["UnitTestType", "UnitTestType"]),
                ("ArtifactSourceUri", ["some-source-uri", "some-source-uri"]),
                ("CreationTime", ["creation-time", "creation-time"]),
                ("LastModifiedTime", ["last-modified-time", "last-modified-time"]),
            ]
        )
    )

    pd.testing.assert_frame_equal(expected_dataframe, actual_dataframe)
