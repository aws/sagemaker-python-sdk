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
from __future__ import absolute_import

import datetime
import unittest.mock

from sagemaker.lineage import artifact, _api_types


def test_create(sagemaker_session):
    sagemaker_session.sagemaker_client.create_artifact.return_value = {
        "ArtifactArn": "bazz",
    }
    obj = artifact.Artifact.create(
        artifact_name="foo",
        source_uri="bar",
        sagemaker_session=sagemaker_session,
    )
    sagemaker_session.sagemaker_client.create_artifact.assert_called_with(
        ArtifactName="foo", Source={"SourceUri": "bar"}
    )
    assert "foo" == obj.artifact_name
    assert "bar" == obj.source.source_uri
    assert "bazz" == obj.artifact_arn


def test_set_tag(sagemaker_session):
    sagemaker_session.sagemaker_client.create_artifact.return_value = {
        "ArtifactArn": "artifact-arn",
    }
    sagemaker_session.sagemaker_client.add_tags.return_value = {
        "Tags": [{"Key": "foo", "Value": "bar"}]
    }
    obj = artifact.Artifact.create(
        artifact_name="foo",
        source_uri="bar",
        sagemaker_session=sagemaker_session,
    )
    obj.set_tag(tag={"Key": "foo", "Value": "bar"})

    sagemaker_session.sagemaker_client.add_tags.assert_called_with = {
        "ResourceArn": "artifact-arn",
        "Tags": [{"Key": "foo", "Value": "bar"}],
    }


def test_set_tags(sagemaker_session):
    sagemaker_session.sagemaker_client.create_artifact.return_value = {
        "ArtifactArn": "artifact-arn",
    }
    obj = artifact.Artifact.create(
        artifact_name="foo",
        source_uri="bar",
        sagemaker_session=sagemaker_session,
    )
    tags = [{"Key": "foo1", "Value": "bar1"}, {"Key": "foo2", "Value": "bar2"}]
    sagemaker_session.sagemaker_client.add_tags.return_value = {"Tags": tags}

    obj.set_tags(tags=tags)
    sagemaker_session.sagemaker_client.add_tags.assert_called_with = {
        "ResourceArn": "artifact-arn",
        "Tags": [{"Key": "foo1", "Value": "bar1"}, {"Key": "foo2", "Value": "bar2"}],
    }


def test_create_with_tags(sagemaker_session):
    sagemaker_session.sagemaker_client.create_artifact.return_value = {
        "ArtifactArn": "bazz",
    }

    tags = [{"Key": "foo", "Value": "bar"}]
    artifact.Artifact.create(
        artifact_name="foo",
        source_uri="bar",
        tags=tags,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.create_artifact.assert_called_with(
        ArtifactName="foo", Source={"SourceUri": "bar"}, Tags=[{"Key": "foo", "Value": "bar"}]
    )


def test_load(sagemaker_session):
    now = datetime.datetime.now(datetime.timezone.utc)

    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactName": "A",
        "Source": {"SourceUri": "B", "SourceTypes": ["C1", "C2", "C3"]},
        "ArtifactArn": "D",
        "ArtifactType": "E",
        "Properties": {"F": "FValue", "G": "GValue"},
        "CreationTime": now,
        "CreatedBy": {},
        "LastModifiedTime": now,
        "LastModifiedBy": {},
    }
    obj = artifact.Artifact.load(artifact_arn="arn", sagemaker_session=sagemaker_session)
    sagemaker_session.sagemaker_client.describe_artifact.assert_called_with(ArtifactArn="arn")
    assert "A" == obj.artifact_name
    assert "B" == obj.source.source_uri
    assert "C1" == obj.source.source_types[0]
    assert "C2" == obj.source.source_types[1]
    assert "C3" == obj.source.source_types[2]
    assert "D" == obj.artifact_arn
    assert "E" == obj.artifact_type
    assert {"F": "FValue", "G": "GValue"} == obj.properties


def test_list(sagemaker_session):
    creation_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=3)
    last_modified_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=4)

    sagemaker_session.sagemaker_client.list_artifacts.side_effect = [
        {
            "ArtifactSummaries": [
                {
                    "ArtifactArn": "A" + str(i),
                    "ArtifactName": "B" + str(i),
                    "Source": {
                        "SourceUri": "D" + str(i),
                        "source_types": [{"SourceIdType": "source_id_type", "Value": "value1"}],
                    },
                    "ArtifactType": "test-type",
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "LastModifiedTime": last_modified_time + datetime.timedelta(hours=i),
                }
                for i in range(10)
            ],
            "NextToken": "100",
        },
        {
            "ArtifactSummaries": [
                {
                    "ArtifactArn": "A" + str(i),
                    "ArtifactName": "B" + str(i),
                    "Source": {
                        "SourceUri": "D" + str(i),
                        "source_types": [{"SourceIdType": "source_id_type", "Value": "value1"}],
                    },
                    "ArtifactType": "test-type",
                    "CreationTime": creation_time + datetime.timedelta(hours=i),
                    "LastModifiedTime": last_modified_time + datetime.timedelta(hours=i),
                }
                for i in range(10, 20)
            ]
        },
    ]

    expected = [
        _api_types.ArtifactSummary(
            artifact_arn="A" + str(i),
            artifact_name="B" + str(i),
            source=_api_types.ArtifactSource(
                source_uri="D" + str(i),
                source_types=[{"SourceIdType": "source_id_type", "Value": "value1"}],
            ),
            artifact_type="test-type",
            creation_time=creation_time + datetime.timedelta(hours=i),
            last_modified_time=last_modified_time + datetime.timedelta(hours=i),
        )
        for i in range(20)
    ]
    result = list(
        artifact.Artifact.list(
            sagemaker_session=sagemaker_session,
            source_uri="foo",
            artifact_type="bar",
            sort_by="CreationTime",
            sort_order="Ascending",
        )
    )

    assert expected == result
    expected_calls = [
        unittest.mock.call(
            SortBy="CreationTime",
            SortOrder="Ascending",
            SourceUri="foo",
            ArtifactType="bar",
        ),
        unittest.mock.call(
            NextToken="100",
            SortBy="CreationTime",
            SortOrder="Ascending",
            SourceUri="foo",
            ArtifactType="bar",
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_artifacts.mock_calls


def test_list_empty(sagemaker_session):
    sagemaker_session.sagemaker_client.list_artifacts.return_value = {"ArtifactSummaries": []}
    assert [] == list(artifact.Artifact.list(sagemaker_session=sagemaker_session))


def test_list_artifacts_call_args(sagemaker_session):
    created_before = datetime.datetime(1999, 10, 12, 0, 0, 0)
    created_after = datetime.datetime(1990, 10, 12, 0, 0, 0)
    source_uri = "foo-source-uri"
    artifact_type = "foo-artifact-type"
    next_token = "thetoken"
    max_results = 99

    sagemaker_session.sagemaker_client.list_artifacts.return_value = {}
    assert [] == list(
        artifact.Artifact.list(
            sagemaker_session=sagemaker_session,
            source_uri=source_uri,
            artifact_type=artifact_type,
            created_before=created_before,
            created_after=created_after,
            next_token=next_token,
            max_results=max_results,
            sort_by="CreationTime",
            sort_order="Ascending",
        )
    )

    expected_calls = [
        unittest.mock.call(
            SourceUri="foo-source-uri",
            ArtifactType="foo-artifact-type",
            CreatedBefore=created_before,
            CreatedAfter=created_after,
            SortBy="CreationTime",
            SortOrder="Ascending",
            NextToken="thetoken",
            MaxResults=99,
        )
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_artifacts.mock_calls


def test_save(sagemaker_session):
    obj = artifact.Artifact(
        sagemaker_session,
        artifact_arn="test-arn",
        artifact_name="foo",
        properties={"k1": "v1", "k2": "v2"},
        properties_to_remove=["r1"],
    )

    sagemaker_session.sagemaker_client.update_artifact.return_value = {}
    obj.save()

    sagemaker_session.sagemaker_client.update_artifact.assert_called_with(
        ArtifactArn="test-arn",
        ArtifactName="foo",
        Properties={"k1": "v1", "k2": "v2"},
        PropertiesToRemove=["r1"],
    )


def test_delete(sagemaker_session):
    obj = artifact.Artifact(sagemaker_session, artifact_arn="foo")
    sagemaker_session.sagemaker_client.delete_artifact.return_value = {}
    obj.delete()
    sagemaker_session.sagemaker_client.delete_artifact.assert_called_with(ArtifactArn="foo")


def test_create_delete_with_association(sagemaker_session):
    obj = artifact.Artifact(sagemaker_session, artifact_arn="foo")

    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": obj.artifact_arn,
                    "SourceName": "X" + str(i),
                    "DestinationArn": "B" + str(i),
                    "DestinationName": "Y" + str(i),
                    "SourceType": "C" + str(i),
                    "DestinationType": "D" + str(i),
                    "AssociationType": "E" + str(i),
                    "CreationTime": None,
                    "CreatedBy": {},
                }
                for i in range(1)
            ],
        },
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "A" + str(i),
                    "SourceName": "X" + str(i),
                    "DestinationArn": obj.artifact_arn,
                    "DestinationName": "Y" + str(i),
                    "SourceType": "C" + str(i),
                    "DestinationType": "D" + str(i),
                    "AssociationType": "E" + str(i),
                    "CreationTime": None,
                    "CreatedBy": {},
                }
                for i in range(1, 2)
            ]
        },
    ]
    sagemaker_session.sagemaker_client.delete_association.return_value = {}
    sagemaker_session.sagemaker_client.delete_artifact.return_value = {}

    obj.delete(disassociate=True)

    delete_with_association_expected_calls = [
        unittest.mock.call(SourceArn=obj.artifact_arn, DestinationArn="B0"),
        unittest.mock.call(SourceArn="A1", DestinationArn=obj.artifact_arn),
    ]
    assert (
        delete_with_association_expected_calls
        == sagemaker_session.sagemaker_client.delete_association.mock_calls
    )


@unittest.mock.patch("sagemaker.lineage.artifact.get_module")
def test_downstream_trials(mock_get_module, sagemaker_session):
    # Mock smexperiments module
    mock_smexperiments = unittest.mock.MagicMock()
    mock_get_module.return_value = mock_smexperiments
    
    sagemaker_session.sagemaker_client.list_associations.side_effect = [
        {
            "AssociationSummaries": [
                {
                    "SourceArn": "A" + str(i),
                    "SourceName": "X" + str(i),
                    "DestinationArn": "B" + str(i),
                    "DestinationName": "Y" + str(i),
                    "SourceType": "C" + str(i),
                    "DestinationType": "D" + str(i),
                    "AssociationType": "E" + str(i),
                    "CreationTime": datetime.datetime.now(),
                    "CreatedBy": {},
                }
                for i in range(10)
            ],
            "NextToken": None,
        }
    ]

    sagemaker_session.sagemaker_client.search.return_value = {
        "Results": [
            {
                "TrialComponent": {
                    "TrialComponentName": "tc-1",
                    "TrialComponentArn": "arn::tc-1",
                    "DisplayName": "TC1",
                    "Parents": [{"TrialName": "test-trial-name"}],
                }
            }
        ]
    }

    obj = artifact.Artifact(
        sagemaker_session=sagemaker_session,
        artifact_arn="test-arn",
        artifact_name="foo",
        properties={"k1": "v1", "k2": "v2"},
        properties_to_remove=["r1"],
    )

    result = obj.downstream_trials(sagemaker_session=sagemaker_session)

    expected_trials = ["test-trial-name"]

    assert expected_trials == result

    expected_calls = [
        unittest.mock.call(
            SourceArn="test-arn",
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.list_associations.mock_calls


@unittest.mock.patch("sagemaker.lineage.artifact.get_module")
def test_downstream_trials_v2(mock_get_module, sagemaker_session):
    # Mock smexperiments module
    mock_smexperiments = unittest.mock.MagicMock()
    mock_get_module.return_value = mock_smexperiments
    
    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": "B" + str(i), "Type": "DataSet", "LineageType": "Artifact"} for i in range(10)
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }
    sagemaker_session.sagemaker_client.search.return_value = {
        "Results": [
            {
                "TrialComponent": {
                    "TrialComponentName": "tc-1",
                    "TrialComponentArn": "arn::tc-1",
                    "DisplayName": "TC1",
                    "Parents": [{"TrialName": "test-trial-name"}],
                }
            }
        ]
    }

    obj = artifact.Artifact(
        sagemaker_session=sagemaker_session,
        artifact_arn="test-arn",
        artifact_name="foo",
        properties={"k1": "v1", "k2": "v2"},
        properties_to_remove=["r1"],
    )

    result = obj.downstream_trials_v2()

    expected_trials = ["test-trial-name"]

    assert expected_trials == result

    expected_calls = [
        unittest.mock.call(
            Direction="Descendants",
            Filters={"LineageTypes": ["TrialComponent"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=["test-arn"],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls


@unittest.mock.patch("sagemaker.lineage.artifact.get_module")
def test_upstream_trials(mock_get_module, sagemaker_session):
    # Mock smexperiments module
    mock_smexperiments = unittest.mock.MagicMock()
    mock_get_module.return_value = mock_smexperiments
    
    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": "B" + str(i), "Type": "DataSet", "LineageType": "Artifact"} for i in range(10)
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }
    sagemaker_session.sagemaker_client.search.return_value = {
        "Results": [
            {
                "TrialComponent": {
                    "TrialComponentName": "tc-1",
                    "TrialComponentArn": "arn::tc-1",
                    "DisplayName": "TC1",
                    "Parents": [{"TrialName": "test-trial-name"}],
                }
            }
        ]
    }

    obj = artifact.Artifact(
        sagemaker_session=sagemaker_session,
        artifact_arn="test-arn",
        artifact_name="foo",
        properties={"k1": "v1", "k2": "v2"},
        properties_to_remove=["r1"],
    )

    result = obj.upstream_trials()

    expected_trials = ["test-trial-name"]

    assert expected_trials == result

    expected_calls = [
        unittest.mock.call(
            Direction="Ascendants",
            Filters={"LineageTypes": ["TrialComponent"]},
            IncludeEdges=False,
            MaxDepth=10,
            StartArns=["test-arn"],
        ),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.query_lineage.mock_calls


def test_s3_uri_artifacts(sagemaker_session):
    obj = artifact.Artifact(
        sagemaker_session=sagemaker_session,
        artifact_arn="test-arn",
        artifact_name="foo",
        source_uri="s3://abced",
        properties={"k1": "v1", "k2": "v2"},
        properties_to_remove=["r1"],
    )
    sagemaker_session.sagemaker_client.list_artifacts.side_effect = [
        {
            "ArtifactSummaries": [
                {
                    "ArtifactArn": "A",
                    "ArtifactName": "B",
                    "Source": {
                        "SourceUri": "D",
                        "source_types": [{"SourceIdType": "source_id_type", "Value": "value1"}],
                    },
                    "ArtifactType": "test-type",
                }
            ],
            "NextToken": "100",
        },
    ]
    result = obj.s3_uri_artifacts(s3_uri="s3://abced")

    expected_calls = [
        unittest.mock.call(SourceUri="s3://abced"),
    ]
    expected_result = {
        "ArtifactSummaries": [
            {
                "ArtifactArn": "A",
                "ArtifactName": "B",
                "Source": {
                    "SourceUri": "D",
                    "source_types": [{"SourceIdType": "source_id_type", "Value": "value1"}],
                },
                "ArtifactType": "test-type",
            }
        ],
        "NextToken": "100",
    }
    assert expected_calls == sagemaker_session.sagemaker_client.list_artifacts.mock_calls
    assert result == expected_result
