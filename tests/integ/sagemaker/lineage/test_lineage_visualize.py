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
"""This module contains code to test SageMaker ``LineageQueryResult.visualize()``"""

import datetime
import logging
import time

import pytest

import sagemaker.lineage.query

from tests.integ.sagemaker.lineage.helpers import name, names, retry, LineageResourceHelper


def test_LineageResourceHelper():
    lineage_resource_helper = LineageResourceHelper()
    try:
        art1 = lineage_resource_helper.create_artifact(artifact_name=name())
        art2 = lineage_resource_helper.create_artifact(artifact_name=name())
        lineage_resource_helper.create_association(source_arn=art1, dest_arn=art2)
        lineage_resource_helper.clean_all()
    except Exception as e:
        print(e)
        assert False


def test_wide_graphs(sagemaker_session):
    lineage_resource_helper = LineageResourceHelper()
    art_root = lineage_resource_helper.create_artifact(artifact_name=name())
    try:
        for i in range(500):
            art = lineage_resource_helper.create_artifact(artifact_name=name())
            lineage_resource_helper.create_association(source_arn=art_root, dest_arn=art)
            time.sleep(0.2)
    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        assert False

    try:
        lq = sagemaker.lineage.query.LineageQuery(sagemaker_session)
        lq_result = lq.query(start_arns=[art_root])
        lq_result.visualize()
    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        assert False

    lineage_resource_helper.clean_all()
