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
"""This module contains helper methods for tests of SageMaker Lineage"""
from __future__ import absolute_import

import uuid
from datetime import datetime
import time


def name():
    return "lineage-integ-{}-{}".format(
        str(uuid.uuid4()), str(datetime.now().timestamp()).split(".")[0]
    )


def names():
    return [
        "lineage-integ-{}-{}".format(
            str(uuid.uuid4()), str(datetime.now().timestamp()).split(".")[0]
        )
        for i in range(3)
    ]


def retry(callable, num_attempts=8):
    assert num_attempts >= 1
    for i in range(num_attempts):
        try:
            return callable()
        except Exception as ex:
            if i == num_attempts - 1:
                raise ex
            print("Retrying", ex)
            time.sleep(2**i)
    assert False, "logic error in retry"


def traverse_graph_back(start_arn, sagemaker_session):
    def visit(arn, visited: set):
        visited.add(arn)
        associations = sagemaker_session.sagemaker_client.list_associations(DestinationArn=arn)[
            "AssociationSummaries"
        ]
        for association in associations:
            if association["SourceArn"] not in visited:
                ret.append(association)
                visit(association["SourceArn"], visited)

        return ret

    ret = []
    return visit(start_arn, set())


def traverse_graph_forward(start_arn, sagemaker_session):
    def visit(arn, visited: set):
        visited.add(arn)
        associations = sagemaker_session.sagemaker_client.list_associations(SourceArn=arn)[
            "AssociationSummaries"
        ]
        for association in associations:
            if association["DestinationArn"] not in visited:
                ret.append(association)
                visit(association["DestinationArn"], visited)

        return ret

    ret = []
    return visit(start_arn, set())
