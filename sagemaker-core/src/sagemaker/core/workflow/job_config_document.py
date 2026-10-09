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
"""Encode a ``CreateJob`` ``JobConfigDocument`` that may carry pipeline variables.

Unlike the other create APIs, ``CreateJob`` flattens the job configuration into
one JSON **string** member, so nothing inside it sits at a position the pipeline
definition serializer can rewrite, and ``json.dumps`` on a ``PipelineVariable``
raises ``TypeError``. The encoder therefore emits the JSON text as a ``Join`` of
literal fragments and the variables, which the pipeline service resolves during
execution like any other variable leaf.

Limitations, both inherent to splicing values into finished JSON text:

* A variable is spliced in as raw text, so a value resolving to text containing
  a double quote or a backslash produces a malformed document. Pipeline
  variables resolve to ARNs, S3 URIs and identifiers in practice.
* A variable always lands in a JSON **string** position; one standing in for a
  number or boolean reaches the service quoted. Parameterise string-valued
  fields only.
"""
from __future__ import absolute_import

import json
import re
import uuid
from typing import List, Union

from sagemaker.core.helper.pipeline_variable import PipelineVariable


def convert_job_config_document_to_string(job_config) -> Union[str, PipelineVariable]:
    """Convert a job config dict to the string-typed ``JobConfigDocument`` value.

    Args:
        job_config (Dict[str, Any]): The job configuration. May hold
            ``PipelineVariable`` values at any depth.

    Returns:
        Union[str, PipelineVariable]: ``json.dumps(job_config)`` when the config
        holds no pipeline variable. Otherwise a ``Join`` over the JSON text and
        those variables -- not yet a string, but the expression the pipeline
        service resolves to the document string during execution.
    """
    # Import locally: sagemaker.core.workflow.functions imports from this package's
    # entities, and a module-level import here would be a cycle.
    from sagemaker.core.workflow.functions import Join

    variables: List[PipelineVariable] = []
    # A run-unique prefix, drawn from characters json.dumps never escapes, so the
    # placeholder survives serialisation verbatim and cannot collide with real content.
    token_prefix = "__sagemaker_pipeline_variable_%s_" % uuid.uuid4().hex

    def _placeholder(obj):
        # json.dumps invokes this for exactly the objects it cannot serialise, so
        # it drives the traversal: each variable is recorded and replaced by a
        # numbered placeholder in the emitted text.
        if isinstance(obj, PipelineVariable):
            variables.append(obj)
            return "%s%d__" % (token_prefix, len(variables) - 1)
        raise TypeError(
            "Object of type %s is not JSON serializable" % obj.__class__.__name__
        )

    document = json.dumps(job_config, default=_placeholder)
    if not variables:
        return document

    # With the capturing group, re.split alternates literal text and placeholder
    # indices: [literal, index, literal, index, ..., literal].
    pieces = re.split(re.escape(token_prefix) + r"(\d+)__", document)
    if (len(pieces) - 1) // 2 != len(variables):
        # Unreachable: every placeholder is a plain string value, so it must survive
        # json.dumps. Raised rather than silently shipping a literal placeholder.
        raise ValueError(
            "[PySDK Error] Could not encode JobConfigDocument: %d of %d pipeline "
            "variables were not found in the serialised document."
            % (len(variables) - (len(pieces) - 1) // 2, len(variables))
        )
    values: List[Union[str, PipelineVariable]] = [
        variables[int(piece)] if index % 2 else piece
        for index, piece in enumerate(pieces)
        if index % 2 or piece
    ]
    return Join(on="", values=values)
