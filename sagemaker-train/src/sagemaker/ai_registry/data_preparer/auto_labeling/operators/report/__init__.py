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
"""What does the run show?

The end-of-loop run report, ``report.html`` and ``report.md`` from the same sections: the
job-produced facts (its summary and held-out report) plus the client facts (the fold's
spot-check verdict, the deliverables, the jurors' responses), or the whole-corpus
people-only case from a local ``labels.jsonl``.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.report.render import (  # noqa: F401
    write_report,
    build_html,
    build_markdown,
)

# The operator, then the builders it writes.
__all__ = [
    # operators
    "write_report",
    # helpers
    "build_html",
    "build_markdown",
]
