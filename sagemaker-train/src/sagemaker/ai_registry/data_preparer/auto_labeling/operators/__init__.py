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
"""Client-side operators of the auto-labeling primitive.

An operator is a function: inputs are parameters, outputs are the returned
:class:`~sagemaker.ai_registry.data_preparer._operator_result.OperatorResult`; a
refusal is raised as
:class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`.
The operators are grouped by the question they answer:

    * :mod:`.inspect` - what is in my data, and can this be labeled?
    * :mod:`.seed` - what am I measuring against, and is it still trustworthy?
    * :mod:`.spec` - what are the rules, and which version ships?
    * :mod:`.job` - run it and read it back.
    * :mod:`.score` - how good is it, and can I prove it?
    * :mod:`.review` - which rows need a person?
    * :mod:`.report` - what does the run show?

Each subsystem lists its operators in ``__all__``; the ``_``-prefixed modules beside
them are private to this primitive. ``from sagemaker.ai_registry.data_preparer.auto_labeling
import prepare_round`` is the import path; where a function sits is not part of it.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import inspect as _inspect
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import job as _job
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import report as _report
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import review as _review
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import score as _score
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import seed as _seed
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import spec as _spec
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import *  # noqa: F401,F403
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import *  # noqa: F401,F403
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.report import *  # noqa: F401,F403
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import *  # noqa: F401,F403
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.score import *  # noqa: F401,F403
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import *  # noqa: F401,F403
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec import *  # noqa: F401,F403

# Client reads and the row rule cut across the subsystems, so they are exported here.
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows import (  # noqa: F401
    read_input_rows,
    read_jsonl,
    head_rows,
    report_client_read,
    print_client_reads,
    client_read_message,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import (  # noqa: F401
    job_tolerance,
    DEFAULT_CORRECT_TOLERANCE,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (  # noqa: F401
    GOLDEN_REVIEW_FIELD,
)

CLIENT_READ_EXPORTS = [
    # helpers
    "read_input_rows",
    "read_jsonl",
    "head_rows",
    "report_client_read",
    "print_client_reads",
    "client_read_message",
    "job_tolerance",
    # constants and exceptions
    "DEFAULT_CORRECT_TOLERANCE",
    "GOLDEN_REVIEW_FIELD",
]

__all__ = (
    *_inspect.__all__,
    *_seed.__all__,
    *_spec.__all__,
    *_job.__all__,
    *_score.__all__,
    *_review.__all__,
    *_report.__all__,
    *CLIENT_READ_EXPORTS,
)
