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
"""Run it and read it back.

The labeling request, the job client, one round end to end, the run log, and the Hub
doors.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round import (  # noqa: F401
    run_labeling_round,
    tune_labeling_spec,
    label_corpus,
    prepare_round,
    submit_round,
    read_round,
    round_summary,
    RoundRefused,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import (  # noqa: F401
    build_labeling_request,
    spec_score_scale,
    scratch_dataset_name,
    MAX_SPEC_BYTES,
    LabelingRequestError,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client import (  # noqa: F401
    upload_text,
    submit_labeling_job,
    CreateJobRefused,
    OptionRefused,
    LabelingBackendUnavailable,
    JobWaitError,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.register import (  # noqa: F401
    resolve_dataset_ref,
    register_labeled_output,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job._scratch import (  # noqa: F401
    scratch_cleanup,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.dt_handoff import (  # noqa: F401
    dt_view,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.run_log import (  # noqa: F401
    labeling_run_record,
    read_labeling_runs,
    latest_labeling_runs,
    labeling_run_invocation,
    labeling_run_for_output,
)

# The operators, then the helpers their callers need beside them, then the names a
# caller catches or compares against.
__all__ = [
    # operators
    "run_labeling_round",
    "tune_labeling_spec",
    "label_corpus",
    "prepare_round",
    "submit_round",
    "read_round",
    "resolve_dataset_ref",
    "register_labeled_output",
    "scratch_cleanup",
    "dt_view",
    "labeling_run_record",
    # helpers
    "round_summary",
    "build_labeling_request",
    "spec_score_scale",
    "scratch_dataset_name",
    "upload_text",
    "submit_labeling_job",
    "read_labeling_runs",
    "latest_labeling_runs",
    "labeling_run_invocation",
    "labeling_run_for_output",
    # constants and exceptions
    "RoundRefused",
    "MAX_SPEC_BYTES",
    "LabelingRequestError",
    "CreateJobRefused",
    "OptionRefused",
    "LabelingBackendUnavailable",
    "JobWaitError",
]
