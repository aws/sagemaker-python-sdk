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
"""SageMaker Experiments module for tracking experiments, trials, and runs."""
from __future__ import absolute_import

# Lazy imports to avoid circular dependencies during package initialization
# Users should import directly from the specific modules:
# from sagemaker.core.experiments.experiment import Experiment
# from sagemaker.core.experiments.run import Run
# etc.

__all__ = [
    "Experiment",
    "Run",
    "_RunContext",
    "_Trial",
    "_TrialComponent",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "Experiment":
        from sagemaker.core.experiments.experiment import Experiment

        return Experiment
    elif name == "Run":
        from sagemaker.core.experiments.run import Run

        return Run
    elif name == "_RunContext":
        from sagemaker.core.experiments._run_context import _RunContext

        return _RunContext
    elif name == "_Trial":
        from sagemaker.core.experiments.trial import _Trial

        return _Trial
    elif name == "_TrialComponent":
        from sagemaker.core.experiments.trial_component import _TrialComponent

        return _TrialComponent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
