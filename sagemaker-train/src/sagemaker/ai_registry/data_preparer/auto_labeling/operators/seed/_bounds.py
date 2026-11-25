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
"""Exact one-sided bounds on a proportion, for sizing a sample that must certify.

The labeling job computes the certificate itself (``AcceptedLowerBound95`` in its
summary). What the client needs is the inverse question before any money is spent:
how many reviewed rows, all correct, does a target need before a one-sided
Clopper-Pearson bound at that confidence can reach it? With zero errors the bound
is ``alpha ** (1 / n)``, so the answer is ``ceil(ln(alpha) / ln(target))``: 19 rows
for 0.85, 29 for 0.90, 59 for 0.95 at ``alpha = 0.05``. One observed error needs
more rows than this; the number here is the floor, not a promise.
"""

from __future__ import absolute_import

import math

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

# The lowest accuracy target a certificate is defined for: the spot-check verdict's
# withdraw count is calibrated for the draws of 0.85 (19), 0.90 (29) and 0.95 (59) rows;
# below 0.85 there is no certificate, only the acceptance bar.
MIN_CALIBRATION_TARGET = 0.85
TARGET_FLOOR_MESSAGE = (
    "certification needs a target of at least 0.85; lower targets route on the bar only"
)


def zero_error_lower_bound(n: int, alpha: float = 0.05) -> float:
    """The one-sided ``(1 - alpha)`` Clopper-Pearson lower bound after ``n`` clean rows.

    ``alpha ** (1 / n)``, or 0.0 for no rows.
    """
    return alpha ** (1.0 / n) if n else 0.0


def best_zero_error_target(n: int, alpha: float = 0.05) -> float:
    """The highest target ``n`` rows with no error can certify, floored to 0.001."""
    return math.floor(zero_error_lower_bound(n, alpha) * 1000) / 1000


def calibration_sample_size(target: float, alpha: float = 0.05) -> int:
    """The smallest number of reviewed rows that can certify ``target``.

    Use it before drawing a seed or a spot-check sample, to know whether the rows
    on hand can reach the target at all: ``n`` rows with zero errors give a one-sided
    ``(1 - alpha)`` Clopper-Pearson lower bound of ``alpha ** (1 / n)``, so the
    smallest ``n`` whose bound reaches ``target`` is ``ceil(ln(alpha) / ln(target))``.
    Do not read it as the rows a run *will* need: a single observed error pushes the
    bound down and the run needs more.

    Args:
        target: The accuracy target, in ``[MIN_CALIBRATION_TARGET, 1)``. Below 0.85
            there is no certificate, only the acceptance bar, and the call is refused.
        alpha: One minus the confidence, in ``(0, 1)``; 0.05 is the job's 95%.

    Returns:
        The row count: 19 for 0.85, 29 for 0.90, 59 for 0.95 at ``alpha = 0.05``.

    Raises:
        DataPreparerOperatorError: ``target`` is not in ``(0, 1)``, is below
            ``MIN_CALIBRATION_TARGET``, or ``alpha`` is not in ``(0, 1)``.

    Example::

        calibration_sample_size(0.90)
        # 29
    """
    if isinstance(target, bool) or not 0.0 < target < 1.0:
        raise DataPreparerOperatorError(f"target must be in (0, 1), got {target!r}")
    if target < MIN_CALIBRATION_TARGET:
        raise DataPreparerOperatorError(f"{TARGET_FLOOR_MESSAGE} (got {target!r})")
    if isinstance(alpha, bool) or not 0.0 < alpha < 1.0:
        raise DataPreparerOperatorError(f"alpha must be in (0, 1), got {alpha!r}")
    n = math.log(alpha) / math.log(target)
    # guard float noise so an exact integer ratio is not bumped up by one
    return max(1, math.ceil(n - 1e-9))
