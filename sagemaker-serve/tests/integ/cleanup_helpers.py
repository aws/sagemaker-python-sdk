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
"""Shared, resilient resource cleanup for sagemaker-serve integration tests.

These tests each ``build()`` a model and ``deploy()`` a real SageMaker endpoint,
then delete it in a ``finally`` block. The historical pattern guarded cleanup on
the object handle returned by ``deploy()``::

    core_model = core_endpoint = None
    try:
        core_model, core_endpoint = build_and_deploy()   # deploy() may fail/hang
        ...
    finally:
        if core_model and core_endpoint:                 # <-- skipped on failure
            cleanup_resources(core_model, core_endpoint)

That guard is the root cause of the endpoint accumulation in the CI account: when
``deploy()`` raises (endpoint goes to ``Failed``) or is killed mid-call (build
timeout), ``core_endpoint`` is never assigned, so the ``if`` is falsy and the
endpoint SageMaker already created is never deleted. Deleting by the *handle* also
can't clean up a resource whose creation call never returned.

The helpers here delete by **name** instead, so cleanup no longer depends on
``deploy()`` returning. A test captures the names it is about to use up front and
always calls :func:`cleanup_by_name` in ``finally``. Each delete is best-effort
(:func:`delete_quietly`): a missing resource or a transient error on one delete
never prevents the others from running.
"""
from __future__ import absolute_import

import logging

from sagemaker.core.resources import Endpoint, EndpointConfig, Model

logger = logging.getLogger(__name__)


def delete_quietly(delete_fn, label):
    """Run a single best-effort delete, logging and swallowing any failure.

    ``delete_fn`` is a zero-arg callable that fetches and deletes one resource
    (e.g. ``lambda: Endpoint.get(endpoint_name=name).delete()``). It is expected
    to raise when the resource does not exist (already cleaned up / never
    created) or on a transient API error; either way we log and continue so a
    later delete in the same cleanup sequence still runs.
    """
    try:
        delete_fn()
        logger.info("Deleted %s", label)
    except Exception as exc:  # noqa: E722 - best-effort cleanup must never raise
        # Debug, not warning: a "does not exist" here is the common, expected
        # case (e.g. the endpoint config was never created because deploy()
        # failed early), and we don't want to red the log for normal cleanup.
        logger.debug("Skipping delete of %s: %s", label, exc)


def cleanup_by_name(
    endpoint_name=None,
    endpoint_config_name=None,
    model_name=None,
):
    """Best-effort delete of an endpoint, its endpoint config, and its model by name.

    Every argument is optional and independent; pass whatever names the test
    intended to create. Deletion order mirrors reverse order of creation
    (endpoint -> endpoint config -> model) so a live endpoint is torn down before
    the config it references. Because deletes are keyed on name (not on the
    objects ``deploy()`` returns), this cleans up resources even when the
    creating call failed or never returned.

    ``endpoint_config_name`` defaults to ``endpoint_name`` when omitted, matching
    ``ModelBuilder.deploy()``'s default of reusing the endpoint name for its
    auto-created endpoint config.
    """
    if endpoint_name and endpoint_config_name is None:
        endpoint_config_name = endpoint_name

    if endpoint_name:
        delete_quietly(
            lambda: Endpoint.get(endpoint_name=endpoint_name).delete(),
            f"Endpoint {endpoint_name}",
        )
    if endpoint_config_name:
        delete_quietly(
            lambda: EndpointConfig.get(
                endpoint_config_name=endpoint_config_name
            ).delete(),
            f"EndpointConfig {endpoint_config_name}",
        )
    if model_name:
        delete_quietly(
            lambda: Model.get(model_name=model_name).delete(),
            f"Model {model_name}",
        )
