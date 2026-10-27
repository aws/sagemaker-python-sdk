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
"""Bounded wait for deep integ tests that train on a HyperPod (EKS) cluster.

``SFTTrainer.train()`` on HyperPod submits a Kubeflow PyTorchJob through the
HyperPod CLI and returns its name. Completion is signalled by
``{s3_output_path}/{job_name}/manifest.json``, which a service-managed component
copies from the cluster to S3 after training. Polling only for that manifest
cannot tell "still training" from "never scheduled" or "succeeded but the
outputs were never synced", so a broken cluster made such tests poll for hours
until the CodeBuild build timed out, and every run left another job queued in
the cluster.

``wait_for_hyperpod_job`` also watches the PyTorchJob itself:

* the job failed -> fail immediately with its condition message;
* the job never starts running within ``schedule_timeout`` (no status, or pods
  that stay Pending) -> skip as a cluster capacity/scheduling problem;
* the job runs longer than ``run_timeout`` -> fail;
* the job succeeded but no manifest appears within ``manifest_timeout`` -> fail
  with a message pointing at the output sync.

Whatever the outcome, the job is removed from the cluster when the wait ends
(``helm uninstall`` of the release the HyperPod CLI created, falling back to
deleting the PyTorchJob), so tests no longer leave jobs behind.
"""

from __future__ import absolute_import

import logging
import subprocess
import time

import pytest

from sagemaker.train.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

PYTORCHJOB_GROUP = "kubeflow.org"
PYTORCHJOB_VERSION = "v1"
PYTORCHJOB_PLURAL = "pytorchjobs"

# Healthy HyperPod Nova Micro SFT jobs in this suite were scheduled within
# seconds and finished in ~25-55 minutes, and the manifest followed within a
# few minutes. Worst case for one test: 30 + 75 + 15 minutes, well inside the
# CI build budget.
DEFAULT_SCHEDULE_TIMEOUT_SECONDS = 30 * 60
DEFAULT_RUN_TIMEOUT_SECONDS = 75 * 60
DEFAULT_MANIFEST_TIMEOUT_SECONDS = 15 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 60


def _load_k8s_clients():
    """Return (CustomObjectsApi, CoreV1Api) for the kubeconfig written by
    ``hyperpod connect-cluster`` during ``train()``.

    ``kubernetes`` is a dependency of the HyperPod CLI, which these tests need
    anyway, so it is imported lazily.
    """
    from kubernetes import client, config

    config.load_kube_config()
    return client.CustomObjectsApi(), client.CoreV1Api()


def _true_condition(job_obj, condition_type):
    for cond in (job_obj.get("status") or {}).get("conditions") or []:
        if cond.get("type") == condition_type and str(cond.get("status")) == "True":
            return cond
    return None


def _pod_phases(core_api, namespace, job_name):
    pods = core_api.list_namespaced_pod(
        namespace, label_selector=f"training.kubeflow.org/job-name={job_name}"
    )
    return [p.status.phase for p in pods.items]


def delete_hyperpod_job(job_name, namespace="kubeflow", custom_api=None):
    """Best-effort removal of a HyperPod CLI job from the cluster.

    The CLI installs each job as a Helm release named after the job (one
    ConfigMap plus one PyTorchJob), so ``helm uninstall`` removes both. If helm is
    unavailable or fails, delete the PyTorchJob directly.
    """
    try:
        result = subprocess.run(
            ["helm", "uninstall", job_name, "--namespace", namespace],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Removed HyperPod job %s (helm uninstall)", job_name)
            return
        logger.warning("helm uninstall %s failed: %s", job_name, result.stderr.strip())
    except FileNotFoundError:
        logger.warning("helm not on PATH; deleting PyTorchJob %s directly", job_name)
    try:
        if custom_api is None:
            custom_api, _ = _load_k8s_clients()
        custom_api.delete_namespaced_custom_object(
            PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, namespace, PYTORCHJOB_PLURAL, job_name
        )
        logger.info("Deleted PyTorchJob %s", job_name)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to delete PyTorchJob %s: %s", job_name, e)


def wait_for_hyperpod_job(
    job_name,
    output_s3_path,
    sagemaker_session=None,
    namespace="kubeflow",
    schedule_timeout=DEFAULT_SCHEDULE_TIMEOUT_SECONDS,
    run_timeout=DEFAULT_RUN_TIMEOUT_SECONDS,
    manifest_timeout=DEFAULT_MANIFEST_TIMEOUT_SECONDS,
    poll_interval=DEFAULT_POLL_INTERVAL_SECONDS,
    cleanup=True,
):
    """Wait for a HyperPod PyTorchJob and its S3 manifest, failing fast.

    Args:
        job_name (str): Job name returned by ``train(wait=False)``.
        output_s3_path (str): The trainer's ``s3_output_path``.
        sagemaker_session: Session used to read the manifest from S3.
        namespace (str): Kubernetes namespace the job was submitted to.
        schedule_timeout (int): Seconds the job may wait to start running before
            the test is skipped.
        run_timeout (int): Seconds the job may run before the test fails.
        manifest_timeout (int): Seconds after the job succeeded that the manifest
            may take to appear before the test fails.
        poll_interval (int): Seconds between checks.
        cleanup (bool): Remove the job from the cluster when the wait ends.

    Returns:
        str: The checkpoint S3 URI from the manifest.
    """
    custom_api, core_api = _load_k8s_clients()
    start = time.time()
    running_since = None
    succeeded_since = None
    # Hard cap so persistent API errors cannot keep the loop alive past the phases.
    max_wait = schedule_timeout + run_timeout + manifest_timeout
    try:
        while time.time() - start < max_wait:
            checkpoint = BaseTrainer._resolve_checkpoint_from_manifest(
                job_name=job_name,
                output_s3_path=output_s3_path,
                sagemaker_session=sagemaker_session,
            )
            if checkpoint:
                logger.info("Job %s produced checkpoint %s", job_name, checkpoint)
                return checkpoint

            now = time.time()
            try:
                job_obj = custom_api.get_namespaced_custom_object(
                    PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, namespace, PYTORCHJOB_PLURAL, job_name
                )
                phases = _pod_phases(core_api, namespace, job_name)
            except Exception as e:  # pylint: disable=broad-except
                if getattr(e, "status", None) == 404:
                    pytest.fail(f"PyTorchJob {job_name} no longer exists in {namespace}.")
                logger.warning("Could not read PyTorchJob %s (will retry): %s", job_name, e)
                time.sleep(poll_interval)
                continue

            failed = _true_condition(job_obj, "Failed")
            if failed:
                pytest.fail(
                    f"HyperPod job {job_name} failed: {failed.get('reason')}: "
                    f"{failed.get('message')}"
                )

            if _true_condition(job_obj, "Succeeded"):
                succeeded_since = succeeded_since or now
                if now - succeeded_since > manifest_timeout:
                    pytest.fail(
                        f"HyperPod job {job_name} succeeded but no manifest appeared at "
                        f"{output_s3_path.rstrip('/')}/{job_name}/manifest.json within "
                        f"{manifest_timeout}s; the cluster's output sync to S3 is not "
                        f"delivering job outputs."
                    )
            else:
                if _true_condition(job_obj, "Running") or "Running" in phases:
                    running_since = running_since or now
                    if now - running_since > run_timeout:
                        pytest.fail(
                            f"HyperPod job {job_name} still running after {run_timeout}s "
                            f"(pod phases: {phases})."
                        )
                elif running_since is None and now - start > schedule_timeout:
                    pytest.skip(
                        f"HyperPod job {job_name} did not start within {schedule_timeout}s "
                        f"(job conditions: {(job_obj.get('status') or {}).get('conditions')}, "
                        f"pod phases: {phases}); the cluster has no schedulable capacity."
                    )
            time.sleep(poll_interval)
        pytest.fail(f"HyperPod job {job_name} did not complete within {max_wait}s.")
    finally:
        if cleanup:
            delete_hyperpod_job(job_name, namespace=namespace, custom_api=custom_api)
