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
"""A run id and the staging directory and S3 prefix it scopes, so concurrent runs do not collide.

Two labeling runs that share an output directory or an S3 base would otherwise write to
the same fixed internal paths: the local staging directory and the ``input/``, ``eval/``,
``labeling/`` and ``output/`` sub-prefixes each stage's job derives. Every run gets a
``run_id``, and :class:`RunContext` scopes those internal paths to it. The caller-chosen
output directory and S3 prefix are already caller-scoped; this isolates only the fixed
sub-paths derived from them.

There is no unscoped layout: every id, including the literal ``"default"``, is a real
sanitized scoped id with its own ``_sdg_work/<id>`` staging directory and ``run-<id>/``
S3 sub-prefix. Two runs that happened to share any one token would otherwise re-open the
overwrite this module prevents.
"""

from __future__ import absolute_import

import hashlib
import os
import re
import shutil
import time
from pathlib import Path

# S3 keys and local dir names: anything else (whitespace, slashes, separators) is squeezed
# to a filesystem- and S3-safe token.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

# The re-attach record a labeling run writes per stage; pruning keeps it.
JOB_RECORD = "job.json"


def sanitize_run_id(run_id):
    """A run id as a safe token for directory names and S3 keys.

    An already-safe id is kept as is. An id the sanitization changes gets a short SHA-256
    of the original appended, so ``run@1`` and ``run#1`` stay distinct. An id with no
    safe character (or a blank one) becomes ``id-`` and a longer hash. ``hashlib``, never
    the builtin ``hash()``, so the token is stable across processes.

    Args:
        run_id: The id as given; ``None`` counts as blank.

    Returns:
        The safe token.
    """
    original = "" if run_id is None else str(run_id)
    stripped = original.strip()
    s = _UNSAFE.sub("-", stripped).strip("-._")
    if not s:
        return "id-" + hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]
    if s == stripped:
        return s
    return s + "-" + hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]


def resolve_run_id(explicit=None, env=None):
    """The sanitized run id: the explicit argument, else the ``RUN_ID`` environment variable.

    Args:
        explicit: The id the caller passed, if any.
        env: The environment mapping; ``os.environ`` when ``None``.

    Returns:
        The safe token (:func:`sanitize_run_id`).
    """
    env = os.environ if env is None else env
    chosen = explicit or env.get("RUN_ID")
    return sanitize_run_id(chosen)


class RunContext:
    """A resolved run id and its staging directory and S3 prefix.

    The same id always gives the same ones, distinct ids distinct ones.
    """

    def __init__(self, run_id):
        """Sanitize ``run_id`` (:func:`sanitize_run_id`) as this context's id."""
        self.run_id = sanitize_run_id(run_id)

    def staging_dir(self, base, name="_sdg_work"):
        """The run-scoped local staging directory ``base/<name>/<run_id>``.

        Args:
            base: Typically the output directory.
            name: The staging directory's name under ``base``.

        Returns:
            The :class:`~pathlib.Path`.
        """
        return Path(base) / name / self.run_id

    def s3_prefix(self, base):
        """The run-scoped S3 base, ``base + run-<run_id>/``, always slash-terminated.

        Every sub-path the backend derives (each stage's ``input/``, ``eval/``,
        ``labeling/`` and ``output/``) lands under this run-private sub-prefix.

        Args:
            base: The caller's S3 prefix.

        Returns:
            The scoped prefix.
        """
        base = base if base.endswith("/") else base + "/"
        return base + "run-" + self.run_id + "/"

    def prune_old_scopes(self, base, keep, stale_after_s=86400, name="_sdg_work"):
        """Best-effort cleanup of old staging directories under ``base/<name>``.

        Only scopes idle longer than ``stale_after_s`` are eligible, since a recent one
        may be a concurrent run still going; the current run is never touched, and the
        ``keep`` newest idle scopes stay. Local only, never S3. Each scope's
        ``<stage>/job.json`` is kept, or a re-run of that run id would submit a duplicate
        job. A directory that cannot be removed is skipped.

        Args:
            base: The directory holding the staging directory.
            keep: How many idle scopes to keep; ``<= 0`` disables pruning.
            stale_after_s: Seconds of idleness before a scope is eligible.
            name: The staging directory's name under ``base``.

        Returns:
            The run ids pruned.
        """
        if not keep or keep <= 0:
            return []
        root = Path(base) / name
        if not root.is_dir():
            return []
        cutoff = time.time() - max(0, stale_after_s)

        def _scope_mtime(d):
            # Newest mtime in the whole subtree: a busy run writes nested files that do
            # not bump the top dir's mtime. A dir that vanishes meanwhile is skipped.
            try:
                newest = d.stat().st_mtime
            except OSError:
                return -1.0
            for sub in d.rglob("*"):
                try:
                    newest = max(newest, sub.stat().st_mtime)
                except OSError:
                    continue
            return newest

        eligible = [
            d
            for d in root.iterdir()
            if d.is_dir() and d.name != self.run_id and 0 <= _scope_mtime(d) <= cutoff
        ]
        eligible.sort(key=_scope_mtime, reverse=True)
        pruned = []
        for d in eligible[keep:]:  # keep the `keep` most-recent idle scopes
            try:
                records = [p for p in d.rglob(JOB_RECORD) if p.is_file()]
                if not records:
                    shutil.rmtree(d)
                else:
                    _prune_all_but(d, set(records))
                pruned.append(d.name)
            except OSError:
                pass  # locked / racing dir: skip, don't fail the run
        return pruned


def _prune_all_but(scope, keep_files):
    """Remove everything under ``scope`` except ``keep_files`` and the dirs holding them."""
    keep_dirs = {p.parent for p in keep_files}
    for k in list(keep_dirs):
        keep_dirs.update(k.parents)
    for p in sorted(scope.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p in keep_files or p in keep_dirs:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def run_context(explicit=None, env=None):
    """A :class:`RunContext` from an explicit id or the ``RUN_ID`` environment variable."""
    return RunContext(resolve_run_id(explicit, env))
