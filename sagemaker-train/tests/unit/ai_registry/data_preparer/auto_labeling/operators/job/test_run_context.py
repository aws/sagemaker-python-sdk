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
"""Unit tests for operators.common.run_context: run-id sanitizing, scoping, pruning."""

from __future__ import absolute_import

import hashlib
import os
import re
from pathlib import Path

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import _run_context as rc

SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
BLANK_TOKEN = "id-" + hashlib.sha256(b"").hexdigest()[:16]


@pytest.mark.parametrize("raw", ["runA", "job-42", "default", "keep._-9", "x" * 300, "a..b"])
def test_safe_ids_are_kept_verbatim(raw):
    assert rc.sanitize_run_id(raw) == raw
    assert rc.RunContext(raw).run_id == raw
    assert rc.sanitize_run_id(f"  {raw}  ") == raw  # surrounding whitespace is not a change


@pytest.mark.parametrize(
    "raw,stem",
    [
        ("a b/c:d", "a-b-c-d"),
        ("weird id/x", "weird-id-x"),
        ("run-1/", "run-1"),
        ("./a", "a"),
        ("a@1", "a-1"),
    ],
)
def test_altered_ids_get_a_deterministic_hash_suffix(raw, stem):
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    assert rc.sanitize_run_id(raw) == f"{stem}-{h}"
    assert rc.sanitize_run_id(raw) == rc.sanitize_run_id(raw)


def test_ids_differing_only_in_unsafe_chars_stay_distinct():
    assert rc.sanitize_run_id("a@1") != rc.sanitize_run_id("a#1")
    assert rc.sanitize_run_id("a b") != rc.sanitize_run_id("a/b")
    assert rc.RunContext("a b").s3_prefix("s3://b/") != rc.RunContext("a/b").s3_prefix("s3://b/")


@pytest.mark.parametrize("raw", [None, "", "   ", "\t", "---", "...", "é", "/", "@#$"])
def test_blank_or_unsafe_only_ids_become_scoped_hash_tokens(raw):
    tok = rc.sanitize_run_id(raw)
    assert tok.startswith("id-") and len(tok) == 19 and SAFE.match(tok)
    assert tok == "id-" + hashlib.sha256(("" if raw is None else raw).encode()).hexdigest()[:16]


def test_blank_and_none_share_the_blank_token():
    assert rc.sanitize_run_id(None) == rc.sanitize_run_id("") == BLANK_TOKEN


@pytest.mark.parametrize("raw", ["../..", "/etc", "a/../../b", "..\\x", "x" * 300, "a b/c:d"])
def test_output_is_always_a_safe_token_that_cannot_escape_the_base(raw):
    tok = rc.sanitize_run_id(raw)
    assert SAFE.match(tok) and "/" not in tok and "\\" not in tok
    staging = rc.RunContext(raw).staging_dir("/o")
    assert str(staging).startswith("/o/_sdg_work/") and staging.parent == Path("/o/_sdg_work")


def test_resolve_precedence_and_sanitizing():
    assert rc.resolve_run_id("explicit", env={"RUN_ID": "fromenv"}) == "explicit"
    assert rc.resolve_run_id(None, env={"RUN_ID": "fromenv"}) == "fromenv"
    assert rc.resolve_run_id(None, env={}) == BLANK_TOKEN
    assert rc.resolve_run_id("", env={"RUN_ID": "fromenv"}) == "fromenv"
    assert rc.resolve_run_id("   ", env={"RUN_ID": "fromenv"}).startswith("id-")  # truthy
    assert re.match(r"^weird-id-x-[0-9a-f]{8}$", rc.resolve_run_id("weird id/x", env={}))
    assert re.match(r"^env-id-y-[0-9a-f]{8}$", rc.resolve_run_id(None, env={"RUN_ID": "env id/y"}))
    assert rc.resolve_run_id("default", env={}) == "default"


def test_resolve_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("RUN_ID", "envglobal")
    assert rc.resolve_run_id() == "envglobal"
    assert rc.run_context().run_id == "envglobal"
    assert rc.run_context("x").run_id == "x"


def test_staging_dir_and_s3_prefix_are_scoped_stable_and_distinct():
    a, b, a2 = rc.RunContext("runA"), rc.RunContext("runB"), rc.RunContext("runA")
    assert a.staging_dir("/o") == Path("/o/_sdg_work/runA") == a2.staging_dir("/o")
    assert a.staging_dir("/o", name="stage") == Path("/o/stage/runA")
    assert a.staging_dir("/o") != b.staging_dir("/o")
    assert a.s3_prefix("s3://bkt/al") == "s3://bkt/al/run-runA/" == a.s3_prefix("s3://bkt/al/")
    assert a.s3_prefix("s3://bkt/a/b/c") == "s3://bkt/a/b/c/run-runA/"
    assert a.s3_prefix("s3://bkt/al") != b.s3_prefix("s3://bkt/al")
    assert rc.RunContext("default").s3_prefix("s3://b/") == "s3://b/run-default/"


# ---- pruning -----------------------------------------------------------------------------


def _mk(base, name, mtime, files=("marker",)):
    d = Path(base) / "_sdg_work" / name
    d.mkdir(parents=True, exist_ok=True)
    for rel in files:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    for p in sorted(d.rglob("*"), key=lambda x: -len(x.parts)):
        os.utime(p, (mtime, mtime))
    os.utime(d, (mtime, mtime))
    return d


def _names(base):
    return sorted(p.name for p in (Path(base) / "_sdg_work").iterdir())


def test_prune_keeps_the_newest_idle_scopes_and_the_current_run(tmp_path):
    for i, mt in enumerate([100, 200, 300, 400, 500]):
        _mk(tmp_path, f"old{i}", mt)
    ctx = rc.RunContext("current")
    _mk(tmp_path, ctx.run_id, 50)  # the active run, oldest, must survive
    pruned = ctx.prune_old_scopes(tmp_path, keep=2)
    assert set(pruned) == {"old0", "old1", "old2"}
    assert _names(tmp_path) == ["current", "old3", "old4"]


def test_a_pruned_scope_keeps_its_job_records(tmp_path):
    for i, mt in enumerate([100, 200, 300]):
        _mk(tmp_path, f"old{i}", mt, files=("val/job.json", "staged/input.jsonl", "marker"))
    pruned = rc.RunContext("current").prune_old_scopes(tmp_path, keep=1)
    assert set(pruned) == {"old0", "old1"}
    for name in ("old0", "old1"):
        d = tmp_path / "_sdg_work" / name
        assert sorted(str(p.relative_to(d)) for p in d.rglob("*")) == ["val", "val/job.json"]
    assert (tmp_path / "_sdg_work" / "old2" / "staged" / "input.jsonl").exists()


def test_keep_zero_or_negative_disables_pruning(tmp_path):
    for i, mt in enumerate([1, 2, 3]):
        _mk(tmp_path, f"s{i}", mt)
    ctx = rc.RunContext("cur")
    assert ctx.prune_old_scopes(tmp_path, 0) == [] and ctx.prune_old_scopes(tmp_path, -5) == []
    assert ctx.prune_old_scopes(tmp_path, None) == []
    assert len(_names(tmp_path)) == 3


def test_recent_scopes_are_not_eligible(tmp_path):
    import time

    _mk(tmp_path, "busy", time.time())
    _mk(tmp_path, "old", 100)
    _mk(tmp_path, "older", 50)
    assert rc.RunContext("cur").prune_old_scopes(tmp_path, keep=1) == ["older"]
    assert _names(tmp_path) == ["busy", "old"]
    assert rc.RunContext("cur").prune_old_scopes(tmp_path, keep=1, stale_after_s=0) == ["old"]


def test_a_busy_scope_counts_its_newest_nested_file(tmp_path):
    import time

    d = _mk(tmp_path, "nested", 100, files=("a/b/c.txt",))
    os.utime(d / "a" / "b" / "c.txt", (time.time(), time.time()))
    _mk(tmp_path, "old", 200)
    assert rc.RunContext("cur").prune_old_scopes(tmp_path, keep=0, stale_after_s=0) == []
    assert rc.RunContext("cur").prune_old_scopes(tmp_path, keep=1) == []  # nested is not idle
    # a negative staleness counts as zero: everything is idle, the newest (nested) is kept
    assert rc.RunContext("cur").prune_old_scopes(tmp_path, keep=1, stale_after_s=-5) == ["old"]


def test_missing_root_and_siblings(tmp_path):
    assert rc.RunContext("cur").prune_old_scopes(tmp_path / "nope", keep=1) == []
    cache = tmp_path / "_sdg_phase1_cache" / "phase1_abc"
    cache.mkdir(parents=True)
    (cache / "generation.py").write_text("x")
    for i, mt in enumerate([1, 2, 3, 4, 5, 6]):
        _mk(tmp_path, f"s{i}", mt)
    (tmp_path / "_sdg_work" / "a-file").write_text("not a scope")
    assert set(rc.RunContext("cur").prune_old_scopes(tmp_path, keep=1)) == {
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
    }
    assert cache.exists() and (tmp_path / "_sdg_work" / "a-file").exists()


def test_prune_all_but_keeps_files_and_their_dirs(tmp_path):
    scope = tmp_path / "scope"
    for rel in ("val/job.json", "val/other.txt", "dev/job.json", "staged/x/y.jsonl", "top.txt"):
        p = scope / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    rc._prune_all_but(scope, {scope / "val" / "job.json", scope / "dev" / "job.json"})
    assert sorted(str(p.relative_to(scope)) for p in scope.rglob("*")) == [
        "dev",
        "dev/job.json",
        "val",
        "val/job.json",
    ]
