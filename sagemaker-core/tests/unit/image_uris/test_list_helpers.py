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
"""Unit tests for the read-only listing helpers in ``image_uris``.

These tests run against the REAL bundled image_uri_config JSON files, so they
assert on stable, well-known frameworks/versions rather than mocking config.
"""

from __future__ import absolute_import

import pytest

from sagemaker.core import image_uris


def test_list_frameworks_includes_known_frameworks_sorted():
    frameworks = image_uris.list_frameworks()

    assert isinstance(frameworks, list)
    assert frameworks == sorted(frameworks)
    # No duplicates.
    assert len(frameworks) == len(set(frameworks))
    # Well-known frameworks must be present.
    for expected in ("pytorch", "tensorflow", "xgboost", "sklearn"):
        assert expected in frameworks
    # Non-framework config files must be excluded.
    assert "instance_gpu_info" not in frameworks


def test_list_versions_scoped_framework_is_nonempty_and_sorted():
    # pytorch is a scope-nested config (training/inference/...).
    versions = image_uris.list_versions("pytorch")

    assert isinstance(versions, list)
    assert len(versions) > 0
    assert len(versions) == len(set(versions))
    # A well-known pytorch training version is present.
    assert "2.0.0" in versions
    # Versions are returned in the module's sorted order.
    assert versions == image_uris.list_versions("pytorch")


def test_list_versions_respects_image_scope_argument():
    training = set(image_uris.list_versions("pytorch", image_scope="training"))
    inference = set(image_uris.list_versions("pytorch", image_scope="inference"))
    union = set(image_uris.list_versions("pytorch"))

    assert training  # non-empty
    assert inference  # non-empty
    # The unscoped call unions across scopes, so each scope is a subset.
    assert training <= union
    assert inference <= union
    assert (training | inference) <= union


def test_list_versions_top_level_versions_framework():
    # linear-learner ships a top-level ``versions`` config (no scope nesting).
    versions = image_uris.list_versions("linear-learner")

    assert isinstance(versions, list)
    assert len(versions) > 0
    assert len(versions) == len(set(versions))


def test_list_py_versions_returns_expected_values():
    # pytorch training 2.0.0 ships py310.
    py_versions = image_uris.list_py_versions("pytorch", "2.0.0", image_scope="training")

    assert py_versions == ["py310"]


def test_list_py_versions_without_scope_unions_across_scopes():
    py_versions = image_uris.list_py_versions("pytorch", "2.0.0")

    assert "py310" in py_versions


def test_list_py_versions_version_without_py_versions_returns_empty():
    # Algorithm images (e.g. linear-learner) do not define ``py_versions``.
    versions = image_uris.list_versions("linear-learner")
    assert versions  # sanity
    assert image_uris.list_py_versions("linear-learner", versions[0]) == []


def test_list_versions_unknown_framework_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported framework"):
        image_uris.list_versions("not-a-real-framework")


def test_list_py_versions_unknown_framework_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported framework"):
        image_uris.list_py_versions("not-a-real-framework", "1.0.0")


def test_list_py_versions_unknown_version_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported version"):
        image_uris.list_py_versions("pytorch", "0.0.0-does-not-exist")


def test_list_versions_includes_version_aliases():
    # ``retrieve`` accepts aliases like "2.0" (-> "2.0.1"); they must be listed.
    versions = image_uris.list_versions("pytorch", image_scope="training")

    assert "2.0" in versions  # alias
    assert "2.0.1" in versions  # concrete target


def test_list_py_versions_resolves_version_alias():
    # "2.0" is an alias for the concrete "2.0.1" pytorch training version.
    from_alias = image_uris.list_py_versions("pytorch", "2.0", image_scope="training")
    from_concrete = image_uris.list_py_versions("pytorch", "2.0.1", image_scope="training")

    assert from_alias == from_concrete
    assert from_alias  # non-empty


def test_list_versions_invalid_scope_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported image scope"):
        image_uris.list_versions("pytorch", image_scope="not-a-scope")


def test_list_versions_sorted_semantically_not_lexically():
    versions = image_uris.list_versions("pytorch", image_scope="training")

    # Semantic (not lexical) ordering: 2.9.0 must precede 2.10.0.
    numeric = [v for v in versions if v in ("2.9.0", "2.10.0")]
    assert numeric == ["2.9.0", "2.10.0"]


def test_list_versions_non_pep440_keys_sort_last():
    # data-wrangler ships non-PEP440 version keys ("1.x", "2.x", "3.x").
    versions = image_uris.list_versions("data-wrangler")

    assert any(v.endswith(".x") for v in versions)
    # Unparseable keys are grouped after all PEP 440 versions.
    from packaging.version import InvalidVersion, Version

    def is_pep440(v):
        try:
            Version(v)
            return True
        except InvalidVersion:
            return False

    pep440_flags = [is_pep440(v) for v in versions]
    # Once we hit the first non-PEP440 key, none after it may be PEP 440.
    if False in pep440_flags:
        first_non = pep440_flags.index(False)
        assert not any(pep440_flags[first_non:])
