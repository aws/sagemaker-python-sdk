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
"""Regression guard: shipped and public DataPreparer text must not contain
internal-only references.

Scans the following paths relative to the repo root:
  - sagemaker-train/src/sagemaker/ai_registry/data_preparer/
  - sagemaker-train/tests/integ/ai_registry/data_preparer/
  - docs/data_preparation/
  - docs/training/data_preparer.rst
  - sagemaker-train/tests/unit/ai_registry/data_preparer/ (hashed internal names only)
  - v3-examples/data-preparer-*.ipynb (except entries in ``_FILE_ALLOWLIST``)

Reports any match as a file:line failure.

Internal names are stored only as SHA-256 hashes so this file does not itself
publish them. To add a name, hash its normalized key and append it to the
matching list:

  identifier names (matched inside one token, any case, e.g. ``FooBar``,
  ``foobar``, ``XFooBar``, ``foo_bar``): key = the CamelCase / underscore parts
  of the name, lowercased and concatenated (``FooBar`` -> ``foobar``)::

      python -c "import hashlib; print(hashlib.sha256(b'foobar').hexdigest())"

  phrases (matched across consecutive words, any case or separator): key = the
  words lowercased and joined by single spaces (``Foo Bar`` -> ``foo bar``).
"""

from __future__ import absolute_import

import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Hashed internal names (see module docstring for how to add one).
# ---------------------------------------------------------------------------
_BANNED_IDENTIFIER_HASHES = frozenset(
    {
        "ec4f5d7aae5c8908eefc7edd0058e37189f26eda72c8dbd5b94e5166ef70e64d",
        "afb1b81f6e54590f7faf23feb2f7f27334cde8840e6a27e007dcee8e75f7305d",
        "8758607d0bcaa8a6efc8f92aa63bf367a6f088001d8fd2f16d1583c356a1a23f",
        "2a0abe451aede7f7139e3b7be00c2adff97b5ef4a50c6b5f4a2165125b55cc15",
        "caddda9701a92268812392382b0d2d4aa1af5ff495159bfaf7904f5d16e2a517",
        "c42e18f7977c78a4ed8895384e5bdd76d66b30f9cb41fb56e35516f3081ce87d",
        "d5220eb13adce7752282ad04c417cb10ae5b698ee92ccd773f8df17c5153b7fa",
        "6d94f54ea44d03bfa323831fc80386adf89f7353f45731671e6eb4bd3055f9ef",
        "b002c7748bf0f65e4a3ab1bffc483eebdc1d126277e6dc5448399da9ad5b7c26",
        "281692523ae08146631c8f5ec31ad076ca035138a92907597ac186325f563757",
        "4f278cdddf52263fe21c64c94932f2b2ec316acecd39a7adcc01eb2e6592a678",
        "e67720146b51ea99c80b5333aceb6b07334d6a827f882dc34ef38c82d96b7c5e",
        "6c6ea56757d47fc8827e5e1a28ec7f6018c593142fc731c274c6e968717535ac",
        "cfce86f824f6456fc848a54ca2362d4f1b463a47c5718dc157a47a394ef63b5d",
        "e05e02f39f81dd081a058f3606eac1715831ee8183d5de6958f310181d9e40a1",
        "a67a9eea354757c05d9f5cc8f2e80c708fc47b6e696991bd1c9292f6c72a5924",
        "f73d3b17f5c5ca935ac16403f4358c6f4bf6ed772104439ee76461702ab2194e",
        "9ff31e78943bf133aa635becbef429d94d775a4d7d07832c163a553c2915eaaf",
        "b109ae687fd57e7962e3d401c43c284b0662cc83547bd04ed76f4db6be25e64c",
        "ff7c72b295126756d83c880cda4ec21d9fc5197636ee4c035a80602e3d366201",
    }
)
_BANNED_PHRASE_HASHES = frozenset(
    {
        "bce3903215e56ecf32bdf102da0115700753c98f19b325ad3a4769aa5b15454f",
        "bca4448761ced653d0fa7217bcd5ded46c9cb1d2e03812be758997fc8577bf36",
        "8ccb0c861c201ac1a589a62baf9c13c7ad760e02b2607cd6bf48ec5dcaef3e41",
        "2b721bb50ce3654dcae442ad216882aa2ac3036a886cd45e6b72802561d20bc9",
        "87dbca0746da73be012a45c7f5a969dd6a7db047cb914fe4bd3b06c76eacebb9",
    }
)

# Identifier parts: CamelCase words, keeping letter/digit runs together so that
# e.g. "ec9x" stays one part and does not contain a shorter banned key.
_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z0-9]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_MAX_PARTS = 6
_MAX_PHRASE_WORDS = 3


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identifier_keys(line: str) -> Iterator[str]:
    """Yield every run of consecutive CamelCase/underscore parts within a token."""
    for token in _TOKEN_RE.findall(line):
        parts = [p.lower() for p in _PART_RE.findall(token)]
        for i in range(len(parts)):
            for j in range(i + 1, min(i + _MAX_PARTS, len(parts)) + 1):
                yield "".join(parts[i:j])


def _phrase_keys(line: str) -> Iterator[str]:
    """Yield every run of 2..3 consecutive words, lowercased and space-joined."""
    words = [w.lower() for w in _WORD_RE.findall(line)]
    for i in range(len(words)):
        for n in range(2, _MAX_PHRASE_WORDS + 1):
            if i + n <= len(words):
                yield " ".join(words[i : i + n])


def _hashed_hits(line: str) -> List[str]:
    hits = [k for k in _identifier_keys(line) if _sha256(k) in _BANNED_IDENTIFIER_HASHES]
    hits += [k for k in _phrase_keys(line) if _sha256(k) in _BANNED_PHRASE_HASHES]
    return hits


# ---------------------------------------------------------------------------
# Generic patterns that must NOT appear in scanned files.
# ---------------------------------------------------------------------------
_BANNED_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("alpha endpoint", re.compile(r"alpha\s+endpoint", re.IGNORECASE)),
    ("internal api/model", re.compile(r"\binternal\s+(?:api|apis|model)\b", re.IGNORECASE)),
    # Backend Java class / file names
    ("java file reference", re.compile(r"\.java\b")),
    # Internal-only service principals (e.g. "<name>.im.aws.internal"); customers
    # can only trust public principals such as "sagemaker.amazonaws.com".
    ("internal service principal", re.compile(r"[a-z0-9-]+\.aws\.internal\b", re.IGNORECASE)),
    # Preview-era install wording; public docs install from PyPI.
    ("beta installation instructions", re.compile(r"beta\s+installation", re.IGNORECASE)),
    ("preview access", re.compile(r"preview\s+access", re.IGNORECASE)),
]

# AWS account ID: 12 consecutive digits, not preceded or followed by another digit.
# Exempt well-known placeholder values.
_ACCOUNT_ID_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")
_ACCOUNT_ID_ALLOWLIST = {"123456789012", "111122223333"}

# ---------------------------------------------------------------------------
# Explicit file-level allowlist. Keep this as small as possible: every entry
# is a file the guard does not protect.
# ---------------------------------------------------------------------------
_FILE_ALLOWLIST = {
    # TODO: the golden SDG-to-DQ notebook still carries saved outputs and
    # account-specific setup cells. It is being reworked separately; remove
    # this entry when that lands so notebooks are fully covered.
    "data-preparer-golden-sdg-to-dq.ipynb",
}


def _repo_root() -> Path:
    """Locate the repo root by walking up from this test file until we find
    the ``sagemaker-train`` directory.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sagemaker-train").is_dir():
            return parent
    raise RuntimeError(
        f"Could not locate repo root from {__file__}. "
        "Expected to find a 'sagemaker-train' directory in an ancestor."
    )


def _collect_files(repo_root: Path) -> List[Path]:
    """Return all text files in the scanned paths."""
    paths: List[Path] = []

    def add_dir(rel: str) -> None:
        d = repo_root / rel
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    paths.append(p)

    def add_file(rel: str) -> None:
        f = repo_root / rel
        if f.is_file():
            paths.append(f)

    def add_glob(pattern: str) -> None:
        for p in sorted(glob.glob(str(repo_root / pattern))):
            pf = Path(p)
            if pf.is_file():
                paths.append(pf)

    add_dir("sagemaker-train/src/sagemaker/ai_registry/data_preparer")
    add_dir("sagemaker-train/tests/integ/ai_registry/data_preparer")
    add_dir("docs/data_preparation")
    add_file("docs/training/data_preparer.rst")
    add_glob("v3-examples/data-preparer-*.ipynb")

    return paths


def _collect_unit_test_files(repo_root: Path) -> List[Path]:
    """Return DataPreparer unit-test files, minus this guard itself.

    These are scanned for hashed internal names only: they use placeholder
    account IDs and generic wording on purpose.
    """
    d = repo_root / "sagemaker-train/tests/unit/ai_registry/data_preparer"
    here = Path(__file__).resolve()
    return [p for p in sorted(d.rglob("*.py")) if p.is_file() and p.resolve() != here]


def _extract_text_lines(path: Path) -> List[str]:
    """Extract plain text lines from a file.

    Notebooks (``.ipynb``) are parsed as JSON; cell sources are concatenated.
    Binary files that cannot be decoded as UTF-8 are silently skipped (they
    contain no internal text references).
    """
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        lines: List[str] = []
        for cell in nb.get("cells", []):
            cell_src = "".join(cell.get("source", []))
            lines.extend(cell_src.splitlines())
        return lines
    try:
        return path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (UnicodeDecodeError, OSError):
        return []


def _scan_files(files: List[Path]) -> List[str]:
    """Scan files and return a list of violation strings (``file:line: message``)."""
    violations: List[str] = []

    for path in files:
        if path.name in _FILE_ALLOWLIST:
            continue

        lines = _extract_text_lines(path)
        for lineno, line in enumerate(lines, start=1):
            for label, pattern in _BANNED_PATTERNS:
                if pattern.search(line):
                    violations.append(f"{path}:{lineno}: banned term [{label}]: {line.strip()}")
            if _hashed_hits(line):
                violations.append(f"{path}:{lineno}: internal name: {line.strip()}")

            for match in _ACCOUNT_ID_RE.finditer(line):
                candidate = match.group(1)
                if candidate not in _ACCOUNT_ID_ALLOWLIST:
                    violations.append(
                        f"{path}:{lineno}: AWS account ID [{candidate}]: {line.strip()}"
                    )

    return violations


# ---------------------------------------------------------------------------
# Self-test: verify the scanner actually catches known patterns in memory.
# ---------------------------------------------------------------------------


def _run_scanner_on_text(text: str) -> List[str]:
    """Run the scanner against an in-memory string (not a file).

    Used by the self-test to verify each banned pattern fires.
    """
    violations: List[str] = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        for label, pattern in _BANNED_PATTERNS:
            if pattern.search(line):
                violations.append(f"<memory>:{lineno}: banned term [{label}]: {line.strip()}")
        if _hashed_hits(line):
            violations.append(f"<memory>:{lineno}: internal name: {line.strip()}")
        for match in _ACCOUNT_ID_RE.finditer(line):
            candidate = match.group(1)
            if candidate not in _ACCOUNT_ID_ALLOWLIST:
                violations.append(
                    f"<memory>:{lineno}: AWS account ID [{candidate}]: {line.strip()}"
                )
    return violations


class TestScannerSelfTest:
    """Verify the scanner's matching rules (in-memory).

    Hashed names are exercised with stand-in terms patched into the hash sets,
    so this file never spells out a real internal name.
    """

    @staticmethod
    def _with_terms(monkeypatch, identifiers=(), phrases=()):
        monkeypatch.setattr(
            __import__(__name__, fromlist=["_"]),
            "_BANNED_IDENTIFIER_HASHES",
            frozenset(_sha256(t) for t in identifiers),
        )
        monkeypatch.setattr(
            __import__(__name__, fromlist=["_"]),
            "_BANNED_PHRASE_HASHES",
            frozenset(_sha256(t) for t in phrases),
        )

    def test_hash_lists_are_well_formed(self):
        for h in _BANNED_IDENTIFIER_HASHES | _BANNED_PHRASE_HASHES:
            assert re.fullmatch(r"[0-9a-f]{64}", h), h
        assert len(_BANNED_IDENTIFIER_HASHES) == 20
        assert len(_BANNED_PHRASE_HASHES) == 5

    @pytest.mark.parametrize(
        "text",
        [
            "routed by FooBarService",
            "see foobar for details",
            "loaded from XFooBar library",
            "call foo_bar() here",
            "FOOBAR",
        ],
    )
    def test_identifier_name_flagged_in_any_form(self, monkeypatch, text):
        self._with_terms(monkeypatch, identifiers=["foobar"])
        assert _run_scanner_on_text(text), f"missed stand-in identifier in {text!r}"

    @pytest.mark.parametrize("text", ["see xfoobar for details", "foobarbaz", "foo bar"])
    def test_identifier_respects_part_boundaries(self, monkeypatch, text):
        self._with_terms(monkeypatch, identifiers=["foobar"])
        assert not _run_scanner_on_text(text), f"false positive on {text!r}"

    def test_digit_runs_stay_one_part(self, monkeypatch):
        # A short key with digits must not match inside a longer lowercase token.
        self._with_terms(monkeypatch, identifiers=["c9x"])
        assert _run_scanner_on_text("uses the C9X fixture")
        assert not _run_scanner_on_text("some ec9x string")

    @pytest.mark.parametrize("text", ["run qux quux update", "Qux-Quux", "doc.qux.quux.com"])
    def test_phrase_flagged_across_words(self, monkeypatch, text):
        self._with_terms(monkeypatch, phrases=["qux quux"])
        assert _run_scanner_on_text(text), f"missed stand-in phrase in {text!r}"

    def test_phrase_not_flagged_when_words_apart(self, monkeypatch):
        self._with_terms(monkeypatch, phrases=["qux quux"])
        assert not _run_scanner_on_text("qux then quux")

    def test_account_id_flagged(self):
        assert _run_scanner_on_text("--account=987654321098"), "Scanner missed 12-digit account ID"

    @pytest.mark.parametrize("account_id", ["123456789012", "111122223333"])
    def test_placeholder_account_ids_not_flagged(self, account_id):
        assert not _run_scanner_on_text(f"account_id = '{account_id}'")

    @pytest.mark.parametrize(
        "text",
        [
            "requires the alpha endpoint to be configured",
            "loaded via an internal model path",
            "calls the internal API directly",
            "spellings come from Something.java",
            '"Service": ["sagemaker.amazonaws.com", "example.im.aws.internal"]',
            "see Beta Installation instructions",
            "provided with your preview access",
        ],
    )
    def test_generic_patterns_flagged(self, text):
        assert _run_scanner_on_text(text), f"Scanner missed {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "see foo.javascript for details",
            '"Service": "sagemaker.amazonaws.com"',
            "pip install --upgrade sagemaker",
        ],
    )
    def test_generic_patterns_not_flagged(self, text):
        assert not _run_scanner_on_text(text), f"Scanner false-positived on {text!r}"

    def test_scanner_returns_empty_for_clean_text(self):
        clean = (
            "import boto3\n"
            "# This notebook uses the public SageMaker SDK.\n"
            "account_id = '123456789012'  # placeholder\n"
        )
        result = _run_scanner_on_text(clean)
        assert not result, f"Scanner false-positived on clean text: {result}"


class TestNoInternalReferences:
    """Integration guard: no banned terms in shipped DataPreparer files."""

    def test_no_internal_references_in_shipped_files(self):
        repo_root = _repo_root()
        files = _collect_files(repo_root)
        assert files, (
            f"No files found to scan under {repo_root}. "
            "Check that the working directory is inside the sagemaker-python-sdk repo."
        )
        violations = _scan_files(files)
        if violations:
            report = "\n".join(violations)
            raise AssertionError(
                f"Found {len(violations)} internal reference(s) in shipped DataPreparer files:\n"
                f"{report}"
            )

    def test_no_internal_names_in_unit_tests(self):
        files = _collect_unit_test_files(_repo_root())
        assert files, "No DataPreparer unit-test files found to scan."
        violations = [
            f"{path}:{lineno}: internal name: {line.strip()}"
            for path in files
            for lineno, line in enumerate(_extract_text_lines(path), start=1)
            if _hashed_hits(line)
        ]
        assert not violations, "Internal names in DataPreparer unit tests:\n" + "\n".join(
            violations
        )
