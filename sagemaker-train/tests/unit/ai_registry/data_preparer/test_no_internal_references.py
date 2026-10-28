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
  - v3-examples/data-preparer-*.ipynb (except entries in ``_FILE_ALLOWLIST``)

Reports any match as a file:line failure.
"""
from __future__ import absolute_import

import glob
import json
import re
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Patterns that must NOT appear in scanned files (case-insensitive).
# ---------------------------------------------------------------------------
_BANNED_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("c2j", re.compile(r"\bc2j\b", re.IGNORECASE)),
    ("duplo", re.compile(r"duplo", re.IGNORECASE)),
    ("isengard", re.compile(r"isengard", re.IGNORECASE)),
    ("ada credentials", re.compile(r"ada\s+credentials", re.IGNORECASE)),
    ("alpha endpoint", re.compile(r"alpha\s+endpoint", re.IGNORECASE)),
    ("internal api/model/c2j", re.compile(r"\binternal\s+(?:api|apis|model|c2j)\b", re.IGNORECASE)),
    ("a2z.com", re.compile(r"a2z\.com", re.IGNORECASE)),
    ("corp.amazon", re.compile(r"corp\.amazon", re.IGNORECASE)),
    ("midway", re.compile(r"\bmidway\b", re.IGNORECASE)),
    # Backend Java class / file names
    ("java file reference", re.compile(r"\.java\b")),
    ("JobMetadataValidator", re.compile(r"JobMetadataValidator")),
    ("networknt validator", re.compile(r"networknt")),
    ("OperatorType enum", re.compile(r"\bOperatorType\b")),
    ("DataCurator service", re.compile(r"DataCurator")),
    ("MetricEngine", re.compile(r"MetricEngine")),
    ("TransformComputeRouter", re.compile(r"TransformComputeRouter")),
    # Internal service names
    ("IronMan internal name", re.compile(r"\bIronMan\b")),
    ("RoleProxy", re.compile(r"\bRoleProxy\b")),
    # Internal-only service principals (e.g. "<name>.im.aws.internal"); customers
    # can only trust public principals such as "sagemaker.amazonaws.com".
    ("internal service principal", re.compile(r"[a-z0-9-]+\.aws\.internal\b", re.IGNORECASE)),
    # Internal infrastructure / implementation details
    ("DDB stream", re.compile(r"DDB stream", re.IGNORECASE)),
    # Internal function names
    ("lambda_dedup_dataset", re.compile(r"lambda_dedup_dataset")),
    ("run_schema_conversion_operation", re.compile(r"run_schema_conversion_operation")),
    # Internal exception type
    ("CustomerFacingError", re.compile(r"CustomerFacingError")),
    # Preview-era install wording and internal caller attribution; public docs
    # install from PyPI and must not tag calls as an internal package.
    ("beta installation instructions", re.compile(r"beta\s+installation", re.IGNORECASE)),
    ("preview access", re.compile(r"preview\s+access", re.IGNORECASE)),
    ("SageMakerAIAgentSkills attribution", re.compile(r"SageMakerAIAgentSkills")),
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
        for match in _ACCOUNT_ID_RE.finditer(line):
            candidate = match.group(1)
            if candidate not in _ACCOUNT_ID_ALLOWLIST:
                violations.append(
                    f"<memory>:{lineno}: AWS account ID [{candidate}]: {line.strip()}"
                )
    return violations


class TestScannerSelfTest:
    """Verify the scanner flags each banned pattern correctly (in-memory)."""

    def test_c2j_flagged(self):
        assert _run_scanner_on_text("Loaded via the internal C2J fixture."), "Scanner missed 'C2J'"

    def test_duplo_flagged(self):
        assert _run_scanner_on_text(
            "generated from the Duplo Smithy model"
        ), "Scanner missed 'duplo'"

    def test_isengard_flagged(self):
        assert _run_scanner_on_text(
            "ada credentials update --provider=isengard"
        ), "Scanner missed 'isengard'"

    def test_ada_credentials_flagged(self):
        assert _run_scanner_on_text(
            "run ada credentials update --account=1234"
        ), "Scanner missed 'ada credentials'"

    def test_alpha_endpoint_flagged(self):
        assert _run_scanner_on_text(
            "requires the alpha endpoint to be configured"
        ), "Scanner missed 'alpha endpoint'"

    def test_internal_model_flagged(self):
        assert _run_scanner_on_text(
            "loaded via an internal model path"
        ), "Scanner missed 'internal model'"

    def test_a2z_com_flagged(self):
        assert _run_scanner_on_text(
            "see https://a2z.com/some/internal/page"
        ), "Scanner missed 'a2z.com'"

    def test_corp_amazon_flagged(self):
        assert _run_scanner_on_text(
            "see https://doc.corp.amazon.com/page"
        ), "Scanner missed 'corp.amazon'"

    def test_midway_flagged(self):
        assert _run_scanner_on_text(
            "authenticate via midway before running"
        ), "Scanner missed 'midway'"

    def test_account_id_flagged(self):
        assert _run_scanner_on_text("--account=925630940947"), "Scanner missed 12-digit account ID"

    def test_placeholder_account_id_not_flagged(self):
        assert not _run_scanner_on_text(
            "account_id = '123456789012'"
        ), "Scanner incorrectly flagged placeholder account ID 123456789012"

    def test_placeholder_account_id_2_not_flagged(self):
        assert not _run_scanner_on_text(
            "account_id = '111122223333'"
        ), "Scanner incorrectly flagged placeholder account ID 111122223333"

    def test_c2j_word_boundary_respected(self):
        # 'ec2j' or 'c2js' should NOT be flagged (word boundary \b)
        assert not _run_scanner_on_text(
            "some ec2j string"
        ), "Scanner incorrectly flagged 'ec2j' (should require word boundary)"

    def test_internal_api_flagged(self):
        assert _run_scanner_on_text(
            "calls the internal API directly"
        ), "Scanner missed 'internal api'"

    def test_scanner_returns_empty_for_clean_text(self):
        clean = (
            "import boto3\n"
            "# This notebook uses the public SageMaker SDK.\n"
            "account_id = '123456789012'  # placeholder\n"
        )
        result = _run_scanner_on_text(clean)
        assert not result, f"Scanner false-positived on clean text: {result}"

    def test_java_file_reference_flagged(self):
        assert _run_scanner_on_text(
            "spellings come from OperatorType.java"
        ), "Scanner missed '.java' file reference"

    def test_java_boundary_not_flagged_for_javascript(self):
        # ".java" should only match when followed by a word boundary.
        # A path like "foo.javascript" must not be flagged.
        assert not _run_scanner_on_text(
            "see foo.javascript for details"
        ), "Scanner incorrectly flagged '.javascript' (should require word boundary after .java)"

    def test_job_metadata_validator_flagged(self):
        assert _run_scanner_on_text(
            "JobMetadataValidator.validate is called here"
        ), "Scanner missed 'JobMetadataValidator'"

    def test_networknt_flagged(self):
        assert _run_scanner_on_text(
            "the networknt JSON schema validator"
        ), "Scanner missed 'networknt'"

    def test_operator_type_enum_flagged(self):
        assert _run_scanner_on_text(
            "see OperatorType for the full list"
        ), "Scanner missed 'OperatorType'"

    def test_data_curator_flagged(self):
        assert _run_scanner_on_text(
            "loaded from AGIDataCurator library"
        ), "Scanner missed 'DataCurator'"

    def test_metric_engine_flagged(self):
        assert _run_scanner_on_text(
            "scores via MetricEngine"
        ), "Scanner missed 'MetricEngine'"

    def test_transform_compute_router_flagged(self):
        assert _run_scanner_on_text(
            "routed by TransformComputeRouter"
        ), "Scanner missed 'TransformComputeRouter'"

    def test_ironman_flagged(self):
        assert _run_scanner_on_text(
            "the IronMan service handles this"
        ), "Scanner missed 'IronMan'"

    def test_role_proxy_flagged(self):
        assert _run_scanner_on_text(
            "credentials vended via RoleProxy"
        ), "Scanner missed 'RoleProxy'"

    def test_internal_service_principal_flagged(self):
        assert _run_scanner_on_text(
            '"Service": ["sagemaker.amazonaws.com", "roleproxy.im.aws.internal"]'
        ), "Scanner missed an internal '.aws.internal' service principal"

    def test_public_service_principal_not_flagged(self):
        assert not _run_scanner_on_text(
            '"Service": "sagemaker.amazonaws.com"'
        ), "Scanner flagged the public SageMaker service principal"

    def test_ddb_stream_flagged(self):
        assert _run_scanner_on_text(
            "triggered off the DDB stream"
        ), "Scanner missed 'DDB stream'"

    def test_lambda_dedup_dataset_flagged(self):
        assert _run_scanner_on_text(
            "lambda_dedup_dataset raises NotImplementedError"
        ), "Scanner missed 'lambda_dedup_dataset'"

    def test_run_schema_conversion_operation_flagged(self):
        assert _run_scanner_on_text(
            "run_schema_conversion_operation compares formats"
        ), "Scanner missed 'run_schema_conversion_operation'"

    def test_customer_facing_error_flagged(self):
        assert _run_scanner_on_text(
            "raise CustomerFacingError('bad input')"
        ), "Scanner missed 'CustomerFacingError'"

    def test_beta_installation_flagged(self):
        assert _run_scanner_on_text(
            "see Beta Installation instructions"
        ), "Scanner missed preview-era 'Beta Installation' wording"

    def test_preview_access_flagged(self):
        assert _run_scanner_on_text(
            "provided with your preview access"
        ), "Scanner missed preview-era 'preview access' wording"

    def test_internal_caller_attribution_flagged(self):
        assert _run_scanner_on_text(
            'os.environ["SAGEMAKER_PYSDK_CREATED_BY"] = "SageMakerAIAgentSkills"'
        ), "Scanner missed internal 'SageMakerAIAgentSkills' attribution"

    def test_public_install_command_not_flagged(self):
        assert not _run_scanner_on_text(
            "pip install --upgrade sagemaker"
        ), "Scanner flagged the public pip install command"


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
