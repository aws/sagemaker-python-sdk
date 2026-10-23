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
"""Unit tests for the SMD custom orchestrator entrypoint (custom_execution_inference.py).

The module under test is copied into the model artifacts as ``code/inference.py`` and
executed inside the serving container, so these tests exec its source the same way the
container does rather than importing it as a package module.
"""

from __future__ import absolute_import

import ast
import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cloudpickle

from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator, CustomOrchestrator

MODULE_NAME = "sagemaker.serve.model_server.smd.custom_execution_inference"
INFERENCE_SOURCE_PATH = Path(importlib.util.find_spec(MODULE_NAME).origin)


class EchoOrchestrator(CustomOrchestrator):
    """Minimal orchestrator used to populate serve.pkl."""

    def handle(self, data, context=None):
        return {"echo": data}


class AsyncEchoOrchestrator(AsyncCustomOrchestrator):
    """Minimal async orchestrator, to exercise the awaited handler branch."""

    async def handle(self, data, context=None):
        return {"async_echo": data}


class _Request:
    def __init__(self, body):
        self.body = body


def _write_artifacts(code_dir: Path, obj=None):
    """Write a serve.pkl/metadata.json pair the way prepare_for_smd does."""
    code_dir.mkdir(parents=True, exist_ok=True)
    buffer = cloudpickle.dumps((EchoOrchestrator(), None) if obj is None else obj)
    code_dir.joinpath("serve.pkl").write_bytes(buffer)
    code_dir.joinpath("metadata.json").write_bytes(
        json.dumps({"sha256_hash": hashlib.sha256(buffer).hexdigest()}).encode()
    )


def _load_inference(code_dir: Path):
    """Import the entrypoint against code_dir, running its import-time preflight.

    The model server imports this file as ``inference``; importing the module for real
    (rather than exec-ing its source) keeps the import-time work attributable to the
    module under test.
    """
    with patch.dict(os.environ, {"SAGEMAKER_INFERENCE_CODE_DIRECTORY": str(code_dir)}):
        module = importlib.import_module(MODULE_NAME)
        return importlib.reload(module)


class TestCustomExecutionInferenceIsSelfContained(unittest.TestCase):
    """The entrypoint must not depend on the container's bundled sagemaker version."""

    def test_no_sagemaker_imports(self):
        """Regression test for issue #6200.

        The entrypoint runs against whichever sagemaker version the SMD image happens to
        bundle, which is not the SDK version that produced the artifacts. Importing
        anything from sagemaker -- the integrity check in particular -- made startup fail
        on images carrying a pre-plain-SHA-256 sagemaker.
        """
        tree = ast.parse(INFERENCE_SOURCE_PATH.read_text())

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        offenders = sorted(
            name for name in imported if name == "sagemaker" or name.startswith("sagemaker.")
        )
        self.assertEqual(offenders, [])


class TestCodeDirectoryResolution(unittest.TestCase):
    """Test _get_code_directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.code_dir = Path(self.temp_dir) / "code"
        _write_artifacts(self.code_dir)
        self.module = _load_inference(self.code_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_uses_env_var_when_set(self):
        with patch.dict(os.environ, {"SAGEMAKER_INFERENCE_CODE_DIRECTORY": "/some/code"}):
            self.assertEqual(str(self.module._get_code_directory()), "/some/code")

    def test_falls_back_to_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(str(self.module._get_code_directory()), "/opt/ml/model/code")


class TestPerformIntegrityCheck(unittest.TestCase):
    """Test the inlined _perform_integrity_check."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.code_dir = Path(self.temp_dir) / "code"
        _write_artifacts(self.code_dir)
        self.module = _load_inference(self.code_dir)
        self.check = self.module._perform_integrity_check
        self.metadata_path = self.code_dir / "metadata.json"
        self.buffer = (self.code_dir / "serve.pkl").read_bytes()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_matching_digest_passes(self):
        self.check(buffer=self.buffer, metadata_path=self.metadata_path)

    def test_no_secret_key_needed(self):
        """The check must not read SAGEMAKER_SERVE_SECRET_KEY (removed for CVE-2026-8596)."""
        with patch.dict(os.environ, {}, clear=True):
            self.check(buffer=self.buffer, metadata_path=self.metadata_path)

    def test_tampered_buffer_fails(self):
        with self.assertRaises(ValueError) as ctx:
            self.check(buffer=self.buffer + b"x", metadata_path=self.metadata_path)
        self.assertIn("Integrity check", str(ctx.exception))

    def test_missing_metadata_fails(self):
        with self.assertRaises(ValueError) as ctx:
            self.check(buffer=self.buffer, metadata_path=self.code_dir / "absent.json")
        self.assertIn("does not exist", str(ctx.exception))

    def test_corrupt_metadata_fails(self):
        self.metadata_path.write_bytes(b"not json")
        with self.assertRaises(ValueError) as ctx:
            self.check(buffer=self.buffer, metadata_path=self.metadata_path)
        self.assertIn("not a valid json file", str(ctx.exception))

    def test_metadata_without_digest_fails(self):
        self.metadata_path.write_bytes(json.dumps({"version": "2023-04-24"}).encode())
        with self.assertRaises(ValueError) as ctx:
            self.check(buffer=self.buffer, metadata_path=self.metadata_path)
        self.assertIn("sha256_hash", str(ctx.exception))


class TestEntrypointStartup(unittest.TestCase):
    """Test the import-time preflight plus handler dispatch."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.code_dir = Path(self.temp_dir) / "code"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_startup_and_handle(self):
        _write_artifacts(self.code_dir)

        module = _load_inference(self.code_dir)

        self.assertIsInstance(module.custom_orchestrator, EchoOrchestrator)
        result = asyncio.run(module.handler(_Request(b"hello")))
        self.assertEqual(result, {"echo": b"hello"})

    def test_startup_and_handle_async(self):
        _write_artifacts(self.code_dir, obj=(AsyncEchoOrchestrator(), None))

        module = _load_inference(self.code_dir)

        self.assertIsInstance(module.custom_orchestrator, AsyncEchoOrchestrator)
        result = asyncio.run(module.handler(_Request(b"hello")))
        self.assertEqual(result, {"async_echo": b"hello"})

    def test_startup_fails_on_tampered_pickle(self):
        _write_artifacts(self.code_dir)
        serve_pkl = self.code_dir / "serve.pkl"
        serve_pkl.write_bytes(serve_pkl.read_bytes() + b"x")

        with self.assertRaises(ValueError):
            _load_inference(self.code_dir)


if __name__ == "__main__":
    unittest.main()
