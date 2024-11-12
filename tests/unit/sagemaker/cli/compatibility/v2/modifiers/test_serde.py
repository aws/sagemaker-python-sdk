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
from __future__ import absolute_import

import ast

import pasta
import pytest

from sagemaker.cli.compatibility.v2.modifiers import serde
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call, ast_import


@pytest.mark.parametrize(
    "src, expected",
    [
        ("sagemaker.predictor._CsvSerializer()", True),
        ("sagemaker.predictor._JsonSerializer()", True),
        ("sagemaker.predictor._NpySerializer()", True),
        ("sagemaker.predictor._CsvDeserializer()", True),
        ("sagemaker.predictor.BytesDeserializer()", True),
        ("sagemaker.predictor.StringDeserializer()", True),
        ("sagemaker.predictor.StreamDeserializer()", True),
        ("sagemaker.predictor._NumpyDeserializer()", True),
        ("sagemaker.predictor._JsonDeserializer()", True),
        ("sagemaker.predictor.OtherClass()", False),
        ("sagemaker.amazon.common.numpy_to_record_serializer()", True),
        ("sagemaker.amazon.common.record_deserializer()", True),
        ("_CsvSerializer()", True),
        ("_JsonSerializer()", True),
        ("_NpySerializer()", True),
        ("_CsvDeserializer()", True),
        ("BytesDeserializer()", True),
        ("StringDeserializer()", True),
        ("StreamDeserializer()", True),
        ("_NumpyDeserializer()", True),
        ("_JsonDeserializer()", True),
        ("numpy_to_record_serializer()", True),
        ("record_deserializer()", True),
        ("OtherClass()", False),
    ],
)
def test_constructor_node_should_be_modified(src, expected):
    modifier = serde.SerdeConstructorRenamer()
    node = ast_call(src)
    assert modifier.node_should_be_modified(node) is expected


@pytest.mark.parametrize(
    "src, expected",
    [
        ("sagemaker.predictor._CsvSerializer()", "serializers.CSVSerializer()"),
        ("sagemaker.predictor._JsonSerializer()", "serializers.JSONSerializer()"),
        ("sagemaker.predictor._NpySerializer()", "serializers.NumpySerializer()"),
        ("sagemaker.predictor._CsvDeserializer()", "deserializers.CSVDeserializer()"),
        ("sagemaker.predictor.BytesDeserializer()", "deserializers.BytesDeserializer()"),
        (
            "sagemaker.predictor.StringDeserializer()",
            "deserializers.StringDeserializer()",
        ),
        (
            "sagemaker.predictor.StreamDeserializer()",
            "deserializers.StreamDeserializer()",
        ),
        ("sagemaker.predictor._NumpyDeserializer()", "deserializers.NumpyDeserializer()"),
        ("sagemaker.predictor._JsonDeserializer()", "deserializers.JSONDeserializer()"),
        (
            "sagemaker.amazon.common.numpy_to_record_serializer()",
            "sagemaker.amazon.common.RecordSerializer()",
        ),
        (
            "sagemaker.amazon.common.record_deserializer()",
            "sagemaker.amazon.common.RecordDeserializer()",
        ),
        ("_CsvSerializer()", "serializers.CSVSerializer()"),
        ("_JsonSerializer()", "serializers.JSONSerializer()"),
        ("_NpySerializer()", "serializers.NumpySerializer()"),
        ("_CsvDeserializer()", "deserializers.CSVDeserializer()"),
        ("BytesDeserializer()", "deserializers.BytesDeserializer()"),
        ("StringDeserializer()", "deserializers.StringDeserializer()"),
        ("StreamDeserializer()", "deserializers.StreamDeserializer()"),
        ("_NumpyDeserializer()", "deserializers.NumpyDeserializer()"),
        ("_JsonDeserializer()", "deserializers.JSONDeserializer()"),
        ("numpy_to_record_serializer()", "RecordSerializer()"),
        ("record_deserializer()", "RecordDeserializer()"),
    ],
)
def test_constructor_modify_node(src, expected):
    modifier = serde.SerdeConstructorRenamer()
    node = ast_call(src)
    modified_node = modifier.modify_node(node)
    assert expected == pasta.dump(modified_node)
    assert isinstance(modified_node, ast.Call)


@pytest.mark.parametrize(
    "src, expected",
    [
        (
            "sagemaker.predictor.csv_serializer",
            True,
        ),
        (
            "sagemaker.predictor.json_serializer",
            True,
        ),
        (
            "sagemaker.predictor.npy_serializer",
            True,
        ),
        (
            "sagemaker.predictor.csv_deserializer",
            True,
        ),
        (
            "sagemaker.predictor.json_deserializer",
            True,
        ),
        (
            "sagemaker.predictor.numpy_deserializer",
            True,
        ),
        (
            "csv_serializer",
            True,
        ),
        (
            "json_serializer",
            True,
        ),
        (
            "npy_serializer",
            True,
        ),
        (
            "csv_deserializer",
            True,
        ),
        (
            "json_deserializer",
            True,
        ),
        (
            "numpy_deserializer",
            True,
        ),
    ],
)
def test_name_node_should_be_modified(src, expected):
    modifier = serde.SerdeObjectRenamer()
    node = ast_call(src)
    assert modifier.node_should_be_modified(node) is True


@pytest.mark.parametrize(
    "src, expected",
    [
        ("sagemaker.predictor.csv_serializer", "serializers.CSVSerializer()"),
        ("sagemaker.predictor.json_serializer", "serializers.JSONSerializer()"),
        ("sagemaker.predictor.npy_serializer", "serializers.NumpySerializer()"),
        ("sagemaker.predictor.csv_deserializer", "deserializers.CSVDeserializer()"),
        ("sagemaker.predictor.json_deserializer", "deserializers.JSONDeserializer()"),
        ("sagemaker.predictor.numpy_deserializer", "deserializers.NumpyDeserializer()"),
        ("csv_serializer", "serializers.CSVSerializer()"),
        ("json_serializer", "serializers.JSONSerializer()"),
        ("npy_serializer", "serializers.NumpySerializer()"),
        ("csv_deserializer", "deserializers.CSVDeserializer()"),
        ("json_deserializer", "deserializers.JSONDeserializer()"),
        ("numpy_deserializer", "deserializers.NumpyDeserializer()"),
    ],
)
def test_name_modify_node(src, expected):
    modifier = serde.SerdeObjectRenamer()
    node = ast_call(src)
    modified_node = modifier.modify_node(node)
    assert expected == pasta.dump(modified_node)
    assert isinstance(modified_node, ast.Call)


@pytest.mark.parametrize(
    "src, expected",
    [
        ("from sagemaker.predictor import _CsvSerializer", True),
        ("from sagemaker.predictor import _JsonSerializer", True),
        ("from sagemaker.predictor import _NpySerializer", True),
        ("from sagemaker.predictor import _CsvDeserializer", True),
        ("from sagemaker.predictor import BytesDeserializer", True),
        ("from sagemaker.predictor import StringDeserializer", True),
        ("from sagemaker.predictor import StreamDeserializer", True),
        ("from sagemaker.predictor import _NumpyDeserializer", True),
        ("from sagemaker.predictor import _JsonDeserializer", True),
        ("from sagemaker.predictor import csv_serializer", True),
        ("from sagemaker.predictor import json_serializer", True),
        ("from sagemaker.predictor import npy_serializer", True),
        ("from sagemaker.predictor import csv_deserializer", True),
        ("from sagemaker.predictor import json_deserializer", True),
        ("from sagemaker.predictor import numpy_deserializer", True),
        ("from sagemaker.predictor import RealTimePredictor, _CsvSerializer", True),
        ("from sagemaker.predictor import RealTimePredictor", False),
        ("from sagemaker.amazon.common import numpy_to_record_serializer", False),
    ],
)
def test_import_from_predictor_node_should_be_modified(src, expected):
    modifier = serde.SerdeImportFromPredictorRenamer()
    node = ast_import(src)
    assert modifier.node_should_be_modified(node) is expected


@pytest.mark.parametrize(
    "src, expected",
    [
        ("from sagemaker.predictor import _CsvSerializer", None),
        ("from sagemaker.predictor import _JsonSerializer", None),
        ("from sagemaker.predictor import _NpySerializer", None),
        ("from sagemaker.predictor import _CsvDeserializer", None),
        ("from sagemaker.predictor import BytesDeserializer", None),
        ("from sagemaker.predictor import StringDeserializer", None),
        ("from sagemaker.predictor import StreamDeserializer", None),
        ("from sagemaker.predictor import _NumpyDeserializer", None),
        ("from sagemaker.predictor import _JsonDeserializer", None),
        ("from sagemaker.predictor import csv_serializer", None),
        ("from sagemaker.predictor import json_serializer", None),
        ("from sagemaker.predictor import npy_serializer", None),
        ("from sagemaker.predictor import csv_deserializer", None),
        ("from sagemaker.predictor import json_deserializer", None),
        ("from sagemaker.predictor import numpy_deserializer", None),
        (
            "from sagemaker.predictor import RealTimePredictor, _NpySerializer",
            "from sagemaker.predictor import RealTimePredictor",
        ),
    ],
)
def test_import_from_predictor_modify_node(src, expected):
    modifier = serde.SerdeImportFromPredictorRenamer()
    node = ast_import(src)
    modified_node = modifier.modify_node(node)
    assert expected == (modified_node and pasta.dump(modified_node))


@pytest.mark.parametrize(
    "import_statement, expected",
    [
        ("from sagemaker.amazon.common import numpy_to_record_serializer", True),
        ("from sagemaker.amazon.common import record_deserializer", True),
        ("from sagemaker.amazon.common import write_spmatrix_to_sparse_tensor", False),
    ],
)
def test_import_from_amazon_common_node_should_be_modified(import_statement, expected):
    modifier = serde.SerdeImportFromAmazonCommonRenamer()
    node = ast_import(import_statement)
    assert modifier.node_should_be_modified(node) is expected


@pytest.mark.parametrize(
    "import_statement, expected",
    [
        (
            "from sagemaker.amazon.common import numpy_to_record_serializer",
            "from sagemaker.amazon.common import RecordSerializer",
        ),
        (
            "from sagemaker.amazon.common import record_deserializer",
            "from sagemaker.amazon.common import RecordDeserializer",
        ),
        (
            "from sagemaker.amazon.common import numpy_to_record_serializer, record_deserializer",
            "from sagemaker.amazon.common import RecordSerializer, RecordDeserializer",
        ),
        (
            "from sagemaker.amazon.common import write_spmatrix_to_sparse_tensor, numpy_to_record_serializer",
            "from sagemaker.amazon.common import write_spmatrix_to_sparse_tensor, RecordSerializer",
        ),
    ],
)
def test_import_from_amazon_common_modify_node(import_statement, expected):
    modifier = serde.SerdeImportFromAmazonCommonRenamer()
    node = ast_import(import_statement)
    modified_node = modifier.modify_node(node)
    assert expected == pasta.dump(modified_node)


@pytest.mark.parametrize(
    "src, expected",
    [
        ("serializers.CSVSerializer()", True),
        ("serializers.JSONSerializer()", True),
        ("serializers.NumpySerializer()", True),
        ("pass", False),
    ],
)
def test_serializer_module_node_should_be_modified(src, expected):
    modifier = serde.SerializerImportInserter()
    node = pasta.parse(src)
    assert modifier.node_should_be_modified(node) is expected


@pytest.mark.parametrize(
    "src, expected",
    [
        (
            "serializers.CSVSerializer()",
            "from sagemaker import serializers\nserializers.CSVSerializer()",
        ),
        (
            "serializers.JSONSerializer()",
            "from sagemaker import serializers\nserializers.JSONSerializer()",
        ),
        (
            "serializers.NumpySerializer()",
            "from sagemaker import serializers\nserializers.NumpySerializer()",
        ),
        (
            "pass\nimport random\nserializers.CSVSerializer()",
            "pass\nfrom sagemaker import serializers\nimport random\nserializers.CSVSerializer()",
        ),
    ],
)
def test_serializer_module_modify_node(src, expected):
    modifier = serde.SerializerImportInserter()
    node = pasta.parse(src)
    modified_node = modifier.modify_node(node)
    assert expected == pasta.dump(modified_node)


@pytest.mark.parametrize(
    "src, expected",
    [
        ("deserializers.CSVDeserializer()", True),
        ("deserializers.BytesDeserializer()", True),
        ("deserializers.StringDeserializer()", True),
        ("deserializers.StreamDeserializer()", True),
        ("deserializers.NumpyDeserializer()", True),
        ("deserializers.JSONDeserializer()", True),
        ("pass", False),
    ],
)
def test_deserializer_module_node_should_be_modified(src, expected):
    modifier = serde.DeserializerImportInserter()
    node = pasta.parse(src)
    assert modifier.node_should_be_modified(node) is expected


@pytest.mark.parametrize(
    "src, expected",
    [
        (
            "deserializers.CSVDeserializer()",
            "from sagemaker import deserializers\ndeserializers.CSVDeserializer()",
        ),
        (
            "deserializers.BytesDeserializer()",
            "from sagemaker import deserializers\ndeserializers.BytesDeserializer()",
        ),
        (
            "deserializers.StringDeserializer()",
            "from sagemaker import deserializers\ndeserializers.StringDeserializer()",
        ),
        (
            "deserializers.StreamDeserializer()",
            "from sagemaker import deserializers\ndeserializers.StreamDeserializer()",
        ),
        (
            "deserializers.NumpyDeserializer()",
            "from sagemaker import deserializers\ndeserializers.NumpyDeserializer()",
        ),
        (
            "deserializers.JSONDeserializer()",
            "from sagemaker import deserializers\ndeserializers.JSONDeserializer()",
        ),
        (
            "pass\nimport random\ndeserializers.CSVDeserializer()",
            "pass\nfrom sagemaker import deserializers\nimport random\ndeserializers.CSVDeserializer()",
        ),
    ],
)
def test_deserializer_module_modify_node(src, expected):
    modifier = serde.DeserializerImportInserter()
    node = pasta.parse(src)
    modified_node = modifier.modify_node(node)
    assert expected == pasta.dump(modified_node)


@pytest.mark.parametrize(
    "src, expected",
    [
        ('estimator.create_model(entry_point="inference.py")', False),
        ("estimator.create_model(serializer=CSVSerializer())", True),
        ("estimator.create_model(deserializer=CSVDeserializer())", True),
        (
            "estimator.create_model(serializer=CSVSerializer(), deserializer=CSVDeserializer())",
            True,
        ),
        ("estimator.deploy(serializer=CSVSerializer())", False),
    ],
)
def test_create_model_call_node_should_be_modified(src, expected):
    modifier = serde.SerdeKeywordRemover()
    node = ast_call(src)
    assert modifier.node_should_be_modified(node) is expected


@pytest.mark.parametrize(
    "src, expected",
    [
        (
            'estimator.create_model(entry_point="inference.py", serializer=CSVSerializer())',
            'estimator.create_model(entry_point="inference.py")',
        ),
        (
            'estimator.create_model(entry_point="inference.py", deserializer=CSVDeserializer())',
            'estimator.create_model(entry_point="inference.py")',
        ),
    ],
)
def test_create_model_call_modify_node(src, expected):
    modifier = serde.SerdeKeywordRemover()
    node = ast_call(src)
    modified_node = modifier.modify_node(node)
    assert expected == pasta.dump(modified_node)
