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
"""Test for JumpStart search_public_hub_models function."""
from __future__ import absolute_import

import pytest
from unittest.mock import patch
from sagemaker.core.jumpstart.search import _Filter, search_public_hub_models
from sagemaker.core.resources import HubContent


@pytest.mark.parametrize(
    "query,keywords,expected",
    [
        ("text-*", ["text-classification"], True),
        ("@task:foo", ["@task:foo"], True),
        ("@task:foo AND bar-*", ["@task:foo", "bar-baz"], True),
        ("@task:foo AND bar-*", ["@task:foo"], False),
        ("@task:foo OR bar-*", ["bar-qux"], True),
        ("@task:foo OR bar-*", ["nothing"], False),
        ("NOT @task:legacy", ["@task:modern"], True),
        ("NOT @task:legacy", ["@task:legacy"], False),
        (
            "(@framework:huggingface OR text-*) AND NOT @provider:qwen",
            ["@framework:huggingface", "text-generator"],
            True,
        ),
        (
            "(@framework:huggingface OR text-*) AND NOT @provider:qwen",
            ["@framework:huggingface", "@provider:qwen"],
            False,
        ),
    ],
)
def test_filter_match(query, keywords, expected):
    f = _Filter(query)
    assert f.match(keywords) == expected


@pytest.mark.parametrize(
    "malicious_query,keywords",
    [
        # Code injection attempts that would work with eval()
        ("__import__('os').system('echo pwned')", ["test"]),
        ("exec('import os; os.system(\"ls\")')", ["test"]),
        ("eval('1+1')", ["test"]),
        ("__builtins__.__import__('os').system('ls')", ["test"]),
        # Attribute access attempts
        ("keywords.__class__.__bases__[0].__subclasses__()", ["test"]),
        # Lambda injection
        ("(lambda: __import__('os').system('ls'))()", ["test"]),
        # Dict/list comprehension injection
        ("[x for x in ().__class__.__bases__[0].__subclasses__()]", ["test"]),
        # Function call injection
        ("open('/etc/passwd').read()", ["test"]),
        # Module access
        ("sys.exit()", ["test"]),
        ("os.system('ls')", ["test"]),
    ],
)
def test_filter_blocks_code_injection(malicious_query, keywords):
    """Test that malicious code injection attempts are safely handled."""
    f = _Filter(malicious_query)
    # Should not execute code, just return False for non-matching patterns
    result = f.match(keywords)
    assert isinstance(result, bool)
    # The filter should safely fail to match rather than execute code
    assert result is False


@pytest.mark.parametrize(
    "injection_query",
    [
        # Various eval-based injection patterns
        "'; __import__('os').system('ls'); '",
        "\"; exec('import os'); \"",
        "') or __import__('os').system('ls') or ('",
        # Nested injection attempts
        "test AND (__import__('os').system('ls'))",
        "NOT (__import__('subprocess').call(['ls']))",
        # String escape attempts
        "test' + str(__import__('os').system('ls')) + '",
    ],
)
def test_filter_injection_variants(injection_query):
    """Test various code injection patterns are blocked."""
    f = _Filter(injection_query)
    result = f.match(["test", "keyword"])
    assert isinstance(result, bool)
    # Should not raise exceptions or execute code
    assert result in [True, False]


def test_filter_no_eval_execution():
    """Verify that expressions are parsed safely without eval()."""
    # This would execute code if eval() was used
    dangerous_expr = "__import__('sys').exit(1)"
    f = _Filter(dangerous_expr)
    
    # Should not crash the program or execute the exit
    result = f.match(["test"])
    assert result is False


def test_filter_safe_ast_parsing():
    """Test that the filter uses AST parsing instead of eval()."""
    f = _Filter("test AND keyword")
    
    # Verify AST is created
    assert f._ast is None  # Not parsed yet
    f.match(["test", "keyword"])
    assert f._ast is not None  # AST created after first match
    
    # Verify it's an AST node, not a string for eval
    from sagemaker.core.jumpstart.search import _ExpressionNode
    assert isinstance(f._ast, _ExpressionNode)


def test_search_public_hub_models():
    mock_models = [
        HubContent(
            hub_content_type="Model",
            hub_content_name="textgen",
            hub_content_arn="arn:example:textgen",
            hub_content_version="1.0",
            document_schema_version="1.0",
            hub_content_display_name="Text Gen",
            hub_content_description="Generates text",
            hub_content_search_keywords=["@task:text-generation", "@framework:huggingface"],
            hub_content_status="Published",
            creation_time="2023-01-01T00:00:00Z",
            hub_name="SageMakerPublicHub",
        ),
        HubContent(
            hub_content_type="Model",
            hub_content_name="qwen-model",
            hub_content_arn="arn:example:qwen",
            hub_content_version="1.0",
            document_schema_version="1.0",
            hub_content_display_name="Qwen",
            hub_content_description="Qwen LLM",
            hub_content_search_keywords=["@provider:qwen"],
            hub_content_status="Published",
            creation_time="2023-01-01T00:00:00Z",
            hub_name="SageMakerPublicHub",
        ),
    ]

    with patch("sagemaker.core.jumpstart.search._list_all_hub_models", return_value=mock_models):
        results = search_public_hub_models(
            "(@task:text-generation OR huggingface) AND NOT @provider:qwen"
        )
        assert len(results) == 1
        assert isinstance(results[0], HubContent)
        assert results[0].hub_content_name == "textgen"
