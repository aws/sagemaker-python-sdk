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
"""Tests for SageMaker Evaluation Module __init__.py."""
from __future__ import absolute_import

import pytest
import inspect


class TestModuleImports:
    """Test cases for module imports and exports."""

    def test_module_can_be_imported(self):
        """Test that the evaluate module can be imported."""
        import sagemaker.train.evaluate
        assert sagemaker.train.evaluate is not None

    def test_base_evaluator_import(self):
        """Test that BaseEvaluator is importable."""
        from sagemaker.train.evaluate import BaseEvaluator
        assert BaseEvaluator is not None

    def test_benchmark_evaluator_import(self):
        """Test that BenchMarkEvaluator is importable."""
        from sagemaker.train.evaluate import BenchMarkEvaluator
        assert BenchMarkEvaluator is not None

    def test_custom_scorer_evaluator_import(self):
        """Test that CustomScorerEvaluator is importable."""
        from sagemaker.train.evaluate import CustomScorerEvaluator
        assert CustomScorerEvaluator is not None

    def test_llm_as_judge_evaluator_import(self):
        """Test that LLMAsJudgeEvaluator is importable."""
        from sagemaker.train.evaluate import LLMAsJudgeEvaluator
        assert LLMAsJudgeEvaluator is not None

    def test_get_benchmarks_import(self):
        """Test that get_benchmarks function is importable."""
        from sagemaker.train.evaluate import get_benchmarks
        assert get_benchmarks is not None
        assert callable(get_benchmarks)

    def test_get_benchmark_properties_import(self):
        """Test that get_benchmark_properties function is importable."""
        from sagemaker.train.evaluate import get_benchmark_properties
        assert get_benchmark_properties is not None
        assert callable(get_benchmark_properties)

    def test_get_builtin_metrics_import(self):
        """Test that get_builtin_metrics function is importable."""
        from sagemaker.train.evaluate import get_builtin_metrics
        assert get_builtin_metrics is not None
        assert callable(get_builtin_metrics)

    def test_evaluation_pipeline_execution_import(self):
        """Test that EvaluationPipelineExecution is importable."""
        from sagemaker.train.evaluate import EvaluationPipelineExecution
        assert EvaluationPipelineExecution is not None

    def test_pipeline_execution_status_import(self):
        """Test that PipelineExecutionStatus is importable."""
        from sagemaker.train.evaluate import PipelineExecutionStatus
        assert PipelineExecutionStatus is not None

    def test_step_detail_import(self):
        """Test that StepDetail is importable."""
        from sagemaker.train.evaluate import StepDetail
        assert StepDetail is not None


class TestModuleAll:
    """Test cases for __all__ exports."""

    def test_all_variable_exists(self):
        """Test that __all__ variable exists."""
        import sagemaker.train.evaluate
        assert hasattr(sagemaker.train.evaluate, "__all__")

    def test_all_is_list(self):
        """Test that __all__ is a list."""
        from sagemaker.train.evaluate import __all__
        assert isinstance(__all__, list)

    def test_all_contains_expected_classes(self):
        """Test that __all__ contains expected evaluator classes."""
        from sagemaker.train.evaluate import __all__
        
        expected_classes = [
            "BaseEvaluator",
            "BenchMarkEvaluator",
            "CustomScorerEvaluator",
            "LLMAsJudgeEvaluator",
        ]
        
        for cls in expected_classes:
            assert cls in __all__, f"{cls} not found in __all__"

    def test_all_contains_expected_functions(self):
        """Test that __all__ contains expected utility functions."""
        from sagemaker.train.evaluate import __all__
        
        expected_functions = [
            "get_benchmarks",
            "get_benchmark_properties",
            "get_builtin_metrics",
        ]
        
        for func in expected_functions:
            assert func in __all__, f"{func} not found in __all__"

    def test_all_contains_expected_execution_classes(self):
        """Test that __all__ contains expected execution classes."""
        from sagemaker.train.evaluate import __all__
        
        expected_classes = [
            "EvaluationPipelineExecution",
            "PipelineExecutionStatus",
            "StepDetail",
        ]
        
        for cls in expected_classes:
            assert cls in __all__, f"{cls} not found in __all__"

    def test_all_items_count(self):
        """Test that __all__ contains exactly 10 items."""
        from sagemaker.train.evaluate import __all__
        assert len(__all__) == 10

    def test_all_items_are_strings(self):
        """Test that all items in __all__ are strings."""
        from sagemaker.train.evaluate import __all__
        for item in __all__:
            assert isinstance(item, str)

    def test_all_items_no_duplicates(self):
        """Test that __all__ has no duplicate entries."""
        from sagemaker.train.evaluate import __all__
        assert len(__all__) == len(set(__all__))


class TestModuleVersion:
    """Test cases for module version."""

    def test_version_exists(self):
        """Test that __version__ attribute exists."""
        import sagemaker.train.evaluate
        assert hasattr(sagemaker.train.evaluate, "__version__")

    def test_version_is_string(self):
        """Test that __version__ is a string."""
        from sagemaker.train.evaluate import __version__
        assert isinstance(__version__, str)

    def test_version_value(self):
        """Test that __version__ has expected value."""
        from sagemaker.train.evaluate import __version__
        assert __version__ == "1.0.0"

    def test_version_format(self):
        """Test that __version__ follows semantic versioning format."""
        from sagemaker.train.evaluate import __version__
        parts = __version__.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()


class TestExportedClasses:
    """Test cases for exported classes."""

    def test_base_evaluator_is_class(self):
        """Test that BaseEvaluator is a class."""
        from sagemaker.train.evaluate import BaseEvaluator
        assert inspect.isclass(BaseEvaluator)

    def test_benchmark_evaluator_is_class(self):
        """Test that BenchMarkEvaluator is a class."""
        from sagemaker.train.evaluate import BenchMarkEvaluator
        assert inspect.isclass(BenchMarkEvaluator)

    def test_custom_scorer_evaluator_is_class(self):
        """Test that CustomScorerEvaluator is a class."""
        from sagemaker.train.evaluate import CustomScorerEvaluator
        assert inspect.isclass(CustomScorerEvaluator)

    def test_llm_as_judge_evaluator_is_class(self):
        """Test that LLMAsJudgeEvaluator is a class."""
        from sagemaker.train.evaluate import LLMAsJudgeEvaluator
        assert inspect.isclass(LLMAsJudgeEvaluator)

    def test_evaluation_pipeline_execution_is_class(self):
        """Test that EvaluationPipelineExecution is a class."""
        from sagemaker.train.evaluate import EvaluationPipelineExecution
        assert inspect.isclass(EvaluationPipelineExecution)

    def test_pipeline_execution_status_is_class(self):
        """Test that PipelineExecutionStatus is a class."""
        from sagemaker.train.evaluate import PipelineExecutionStatus
        assert inspect.isclass(PipelineExecutionStatus)

    def test_step_detail_is_class(self):
        """Test that StepDetail is a class."""
        from sagemaker.train.evaluate import StepDetail
        assert inspect.isclass(StepDetail)


class TestExportedFunctions:
    """Test cases for exported functions."""

    def test_get_benchmarks_is_function(self):
        """Test that get_benchmarks is a function."""
        from sagemaker.train.evaluate import get_benchmarks
        assert callable(get_benchmarks)

    def test_get_benchmark_properties_is_function(self):
        """Test that get_benchmark_properties is a function."""
        from sagemaker.train.evaluate import get_benchmark_properties
        assert callable(get_benchmark_properties)

    def test_get_builtin_metrics_is_function(self):
        """Test that get_builtin_metrics is a function."""
        from sagemaker.train.evaluate import get_builtin_metrics
        assert callable(get_builtin_metrics)


class TestImportOrigins:
    """Test cases verifying import origins."""

    def test_base_evaluator_origin(self):
        """Test that BaseEvaluator comes from base_evaluator module."""
        from sagemaker.train.evaluate import BaseEvaluator
        from sagemaker.train.evaluate.base_evaluator import BaseEvaluator as OriginalBaseEvaluator
        assert BaseEvaluator is OriginalBaseEvaluator

    def test_benchmark_evaluator_origin(self):
        """Test that BenchMarkEvaluator comes from benchmark_evaluator module."""
        from sagemaker.train.evaluate import BenchMarkEvaluator
        from sagemaker.train.evaluate.benchmark_evaluator import BenchMarkEvaluator as OriginalBenchMarkEvaluator
        assert BenchMarkEvaluator is OriginalBenchMarkEvaluator

    def test_custom_scorer_evaluator_origin(self):
        """Test that CustomScorerEvaluator comes from custom_scorer_evaluator module."""
        from sagemaker.train.evaluate import CustomScorerEvaluator
        from sagemaker.train.evaluate.custom_scorer_evaluator import CustomScorerEvaluator as OriginalCustomScorerEvaluator
        assert CustomScorerEvaluator is OriginalCustomScorerEvaluator

    def test_llm_as_judge_evaluator_origin(self):
        """Test that LLMAsJudgeEvaluator comes from llm_as_judge_evaluator module."""
        from sagemaker.train.evaluate import LLMAsJudgeEvaluator
        from sagemaker.train.evaluate.llm_as_judge_evaluator import LLMAsJudgeEvaluator as OriginalLLMAsJudgeEvaluator
        assert LLMAsJudgeEvaluator is OriginalLLMAsJudgeEvaluator

    def test_get_benchmarks_origin(self):
        """Test that get_benchmarks comes from benchmark_evaluator module."""
        from sagemaker.train.evaluate import get_benchmarks
        from sagemaker.train.evaluate.benchmark_evaluator import get_benchmarks as original_get_benchmarks
        assert get_benchmarks is original_get_benchmarks

    def test_get_benchmark_properties_origin(self):
        """Test that get_benchmark_properties comes from benchmark_evaluator module."""
        from sagemaker.train.evaluate import get_benchmark_properties
        from sagemaker.train.evaluate.benchmark_evaluator import get_benchmark_properties as original_get_benchmark_properties
        assert get_benchmark_properties is original_get_benchmark_properties

    def test_get_builtin_metrics_origin(self):
        """Test that get_builtin_metrics comes from custom_scorer_evaluator module."""
        from sagemaker.train.evaluate import get_builtin_metrics
        from sagemaker.train.evaluate.custom_scorer_evaluator import get_builtin_metrics as original_get_builtin_metrics
        assert get_builtin_metrics is original_get_builtin_metrics

    def test_evaluation_pipeline_execution_origin(self):
        """Test that EvaluationPipelineExecution comes from execution module."""
        from sagemaker.train.evaluate import EvaluationPipelineExecution
        from sagemaker.train.evaluate.execution import EvaluationPipelineExecution as OriginalEvaluationPipelineExecution
        assert EvaluationPipelineExecution is OriginalEvaluationPipelineExecution

    def test_pipeline_execution_status_origin(self):
        """Test that PipelineExecutionStatus comes from execution module."""
        from sagemaker.train.evaluate import PipelineExecutionStatus
        from sagemaker.train.evaluate.execution import PipelineExecutionStatus as OriginalPipelineExecutionStatus
        assert PipelineExecutionStatus is OriginalPipelineExecutionStatus

    def test_step_detail_origin(self):
        """Test that StepDetail comes from execution module."""
        from sagemaker.train.evaluate import StepDetail
        from sagemaker.train.evaluate.execution import StepDetail as OriginalStepDetail
        assert StepDetail is OriginalStepDetail


class TestWildcardImport:
    """Test cases for wildcard imports."""

    def test_wildcard_import_includes_all_items(self):
        """Test that wildcard import includes all __all__ items."""
        import sagemaker.train.evaluate
        from sagemaker.train.evaluate import __all__
        
        for item in __all__:
            assert hasattr(sagemaker.train.evaluate, item), f"{item} not accessible via wildcard import"

    def test_wildcard_import_excludes_private_imports(self):
        """Test that wildcard import doesn't expose private modules."""
        import sagemaker.train.evaluate
        
        # These should not be directly accessible
        private_modules = ["base_evaluator", "benchmark_evaluator", "custom_scorer_evaluator", 
                          "llm_as_judge_evaluator", "execution"]
        
        for module in private_modules:
            # Check if it's accessible (it shouldn't be via __all__)
            if hasattr(sagemaker.train.evaluate, module):
                # If it exists, it shouldn't be in __all__
                from sagemaker.train.evaluate import __all__
                assert module not in __all__


class TestModuleDocstring:
    """Test cases for module documentation."""

    def test_module_has_docstring(self):
        """Test that the module has a docstring."""
        import sagemaker.train.evaluate
        assert sagemaker.train.evaluate.__doc__ is not None

    def test_module_docstring_is_string(self):
        """Test that module docstring is a string."""
        import sagemaker.train.evaluate
        assert isinstance(sagemaker.train.evaluate.__doc__, str)

    def test_module_docstring_not_empty(self):
        """Test that module docstring is not empty."""
        import sagemaker.train.evaluate
        assert len(sagemaker.train.evaluate.__doc__.strip()) > 0

    def test_module_docstring_contains_key_info(self):
        """Test that module docstring contains key information."""
        import sagemaker.train.evaluate
        docstring = sagemaker.train.evaluate.__doc__
        
        # Check for key sections
        assert "SageMaker Model Evaluation Module" in docstring
        assert "Classes:" in docstring or "classes:" in docstring.lower()


class TestNamespaceIsolation:
    """Test cases for namespace isolation."""

    def test_no_unexpected_exports(self):
        """Test that only expected items are in __all__."""
        from sagemaker.train.evaluate import __all__
        
        expected_items = {
            # Evaluator classes
            "BaseEvaluator",
            "BenchMarkEvaluator",
            "CustomScorerEvaluator",
            "LLMAsJudgeEvaluator",
            # Benchmark utility functions
            "get_benchmarks",
            "get_benchmark_properties",
            # Custom scorer utility functions
            "get_builtin_metrics",
            # Execution classes
            "EvaluationPipelineExecution",
            "PipelineExecutionStatus",
            "StepDetail",
        }
        
        actual_items = set(__all__)
        assert actual_items == expected_items

    def test_module_level_imports_work(self):
        """Test that all module-level imports work."""
        try:
            from sagemaker.train.evaluate import (
                BaseEvaluator,
                BenchMarkEvaluator,
                CustomScorerEvaluator,
                LLMAsJudgeEvaluator,
                get_benchmarks,
                get_benchmark_properties,
                get_builtin_metrics,
                EvaluationPipelineExecution,
                PipelineExecutionStatus,
                StepDetail,
            )
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


class TestNoAWSCredentialsRequired:
    """Test that imports don't require AWS credentials."""

    def test_imports_work_without_credentials(self):
        """Test that all imports work without AWS credentials."""
        import os
        
        # Save current environment
        saved_env = {}
        aws_keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
        for key in aws_keys:
            saved_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
        
        try:
            # Try importing with no credentials
            from sagemaker.train.evaluate import (
                BaseEvaluator,
                BenchMarkEvaluator,
                CustomScorerEvaluator,
                LLMAsJudgeEvaluator,
                get_benchmarks,
                get_benchmark_properties,
                get_builtin_metrics,
                EvaluationPipelineExecution,
                PipelineExecutionStatus,
                StepDetail,
            )
            assert True
        except Exception as e:
            pytest.fail(f"Import failed without credentials: {e}")
        finally:
            # Restore environment
            for key, value in saved_env.items():
                if value is not None:
                    os.environ[key] = value
