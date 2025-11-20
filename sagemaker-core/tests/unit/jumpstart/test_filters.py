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

import pytest
from unittest.mock import Mock

from sagemaker.core.jumpstart.filters import (
    BooleanValues,
    FilterOperators,
    Operand,
    Operator,
    And,
    Or,
    Not,
    Identity,
    Constant,
    ModelFilter,
    parse_filter_string,
    evaluate_filter_expression,
    _negate_boolean,
    _evaluate_filter_expression_equals,
    _evaluate_filter_expression_in,
    _evaluate_filter_expression_includes,
    _evaluate_filter_expression_begins_with,
    _evaluate_filter_expression_ends_with,
)


class TestBooleanValues:
    """Test cases for BooleanValues enum"""

    def test_boolean_values(self):
        """Test BooleanValues enum values"""
        assert BooleanValues.TRUE == "true"
        assert BooleanValues.FALSE == "false"
        assert BooleanValues.UNKNOWN == "unknown"
        assert BooleanValues.UNEVALUATED == "unevaluated"


class TestFilterOperators:
    """Test cases for FilterOperators enum"""

    def test_filter_operators(self):
        """Test FilterOperators enum values"""
        assert FilterOperators.EQUALS == "equals"
        assert FilterOperators.NOT_EQUALS == "not_equals"
        assert FilterOperators.IN == "in"
        assert FilterOperators.NOT_IN == "not_in"
        assert FilterOperators.INCLUDES == "includes"
        assert FilterOperators.NOT_INCLUDES == "not_includes"
        assert FilterOperators.BEGINS_WITH == "begins_with"
        assert FilterOperators.ENDS_WITH == "ends_with"


class TestOperand:
    """Test cases for Operand class"""

    def test_init(self):
        """Test Operand initialization"""
        operand = Operand("test_value")
        assert operand.unresolved_value == "test_value"
        assert operand.resolved_value == BooleanValues.UNEVALUATED

    def test_init_with_resolved_value(self):
        """Test Operand initialization with resolved value"""
        operand = Operand("test", resolved_value=BooleanValues.TRUE)
        assert operand.resolved_value == BooleanValues.TRUE

    def test_resolved_value_setter(self):
        """Test resolved_value setter"""
        operand = Operand("test")
        operand.resolved_value = BooleanValues.FALSE
        assert operand.resolved_value == BooleanValues.FALSE

    def test_resolved_value_setter_invalid_type(self):
        """Test resolved_value setter with invalid type"""
        operand = Operand("test")
        with pytest.raises(RuntimeError, match="Resolved value must be of type BooleanValues"):
            operand.resolved_value = "invalid"

    def test_validate_operand_with_true_string(self):
        """Test validate_operand with 'true' string"""
        operand = Operand.validate_operand("true")
        assert isinstance(operand, Operand)
        assert operand.resolved_value == BooleanValues.TRUE

    def test_validate_operand_with_false_string(self):
        """Test validate_operand with 'false' string"""
        operand = Operand.validate_operand("false")
        assert isinstance(operand, Operand)
        assert operand.resolved_value == BooleanValues.FALSE

    def test_validate_operand_with_unknown_string(self):
        """Test validate_operand with 'unknown' string"""
        operand = Operand.validate_operand("unknown")
        assert isinstance(operand, Operand)
        assert operand.resolved_value == BooleanValues.UNKNOWN

    def test_validate_operand_with_filter_string(self):
        """Test validate_operand with filter string"""
        operand = Operand.validate_operand("task == ic")
        assert isinstance(operand, Operand)

    def test_validate_operand_with_operand_instance(self):
        """Test validate_operand with Operand instance"""
        original = Operand("test")
        result = Operand.validate_operand(original)
        assert result is original

    def test_validate_operand_with_invalid_type(self):
        """Test validate_operand with invalid type"""
        with pytest.raises(RuntimeError, match="Operand .* is not supported"):
            Operand.validate_operand(123)

    def test_iter(self):
        """Test __iter__ method"""
        operand = Operand("test")
        result = list(operand)
        assert len(result) == 1
        assert result[0] is operand


class TestAnd:
    """Test cases for And operator"""

    def test_init(self):
        """Test And initialization"""
        op1 = Operand("test1", BooleanValues.TRUE)
        op2 = Operand("test2", BooleanValues.TRUE)
        and_op = And(op1, op2)
        assert len(and_op.operands) == 2

    def test_eval_all_true(self):
        """Test And.eval with all true operands"""
        op1 = Operand("test1", BooleanValues.TRUE)
        op2 = Operand("test2", BooleanValues.TRUE)
        and_op = And(op1, op2)
        and_op.eval()
        assert and_op.resolved_value == BooleanValues.TRUE

    def test_eval_one_false(self):
        """Test And.eval with one false operand"""
        op1 = Operand("test1", BooleanValues.TRUE)
        op2 = Operand("test2", BooleanValues.FALSE)
        and_op = And(op1, op2)
        and_op.eval()
        assert and_op.resolved_value == BooleanValues.FALSE

    def test_eval_with_unknown(self):
        """Test And.eval with unknown operand"""
        op1 = Operand("test1", BooleanValues.TRUE)
        op2 = Operand("test2", BooleanValues.UNKNOWN)
        and_op = And(op1, op2)
        and_op.eval()
        assert and_op.resolved_value == BooleanValues.UNKNOWN

    def test_eval_unevaluated_operand(self):
        """Test And.eval with unevaluated operand that remains unevaluated"""
        op1 = Operand("test1", BooleanValues.TRUE)
        op2 = Operand("test2", BooleanValues.UNEVALUATED)
        op2.eval = Mock()
        
        and_op = And(op1, op2)
        with pytest.raises(RuntimeError, match="Operand remains unevaluated"):
            and_op.eval()

    def test_iter(self):
        """Test And.__iter__"""
        op1 = Operand("test1", BooleanValues.TRUE)
        op2 = Operand("test2", BooleanValues.TRUE)
        and_op = And(op1, op2)
        result = list(and_op)
        assert len(result) == 3  # 2 operands + and_op itself


class TestOr:
    """Test cases for Or operator"""

    def test_init(self):
        """Test Or initialization"""
        op1 = Operand("test1", BooleanValues.FALSE)
        op2 = Operand("test2", BooleanValues.FALSE)
        or_op = Or(op1, op2)
        assert len(or_op.operands) == 2

    def test_eval_all_false(self):
        """Test Or.eval with all false operands"""
        op1 = Operand("test1", BooleanValues.FALSE)
        op2 = Operand("test2", BooleanValues.FALSE)
        or_op = Or(op1, op2)
        or_op.eval()
        assert or_op.resolved_value == BooleanValues.FALSE

    def test_eval_one_true(self):
        """Test Or.eval with one true operand"""
        op1 = Operand("test1", BooleanValues.FALSE)
        op2 = Operand("test2", BooleanValues.TRUE)
        or_op = Or(op1, op2)
        or_op.eval()
        assert or_op.resolved_value == BooleanValues.TRUE

    def test_eval_with_unknown(self):
        """Test Or.eval with unknown operand"""
        op1 = Operand("test1", BooleanValues.FALSE)
        op2 = Operand("test2", BooleanValues.UNKNOWN)
        or_op = Or(op1, op2)
        or_op.eval()
        assert or_op.resolved_value == BooleanValues.UNKNOWN

    def test_iter(self):
        """Test Or.__iter__"""
        op1 = Operand("test1", BooleanValues.FALSE)
        op2 = Operand("test2", BooleanValues.TRUE)
        or_op = Or(op1, op2)
        result = list(or_op)
        assert len(result) == 3


class TestNot:
    """Test cases for Not operator"""

    def test_init(self):
        """Test Not initialization"""
        op = Operand("test", BooleanValues.TRUE)
        not_op = Not(op)
        assert not_op.operand is op

    def test_eval_true_to_false(self):
        """Test Not.eval with true operand"""
        op = Operand("test", BooleanValues.TRUE)
        not_op = Not(op)
        not_op.eval()
        assert not_op.resolved_value == BooleanValues.FALSE

    def test_eval_false_to_true(self):
        """Test Not.eval with false operand"""
        op = Operand("test", BooleanValues.FALSE)
        not_op = Not(op)
        not_op.eval()
        assert not_op.resolved_value == BooleanValues.TRUE

    def test_eval_unknown_stays_unknown(self):
        """Test Not.eval with unknown operand"""
        op = Operand("test", BooleanValues.UNKNOWN)
        not_op = Not(op)
        not_op.eval()
        assert not_op.resolved_value == BooleanValues.UNKNOWN

    def test_iter(self):
        """Test Not.__iter__"""
        op = Operand("test", BooleanValues.TRUE)
        not_op = Not(op)
        result = list(not_op)
        assert len(result) == 2


class TestIdentity:
    """Test cases for Identity operator"""

    def test_init(self):
        """Test Identity initialization"""
        op = Operand("test", BooleanValues.TRUE)
        identity = Identity(op)
        assert identity.operand is op

    def test_eval(self):
        """Test Identity.eval"""
        op = Operand("test", BooleanValues.TRUE)
        identity = Identity(op)
        identity.eval()
        assert identity.resolved_value == BooleanValues.TRUE

    def test_iter(self):
        """Test Identity.__iter__"""
        op = Operand("test", BooleanValues.TRUE)
        identity = Identity(op)
        result = list(identity)
        assert len(result) == 2


class TestConstant:
    """Test cases for Constant operator"""

    def test_init(self):
        """Test Constant initialization"""
        constant = Constant(BooleanValues.TRUE)
        assert constant.resolved_value == BooleanValues.TRUE

    def test_eval(self):
        """Test Constant.eval"""
        constant = Constant(BooleanValues.FALSE)
        constant.eval()
        assert constant.resolved_value == BooleanValues.FALSE


class TestModelFilter:
    """Test cases for ModelFilter"""

    def test_init(self):
        """Test ModelFilter initialization"""
        filter_obj = ModelFilter(key="task", value="ic", operator="==")
        assert filter_obj.key == "task"
        assert filter_obj.value == "ic"
        assert filter_obj.operator == "=="

    def test_set_key(self):
        """Test set_key method"""
        filter_obj = ModelFilter(key="task", value="ic", operator="==")
        filter_obj.set_key("framework")
        assert filter_obj.key == "framework"

    def test_set_value(self):
        """Test set_value method"""
        filter_obj = ModelFilter(key="task", value="ic", operator="==")
        filter_obj.set_value("pytorch")
        assert filter_obj.value == "pytorch"


class TestParseFilterString:
    """Test cases for parse_filter_string"""

    def test_parse_equals(self):
        """Test parsing equals operator"""
        result = parse_filter_string("task == ic")
        assert result.key == "task"
        assert result.value == "ic"
        assert result.operator == "=="

    def test_parse_not_equals(self):
        """Test parsing not equals operator"""
        result = parse_filter_string("task != ic")
        assert result.key == "task"
        assert result.value == "ic"
        assert result.operator == "!="

    def test_parse_in(self):
        """Test parsing in operator"""
        result = parse_filter_string("task in ['ic', 'od']")
        assert result.key == "task"
        assert result.value == "['ic', 'od']"
        assert result.operator == "in"

    def test_parse_includes(self):
        """Test parsing includes operator"""
        result = parse_filter_string("description includes pytorch")
        assert result.key == "description"
        assert result.value == "pytorch"
        assert result.operator == "includes"

    def test_parse_begins_with(self):
        """Test parsing begins with operator"""
        result = parse_filter_string("model_id begins with huggingface")
        assert result.key == "model_id"
        assert result.value == "huggingface"
        assert result.operator == "begins with"

    def test_parse_ends_with(self):
        """Test parsing ends with operator"""
        result = parse_filter_string("model_id ends with v1.0")
        assert result.key == "model_id"
        assert result.value == "v1.0"
        assert result.operator == "ends with"

    def test_parse_invalid_filter(self):
        """Test parsing invalid filter string"""
        with pytest.raises(ValueError, match="Cannot parse filter string"):
            parse_filter_string("invalid filter")


class TestEvaluateFilterExpression:
    """Test cases for evaluate_filter_expression"""

    def test_equals_true(self):
        """Test equals operator returns true"""
        filter_obj = ModelFilter(key="task", value="ic", operator="==")
        result = evaluate_filter_expression(filter_obj, "ic")
        assert result == BooleanValues.TRUE

    def test_equals_false(self):
        """Test equals operator returns false"""
        filter_obj = ModelFilter(key="task", value="ic", operator="==")
        result = evaluate_filter_expression(filter_obj, "od")
        assert result == BooleanValues.FALSE

    def test_equals_with_none(self):
        """Test equals operator with None value"""
        filter_obj = ModelFilter(key="task", value="ic", operator="==")
        result = evaluate_filter_expression(filter_obj, None)
        assert result == BooleanValues.FALSE

    def test_equals_with_boolean(self):
        """Test equals operator with boolean value"""
        filter_obj = ModelFilter(key="enabled", value="true", operator="==")
        result = evaluate_filter_expression(filter_obj, True)
        assert result == BooleanValues.TRUE

    def test_not_equals(self):
        """Test not equals operator"""
        filter_obj = ModelFilter(key="task", value="ic", operator="!=")
        result = evaluate_filter_expression(filter_obj, "od")
        assert result == BooleanValues.TRUE

    def test_in_operator(self):
        """Test in operator"""
        filter_obj = ModelFilter(key="task", value="['ic', 'od']", operator="in")
        result = evaluate_filter_expression(filter_obj, "ic")
        assert result == BooleanValues.TRUE

    def test_in_operator_false(self):
        """Test in operator returns false"""
        filter_obj = ModelFilter(key="task", value="['ic', 'od']", operator="in")
        result = evaluate_filter_expression(filter_obj, "nlp")
        assert result == BooleanValues.FALSE

    def test_not_in_operator(self):
        """Test not in operator"""
        filter_obj = ModelFilter(key="task", value="['ic', 'od']", operator="not in")
        result = evaluate_filter_expression(filter_obj, "nlp")
        assert result == BooleanValues.TRUE

    def test_includes_operator(self):
        """Test includes operator"""
        filter_obj = ModelFilter(key="description", value="pytorch", operator="includes")
        result = evaluate_filter_expression(filter_obj, "This is a pytorch model")
        assert result == BooleanValues.TRUE

    def test_includes_operator_false(self):
        """Test includes operator returns false"""
        filter_obj = ModelFilter(key="description", value="pytorch", operator="includes")
        result = evaluate_filter_expression(filter_obj, "This is a tensorflow model")
        assert result == BooleanValues.FALSE

    def test_not_includes_operator(self):
        """Test not includes operator"""
        filter_obj = ModelFilter(key="description", value="pytorch", operator="not includes")
        result = evaluate_filter_expression(filter_obj, "This is a tensorflow model")
        assert result == BooleanValues.TRUE

    def test_begins_with_operator(self):
        """Test begins with operator"""
        filter_obj = ModelFilter(key="model_id", value="huggingface", operator="begins with")
        result = evaluate_filter_expression(filter_obj, "huggingface-bert-base")
        assert result == BooleanValues.TRUE

    def test_begins_with_operator_false(self):
        """Test begins with operator returns false"""
        filter_obj = ModelFilter(key="model_id", value="huggingface", operator="begins with")
        result = evaluate_filter_expression(filter_obj, "pytorch-bert-base")
        assert result == BooleanValues.FALSE

    def test_ends_with_operator(self):
        """Test ends with operator"""
        filter_obj = ModelFilter(key="model_id", value="v1.0", operator="ends with")
        result = evaluate_filter_expression(filter_obj, "model-v1.0")
        assert result == BooleanValues.TRUE

    def test_ends_with_operator_false(self):
        """Test ends with operator returns false"""
        filter_obj = ModelFilter(key="model_id", value="v1.0", operator="ends with")
        result = evaluate_filter_expression(filter_obj, "model-v2.0")
        assert result == BooleanValues.FALSE

    def test_invalid_operator(self):
        """Test invalid operator raises error"""
        filter_obj = ModelFilter(key="task", value="ic", operator="invalid")
        with pytest.raises(RuntimeError, match="Bad operator"):
            evaluate_filter_expression(filter_obj, "ic")


class TestHelperFunctions:
    """Test cases for helper functions"""

    def test_negate_boolean_true(self):
        """Test _negate_boolean with TRUE"""
        result = _negate_boolean(BooleanValues.TRUE)
        assert result == BooleanValues.FALSE

    def test_negate_boolean_false(self):
        """Test _negate_boolean with FALSE"""
        result = _negate_boolean(BooleanValues.FALSE)
        assert result == BooleanValues.TRUE

    def test_negate_boolean_unknown(self):
        """Test _negate_boolean with UNKNOWN"""
        result = _negate_boolean(BooleanValues.UNKNOWN)
        assert result == BooleanValues.UNKNOWN

    def test_evaluate_filter_expression_in_with_list_value(self):
        """Test _evaluate_filter_expression_in with list cached value"""
        filter_obj = ModelFilter(key="task", value="ic", operator="in")
        result = _evaluate_filter_expression_in(filter_obj, ["ic", "od"])
        assert result == BooleanValues.FALSE
