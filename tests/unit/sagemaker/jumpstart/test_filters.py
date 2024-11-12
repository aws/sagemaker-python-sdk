from __future__ import absolute_import

from unittest import TestCase

from sagemaker.jumpstart.filters import (
    FILTER_OPERATOR_STRING_MAPPINGS,
    And,
    BooleanValues,
    FilterOperators,
    Identity,
    ModelFilter,
    Not,
    Or,
    parse_filter_string,
)
from sagemaker.jumpstart.filters import evaluate_filter_expression


class TestEvaluateFilterExpression(TestCase):
    def test_equals(self):

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="equals", value="5"), 5
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="equals", value="True"), True
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="equals", value="mom"), "mom"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="equals", value="5"), 6
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="equals", value="True"), False
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="equals", value="mom"), "moms"
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="is", value="5"), 5
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="is", value="True"), True
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="is", value="mom"), "mom"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="is", value="5"), 6
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="is", value="True"), False
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="is", value="mom"), "moms"
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="==", value="5"), 5
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="==", value="True"), True
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="==", value="mom"), "mom"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="==", value="5"), 6
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="==", value="True"), False
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="==", value="mom"), "moms"
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="===", value="5"), 5
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="===", value="True"), True
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="===", value="mom"), "mom"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="===", value="5"), 6
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="===", value="True"), False
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="===", value="mom"), "moms"
        )

    def test_not_equals(self):

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not equals", value="5"), 5
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not equals", value="True"), True
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not equals", value="mom"), "mom"
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="is not", value="5"), 6
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="!=", value="True"), False
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not equals", value="mom"), "moms"
        )

    def test_in(self):

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="in", value="daddy"), "dad"
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="in", value='["mom", "dad"]'), "dad"
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="in", value='["mom", 1]'), 1
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="in", value='["mom", "fsdfdsfsd", False]'), False
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="in", value='["mom"]'), "dad"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="in", value='["mom"]'), 1
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="in", value='["mom", "fsdfdsfsd"]'), False
        )

    def test_not_in(self):

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not in", value="daddy"), "dad"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not in", value='["mom", "dad"]'), "dad"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not in", value='["mom", 1]'), 1
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not in", value='["mom", "fsdfdsfsd", False]'), False
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not in", value='["mom"]'), "dad"
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not in", value='["mom"]'), 1
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not in", value='["mom", "fsdfdsfsd"]'), False
        )

    def test_includes(self):

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="includes", value="dad"), "daddy"
        )

        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="includes", value="dad"), ["dad"]
        )

    def test_not_includes(self):

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not includes", value="dad"), "daddy"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="not includes", value="dad"), ["dad"]
        )

    def test_begins_with(self):
        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="begins with", value="dad"), "daddy"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="begins with", value="mm"), "mommy"
        )

    def test_ends_with(self):
        assert BooleanValues.TRUE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="ends with", value="car"), "racecar"
        )

        assert BooleanValues.FALSE == evaluate_filter_expression(
            ModelFilter(key="hello", operator="begins with", value="ace"), "racecar"
        )


def test_parse_filter_string():

    values = ["ic", "tc", "od", "pytorch", "huggingface", str(True), str(False), "od1"]
    for key in ["task", "framework", "training_supported", "model_id"]:
        for value in values:
            for operator in (
                FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_EQUALS]
                + FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.EQUALS]
            ):
                model_filter = parse_filter_string(f"{key} {operator} {value}")
                assert model_filter.key == key
                assert model_filter.operator == operator
                assert model_filter.value == value

                is_alphabetic_operator = any(character.isalpha() for character in operator)
                if not is_alphabetic_operator:
                    model_filter = parse_filter_string(f"{key}{operator}{value}")
                    assert model_filter.key == key
                    assert model_filter.operator == operator
                    assert model_filter.value == value

    for key in ["task", "framework", "training_supported", "model_id"]:
        for value in [
            str([]),
            str(["some-val"]),
            str(["some-val-1", "some-val-2"]),
            str([True]),
            str([True, False]),
            str([1]),
            str([1, 2]),
        ]:
            for operator in (
                FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_IN]
                + FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.IN]
            ):
                value = sorted(list(value))
                model_filter = parse_filter_string(f"{key} {operator} {value}")
                assert model_filter.key == key
                assert model_filter.operator == operator
                assert model_filter.value == str(value)


class TestFilterOperators(TestCase):
    def test_and(self):
        operation = And("True", "True")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value

        operation = And("True", "False")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.FALSE == resolved_value

        operation = And("True", "Unknown")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.UNKNOWN == resolved_value

        operation = And("False", "False")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.FALSE == resolved_value

        operation = And("False", "False", "False")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.FALSE == resolved_value

    def test_or(self):
        operation = Or("True", "True")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value

        operation = Or("True", "False")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value

        operation = Or("True", "Unknown")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value

        operation = Or("False", "False")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.FALSE == resolved_value

        operation = Or("False", "False", "False")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.FALSE == resolved_value

        operation = Or("False", "False", "True")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value

    def test_not(self):
        operation = Not("False")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value

        operation = Not("TrUe")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.FALSE == resolved_value

        operation = Not("unknown")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.UNKNOWN == resolved_value

    def test_identity(self):
        operation = Identity("False")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.FALSE == resolved_value

        operation = Identity("TrUe")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value

        operation = Identity("unknown")
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.UNKNOWN == resolved_value

    def test_complex_expressions(self):
        operation = Not(
            And(Or("False", "True", "UnknoWn"), "TrUE", Or("False", "False", And("True")))
        )
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.FALSE == resolved_value

        operation = Not(
            And(
                Or("False", "True", "UnknoWn"),
                Identity(Not(And("True", "True"))),
                Or("False", "False", And("True")),
            )
        )
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value

        operation = Not(
            And(
                Or("False", "True", "UnknoWn"),
                Identity(Not(And("True", "True"))),
                Not(Or("False", "False", And("True"))),
            )
        )
        operation.eval()

        resolved_value = operation.resolved_value
        assert BooleanValues.TRUE == resolved_value
