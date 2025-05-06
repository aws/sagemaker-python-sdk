import abc

from enum import EnumMeta
from typing import Any, Dict, List, Union

PrimitiveType = Union[str, int, bool, float, None]
RequestType = Union[Dict[str, Any], List[Dict[str, Any]]]


class Entity(abc.ABC):
    """Base object for workflow entities.
    Entities must implement the to_request method.
    """

    @abc.abstractmethod
    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""


class DefaultEnumMeta(EnumMeta):
    """An EnumMeta which defaults to the first value in the Enum list."""

    default = object()

    def __call__(cls, *args, value=default, **kwargs):
        """Defaults to the first value in the Enum list."""
        if value is DefaultEnumMeta.default:
            return next(iter(cls))
        return super().__call__(value, *args, **kwargs)

    factory = __call__


class Expression(abc.ABC):
    """Base object for expressions.
    Expressions must implement the expr property.
    """

    @property
    @abc.abstractmethod
    def expr(self) -> RequestType:
        """Get the expression structure for workflow service calls."""


class PipelineVariable(Expression):
    """Base object for pipeline variables
    PipelineVariable subclasses must implement the expr property. Its subclasses include:
    :class:`~sagemaker.workflow.parameters.Parameter`,
    :class:`~sagemaker.workflow.properties.Properties`,
    :class:`~sagemaker.workflow.functions.Join`,
    :class:`~sagemaker.workflow.functions.JsonGet`,
    :class:`~sagemaker.workflow.execution_variables.ExecutionVariable`.
    :class:`~sagemaker.workflow.step_outputs.StepOutput`.
    """

    def __add__(self, other: Union[Expression, PrimitiveType]):
        """Add function for PipelineVariable
        Args:
            other (Union[Expression, PrimitiveType]): The other object to be concatenated.
        Always raise an error since pipeline variables do not support concatenation
        """

        raise TypeError("Pipeline variables do not support concatenation.")
