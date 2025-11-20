import abc
from typing import Dict, List, Union, Any

try:
    from pydantic_core import core_schema
except ImportError:
    core_schema = None

PrimitiveType = Union[str, int, bool, float, None]
RequestType = Union[Dict[str, Any], List[Dict[str, Any]]]


class PipelineVariable(abc.ABC):
    """PipelineVariables are placeholders for strings that are unknown before Pipeline execution

    PipelineVariable subclasses must implement the expr property. Its subclasses include:
    :class:`~sagemaker.workflow.parameters.Parameter`,
    :class:`~sagemaker.workflow.properties.Properties`,
    :class:`~sagemaker.workflow.functions.Join`,
    :class:`~sagemaker.workflow.functions.JsonGet`,
    :class:`~sagemaker.workflow.execution_variables.ExecutionVariable`.
    :class:`~sagemaker.workflow.step_outputs.StepOutput`.
    """

    def __add__(self, other: Union["PipelineVariable", PrimitiveType]):
        """Add function for PipelineVariable

        Args:
            other (Union[PipelineVariable, PrimitiveType]): The other object to be concatenated.

        Always raise an error since pipeline variables do not support concatenation
        """

        raise TypeError("Pipeline variables do not support concatenation.")

    def __str__(self):
        """Override built-in String function for PipelineVariable"""
        raise TypeError(
            "Pipeline variables do not support __str__ operation. "
            "Please use `.to_string()` to convert it to string type in execution time "
            "or use `.expr` to translate it to Json for display purpose in Python SDK."
        )

    def __int__(self):
        """Override built-in Integer function for PipelineVariable"""
        raise TypeError("Pipeline variables do not support __int__ operation.")

    def __float__(self):
        """Override built-in Float function for PipelineVariable"""
        raise TypeError("Pipeline variables do not support __float__ operation.")

    def to_string(self):
        """Prompt the pipeline to convert the pipeline variable to String in runtime"""
        from sagemaker.core.workflow.functions import Join

        return Join(on="", values=[self])

    @property
    @abc.abstractmethod
    def expr(self) -> RequestType:
        """Get the expression structure for workflow service calls."""

    @property
    def _pickleable(self):
        """A pickleable object that can be used in a function step."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _referenced_steps(self) -> List[Any]:
        """List of steps that this variable is generated from."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Return a Pydantic core schema for PipelineVariable validation."""
        if core_schema is None:
            raise ImportError("pydantic-core is required for Pydantic validation")
        return core_schema.is_instance_schema(cls)


# This is a type that could be either string or pipeline variable
StrPipeVar = Union[str, PipelineVariable]
