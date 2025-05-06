"""Defines Types etc. used in workflow."""

from __future__ import absolute_import

from sagemaker.utils.workflow.entities import Expression
from sagemaker.utils.workflow.parameters import ParameterString


def is_pipeline_variable(var: object) -> bool:
    """Check if the variable is a pipeline variable
    Args:
        var (object): The variable to be verified.
    Returns:
         bool: True if it is, False otherwise.
    """

    # Currently Expression is on top of all kinds of pipeline variables
    # as well as PipelineExperimentConfigProperty and PropertyFile
    # TODO: We should deprecate the Expression and replace it with PipelineVariable
    return isinstance(var, Expression)


def is_pipeline_parameter_string(var: object) -> bool:
    """Check if the variable is a pipeline parameter string
    Args:
        var (object): The variable to be verified.
    Returns:
         bool: True if it is, False otherwise.
    """
    return isinstance(var, ParameterString)
