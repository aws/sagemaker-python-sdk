import inspect
import logging
from functools import wraps
from sagemaker.utils.workflow.parameters import Parameter

logger = logging.getLogger(__name__)


def override_pipeline_parameter_var(func):
    """A decorator to override pipeline Parameters passed into a function
    This is a temporary decorator to override pipeline Parameter objects with their default value
    and display warning information to instruct users to update their code.
    This decorator can help to give a grace period for users to update their code when
    we make changes to explicitly prevent passing any pipeline variables to a function.
    We should remove this decorator after the grace period.
    """
    warning_msg_template = (
        "The input argument %s of function (%s) is a pipeline variable (%s), "
        "which is interpreted in pipeline execution time only. "
        "As the function needs to evaluate the argument value in SDK compile time, "
        "the default_value of this Parameter object will be used to override it. "
        "Please make sure the default_value is valid."
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = "{}.{}".format(func.__module__, func.__name__)
        params = inspect.signature(func).parameters
        args = list(args)
        for i, (arg_name, _) in enumerate(params.items()):
            if i >= len(args):
                break
            if isinstance(args[i], Parameter):
                logger.warning(warning_msg_template, arg_name, func_name, type(args[i]))
                args[i] = args[i].default_value
        args = tuple(args)

        for arg_name, value in kwargs.items():
            if isinstance(value, Parameter):
                logger.warning(warning_msg_template, arg_name, func_name, type(value))
                kwargs[arg_name] = value.default_value
        return func(*args, **kwargs)

    return wrapper
