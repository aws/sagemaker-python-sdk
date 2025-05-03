from enum import Enum

from sagemaker.core.utils.utils import remove_html_tags


class MethodType(Enum):
    CLASS = "class"
    OBJECT = "object"
    STATIC = "static"


class Method:
    """
    A class to store the information of methods to be generated
    """

    operation_name: str
    resource_name: str
    method_name: str
    return_type: str
    method_type: MethodType
    service_name: str
    docstring_title: str

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_docstring_title(self, operation):
        documentation = operation.get("documentation")
        title = remove_html_tags(documentation) if documentation else None
        self.docstring_title = title.split(".")[0] + "." if title else None

    # TODO: add some templates for common methods
