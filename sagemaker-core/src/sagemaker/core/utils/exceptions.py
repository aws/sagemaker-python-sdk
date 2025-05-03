class SageMakerCoreError(Exception):
    """Base class for all exceptions in SageMaker Core"""

    fmt = "An unspecified error occurred."

    def __init__(self, **kwargs):
        """Initialize a SageMakerCoreError exception.

        Args:
            **kwargs: Keyword arguments to be formatted into the custom error message template.
        """
        msg = self.fmt.format(**kwargs)
        Exception.__init__(self, msg)


### Generic Validation Errors
class ValidationError(SageMakerCoreError):
    """Raised when a validation error occurs."""

    fmt = "An error occurred while validating user input/setup. {message}"

    def __init__(self, message="", **kwargs):
        """Initialize a ValidationError exception.

        Args:
            message (str): A message describing the error.
        """
        super().__init__(message=message, **kwargs)


### Waiter Errors
class WaiterError(SageMakerCoreError):
    """Raised when an error occurs while waiting."""

    fmt = "An error occurred while waiting for {resource_type}. Final Resource State: {status}."

    def __init__(self, resource_type="(Unkown)", status="(Unkown)", **kwargs):
        """Initialize a WaiterError exception.

        Args:
            resource_type (str): The type of resource being waited on.
            status (str): The final status of the resource.
        """
        super().__init__(resource_type=resource_type, status=status, **kwargs)


class FailedStatusError(WaiterError):
    """Raised when a resource enters a failed state."""

    fmt = "Encountered unexpected failed state while waiting for {resource_type}. Final Resource State: {status}. Failure Reason: {reason}"

    def __init__(self, resource_type="(Unkown)", status="(Unkown)", reason="(Unkown)"):
        """Initialize a FailedStatusError exception.

        Args:
            resource_type (str): The type of resource being waited on.
            status (str): The final status of the resource.
            reason (str): The reason the resource entered a failed state.
        """
        super().__init__(resource_type=resource_type, status=status, reason=reason)


class DeleteFailedStatusError(WaiterError):
    """Raised when a resource enters a delete_failed state."""

    fmt = "Encountered unexpected delete_failed state while deleting {resource_type}. Failure Reason: {reason}"

    def __init__(self, resource_type="(Unkown)", reason="(Unkown)"):
        """Initialize a FailedStatusError exception.

        Args:
            resource_type (str): The type of resource being waited on.
            status (str): The final status of the resource.
            reason (str): The reason the resource entered a failed state.
        """
        super().__init__(resource_type=resource_type, reason=reason)


class TimeoutExceededError(WaiterError):
    """Raised when a specified timeout is exceeded"""

    fmt = "Timeout exceeded while waiting for {resource_type}. Final Resource State: {status}. Increase the timeout and try again."

    def __init__(self, resource_type="(Unkown)", status="(Unkown)", reason="(Unkown)"):
        """Initialize a TimeoutExceededError exception.
        Args:
            resource_type (str): The type of resource being waited on.
            status (str): The final status of the resource.
            reason (str): The reason the resource entered a failed state.
        """
        super().__init__(resource_type=resource_type, status=status, reason=reason)


### Intelligent Defaults Errors
class IntelligentDefaultsError(SageMakerCoreError):
    """Raised when an error occurs in the Intelligent Defaults"""

    fmt = "An error occurred while loading Intelligent Default. {message}"

    def __init__(self, message="", **kwargs):
        """Initialize an IntelligentDefaultsError exception.
        Args:
            message (str): A message describing the error.
        """
        super().__init__(message=message, **kwargs)


class LocalConfigNotFoundError(IntelligentDefaultsError):
    """Raised when a configuration file is not found in local file system"""

    fmt = "Failed to load configuration file from location: {file_path}. {message}"

    def __init__(self, file_path="(Unkown)", message=""):
        """Initialize a LocalConfigNotFoundError exception.
        Args:
            file_path (str): The path to the configuration file.
            message (str): A message describing the error.
        """
        super().__init__(file_path=file_path, message=message)


class S3ConfigNotFoundError(IntelligentDefaultsError):
    """Raised when a configuration file is not found in S3"""

    fmt = "Failed to load configuration file from S3 location: {s3_uri}. {message}"

    def __init__(self, s3_uri="(Unkown)", message=""):
        """Initialize a S3ConfigNotFoundError exception.
        Args:
            s3_uri (str): The S3 URI path to the configuration file.
            message (str): A message describing the error.
        """
        super().__init__(s3_uri=s3_uri, message=message)


class ConfigSchemaValidationError(IntelligentDefaultsError, ValidationError):
    """Raised when a configuration file does not adhere to the schema"""

    fmt = "Failed to validate configuration file from location: {file_path}. {message}"

    def __init__(self, file_path="(Unkown)", message=""):
        """Initialize a ConfigSchemaValidationError exception.
        Args:
            file_path (str): The path to the configuration file.
            message (str): A message describing the error.
        """
        super().__init__(file_path=file_path, message=message)
