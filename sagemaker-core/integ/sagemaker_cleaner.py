import datetime
from sagemaker.core.resources import Model, EndpointConfig, Endpoint


class SageMakerCleaner:
    """Provides methods to cleanup SageMaker resources"""

    def __init__(self):
        """Initialize a SageMakerCleaner

        Args:
            client (client): A boto3 Session SageMaker Client
        """
        self.resource_tracker = {"total_deleted": 0, "total_failed": 0}

    def handle_cleanup(self, before_timestamp, after_timestamp):
        """Handles deletion for Sagmeker resources

        Args:
            before_timestamp (datetime): timestamp for 'CreationTimeBefore' or 'CreatedBefore' boto3 parameter
            after_timestamp (datetime): timestamp for 'CreationTimeAfter' or 'CreatedAfter' boto3 parameter
        """
        RESOURCE_TYPE_ORDER = [
            "Endpoints",
            "EndpointConfigs",
            "Models",
        ]

        CLEANUP_METHODS = {
            "Endpoints": self.cleanup_endpoints,
            "EndpointConfigs": self.cleanup_endpoint_configs,
            "Models": self.cleanup_models,
        }
        for resource_type in RESOURCE_TYPE_ORDER:
            CLEANUP_METHODS[resource_type](before_timestamp, after_timestamp)

    def cleanup_endpoints(self, creation_time_before, creation_time_after):
        """Deletes Models before a given timestamp

        Args:
            creation_time_before (datetime): timestamp for 'CreationTimeBefore' or 'CreatedBefore' boto3 parameter
            creation_time_after (datetime): timestamp for 'CreationTimeAfter' or 'CreatedAfter' boto3 parameter
        """
        endpoints = Endpoint.get_all(
            creation_time_before=creation_time_before, creation_time_after=creation_time_after
        )
        for endpoint in endpoints:
            try:
                endpoint.delete()
            except:
                self._track_resource(failed=1)
        self._track_resource(deleted=1)

    def cleanup_endpoint_configs(self, creation_time_before, creation_time_after):
        """Deletes Models before a given timestamp

        Args:
            creation_time_before (datetime): timestamp for 'CreationTimeBefore' or 'CreatedBefore' boto3 parameter
            creation_time_after (datetime): timestamp for 'CreationTimeAfter' or 'CreatedAfter' boto3 parameter
        """
        endpoint_configs = EndpointConfig.get_all(
            creation_time_before=creation_time_before, creation_time_after=creation_time_after
        )
        for endpoint_config in endpoint_configs:
            try:
                endpoint_config.delete()
            except:
                self._track_resource(failed=1)
        self._track_resource(deleted=1)

    def cleanup_models(self, creation_time_before, creation_time_after):
        """Deletes Models before a given timestamp

        Args:
            creation_time_before (datetime): timestamp for 'CreationTimeBefore' or 'CreatedBefore' boto3 parameter
            creation_time_after (datetime): timestamp for 'CreationTimeAfter' or 'CreatedAfter' boto3 parameter
        """
        models = Model.get_all(
            creation_time_before=creation_time_before, creation_time_after=creation_time_after
        )
        for model in models:
            try:
                model.delete()
            except:
                self._track_resource(failed=1)
        self._track_resource(deleted=1)

    def _track_resource(self, deleted=0, failed=0):
        """Updates the resource tracker with # of deleted, or failed resources

        Args:
            deleted (int): # of deleted resources to add to tracker
            failed (int): # of failed resources to add to tracker
        """
        self.resource_tracker["total_deleted"] += deleted
        self.resource_tracker["total_failed"] += failed


def handle_cleanup():
    region = "us-west-2"
    print(f"\n\n=========== Cleaning SageMaker Resources in {region} ===========")
    before_timestamp = datetime.datetime.now(datetime.timezone.utc)
    after_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(weeks=2)

    sagemaker_cleaner = SageMakerCleaner()
    sagemaker_cleaner.handle_cleanup(before_timestamp, after_timestamp)

    print(f"resource_tracker: {sagemaker_cleaner.resource_tracker}")
