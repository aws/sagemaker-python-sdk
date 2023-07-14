import boto3
from botocore.exceptions import ClientError

from sagemaker.jumpstart.curated_hub.utils import PublicModelId
from sagemaker.jumpstart.types import JumpStartModelSpecs


class CuratedHubClient:
    """Calls SageMaker Hub APIs for the curated hub."""

    def __init__(self, curated_hub_name: str, region: str) -> None:
        """Sets up region and underlying client."""
        self.curated_hub_name = curated_hub_name
        self._region = region
        self._sm_client = boto3.client("sagemaker", region_name=self._region)

    def desribe_model(self, model_specs: JumpStartModelSpecs):
        self._sm_client.describe_hub_content(
                HubName=self.curated_hub_name,
                HubContentName=model_specs.model_id,
                HubContentType="Model",
                HubContentVersion=model_specs.version,
            )

    def delete_model(self, model_id: PublicModelId):
        print(f"Deleting model {model_id.id} from curated hub...")
        content_versions = self._list_hub_content_versions_no_content_noop(model_id.id)

        print(f"Found {len(content_versions)} versions of {model_id.id}. Deleting all versions...")

        for content_version in content_versions:
            self._sm_client.delete_hub_content(
                HubName=self.curated_hub_name,
                HubContentName=model_id.id,
                HubContentType="Model",
                HubContentVersion=content_version["HubContentVersion"],
            )

        print(f"Deleting model {model_id.id} from curated hub complete!")

    def _list_hub_content_versions_no_content_noop(self, hub_content_name: str):
        content_versions = []
        try:
            response = self._sm_client.list_hub_content_versions(
                HubName=self.curated_hub_name,
                HubContentName=hub_content_name,
                HubContentType="Model",
            )
            content_versions = response.pop("HubContentSummaries")
        except ClientError as ex:
            if ex.response["Error"]["Code"] != "ResourceNotFound":
                raise ex

        return content_versions
