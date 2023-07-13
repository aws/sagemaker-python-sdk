from dataclasses import dataclass
from sagemaker.jumpstart.curated_hub.utils import construct_s3_uri
from typing import Dict

@dataclass
class S3ObjectReference:
    bucket: str
    key: str

    def format_for_s3_copy(self) -> Dict[str, str]:
        return {
          "Bucket": self.bucket,
          "Key": self.key,
        }
    
    def get_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}" 
        
def create_s3_object_reference_from_bucket_and_key(bucket: str, key: str) -> S3ObjectReference:
    return S3ObjectReference(
        bucket=bucket,
        key=key
    )

def create_s3_object_reference_from_uri(s3_uri: str) -> S3ObjectReference:
    uri_with_s3_prefix_removed = s3_uri.replace("s3://", "", 1)
    uri_split = uri_with_s3_prefix_removed.split("/")

    return S3ObjectReference(
        bucket=uri_split[0],
        key="/".join(uri_split[1:]),
    )
        