import re

import boto3
from botocore.exceptions import ClientError


def is_bucket_accessible(bucket_name):
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 403:
            print(f"Bucket {bucket_name} exists, but you don't have permission to access it.")
        elif error_code == 404:
            print(f"Bucket {bucket_name} does not exist.")
        else:
            print(f"Error checking bucket {bucket_name}: {e}")
        return False


def validate_s3_references(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    s3_pattern = re.compile(r"s3:\/\/([a-zA-Z0-9._-]+)")
    matches = s3_pattern.findall(content)

    invalid_buckets = []
    for bucket in matches:
        if not is_bucket_accessible(bucket):
            invalid_buckets.append(bucket)

    return invalid_buckets


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python s3_bucket_validator.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    invalid_buckets = validate_s3_references(file_path)

    if invalid_buckets:
        print(f"Invalid or inaccessible S3 buckets found: {', '.join(invalid_buckets)}")
        sys.exit(1)
    else:
        print("All referenced S3 buckets are valid and accessible.")
