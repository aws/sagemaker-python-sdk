import boto3
import json
from packaging.version import Version

JUMPSTART_BUCKET = "jumpstart-cache-prod-us-west-2"
SDK_MANIFEST_FILE = "models_manifest.json"


def get_jumpstart_sdk_manifest():
    s3_client = boto3.client("s3")
    studio_metadata = (
        s3_client.get_object(Bucket=JUMPSTART_BUCKET, Key=SDK_MANIFEST_FILE)["Body"]
        .read()
        .decode("utf-8")
    )
    return json.loads(studio_metadata)


def get_jumpstart_sdk_spec(key):
    s3_client = boto3.client("s3")
    spec = s3_client.get_object(Bucket=JUMPSTART_BUCKET, Key=key)["Body"].read().decode("utf-8")
    return json.loads(spec)

def create_jumpstart_model_table():
    sdk_manifest = get_jumpstart_sdk_manifest()
    sdk_manifest_top_versions_for_models = {}

    for model in sdk_manifest:
        if model["model_id"] not in sdk_manifest_top_versions_for_models:
            sdk_manifest_top_versions_for_models[model["model_id"]] = model
        else:
            if Version(sdk_manifest_top_versions_for_models[model["model_id"]]["version"]) < Version(
                model["version"]
            ):
                sdk_manifest_top_versions_for_models[model["model_id"]] = model

    f = open("jumpstart/jumpstart.rst", "w")

    f.write('==================================\n')
    f.write('JumpStart Available Model Table\n')
    f.write('==================================\n')
    f.write('\n')
    f.write('JumpStart for the SageMaker Python SDK uses model ids and model versions to access the necessary utilities. This table serves to provide the core material plus some extra information that can be useful in selecting the correct model id and corresponding parameters.\n')
    f.write('\n')
    f.write('If you want to automatically use the latest version of the model, use "*" for the `model_version` attribute. We highly suggest pinning an exact model version however.\n')
    f.write('\n')
    f.write('.. list-table:: Available Models\n')
    f.write('   :widths: 50 20 20 20\n')
    f.write('   :header-rows: 1\n')
    f.write('   :class: datatable\n')
    f.write('\n')
    f.write('   * - Model ID\n')
    f.write('     - Fine Tunable?\n')
    f.write('     - Latest Version\n')
    f.write('     - Min SDK Version\n')

    for model in sorted(sdk_manifest, key=lambda elt: elt["model_id"]):
        model_spec = get_jumpstart_sdk_spec(model["spec_key"])
        f.write('   * - {}\n'.format(model["model_id"]))
        f.write('     - {}\n'.format(model_spec["training_supported"]))
        f.write('     - {}\n'.format(model["version"]))
        f.write('     - {}\n'.format(model["min_version"]))

    f.close()