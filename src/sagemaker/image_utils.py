from typing import Optional, Tuple

from sagemaker.image_uris import config_for_framework, retrieve
from packaging.version import Version

def get_latest_container_image(framework: str,
                               image_scope: str,
                               instance_type: Optional[str] = None,
                               py_version: Optional[str] = None,
                               region: str = "us-west-2",
                               version: Optional[str] = None) -> Tuple[str, str]:
    try:
        framework_config = config_for_framework(framework)
    except FileNotFoundError:
        raise ValueError("Invalid framework {}".format(framework))

    if not framework_config:
        raise ValueError("Invalid framework {}".format(framework))

    if not version:
        version = _fetch_latest_version_from_config(framework_config, image_scope)
    image_uri = retrieve(framework=framework,
                         region=region,
                         version=version,
                         instance_type=instance_type,
                         py_version=py_version
                         )
    return image_uri, version


def _fetch_latest_version_from_config(framework_config: dict, image_scope: str) -> str:
    if image_scope in framework_config:
        if image_scope_config := framework_config[image_scope]:
            if version_aliases := image_scope_config["version_aliases"]:
                if latest_version := version_aliases["latest"]:
                    return latest_version
    versions = list(framework_config["versions"].keys())
    top_version = versions[0]
    bottom_version = versions[-1]

    if Version(top_version) >= Version(bottom_version):
        return top_version
    return bottom_version
