from typing import Optional, Tuple

from sagemaker.image_uris import config_for_framework, retrieve


def get_latest_container_image(framework: str,
                               image_scope: str,
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
                         version=version)
    return image_uri, version


def _fetch_latest_version_from_config(framework_config: dict, image_scope: str) -> str:
    return framework_config.get(image_scope).get("version_aliases").get("latest")
