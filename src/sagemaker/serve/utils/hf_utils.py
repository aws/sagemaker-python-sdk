import json
import urllib.request
from json import JSONDecodeError
from urllib.error import HTTPError, URLError
import logging

logger = logging.getLogger(__name__)


def _get_model_config_properties_from_hf(model_id: str, hf_hub_token: str = None):
    """Placeholder docstring"""

    config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    model_config = None
    try:
        if hf_hub_token:
            config_url = urllib.request.Request(
                config_url, headers={"Authorization": "Bearer " + hf_hub_token}
            )
        with urllib.request.urlopen(config_url) as response:
            model_config = json.load(response)
    except (HTTPError, URLError, TimeoutError, JSONDecodeError) as e:
        if "HTTP Error 401: Unauthorized" in str(e):
            raise ValueError(
                "Trying to access a gated/private HuggingFace model without valid credentials. "
                "Please provide a HUGGING_FACE_HUB_TOKEN in env_vars"
            )
        logger.warning(
            "Exception encountered while trying to read config file %s. " "Details: %s",
            config_url,
            e,
        )
    if not model_config:
        raise ValueError(
            f"Did not find a config.json or model_index.json file in huggingface hub for "
            f"{model_id}. Please make sure a config.json exists (or model_index.json for Stable "
            f"Diffusion Models) for this model in the huggingface hub"
        )
    return model_config
