import logging
import os
import boto3
import json
from typing import Dict, Set, Type
from sagemaker_core.helper.session_helper import Session
from sagemaker.utils.jumpstart.types import JumpStartLaunchedRegionInfo
from sagemaker.utils.jumpstart.enums import JumpStartScriptScope

ENV_VARIABLE_DISABLE_JUMPSTART_LOGGING = "DISABLE_JUMPSTART_LOGGING"
JUMPSTART_LOGGER = logging.getLogger("sagemaker.utils.jumpstart")

# disable logging if env var is set
JUMPSTART_LOGGER.addHandler(
    type(
        "",
        (logging.StreamHandler,),
        {
            "emit": lambda self, *args, **kwargs: (
                logging.StreamHandler.emit(self, *args, **kwargs)
                if not os.environ.get(ENV_VARIABLE_DISABLE_JUMPSTART_LOGGING)
                else None
            )
        },
    )()
)

def _load_region_config(filepath: str) -> Set[JumpStartLaunchedRegionInfo]:
    """Load the JumpStart region config from a JSON file."""
    debug_msg = f"Loading JumpStart region config from '{filepath}'."
    JUMPSTART_LOGGER.debug(debug_msg)
    try:
        with open(filepath) as f:
            config = json.load(f)

        return {
            JumpStartLaunchedRegionInfo(
                region_name=region,
                content_bucket=data["content_bucket"],
                gated_content_bucket=data.get("gated_content_bucket"),
                neo_content_bucket=data.get("neo_content_bucket"),
            )
            for region, data in config.items()
        }
    except Exception:  # pylint: disable=W0703
        JUMPSTART_LOGGER.error("Unable to load JumpStart region config.", exc_info=True)
        return set()

_CURRENT_FILE_DIRECTORY_PATH = os.path.dirname(os.path.realpath(__file__))
REGION_CONFIG_JSON_FILENAME = "region_config.json"
REGION_CONFIG_JSON_FILEPATH = os.path.join(
    _CURRENT_FILE_DIRECTORY_PATH, REGION_CONFIG_JSON_FILENAME
)

JUMPSTART_LAUNCHED_REGIONS: Set[JumpStartLaunchedRegionInfo] = _load_region_config(
    REGION_CONFIG_JSON_FILEPATH
)

JUMPSTART_REGION_NAME_SET = {region.region_name for region in JUMPSTART_LAUNCHED_REGIONS}

JUMPSTART_DEFAULT_REGION_NAME = boto3.session.Session().region_name or "us-west-2"

try:
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION = Session(
        boto3.Session(region_name=JUMPSTART_DEFAULT_REGION_NAME)
    )
except Exception as e:  # pylint: disable=W0703
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION = None
    JUMPSTART_LOGGER.warning(
        "Unable to create default JumpStart SageMaker Session due to the following error: %s.",
        str(e),
    )

MODEL_ID_LIST_WEB_URL = "https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html"

SUPPORTED_JUMPSTART_SCOPES = set(scope.value for scope in JumpStartScriptScope)