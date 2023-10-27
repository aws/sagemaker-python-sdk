"""Detects the image to deploy model"""
from __future__ import absolute_import
from typing import Tuple, List
import platform
import logging
from sagemaker import image_uris

logger = logging.getLogger(__name__)

_VERSION_DETECTION_ERROR = "Framework version was unable to be found for the provided %s model.\
    The latest supported framework version will be used."
_CASTING_WARNING = "Could not find the framework version %s in supported framework versions\
    for the DLC. Mapping to the nearest latest minor version.\
        The available compatible versions are as follows %s"


def auto_detect_container(model, region: str, instance_type: str) -> str:
    """Auto detect the container off of model and instance type"""

    logger.info("Autodetecting image since image_uri was not provided in ModelBuilder()")

    if not instance_type:
        raise ValueError(
            "Instance type is not specified.\
                Unable to detect if the container needs to be GPU or CPU."
        )

    logger.warning(
        "Auto detection is only supported for single models DLCs with a framework backend."
    )

    model_base = _get_model_base(model)

    py_tuple = platform.python_version_tuple()

    fw, fw_version = _detect_framework_and_version(str(model_base))
    logger.info("Autodetected framework is %s", fw)
    logger.info("Autodetected framework version is %s", fw_version)

    try:
        dlc = image_uris.retrieve(
            framework=fw,
            region=region,
            version=fw_version if not fw_version else _cast_to_compatible_version(fw, fw_version),
            image_scope="inference",
            py_version=f"py{py_tuple[0]}{py_tuple[1]}",
            instance_type=instance_type,
        )
    except ValueError as e:
        logger.exception(e)
        raise ValueError(
            "Unable to auto detect a DLC for framework %s, framework version %s.\
                Please manually provide image_uri to ModelBuilder()"
            % (fw, fw_version)
        )

    logger.info("Auto detected %s. Proceeding with the the deployment.", dlc)

    return dlc


def _cast_to_compatible_version(framework: str, fw_version: str) -> str:
    """Placeholder docstring"""
    config = image_uris._config_for_framework_and_scope(framework, "inference", None)
    available_versions = list(config["versions"].keys())
    earliest_upcast_version = None
    latest_downcast_version = None

    split_vs = list(map(int, fw_version.split(".")))

    for version in available_versions:
        up_cast, down_cast, found = _find_compatible_vs(split_vs, version)
        if found:
            logger.info("Framework version %s found in available versions", version)
            return found
        if up_cast:
            if not earliest_upcast_version or _later_version(earliest_upcast_version, up_cast):
                logger.info("set to %s", up_cast)
                earliest_upcast_version = up_cast
        if down_cast:
            if not latest_downcast_version or not _later_version(
                latest_downcast_version, down_cast
            ):
                latest_downcast_version = down_cast

    if earliest_upcast_version:
        logger.warning(_CASTING_WARNING, fw_version, available_versions)
        return earliest_upcast_version

    if latest_downcast_version:
        logger.warning(_CASTING_WARNING, fw_version, available_versions)
        return latest_downcast_version

    raise ValueError(
        "Auto detection could not find a compatible DLC version mapped to framework %s,\
            framework version %s. The available compatible versions\
                are as follows %s."
        % (framework, fw_version, available_versions)
    )


def _later_version(current: str, found: str) -> bool:
    """Placeholder docstring"""
    split_current = current.split(".")
    split_minor_current = split_current[1].split("-")
    split_found = found.split(".")
    split_minor_found = split_found[1].split("-")

    major_current = int(split_current[0])
    major_found = int(split_found[0])

    # major versions should always be equal. but check for safety
    if major_current == major_found:
        mini_current = (
            int(split_current[2]) if len(split_minor_current) == 1 else int(split_minor_current[1])
        )
        mini_found = (
            int(split_found[2]) if len(split_minor_found) == 1 else int(split_minor_found[1])
        )
        return mini_current > mini_found

    return major_current > major_found


def _find_compatible_vs(split_vs: List[int], supported_vs: str) -> Tuple[str, str, str]:
    """Placeholder docstring"""
    earliest_upcast_version = None
    latest_downcast_version = None
    found_version = None

    split_supported_vs = supported_vs.split(".")

    # if same major version
    if split_vs[0] == int(split_supported_vs[0]):
        # if no minor or mini version
        if len(split_supported_vs) == 1:
            if len(split_vs) == 1:
                return (None, None, supported_vs)
            return (None, None, None)

        # the minor and mini could be joined as such 1.2-1
        split_supported_minor = split_supported_vs[1].split("-")
        converted_supported_minor = int(split_supported_minor[0])

        # if same minor version
        if split_vs[1] == converted_supported_minor:
            mini = (
                int(split_supported_vs[2])
                if len(split_supported_minor) == 1
                else int(split_supported_minor[1])
            )
            if split_vs[2] == mini:
                found_version = supported_vs
            elif split_vs[2] < mini:
                earliest_upcast_version = supported_vs
            else:
                latest_downcast_version = supported_vs
        elif split_vs[1] < converted_supported_minor:
            earliest_upcast_version = supported_vs

    return (earliest_upcast_version, latest_downcast_version, found_version)


def _detect_framework_and_version(model_base: str) -> Tuple[str, str]:
    """Parse fw based off the base model object and get version if possible"""
    fw = ""
    vs = ""
    if "torch" in model_base:
        fw = "pytorch"
        try:
            import torch

            vs = torch.__version__.split("+")[0]
        except ImportError:
            logger.warning(_VERSION_DETECTION_ERROR, fw)
    elif "xgb" in model_base:
        fw = "xgboost"
        try:
            import xgboost

            vs = xgboost.__version__
        except ImportError:
            logger.warning(_VERSION_DETECTION_ERROR, fw)
    elif "keras" in model_base or "tensorflow" in model_base:
        fw = "tensorflow"
        try:
            import tensorflow

            vs = tensorflow.__version__
        except ImportError:
            logger.warning(_VERSION_DETECTION_ERROR, fw)

    else:
        raise Exception(
            "Unable to determine required container for model base %s.\
                Please specify container in model builder"
            % model_base
        )

    return (fw, vs)


def _get_model_base(model: object) -> type:
    """Placeholder docstring"""
    model_base = model.__class__.__base__

    # for cases such as xgb.Booster where there is no inherited base class
    if object == model_base:
        model_base = model.__class__

    return model_base
