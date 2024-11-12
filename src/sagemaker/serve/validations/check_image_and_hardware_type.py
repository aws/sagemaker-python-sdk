"""Validate if image_uri is compatible with instance_type"""

from __future__ import absolute_import
import logging

from sagemaker.serve.utils.types import ModelServer, HardwareType

logger = logging.getLogger(__name__)

GPU_INSTANCE_FAMILIES = {
    "ml.g4dn",
    "ml.g5",
    "ml.p3",
    "ml.p3dn",
    "ml.p4",
    "ml.p4d",
    "ml.p4de",
}


INF1_INSTANCE_FAMILIES = {"ml.inf1"}
INF2_INSTANCE_FAMILIES = {"ml.inf2"}

GRAVITON_INSTANCE_FAMILIES = {
    "ml.c7g",
    "ml.m6g",
    "ml.m6gd",
    "ml.c6g",
    "ml.c6gd",
    "ml.c6gn",
    "ml.r6g",
    "ml.r6gd",
}


def validate_image_uri_and_hardware(image_uri: str, instance_type: str, model_server: ModelServer):
    """Placeholder docstring"""
    if "xgboost" in image_uri:
        # xgboost container does not care about hardware type
        # hence skipping validation
        return

    hardware_type_of_instance = detect_hardware_type_of_instance(instance_type=instance_type)
    if model_server == ModelServer.TORCHSERVE:
        hardware_type_of_image = detect_torchserve_image_hardware_type(image_uri=image_uri)
    elif model_server == ModelServer.TRITON:
        hardware_type_of_image = detect_triton_image_hardware_type(image_uri=image_uri)
    else:
        logger.info("Skipping validation of image_uri and instance_type compatibility.")
        return

    if hardware_type_of_image != hardware_type_of_instance:
        logger.warning(
            (
                "Detected potential incompatibility of image_uri and instance_type. "
                "Your image_uri %s is for %s but instance_type %s is a %s instance type. "
                "This might lead to sub-optimal performance or deployment failure. "
                "Please provide the same `instance_type` to "
                "ModelBuilder as well as in model.deploy(). "
                "Alternatively, directly provide `image_uri` to ModelBuilder. "
            ),
            image_uri,
            hardware_type_of_image,
            instance_type,
            hardware_type_of_instance,
        )

    return


def detect_hardware_type_of_instance(instance_type: str) -> HardwareType:
    """Placeholder docstring"""
    instance_family = instance_type.rsplit(".", 1)[0]
    if instance_family in GPU_INSTANCE_FAMILIES:
        return HardwareType.GPU
    if instance_family in INF1_INSTANCE_FAMILIES:
        return HardwareType.INFERENTIA_1
    if instance_family in INF2_INSTANCE_FAMILIES:
        return HardwareType.INFERENTIA_2
    if instance_family in GRAVITON_INSTANCE_FAMILIES:
        return HardwareType.GRAVITON
    return HardwareType.CPU


def detect_triton_image_hardware_type(image_uri: str) -> HardwareType:
    """Placeholder docstring"""
    return HardwareType.CPU if "cpu" in image_uri else HardwareType.GPU


def detect_torchserve_image_hardware_type(image_uri: str) -> HardwareType:
    """Placeholder docstring"""
    if "neuronx" in image_uri:
        return HardwareType.INFERENTIA_2
    if "neuron" in image_uri:
        return HardwareType.INFERENTIA_1
    if "graviton" in image_uri:
        return HardwareType.GRAVITON
    if "cpu" in image_uri:
        return HardwareType.CPU

    return HardwareType.GPU
