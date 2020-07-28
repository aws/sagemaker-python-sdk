# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A Python script to upgrade framework versions"""
from __future__ import absolute_import

import argparse
import json


def _read_json_to_dict(filename):
    """Read a json file into a Python dictionary

    Args:
        filename (str): Name of the json file.

    Returns:
        dict: A Python Dictionary
    """
    with open(filename, "r") as f:
        content = json.load(f)
        return content


def _write_dict_to_json(filename, content):
    """Write a Python dictionary to a json file.

    Args:
        filename (str): Name of the target json file.
        content (dict): Dictionary to be written to the json file.
    """
    with open(filename, "w") as f:
        json.dump(content, f, indent=4)


def _read_framework_region_regestries(framework):  # pylint: disable=W0621
    """Read framework's corresponding region registries to a Python dictionary.

    Args:
        framework (str): Name of the framework (e.g. tensorflow, pytorch, mxnet)

    Returns:
        dict: Dictionary of region to registry mapping.
    """
    with open("framework-region-registry.json", "r") as f:
        frameworks = json.load(f)
        return frameworks[framework]


def add_dlc_framework_version(
    content,  # pylint: disable=W0621
    framework,  # pylint: disable=W0621
    short_version,  # pylint: disable=W0621
    full_version,  # pylint: disable=W0621
    image_type,  # pylint: disable=W0621
    processors,  # pylint: disable=W0621
    py_versions,  # pylint: disable=W0621
    registries,  # pylint: disable=W0621
):
    """Update framework image uri json file with new version information.

    Args:
        content (dict): Existing framework image uri information read from "<framework>.json" file.
        framework (str): Framework name (e.g. tensorflow, pytorch, mxnet)
        short_version (str): Abbreviated framework version (e.g. 1.0, 1.5)
        full_version (str): Complete framework version (e.g. 1.0.0, 1.5.2)
        image_type (str): Framework image type, it could be "training", "inference" or "eia"
        processors (list): Supported processors (e.g. ["cpu", "gpu"])
        py_versions (list): Supported Python versions (e.g. ["py3", "py37"])
    """
    for processor in processors:
        if processor not in content[image_type]["processors"]:
            content[image_type]["processors"].append(processor)
    content[image_type]["version_aliases"][short_version] = full_version
    repo = "{}-{}{}"
    if image_type == "eia":
        repo = repo.format(framework, "inference", "-eia")
    else:
        repo = repo.format(framework, image_type, "")
    add_version = {
        "registries": registries,
        "repository": repo,
        "py_versions": py_versions,
    }
    content[image_type]["versions"][full_version] = add_version


def update_json(
    framework, short_version, full_version, image_type, processors, py_versions
):  # pylint: disable=W0621
    """Read framework image uri information from json file to a dictionary, update it with new
    framework version information, then write the dictionary back to json file.

    Args:
        framework (str): Framework name (e.g. tensorflow, pytorch, mxnet)
        short_version (str): Abbreviated framework version (e.g. 1.0, 1.5)
        full_version (str): Complete framework version (e.g. 1.0.0, 1.5.2)
        image_type (str): Framework image type, it could be "training", "inference" or "eia"
        processors (str): Supported processors (e.g. "cpu,gpu")
        py_versions (str): Supported Python versions (e.g. "py3,py37")
     """
    filename = "{}.json".format(framework)
    content = _read_json_to_dict(filename)
    py_versions = py_versions.split(",")
    processors = processors.split(",")
    registries = _read_framework_region_regestries(framework)["registries"]
    add_dlc_framework_version(
        content,
        framework,
        short_version,
        full_version,
        image_type,
        processors,
        py_versions,
        registries,
    )
    _write_dict_to_json(filename, content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework upgrade tool.")
    parser.add_argument("--framework", help="Name of the framework (e.g. tensorflow, mxnet, etc.)")
    parser.add_argument("--short-version", help="Abbreviated framework version (e.g. 2.0)")
    parser.add_argument("--full-version", help="Full framework version (e.g. 2.0.1)")
    parser.add_argument("--image-type", help="Framework image type (e.g. training, inference, eia)")
    parser.add_argument("--processors", help="Suppoted processors (e.g. cpu, gpu)")
    parser.add_argument("--py-versions", help="Supported Python versions (e.g. py3,py37)")

    args = parser.parse_args()
    framework = args.framework
    short_version = args.short_version
    full_version = args.full_version
    image_type = args.image_type
    processors = args.processors
    py_versions = args.py_versions

    update_json(framework, short_version, full_version, image_type, processors, py_versions)
