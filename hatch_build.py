from __future__ import absolute_import

import os
import sys

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        metadata["optional-dependencies"] = get_optional_dependencies(self.root)


def get_optional_dependencies(root):

    def read_feature_deps(feature):
        req_file = os.path.join(root, "requirements", "extras", f"{feature}_requirements.txt")
        with open(req_file, encoding="utf-8") as f:
            return list(filter(lambda d: not d.startswith("#"), f.read().splitlines()))

    optional_dependencies = {"all": []}

    for feature in ("feature-processor", "huggingface", "local", "scipy", "sagemaker-mlflow"):
        dependencies = read_feature_deps(feature)
        optional_dependencies[feature] = dependencies
        optional_dependencies["all"].extend(dependencies)

    # Test dependencies come last because we don't want them in `all`
    optional_dependencies["test"] = read_feature_deps("test")
    optional_dependencies["test"].extend(optional_dependencies["all"])

    # remove torch and torchvision if python version is not 3.10/3.11
    if sys.version_info.minor not in (10, 11):
        optional_dependencies["test"] = list(
            filter(
                lambda d: not d.startswith(
                    ("sentencepiece", "transformers", "torch", "torchvision")
                ),
                optional_dependencies["test"],
            )
        )

    return optional_dependencies
