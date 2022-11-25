import sys

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        optional_dependencies = metadata["optional-dependencies"]
        optional_dependencies["all"] = [
            dependency
            for optional_dependencies in optional_dependencies.values()
            for dependency in optional_dependencies
        ]
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
