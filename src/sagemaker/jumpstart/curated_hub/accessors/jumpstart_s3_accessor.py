from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectReference,
)


class JumpstartS3Accessor:
    def get_bucket(self) -> str:
        raise Exception("Not implemented")

    def get_inference_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        raise Exception("Not implemented")

    def get_training_artifact_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        raise Exception("Not implemented")

    def get_inference_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        raise Exception("Not implemented")

    def get_training_script_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        raise Exception("Not implemented")

    def get_default_training_dataset_s3_reference(
        self, model_specs: JumpStartModelSpecs
    ) -> S3ObjectReference:
        raise Exception("Not implemented")

    def get_demo_notebook_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        raise Exception("Not implemented")

    def get_markdown_s3_reference(self, model_specs: JumpStartModelSpecs) -> S3ObjectReference:
        raise Exception("Not implemented")
