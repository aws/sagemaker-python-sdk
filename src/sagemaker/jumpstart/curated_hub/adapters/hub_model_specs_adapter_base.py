
from sagemaker.jumpstart.types import (
    JumpStartModelHeader,
    JumpStartModelSpecs,
    JumpStartVersionedModelId,
)


class HubModelSpecsAdapterBase:
    def __init__(self):
        return
    
    def convert(self, model_specs: JumpStartModelSpecs) -> str:
        """Load in the file for extracting text."""
        pass

