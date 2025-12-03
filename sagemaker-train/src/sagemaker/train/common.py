from typing import Dict, Any
from enum import Enum
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

JOB_TYPE = "FineTuning"

class TrainingType(Enum):
    """Training types for fine-tuning."""
    LORA = "LORA"
    FULL = "FULL"


class CustomizationTechnique(Enum):
    """Customization techniques for fine-tuning."""
    SFT = "SFT"
    RLVR = "RLVR"
    RLAIF = "RLAIF"
    DPO = "DPO"


class FineTuningOptions:
    """Dynamic class for fine-tuning options with validation."""
    
    def __init__(self, options_dict: Dict[str, Any]):
        self._specs = options_dict.copy()
        self._initialized = False
        # Extract default values and set as attributes (no validation during init)
        for key, spec in options_dict.items():
            default_value = spec.get('default') if isinstance(spec, dict) else spec
            super().__setattr__(key, default_value)
        self._initialized = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary for hyperparameters with string values."""
        return {k: str(getattr(self, k)) for k in self._specs.keys()}
    
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif hasattr(self, '_specs') and name in self._specs:
            # Only validate if initialized (user is setting values)
            if getattr(self, '_initialized', False):
                spec = self._specs[name]
                if isinstance(spec, dict):
                    self._validate_value(name, value, spec)
            super().__setattr__(name, value)
        elif hasattr(self, '_specs'):
            raise AttributeError(f"'{name}' is not a valid fine-tuning option. Valid options: {list(self._specs.keys())}")
        else:
            super().__setattr__(name, value)
    
    def _validate_value(self, name: str, value: Any, spec: Dict[str, Any]):
        """Validate value against parameter specification."""
        # Type validation
        expected_type = spec.get('type')
        if expected_type == 'float' and not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {type(value).__name__}")
        elif expected_type == 'integer' and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
        elif expected_type == 'string' and not isinstance(value, str):
            raise ValueError(f"{name} must be a string, got {type(value).__name__}")
        
        # Range validation
        if 'min' in spec and value < spec['min']:
            raise ValueError(f"{name} must be >= {spec['min']}, got {value}")
        if 'max' in spec and value > spec['max']:
            raise ValueError(f"{name} must be <= {spec['max']}, got {value}")
        
        # Enum validation
        if 'enum' in spec and value not in spec['enum']:
            raise ValueError(f"{name} must be one of {spec['enum']}, got {value}")
    
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="FineTuningOptions.get_info")
    def get_info(self, param_name: str = None):
        """Display parameter information in a user-friendly format."""
        if param_name:
            if param_name not in self._specs:
                raise ValueError(f"Parameter '{param_name}' not found. Available: {list(self._specs.keys())}")
            params_to_show = {param_name: self._specs[param_name]}
        else:
            params_to_show = self._specs
        
        for name, spec in params_to_show.items():
            if isinstance(spec, dict):
                print(f"\n{name}:")
                print(f"  Current value: {getattr(self, name)}")
                print(f"  Type: {spec.get('type', 'unknown')}")
                print(f"  Default: {spec.get('default', 'N/A')}")
                if 'min' in spec and 'max' in spec:
                    print(f"  Range: {spec['min']} - {spec['max']}")
                elif 'min' in spec:
                    print(f"  Min: {spec['min']}")
                elif 'max' in spec:
                    print(f"  Max: {spec['max']}")
                if 'enum' in spec:
                    print(f"  Valid options: {spec['enum']}")
                if spec.get('required'):
                    print(f"  Required: Yes")
            else:
                print(f"\n{name}: {getattr(self, name)}")
