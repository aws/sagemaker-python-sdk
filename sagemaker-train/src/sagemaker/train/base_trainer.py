import copy
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Tag, Networking, InputData, Channel
from sagemaker.core.shapes import shapes
from sagemaker.core.resources import TrainingJob
from sagemaker.train.common_utils.recipe_utils import _is_nova_model, resolve_recipe
from sagemaker.train.recipe_resolver import flatten_resolved_recipe


class BaseTrainer(ABC):
    """Abstract base class for all SageMaker training workflows.

    This class provides the common interface and shared functionality for all trainer implementations
    including SFT, DPO, RLVR, and RLAIF trainers. It defines the standard parameters and abstract
    methods that concrete trainer classes must implement.

    Parameters:
        sagemaker_session (Optional[Session]):
            The SageMaker session for managing API calls and resources.
            If not specified, a default session will be created.
        role (Optional[str]):
            The IAM role ARN for the training job execution.
            If not specified, the default SageMaker execution role will be used.
        base_job_name (Optional[str]):
            The base name for training jobs. A unique suffix will be appended.
            If not specified, a default name will be generated based on the trainer type.
        tags (Optional[List[Tag]]):
            List of tags to apply to the training job for resource management and billing.
        hyperparameters (Optional[Dict[str, Any]]):
            Dictionary of hyperparameters for the training job.
            Trainer-specific defaults will be applied if not specified.
        output_data_config (Optional[shapes.OutputDataConfig]):
            Configuration for training job outputs including S3 paths and encryption.
            If not specified, default output configuration will be used.
        input_data_config (Optional[List[Union[Channel, InputData]]]):
            List of input data channels for the training job.
            Can include training and validation datasets.
        environment (Optional[Dict[str, str]]):
            Environment variables to set in the training container.
    """
    
    # Class-level attributes with default values
    sagemaker_session: Optional[Session] = None
    role: Optional[str] = None
    base_job_name: Optional[str] = None
    tags: Optional[List[Tag]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    output_data_config: Optional[shapes.OutputDataConfig] = None
    input_data_config: Optional[List[Union[Channel, InputData]]] = None
    environment: Optional[Dict[str, str]] = None
    latest_training_job: Optional[TrainingJob] = None

    def __init__(
        self,
        sagemaker_session: Optional[Session] = None,
        role: Optional[str] = None,
        base_job_name: Optional[str] = None,
        tags: Optional[List[Tag]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_data_config: Optional[shapes.OutputDataConfig] = None,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        environment: Optional[Dict[str, str]] = None,
    ):
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.base_job_name = base_job_name
        self.tags = tags
        self.hyperparameters = hyperparameters or {}
        self.output_data_config = output_data_config
        self.input_data_config = input_data_config
        self.environment = environment or {}

    def _is_nova_model_for_telemetry(self) -> bool:
        """Check if the model is a Nova model for telemetry tracking."""
        model_name = getattr(self, "_model_name", None)
        return _is_nova_model(model_name) if model_name else False

    def get_resolved_recipe(self) -> Dict[str, Any]:
        """Return the fully resolved recipe configuration.

        Shows the final merged result of base defaults + user recipe + overrides
        after interpolation resolution and validation. Callable before or after train().

        Returns:
            dict: Deep copy of the resolved recipe configuration.

        Raises:
            ValueError: If no recipe or overrides were provided at construction time.
        """
        if getattr(self, '_resolved_recipe_cache', None) is not None:
            return copy.deepcopy(self._resolved_recipe_cache)

        recipe_path = getattr(self, '_recipe_path', None)
        overrides = getattr(self, '_overrides', None)

        if not recipe_path and not overrides:
            raise ValueError(
                "get_resolved_recipe() requires a 'recipe' or 'overrides' to be provided "
                "at construction time."
            )

        override_spec = {}
        full_recipe_template = None
        if hasattr(self, 'hyperparameters') and hasattr(self.hyperparameters, '_specs'):
            override_spec = self.hyperparameters._specs
        frt = getattr(self.hyperparameters, '_full_recipe_template', None) if hasattr(self, 'hyperparameters') else None
        if isinstance(frt, dict):
            full_recipe_template = frt

        resolved = resolve_recipe(
            recipe_path=recipe_path,
            overrides=overrides,
            override_spec=override_spec,
            template_section="training_config",
            protected_keys={"model_type", "model_name_or_path"},
            full_recipe_template=full_recipe_template,
        )

        self._resolved_recipe_cache = resolved
        return copy.deepcopy(resolved)

    def _apply_recipe_to_hyperparameters(self, final_hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resolved recipe values to final_hyperparameters dict.

        If recipe/overrides were provided, merges resolved values into the
        hyperparameters dict. All leaf values from the resolved recipe are
        applied — including keys not in the Hub spec subset — enabling
        power users to override any parameter in the full recipe.
        Values are converted to strings (matching the SageMaker API
        expectation for hyperparameter values).

        Args:
            final_hyperparameters: The hyperparameters dict from to_dict().

        Returns:
            The updated hyperparameters dict with recipe values applied.
        """
        if not getattr(self, '_recipe_path', None) and not getattr(self, '_overrides', None):
            return final_hyperparameters

        resolved = self.get_resolved_recipe()
        flat = flatten_resolved_recipe(resolved)
        for k, v in flat.items():
            if v is not None:
                final_hyperparameters[k] = str(v) if not isinstance(v, str) else v

        return final_hyperparameters

    @abstractmethod
    def train(self, input_data_config: List[InputData], wait: bool = True, logs: bool = True, wait_timeout: Optional[int] = None):
        """Common training method that calls the specific implementation."""
        pass
