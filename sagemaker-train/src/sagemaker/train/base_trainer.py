from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Tag, Networking, InputData, Channel
from sagemaker.core.shapes import shapes
from sagemaker.core.resources import TrainingJob


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

    @abstractmethod
    def train(self, input_data_config: List[InputData], wait: bool = True, logs: bool = True):
        """Common training method that calls the specific implementation."""
        pass
