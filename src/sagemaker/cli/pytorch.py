"""Placeholder docstring"""

from __future__ import absolute_import

from sagemaker.cli.common import HostCommand, TrainCommand


def train(args):
    """

    Args:
        args:

    Returns:

    """
    PytorchTrainCommand(args).start()


def host(args):
    """
    Args:
        args:
    """
    PytorchHostCommand(args).start()


class PytorchTrainCommand(TrainCommand):
    """Placeholder docstring"""

    def create_estimator(self):
        from sagemaker.pytorch import PyTorch

        return PyTorch(
            base_job_name=self.job_name,
            train_instance_count=self.instance_count,
            train_instance_type=self.instance_type,
            entry_point=self.script,
            hyperparameters=self.hyperparameters,
            py_version=self.python,
        )


class PytorchHostCommand(HostCommand):
    """Placeholder docstring"""

    def create_model(self, model_url):
        """
        Args:
            model_url:
        """
        from sagemaker.pytorch.model import PyTorchModel

        return PyTorchModel(
            model_data=model_url,
            role=self.role_name,
            entry_point=self.script,
            py_version=self.python,
            name=self.endpoint_name,
            env=self.environment,
        )
