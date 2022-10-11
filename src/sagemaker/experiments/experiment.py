# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Contains the SageMaker Experiment class."""
from __future__ import absolute_import

from sagemaker.apiutils import _base_types


class _Experiment(_base_types.Record):
    """An Amazon SageMaker experiment, which is a collection of related trials.

    New experiments are created by calling `experiments.experiment._Experiment.create`.
    Existing experiments can be reloaded by calling `experiments.experiment._Experiment.load`.

    Attributes:
        experiment_name (str): The name of the experiment. The name must be unique
            within an account.
        display_name (str): Name of the experiment that will appear in UI,
            such as SageMaker Studio.
        description (str): A description of the experiment.
        tags (List[Dict[str, str]]): A list of tags to associate with the experiment.
    """

    experiment_name = None
    display_name = None
    description = None
    tags = None

    _boto_create_method = "create_experiment"
    _boto_load_method = "describe_experiment"
    _boto_update_method = "update_experiment"
    _boto_delete_method = "delete_experiment"

    _boto_update_members = ["experiment_name", "description", "display_name"]
    _boto_delete_members = ["experiment_name"]

    def save(self):
        """Save the state of this Experiment to SageMaker.

        Returns:
            dict: Update experiment API response.
        """
        return self._invoke_api(self._boto_update_method, self._boto_update_members)

    def delete(self):
        """Delete this Experiment from SageMaker.

        Deleting an Experiment does not delete associated Trials and their Trial Components.
        It requires that each Trial in the Experiment is first deleted.

        Returns:
            dict: Delete experiment API response.
        """
        return self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    @classmethod
    def load(cls, experiment_name, sagemaker_session=None):
        """Load an existing experiment and return an `_Experiment` object representing it.

        Args:
            experiment_name: (str): Name of the experiment
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            experiments.experiment._Experiment: A SageMaker `_Experiment` object
        """
        return cls._construct(
            cls._boto_load_method,
            experiment_name=experiment_name,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def create(
        cls,
        experiment_name,
        display_name=None,
        description=None,
        tags=None,
        sagemaker_session=None,
    ):
        """Create a new experiment in SageMaker and return an `_Experiment` object.

        Args:
            experiment_name: (str): Name of the experiment. Must be unique. Required.
            display_name: (str): Name of the experiment that will appear in UI,
                such as SageMaker Studio (default: None).
            description: (str): Description of the experiment (default: None).
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
            tags (List[Dict[str, str]]): A list of tags to associate with the experiment
                (default: None).

        Returns:
            experiments.experiment._Experiment: A SageMaker `_Experiment` object
        """
        return cls._construct(
            cls._boto_create_method,
            experiment_name=experiment_name,
            display_name=display_name,
            description=description,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def _load_or_create(
        cls,
        experiment_name,
        display_name=None,
        description=None,
        tags=None,
        sagemaker_session=None,
    ):
        """Load an experiment by name and create a new one if it does not exist.

        Args:
            experiment_name: (str): Name of the experiment. Must be unique. Required.
            display_name: (str): Name of the experiment that will appear in UI,
                such as SageMaker Studio (default: None). This is used only when the
                given `experiment_name` does not exist and a new experiment has to be created.
            description: (str): Description of the experiment (default: None).
                This is used only when the given `experiment_name` does not exist and
                a new experiment has to be created.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
            tags (List[Dict[str, str]]): A list of tags to associate with the experiment
                (default: None). This is used only when the given `experiment_name` does not
                exist and a new experiment has to be created.

        Returns:
            experiments.experiment._Experiment: A SageMaker `_Experiment` object
        """
        sagemaker_client = sagemaker_session.sagemaker_client
        try:
            experiment = _Experiment.load(experiment_name, sagemaker_session)
        except sagemaker_client.exceptions.ResourceNotFound:
            experiment = _Experiment.create(
                experiment_name=experiment_name,
                display_name=display_name,
                description=description,
                tags=tags,
                sagemaker_session=sagemaker_session,
            )
        return experiment
