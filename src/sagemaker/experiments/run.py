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
"""Contains the SageMaker Experiment Run class."""
from __future__ import absolute_import

import datetime
import logging
import os
from math import isnan, isinf
from numbers import Number
from os.path import join
from typing import Optional, List, Dict, TYPE_CHECKING

import dateutil

from sagemaker.apiutils import _utils
from sagemaker.experiments import _api_types
from sagemaker.experiments._api_types import TrialComponentArtifact, _TrialComponentStatusType
from sagemaker.experiments._helper import (
    _ArtifactUploader,
    _LineageArtifactTracker,
)
from sagemaker.experiments._environment import _RunEnvironment, _EnvironmentType
from sagemaker.experiments._run_context import _RunContext
from sagemaker.experiments.experiment import _Experiment
from sagemaker.experiments.metrics import _MetricsManager
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent

from sagemaker.utils import (
    get_module,
    retry_with_backoff,
    unique_name_from_base,
)

from sagemaker.experiments._utils import (
    guess_media_type,
    resolve_artifact_name,
    verify_length_of_true_and_predicted,
    validate_invoked_inside_run_context,
)

if TYPE_CHECKING:
    from sagemaker import Session

logger = logging.getLogger(__name__)

RUN_NAME_BASE = "Sagemaker-Run"
TRIAL_NAME_TEMPLATE = "Default-Run-Group-{}"
MAX_RUN_TC_ARTIFACTS_LEN = 30
MAX_NAME_LEN_IN_BACKEND = 120
UNKNOWN_NAME = "unknown"
EXP_NAME_BASE = "Sagemaker-Experiment"
EXPERIMENT_NAME = "ExperimentName"
TRIAL_NAME = "TrialName"
RUN_NAME = "RunName"
TRIAL_COMPONENT_NAME = "TrialComponentName"
DELIMITER = "-"
RUN_TC_TAG_KEY = "sagemaker:trial-component-source"
RUN_TC_TAG_VALUE = "run"


class Run(object):
    """A collection of parameters, metrics, and artifacts to create a ML model."""

    def __init__(
        self,
        experiment_name,
        run_name,
        trial_component,
        sagemaker_session,
        run_group_name=None,
        experiment=None,
        trial=None,
    ):
        """Construct a `Run` instance"""
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_group_name = run_group_name
        self._experiment = experiment
        self._trial = trial
        self._trial_component = trial_component

        self._artifact_uploader = _ArtifactUploader(
            trial_component_name=self._trial_component.trial_component_name,
            sagemaker_session=sagemaker_session,
        )
        self._lineage_artifact_tracker = _LineageArtifactTracker(
            trial_component_arn=self._trial_component.trial_component_arn,
            sagemaker_session=sagemaker_session,
        )
        self._metrics_manager = _MetricsManager(
            resource_arn=self._trial_component.trial_component_arn,
            sagemaker_session=sagemaker_session,
        )
        self._inside_init_context = False
        self._inside_load_context = False
        self._in_load = False

    @classmethod
    def init(
        cls,
        experiment_name: str,
        run_name: Optional[str] = None,
        experiment_display_name: Optional[str] = None,
        run_display_name: Optional[str] = None,
        tags: Optional[List[Dict[str, str]]] = None,
        sagemaker_session: Optional["Session"] = None,
    ):
        """The primary method used to init a run class and setup experiment tracking.

        Args:
            experiment_name (str): The name of the experiment. The name must be unique
                within an account.
            run_name (str): The name of the run. If it is not specified, one is auto generated.
            experiment_display_name (str): Name of the experiment that will appear in UI,
                such as SageMaker Studio. (default: None). This display name is used in
                a create experiment call. If an experiment with the specified name already exists,
                this display name won't take effect.
            run_display_name (str): The display name of the run used in UI (default: None).
                This display name is used in a create run call. If a run with the
                specified name already exists, this display name won't take effect.
            tags (List[Dict[str, str]]): A list of tags to be used for all create calls,
                e.g. to create an experiment, a run group, etc. (default: None).
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or _utils.default_session()
        run_name = run_name or unique_name_from_base(RUN_NAME_BASE)
        trial_component_name = Run._generate_trial_component_name(
            run_name=run_name, experiment_name=experiment_name
        )
        run_group_name = Run._generate_trial_name(experiment_name)

        experiment = _Experiment._load_or_create(
            experiment_name=experiment_name,
            display_name=experiment_display_name,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

        trial = _Trial._load_or_create(
            experiment_name=experiment_name,
            trial_name=run_group_name,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

        run_tc = _TrialComponent._load_or_create(
            trial_component_name=trial_component_name,
            display_name=run_display_name,
            tags=Run._append_run_tc_label_to_tags(tags),
            sagemaker_session=sagemaker_session,
        )

        trial.add_trial_component(run_tc)

        return cls(
            experiment_name=experiment_name,
            run_group_name=run_group_name,
            run_name=run_name,
            experiment=experiment,
            trial=trial,
            trial_component=run_tc,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def load(
        cls,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        sagemaker_session: Optional["Session"] = None,
    ):
        """Load a Run Trial Component by the run name or from the job environment.

        Args:
            run_name (str): The name of the Run to be loaded (default: None).
                If it is None, the `RunName` in the `ExperimentConfig` of the job will be
                fetched to load the Run Trial Component.
            experiment_name (str): The name of the Experiment that the to be loaded Run
                is associated with (default: None).
                Note: the experiment_name must be supplied along with a valid run_name.
                Otherwise, it will be ignored.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or _utils.default_session()
        environment = _RunEnvironment.load()

        Run._verify_load_input_names(run_name=run_name, experiment_name=experiment_name)

        if run_name or environment:
            if run_name:
                logger.warning(
                    "run_name is explicitly supplied in Run.load, "
                    "which will be prioritized to load the Run object. "
                    "In other words, the run name in the experiment config, fetched from the "
                    "job environment or the current Run context, will be ignored."
                )
                run_tc, exp_config = Run._get_tc_and_exp_config_by_run_name(
                    run_name=run_name,
                    sagemaker_session=sagemaker_session,
                    experiment_name=experiment_name,
                )
            else:
                run_tc, exp_config = Run._get_tc_and_exp_config_from_job_env(
                    environment=environment, sagemaker_session=sagemaker_session
                )
            run_name = exp_config[RUN_NAME].replace(
                "{}{}".format(exp_config[EXPERIMENT_NAME], DELIMITER), "", 1
            )
            run_instance = cls(
                experiment_name=exp_config[EXPERIMENT_NAME],
                run_name=run_name,
                run_group_name=exp_config[TRIAL_NAME],
                trial_component=run_tc,
                sagemaker_session=sagemaker_session,
            )
        elif _RunContext.get_current_run():
            run_instance = _RunContext.get_current_run()
        else:
            raise RuntimeError(
                "Failed to load a Run object. "
                "Please make sure a Run object has been initialized already."
            )

        run_instance._in_load = True
        return run_instance

    @property
    def _experiment_config(self):
        """Get experiment config from Run attributes."""
        return {
            EXPERIMENT_NAME: self.experiment_name,
            TRIAL_NAME: self.run_group_name,
            RUN_NAME: self._trial_component.trial_component_name,
        }

    @validate_invoked_inside_run_context
    def log_parameter(self, name, value):
        """Record a single parameter value for this run trial component.

        Overwrites any previous value recorded for the specified parameter name.

        Args:
            name (str): The name of the parameter.
            value (str or numbers.Number): The value of the parameter.
        """
        if self._is_input_valid("parameter", name, value):
            self._trial_component.parameters[name] = value

    @validate_invoked_inside_run_context
    def log_parameters(self, parameters):
        """Record a collection of parameter values for this run trial component.

        Args:
            parameters (dict[str, str or numbers.Number]): The parameters to record.
        """
        filtered_parameters = {
            key: value
            for (key, value) in parameters.items()
            if self._is_input_valid("parameter", key, value)
        }
        self._trial_component.parameters.update(filtered_parameters)

    @validate_invoked_inside_run_context
    def log_metric(self, name, value, timestamp=None, step=None):
        """Record a custom scalar metric value for this run trial component.

        Note:
             1. This method is for manual custom metrics, for automatic metrics see the
             `enable_sagemaker_metrics` parameter on the `estimator` class.
             2. Metrics logged with this method will only appear in SageMaker when this method
             is called from a training job host.

        Args:
            name (str): The name of the metric.
            value (float): The value of the metric.
            timestamp (datetime.datetime): The timestamp of the metric.
                If not specified, the current UTC time will be used.
            step (int): The integer iteration number of the metric value (default: None).
        """
        if self._is_input_valid("metric", name, value):
            self._metrics_manager.log_metric(
                metric_name=name, value=value, timestamp=timestamp, step=step
            )

    @validate_invoked_inside_run_context
    def log_precision_recall(
        self,
        y_true,
        predicted_probabilities,
        positive_label=None,
        title=None,
        is_output=True,
        no_skill=None,
    ):
        """Create and log a precision recall graph artifact for Studio UI to render.

        The artifact is stored in S3 and represented as a lineage artifact
        with an association with the run trial component.

        You can view the artifact in the charts tab of the Trial Component UI.
        If your job is created by a pipeline execution you can view the artifact
        by selecting the corresponding step in the pipelines UI.
        See also `SageMaker Pipelines <https://aws.amazon.com/sagemaker/pipelines/>`_

        This method requires sklearn library.

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            predicted_probabilities (list or array): Estimated/predicted probabilities.
            positive_label (str or int): Label of the positive class (default: None).
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                trial component. Defaults to True (output artifact).
                If set to False then represented as input association.
            no_skill (int): The precision threshold under which the classifier cannot discriminate
                between the classes and would predict a random class or a constant class in
                all cases (default: None).
        """

        verify_length_of_true_and_predicted(
            true_labels=y_true,
            predicted_attrs=predicted_probabilities,
            predicted_attrs_name="predicted probabilities",
        )

        get_module("sklearn")
        from sklearn.metrics import precision_recall_curve, average_precision_score

        kwargs = {}
        if positive_label is not None:
            kwargs["pos_label"] = positive_label

        precision, recall, _ = precision_recall_curve(y_true, predicted_probabilities, **kwargs)

        kwargs["average"] = "micro"
        ap = average_precision_score(y_true, predicted_probabilities, **kwargs)

        data = {
            "type": "PrecisionRecallCurve",
            "version": 0,
            "title": title,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "averagePrecisionScore": ap,
            "noSkill": no_skill,
        }
        self._log_graph_artifact(
            artifact_name=title, data=data, graph_type="PrecisionRecallCurve", is_output=is_output
        )

    @validate_invoked_inside_run_context
    def log_roc_curve(
        self,
        y_true,
        y_score,
        title=None,
        is_output=True,
    ):
        """Create and log a receiver operating characteristic (ROC curve) artifact.

        The artifact is stored in S3 and represented as a lineage artifact
        with an association with the run trial component.

        You can view the artifact in the charts tab of the Trial Component UI.
        If your job is created by a pipeline execution you can view the artifact
        by selecting the corresponding step in the pipelines UI.
        See also `SageMaker Pipelines <https://aws.amazon.com/sagemaker/pipelines/>`_

        This method requires sklearn library.

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            y_score (list or array): Estimated/predicted probabilities.
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                trial component. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        verify_length_of_true_and_predicted(
            true_labels=y_true, predicted_attrs=y_score, predicted_attrs_name="predicted scores"
        )

        get_module("sklearn")
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_score)

        auc = auc(fpr, tpr)

        data = {
            "type": "ROCCurve",
            "version": 0,
            "title": title,
            "falsePositiveRate": fpr.tolist(),
            "truePositiveRate": tpr.tolist(),
            "areaUnderCurve": auc,
        }
        self._log_graph_artifact(
            artifact_name=title, data=data, graph_type="ROCCurve", is_output=is_output
        )

    @validate_invoked_inside_run_context
    def log_confusion_matrix(
        self,
        y_true,
        y_pred,
        title=None,
        is_output=True,
    ):
        """Create and log a confusion matrix artifact.

        The artifact is stored in S3 and represented as a lineage artifact
        with an association with the run trial component.

        You can view the artifact in the charts tab of the Trial Component UI.
        If your job is created by a pipeline execution you can view the
        artifact by selecting the corresponding step in the pipelines UI.
        See also `SageMaker Pipelines <https://aws.amazon.com/sagemaker/pipelines/>`_
        This method requires sklearn library.

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            y_pred (list or array): Predicted labels.
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                trial component. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        verify_length_of_true_and_predicted(
            true_labels=y_true, predicted_attrs=y_pred, predicted_attrs_name="predicted labels"
        )

        get_module("sklearn")
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(y_true, y_pred)

        data = {
            "type": "ConfusionMatrix",
            "version": 0,
            "title": title,
            "confusionMatrix": matrix.tolist(),
        }
        self._log_graph_artifact(
            artifact_name=title, data=data, graph_type="ConfusionMatrix", is_output=is_output
        )

    @validate_invoked_inside_run_context
    def log_output(self, name, value, media_type=None):
        """Record a single output artifact for this run trial component.

        Overwrites any previous value recorded for the specified output name.

        Args:
            name (str): The name of the output value.
            value (str): The value.
            media_type (str): The MediaType (MIME type) of the value (default: None).
        """
        self._verify_trial_component_artifacts_length(is_output=True)
        self._trial_component.output_artifacts[name] = TrialComponentArtifact(
            value, media_type=media_type
        )

    @validate_invoked_inside_run_context
    def log_input(self, name, value, media_type=None):
        """Record a single input artifact for this run trial component.

        Overwrites any previous value recorded for the specified input name.

        Args:
            name (str): The name of the input value.
            value (str): The value.
            media_type (str): The MediaType (MIME type) of the value (default: None).
        """
        self._verify_trial_component_artifacts_length(is_output=False)
        self._trial_component.input_artifacts[name] = TrialComponentArtifact(
            value, media_type=media_type
        )

    @validate_invoked_inside_run_context
    def log_artifact_file(self, file_path, name=None, media_type=None, is_output=True):
        """Upload a file to s3 and store it as an input/output artifact in this trial component.

        Args:
            file_path (str): The path of the local file to upload.
            name (str): The name of the artifact (default: None).
            media_type (str): The MediaType (MIME type) of the file.
                If not specified, this library will attempt to infer the media type
                from the file extension of `file_path`.
            is_output (bool): Determines direction of association to the
                trial component. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        self._verify_trial_component_artifacts_length(is_output)
        media_type = media_type or guess_media_type(file_path)
        name = name or resolve_artifact_name(file_path)
        s3_uri, _ = self._artifact_uploader.upload_artifact(file_path)
        if is_output:
            self._trial_component.output_artifacts[name] = TrialComponentArtifact(
                value=s3_uri, media_type=media_type
            )
        else:
            self._trial_component.input_artifacts[name] = TrialComponentArtifact(
                value=s3_uri, media_type=media_type
            )

    @validate_invoked_inside_run_context
    def log_artifact_directory(self, directory, media_type=None, is_output=True):
        """Upload files under directory to s3 and log as artifacts in this trial component.

        The file name is used as the artifact name

        Args:
            directory (str): The directory of the local files to upload.
            media_type (str): The MediaType (MIME type) of the file.
                If not specified, this library will attempt to infer the media type
                from the file extension of `file_path`.
            is_output (bool): Determines direction of association to the
                trial component. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        for dir_file in os.listdir(directory):
            file_path = join(directory, dir_file)
            artifact_name = os.path.splitext(dir_file)[0]
            self.log_artifact_file(
                file_path=file_path, name=artifact_name, media_type=media_type, is_output=is_output
            )

    @validate_invoked_inside_run_context
    def log_lineage_artifact(self, file_path, name=None, media_type=None, is_output=True):
        """Upload a file to S3 and creates a lineage Artifact associated with this trial component.

        Args:
            file_path (str): The path of the local file to upload.
            name (str): The name of the artifact (default: None).
            media_type (str): The MediaType (MIME type) of the file.
                If not specified, this library will attempt to infer the media type
                from the file extension of `file_path`.
            is_output (bool): Determines direction of association to the
                trial component. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        media_type = media_type or guess_media_type(file_path)
        name = name or resolve_artifact_name(file_path)
        s3_uri, etag = self._artifact_uploader.upload_artifact(file_path)
        if is_output:
            self._lineage_artifact_tracker.add_output_artifact(
                name=name, source_uri=s3_uri, etag=etag, artifact_type=media_type
            )
        else:
            self._lineage_artifact_tracker.add_input_artifact(
                name=name, source_uri=s3_uri, etag=etag, artifact_type=media_type
            )

    @classmethod
    def list(
        cls,
        experiment_name,
        created_before=None,
        created_after=None,
        sort_by="CreationTime",
        sort_order="Descending",
        sagemaker_session=None,
        max_results=None,
        next_token=None,
    ):
        """Return a list of `Run` objects matching the given criteria.

        Args:
            experiment_name (str): Only trial components related to the specified experiment
                are returned.
            created_before (datetime.datetime): Return trial components created before this instant
                (default: None).
            created_after (datetime.datetime): Return trial components created after this instant
                (default: None).
            sort_by (str): Which property to sort results by. One of 'Name', 'CreationTime'
                (default: 'CreationTime').
            sort_order (str): One of 'Ascending', or 'Descending' (default: 'Descending').
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
            max_results (int): maximum number of trial components to retrieve (default: None).
            next_token (str): token for next page of results (default: None).

        Returns:
            list: A list of `Run` objects.
        """
        tc_summaries = _TrialComponent.list(
            experiment_name=experiment_name,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            sagemaker_session=sagemaker_session,
            max_results=max_results,
            next_token=next_token,
        )
        run_list = []
        for tc_summary in tc_summaries:
            tc = _TrialComponent.load(
                trial_component_name=tc_summary.trial_component_name,
                sagemaker_session=sagemaker_session,
            )
            run_instance = cls(
                experiment_name=experiment_name,
                run_name=tc_summary.trial_component_name,
                trial_component=tc,
                sagemaker_session=sagemaker_session,
            )
            run_list.append(run_instance)
        return run_list

    def close(self):
        """Persist any data saved locally."""
        try:
            # Update the trial component with additions from the Run object
            self._trial_component.save()
            # Create Lineage entities for the artifacts
            self._lineage_artifact_tracker.save()
        finally:
            if self._metrics_manager:
                self._metrics_manager.close()

    @staticmethod
    def _generate_trial_name(base_name):
        """Generate the reserved trial name based on run name

        Args:
            base_name (str): The run_name of this `Run` object.
        """
        available_length = MAX_NAME_LEN_IN_BACKEND - len(TRIAL_NAME_TEMPLATE)
        return TRIAL_NAME_TEMPLATE.format(base_name[:available_length])

    @staticmethod
    def _is_input_valid(input_type, field_name, field_value):
        """Check if the input is valid or not

        Args:
            input_type (str): The type of the input, one of `parameter`, `metric`.
            field_name (str): The name of the field to be checked.
            field_value (str or numbers.Number): The value of the field to be checked.
        """
        if isinstance(field_value, Number) and (isnan(field_value) or isinf(field_value)):
            logger.warning(
                "Failed to log %s %s. Received invalid value: %s.",
                input_type,
                field_name,
                field_value,
            )
            return False
        return True

    def _log_graph_artifact(self, data, graph_type, is_output, artifact_name=None):
        """Log an artifact.

        Logs an artifact by uploading data to S3, creating an artifact, and associating that
        artifact with the run trial component.

        Args:
            data (dict): Artifacts data that will be saved to S3.
            graph_type (str):  The type of the artifact.
            is_output (bool): Determines direction of association to the
                trial component. Defaults to True (output artifact).
                If set to False then represented as input association.
            artifact_name (str): Name of the artifact (default: None).
        """
        # generate an artifact name
        if not artifact_name:
            unique_name_from_base(graph_type)

        # create a json file in S3
        s3_uri, etag = self._artifact_uploader.upload_object_artifact(
            artifact_name, data, file_extension="json"
        )

        # create an artifact and association for the table
        if is_output:
            self._lineage_artifact_tracker.add_output_artifact(
                name=artifact_name, source_uri=s3_uri, etag=etag, artifact_type=graph_type
            )
        else:
            self._lineage_artifact_tracker.add_input_artifact(
                name=artifact_name, source_uri=s3_uri, etag=etag, artifact_type=graph_type
            )

    def _verify_trial_component_artifacts_length(self, is_output):
        """Verify the length of trial component artifacts

        Args:
            is_output (bool): Determines direction of association to the
                trial component.

        Raises:
            ValueError: If the length of trial component artifacts exceeds the limit.
        """
        err_msg_template = "Cannot add more than {} {}_artifacts under run trial_component"
        if is_output:
            if len(self._trial_component.output_artifacts) >= MAX_RUN_TC_ARTIFACTS_LEN:
                raise ValueError(err_msg_template.format(MAX_RUN_TC_ARTIFACTS_LEN, "output"))
        else:
            if len(self._trial_component.input_artifacts) >= MAX_RUN_TC_ARTIFACTS_LEN:
                raise ValueError(err_msg_template.format(MAX_RUN_TC_ARTIFACTS_LEN, "input"))

    @staticmethod
    def _get_tc_and_exp_config_by_run_name(
        run_name: str,
        experiment_name: str,
        sagemaker_session: "Session",
    ):
        """Retrieve a trial component and experiment config by the run_name etc.

        Args:
            run_name (str): The name of a run object.
            experiment_name (str): The name of the Experiment the run object is
                associated with.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
        """
        trial_component_name = Run._generate_trial_component_name(
            run_name=run_name, experiment_name=experiment_name
        )
        run_tc = _TrialComponent.load(
            trial_component_name=trial_component_name,
            sagemaker_session=sagemaker_session,
        )
        exp_config = {
            EXPERIMENT_NAME: experiment_name,
            TRIAL_NAME: Run._generate_trial_name(experiment_name),
            RUN_NAME: trial_component_name,
        }

        return run_tc, exp_config

    @staticmethod
    def _get_tc_and_exp_config_from_job_env(
        environment: _RunEnvironment,
        sagemaker_session: "Session",
    ):
        """Retrieve a trial component and experiment config from the job environment.

        Note: Only Training Job is supported at this point.

        Args:
            environment (_RunEnvironment): The run environment object with job specific data.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
        """
        job_name = environment.source_arn.split("/")[-1]
        if environment.environment_type == _EnvironmentType.SageMakerTrainingJob:
            job_response = retry_with_backoff(
                callable_func=lambda: sagemaker_session.describe_training_job(job_name),
                num_attempts=4,
            )
        else:  # environment.environment_type == _EnvironmentType.SageMakerProcessingJob
            job_response = retry_with_backoff(
                callable_func=lambda: sagemaker_session.describe_processing_job(job_name),
                num_attempts=4,
            )
        job_exp_config = job_response.get("ExperimentConfig", dict())
        if job_exp_config.get(RUN_NAME, None):
            # The run with RunName has been created outside of the job env.
            # The job env already ensures the trial_component/run given in experiment config exists
            # otherwise it fails to create the job.
            tc = _TrialComponent.load(
                trial_component_name=job_exp_config[RUN_NAME], sagemaker_session=sagemaker_session
            )
            return tc, job_exp_config
        raise RuntimeError(
            "Not able to fetch RunName in ExperimentConfig of the sagemaker job. "
            "Please make sure the ExperimentConfig is correctly set."
        )

    @staticmethod
    def _generate_trial_component_name(run_name: str, experiment_name: str) -> str:
        """Generate the TrialComponentName based on run_name and experiment_name

        Args:
            run_name (str): The run_name supplied by the user.
            experiment_name (str): The experiment_name supplied by the user,
                which is prepended to the run_name to generate the TrialComponentName.

        Returns:
            str: The TrialComponentName used to create a trial component
                which is unique in an account.

        Raises:
            ValueError: If either the run_name or the experiment_name exceeds
                the length limit.
        """
        buffer = 1  # leave length buffers for delimiters
        max_len = int(MAX_NAME_LEN_IN_BACKEND / 2) - buffer
        err_msg_template = "The {} (length: {}) must have length less than or equal to {}"
        if len(run_name) > max_len:
            raise ValueError(err_msg_template.format("run_name", len(run_name), max_len))
        if len(experiment_name) > max_len:
            raise ValueError(
                err_msg_template.format("experiment_name", len(experiment_name), max_len)
            )
        return "{}{}{}".format(experiment_name, DELIMITER, run_name)

    @staticmethod
    def _verify_load_input_names(
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Verify the run_name and the experiment_name inputs in Run.load.

        Args:
            run_name (str): The run_name supplied by the user (default: None).
            experiment_name (str): The experiment_name supplied by the user
                (default: None).

        Raises:
            ValueError: If run_name is supplied while experiment_name is not.
        """
        if not run_name and experiment_name:
            logger.warning(
                "No run_name is supplied. Ignoring the provided experiment_name "
                "since it only takes effect along with run_name. "
                "Will load the Run object from the job environment or current Run context."
            )
        if run_name and not experiment_name:
            raise ValueError(
                "Invalid input: experiment_name is missing when run_name is supplied. "
                "Please supply valid experiment_name when the run_name is not None."
            )

    @staticmethod
    def _append_run_tc_label_to_tags(tags: Optional[List[Dict[str, str]]] = None) -> list:
        """Append the Run TrialComponent label to tags used to create a trial component.

        Args:
            tags (List[Dict[str, str]]): The tags supplied by users to initialize a Run object.

        Returns:
            list: The updated tags with the appended Run TrialComponent label.
        """
        if not tags:
            tags = []
        tags.append({"Key": RUN_TC_TAG_KEY, "Value": RUN_TC_TAG_VALUE})
        return tags

    def __enter__(self):
        """Updates the start time of the tracked trial component.

        Returns:
            object: self.
        """
        nested_with_err_msg_template = (
            "It is not allowed to use nested 'with' statements on the {}."
        )
        if self._in_load:
            if self._inside_load_context:
                raise RuntimeError(nested_with_err_msg_template.format("Run.load"))
            self._inside_load_context = True
        else:
            if _RunContext.get_current_run():
                raise RuntimeError(nested_with_err_msg_template.format("Run.init"))
            self._inside_init_context = True
            _RunContext.add_run_object(self)

        if not self._trial_component.start_time:
            start_time = datetime.datetime.now(dateutil.tz.tzlocal())
            self._trial_component.start_time = start_time
        self._trial_component.status = _api_types.TrialComponentStatus(
            primary_status=_TrialComponentStatusType.InProgress.value,
            message="Within a Run context",
        )
        # Save the start_time and status changes to backend
        self._trial_component.save()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Updates the end time of the tracked trial component.

        Args:
            exc_type (str): The exception type.
            exc_value (str): The exception value.
            exc_traceback (str): The stack trace of the exception.
        """
        if self._in_load:
            self._inside_load_context = False
            self._in_load = False
        else:
            self._inside_init_context = False
            _RunContext.drop_current_run()

        end_time = datetime.datetime.now(dateutil.tz.tzlocal())
        self._trial_component.end_time = end_time
        if exc_value:
            self._trial_component.status = _api_types.TrialComponentStatus(
                primary_status=_TrialComponentStatusType.Failed.value, message=str(exc_value)
            )
        else:
            self._trial_component.status = _api_types.TrialComponentStatus(
                primary_status=_TrialComponentStatusType.Completed.value
            )

        self.close()
