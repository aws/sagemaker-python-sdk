# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Scrapper utilities to support repacking of models."""
from __future__ import absolute_import

import os
import shutil
import tarfile
import tempfile

from typing import List

from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.s3 import (
    S3Downloader,
    S3Uploader,
)
from sagemaker.estimator import EstimatorBase
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import (
    StepTypeEnum,
    TrainingStep,
    Step,
)

FRAMEWORK_VERSION = "0.23-1"
INSTANCE_TYPE = "ml.m5.large"
REPACK_SCRIPT = "_repack_model.py"


class _RepackModelStep(TrainingStep):
    """Repacks model artifacts with inference entry point.

    Attributes:
        name (str): The name of the training step.
        step_type (StepTypeEnum): The type of the step with value `StepTypeEnum.Training`.
        estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
        inputs (TrainingInput): A `sagemaker.inputs.TrainingInput` instance. Defaults to `None`.
    """

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        model_data: str,
        entry_point: str,
        source_dir: str = None,
        dependencies: List = None,
        depends_on: List[str] = None,
        **kwargs,
    ):
        """Constructs a TrainingStep, given an `EstimatorBase` instance.

        In addition to the estimator instance, the other arguments are those that are supplied to
        the `fit` method of the `sagemaker.estimator.Estimator`.

        Args:
            name (str): The name of the training step.
            estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
            inputs (TrainingInput): A `sagemaker.inputs.TrainingInput` instance. Defaults to `None`.
        """
        # yeah, go ahead and save the originals for now
        self._estimator = estimator
        self._model_data = model_data
        if isinstance(model_data, Properties):
            self._model_prefix = model_data
            self._model_archive = "model.tar.gz"
        else:
            self._model_prefix = "/".join(self._model_data.split("/")[:-1])
            self._model_archive = self._model_data.split("/")[-1]
        self._entry_point = entry_point
        self._entry_point_basename = os.path.basename(self._entry_point)
        self._source_dir = source_dir
        self._dependencies = dependencies

        # the real estimator and inputs
        repacker = SKLearn(
            framework_version=FRAMEWORK_VERSION,
            instance_type=INSTANCE_TYPE,
            entry_point=REPACK_SCRIPT,
            source_dir=self._source_dir,
            dependencies=self._dependencies,
            sagemaker_session=self._estimator.sagemaker_session,
            role=self._estimator.role,
            hyperparameters={
                "inference_script": self._entry_point_basename,
                "model_archive": self._model_archive,
            },
            **kwargs,
        )
        repacker.disable_profiler = True
        inputs = TrainingInput(self._model_prefix)

        # super!
        super(_RepackModelStep, self).__init__(
            name=name, depends_on=depends_on, estimator=repacker, inputs=inputs
        )

    def _prepare_for_repacking(self):
        """Prepares the source for the estimator."""
        if self._source_dir is None:
            self._establish_source_dir()

        self._inject_repack_script()

    def _establish_source_dir(self):
        """If the source_dir is None, creates it for the repacking job.

        It performs the following:
            1) creates a source directory
            2) copies the inference_entry_point inside it
            3) copies the repack_model.py inside it
            4) sets the source dir for the repacking estimator
        """
        self._source_dir = tempfile.mkdtemp()
        self.estimator.source_dir = self._source_dir

        shutil.copy2(self._entry_point, os.path.join(self._source_dir, self._entry_point_basename))
        self._entry_point = self._entry_point_basename

    def _inject_repack_script(self):
        """Injects the _repack_model.py script where it belongs.

        If the source_dir is an S3 path:
            1) downloads the source_dir tar.gz
            2) copies the _repack_model.py script where it belongs
            3) uploads the mutated source_dir

        If the source_dir is a local path:
            1) copies the _repack_model.py script into the source dir
        """
        fname = os.path.join(os.path.dirname(__file__), REPACK_SCRIPT)
        if self._source_dir.lower().startswith("s3://"):
            with tempfile.TemporaryDirectory() as tmp:
                local_path = os.path.join(tmp, "local.tar.gz")

                S3Downloader.download(
                    s3_uri=self._source_dir,
                    local_path=local_path,
                    sagemaker_session=self._estimator.sagemaker_session,
                )

                src_dir = os.path.join(tmp, "src")
                with tarfile.open(name=local_path, mode="r:gz") as tf:
                    tf.extractall(path=src_dir)

                shutil.copy2(fname, os.path.join(src_dir, REPACK_SCRIPT))
                with tarfile.open(name=local_path, mode="w:gz") as tf:
                    tf.add(src_dir, arcname=".")

                S3Uploader.upload(
                    local_path=local_path,
                    desired_s3_uri=self._source_dir,
                    sagemaker_session=self._estimator.sagemaker_session,
                )
        else:
            shutil.copy2(fname, os.path.join(self._source_dir, REPACK_SCRIPT))

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that are used to call `create_training_job`.

        This first prepares the source bundle for repackinglby placing artifacts
        in locations which the training container will make available to the
        repacking script and then gets the arguments for the training job.
        """
        self._prepare_for_repacking()
        return super(_RepackModelStep, self).arguments

    @property
    def properties(self):
        """A Properties object representing the DescribeTrainingJobResponse data model."""
        return self._properties


class _RegisterModelStep(Step):
    """Register model step in workflow that creates a model package.

    Attributes:
        name (str): The name of the training step.
        step_type (StepTypeEnum): The type of the step with value `StepTypeEnum.Training`.
        estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
        model_data: the S3 URI to the model data from training.
        content_types (list): The supported MIME types for the input data (default: None).
        response_types (list): The supported MIME types for the output data (default: None).
        inference_instances (list): A list of the instance types that are used to
            generate inferences in real-time (default: None).
        transform_instances (list): A list of the instance types on which a transformation
            job can be run or on which an endpoint can be deployed (default: None).
        model_package_group_name (str): Model Package Group name, exclusive to
            `model_package_name`, using `model_package_group_name` makes the Model Package
            versioned (default: None).
        image_uri (str): The container image uri for Model Package, if not specified,
            Estimator's training container image will be used (default: None).
        compile_model_family (str): Instance family for compiled model, if specified, a compiled
            model will be used (default: None).
        **kwargs: additional arguments to `create_model`.
    """

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        model_data,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_group_name=None,
        model_metrics=None,
        metadata_properties=None,
        approval_status="PendingManualApproval",
        image_uri=None,
        compile_model_family=None,
        description=None,
        depends_on: List[str] = None,
        tags=None,
        **kwargs,
    ):
        """Constructor of a register model step.

        Args:
            name (str): The name of the training step.
            step_type (StepTypeEnum): The type of the step with value `StepTypeEnum.Training`.
            estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
            model_data: the S3 URI to the model data from training.
            content_types (list): The supported MIME types for the input data (default: None).
            response_types (list): The supported MIME types for the output data (default: None).
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time (default: None).
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed (default: None).
            model_package_group_name (str): Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties object (default: None).
            approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
                or "PendingManualApproval" (default: "PendingManualApproval").
            image_uri (str): The container image uri for Model Package, if not specified,
                Estimator's training container image will be used (default: None).
            compile_model_family (str): Instance family for compiled model, if specified, a compiled
                model will be used (default: None).
            description (str): Model Package description (default: None).
            depends_on (List[str]): A list of step names this `sagemaker.workflow.steps.TrainingStep`
                depends on
            **kwargs: additional arguments to `create_model`.
        """
        super(_RegisterModelStep, self).__init__(name, StepTypeEnum.REGISTER_MODEL, depends_on)
        self.estimator = estimator
        self.model_data = model_data
        self.content_types = content_types
        self.response_types = response_types
        self.inference_instances = inference_instances
        self.transform_instances = transform_instances
        self.model_package_group_name = model_package_group_name
        self.tags = tags
        self.model_metrics = model_metrics
        self.metadata_properties = metadata_properties
        self.approval_status = approval_status
        self.image_uri = image_uri
        self.compile_model_family = compile_model_family
        self.description = description
        self.kwargs = kwargs

        self._properties = Properties(
            path=f"Steps.{name}", shape_name="DescribeModelPackageResponse"
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that are used to call `create_model_package`."""
        model_name = self.name
        if self.compile_model_family:
            model = self.estimator._compiled_models[self.compile_model_family]
        else:
            # create_model wants the estimator to have a model_data attribute...
            self.estimator._current_job_name = model_name

            # placeholder. replaced with model_data later
            output_path = self.estimator.output_path
            self.estimator.output_path = "/tmp"

            # create the model, but custom funky framework stuff going on in some places
            if self.image_uri:
                model = self.estimator.create_model(image_uri=self.image_uri, **self.kwargs)
            else:
                model = self.estimator.create_model(**self.kwargs)
            model.model_data = self.model_data

            # reset placeholder
            self.estimator.output_path = output_path

            # yeah, there is some framework stuff going on that we need to pull in here
            if model.image_uri is None:
                region_name = self.estimator.sagemaker_session.boto_session.region_name
                model.image_uri = image_uris.retrieve(
                    model._framework_name,
                    region_name,
                    version=model.framework_version,
                    py_version=model.py_version if hasattr(model, "py_version") else None,
                    instance_type=self.kwargs.get("instance_type", self.estimator.instance_type),
                    accelerator_type=self.kwargs.get("accelerator_type"),
                    image_scope="inference",
                )
        model.name = model_name

        model_package_args = model._get_model_package_args(
            content_types=self.content_types,
            response_types=self.response_types,
            inference_instances=self.inference_instances,
            transform_instances=self.transform_instances,
            model_package_group_name=self.model_package_group_name,
            model_metrics=self.model_metrics,
            metadata_properties=self.metadata_properties,
            approval_status=self.approval_status,
            description=self.description,
            tags=self.tags,
        )
        request_dict = model.sagemaker_session._get_create_model_package_request(
            **model_package_args
        )

        # these are not available in the workflow service and will cause rejection
        if "CertifyForMarketplace" in request_dict:
            request_dict.pop("CertifyForMarketplace")
        if "Description" in request_dict:
            request_dict.pop("Description")

        return request_dict

    @property
    def properties(self):
        """A Properties object representing the DescribeTrainingJobResponse data model."""
        return self._properties
