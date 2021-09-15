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
"""Scrapper utilities to support repacking of models."""
from __future__ import absolute_import

import os
import shutil
import tarfile
import tempfile
from typing import List, Union
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
from sagemaker.session import get_create_model_package_request
from sagemaker.session import get_model_package_args
from sagemaker.workflow.steps import (
    StepTypeEnum,
    TrainingStep,
    Step,
)

FRAMEWORK_VERSION = "0.23-1"
INSTANCE_TYPE = "ml.m5.large"
REPACK_SCRIPT = "_repack_model.py"


class _RepackModelStep(TrainingStep):
    """Repacks model artifacts with custom inference entry points.

    The SDK automatically adds this step to pipelines that have RegisterModelSteps with models
    that have a custom entry point.
    """

    def __init__(
        self,
        name: str,
        sagemaker_session,
        role,
        model_data: str,
        entry_point: str,
        display_name: str = None,
        description: str = None,
        source_dir: str = None,
        dependencies: List = None,
        depends_on: Union[List[str], List[Step]] = None,
        subnets=None,
        security_group_ids=None,
        **kwargs,
    ):
        """Base class initializer.

        Args:
            name (str): The name of the training step.
            sagemaker_session (sagemaker.session.Session): Session object which manages
                    interactions with Amazon SageMaker APIs and any other AWS services needed. If
                    not specified, the estimator creates one using the default
                    AWS configuration chain.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                    SageMaker training jobs and APIs that create Amazon SageMaker
                    endpoints use this role to access training data and model
                    artifacts. After the endpoint is created, the inference code
                    might use the IAM role, if it needs to access an AWS resource.
            model_data (str): The S3 location of a SageMaker model data
                    ``.tar.gz`` file (default: None).
            entry_point (str): Path (absolute or relative) to the local Python
                    source file which should be executed as the entry point to
                    inference. If ``source_dir`` is specified, then ``entry_point``
                    must point to a file located at the root of ``source_dir``.
                    If 'git_config' is provided, 'entry_point' should be
                    a relative location to the Python source file in the Git repo.

                    Example:
                        With the following GitHub repo directory structure:

                        >>> |----- README.md
                        >>> |----- src
                        >>>         |----- train.py
                        >>>         |----- test.py

                        You can assign entry_point='src/train.py'.
            source_dir (str): A relative location to a directory with other training
                or model hosting source code dependencies aside from the entry point
                file in the Git repo (default: None). Structure within this
                directory are preserved when training on Amazon SageMaker.
            dependencies (list[str]): A list of paths to directories (absolute
                    or relative) with any additional libraries that will be exported
                    to the container (default: []). The library folders will be
                    copied to SageMaker in the same folder where the entrypoint is
                    copied. If 'git_config' is provided, 'dependencies' should be a
                    list of relative locations to directories with any additional
                    libraries needed in the Git repo.

                    .. admonition:: Example

                        The following call

                        >>> Estimator(entry_point='train.py',
                        ...           dependencies=['my/libs/common', 'virtual-env'])

                        results in the following inside the container:

                        >>> $ ls

                        >>> opt/ml/code
                        >>>     |------ train.py
                        >>>     |------ common
                        >>>     |------ virtual-env

                    This is not supported with "local code" in Local Mode.
            depends_on (List[str] or List[Step]): A list of step names or instances
                    this step depends on
            subnets (list[str]): List of subnet ids. If not specified, the re-packing
                    job will be created without VPC config.
            security_group_ids (list[str]): List of security group ids. If not
                specified, the re-packing job will be created without VPC config.
        """
        self._model_data = model_data
        self.sagemaker_session = sagemaker_session
        self.role = role
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
            sagemaker_session=self.sagemaker_session,
            role=self.role,
            hyperparameters={
                "inference_script": self._entry_point_basename,
                "model_archive": self._model_archive,
            },
            subnets=subnets,
            security_group_ids=security_group_ids,
            **kwargs,
        )
        repacker.disable_profiler = True
        inputs = TrainingInput(self._model_prefix)

        # super!
        super(_RepackModelStep, self).__init__(
            name=name,
            display_name=display_name,
            description=description,
            depends_on=depends_on,
            estimator=repacker,
            inputs=inputs,
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
                    sagemaker_session=self.sagemaker_session,
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
                    sagemaker_session=self.sagemaker_session,
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
        container_def_list (list): A list of container definitions.
        **kwargs: additional arguments to `create_model`.
    """

    def __init__(
        self,
        name: str,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        estimator: EstimatorBase = None,
        model_data=None,
        model_package_group_name=None,
        model_metrics=None,
        metadata_properties=None,
        approval_status="PendingManualApproval",
        image_uri=None,
        compile_model_family=None,
        display_name: str = None,
        description=None,
        depends_on: Union[List[str], List[Step]] = None,
        tags=None,
        container_def_list=None,
        **kwargs,
    ):
        """Constructor of a register model step.

        Args:
            name (str): The name of the training step.
            step_type (StepTypeEnum): The type of the step with value
                `StepTypeEnum.Training`.
            estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
            model_data: the S3 URI to the model data from training.
            content_types (list): The supported MIME types for the
                input data (default: None).
            response_types (list): The supported MIME types for
                the output data (default: None).
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time (default: None).
            transform_instances (list): A list of the instance types on which a
                transformation job can be run or on which an endpoint
                can be deployed (default: None).
            model_package_group_name (str): Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name`
                makes the Model Package versioned (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties object
                (default: None).
            approval_status (str): Model Approval Status, values can be "Approved",
                "Rejected", or "PendingManualApproval"
                (default: "PendingManualApproval").
            image_uri (str): The container image uri for Model Package, if not specified,
                Estimator's training container image will be used (default: None).
            compile_model_family (str): Instance family for compiled model,
                if specified, a compiled model will be used (default: None).
            description (str): Model Package description (default: None).
            depends_on (List[str] or List[Step]): A list of step names or instances
                this step depends on
            **kwargs: additional arguments to `create_model`.
        """
        super(_RegisterModelStep, self).__init__(
            name, display_name, description, StepTypeEnum.REGISTER_MODEL, depends_on
        )
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
        self.tags = tags
        self.kwargs = kwargs
        self.container_def_list = container_def_list

        self._properties = Properties(path=f"Steps.{name}", shape_name="DescribeModelPackageOutput")

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that are used to call `create_model_package`."""
        model_name = self.name

        if self.container_def_list is None:
            if self.compile_model_family:
                model = self.estimator._compiled_models[self.compile_model_family]
                self.model_data = model.model_data
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
                    self.image_uri = model.image_uri

                if self.model_data is None:
                    self.model_data = model.model_data

                # reset placeholder
                self.estimator.output_path = output_path

                # yeah, there is some framework stuff going on that we need to pull in here
                if self.image_uri is None:
                    region_name = self.estimator.sagemaker_session.boto_session.region_name
                    self.image_uri = image_uris.retrieve(
                        model._framework_name,
                        region_name,
                        version=model.framework_version,
                        py_version=model.py_version if hasattr(model, "py_version") else None,
                        instance_type=self.kwargs.get(
                            "instance_type", self.estimator.instance_type
                        ),
                        accelerator_type=self.kwargs.get("accelerator_type"),
                        image_scope="inference",
                    )
                    model.name = model_name
                    model.model_data = self.model_data

        model_package_args = get_model_package_args(
            content_types=self.content_types,
            response_types=self.response_types,
            inference_instances=self.inference_instances,
            transform_instances=self.transform_instances,
            model_package_group_name=self.model_package_group_name,
            model_data=self.model_data,
            image_uri=self.image_uri,
            model_metrics=self.model_metrics,
            metadata_properties=self.metadata_properties,
            approval_status=self.approval_status,
            description=self.description,
            tags=self.tags,
            container_def_list=self.container_def_list,
        )

        request_dict = get_create_model_package_request(**model_package_args)
        # these are not available in the workflow service and will cause rejection
        if "CertifyForMarketplace" in request_dict:
            request_dict.pop("CertifyForMarketplace")
        if "Description" in request_dict:
            request_dict.pop("Description")

        return request_dict

    @property
    def properties(self):
        """A Properties object representing the DescribeModelPackageOutput data model."""
        return self._properties
