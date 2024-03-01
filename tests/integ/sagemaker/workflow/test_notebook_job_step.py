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

from __future__ import absolute_import

import os.path
import tarfile
import logging
import nbformat as nbf
import pytest

from sagemaker import get_execution_role
from sagemaker.s3 import S3Downloader
from sagemaker.s3_utils import s3_path_join
from sagemaker.utils import name_from_base
from sagemaker.workflow import ParameterString
from sagemaker.workflow.notebook_job_step import NotebookJobStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterBoolean
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.utils import _tmpdir
from tests.integ.retry import retries
from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from tests.integ import kms_utils
from sagemaker.workflow.function_step import step

SAGE_MAKER_ROLE_NAME = "SageMakerRole"

PIPELINE_NAME = "pipeline-test-from-pysdk"
STEP_NAME = "NotebookStepFromPySDKTest"
DISPLAY_NAME = "MyNotebookStep From PySDK Test"
DESCRIPTION = "Description for the step"
NOTEBOOK_JOB_NAME_Prefix = "PySDKTest"

COSMOS_IMAGE_V0_URI = (
    "542918446943.dkr.ecr.us-west-2.amazonaws.com/sagemaker-distribution-prod@sha256"
    ":b49a54ff7ca519dd57cf1a3a7a675f3db5c591fab456f9374e390fb68a99edaf"
)
KERNEL_NAME = "python3"


def test_happycase_minimum_input(sagemaker_session):
    pipeline_name = name_from_base(PIPELINE_NAME, max_length=63)
    input_notebook = "tests/data/workflow/notebook_job_step/simple.ipynb"

    notebook_job_step = NotebookJobStep(
        input_notebook=input_notebook,
        image_uri=COSMOS_IMAGE_V0_URI,
        kernel_name=KERNEL_NAME,
        role=SAGE_MAKER_ROLE_NAME,
    )

    # TODO - will be removed when cosmos image is ready
    notebook_job_step = _config_manual_dependency_installation_short_term_workaround(
        notebook_job_step
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[notebook_job_step],
        sagemaker_session=sagemaker_session,
    )
    logging.info(f"Notebook job step test pipeline definition: {pipeline.definition()}")

    try:
        pipeline.create(get_role(sagemaker_session))
        execution_steps = _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={},
        )

        job_description = _get_training_job_details(execution_steps[0], sagemaker_session)

        # verify the training job is created consistent with input.
        training_job_name = job_description["TrainingJobName"]
        assert training_job_name.startswith("simple-ipynb-")
        assert job_description["AlgorithmSpecification"]["TrainingImage"] == COSMOS_IMAGE_V0_URI
        assert SAGE_MAKER_ROLE_NAME in job_description["RoleArn"]

        assert job_description["Environment"]["SM_INPUT_NOTEBOOK_NAME"] == os.path.basename(
            input_notebook
        )
        assert job_description["Environment"]["SM_KERNEL_NAME"] == KERNEL_NAME
        assert job_description["TrainingJobStatus"] == "Completed"

        # verify the output notebook
        output_s3_uri = s3_path_join(
            job_description["OutputDataConfig"]["S3OutputPath"],
            training_job_name,
            "output",
            "output.tar.gz",
        )
        output_notebook_name = job_description["Environment"]["SM_OUTPUT_NOTEBOOK_NAME"]

        def verify_notebook_for_happy_case(cells):
            # happy case is using tests/data/workflow/notebook_job_step/simple.ipynb which print
            # aws-sagemaker.
            assert NotebookUtils.search_output_of_cells(cells, "aws-sagemaker")

        _download_and_verify_output_notebook(
            output_s3_uri, output_notebook_name, verify_notebook_for_happy_case, sagemaker_session
        )

        # upsert and run a second time
        pipeline.upsert(get_role(sagemaker_session))
        execution_steps = _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={},
        )

        job_description = _get_training_job_details(execution_steps[0], sagemaker_session)
        assert job_description["TrainingJobStatus"] == "Completed"
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


@pytest.mark.skip(
    reason="This test is skipped temporarily due to failures. Need to re-enable later after fix."
)
def test_notebook_job_with_more_configuration(sagemaker_session):
    """This test case is for more complex job configuration.
    1. a parent notebook file with %run magic to execute 'subfolder/sub.ipynb' and the
    simple.ipynb
    2. the notebook job has parameters used by parent and sub notebooks
    3. the environment variables are passed from step and init script
    4. pipeline variables are used for a sanity testing purpose
    5. sub notebook is uploaded as additional_dependencies
    6. init script will be used
    7. additional tags are also attached
    8. apply kms key
    """
    pipeline_name = name_from_base(PIPELINE_NAME, max_length=63)
    notebook_job_name = f"{NOTEBOOK_JOB_NAME_Prefix}-case2-configs"

    input_notebook = "tests/data/workflow/notebook_job_step/notebook1_happypath.ipynb"
    folder_with_sub_notebook = "tests/data/workflow/notebook_job_step/subfolder"
    simple_notebook_path = "tests/data/workflow/notebook_job_step/simple.ipynb"
    init_script = "tests/data/workflow/notebook_job_step/my-init.sh"

    # prepare parameter
    company_parameter = ParameterString(name="company", default_value="Amazon_FromParameter")
    notebook_job_parameters = {
        "company": company_parameter,
        "company2": "Amazon2",
    }

    # prepare env vars
    env_parameter = ParameterString(name="env_key", default_value="EnvValueDefinedByParameter")
    environment_variables = {"env_key": env_parameter, "env_key2": "env_value_abc2"}

    # prepare tags
    tags = {
        "company": company_parameter,
        "sagemaker:user-profile-name": "studio-user",
        "sagemaker:is-scheduling-notebook-job": "true",
    }

    # ParameterInteger for volume
    volume_size_parameter = ParameterInteger(name="volume_size", default_value=40)

    # ParameterBoolean
    encrypt_container_traffic_parameter = ParameterBoolean(
        name="encrypt_container_traffic", default_value=False
    )

    kms_key_id = kms_utils.get_or_create_kms_key(
        sagemaker_session, role_arn=get_role(sagemaker_session)
    )

    notebook_job_step = NotebookJobStep(
        name=STEP_NAME,
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
        notebook_job_name=notebook_job_name,
        image_uri=COSMOS_IMAGE_V0_URI,
        kernel_name=KERNEL_NAME,
        role=SAGE_MAKER_ROLE_NAME,
        input_notebook=input_notebook,
        initialization_script=init_script,
        additional_dependencies=[simple_notebook_path, folder_with_sub_notebook],
        parameters=notebook_job_parameters,
        environment_variables=environment_variables,
        tags=tags,
        volume_size=volume_size_parameter,
        encrypt_inter_container_traffic=encrypt_container_traffic_parameter,
        s3_kms_key=kms_key_id,
        volume_kms_key=kms_key_id,
    )

    # TODO - will be removed when cosmos image is ready
    notebook_job_step = _config_manual_dependency_installation_short_term_workaround(
        notebook_job_step
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            company_parameter,
            env_parameter,
            volume_size_parameter,
            encrypt_container_traffic_parameter,
        ],
        steps=[notebook_job_step],
        sagemaker_session=sagemaker_session,
    )
    logging.info(f"Notebook job step test pipeline definition: {pipeline.definition()}")

    try:
        pipeline.create(get_role(sagemaker_session))
        execution_steps = _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={},
        )

        job_description = _get_training_job_details(execution_steps[0], sagemaker_session)

        # verify the training job is created consistent with input.
        assert job_description["AlgorithmSpecification"]["TrainingImage"] == COSMOS_IMAGE_V0_URI
        assert SAGE_MAKER_ROLE_NAME in job_description["RoleArn"]

        # verify notebook job parameters
        assert job_description["HyperParameters"]["company"] == "Amazon_FromParameter"
        assert job_description["HyperParameters"]["company2"] == "Amazon2"

        # verify custom env variables
        assert job_description["Environment"]["env_key"] == "EnvValueDefinedByParameter"
        assert job_description["Environment"]["env_key2"] == "env_value_abc2"

        # verify passed in volume number
        assert job_description["ResourceConfig"]["VolumeSizeInGB"] == 40

        # verify passed in encrypt_container_traffic_parameter
        assert not job_description["EnableInterContainerTrafficEncryption"]

        # verify kms key
        assert job_description["OutputDataConfig"]["KmsKeyId"] == kms_key_id
        assert job_description["ResourceConfig"]["VolumeKmsKeyId"] == kms_key_id

        # verify tags
        tags_from_job = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=job_description["TrainingJobArn"]
        )["Tags"]
        tags_dict = _convert_to_flatten_dict(tags_from_job)
        # tags set by client code
        assert tags_dict["company"] == "Amazon_FromParameter"
        assert tags_dict["sagemaker:user-profile-name"] == "studio-user"
        assert tags_dict["sagemaker:is-scheduling-notebook-job"] == "true"

        # system tags
        assert tags_dict["sagemaker:is-studio-archived"] == "false"
        assert tags_dict["sagemaker:name"] == notebook_job_name
        assert tags_dict["sagemaker:notebook-job-origin"] == "PIPELINE_STEP"

        # verify the job output
        assert job_description["TrainingJobStatus"] == "Completed"

        output_s3_uri = s3_path_join(
            job_description["OutputDataConfig"]["S3OutputPath"],
            job_description["TrainingJobName"],
            "output",
            "output.tar.gz",
        )
        output_notebook_name = job_description["Environment"]["SM_OUTPUT_NOTEBOOK_NAME"]

        def verify_notebook(cells):
            # verify the env variable set by the init script
            # logging for better troubleshooting
            logging.info(cells)
            assert NotebookUtils.search_output_of_cells(
                cells, "ParentNotebook: ENV_VAR_FROM_INIT_SCRIPT=FROM_INIT_SCRIPT"
            )
            assert NotebookUtils.search_output_of_cells(
                cells, "SubNotebook: ENV_VAR_FROM_INIT_SCRIPT=FROM_INIT_SCRIPT"
            )

            # verify the env from the step def
            assert NotebookUtils.search_output_of_cells(
                cells, "ParentNotebook: env_key=EnvValueDefinedByParameter"
            )
            assert NotebookUtils.search_output_of_cells(
                cells, "SubNotebook: env_key=EnvValueDefinedByParameter"
            )

            # verify parameters
            assert NotebookUtils.search_output_of_cells(
                cells, "ParentNotebook: company_is_Amazon_FromParameter"
            )
            assert NotebookUtils.search_output_of_cells(
                cells, "SubNotebook: company_is_Amazon_FromParameter"
            )

            # verify the output of %run simple.ipynb
            assert NotebookUtils.search_output_of_cells(cells, "aws-sagemaker")

        _download_and_verify_output_notebook(
            output_s3_uri,
            output_notebook_name,
            verify_notebook,
            sagemaker_session,
        )
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_notebook_job_dependencies_and_outputs(sagemaker_session):
    pipeline_name = name_from_base(PIPELINE_NAME, max_length=63)
    notebook_job_name_base = f"{NOTEBOOK_JOB_NAME_Prefix}-case3-outputs"

    tags = {
        "sagemaker:user-profile-name": "studio-user",
        "sagemaker:is-scheduling-notebook-job": "true",
    }

    input_notebook = "tests/data/workflow/notebook_job_step/simple.ipynb"
    notebook_job_step1 = NotebookJobStep(
        name=f"{STEP_NAME}-1",
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
        notebook_job_name=f"{notebook_job_name_base}-a",
        input_notebook=input_notebook,
        image_uri=COSMOS_IMAGE_V0_URI,
        kernel_name=KERNEL_NAME,
        role=SAGE_MAKER_ROLE_NAME,
        tags=tags,
    )

    # TODO - will be removed when cosmos image is ready
    notebook_job_step1 = _config_manual_dependency_installation_short_term_workaround(
        notebook_job_step1
    )

    input_notebook2 = "tests/data/workflow/notebook_job_step/step2_notebook.ipynb"
    notebook_job_step2 = NotebookJobStep(
        name=f"{STEP_NAME}-2",
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
        notebook_job_name=f"{notebook_job_name_base}-b",
        input_notebook=input_notebook2,
        image_uri=COSMOS_IMAGE_V0_URI,
        kernel_name=KERNEL_NAME,
        role=SAGE_MAKER_ROLE_NAME,
        parameters={
            "step1_JobName": notebook_job_step1.properties.ComputingJobName,
            "step1_JobStatus": notebook_job_step1.properties.ComputingJobStatus,
            "step1_NotebookJobInput": notebook_job_step1.properties.NotebookJobInputLocation,
            "step1_NotebookJobOutput": notebook_job_step1.properties.NotebookJobOutputLocationPrefix,
            "step1_InputNotebookName": notebook_job_step1.properties.InputNotebookName,
            "step1_OutputNotebookName": notebook_job_step1.properties.OutputNotebookName,
        },
        tags=tags,
    )
    notebook_job_step2.add_depends_on([notebook_job_step1])

    # TODO - will be removed when cosmos image is ready
    notebook_job_step2 = _config_manual_dependency_installation_short_term_workaround(
        notebook_job_step2
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[notebook_job_step1, notebook_job_step2],
        sagemaker_session=sagemaker_session,
    )
    logging.info(f"Notebook job step test pipeline definition: {pipeline.definition()}")

    try:
        pipeline.create(get_role(sagemaker_session))
        execution_steps = _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={},
        )

        job_description1 = None
        job_description2 = None
        for execution_step in execution_steps:
            if execution_step["StepName"] == f"{STEP_NAME}-1":
                job_description1 = _get_training_job_details(execution_step, sagemaker_session)
            elif execution_step["StepName"] == f"{STEP_NAME}-2":
                job_description2 = _get_training_job_details(execution_step, sagemaker_session)

        step1_job_name = job_description1["TrainingJobName"]
        step1_job_status = job_description1["TrainingJobStatus"]

        step1_job_input = job_description1["InputDataConfig"][0]["DataSource"]["S3DataSource"][
            "S3Uri"
        ]
        step1_job_input_notebook = job_description1["Environment"]["SM_INPUT_NOTEBOOK_NAME"]
        step1_job_output = job_description1["OutputDataConfig"]["S3OutputPath"]
        step1_job_output_notebook = job_description1["Environment"]["SM_OUTPUT_NOTEBOOK_NAME"]

        # verify output notebook of step 2
        output_s3_uri = s3_path_join(
            job_description2["OutputDataConfig"]["S3OutputPath"],
            job_description2["TrainingJobName"],
            "output",
            "output.tar.gz",
        )
        output_notebook_name = job_description2["Environment"]["SM_OUTPUT_NOTEBOOK_NAME"]

        def verify_notebook(cells):
            # logging for better troubleshooting
            logging.info(f"step1_JobName={step1_job_name}")
            logging.info(f"step1_JobStatus={step1_job_status}")
            logging.info(f"step1_NotebookJobInput={step1_job_input}")
            logging.info(f"step1_InputNotebookName={step1_job_input_notebook}")
            logging.info(f"step1_NotebookJobOutput={step1_job_output}")
            logging.info(f"step1_OutputNotebookName={step1_job_output_notebook}")
            logging.info(f"cells: {cells}")

            assert NotebookUtils.search_output_of_cells(cells, f"step1_JobName={step1_job_name}")
            assert NotebookUtils.search_output_of_cells(
                cells, f"step1_JobStatus={step1_job_status}"
            )
            assert NotebookUtils.search_output_of_cells(
                cells, f"step1_NotebookJobInput={step1_job_input}"
            )
            assert NotebookUtils.search_output_of_cells(
                cells, f"step1_InputNotebookName={step1_job_input_notebook}"
            )
            assert NotebookUtils.search_output_of_cells(
                cells, f"step1_NotebookJobOutput={step1_job_output}"
            )
            assert NotebookUtils.search_output_of_cells(
                cells, f"step1_OutputNotebookName={step1_job_output_notebook}"
            )

        _download_and_verify_output_notebook(
            output_s3_uri, output_notebook_name, verify_notebook, sagemaker_session
        )
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def test_notebook_job_depend_on_function_step(sagemaker_session, dummy_container_without_error):

    pipeline_name = name_from_base(PIPELINE_NAME, max_length=63)
    notebook_job_name_base = f"{NOTEBOOK_JOB_NAME_Prefix}-case4-depend-on"

    tags = {
        "sagemaker:user-profile-name": "studio-user",
        "sagemaker:is-scheduling-notebook-job": "true",
    }

    @step(
        role=SAGE_MAKER_ROLE_NAME,
        instance_type="ml.m5.large",
        keep_alive_period_in_seconds=60,
        image_uri=dummy_container_without_error,
    )
    def get_env_value():
        """return a static string"""
        return "env_value_from_function_step"

    env_value = get_env_value()

    input_notebook = "tests/data/workflow/notebook_job_step/simple.ipynb"
    notebook_job_step1 = NotebookJobStep(
        name=f"{STEP_NAME}-1",
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
        notebook_job_name=f"{notebook_job_name_base}-a",
        input_notebook=input_notebook,
        image_uri=COSMOS_IMAGE_V0_URI,
        kernel_name=KERNEL_NAME,
        role=SAGE_MAKER_ROLE_NAME,
        tags=tags,
        environment_variables={"env_key": env_value},
    )

    # TODO - will be removed when cosmos image is ready
    notebook_job_step1 = _config_manual_dependency_installation_short_term_workaround(
        notebook_job_step1
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[env_value, notebook_job_step1],
        sagemaker_session=sagemaker_session,
    )

    logging.info(f"Notebook job step test pipeline definition: {pipeline.definition()}")

    try:
        pipeline.create(get_role(sagemaker_session))
        execution_steps = _start_and_verify_execution_with_retry(
            pipeline=pipeline,
            parameters={},
        )

        job_description1 = None
        for execution_step in execution_steps:
            if execution_step["StepName"] == f"{STEP_NAME}-1":
                job_description1 = _get_training_job_details(execution_step, sagemaker_session)

        # verify the output of function step is in notebook job's training job def
        assert job_description1["Environment"]["env_key"] == "env_value_from_function_step"
    finally:
        try:
            pipeline.delete()
        except Exception as error:
            logging.error(error)


def _get_training_job_details(notebook_job_step, sagemaker_session):
    training_job_arn = notebook_job_step["Metadata"]["TrainingJob"]["Arn"]

    return sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=training_job_arn.split("/")[1]
    )


def _download_and_verify_output_notebook(
    output_s3_uri, output_notebook_name, verification, sagemaker_session, kms_key=None
):
    with _tmpdir() as temp_output_folder:
        S3Downloader.download(
            output_s3_uri,
            temp_output_folder,
            sagemaker_session=sagemaker_session,
            kms_key=kms_key,
        )

        with tarfile.open(os.path.join(temp_output_folder, "output.tar.gz"), "r:gz") as tar:
            tar.extract(output_notebook_name, temp_output_folder)

        notebook_cells = NotebookUtils.get_notebook_cells(
            os.path.join(temp_output_folder, output_notebook_name)
        )
        verification(notebook_cells)


def get_role(sagemaker_session):
    return get_execution_role(sagemaker_session)


def _convert_to_flatten_dict(list_of_tag_dicts):
    """Method to convert tags format.

    from:
        [{"Key": "tag1", "Value": "value1"}, {"Key": "tag2", "Value": "value2"}]
    to:
        {"tag1": "value1", "tag2":"value2"}
    """
    result_dict = {}
    for entry in list_of_tag_dicts:
        key = entry["Key"]
        value = entry["Value"]
        result_dict[key] = value
    return result_dict


def _start_and_verify_execution_with_retry(pipeline: Pipeline, parameters: dict) -> list:
    for _ in retries(
        max_retry_count=3,
        exception_message_prefix="Waiting for a successful execution of pipeline",
        seconds_to_sleep=10,
    ):
        execution = pipeline.start(parameters=parameters)
        wait_pipeline_execution(execution=execution)
        execution_steps = execution.list_steps()

        for execution_step in execution_steps:
            failure_reason = execution_step.get("FailureReason", "")
            if failure_reason != "":
                logging.error(f"Pipeline execution failed with error: {failure_reason}. Retrying..")
                continue

        for execution_step in execution_steps:
            if execution_step["StepStatus"] != "Succeeded":
                logging.error(f"execution_steps: {execution_steps}")
            assert execution_step["StepStatus"] == "Succeeded"
        return execution_steps


def _config_manual_dependency_installation_short_term_workaround(notebook_step):
    notebook_step._scheduler_container_entry_point = ["/bin/bash"]

    container_arguments = [
        "-l",
        "-c",
        "pip install sagemaker-headless-execution-driver && exec amazon_sagemaker_scheduler",
    ]

    notebook_step._scheduler_container_arguments = container_arguments

    return notebook_step


# TODO - potentially move to a common place for reuse.
class NotebookUtils:
    @classmethod
    def get_notebook_cells(cls, notebook_path):
        return nbf.read(notebook_path, nbf.NO_CONVERT).cells

    # return the first cell containing the target string
    @classmethod
    def search_output_of_cells(cls, cells, target):
        for cell in cells:
            if "outputs" in cell:
                for output in cell["outputs"]:
                    if "text" in output and output["text"].find(target) > -1:
                        return cell
                    if "output_type" in output and output["output_type"] == "display_data":
                        if str(output["data"]).find(target) > -1:
                            return cell
        return None

    # return true if there is error
    @classmethod
    def search_error_from_output_of_cells(cls, cells):
        for cell in cells:
            if "outputs" in cell:
                for output in cell["outputs"]:
                    if output.output_type == "error":
                        return True
        return False
