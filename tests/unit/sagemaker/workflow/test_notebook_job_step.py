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

import unittest
from mock import Mock, patch

from sagemaker.workflow.notebook_job_step import NotebookJobStep
from sagemaker.workflow.functions import Join

REGION = "us-west-2"
PIPELINE_NAME = "test-pipeline-name"
STEP_NAME = "NotebookStep"
DISPLAY_NAME = "Display Name"
DESCRIPTION = "Description"
NOTEBOOK_JOB_NAME = "MyNotebookJob"
INPUT_NOTEBOOK = "tests/data/workflow/notebook_job_step/notebook1_happypath.ipynb"
IMAGE_URI = "236514542706.dkr.ecr.us-west-2.amazonaws.com/sagemaker-data-science-310@1"
KERNEL_NAME = "python3"
ROLE = "arn:aws:iam::123456789012:role/MySageMakerRole"
DEFAULT_S3_BUCKET = "my-bucket"
S3_ROOT_URI = "s3://my-bucket/abc"


def mock_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        default_bucket_prefix="abc",
        sagemaker_config=None,
    )
    session_mock.upload_data = Mock(name="upload_data", return_value="s3_uri_to_uploaded_data")
    session_mock.download_data = Mock(name="download_data")
    session_mock.get_caller_identity_arn = Mock(name="get_caller_identity_arn", return_value=ROLE)
    session_mock.default_bucket = Mock(name="default_bucket", return_value=DEFAULT_S3_BUCKET)
    session_mock.expand_role = Mock(name="expand_role", return_value=ROLE)
    return session_mock


class TestNotebookJobStep(unittest.TestCase):
    def setUp(self):
        # setup mock for compilation_context
        self.patcher = patch("sagemaker.workflow.notebook_job_step.load_step_compilation_context")
        self.mock_load_step_compilation_context = self.patcher.start()
        self.mocked_session = mock_session()
        mock_compilation_context = Mock(
            sagemaker_session=self.mocked_session, pipeline_name=PIPELINE_NAME
        )
        self.mock_load_step_compilation_context.return_value = mock_compilation_context

    def test_initialization(self):
        parameters = {"param1": "value1", "param2": "value2"}
        environment_variables = {"env_var1": "value1", "env_var2": "value2"}
        initialization_script = "tests/data/workflow/notebook_job_step/my-init.sh"
        s3_kms_key = "arn:aws:kms:us-east-1:123456789012:key/abc123"
        instance_type = "ml.m5.large"
        volume_size = 50
        volume_kms_key = "arn:aws:kms:us-east-1:123456789012:key/xyz456"
        encrypt_inter_container_traffic = True
        security_group_ids = ["sg-12345678", "sg-87654321"]
        subnets = ["subnet-12345678", "subnet-87654321"]
        max_retry_attempts = 2
        max_runtime_in_seconds = 3600
        tags = {"TagKey": "TagValue", "TagKey2": "TagValue2"}
        additional_dependencies = [".", "tests/data/workflow/notebook_job_step/subfolder"]
        retry_policies = []
        depends_on = []

        # Create a NotebookJobStep instance
        notebook_step = NotebookJobStep(
            name=STEP_NAME,
            display_name=DISPLAY_NAME,
            description=DESCRIPTION,
            notebook_job_name=NOTEBOOK_JOB_NAME,
            input_notebook=INPUT_NOTEBOOK,
            image_uri=IMAGE_URI,
            kernel_name=KERNEL_NAME,
            role=ROLE,
            s3_root_uri=S3_ROOT_URI,
            parameters=parameters,
            environment_variables=environment_variables,
            initialization_script=initialization_script,
            s3_kms_key=s3_kms_key,
            instance_type=instance_type,
            volume_size=volume_size,
            volume_kms_key=volume_kms_key,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            security_group_ids=security_group_ids,
            subnets=subnets,
            max_retry_attempts=max_retry_attempts,
            max_runtime_in_seconds=max_runtime_in_seconds,
            tags=tags,
            additional_dependencies=additional_dependencies,
            retry_policies=retry_policies,
            depends_on=depends_on,
        )

        # Assert that the attributes are set correctly
        self.assertEqual(notebook_step.name, STEP_NAME)
        self.assertEqual(notebook_step.display_name, DISPLAY_NAME)
        self.assertEqual(notebook_step.description, DESCRIPTION)
        self.assertEqual(notebook_step.notebook_job_name, NOTEBOOK_JOB_NAME)
        self.assertEqual(notebook_step.input_notebook, INPUT_NOTEBOOK)
        self.assertEqual(notebook_step.image_uri, IMAGE_URI)
        self.assertEqual(notebook_step.kernel_name, KERNEL_NAME)
        self.assertEqual(notebook_step.role, ROLE)
        self.assertEqual(notebook_step.s3_root_uri, S3_ROOT_URI)
        self.assertEqual(notebook_step.parameters, parameters)
        self.assertEqual(notebook_step.environment_variables, environment_variables)
        self.assertEqual(notebook_step.initialization_script, initialization_script)
        self.assertEqual(notebook_step.s3_kms_key, s3_kms_key)
        self.assertEqual(notebook_step.instance_type, instance_type)
        self.assertEqual(notebook_step.volume_size, volume_size)
        self.assertEqual(notebook_step.volume_kms_key, volume_kms_key)
        self.assertEqual(
            notebook_step.encrypt_inter_container_traffic,
            encrypt_inter_container_traffic,
        )
        self.assertEqual(notebook_step.security_group_ids, security_group_ids)
        self.assertEqual(notebook_step.subnets, subnets)
        self.assertEqual(notebook_step.max_retry_attempts, max_retry_attempts)
        self.assertEqual(notebook_step.max_runtime_in_seconds, max_runtime_in_seconds)
        self.assertEqual(notebook_step.tags, tags)
        self.assertEqual(notebook_step.additional_dependencies, additional_dependencies)
        self.assertEqual(notebook_step.retry_policies, retry_policies)
        self.assertEqual(notebook_step.depends_on, depends_on)

    @patch("sagemaker.workflow.notebook_job_step.name_from_base")
    def test_initialization_for_optional_fields(self, mock_name_from_base):
        derived_notebook_job_name = "job_name_abc"
        mock_name_from_base.return_value = derived_notebook_job_name

        notebook_step = NotebookJobStep(
            input_notebook=INPUT_NOTEBOOK,
            image_uri=IMAGE_URI,
            kernel_name=KERNEL_NAME,
        )
        # call arguments to trigger default value resolving logic
        notebook_step.arguments

        # Verify the default values for optional fields
        self.assertIsNone(notebook_step.display_name)
        self.assertIsNone(notebook_step.description)
        self.assertEqual(notebook_step.name, derived_notebook_job_name)
        self.assertEqual(notebook_step.notebook_job_name, derived_notebook_job_name)
        mock_name_from_base.assert_any_call("input")
        mock_name_from_base.assert_any_call("notebook1-happypath-ipynb")

        self.assertEqual(notebook_step.s3_root_uri, S3_ROOT_URI)
        self.assertIsNone(notebook_step.parameters)
        self.assertIsNone(notebook_step.environment_variables)
        self.assertIsNone(notebook_step.initialization_script)
        self.assertIsNone(notebook_step.s3_kms_key)
        self.assertEqual(notebook_step.instance_type, "ml.m5.large")
        self.assertEqual(notebook_step.volume_size, 30)
        self.assertIsNone(notebook_step.volume_kms_key)
        self.assertTrue(notebook_step.encrypt_inter_container_traffic)
        self.assertIsNone(notebook_step.security_group_ids)
        self.assertIsNone(notebook_step.subnets)
        self.assertEqual(notebook_step.max_retry_attempts, 1)
        self.assertEqual(notebook_step.max_runtime_in_seconds, 2 * 24 * 60 * 60)
        self.assertEqual(notebook_step.role, ROLE)
        self.assertIsNone(notebook_step.tags)
        self.assertIsNone(notebook_step.additional_dependencies)
        # ConfigurableRetryStep added default value as []
        self.assertEqual(notebook_step.retry_policies, [])
        self.assertIsNone(notebook_step.depends_on)

    def test_invalid_inputs_required_fields_passed_as_none(self):
        with self.assertRaises(ValueError) as context:
            NotebookJobStep(
                name=STEP_NAME,
                display_name=DISPLAY_NAME,
                description=DESCRIPTION,
                notebook_job_name=None,
                input_notebook=None,
                image_uri=None,
                kernel_name=None,
            ).arguments
        self.assertTrue(
            "Notebook Job Name(None) is not valid. Valid name should start with "
            "letters and contain only letters, numbers, hyphens, and underscores."
            in str(context.exception)
        )
        self.assertTrue(
            "The required input notebook(None) is not a valid file." in str(context.exception)
        )
        self.assertTrue(
            "The image uri(specified as None) is required and should be hosted in "
            "same region of the session(us-west-2)." in str(context.exception)
        )
        self.assertTrue("The kernel name is required." in str(context.exception))

    def test_invalid_paths_to_upload(self):
        with self.assertRaises(ValueError) as context:
            NotebookJobStep(
                name=STEP_NAME,
                display_name=DISPLAY_NAME,
                description=DESCRIPTION,
                notebook_job_name=NOTEBOOK_JOB_NAME,
                input_notebook="path/non-existing-file",
                image_uri=IMAGE_URI,
                kernel_name=KERNEL_NAME,
                initialization_script="non-existing-script",
                additional_dependencies=["/tmp/non-existing-folder", "path2/non-existing-file"],
            ).arguments

        self.assertTrue(
            "The required input notebook(path/non-existing-file) is not a valid file."
            in str(context.exception)
        )
        self.assertTrue(
            "The initialization script(path/non-existing-file) is not a valid file."
            in str(context.exception)
        )
        self.assertTrue(
            "The path(/tmp/non-existing-folder) specified in additional dependencies "
            "does not exist." in str(context.exception)
        )
        self.assertTrue(
            "The path(path2/non-existing-file) specified in additional dependencies "
            "does not exist." in str(context.exception)
        )

    def test_image_uri_is_not_in_the_expected_region(self):
        with self.assertRaises(ValueError) as context:
            return NotebookJobStep(
                name=STEP_NAME,
                display_name=DISPLAY_NAME,
                description=DESCRIPTION,
                notebook_job_name=NOTEBOOK_JOB_NAME,
                input_notebook=INPUT_NOTEBOOK,
                image_uri="236514542706.dkr.ecr.us-east-9.amazonaws.com/sagemaker-data-science",
                kernel_name=KERNEL_NAME,
            ).arguments

        self.assertTrue(
            "The image uri(specified as 236514542706.dkr.ecr.us-east-9.amazonaws.com/"
            "sagemaker-data-science) is required and should be hosted in "
            "same region of the session(us-west-2)." in str(context.exception)
        )

    def test_invalid_notebook_job_name(self):
        self._do_test_for_invalid_notebook_job_name("notebo okjob")
        self._do_test_for_invalid_notebook_job_name("123notebookjob")
        self._do_test_for_invalid_notebook_job_name("_Notebookjob")
        self._do_test_for_invalid_notebook_job_name("-notebookjob")
        self._do_test_for_invalid_notebook_job_name(".notebookjob")
        self._do_test_for_invalid_notebook_job_name("noteb.ookjob")
        self._do_test_for_invalid_notebook_job_name("~noteb~ookjob")
        self._do_test_for_invalid_notebook_job_name("noteb@#ookjob")

    def _do_test_for_invalid_notebook_job_name(self, job_name):
        with self.assertRaises(ValueError) as context:
            NotebookJobStep(
                name=STEP_NAME,
                display_name=DISPLAY_NAME,
                description=DESCRIPTION,
                notebook_job_name=job_name,
                input_notebook=INPUT_NOTEBOOK,
                image_uri=IMAGE_URI,
                kernel_name=KERNEL_NAME,
            ).arguments
        self.assertTrue(
            f"Notebook Job Name({job_name}) is not valid. Valid name should start with letters "
            "and contain only letters, numbers, hyphens, and underscores." in str(context.exception)
        )

    def test_valid_notebook_job_name(self):
        self._do_test_for_valid_notebook_job_name("NotebookJob123")
        self._do_test_for_valid_notebook_job_name("Notebo---okJob-123--")
        self._do_test_for_valid_notebook_job_name("Notebo___okJob_123__")
        self._do_test_for_valid_notebook_job_name("notebookjob1234567890")

    def _do_test_for_valid_notebook_job_name(self, job_name):
        # no error is expected.
        NotebookJobStep(
            name=STEP_NAME,
            display_name=DISPLAY_NAME,
            description=DESCRIPTION,
            notebook_job_name=job_name,
            input_notebook=INPUT_NOTEBOOK,
            image_uri=IMAGE_URI,
            kernel_name=KERNEL_NAME,
        )

    def test_properties(self):
        notebook_step = self._create_step_with_required_fields()
        properties = notebook_step.properties

        self.assertEqual(
            properties.ComputingJobName.expr, {"Get": "Steps.NotebookStep.TrainingJobName"}
        )

        self.assertEqual(
            properties.ComputingJobStatus.expr, {"Get": "Steps.NotebookStep.TrainingJobStatus"}
        )

        self.assertEqual(
            properties.NotebookJobInputLocation.expr,
            {"Get": "Steps.NotebookStep.InputDataConfig[0].DataSource.S3DataSource.S3Uri"},
        )

        self.assertEqual(
            properties.NotebookJobOutputLocationPrefix.expr,
            {"Get": "Steps.NotebookStep.OutputDataConfig.S3OutputPath"},
        )

        self.assertEqual(
            properties.InputNotebookName.expr,
            {"Get": "Steps.NotebookStep.Environment['SM_INPUT_NOTEBOOK_NAME']"},
        )

        self.assertEqual(
            properties.OutputNotebookName.expr,
            {"Get": "Steps.NotebookStep.Environment['SM_OUTPUT_NOTEBOOK_NAME']"},
        )

    @patch("sagemaker.workflow.notebook_job_step.ExecutionVariables")
    @patch("sagemaker.workflow.notebook_job_step.name_from_base")
    def test_to_request_step_with_required_fields(
        self, mock_name_from_base, mock_execution_variables
    ):
        self.maxDiff = None
        expected_input_folder = "input-abc"
        mock_name_from_base.return_value = expected_input_folder
        mock_execution_variables.PIPELINE_EXECUTION_ID = Mock()

        notebook_step = self._create_step_with_required_fields()
        request = notebook_step.to_request()

        expected_request = {
            "Name": "NotebookStep",
            "Type": "Training",
            "Arguments": {
                "TrainingJobName": f"{notebook_step._underlying_job_prefix}",
                "RoleArn": "arn:aws:iam::123456789012:role/MySageMakerRole",
                "RetryStrategy": {"MaximumRetryAttempts": 1},
                "StoppingCondition": {"MaxRuntimeInSeconds": 172800},
                "EnableInterContainerTrafficEncryption": True,
                "AlgorithmSpecification": {
                    "TrainingImage": "236514542706.dkr.ecr.us-west-2.amazonaws.com/sagemaker-data-science-310@1",
                    "TrainingInputMode": "File",
                    "ContainerEntrypoint": ["amazon_sagemaker_scheduler"],
                },
                "InputDataConfig": [
                    {
                        "ChannelName": "sagemaker_headless_execution_pipelinestep",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": f"s3://my-bucket/abc/{PIPELINE_NAME}"
                                f"/NotebookStep/{expected_input_folder}",
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                    },
                ],
                "OutputDataConfig": {
                    "S3OutputPath": Join(
                        "/",
                        [
                            S3_ROOT_URI,
                            PIPELINE_NAME,
                            mock_execution_variables.PIPELINE_EXECUTION_ID,
                            STEP_NAME,
                        ],
                    )
                },
                "ResourceConfig": {
                    "VolumeSizeInGB": 30,
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.large",
                },
                "Tags": [
                    {"Key": "sagemaker:name", "Value": "MyNotebookJob"},
                    {"Key": "sagemaker:notebook-name", "Value": "notebook1_happypath.ipynb"},
                    {"Key": "sagemaker:notebook-job-origin", "Value": "PIPELINE_STEP"},
                    {"Key": "sagemaker:is-studio-archived", "Value": "false"},
                ],
                "Environment": {
                    "AWS_DEFAULT_REGION": "us-west-2",
                    "SM_JOB_DEF_VERSION": "1.0",
                    "SM_ENV_NAME": "sagemaker-default-env",
                    "SM_SKIP_EFS_SIMULATION": "true",
                    "SM_EXECUTION_INPUT_PATH": "/opt/ml/input/data/sagemaker_headless_execution_pipelinestep",
                    "SM_KERNEL_NAME": "python3",
                    "SM_INPUT_NOTEBOOK_NAME": "notebook1_happypath.ipynb",
                    "SM_OUTPUT_NOTEBOOK_NAME": f"{notebook_step._underlying_job_prefix}.ipynb",
                },
            },
            "DisplayName": "Display Name",
            "Description": "Description",
        }
        self.assertEqual(request, expected_request)
        # Only verify it's called once as the logic uses a transient tmp folder with random name
        # The uploading function is also tested in integ testing.

        self.mocked_session.upload_data.assert_called_once()
        mock_name_from_base.assert_called_with("input")

    @patch("sagemaker.workflow.notebook_job_step.ExecutionVariables")
    @patch("sagemaker.workflow.notebook_job_step.name_from_base")
    def test_to_request_step_with_all_fields(self, mock_name_from_base, mock_execution_variables):
        self.maxDiff = None
        parameters = {"param1": "value1", "param2": "value2"}
        environment_variables = {"env_var1": "value1", "env_var2": "value2"}
        initialization_script = "tests/data/workflow/notebook_job_step/my-init.sh"
        s3_kms_key = "arn:aws:kms:us-east-1:123456789012:key/abc123"
        instance_type = "ml.m5.large"
        volume_size = 50
        volume_kms_key = "arn:aws:kms:us-east-1:123456789012:key/xyz456"
        encrypt_inter_container_traffic = True
        security_group_ids = ["sg-12345678", "sg-87654321"]
        subnets = ["subnet-12345678", "subnet-87654321"]
        max_retry_attempts = 2
        max_runtime_in_seconds = 3600
        tags = {"TagKey": "TagValue", "TagKey2": "TagValue2"}
        additional_dependencies = [
            "tests/data/workflow/notebook_job_step",
            "tests/data/workflow/notebook_job_step/subfolder",
        ]
        retry_policies = []
        depends_on = []

        expected_input_folder = "input-abc"
        mock_name_from_base.return_value = expected_input_folder
        mock_execution_variables.PIPELINE_EXECUTION_ID = Mock()

        # Create a NotebookJobStep instance
        notebook_step = NotebookJobStep(
            name=STEP_NAME,
            display_name=DISPLAY_NAME,
            description=DESCRIPTION,
            notebook_job_name=NOTEBOOK_JOB_NAME,
            input_notebook=INPUT_NOTEBOOK,
            image_uri=IMAGE_URI,
            kernel_name=KERNEL_NAME,
            role=ROLE,
            s3_root_uri=S3_ROOT_URI,
            parameters=parameters,
            environment_variables=environment_variables,
            initialization_script=initialization_script,
            s3_kms_key=s3_kms_key,
            instance_type=instance_type,
            volume_size=volume_size,
            volume_kms_key=volume_kms_key,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            security_group_ids=security_group_ids,
            subnets=subnets,
            max_retry_attempts=max_retry_attempts,
            max_runtime_in_seconds=max_runtime_in_seconds,
            tags=tags,
            additional_dependencies=additional_dependencies,
            retry_policies=retry_policies,
            depends_on=depends_on,
        )

        request = notebook_step.to_request()
        expected_request = {
            "Name": "NotebookStep",
            "Type": "Training",
            "Arguments": {
                "TrainingJobName": f"{notebook_step._underlying_job_prefix}",
                "RoleArn": "arn:aws:iam::123456789012:role/MySageMakerRole",
                "RetryStrategy": {"MaximumRetryAttempts": 2},
                "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
                "EnableInterContainerTrafficEncryption": True,
                "AlgorithmSpecification": {
                    "TrainingImage": "236514542706.dkr.ecr.us-west-2.amazonaws.com/sagemaker-data-science-310@1",
                    "TrainingInputMode": "File",
                    "ContainerEntrypoint": ["amazon_sagemaker_scheduler"],
                },
                "InputDataConfig": [
                    {
                        "ChannelName": "sagemaker_headless_execution_pipelinestep",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": f"s3://my-bucket/abc/{PIPELINE_NAME}"
                                f"/NotebookStep/{expected_input_folder}",
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                    }
                ],
                "OutputDataConfig": {
                    "S3OutputPath": Join(
                        "/",
                        [
                            S3_ROOT_URI,
                            PIPELINE_NAME,
                            mock_execution_variables.PIPELINE_EXECUTION_ID,
                            STEP_NAME,
                        ],
                    ),
                    "KmsKeyId": "arn:aws:kms:us-east-1:123456789012:key/abc123",
                },
                "ResourceConfig": {
                    "VolumeSizeInGB": 50,
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.large",
                    "VolumeKmsKeyId": "arn:aws:kms:us-east-1:123456789012:key/xyz456",
                },
                "VpcConfig": {
                    "Subnets": ["subnet-12345678", "subnet-87654321"],
                    "SecurityGroupIds": ["sg-12345678", "sg-87654321"],
                },
                "Tags": [
                    {"Key": "TagKey", "Value": "TagValue"},
                    {"Key": "TagKey2", "Value": "TagValue2"},
                    {"Key": "sagemaker:name", "Value": "MyNotebookJob"},
                    {"Key": "sagemaker:notebook-name", "Value": "notebook1_happypath.ipynb"},
                    {"Key": "sagemaker:notebook-job-origin", "Value": "PIPELINE_STEP"},
                    {"Key": "sagemaker:is-studio-archived", "Value": "false"},
                ],
                "Environment": {
                    "env_var1": "value1",
                    "env_var2": "value2",
                    "AWS_DEFAULT_REGION": "us-west-2",
                    "SM_JOB_DEF_VERSION": "1.0",
                    "SM_ENV_NAME": "sagemaker-default-env",
                    "SM_SKIP_EFS_SIMULATION": "true",
                    "SM_EXECUTION_INPUT_PATH": "/opt/ml/input/data/sagemaker_headless_execution_pipelinestep",
                    "SM_KERNEL_NAME": "python3",
                    "SM_INPUT_NOTEBOOK_NAME": "notebook1_happypath.ipynb",
                    "SM_OUTPUT_NOTEBOOK_NAME": f"{notebook_step._underlying_job_prefix}.ipynb",
                    "SM_INIT_SCRIPT": "my-init.sh",
                },
                "HyperParameters": {"param1": "value1", "param2": "value2"},
            },
            "DisplayName": "Display Name",
            "Description": "Description",
        }
        self.assertEqual(request, expected_request)
        # only verify it's called once as the logic uses a transient tmp folder with random name
        # The uploading function is also tested in integ testing.
        self.mocked_session.upload_data.assert_called_once()
        mock_name_from_base.assert_called_with("input")

    def test_expected_error_on_calling_depends_on(self):
        notebook_step = self._create_step_with_required_fields()
        with self.assertRaises(ValueError) as context:
            notebook_step.depends_on = []

        expected_error_message = (
            "Cannot set depends_on for a NotebookJobStep. Use "
            "add_depends_on instead to extend the list."
        )
        self.assertEqual(str(context.exception), expected_error_message)

    def _create_step_with_required_fields(self):
        return NotebookJobStep(
            name=STEP_NAME,
            display_name=DISPLAY_NAME,
            description=DESCRIPTION,
            notebook_job_name=NOTEBOOK_JOB_NAME,
            input_notebook=INPUT_NOTEBOOK,
            image_uri=IMAGE_URI,
            kernel_name=KERNEL_NAME,
        )
