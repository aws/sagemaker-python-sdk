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
from __future__ import print_function, absolute_import
import os

import pytest
from mock import Mock, patch
from botocore.exceptions import ClientError
from sagemaker import lambda_helper

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
LAMBDA_ARN = "arn:aws:lambda:us-west-2:123456789012:function:test_function"
FUNCTION_NAME = "test_function"
EXECUTION_ROLE = "arn:aws:iam::123456789012:role/execution_role"
SCRIPT = "test_function.py"
HANDLER = "test_function.lambda_handler"
ZIPPED_CODE_DIR = "code.zip"
S3_BUCKET = "sagemaker-us-west-2-123456789012"
ZIPPED_CODE = "Zipped code"
S3_KEY = "{}/{}/{}".format("lambda", FUNCTION_NAME, "code")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        # default_bucket=S3_BUCKET,
        default_bucket_prefix=None,
    )
    return session_mock


@pytest.fixture()
def s3_client(sagemaker_session):
    return sagemaker_session.boto_session.client(
        "s3", region_name=sagemaker_session.boto_region_name
    )


def test_lambda_object_with_arn_happycase():
    lambda_obj = lambda_helper.Lambda(function_arn=LAMBDA_ARN, session=sagemaker_session)
    assert lambda_obj.function_arn == LAMBDA_ARN


def test_lambda_object_with_name_happycase1():
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
    )

    assert lambda_obj.function_name == FUNCTION_NAME
    assert lambda_obj.execution_role_arn == EXECUTION_ROLE
    assert lambda_obj.script == SCRIPT
    assert lambda_obj.handler == HANDLER
    assert lambda_obj.timeout == 120
    assert lambda_obj.memory_size == 128
    assert lambda_obj.runtime == "python3.8"


def test_lambda_object_with_name_happycase2():
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        zipped_code_dir=ZIPPED_CODE_DIR,
        s3_bucket=S3_BUCKET,
        handler=HANDLER,
        session=sagemaker_session,
    )

    assert lambda_obj.function_name == FUNCTION_NAME
    assert lambda_obj.execution_role_arn == EXECUTION_ROLE
    assert lambda_obj.zipped_code_dir == ZIPPED_CODE_DIR
    assert lambda_obj.s3_bucket == S3_BUCKET
    assert lambda_obj.handler == HANDLER
    assert lambda_obj.timeout == 120
    assert lambda_obj.memory_size == 128
    assert lambda_obj.runtime == "python3.8"


def test_lambda_object_with_no_name_and_arn_error():
    with pytest.raises(ValueError) as error:
        lambda_helper.Lambda(
            execution_role_arn=EXECUTION_ROLE,
            script=SCRIPT,
            handler=HANDLER,
            session=sagemaker_session,
        )
    assert "Either function_arn or function_name must be provided" in str(error)


def test_lambda_object_no_code_error():
    with pytest.raises(ValueError) as error:
        lambda_helper.Lambda(
            function_name=FUNCTION_NAME,
            execution_role_arn=EXECUTION_ROLE,
            handler=HANDLER,
            session=sagemaker_session,
        )
    assert "Either zipped_code_dir or script must be provided" in str(error)


def test_lambda_object_both_script_and_code_dir_error_with_name():
    with pytest.raises(ValueError) as error:
        lambda_helper.Lambda(
            function_name=FUNCTION_NAME,
            execution_role_arn=EXECUTION_ROLE,
            script=SCRIPT,
            zipped_code_dir=ZIPPED_CODE_DIR,
            handler=HANDLER,
            session=sagemaker_session,
        )
    assert "Provide either script or zipped_code_dir, not both." in str(error)


def test_lambda_object_both_script_and_code_dir_error_with_arn():
    with pytest.raises(ValueError) as error:
        lambda_helper.Lambda(
            function_arn=LAMBDA_ARN,
            script=SCRIPT,
            zipped_code_dir=ZIPPED_CODE_DIR,
            session=sagemaker_session,
        )
    assert "Provide either script or zipped_code_dir, not both." in str(error)


def test_lambda_object_no_handler_error():
    with pytest.raises(ValueError) as error:
        lambda_helper.Lambda(
            function_name=FUNCTION_NAME,
            execution_role_arn=EXECUTION_ROLE,
            zipped_code_dir=ZIPPED_CODE_DIR,
            s3_bucket=S3_BUCKET,
            session=sagemaker_session,
        )
    assert "Lambda handler must be provided." in str(error)


def test_lambda_object_no_execution_role_error():
    with pytest.raises(ValueError) as error:
        lambda_helper.Lambda(
            function_name=FUNCTION_NAME,
            zipped_code_dir=ZIPPED_CODE_DIR,
            s3_bucket=S3_BUCKET,
            handler=HANDLER,
            session=sagemaker_session,
        )
    assert "execution_role_arn must be provided." in str(error)


@patch("sagemaker.lambda_helper._zip_lambda_code", return_value=ZIPPED_CODE)
def test_create_lambda_happycase1(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
    )

    lambda_obj.create()
    code = {"ZipFile": ZIPPED_CODE}

    sagemaker_session.lambda_client.create_function.assert_called_with(
        FunctionName=FUNCTION_NAME,
        Runtime="python3.8",
        Handler=HANDLER,
        Role=EXECUTION_ROLE,
        Code=code,
        Timeout=120,
        MemorySize=128,
        VpcConfig={},
        Environment={},
        Layers=[],
    )


@patch("sagemaker.lambda_helper._upload_to_s3", return_value=S3_KEY)
def test_create_lambda_happycase2(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        zipped_code_dir=ZIPPED_CODE_DIR,
        s3_bucket=S3_BUCKET,
        handler=HANDLER,
        session=sagemaker_session,
    )

    lambda_obj.create()
    code = {"S3Bucket": lambda_obj.s3_bucket, "S3Key": S3_KEY}

    sagemaker_session.lambda_client.create_function.assert_called_with(
        FunctionName=FUNCTION_NAME,
        Runtime="python3.8",
        Handler=HANDLER,
        Role=EXECUTION_ROLE,
        Code=code,
        Timeout=120,
        MemorySize=128,
        VpcConfig={},
        Environment={},
        Layers=[],
    )


@patch("sagemaker.lambda_helper._zip_lambda_code", return_value=ZIPPED_CODE)
def test_create_lambda_happycase3(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
        environment={"Name": "my-test-lambda"},
        vpc_config={
            "SubnetIds": ["test-subnet-1"],
            "SecurityGroupIds": ["sec-group-1"],
        },
        layers=["my-test-layer-1", "my-test-layer-2"],
    )

    lambda_obj.create()
    code = {"ZipFile": ZIPPED_CODE}

    sagemaker_session.lambda_client.create_function.assert_called_with(
        FunctionName=FUNCTION_NAME,
        Runtime="python3.8",
        Handler=HANDLER,
        Role=EXECUTION_ROLE,
        Code=code,
        Timeout=120,
        MemorySize=128,
        VpcConfig={"SubnetIds": ["test-subnet-1"], "SecurityGroupIds": ["sec-group-1"]},
        Environment={"Name": "my-test-lambda"},
        Layers=["my-test-layer-1", "my-test-layer-2"],
    )


def test_create_lambda_no_function_name_error(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_arn=LAMBDA_ARN,
        execution_role_arn=EXECUTION_ROLE,
        zipped_code_dir=ZIPPED_CODE_DIR,
        s3_bucket=S3_BUCKET,
        handler=HANDLER,
        session=sagemaker_session,
    )

    with pytest.raises(ValueError) as error:
        lambda_obj.create()
    assert "FunctionName must be provided to create a Lambda function" in str(error)


@patch("sagemaker.lambda_helper._upload_to_s3", return_value=S3_KEY)
def test_create_lambda_client_error(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        zipped_code_dir=ZIPPED_CODE_DIR,
        s3_bucket=S3_BUCKET,
        handler=HANDLER,
        session=sagemaker_session,
    )
    sagemaker_session.lambda_client.create_function.side_effect = ClientError(
        {
            "Error": {
                "Code": "ResourceConflictException",
                "Message": "Function already exists",
            }
        },
        "CreateFunction",
    )

    with pytest.raises(ValueError) as error:
        lambda_obj.create()

    assert "Function already exists" in str(error)


@patch("sagemaker.lambda_helper._zip_lambda_code", return_value=ZIPPED_CODE)
def test_update_lambda_happycase1(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
    )

    lambda_obj.update()

    sagemaker_session.lambda_client.update_function_code.assert_called_with(
        FunctionName=FUNCTION_NAME,
        ZipFile=ZIPPED_CODE,
    )


@patch("sagemaker.lambda_helper._upload_to_s3", return_value=S3_KEY)
def test_update_lambda_happycase2(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_arn=LAMBDA_ARN,
        execution_role_arn=EXECUTION_ROLE,
        zipped_code_dir=ZIPPED_CODE_DIR,
        s3_bucket=S3_BUCKET,
        handler=HANDLER,
        session=sagemaker_session,
    )

    lambda_obj.update()

    sagemaker_session.lambda_client.update_function_code.assert_called_with(
        FunctionName=LAMBDA_ARN,
        S3Bucket=S3_BUCKET,
        S3Key=S3_KEY,
    )


@patch("sagemaker.lambda_helper._zip_lambda_code", return_value=ZIPPED_CODE)
def test_update_lambda_happycase3(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
        environment={"Name": "my-test-lambda"},
        vpc_config={
            "SubnetIds": ["test-subnet-1"],
            "SecurityGroupIds": ["sec-group-1"],
        },
    )

    lambda_obj.update()

    sagemaker_session.lambda_client.update_function_code.assert_called_with(
        FunctionName=FUNCTION_NAME,
        ZipFile=ZIPPED_CODE,
    )


@patch("sagemaker.lambda_helper._upload_to_s3", return_value=S3_KEY)
def test_update_lambda_s3bucket_not_provided(s3_upload, sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_arn=LAMBDA_ARN,
        execution_role_arn=EXECUTION_ROLE,
        zipped_code_dir=ZIPPED_CODE_DIR,
        handler=HANDLER,
        session=sagemaker_session,
    )

    lambda_obj.update()

    sagemaker_session.lambda_client.update_function_code.assert_called_with(
        FunctionName=LAMBDA_ARN,
        S3Bucket=sagemaker_session.default_bucket(),
        S3Key=s3_upload.return_value,
    )


@patch("sagemaker.lambda_helper._zip_lambda_code", return_value=ZIPPED_CODE)
def test_update_lambda_client_error(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
    )

    sagemaker_session.lambda_client.update_function_code.side_effect = ClientError(
        {"Error": {"Code": "InvalidCodeException", "Message": "Cannot update code"}},
        "UpdateFunction",
    )
    with pytest.raises(ValueError) as error:
        lambda_obj.update()

    assert "Cannot update code" in str(error)


@patch("sagemaker.lambda_helper._zip_lambda_code", return_value=ZIPPED_CODE)
def test_upsert_lambda_happycase1(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
    )

    code = {"ZipFile": ZIPPED_CODE}
    lambda_obj.upsert()

    sagemaker_session.lambda_client.create_function.assert_called_with(
        FunctionName=FUNCTION_NAME,
        Runtime="python3.8",
        Handler=HANDLER,
        Role=EXECUTION_ROLE,
        Code=code,
        Timeout=120,
        MemorySize=128,
        VpcConfig={},
        Environment={},
        Layers=[],
    )


@patch("sagemaker.lambda_helper._zip_lambda_code", return_value=ZIPPED_CODE)
def test_upsert_lambda_happycase2(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
    )

    sagemaker_session.lambda_client.create_function.side_effect = ClientError(
        {
            "Error": {
                "Code": "ResourceConflictException",
                "Message": "Lambda already exists",
            }
        },
        "CreateFunction",
    )

    lambda_obj.upsert()

    sagemaker_session.lambda_client.update_function_code.assert_called_once_with(
        FunctionName=FUNCTION_NAME, ZipFile=ZIPPED_CODE
    )


@patch("sagemaker.lambda_helper._zip_lambda_code", return_value=ZIPPED_CODE)
def test_upsert_lambda_client_error(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(
        function_name=FUNCTION_NAME,
        execution_role_arn=EXECUTION_ROLE,
        script=SCRIPT,
        handler=HANDLER,
        session=sagemaker_session,
    )

    sagemaker_session.lambda_client.create_function.side_effect = ClientError(
        {
            "Error": {
                "Code": "ResourceConflictException",
                "Message": "Lambda already exists",
            }
        },
        "CreateFunction",
    )

    sagemaker_session.lambda_client.update_function_code.side_effect = ClientError(
        {
            "Error": {
                "Code": "ResourceConflictException",
                "Message": "Cannot update code",
            }
        },
        "UpdateFunctionCode",
    )

    with pytest.raises(ValueError) as error:
        lambda_obj.upsert()

    assert "Cannot update code" in str(error)


def test_invoke_lambda_happycase(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(function_arn=LAMBDA_ARN, session=sagemaker_session)
    lambda_obj.invoke()

    sagemaker_session.lambda_client.invoke.assert_called_with(
        FunctionName=LAMBDA_ARN, InvocationType="RequestResponse"
    )


def test_invoke_lambda_client_error(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(function_arn=LAMBDA_ARN, session=sagemaker_session)

    sagemaker_session.lambda_client.invoke.side_effect = ClientError(
        {"Error": {"Code": "InvalidCodeException", "Message": "invoke failed"}},
        "Invoke",
    )
    with pytest.raises(ValueError) as error:
        lambda_obj.invoke()

    assert "invoke failed" in str(error)


def test_delete_lambda_happycase(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(function_arn=LAMBDA_ARN, session=sagemaker_session)
    lambda_obj.delete()
    sagemaker_session.lambda_client.delete_function.assert_called_with(FunctionName=LAMBDA_ARN)


def test_delete_lambda_client_error(sagemaker_session):
    lambda_obj = lambda_helper.Lambda(function_arn=LAMBDA_ARN, session=sagemaker_session)

    sagemaker_session.lambda_client.delete_function.side_effect = ClientError(
        {"Error": {"Code": "Invalid", "Message": "Delete failed"}}, "Invoke"
    )
    with pytest.raises(ValueError) as error:
        lambda_obj.delete()

    assert "Delete failed" in str(error)


def test_upload_to_s3(s3_client):
    key = lambda_helper._upload_to_s3(s3_client, FUNCTION_NAME, ZIPPED_CODE_DIR, S3_BUCKET)
    s3_client.upload_file.assert_called_with(ZIPPED_CODE_DIR, S3_BUCKET, key)
    assert key == S3_KEY


def test_zip_lambda_code():
    code = lambda_helper._zip_lambda_code(os.path.join(DATA_DIR, "dummy_script.py"))
    assert code is not None
