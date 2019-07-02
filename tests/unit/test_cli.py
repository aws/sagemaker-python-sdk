# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
import sagemaker.cli.main as cli
from mock import patch

COMMON_ARGS = (
    "--role-name myrole --data mydata --script myscript --job-name myjob --bucket-name mybucket "
    + "--python py3 --instance-type myinstance --instance-count 2"
)

TRAIN_ARGS = "--hyperparameters myhyperparameters.json"

LOG_ARGS = "--log-level debug --botocore-log-level debug"

HOST_ARGS = "--env ENV1=env1 ENV2=env2"


def assert_common_defaults(args):
    assert args.data == "./data"
    assert args.script == "./script.py"
    assert args.job_name is None
    assert args.bucket_name is None
    assert args.python == "py2"
    assert args.instance_type == "ml.m4.xlarge"
    assert args.instance_count == 1
    assert args.log_level == "info"
    assert args.botocore_log_level == "warning"


def assert_common_non_defaults(args):
    assert args.data == "mydata"
    assert args.script == "myscript"
    assert args.job_name == "myjob"
    assert args.bucket_name == "mybucket"
    assert args.role_name == "myrole"
    assert args.python == "py3"
    assert args.instance_type == "myinstance"
    assert args.instance_count == 2
    assert args.log_level == "debug"
    assert args.botocore_log_level == "debug"


def assert_train_defaults(args):
    assert args.hyperparameters == "./hyperparameters.json"


def assert_train_non_defaults(args):
    assert args.hyperparameters == "myhyperparameters.json"


def assert_host_defaults(args):
    assert args.env == []


def assert_host_non_defaults(args):
    assert args.env == ["ENV1=env1", "ENV2=env2"]


def test_args_mxnet_train_defaults():
    args = cli.parse_arguments("mxnet train --role-name role".split())
    assert_common_defaults(args)
    assert_train_defaults(args)
    assert args.func.__module__ == "sagemaker.cli.mxnet"
    assert args.func.__name__ == "train"


def test_args_mxnet_train_non_defaults():
    args = cli.parse_arguments(
        "{} mxnet train --role-name role {} {}".format(LOG_ARGS, COMMON_ARGS, TRAIN_ARGS).split()
    )
    assert_common_non_defaults(args)
    assert_train_non_defaults(args)
    assert args.func.__module__ == "sagemaker.cli.mxnet"
    assert args.func.__name__ == "train"


def test_args_mxnet_host_defaults():
    args = cli.parse_arguments("mxnet host --role-name role".split())
    assert_common_defaults(args)
    assert_host_defaults(args)
    assert args.func.__module__ == "sagemaker.cli.mxnet"
    assert args.func.__name__ == "host"


def test_args_mxnet_host_non_defaults():
    args = cli.parse_arguments(
        "{} mxnet host --role-name role {} {}".format(LOG_ARGS, COMMON_ARGS, HOST_ARGS).split()
    )
    assert_common_non_defaults(args)
    assert_host_non_defaults(args)
    assert args.func.__module__ == "sagemaker.cli.mxnet"
    assert args.func.__name__ == "host"


def test_args_tensorflow_train_defaults():
    args = cli.parse_arguments("tensorflow train --role-name role".split())
    assert_common_defaults(args)
    assert_train_defaults(args)
    assert args.training_steps is None
    assert args.evaluation_steps is None
    assert args.func.__module__ == "sagemaker.cli.tensorflow"
    assert args.func.__name__ == "train"


def test_args_tensorflow_train_non_defaults():
    args = cli.parse_arguments(
        "{} tensorflow train --role-name role --training-steps 10 --evaluation-steps 5 {} {}".format(
            LOG_ARGS, COMMON_ARGS, TRAIN_ARGS
        ).split()
    )
    assert_common_non_defaults(args)
    assert_train_non_defaults(args)
    assert args.training_steps == 10
    assert args.evaluation_steps == 5
    assert args.func.__module__ == "sagemaker.cli.tensorflow"
    assert args.func.__name__ == "train"


def test_args_tensorflow_host_defaults():
    args = cli.parse_arguments("tensorflow host --role-name role".split())
    assert_common_defaults(args)
    assert_host_defaults(args)
    assert args.func.__module__ == "sagemaker.cli.tensorflow"
    assert args.func.__name__ == "host"


def test_args_tensorflow_host_non_defaults():
    args = cli.parse_arguments(
        "{} tensorflow host --role-name role {} {}".format(LOG_ARGS, COMMON_ARGS, HOST_ARGS).split()
    )
    assert_common_non_defaults(args)
    assert_host_non_defaults(args)
    assert args.func.__module__ == "sagemaker.cli.tensorflow"
    assert args.func.__name__ == "host"


def test_args_invalid_framework():
    with pytest.raises(SystemExit):
        cli.parse_arguments("fakeframework train --role-name role".split())


def test_args_invalid_subcommand():
    with pytest.raises(SystemExit):
        cli.parse_arguments("mxnet drain".split())


def test_args_invalid_args():
    with pytest.raises(SystemExit):
        cli.parse_arguments("tensorflow train --role-name role --notdata foo".split())


def test_args_invalid_mxnet_python():
    with pytest.raises(SystemExit):
        cli.parse_arguments("mxnet train --role-name role nython py2".split())


def test_args_invalid_host_args_in_train():
    with pytest.raises(SystemExit):
        cli.parse_arguments("mxnet train --role-name role --env FOO=bar".split())


def test_args_invalid_train_args_in_host():
    with pytest.raises(SystemExit):
        cli.parse_arguments("tensorflow host --role-name role --hyperparameters foo.json".split())


@patch("sagemaker.mxnet.estimator.MXNet")
@patch("sagemaker.Session")
def test_mxnet_train(session, estimator):
    args = cli.parse_arguments("mxnet train --role-name role".split())
    args.func(args)
    session.return_value.upload_data.assert_called()
    estimator.assert_called()
    estimator.return_value.fit.assert_called()


@patch("sagemaker.mxnet.model.MXNetModel")
@patch("sagemaker.cli.common.HostCommand.upload_model")
@patch("sagemaker.Session")
def test_mxnet_host(session, upload_model, model):
    args = cli.parse_arguments("mxnet host --role-name role".split())
    args.func(args)
    session.assert_called()
    upload_model.assert_called()
    model.assert_called()
    model.return_value.deploy.assert_called()
