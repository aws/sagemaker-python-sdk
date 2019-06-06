# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import base64
import collections
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import sagemaker

Request = collections.namedtuple('Request', ['input', 'content_type', 'accept', 'response'])


def build(expected_hyperparameters=None,
          expected_inputdataconfig=None,
          expected_hosts=None,
          expected_envvars=None,
          expected_requests=None,
          name='sagemaker-dummy-container'):
    with _tmpdir() as tmp:
        container_dir = os.path.join(os.path.dirname(__file__), 'dummy_container')
        tmp_container_dir = os.path.join(tmp, 'dummy_container')

        shutil.copytree(container_dir, tmp_container_dir)

        assertions = {}
        if expected_hyperparameters:
            assertions['expected_hyperparameters'] = expected_hyperparameters

        if expected_inputdataconfig:
            assertions['expected_inputdataconfig'] = expected_inputdataconfig

        if expected_hosts:
            assertions['expected_hosts'] = expected_hosts

        if expected_envvars:
            assertions['expected_envvars'] = expected_envvars

        if expected_requests:
            assertions['expected_requests'] = {
                r.input: {
                    'content_type': r.content_type,
                    'accept': r.accept,
                    'response': r.response
                } for r in expected_requests
            }

        with open(os.path.join(tmp_container_dir, 'assert.json'), mode='w') as f:
            json.dump(assertions, f)

        _execute(['docker', 'build', '-t', name, '.'], cwd=tmp_container_dir)
        return name


def build_and_push(expected_hyperparameters=None,
                   expected_inputdataconfig=None,
                   expected_hosts=None,
                   expected_envvars=None,
                   expected_requests=None,
                   sagemaker_session=None):
    name = build(expected_hyperparameters, expected_inputdataconfig, expected_hosts,
                 expected_envvars, expected_requests)

    if isinstance(sagemaker_session, sagemaker.LocalSession):
        return name
    else:
        boto_session = sagemaker_session.boto_session
        aws_account = boto_session.client("sts").get_caller_identity()['Account']
        aws_region = boto_session.region_name
        ecr_client = boto_session.client('ecr', region_name=aws_region)

        _create_ecr_repo(ecr_client, 'sagemaker-dummy-container')
        _ecr_login(ecr_client, aws_account)

        return _push(aws_account, aws_region, name)


def _push(aws_account, aws_region, tag):
    ecr_repo = '%s.dkr.ecr.%s.amazonaws.com' % (aws_account, aws_region)
    ecr_tag = '%s/%s' % (ecr_repo, tag)
    _execute(['docker', 'tag', tag, ecr_tag])
    print("Pushing docker image to ECR repository %s/%s\n" % (ecr_repo, tag))
    _execute(['docker', 'push', ecr_tag])
    return ecr_tag


def _create_ecr_repo(ecr_client, repository_name):
    try:

        ecr_client.describe_repositories(repositoryNames=[repository_name])['repositories']

    except ecr_client.exceptions.RepositoryNotFoundException:

        ecr_client.create_repository(repositoryName=repository_name)

        print("Created new ECR repository: %s" % repository_name)


def _ecr_login(ecr_client, aws_account):
    auth = ecr_client.get_authorization_token(registryIds=[aws_account])
    authorization_data = auth['authorizationData'][0]

    raw_token = base64.b64decode(authorization_data['authorizationToken'])
    token = raw_token.decode('utf-8').strip('AWS:')
    ecr_url = auth['authorizationData'][0]['proxyEndpoint']

    cmd = ['docker', 'login', '-u', 'AWS', '-p', token, ecr_url]
    _execute(cmd)


def _execute(command, cwd=None):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=cwd)
    try:
        _stream_output(process)
    except RuntimeError as e:
        # _stream_output() doesn't have the command line. We will handle the exception
        # which contains the exit code and append the command line to it.
        msg = "Failed to run: %s, %s" % (command, str(e))
        raise RuntimeError(msg)


def _stream_output(process):
    """Stream the output of a process to stdout
    This function takes an existing process that will be polled for output. Only stdout
    will be polled and sent to sys.stdout.
    Args:
        process(subprocess.Popen): a process that has been started with
            stdout=PIPE and stderr=STDOUT
    Returns (int): process exit code
    """
    exit_code = None

    while exit_code is None:
        stdout = process.stdout.readline().decode("utf-8")
        sys.stdout.write(stdout)
        exit_code = process.poll()

    if exit_code != 0:
        raise RuntimeError("Process exited with code: %s" % exit_code)


@contextlib.contextmanager
def _tmpdir(suffix='', prefix='tmp', dir=None):  # type: (str, str, str) -> None
    """Create a temporary directory with a context manager. The file is deleted when the context exits.
    The prefix, suffix, and dir arguments are the same as for mkstemp().
    Args:
        suffix (str):  If suffix is specified, the file name will end with that suffix, otherwise there will be no
                        suffix.
        prefix (str):  If prefix is specified, the file name will begin with that prefix; otherwise,
                        a default prefix is used.
        dir (str):  If dir is specified, the file will be created in that directory; otherwise, a default directory is
                        used.
    Returns:
        str: path to the directory
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    yield tmp
    shutil.rmtree(tmp)
