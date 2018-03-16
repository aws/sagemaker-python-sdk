# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from subprocess import Popen, STDOUT, CalledProcessError
from time import sleep

import yaml

CONTAINER_PREFIX = "algo"
DOCKER_COMPOSE_FILENAME = 'docker-compose.yaml'

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SageMakerContainer(object):

    def __init__(self, instance_type, instance_count, image, sagemaker_session=None):
        from sagemaker.local_session import LocalSession
        self.sagemaker_session = sagemaker_session or LocalSession()
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.image = image
        self.hosts = ['{}-{}'.format(CONTAINER_PREFIX, i) for i in range(1, self.instance_count + 1)]
        self.container_root = None
        self.container = None

    def train(self, input_data_config, hyperparameters):
        """Run a training job locally using docker-compose.
        Args:
            input_data_config (dict): The Input Data Configuration, this contains data such as the
                channels to be used for training.
            hyperparameters (dict): The HyperParameters for the training job.

        Returns (str): Location of the trained model.

        """
        self.container_root = _create_tmp_folder()

        data_dir = tempfile.mkdtemp()
        bucket_name = self.sagemaker_session.default_bucket()
        bucket_name_with_prefix = 's3://{}/'.format(bucket_name)
        volumes = []

        # Set up the channels for the containers. For local data we will
        # mount the local directory to the container. For S3 Data we will download the S3 data
        # first.
        for channel in input_data_config:
            uri = channel['DataSource']['S3DataSource']['S3Uri']
            key = uri[len(bucket_name_with_prefix):]

            channel_name = channel['ChannelName']
            channel_dir = os.path.join(data_dir, channel_name)
            os.mkdir(channel_dir)

            if uri.lower().startswith("s3://"):
                self._download_folder(bucket_name, key, channel_dir)
            else:
                volumes.append(Volume(uri, channel=channel_name))

        # Create the docker compose files for each container that we will create
        # Each container will map the additional local volumes (if any).
        for host in self.hosts:
            _prepare_config_file_directory(self.container_root, host)
            self.write_config_files(host, hyperparameters, input_data_config)
            shutil.copytree(data_dir, os.path.join(self.container_root, host, 'input', 'data'))

        compose_data = self._generate_compose_file('train', additional_volumes=volumes)
        compose_command = self._compose()
        _stream_output(compose_command)
        _cleanup()

        s3_model_artifacts = self.retrieve_model_artifacts(compose_data)
        return s3_model_artifacts

    def serve(self, primary_container):
        """Host a local endpoint using docker-compose.
        Args:
            primary_container (dict): dictionary containing the container runtime settings
                for serving. Expected keys:
                - 'ModelDataUrl' pointing to either an s3 location or a local file
                - 'Environment' a dictionary of environment variables to be passed to the hosting container.

        """
        logger.info("serving")

        self.container_root = _create_tmp_folder()
        logger.info('creating hosting dir in {}'.format(self.container_root))

        model_dir = primary_container['ModelDataUrl']
        if not model_dir.lower().startswith("s3://"):
            for h in self.hosts:
                host_dir = os.path.join(self.container_root, h)
                os.makedirs(host_dir)
                shutil.copytree(model_dir, os.path.join(self.container_root, h, 'model'))

        env_vars = ['{}={}'.format(k, v) for k, v in primary_container['Environment'].items()]

        self._generate_compose_file('serve', additional_env_vars=env_vars)
        compose_command = self._compose()
        self.container = _Container(self.container_root, compose_command)
        self.container.up()

    def stop_serving(self):
        """Stop the serving container.

        The serving container runs in async mode to allow the SDK to do other tasks.
        """
        if self.container:
            self.container.down()

    def retrieve_model_artifacts(self, compose_data):
        """Get the model artifacts from all the container nodes.

        Used after training completes to gather the data from all the individual containers. As the
        official SageMaker Training Service, it will override duplicate files if multiple containers have
        the same file names.

        Args:
            compose_data(dict): Docker-Compose configuration in dictionary format.

        Returns: Local path to the collected model artifacts.

        """
        # Grab the model artifacts from all the Nodes.
        s3_model_artifacts = os.path.join(self.container_root, 's3_model_artifacts')
        os.mkdir(s3_model_artifacts)

        for host in self.hosts:
            volumes = compose_data['services'][str(host)]['volumes']

            for volume in volumes:
                container_dir, host_dir = volume.split(':')
                if host_dir == '/opt/ml/model':
                    self._recursive_copy(container_dir, s3_model_artifacts)

        return s3_model_artifacts

    def write_config_files(self, host, hyperparameters, input_data_config):
        """Write the config files for the training containers.

        This method writes the hyperparameters, resources and input data configuration files.

        Args:
            host (str): Host to write the configuration for
            hyperparameters (dict): Hyperparameters for training.
            input_data_config (dict): Training input channels to be used for training.

        Returns:

        """

        config_path = os.path.join(self.container_root, host, 'input', 'config')

        resource_config = {
            'current_host': host,
            'hosts': self.hosts
        }

        json_input_data_config = {
            c['ChannelName']: {'ContentType': 'application/octet-stream'} for c in input_data_config
        }

        _write_json_file(os.path.join(config_path, 'hyperparameters.json'), hyperparameters)
        _write_json_file(os.path.join(config_path, 'resourceconfig.json'), resource_config)
        _write_json_file(os.path.join(config_path, 'inputdataconfig.json'), json_input_data_config)

    def _recursive_copy(self, src, dst):
        for root, dirs, files in os.walk(src):
            root = os.path.relpath(root, src)
            current_path = os.path.join(src, root)
            target_path = os.path.join(dst, root)

            for file in files:
                shutil.copy(os.path.join(current_path, file), os.path.join(target_path, file))
            for dir in dirs:
                new_dir = os.path.join(target_path, dir)
                if not os.path.exists(new_dir):
                    os.mkdir(os.path.join(target_path, dir))

    def _download_folder(self, bucket_name, prefix, target):
        session = self.sagemaker_session.boto_session

        s3 = session.resource('s3')
        bucket = s3.Bucket(bucket_name)

        for obj_sum in bucket.objects.filter(Prefix=prefix):

            obj = s3.Object(obj_sum.bucket_name, obj_sum.key)
            file_path = os.path.join(target, obj_sum.key[len(prefix) + 1:])

            try:
                os.makedirs(os.path.dirname(file_path))
            except os.error:
                pass

            obj.download_file(file_path)

    def _generate_compose_file(self, command, additional_volumes=None, additional_env_vars=None):
        session = self.sagemaker_session.boto_session
        additional_env_vars = additional_env_vars or []
        additional_volumes = additional_volumes or {}
        environment = []
        optml_dirs = set()

        aws_creds = _aws_credentials(session)
        if aws_creds is not None:
            environment.extend(aws_creds)

        environment.extend(additional_env_vars)

        if command == 'train':
            optml_dirs = {'output', 'input'}

        services = {
            h: self._create_docker_host(h, environment, optml_dirs,
                                        command, additional_volumes) for h in self.hosts
        }

        content = {
            # Some legacy hosts only support the 2.1 format.
            'version': '2.1',
            'services': services
        }

        filename = os.path.join(self.container_root, DOCKER_COMPOSE_FILENAME)
        yaml_content = yaml.dump(content, default_flow_style=False)
        logger.info('docker compose file: \n{}'.format(yaml_content))
        with open(filename, 'w') as f:
            f.write(yaml_content)

        return content

    def _compose(self, detached=False):
        compose_cmd = 'nvidia-docker-compose' if self.instance_type == "local_gpu" else 'docker-compose'

        command = [
            compose_cmd,
            '-f',
            os.path.join(self.container_root, DOCKER_COMPOSE_FILENAME),
            'up',
            '--build',
            '--abort-on-container-exit'
        ]

        if detached:
            command.append('-d')

        logger.info('docker command: {}'.format(' '.join(command)))
        return command

    def _create_docker_host(self, host, environment, optml_subdirs, command, volumes):
        optml_volumes = self._optml_volumes(host, optml_subdirs)
        optml_volumes.extend(volumes)

        host_config = {
            'image': self.image,
            'stdin_open': True,
            'tty': True,
            'volumes': [v.map for v in optml_volumes],
            'environment': environment,
            'command': command
        }

        if command == 'serve':
            host_config.update({
                'ports': [
                    '8080:8080'
                ]
            })

        return host_config

    def _optml_volumes(self, host, subdirs, single_model_dir=False):
        """
        It takes a folder with the necessary files for training and creates a list of opt volumes that
        the Container needs to start.
        If args.single_model_dir is True, all the hosts will point the opt/ml/model subdir to the first container.
        That is useful for distributed training, so all the containers can read and write the same checkpoints.

        :param opt_root_folder: root folder with the contents to be mapped to the container
        :param host: host name of the container
        :param subdirs: list of subdirs that will be mapped. Example: ['input', 'output', 'model']
        :return:
        """
        volumes = []

        # If it is single mode dir we want to map the same model dir and share between hosts
        if single_model_dir:
            host_dir = os.path.join(self.container_root, 'algo-1/model')
            volume = Volume(host_dir, '/opt/ml/model')
            volumes.append(volume)
        else:
            # else we want to add model to the list of subdirs so it will be created for each container.
            subdirs.add('model')

        for subdir in subdirs:
            host_dir = os.path.join(self.container_root, host, subdir)
            container_dir = '/opt/ml/{}'.format(subdir)
            volume = Volume(host_dir, container_dir)
            volumes.append(volume)

        return volumes


class _Container(object):
    def __init__(self, tmpdir, command, startup_delay=5):
        self.command = command
        self.compose_file = os.path.join(tmpdir, DOCKER_COMPOSE_FILENAME)
        self.startup_delay = startup_delay
        self._process = None

    def up(self):
        self._process = Popen(self.command)
        sleep(self.startup_delay)

    def down(self):
        self._process.terminate()
        _cleanup()


class Volume(object):

    def __init__(self, host_dir, container_dir=None, channel=None):
        # that is necessary because docker cannot mount mac temp folders.
        if not container_dir and not channel:
            raise ValueError('Either container_dir or channel must be declared.')

        if container_dir and channel:
            raise ValueError('container_dir and channel cannot be declared together.')

        self.container_dir = container_dir if container_dir else os.path.join('/opt/ml/input/data', channel)

        self.host_dir = os.path.join('/private', host_dir) if host_dir.startswith('/var') else host_dir

        self.map = '{}:{}'.format(self.host_dir, self.container_dir)


def _cleanup():
    _chain_docker_cmds('docker images -f dangling=true -q', 'docker rmi -f')
    _check_output('docker network prune -f'.split(' '))


def _chain_docker_cmds(cmd, cmd2):
    docker_tags = _check_output(cmd).split('\n')

    if any(docker_tags):
        try:
            _check_output(cmd2.split() + docker_tags, stderr=STDOUT)
        except CalledProcessError:
            pass


def _stream_output(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    exit_code = None
    while exit_code is None:
        stdout = process.stdout.readline().decode("utf-8")
        sys.stdout.write(stdout)

        exit_code = process.poll()

    if exit_code != 0:
        raise Exception("Failed to run %s, exit code: %s" % (",".join(cmd), exit_code))

    return exit_code


def _check_output(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")

    success = True
    try:
        output = subprocess.check_output(cmd, *popenargs, **kwargs)
    except subprocess.CalledProcessError as e:
        output = e.output
        success = False

    output = output.decode("utf-8")
    if not success:
        raise Exception("Failed to run %s" % ",".join(cmd))

    return output


def _create_tmp_folder():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    return os.path.abspath(dir)


def _prepare_config_file_directory(root, host):
    for d in ['input', 'input/config', 'output', 'model']:
        os.makedirs(os.path.join(root, host, d))


def _aws_credentials(session):
    try:
        creds = session.get_credentials()
        access_key = creds.access_key
        secret_key = creds.secret_key

        # if there is a Token as part of the credentials, it is not safe to
        # pass them as environment variables because the Token is not static, this is the case
        # when running under an IAM Role in EC2 for example. By not passing credentials the
        # SDK in the container will look for the credentials in the EC2 Metadata Service.
        if creds.token is None:
            return [
                'AWS_ACCESS_KEY_ID=%s' % (str(access_key)),
                'AWS_SECRET_ACCESS_KEY=%s' % (str(secret_key))
            ]
        else:
            return None
    except Exception as e:
        logger.info('Could not get AWS creds: %s' % e)

    return None


def _write_json_file(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f)
