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

import base64
import errno
import json
import logging
import os
import platform
import random
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
from subprocess import Popen
from six.moves.urllib.parse import urlparse
from time import sleep

import yaml

import sagemaker
from sagemaker.utils import get_config_value

CONTAINER_PREFIX = "algo"
DOCKER_COMPOSE_FILENAME = 'docker-compose.yaml'

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class _SageMakerContainer(object):
    """Handle the lifecycle and configuration of a local docker container execution.

    This class is responsible for creating the directories and configuration files that
    the docker containers will use for either training or serving.
    """

    def __init__(self, instance_type, instance_count, image, sagemaker_session=None):
        """Initialize a SageMakerContainer instance

        It uses a :class:`sagemaker.session.Session` for general interaction with user configuration
        such as getting the default sagemaker S3 bucket. However this class does not call any of the
        SageMaker APIs.

        Args:
            instance_type (str): The instance type to use. Either 'local' or 'local_gpu'
            instance_count (int): The number of instances to create.
            image (str): docker image to use.
            sagemaker_session (sagemaker.session.Session): a sagemaker session to use when interacting
                with SageMaker.
        """
        from sagemaker.local.local_session import LocalSession
        self.sagemaker_session = sagemaker_session or LocalSession()
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.image = image
        # Since we are using a single docker network, Generate a random suffix to attach to the container names.
        #  This way multiple jobs can run in parallel.
        suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        self.hosts = ['{}-{}-{}'.format(CONTAINER_PREFIX, i, suffix) for i in range(1, self.instance_count + 1)]
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
        self.container_root = self._create_tmp_folder()
        os.mkdir(os.path.join(self.container_root, 'output'))
        # A shared directory for all the containers. It is only mounted if the training script is
        # Local.
        shared_dir = os.path.join(self.container_root, 'shared')
        os.mkdir(shared_dir)

        data_dir = self._create_tmp_folder()
        volumes = []

        # Set up the channels for the containers. For local data we will
        # mount the local directory to the container. For S3 Data we will download the S3 data
        # first.
        for channel in input_data_config:
            if channel['DataSource'] and 'S3DataSource' in channel['DataSource']:
                uri = channel['DataSource']['S3DataSource']['S3Uri']
            elif channel['DataSource'] and 'FileDataSource' in channel['DataSource']:
                uri = channel['DataSource']['FileDataSource']['FileUri']
            else:
                raise ValueError('Need channel[\'DataSource\'] to have [\'S3DataSource\'] or [\'FileDataSource\']')

            parsed_uri = urlparse(uri)
            key = parsed_uri.path.lstrip('/')

            channel_name = channel['ChannelName']
            channel_dir = os.path.join(data_dir, channel_name)
            os.mkdir(channel_dir)

            if parsed_uri.scheme == 's3':
                bucket_name = parsed_uri.netloc
                self._download_folder(bucket_name, key, channel_dir)
            elif parsed_uri.scheme == 'file':
                path = parsed_uri.path
                volumes.append(_Volume(path, channel=channel_name))
            else:
                raise ValueError('Unknown URI scheme {}'.format(parsed_uri.scheme))

        # If the training script directory is a local directory, mount it to the container.
        training_dir = json.loads(hyperparameters[sagemaker.estimator.DIR_PARAM_NAME])
        parsed_uri = urlparse(training_dir)
        if parsed_uri.scheme == 'file':
            volumes.append(_Volume(parsed_uri.path, '/opt/ml/code'))
            # Also mount a directory that all the containers can access.
            volumes.append(_Volume(shared_dir, '/opt/ml/shared'))

        # Create the configuration files for each container that we will create
        # Each container will map the additional local volumes (if any).
        for host in self.hosts:
            _create_config_file_directories(self.container_root, host)
            self.write_config_files(host, hyperparameters, input_data_config)
            shutil.copytree(data_dir, os.path.join(self.container_root, host, 'input', 'data'))

        compose_data = self._generate_compose_file('train', additional_volumes=volumes)
        compose_command = self._compose()

        _ecr_login_if_needed(self.sagemaker_session.boto_session, self.image)
        _execute_and_stream_output(compose_command)

        s3_artifacts = self.retrieve_artifacts(compose_data)

        # free up the training data directory as it may contain
        # lots of data downloaded from S3. This doesn't delete any local
        # data that was just mounted to the container.
        _delete_tree(data_dir)
        _delete_tree(shared_dir)
        # Also free the container config files.
        for host in self.hosts:
            container_config_path = os.path.join(self.container_root, host)
            _delete_tree(container_config_path)

        self._cleanup()
        # Print our Job Complete line to have a simmilar experience to training on SageMaker where you
        # see this line at the end.
        print('===== Job Complete =====')
        return s3_artifacts

    def serve(self, primary_container):
        """Host a local endpoint using docker-compose.
        Args:
            primary_container (dict): dictionary containing the container runtime settings
                for serving. Expected keys:
                - 'ModelDataUrl' pointing to a local file
                - 'Environment' a dictionary of environment variables to be passed to the hosting container.

        """
        logger.info("serving")

        self.container_root = self._create_tmp_folder()
        logger.info('creating hosting dir in {}'.format(self.container_root))

        model_dir = primary_container['ModelDataUrl']
        if not model_dir.lower().startswith("s3://"):
            for h in self.hosts:
                host_dir = os.path.join(self.container_root, h)
                os.makedirs(host_dir)
                shutil.copytree(model_dir, os.path.join(self.container_root, h, 'model'))

        env_vars = ['{}={}'.format(k, v) for k, v in primary_container['Environment'].items()]

        _ecr_login_if_needed(self.sagemaker_session.boto_session, self.image)

        # If the user script was passed as a file:// mount it to the container.
        script_dir = primary_container['Environment'][sagemaker.estimator.DIR_PARAM_NAME.upper()]
        parsed_uri = urlparse(script_dir)
        volumes = []
        if parsed_uri.scheme == 'file':
            volumes.append(_Volume(parsed_uri.path, '/opt/ml/code'))

        self._generate_compose_file('serve',
                                    additional_env_vars=env_vars,
                                    additional_volumes=volumes)
        compose_command = self._compose()
        self.container = _HostingContainer(compose_command)
        self.container.up()

    def stop_serving(self):
        """Stop the serving container.

        The serving container runs in async mode to allow the SDK to do other tasks.
        """
        if self.container:
            self.container.down()
            self._cleanup()
        # for serving we can delete everything in the container root.
        _delete_tree(self.container_root)

    def retrieve_artifacts(self, compose_data):
        """Get the model artifacts from all the container nodes.

        Used after training completes to gather the data from all the individual containers. As the
        official SageMaker Training Service, it will override duplicate files if multiple containers have
        the same file names.

        Args:
            compose_data(dict): Docker-Compose configuration in dictionary format.

        Returns: Local path to the collected model artifacts.

        """
        # Grab the model artifacts from all the Nodes.
        s3_artifacts = os.path.join(self.container_root, 's3_artifacts')
        os.mkdir(s3_artifacts)

        s3_model_artifacts = os.path.join(s3_artifacts, 'model')
        s3_output_artifacts = os.path.join(s3_artifacts, 'output')
        os.mkdir(s3_model_artifacts)
        os.mkdir(s3_output_artifacts)

        for host in self.hosts:
            volumes = compose_data['services'][str(host)]['volumes']

            for volume in volumes:
                host_dir, container_dir = volume.split(':')
                if container_dir == '/opt/ml/model':
                    self._recursive_copy(host_dir, s3_model_artifacts)
                elif container_dir == '/opt/ml/output':
                    self._recursive_copy(host_dir, s3_output_artifacts)

        return s3_model_artifacts

    def write_config_files(self, host, hyperparameters, input_data_config):
        """Write the config files for the training containers.

        This method writes the hyperparameters, resources and input data configuration files.

        Args:
            host (str): Host to write the configuration for
            hyperparameters (dict): Hyperparameters for training.
            input_data_config (dict): Training input channels to be used for training.

        Returns: None

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
        boto_session = self.sagemaker_session.boto_session

        s3 = boto_session.resource('s3')
        bucket = s3.Bucket(bucket_name)

        for obj_sum in bucket.objects.filter(Prefix=prefix):
            obj = s3.Object(obj_sum.bucket_name, obj_sum.key)
            file_path = os.path.join(target, obj_sum.key[len(prefix) + 1:])

            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            obj.download_file(file_path)

    def _generate_compose_file(self, command, additional_volumes=None, additional_env_vars=None):
        """Writes a config file describing a training/hosting  environment.

        This method generates a docker compose configuration file, it has an entry for each container
        that will be created (based on self.hosts). it calls
        :meth:~sagemaker.local_session.SageMakerContainer._create_docker_host to generate the config
        for each individual container.

        Args:
            command (str): either 'train' or 'serve'
            additional_volumes (list): a list of volumes that will be mapped to the containers
            additional_env_vars (dict): a dictionary with additional environment variables to be
                passed on to the containers.

        Returns: (dict) A dictionary representation of the configuration that was written.

        """
        boto_session = self.sagemaker_session.boto_session
        additional_env_vars = additional_env_vars or []
        additional_volumes = additional_volumes or {}
        environment = []
        optml_dirs = set()

        aws_creds = _aws_credentials(boto_session)
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
            'services': services,
            'networks': {
                'sagemaker-local': {'name': 'sagemaker-local'}
            }

        }

        docker_compose_path = os.path.join(self.container_root, DOCKER_COMPOSE_FILENAME)
        yaml_content = yaml.dump(content, default_flow_style=False)
        logger.info('docker compose file: \n{}'.format(yaml_content))
        with open(docker_compose_path, 'w') as f:
            f.write(yaml_content)

        return content

    def _compose(self, detached=False):
        compose_cmd = 'docker-compose'

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
        optml_volumes = self._build_optml_volumes(host, optml_subdirs)
        optml_volumes.extend(volumes)

        host_config = {
            'image': self.image,
            'stdin_open': True,
            'tty': True,
            'volumes': [v.map for v in optml_volumes],
            'environment': environment,
            'command': command,
            'networks': {
                'sagemaker-local': {
                    'aliases': [host]
                }
            }
        }

        if command == 'serve':
            serving_port = get_config_value('local.serving_port',
                                            self.sagemaker_session.config) or 8080
            host_config.update({
                'ports': [
                    '%s:8080' % serving_port
                ]
            })

        return host_config

    def _create_tmp_folder(self):
        root_dir = get_config_value('local.container_root', self.sagemaker_session.config)
        if root_dir:
            root_dir = os.path.abspath(root_dir)

        dir = tempfile.mkdtemp(dir=root_dir)

        # Docker cannot mount Mac OS /var folder properly see
        # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
        # Only apply this workaround if the user didn't provide an alternate storage root dir.
        if root_dir is None and platform.system() == 'Darwin':
            dir = '/private{}'.format(dir)

        return os.path.abspath(dir)

    def _build_optml_volumes(self, host, subdirs):
        """Generate a list of :class:`~sagemaker.local_session.Volume` required for the container to start.

        It takes a folder with the necessary files for training and creates a list of opt volumes that
        the Container needs to start.

        Args:
            host (str): container for which the volumes will be generated.
            subdirs (list): list of subdirectories that will be mapped. For example: ['input', 'output', 'model']

        Returns: (list) List of :class:`~sagemaker.local_session.Volume`
        """
        volumes = []

        # Ensure that model is in the subdirs
        if 'model' not in subdirs:
            subdirs.add('model')

        for subdir in subdirs:
            host_dir = os.path.join(self.container_root, host, subdir)
            container_dir = '/opt/ml/{}'.format(subdir)
            volume = _Volume(host_dir, container_dir)
            volumes.append(volume)

        return volumes

    def _cleanup(self):
        # we don't need to cleanup anything at the moment
        pass


class _HostingContainer(object):
    def __init__(self, command, startup_delay=5):
        self.command = command
        self.startup_delay = startup_delay
        self.process = None

    def up(self):
        self.process = Popen(self.command)
        sleep(self.startup_delay)

    def down(self):
        self.process.terminate()


class _Volume(object):
    """Represent a Volume that will be mapped to a container.

    """

    def __init__(self, host_dir, container_dir=None, channel=None):
        """Create a Volume instance

        the container path can be provided as a container_dir or as a channel name but not both.
        Args:
            host_dir (str): path to the volume data in the host
            container_dir (str): path inside the container that host_dir will be mapped to
            channel (str): channel name that the host_dir represents. It will be mapped as
                /opt/ml/input/data/<channel> in the container.
        """
        if not container_dir and not channel:
            raise ValueError('Either container_dir or channel must be declared.')

        if container_dir and channel:
            raise ValueError('container_dir and channel cannot be declared together.')

        self.container_dir = container_dir if container_dir else os.path.join('/opt/ml/input/data', channel)
        self.host_dir = host_dir
        if platform.system() == 'Darwin' and host_dir.startswith('/var'):
            self.host_dir = os.path.join('/private', host_dir)

        self.map = '{}:{}'.format(self.host_dir, self.container_dir)


def _execute_and_stream_output(cmd):
    """Execute a command and stream the output to stdout

    Args:
        cmd(str or List): either a string or a List (in Popen Format) with the command to execute.

    Returns (int): command exit code
    """
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
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
        cmd = shlex.split(cmd)

    success = True
    try:
        output = subprocess.check_output(cmd, *popenargs, **kwargs)
    except subprocess.CalledProcessError as e:
        output = e.output
        success = False

    output = output.decode("utf-8")
    if not success:
        logger.error("Command output: %s" % output)
        raise Exception("Failed to run %s" % ",".join(cmd))

    return output


def _create_config_file_directories(root, host):
    for d in ['input', 'input/config', 'output', 'model']:
        os.makedirs(os.path.join(root, host, d))


def _delete_tree(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        # on Linux, when docker writes to any mounted volume, it uses the container's user. In most cases
        # this is root. When the container exits and we try to delete them we can't because root owns those
        # files. We expect this to happen, so we handle EACCESS. Any other error we will raise the
        # exception up.
        if exc.errno == errno.EACCES:
            logger.warning("Failed to delete: %s Please remove it manually." % path)
        else:
            logger.error("Failed to delete: %s" % path)
            raise


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


def _ecr_login_if_needed(boto_session, image):
    # Only ECR images need login
    if not ('dkr.ecr' in image and 'amazonaws.com' in image):
        return

    # do we have the image?
    if _check_output('docker images -q %s' % image).strip():
        return

    if not boto_session:
        raise RuntimeError('A boto session is required to login to ECR.'
                           'Please pull the image: %s manually.' % image)

    ecr = boto_session.client('ecr')
    auth = ecr.get_authorization_token(registryIds=[image.split('.')[0]])
    authorization_data = auth['authorizationData'][0]

    raw_token = base64.b64decode(authorization_data['authorizationToken'])
    token = raw_token.decode('utf-8').strip('AWS:')
    ecr_url = auth['authorizationData'][0]['proxyEndpoint']

    cmd = "docker login -u AWS -p %s %s" % (token, ecr_url)
    subprocess.check_output(cmd, shell=True)
