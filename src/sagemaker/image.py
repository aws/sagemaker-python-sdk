import json
import logging
import platform
import shutil
import subprocess
import tempfile
from subprocess import Popen, STDOUT, CalledProcessError
from time import sleep

import boto3
import docker
import os
import yaml
import sys
from os.path import join, abspath, dirname

CONTAINER_PREFIX = "algo"
DOCKER_COMPOSE_FILENAME = 'docker-compose.yaml'

client = docker.from_env()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def train(algorithm_specification, input_data_config, resource_config, hyperparameters, session=None):
    logger.info("training")

    session = session or boto3.Session()

    image_name = algorithm_specification['TrainingImage']
    instance_type = resource_config['InstanceType']
    instance_count = resource_config['InstanceCount']

    hosts = _host_names(instance_count)
    master = hosts[0]

    optml = _opt_folder()
    tmpdir = abspath(optml)
    data_dir = tempfile.mkdtemp()
    volumes = []

    for channel in input_data_config:
        uri = channel['DataSource']['S3DataSource']['S3Uri']

        bucket_name = _default_bucket(session)
        bucket_name_with_prefix = 'S3://{}/'.format(bucket_name)
        key = uri[len(bucket_name_with_prefix):]

        channel_name = channel['ChannelName']
        channel_dir = join(data_dir, channel_name)
        os.mkdir(channel_dir)

        if _is_s3_path(uri):
            _download_folder(bucket_name, key, channel_dir, session)
        else:
            volumes.append(Volume(uri, channel=channel_name))

    for host in hosts:
        for d in ['input', 'input/config', 'output', 'model']:
            os.makedirs(join(tmpdir, host, d))

        _write_config_files(host, hosts, hyperparameters, input_data_config, tmpdir)

        shutil.copytree(data_dir, join(tmpdir, host, 'input', 'data'))

    content = _compose_info('train', tmpdir, hosts, image_name, additional_volumes=volumes)

    _write_compose_file(content, tmpdir)

    logger.info("training dir: \n{}".format(_check_output(['ls', '-lR', tmpdir])))

    command = _compose(tmpdir, instance_type)
    _stream_output(command)
    _cleanup()

    # Grab the model artifacts from the master node [ This works for TF, but we need to revisit it
    # for MXNet and possibly other frameworks ].
    s3_model_artifacts = join(tmpdir, 's3_model_artifacts')
    volumes = content['services'][str(master)]['volumes']

    for volume in volumes:
        container_dir, host_dir = volume.split(':')
        if host_dir == '/opt/ml/model':
            shutil.copytree(container_dir, s3_model_artifacts)

    return s3_model_artifacts


def serve(primary_container, production_variant, session=None):
    logger.info("serving")
    instance_type = production_variant['InstanceType']
    instance_count = production_variant['InitialInstanceCount']

    optml = _opt_folder()
    tmpdir = os.path.abspath(optml)
    logger.info('creating hosting dir in {}'.format(tmpdir))

    hosts = _host_names(instance_count)
    logger.info('creating hosts: {}'.format(hosts))

    model_dir = primary_container['ModelDataUrl']
    if not _is_s3_path(model_dir):
        for h in hosts:
            host_dir = os.path.join(tmpdir, h)
            os.makedirs(host_dir)
            shutil.copytree(model_dir, os.path.join(tmpdir, h, 'model'))

    env_vars = ['{}={}'.format(k, v) for k, v in primary_container['Environment'].items()]

    content = _compose_info('serve', tmpdir, hosts, primary_container['Image'], additional_env_vars=env_vars)

    _write_compose_file(content, tmpdir)

    command = _compose(tmpdir, instance_type)
    return _Container(tmpdir, command)


def _is_s3_path(uri):
    return uri.startswith('S3://') or uri.startswith('s3://')


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


CYAN_COLOR = '\033[36m'
END_COLOR = '\033[0m'


def _check_call(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    _print_cmd(cmd, *popenargs, **kwargs)
    return subprocess.check_call(cmd, *popenargs, **kwargs)


def _stream_output(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    #_print_cmd(cmd, *popenargs, **kwargs)
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
    _print_cmd(cmd, *popenargs, **kwargs)
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


def _print_cmd(cmd, *popenargs, **kwargs):
    logger.info('executing docker command: {}{}{}'.format(CYAN_COLOR, ' '.join(cmd), END_COLOR))
    if 'cwd' in kwargs:
        logger.info('in the folder {}'.format(CYAN_COLOR, ' '.join(kwargs['cwd']), END_COLOR))
    # sys.stdout.flush()


def _download_folder(bucket_name, prefix, target, session=None):
    session = session or boto3.Session()

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    for obj_sum in bucket.objects.filter(Prefix=prefix):

        obj = s3.Object(obj_sum.bucket_name, obj_sum.key)

        file_path = join(target, obj_sum.key[len(prefix) + 1:])

        try:
            os.makedirs(dirname(file_path))
        except:
            pass

        try:
            obj.download_file(file_path)
        except OSError as e:
            if e.errno == 21: # it is a directory
                pass


def _compose(tmpdir, instance_type, detached=False):
    compose_cmd = 'nvidia-docker-compose' if instance_type[3] in ['g', 'p'] else 'docker-compose'

    command = [
        compose_cmd,
        '-f',
        join(tmpdir, DOCKER_COMPOSE_FILENAME),
        'up',
        '--build',
        '--abort-on-container-exit'
    ]

    if detached:
        command.append('-d')

    logger.info('docker command: {}'.format(' '.join(command)))

    return command


def _shutdown(compose_file):
    logger.info("shutting down")
    _check_call(['docker-compose', '-f', compose_file, 'down'])


def _write_compose_file(content, tmpdir):
    filename = join(tmpdir, DOCKER_COMPOSE_FILENAME)
    content = yaml.dump(content, default_flow_style=False)
    logger.info('docker compose file: \n{}'.format(content))
    with open(filename, 'w') as f:
        f.write(content)


def _compose_info(command, tmpdir, hosts, image, additional_volumes=None, additional_env_vars=None):
    environment = []
    additional_env_vars = additional_env_vars or []
    additional_volumes = additional_volumes or {}

    session = boto3.Session()

    optml_dirs = set()
    if command == 'train':
        optml_dirs = {'output', 'input'}

    environment.extend(_aws_credentials(session))

    environment.extend(additional_env_vars)

    services = {h: _create_docker_host(tmpdir, h, image, environment, optml_dirs, command, additional_volumes)
                for h in hosts}

    content = {
        # docker version on ACC hosts only supports compose 2.1 format
        'version': '2.1',
        'services': services
    }

    return content


def _aws_credentials(session):
    try:
        creds = session.get_credentials()
        access_key = creds.access_key
        secret_key = creds.secret_key

        return [
            'AWS_ACCESS_KEY_ID=%s' % (str(access_key)),
            'AWS_SECRET_ACCESS_KEY=%s' % (str(secret_key))
        ]
    except Exception as e:
        logger.info('Could not get AWS creds: %s' % e)

    return []


def _create_docker_host(tmpdir, host, image, environment, optml_subdirs, command, volumes):
    optml_volumes = _optml_volumes(tmpdir, host, optml_subdirs)
    optml_volumes.extend(volumes)

    host_config = {
        'image': image,
        'stdin_open': True,
        'tty': True,
        'volumes': [v.map for v in optml_volumes],
        'environment': environment,
        'command': command,
    }

    if command == 'serve':
        host_config.update({
            'ports': [
                '8080:8080'
            ]
        })

    return host_config


class Volume(object):

    def __init__(self, host_dir, container_dir=None, channel=None):
        # that is necessary because docker cannot mount mac temp folders.
        if not container_dir and not channel:
            raise ValueError('Either container_dir or channel must be declared.')

        if container_dir and channel:
            raise ValueError('container_dir and channel cannot be declared together.')

        self.container_dir = container_dir if container_dir else join('/opt/ml/input/data', channel)

        self.host_dir = join('/private', host_dir) if host_dir.startswith('/var') else host_dir

        self.map = '{}:{}'.format(self.host_dir, self.container_dir)


def _optml_volumes(opt_root_folder, host, subdirs, single_model_dir=False):
    """
    It takes a folder with the necessary files for training and creates a list of opt volumes that
    the Container needs to start.
    If args.single_model_dir is True, all the hosts will point the opt/ml/model subdir to the first container. That is
    useful for distributed training, so all the containers can read and write the same checkpoints.

    :param opt_root_folder: root folder with the contents to be mapped to the container
    :param host: host name of the container
    :param subdirs: list of subdirs that will be mapped. Example: ['input', 'output', 'model']
    :return:
    """
    volumes = []

    # If it is single mode dir we want to map the same model dir and share between hosts
    if single_model_dir:
        host_dir = join(opt_root_folder, 'algo-1/model')
        volume = Volume(host_dir, '/opt/ml/model')
        volumes.append(volume)
    else:
        # else we want to add model to the list of subdirs so it will be created for each container.
        subdirs.add('model')

    for subdir in subdirs:
        host_dir = join(opt_root_folder, host, subdir)
        container_dir = '/opt/ml/{}'.format(subdir)
        volume = Volume(host_dir, container_dir)
        volumes.append(volume)

    return volumes


def _default_bucket(boto_session):
    account = boto_session.client('sts').get_caller_identity()['Account']
    region = boto_session.region_name
    bucket = 'sagemaker-{}-{}'.format(region, account)

    return bucket


def _opt_folder():
    tmp = tempfile.mkdtemp()
    os.mkdir(join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    opt_ml_dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    return opt_ml_dir


def _write_json_file(tmpdir, host, filename, content):
    folder = join(tmpdir, host, 'input', 'config', filename)

    with open(folder, 'w') as f:
        json.dump(content, f)


def _host_names(cluster_size):
    return ['{}-{}'.format(CONTAINER_PREFIX, i) for i in range(1, cluster_size + 1)]


def _write_config_files(host, hosts, hyperparameters, input_data_config, tmpdir):
    _write_json_file(tmpdir, host, 'hyperparameters.json', hyperparameters)
    content = {
        'current_host': host,
        'hosts': hosts
    }
    _write_json_file(tmpdir, host, 'resourceconfig.json', content)

    _write_json_file(tmpdir, host, 'inputdataconfig.json', _format_input_data_config(input_data_config))


def _format_input_data_config(input_data_config):
    channel_names = [channel['ChannelName'] for channel in input_data_config]
    config = {c: {'ContentType': 'application/octet-stream'} for c in channel_names}
    logger.info('input data config: {}'.format(config))
    return config
