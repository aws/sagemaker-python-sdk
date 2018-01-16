# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
import logging
import sys

import sagemaker
import sagemaker.cli.mxnet
import sagemaker.cli.tensorflow

logger = logging.getLogger(__name__)

DEFAULT_LOG_LEVEL = 'info'
DEFAULT_BOTOCORE_LOG_LEVEL = 'warning'


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Launch SageMaker training jobs or hosting endpoints')
    parser.set_defaults(func=lambda x: parser.print_usage())

    # common args for training/hosting/all frameworks
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--role-name', help='SageMaker execution role name', type=str, required=True)
    common_parser.add_argument('--data', help='path to training data or model files', type=str, default='./data')
    common_parser.add_argument('--script', help='path to script', type=str, default='./script.py')
    common_parser.add_argument('--job-name', help='job or endpoint name', type=str, default=None)
    common_parser.add_argument('--bucket-name', help='S3 bucket for training/model data and script files',
                               type=str, default=None)
    common_parser.add_argument('--python', help='python version', type=str, default='py2')

    instance_group = common_parser.add_argument_group('instance settings')
    instance_group.add_argument('--instance-type', type=str, help='instance type', default='ml.m4.xlarge')
    instance_group.add_argument('--instance-count', type=int, help='instance count', default=1)

    # common training args
    common_train_parser = argparse.ArgumentParser(add_help=False)
    common_train_parser.add_argument('--hyperparameters', help='path to training hyperparameters file',
                                     type=str, default='./hyperparameters.json')

    # common hosting args
    common_host_parser = argparse.ArgumentParser(add_help=False)
    common_host_parser.add_argument('--env', help='hosting environment variable(s)', type=str, nargs='*', default=[])

    subparsers = parser.add_subparsers()

    # framework/algo subcommands
    mxnet_parser = subparsers.add_parser('mxnet', help='use MXNet', parents=[])
    mxnet_subparsers = mxnet_parser.add_subparsers()
    mxnet_train_parser = mxnet_subparsers.add_parser('train',
                                                     help='start a training job',
                                                     parents=[common_parser, common_train_parser])
    mxnet_train_parser.set_defaults(func=sagemaker.cli.mxnet.train)

    mxnet_host_parser = mxnet_subparsers.add_parser('host',
                                                    help='start a hosting endpoint',
                                                    parents=[common_parser, common_host_parser])
    mxnet_host_parser.set_defaults(func=sagemaker.cli.mxnet.host)

    tensorflow_parser = subparsers.add_parser('tensorflow', help='use TensorFlow', parents=[])
    tensorflow_subparsers = tensorflow_parser.add_subparsers()
    tensorflow_train_parser = tensorflow_subparsers.add_parser('train',
                                                               help='start a training job',
                                                               parents=[common_parser, common_train_parser])
    tensorflow_train_parser.add_argument('--training-steps',
                                         help='number of training steps (tensorflow only)', type=int, default=None)
    tensorflow_train_parser.add_argument('--evaluation-steps',
                                         help='number of evaluation steps (tensorflow only)', type=int, default=None)
    tensorflow_train_parser.set_defaults(func=sagemaker.cli.tensorflow.train)

    tensorflow_host_parser = tensorflow_subparsers.add_parser('host',
                                                              help='start a hosting endpoint',
                                                              parents=[common_parser, common_host_parser])
    tensorflow_host_parser.set_defaults(func=sagemaker.cli.tensorflow.host)

    log_group = parser.add_argument_group('optional log settings')
    log_group.add_argument('--log-level', help='log level for this command', type=str, default=DEFAULT_LOG_LEVEL)
    log_group.add_argument('--botocore-log-level', help='log level for botocore', type=str,
                           default=DEFAULT_BOTOCORE_LOG_LEVEL)

    return parser.parse_args(args)


def configure_logging(args):
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    log_level = logging.getLevelName(args.log_level.upper())
    logging.basicConfig(format=log_format, level=log_level)
    logging.getLogger("botocore").setLevel(args.botocore_log_level.upper())


def main():
    args = parse_arguments(sys.argv[1:])
    configure_logging(args)
    logger.debug('args: {}'.format(args))
    args.func(args)


if __name__ == '__main__':
    main()
