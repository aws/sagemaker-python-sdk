from __future__ import absolute_import

import argparse
import logging
import sys

import sagemaker

logger = logging.getLogger(__name__)


def parse_arguments(args):
    # common arguments
    common_parser = argparse.ArgumentParser(add_help=False)

    # image-related settings
    image_mtx = common_parser.add_mutually_exclusive_group(required=True)
    image_mtx.add_argument('--tf', help='use a TensorFlow container image', action='store_true')
    image_mtx.add_argument('--mx', help='use an MXNet container image', action='store_true')

    # path to data and script files
    common_parser.add_argument('--data', help='path to training data or model files', type=str, default='./data')
    common_parser.add_argument('--script', help='path to script', type=str, default='./script.py')
    common_parser.add_argument('--job-name', help='job or endpoint name', type=str, default=None)
    common_parser.add_argument('--bucket-name', help='S3 bucket', type=str, default=None)
    common_parser.add_argument('--role-name', help='SageMaker execution role name', type=str,
                               default='AmazonSageMakerFullAccess')

    instance_group = common_parser.add_argument_group('instance settings')
    instance_group.add_argument('--instance-type', type=str, help='instance type', default='ml.m4.xlarge')
    instance_group.add_argument('--instance-count', type=int, help='instance count', default=1)

    image_group = common_parser.add_argument_group('other container image settings')
    image_group.add_argument('--python', help='python version (mxnet only)', type=str, default='py2')

    parser = argparse.ArgumentParser(description='Launch SageMaker training jobs or hosting endpoints')
    parser.set_defaults(func=lambda x: parser.print_usage())

    log_group = parser.add_argument_group('log settings')
    log_group.add_argument('--log-level', help='log level for this command', type=str, default='info')
    log_group.add_argument('--botocore-log-level', help='log level for botocore', type=str, default='warning')

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='start a training job', parents=[common_parser])
    train_group = train_parser.add_argument_group('training settings')
    train_group.add_argument('--hyperparameters', help='path to training hyperparameters file',
                             type=str, default='./hyperparameters.json')
    train_group.add_argument('--training-steps',
                             help='number of training steps (tensorflow only)', type=int, default=None)
    train_group.add_argument('--evaluation-steps',
                             help='number of evaluation steps (tensorflow only)', type=int, default=None)
    train_parser.set_defaults(mode='train')
    train_parser.set_defaults(func=sagemaker.cli.train)

    host_parser = subparsers.add_parser('host', help='start a hosting endpoint', parents=[common_parser])
    host_group = host_parser.add_argument_group('hosting settings')
    host_group.add_argument('--env', help='hosting environment variable(s)', type=str, nargs='*', default=[])
    train_parser.set_defaults(mode='host')
    host_parser.set_defaults(func=sagemaker.cli.host)

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
