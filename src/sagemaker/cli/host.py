from __future__ import absolute_import

import os
import shutil
import tarfile
import tempfile

import sagemaker


def host(args):
    return HostingCommand(args).start()


class HostingCommand(object):
    def __init__(self, args):
        self.endpoint_name = args.job_name
        self.bucket = args.bucket_name  # may be None
        self.role_name = args.role_name
        self.data = args.data
        self.script = args.script
        self.python = args.python
        self.instance_type = args.instance_type
        self.instance_count = args.instance_count
        self.framework = 'tensorflow' if args.tf else 'mxnet' if args.mx else 'undefined'
        self.environment = {k: v for k, v in (kv.split('=') for kv in args.env)}

        self.session = sagemaker.Session()

    def upload_model(self):
        prefix = '{}/model'.format(self.endpoint_name)

        archive = self.create_model_archive(self.data)
        model_uri = self.session.upload_data(path=archive, bucket=self.bucket, key_prefix=prefix)
        shutil.rmtree(os.path.dirname(archive))

        return model_uri

    @staticmethod
    def create_model_archive(src):
        if os.path.isdir(src):
            arcname = '.'
        else:
            arcname = os.path.basename(src)

        tmp = tempfile.mkdtemp()
        archive = os.path.join(tmp, 'model.tar.gz')

        with tarfile.open(archive, mode='w:gz') as t:
            t.add(src, arcname=arcname)
        t.close()
        return archive

    def create_model(self, model_url):
        if self.framework == 'tensorflow':
            from sagemaker.tensorflow.model import TensorFlowModel
            return TensorFlowModel(model_data=model_url, role=self.role_name, entry_point=self.script,
                                   name=self.endpoint_name, env=self.environment)
        elif self.framework == 'mxnet':
            from sagemaker.mxnet.model import MXNetModel
            return MXNetModel(model_data=model_url, role=self.role_name, entry_point=self.script,
                              py_version=self.python, name=self.endpoint_name, env=self.environment)
        else:
            raise ValueError('unsupported framework value: {}'.format(self.framework))

    def start(self):
        model_url = self.upload_model()
        model = self.create_model(model_url)
        predictor = model.deploy(initial_instance_count=self.instance_count,
                                 instance_type=self.instance_type)

        return predictor
