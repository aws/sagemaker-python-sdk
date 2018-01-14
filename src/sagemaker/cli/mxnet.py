from sagemaker.cli.common import HostCommand, TrainCommand


def train(args):
    MXNetTrainCommand(args).start()


def host(args):
    MXNetHostCommand(args).start()


class MXNetTrainCommand(TrainCommand):
    def __init__(self, args):
        super(MXNetTrainCommand, self).__init__(args)

    def create_estimator(self):
        from sagemaker.mxnet.estimator import MXNet
        return MXNet(self.script,
                     role=self.role_name,
                     base_job_name=self.job_name,
                     train_instance_count=self.instance_count,
                     train_instance_type=self.instance_type,
                     hyperparameters=self.hyperparameters,
                     py_version=self.python)


class MXNetHostCommand(HostCommand):
    def __init__(self, args):
        super(MXNetHostCommand, self).__init__(args)

    def create_model(self, model_url):
        from sagemaker.mxnet.model import MXNetModel
        return MXNetModel(model_data=model_url, role=self.role_name, entry_point=self.script,
                          py_version=self.python, name=self.endpoint_name, env=self.environment)
