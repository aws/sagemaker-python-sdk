from sagemaker.cli.common import HostCommand, TrainCommand


def train(args):
    TensorFlowTrainCommand(args).start()


def host(args):
    TensorFlowHostCommand(args).start()


class TensorFlowTrainCommand(TrainCommand):
    def __init__(self, args):
        super(TensorFlowTrainCommand, self).__init__(args)
        self.training_steps = args.training_steps
        self.evaluation_steps = args.evaluation_steps

    def create_estimator(self):
        from sagemaker.tensorflow import TensorFlow
        return TensorFlow(training_steps=self.training_steps,
                          evaluation_steps=self.evaluation_steps,
                          py_version=self.python,
                          entry_point=self.script,
                          role=self.role_name,
                          base_job_name=self.job_name,
                          train_instance_count=self.instance_count,
                          train_instance_type=self.instance_type,
                          hyperparameters=self.hyperparameters)


class TensorFlowHostCommand(HostCommand):
    def __init__(self, args):
        super(TensorFlowHostCommand, self).__init__(args)

    def create_model(self, model_url):
        from sagemaker.tensorflow.model import TensorFlowModel
        return TensorFlowModel(model_data=model_url, role=self.role_name, entry_point=self.script,
                               py_version=self.python, name=self.endpoint_name, env=self.environment)
