import datetime


class _LocalTrainingJob(object):
    """Defines and starts a local training job."""

    _STARTING = "Starting"
    _TRAINING = "Training"
    _COMPLETED = "Completed"
    _states = ["Starting", "Training", "Completed"]

    def __init__(self, container):
        """Creates a local training job.

        Args:
            container: the local container object.
        """
        self.container = container
        self.model_artifacts = None
        self.state = "created"
        self.start_time = None
        self.end_time = None
        self.environment = None
        self.training_job_name = ""
        self.output_data_config = None

    def start(self, input_data_config, output_data_config, hyperparameters, environment, job_name):
        """Starts a local training job.

        Args:
            input_data_config (dict): The Input Data Configuration, this contains data such as the
                channels to be used for training.
            output_data_config (dict): The configuration of the output data.
            hyperparameters (dict): The HyperParameters for the training job.
            environment (dict): The collection of environment variables passed to the job.
            job_name (str): Name of the local training job being run.

        Raises:
            ValueError: If the input data configuration is not valid.
            RuntimeError: If the data distribution type is not supported.
        """
        for channel in input_data_config:
            if channel["DataSource"] and "S3DataSource" in channel["DataSource"]:
                data_distribution = channel["DataSource"]["S3DataSource"].get(
                    "S3DataDistributionType", None
                )
                data_uri = channel["DataSource"]["S3DataSource"]["S3Uri"]
            elif channel["DataSource"] and "FileDataSource" in channel["DataSource"]:
                data_distribution = channel["DataSource"]["FileDataSource"][
                    "FileDataDistributionType"
                ]
                data_uri = channel["DataSource"]["FileDataSource"]["FileUri"]
            else:
                raise ValueError(
                    "Need channel['DataSource'] to have ['S3DataSource'] or ['FileDataSource']"
                )

            # use a single Data URI - this makes handling S3 and File Data easier down the stack
            channel["DataUri"] = data_uri

            supported_distributions = ["FullyReplicated"]
            if data_distribution and data_distribution not in supported_distributions:
                raise RuntimeError(
                    "Invalid DataDistribution: '{}'. Local mode currently supports: {}.".format(
                        data_distribution, ", ".join(supported_distributions)
                    )
                )

        self.start_time = datetime.datetime.now()
        self.state = self._TRAINING
        self.environment = environment

        self.model_artifacts = self.container.train(
            input_data_config, output_data_config, hyperparameters, environment, job_name
        )
        self.end_time = datetime.datetime.now()
        self.state = self._COMPLETED
        self.training_job_name = job_name
        self.output_data_config = output_data_config

    def describe(self):
        """Placeholder docstring"""
        response = {
            "TrainingJobName": self.training_job_name,
            "TrainingJobArn": "unused-arn",
            "ResourceConfig": {"InstanceCount": self.container.instance_count},
            "TrainingJobStatus": self.state,
            "TrainingStartTime": self.start_time,
            "TrainingEndTime": self.end_time,
            "ModelArtifacts": {"S3ModelArtifacts": self.model_artifacts},
            "OutputDataConfig": self.output_data_config,
            "Environment": self.environment,
            "AlgorithmSpecification": {
                "ContainerEntrypoint": self.container.container_entrypoint,
            },
        }
        return response
