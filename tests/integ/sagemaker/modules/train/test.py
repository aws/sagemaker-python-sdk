# import os
# import tempfile
# from tests.integ import DATA_DIR
# import tests.integ.lock as lock

# from sagemaker.modules.configs import Compute, InputData, SourceCode
# from sagemaker.modules.distributed import Torchrun
# from sagemaker.modules.train.model_trainer import Mode, ModelTrainer
# import subprocess
# from sagemaker.modules import Session

# DEFAULT_CPU_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"
# CWD = os.getcwd()
# SOURCE_DIR = os.path.join(DATA_DIR, "modules/local_script")
# LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_test_local_mode_lock")

# source_code = SourceCode(
#     source_dir=SOURCE_DIR,
#     entry_script="local_training_script.py",
# )

# compute = Compute(
#     instance_type="local_cpu",
#     instance_count=1,
# )

# session = Session()
# bucket = session.default_bucket()
# session.upload_data(
#     path=os.path.join(SOURCE_DIR, "data/train/"),
#     bucket=bucket,
#     key_prefix="data/train",
# )
# session.upload_data(
#     path=os.path.join(SOURCE_DIR, "data/test/"),
#     bucket=bucket,
#     key_prefix="data/test",
# )

# train_data = InputData(channel_name="train", data_source=f"s3://{bucket}/data/train/")

# test_data = InputData(channel_name="test", data_source=f"s3://{bucket}/data/test/")

# model_trainer = ModelTrainer(
#     training_image=DEFAULT_CPU_IMAGE,
#     source_code=source_code,
#     compute=compute,
#     input_data_config=[train_data, test_data],
#     base_job_name="local_mode_single_container_local_data",
#     training_mode=Mode.LOCAL_CONTAINER,
#     remove_inputs_and_container_artifacts=False,
# )

# model_trainer.train()

from sagemaker.local import LocalSession

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}