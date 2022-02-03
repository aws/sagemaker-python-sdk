import os

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

# Based on https://github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst#python-api
ray.init(log_to_driver=False)
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = int(os.environ.get("SM_NUM_GPUS", 0))
checkpoint_dir = os.environ.get("SM_MODEL_DIR", "/tmp")
config["num_workers"] = 1
agent = ppo.PPOTrainer(config=config, env="CartPole-v0")

# Can optionally call agent.restore(path) to load a checkpoint.

for i in range(5):
    # Perform one iteration of training the policy with PPO
    result = agent.train()
    print(pretty_print(result))

    checkpoint = agent.save(checkpoint_dir=checkpoint_dir)
    print("checkpoint saved at", checkpoint)
