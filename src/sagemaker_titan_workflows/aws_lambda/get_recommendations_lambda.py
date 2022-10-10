import math

from titan_ml.dataset.optimus import OptimusNaming, load_optimus_data
from titan_ml.instance_info import InstanceInfos
from titan_ml.optimizer.search_space import choice
from titan_ml.recommendation.pareto_select_recommender import ParetoSelectRecommender

df, input_columns, output_columns = load_optimus_data()

# fit recommender on all data
recommender = ParetoSelectRecommender(
    input_columns=input_columns, output_columns=output_columns
)
transformer = recommender.fit(df)

instance_info = InstanceInfos()
instances = instance_info.instances

integer_env_vars = ["OMP_NUM_THREADS", "TS_DEFAULT_WORKERS_PER_MODEL"]


def get_config(instance):
    num_gpu = instance_info(instance).num_gpu
    num_cpu = instance_info(instance).num_cpu

    if num_gpu > 0:
        return {OptimusNaming.num_workers: choice(range(1, num_cpu))}

    return {
        OptimusNaming.omp_num_threads: choice(range(1, num_cpu)),
        OptimusNaming.num_workers: choice(range(1, num_cpu)),
    }


default_env = {instance: get_config(instance) for instance in instances}


def get_recommendations_handler(event, context):
    nearest_model_name = event["CustomerModelDetails"].get("NearestModelName")
    framework = event["CustomerModelDetails"].get("Framework")
    count = event["Count"]
    instance_types = event["InstanceTypes"]

    # filter only instances supported
    env = {k: v for (k, v) in default_env.items() if k in instance_types}

    recommendations = recommender.transform(
        num_recommendation=count,
        nearest_model_name=nearest_model_name,
        framework=framework,
        env=env if env else default_env,
    )

    return [
        {
            "instanceType": x.instanceType,
            "env": {
                str(k): str(int(v)) if k in integer_env_vars else str(v)
                for k, v in x.env.items()
                if not math.isnan(v)
            },
        }
        for x in recommendations
    ]
