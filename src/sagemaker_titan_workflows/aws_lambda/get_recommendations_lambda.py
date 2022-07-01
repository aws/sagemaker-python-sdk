from titan_ml.dataset.optimus import OptimusNaming, load_optimus_data
from titan_ml.instance_info import InstanceInfos
from titan_ml.optimizer.search_space import choice
from titan_ml.recommendation.pareto_select_recommender import \
    ParetoSelectRecommender

df, input_columns, output_columns = load_optimus_data()

# fit recommender on all data
recommender = ParetoSelectRecommender(
    input_columns=input_columns, output_columns=output_columns
)
transformer = recommender.fit(df)

instance_info = InstanceInfos()
instances = instance_info.instances
env_variables_of_instance = {
    instance: {
        # we want to allow threads less than the number of cpus
        OptimusNaming.omp_num_threads: choice(range(1, instance_info(instance).num_cpu))
        if instance_info(instance).num_gpu == 0
        # gpu instances shouldn't have multiple threads
        else choice([1]),
        # we want to allow workers less than the number of cpus
        OptimusNaming.num_workers: choice(range(1, instance_info(instance).num_cpu)),
    }
    for instance in instances
}


def get_recommendations_handler(event, context):
    nearest_model_name = event["NearestModelName"]
    framework = event["Framework"]

    recommendations = recommender.transform(
        num_recommendation=10,
        nearest_model_name=nearest_model_name,
        framework=framework,
        env=env_variables_of_instance,
    )

    return [
        {
            "InstanceType": x.instanceType,
            "EnvironmentVariables": {str(k): str(v) for k, v in x.env.items()},
        }
        for x in recommendations
    ]
