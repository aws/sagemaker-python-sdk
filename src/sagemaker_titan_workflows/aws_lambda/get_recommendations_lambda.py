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


def get_recommendations_handler(event, context):
    nearest_model_name = event["NearestModelName"]
    framework = event["Framework"]

    recommendations = recommender.transform(
        num_recommendation=10,
        nearest_model_name=nearest_model_name,
        framework=framework,
    )

    return [x.instanceType for x in recommendations]
