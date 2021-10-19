import argparse
import os

from sagemaker_xgboost_container.data_utils import get_dmatrix

import xgboost as xgb

model_filename = "xgboost-model"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/abalone"),
    )

    args, _ = parser.parse_known_args()

    dtrain = get_dmatrix(args.train, "libsvm")

    params = {
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weight": 6,
        "subsample": 0.7,
        "verbosity": 2,
        "objective": "reg:squarederror",
        "tree_method": "auto",
        "predictor": "auto",
    }

    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=50)
    booster.save_model(args.model_dir + "/" + model_filename)


def model_fn(model_dir):
    """Deserialize and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, model_filename))
    return booster
