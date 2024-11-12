import os
import pickle as pkl
import json
import numpy as np
import xgboost as xgb

from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)

from sagemaker_xgboost_container import encoder as xgb_encoders


def input_fn(input_data, content_type):
    if content_type == content_types.JSON:
        obj = json.loads(input_data)
        features = obj["instances"][0]["features"]
        array = np.array(features).reshape((1, -1))
        return xgb.DMatrix(array)
    else:
        return xgb_encoders.decode(input_data, content_type)


def model_fn(model_dir):
    model_file = model_dir + "/model.bin"
    model = pkl.load(open(model_file, "rb"))
    return model


def output_fn(prediction, accept):

    pred_array_value = np.array(prediction)
    score = pred_array_value[0]

    if accept == "application/json":
        predicted_label = 1 if score > 0.5 else 0
        return_value = {
            "predictions": [{"score": score.astype(float), "predicted_label": predicted_label}]
        }
        return worker.Response(json.dumps(return_value), mimetype=accept)
    elif accept == "text/csv":
        return_value = "yes" if score > 0.5 else "no"
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported.".format(accept))
