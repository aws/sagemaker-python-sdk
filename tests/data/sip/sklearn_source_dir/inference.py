from __future__ import print_function

import sys
from io import StringIO
import os
import csv
import json
import numpy as np
import pandas as pd

from sklearn.externals import joblib

from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)

feature_columns_names = [
    "turbine_id",
    "turbine_type",
    "wind_speed",
    "rpm_blade",
    "oil_temperature",
    "oil_level",
    "temperature",
    "humidity",
    "vibrations_frequency",
    "pressure",
    "wind_direction",
]


def input_fn(input_data, content_type):
    print(input_data)

    if content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header=None)
        if len(df.columns) == len(feature_columns_names):
            df.columns = feature_columns_names
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def predict_fn(input_data, model):
    features = model.transform(input_data)
    return features


def output_fn(prediction, accept):
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})
        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported.".format(accept))


def model_fn(model_dir):
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
