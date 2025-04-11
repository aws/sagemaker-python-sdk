# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import gzip
import json
import mxnet as mx
import numpy as np
import os
import struct


# --- this example demonstrates how to extend default behavior during model hosting ---


# --- Model preparation ---
# it is possible to specify own code to load the model, otherwise a default model loading takes place
def model_fn(path_to_model_files):
    from mxnet.io import DataDesc

    loaded_symbol = mx.symbol.load(os.path.join(path_to_model_files, "symbol"))
    created_module = mx.mod.Module(symbol=loaded_symbol)
    created_module.bind([DataDesc("data", (1, 1, 28, 28))])
    created_module.load_params(os.path.join(path_to_model_files, "params"))
    return created_module


# --- Option 1 - provide just 1 entry point for end2end prediction ---
# if this function is specified, no other overwriting described in Option 2 will have effect
# returns serialized data and content type it has used
def transform_fn(model, request_data, input_content_type, requested_output_content_type):
    # for demonstration purposes we will be calling handlers from Option2
    return (
        output_fn(
            process_request_fn(model, request_data, input_content_type),
            requested_output_content_type,
        ),
        requested_output_content_type,
    )


# --- Option 2 - overwrite container's default input/output behavior with handlers ---
# there are 2 data handlers: input and output, you need to conform to their interface to fit into default execution
def process_request_fn(model, data, input_content_type):
    if input_content_type == "text/s3_file_path":
        prediction_input = handle_s3_file_path(data)
    elif input_content_type == "application/json":
        prediction_input = handle_json_input(data)
    else:
        raise NotImplementedError(
            "This model doesnt support requested input type: " + input_content_type
        )

    return model.predict(prediction_input)


# for this example S3 path points to a file that is same format as in test/images.gz
def handle_s3_file_path(path):
    import sys

    if sys.version_info.major == 2:
        import urlparse

        parse_cmd = urlparse.urlparse
    else:
        import urllib

        parse_cmd = urllib.parse.urlparse

    import boto3
    from botocore.exceptions import ClientError

    # parse the path
    parsed_url = parse_cmd(path)

    # get S3 client
    s3 = boto3.resource("s3")

    # read file content and pass it down
    obj = s3.Object(parsed_url.netloc, parsed_url.path.lstrip("/"))
    print("loading file: " + str(obj))

    try:
        data = obj.get()["Body"]
    except ClientError as ce:
        raise ValueError(
            "Can't download from S3 path: " + path + " : " + ce.response["Error"]["Message"]
        )

    import StringIO

    buf = StringIO(data.read())
    img = gzip.GzipFile(mode="rb", fileobj=buf)

    _, _, rows, cols = struct.unpack(">IIII", img.read(16))
    images = np.fromstring(img.read(), dtype=np.uint8).reshape(10000, rows, cols)
    images = images.reshape(images.shape[0], 1, 28, 28).astype(np.float32) / 255

    return mx.io.NDArrayIter(images, None, 1)


# for this example it is assumed that the client is passing data that can be "directly" provided to the model
def handle_json_input(data):
    nda = mx.nd.array(json.loads(data))
    return mx.io.NDArrayIter(nda, None, 1)


def output_fn(prediction_output, requested_output_content_type):
    # output from the model is NDArray

    data_to_return = prediction_output.asnumpy()

    if requested_output_content_type == "application/json":
        json.dumps(data_to_return.tolist), requested_output_content_type

    raise NotImplementedError(
        "Model doesn't support requested output type: " + requested_output_content_type
    )
