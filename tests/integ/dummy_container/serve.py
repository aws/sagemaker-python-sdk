# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import json

import flask

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
OCTET_STREAM_CONTENT_TYPE = "application/octet-stream"
ANY_CONTENT_TYPE = '*/*'
UTF8_CONTENT_TYPES = [JSON_CONTENT_TYPE, CSV_CONTENT_TYPE]

app = flask.Flask(__name__)


def read_assertions():
    with open('/assert.json') as f:
        return json.load(f)


assertions = read_assertions()


@app.route("/ping")
def ping():
    return "OK"


@app.route("/invocations", methods=["POST"])
def invocations():
    content_type = flask.request.headers.get('Content-Type', JSON_CONTENT_TYPE)
    content_type = flask.request.headers.get('ContentType', content_type)

    output_content_type = flask.request.headers.get('Accept', JSON_CONTENT_TYPE)

    if content_type in UTF8_CONTENT_TYPES:
        content = flask.request.get_data().decode('utf-8')
    else:
        content = flask.request.get_data()

    print('================ Request ==================')
    print(content)
    print('===========================================')
    expected_request = assertions['expected_requests'][content]

    expected_content_type = expected_request['content_type']
    assert content_type == expected_content_type, '%s != %s' % (content_type, expected_content_type)

    response = expected_request['response']
    print('================ Response ==================')
    print(response)
    print('============================================')
    return flask.Response(response=response, mimetype=output_content_type)
