from __future__ import print_function

import json
import mxnet as mx
from mxnet import gluon


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    symbol = mx.sym.load("%s/model.json" % model_dir)
    outputs = mx.symbol.softmax(data=symbol, name="softmax_label")
    inputs = mx.sym.var("data")
    param_dict = gluon.ParameterDict("model_")
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net.load_params("%s/model.params" % model_dir, ctx=mx.cpu())
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    parsed = json.loads(data)
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist())
    return response_body, output_content_type
