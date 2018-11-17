# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import alexa


def model_fn(anything):
    return alexa


def predict_fn(input_object, model):
    return input_object


if __name__ == '__main__':
    with open('/opt/ml/model/answer', 'w') as model:
        model.write(str(alexa.question('How many roads must a man walk down?')))
