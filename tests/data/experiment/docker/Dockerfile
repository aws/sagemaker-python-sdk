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

FROM ubuntu:22.04

ARG script
ARG library
ARG botomodel
ARG metricsmodel

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        curl \
        ca-certificates \
        awscli \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py


WORKDIR /root

COPY $library .

# use a custom model
# TODO: coumment out these four lines once the new API changes for Run release
COPY $botomodel sagemaker-2017-07-24.normal.json
RUN aws configure add-model --service-name sagemaker --service-model file://sagemaker-2017-07-24.normal.json
COPY $metricsmodel sagemaker-metrics-2022-09-30.normal.json
RUN aws configure add-model --service-name sagemaker-metrics --service-model file://sagemaker-metrics-2022-09-30.normal.json

RUN python3 -m pip install $(basename $library)

COPY $script script.py

ENTRYPOINT ["python3", "./script.py"]