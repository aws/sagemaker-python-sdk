#!/bin/bash

python ./download_training_data.py
sagemaker mxnet train --role-name <your-sagemaker-execution-role>
