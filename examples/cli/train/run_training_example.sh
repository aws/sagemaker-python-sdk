#!/bin/bash

python ./download_training_data.py
sagemaker train --mx
