#!/usr/bin/env bash

python benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_batches=500 --model vgg16 --variable_update horovod --horovod_device gpu --use_fp16 --summary_verbosity 1 --save_summaries_steps 10 --train_dir /opt/ml/model --eval_dir /opt/ml/model --batch_size 32
