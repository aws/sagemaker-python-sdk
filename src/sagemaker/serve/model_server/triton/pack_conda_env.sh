#!/bin/bash
conda=$1
output_path=$2

export PYTHONNOUSERSITE=True
rm $output_path
conda-pack -o $output_path