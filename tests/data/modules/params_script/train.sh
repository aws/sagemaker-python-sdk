#!/bin/bash
set -e

echo "Do some extra work here..."

CMD="python train.py $@"
echo "Executing Command: $CMD"

python train.py "$@"

echo "Done!"
