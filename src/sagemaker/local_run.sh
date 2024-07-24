#!/bin/bash

# Entry point for running the service locally

# Verify if uvicorn and fastapi are installed
if ! command -v uvicorn &> /dev/null
then
    echo "uvicorn could not be found, installing..."
    pip install "uvicorn[standard]"
fi

echo "Starting the service locally"
uvicorn $1:app --port 8000 --host 0.0.0.0 --reload &
UVICORN_PID=$!

echo $UVICORN_PID

echo "Wait for the service to be completely ready"
# uvicorn takes couple of seconds to launch the service
sleep 10

exit 0
