#!/bin/bash
set -e

# Cleanup function
cleanup() {
    echo "Stopping processes..."
    # Kill any Xvfb or Java processes we might have spawned implicitly?
    # Actually Python will manage them via start_infrastructure.
    exit 0
}
trap cleanup SIGINT SIGTERM

# We do NOT start Xvfb/Server/GUI here anymore.
# They are managed by the Python Env per worker.

echo "Starting Training Agent..."
# Run the Python training script
# Pass arguments if any
python -m src.train "$@" &
TRAIN_PID=$!

# Wait for training to finish
wait $TRAIN_PID
