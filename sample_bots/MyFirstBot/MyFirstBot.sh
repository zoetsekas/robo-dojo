#!/bin/sh
set -e

# Change to script directory
cd -- "$(dirname -- "$0")"

# Install dependencies (relative to script dir)
../deps/install-dependencies.sh

# Try to use venv python first (correct path: ../deps/venv)
if [ -x "../deps/venv/bin/python" ]; then
    exec "../deps/venv/bin/python" "MyFirstBot.py"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "MyFirstBot.py"
elif command -v python >/dev/null 2>&1; then
    exec python "MyFirstBot.py"
else
    echo "Error: Python not found. Please install python3 or python." >&2
    exit 1
fi