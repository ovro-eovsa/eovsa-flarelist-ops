#!/bin/bash

# This script wraps the flarelist2sql.py Python script, passing through any arguments directly.
# Ensure you're in the 'user' group for correct permissions: use 'newgrp user' beforehand.

PYTHON_SCRIPT_PATH="$HOME/eovsa-flarelist-ops/flarelist2sql.py"

# Check for the help option explicitly to display Python script's help
if [[ " $* " == *" --help "* ]] || [[ " $* " == *" -h "* ]]; then
    python $PYTHON_SCRIPT_PATH --help
else
    # Pass all arguments to the Python script
    python $PYTHON_SCRIPT_PATH "$@"
fi