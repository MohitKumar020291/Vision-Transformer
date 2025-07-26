#!/bin/bash

test_file="$1"

echo "Running script with test file: $test_file"

if [ "$test_file" = "kernel" ]; then
    source vitm/Kernels/run.sh
fi