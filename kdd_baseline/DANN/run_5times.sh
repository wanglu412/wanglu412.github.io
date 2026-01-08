#!/bin/bash
# Script to run a command 5 times and calculate test AUC statistics
# Usage: ./run_5times.sh <command>
# Example: ./run_5times.sh "python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10"

if [ $# -eq 0 ]; then
    echo "Error: No command provided"
    echo "Usage: ./run_5times.sh \"your command here\""
    echo "Example: ./run_5times.sh \"python train_dann.py --dataset goodhiv_scaffold_covariate\""
    exit 1
fi

# Redirect to Python script for better statistics handling
python run_multiple.py 5 "$@"
