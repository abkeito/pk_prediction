#!/bin/sh
#SBATCH -p v
#SBATCH -n 2

poetry run python src/transformer_classification/main.py