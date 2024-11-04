#!/bin/sh
#SBATCH -p v
#SBATCH -n 2

poetry run python src/classify_prediction/model_train.py