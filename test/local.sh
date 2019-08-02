#!/bin/bash


cd atoms
#TODO: rename image URIs and check whether docker run is successful or not.
export PATH=~/google-cloud-sdk/bin:$PATH
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
# --train-files /NumeraiData/Numerai_test/sample_data.csv

# Logistic Regression
docker run $IMAGE_URI --C 1 --penalty l2

# Light Gradient Boosting Classifier
docker run $IMAGE_URI --n_estimators 10 --max_depth 2 --debug 1

# XGBoost Classifier
docker run $IMAGE_URI --n_estimators 10 --max_depth 2 --debug 1

# Auto-sklearn Classifier
docker run $IMAGE_URI --debug 1

# Random Forest


# LDA


# QDA



# Bayesian Modeling


# SuperLearner Classifier
