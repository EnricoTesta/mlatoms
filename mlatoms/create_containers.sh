#!/bin/bash

export PATH=~/google-cloud-sdk/bin:$PATH
# --train-files /NumeraiData/Numerai_test/sample_data.csv

# Logistic Regression

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=sklearn_logisticregression
export IMAGE_TAG=sklearn_logreg
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f LogisticRegression_DockerFile -t $IMAGE_URI ./

docker run $IMAGE_URI --C 1 --penalty l2

docker push $IMAGE_URI


# Light Gradient Boosting Classifier

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=lgbm_classifier
export IMAGE_TAG=lgbm_classifier
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f lgbm_classifier_DockerFile -t $IMAGE_URI ./

docker run $IMAGE_URI --n_estimators 10 --max_depth 2 --debug 1

docker push $IMAGE_URI


# XGBoost Classifier

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=xgb_classifier
export IMAGE_TAG=xgb_classifier
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f xgb_classifier_DockerFile -t $IMAGE_URI ./

docker run $IMAGE_URI --n_estimators 10 --max_depth 2 --debug 1

docker push $IMAGE_URI


# Auto-sklearn Classifier

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=askl_classifier
export IMAGE_TAG=askl_classifier
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f askl_classifier_DockerFile -t $IMAGE_URI ./

docker run $IMAGE_URI --debug 1

docker push $IMAGE_URI


# Random Forest


# LDA


# QDA



# Bayesian Modeling


# SuperLearner Classifier
