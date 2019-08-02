#!/bin/bash

cd atoms

export PATH=~/google-cloud-sdk/bin:$PATH
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
# --train-files /NumeraiData/Numerai_test/sample_data.csv

# Logistic Regression
export IMAGE_REPO_NAME=classification_sklearn_logisticregression
export IMAGE_TAG=class_skl_logreg
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f docker_logisticRegression_classifier -t $IMAGE_URI ./

docker push $IMAGE_URI


# Light Gradient Boosting Classifier

export IMAGE_REPO_NAME=classification_lgbm
export IMAGE_TAG=class_lgbm
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f docker_lgbm_classifier -t $IMAGE_URI ./

docker push $IMAGE_URI


# XGBoost Classifier

export IMAGE_REPO_NAME=classification_xgboost
export IMAGE_TAG=class_xgb
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f docker_xgb_classifier -t $IMAGE_URI ./

docker push $IMAGE_URI


# Auto-sklearn Classifier

export IMAGE_REPO_NAME=classification_autosklearn
export IMAGE_TAG=class_askl
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f docker_askl_classifier -t $IMAGE_URI ./

docker push $IMAGE_URI


# Random Forest


# LDA


# QDA



# Bayesian Modeling


# SuperLearner Classifier
