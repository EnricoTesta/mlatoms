#!/bin/bash

export PATH=~/google-cloud-sdk/bin:$PATH
# --train-files /NumeraiData/Numerai_test/sample_data.csv

# Job Request with HPT
#export BUCKET_NAME=ml_train_deploy_test
#export MODEL_DIR=sklearn_model_$(date +%Y%m%d_%H%M%S)
#export REGION=us-central1
#export JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)
#
#gcloud beta ai-platform jobs submit training $JOB_NAME \
#  --region $REGION \
#  --master-image-uri $IMAGE_URI \
#  --config /Numerai_2.0/mlatoms/ht_config.yml \
#  -- \
#  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR \
#  --train-files=gs://$BUCKET_NAME/sample_data/sample_data.csv

# Logistic Regression
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=sklearn_logisticregression
export IMAGE_TAG=sklearn_logreg
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

export BUCKET_NAME=ml_train_deploy_test
export MODEL_DIR=logistic_regression_$(date +%Y%m%d_%H%M%S)
export REGION=us-central1
export JOB_NAME=logistic_regression_job_$(date +%Y%m%d_%H%M%S)

gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR \
  --train-files=gs://$BUCKET_NAME/sample_data/sample_data.csv


# Light Gradient Boosting Classifier
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=lgbm_classifier
export IMAGE_TAG=lgbm_classifier
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

export BUCKET_NAME=ml_train_deploy_test
export MODEL_DIR=lgbm_classifier_$(date +%Y%m%d_%H%M%S)
export REGION=us-central1
export JOB_NAME=lgbm_classifier_job_$(date +%Y%m%d_%H%M%S)

gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR \
  --train-files=gs://$BUCKET_NAME/sample_data/sample_data.csv


# XGBoost Classifier
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=xgb_classifier
export IMAGE_TAG=xgb_classifier
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

export BUCKET_NAME=ml_train_deploy_test
export MODEL_DIR=xgb_classifier_$(date +%Y%m%d_%H%M%S)
export REGION=us-central1
export JOB_NAME=xgb_classifier_job_$(date +%Y%m%d_%H%M%S)

gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR \
  --train-files=gs://$BUCKET_NAME/sample_data/sample_data.csv

# AutoSKLearn
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=askl_classifier
export IMAGE_TAG=askl_classifier
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

export BUCKET_NAME=ml_train_deploy_test
export MODEL_DIR=askl_classifier_$(date +%Y%m%d_%H%M%S)
export REGION=us-central1
export JOB_NAME=askl_classifier_job_$(date +%Y%m%d_%H%M%S)

gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR \
  --train-files=gs://$BUCKET_NAME/sample_data/sample_data.csv
