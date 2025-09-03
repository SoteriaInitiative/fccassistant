#!/usr/bin/env bash
echo "Please consult README.md in case of issues."
gcloud auth login --account=$GCP_ACCOUNT
gcloud config set project $PROJECT_ID
gcloud storage buckets create gs://$BUCKET --location=$REGION
pip install -r requirements.txt