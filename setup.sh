#!/usr/bin/env bash
echo "Using account fccassistant27@gmail.com, project fcc-assistant-mvp, location us-central1, bucket fcc-assistant-84."
echo "NOTE: Change above settings in the file to adjust for your environment!"
gcloud auth login --account=fccassistant27@gmail.com
gcloud config set project fcc-assistant-mvp
gcloud storage buckets create gs://fcc-assistant-84 --location=us-central1
pip install -r requirements.txt