#!/usr/bin/env bash
gcloud auth login --account=fccassistant27@gmail.com
gcloud config set project fcc-assistant-mvp
pip install -r requirements.txt