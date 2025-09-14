# OFAC FAQ AI Assistant

This application is designed primarily as a demonstrator and is ***NOT*** fit for
production deployment. The purpose is to highlight the need to train LLM based AI agents
on specific narrow domains in order to receive reasonably accurate results appropriate for 
use in a financial crimes compliance setting.

The application is released under the MIT License and free to use for commercial purpose, 
with or without attribution. 

> NOTE: There is likely no or very little maintenance and there might be breaking changes in the future.

Nonetheless, this could very well serve as a template for a corporate deployment. The individual components
are kept simple enough to allow adaptations and combination with other agents -
the entire model incl. data load is in the ``model`` folder.

Excessive complexity for security/identity management has been omitted and there is no MCP either.

The application comes with a full OFAC FAQ extract at time of release, for convenience. You may want
to update the file in ``data/ofac_faq_full.pdf`` with a more recent version for best results.

This is a minimal app that retrieves context from a FAISS index in GCS and calls Vertex AI. 
Includes a model switch (base/tuned) and hard-coded login/password frontend. 
Activation of CloudRun API/Services is assumed prior to starting the app.

## Quickstart

The quickstart lets you train the model and run a hard-coded query as defined in ``app/query.py``.
For a web based chat app see further below.

> NOTE: Update the ``config.py`` with environment and application parameters that match the
> settings below or you will receive errors!

Set the initial environment parameters - do replace with your own!
```bash
export GCP_ACCOUNT="fccassistant27@gmail.com"
export PROJECT_ID="fcc-assistant-mvp"
export PROJECT_NUM="526463352197"
export GCS_PREFIX="gs://fcc-assistant-84/index"
export SA_EMAIL="526463352197-compute@developer.gserviceaccount.com"
export REGION="us-central1"
export BUCKET="gs://fcc-assistant-84"
```

Initiate the local environment - NOTE: Check ``setup.sh`` for any local settings you may need!
```bash
./setup.sh
```

Setting identity and access management permissions for GCP:
```bash
gcloud storage buckets add-iam-policy-binding $BUCKET \                                                                 
  --member="serviceAccount:${SA_EMAIL}" \                                                                    
  --role="roles/storage.objectViewer"
gcloud projects add-iam-policy-binding $PROJECT_ID \                                                  
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user"
gcloud ai models add-iam-policy-binding $MODEL_ID \                                                                  
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.modelUser" \
  --region="${REGION}"
```

Data load, build embeddings, tune the model and test with a basic hard-coded query 
```bash
./run.sh
```

Congratulations! That is it you have deployed and tested you first Retrival Augmented Generative model.

## Web Chat App
The web based chat assumes that you have completed the quickstart and will deploy a simple 
Python Flask app with hard-coded login credentials - see below.

Collect the model and endpoint id below from the GCP Vertex Console or run:
```bash
gcloud ai models list --region=$REGION
gcloud ai endpoints list --region=$REGION 
```

When you have obtained the values update the below and export the parameters.
```bash
export MODEL_ID="2867371294100291584"
export MODEL_ENDPOINT_ID="1429685073992482816"
```

Set the application parameters - NOTE: replace with your own!
```bash
export MODEL_BASE="gemini-2.5-flash"
export ALLOWED_USER_PW="demo@example.com:P@ssw0Rd"
export SERVICE_APP_NAME="acams-ai-agent"
```

Initial deployment of the app. Define your runtime capacity (compute/memory) as required.
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_APP_NAME
gcloud run deploy $SERVICE_APP_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_APP_NAME \
  --platform managed \
  --region us-central1 \
  --min-instances=0 \
  --max-instances=3 \
  --concurrency=80 \
  --cpu=1 --memory=512Mi \
  --cpu-throttling \
  --allow-unauthenticated \
  --set-env-vars PROJECT_ID=$PROJECT_ID,LOCATION=$REGION,BASE_MODEL_NAME=gemini-2.5-flash,GCS_PREFIX=$GCS_PREFIX,ALLOW_CORS_ALL=1,ALLOWED_USERS=demo@example.com:P@ssw0Rd,TUNED_MODEL_NAME=projects/$PROJECT_ID/locations/us-central1/endpoints/$ENDPOINT_ID 
```

For running a local non-authenticated version
```bash
export PROJECT_ID=$PROJECT_ID
export GCS_PREFIX=gs://$BUCKET/$GCS_PREFIX
export DISABLE_AUTH=1
uvicorn backend.main:app --reload --port 8080
```

For updating/rebuilding and redeploy the service run:
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_APP_NAME
gcloud run services update $SERVICE_APP_NAME \
  --region=${REGION} \
  --service-account=${SA_EMAIL} \
  --set-env-vars PROJECT_ID=$PROJECT_ID,LOCATION=$REGION,BASE_MODEL_NAME=$MODEL_BASE,TUNED_MODEL_NAME=fccassistant-ofac-fatf-gemini-sft,GCS_PREFIX=$GCS_PREFIX,ALLOW_CORS_ALL=1,ALLOWED_USERS=$ALLOWED_USER_PW 
```
