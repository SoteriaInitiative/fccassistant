# RAG Chat on Cloud Run (Vertex + FAISS, Firebase Auth)

Minimal app that retrieves context from a FAISS index in GCS and calls Vertex AI. 
Includes a model switch (base/tuned) and Firebase-authenticated frontend.
This assumes Cloud Run and Firebase API/Servies have been activated, models/embeddings created.

## Deploy quickly

```bash
# Setting parameters
PROJECT_ID="fcc-assistant-mvp"
PROJECT_NUM="526463352197"
GCS_PREFIX="gs://fcc-assistant-84/index"
SERVICE_NAME="ask-simons"
SA_EMAIL="526463352197-compute@developer.gserviceaccount.com"
REGION="us-central1"
BUCKET="gs://fcc-assistant-84"
MODEL_ID="2867371294100291584"
MODEL_ENDPOINT_ID="1429685073992482816"
MODEL_BASE="gemini-2.5-flash"
ALLOWED_USER_PW="demo@example.com:P@ssw0Rd"
SERVICE_APP_NAME="acams-ai"

# Initial deployment of the app
gcloud builds submit --tag gcr.io/$PROJECT_ID/ask-simons
gcloud run deploy ask-simons \
  --image gcr.io/$PROJECT_ID/ask-simons \
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


Setting IAM permissions:
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

> NOTE: Ensure to create a Firebase app in Blaze (pay as you go) plan
> and define authorized domains in Firebase (Azthentication>Settings>Authorized Domains)
> using the Service URL link returned from Cloud run.

Replace the Firebase web config in `frontend/index.html`. 
For local dev, set `DISABLE_AUTH=1` and run:

```bash
export PROJECT_ID=your-proj
export GCS_PREFIX=gs://your-bucket/your-prefix
export DISABLE_AUTH=1
uvicorn backend.main:app --reload --port 8080
```

For updating/rebuilding and redeploy the service run:
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/ask-simons
gcloud run services update ask-simons \
  --region=${REGION} \
  --service-account=${SA_EMAIL} \
  --set-env-vars PROJECT_ID=$PROJECT_ID,LOCATION=$REGION,BASE_MODEL_NAME=gemini-2.5-flash,TUNED_MODEL_NAME=fccassistant-ofac-fatf-gemini-sft,GCS_PREFIX=$GCS_PREFIX,ALLOW_CORS_ALL=1,ALLOWED_USERS=demo@example.com:P@ssw0Rd 
```
