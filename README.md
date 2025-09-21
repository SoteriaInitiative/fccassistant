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

## Quickstart Google

The quickstart lets you train the model and run a hard-coded query as defined in ``app/query.py``.
For a web based chat app see further below.

> NOTE: Update the ``config.py`` with environment and application parameters that match the
> settings below or you will receive errors!

Set the initial environment parameters - do replace with your own!
```bash
export GCP_ACCOUNT="GCP_ACCOUNT_EMAIL"
export PROJECT_ID="GCP_PROJECT_ID"
export PROJECT_NUM="GCP_PROJECT_NUM"
export GCS_PREFIX="gs://GCP_BUCKET/index"
export SA_EMAIL="GCP_PROJECT_NUM-compute@developer.gserviceaccount.com"
export REGION="GCP_REGION"
export BUCKET="gs://GCP_BUCKET"
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

## Quickstart AWS
The quickstart lets you train the model and run a hard-coded query. For a web based chat app see further below.

The AWS account executing the below must have the following permissions assigned in IAM
because you will create, delete resources including service roles:
1. AdministratorAccess
2. AmazonBedrockFullAccess

Additionally, if you want to leverage fine-tuning (recommended), you will need to purchase
[Provisioned Throughput for Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/prov-thru-purchase.html)
(use 'No Commitment', 1 Model Unit) - even when just testing in contrast to GCP AWS charges for any inference.
Ensure that you delete the provisioned throughput immediately after use, it's expensive!

If you are on a new account, the above will only work if you have the right service quotas available. You may need to 
[request service quote increase ](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html)
for the service "Bedrock" and the quota "Model units no-commitment Provisioned Throughputs across custom models".
The request can take several days to complete.

The purchase is only possible after the custom model has been fine-tuned, given the significant cost implication
the ```deploy.sh``` stops after the model fine-tuning completes and asks if the deployment should continue and use
the custom model or not. When the fine-tuned model is not used, inference will be limited to the RAG/knowledge base.

> NOTE: Update the ``deploy.sh`` with environment and application parameters that match the
> settings below, or you will receive errors - the AWS configuration does _NOT_ use ```config.py```!

Set the initial environment parameters - do replace with your own!
```bash
export GCP_ACCOUNT="GCP_ACCOUNT_EMAIL"
export PROJECT_ID="GCP_PROJECT_ID"
export PROJECT_NUM="GCP_PROJECT_NUM"
export GCS_PREFIX="gs://GCP_BUCKET/index"
export SA_EMAIL="GCP_PROJECT_NUM-compute@developer.gserviceaccount.com"
export REGION="GCP_REGION"
export BUCKET="gs://GCP_BUCKET"
```

In case you plan to change any of the model functionality (different embedding, chunking or tuning settings)
you may need to change either file sin the ```lambdas``` or ```templates``` folders. If that is the case ensure
that you are running the below _before_ executing ```deploy.sh```
```bash
./build.sh
```
The build script creates important infrastructure used by the AWS CloudFormation stack, especially regarding the
creation of chunks and embeddings (model fine-tuning is fully controlled in ```tuning_nova.py```).

Execute the deployment - NOTE: Check ``deploy.sh`` for any local settings you may need!
```bash
./deploy.sh
```

Please consider removing the deployed solution when and if you do not need it. Both the provisioned throughput and the
Open Search Service are expensive resources and are charged even when not used.
```bash
./destory.sh
```


## Web Chat App Google
The web based chat assumes that you have completed the quickstart and will deploy a simple 
Python Flask app with hard-coded login credentials - see below.

1) Collect the model and endpoint id below from the GCP Vertex Console or run:
```bash
gcloud ai models list --region=$REGION
gcloud ai endpoints list --region=$REGION 
```

2) When you have obtained the values update the below and export the parameters.
```bash
export MODEL_ID="GCP_MODEL_ID"
export MODEL_ENDPOINT_ID="MODEL_ENDPOINT_ID"
```

3) Set the application parameters - NOTE: replace with your own!
```bash
export MODEL_BASE="gemini-2.5-flash"
export ALLOWED_USER_PW="demo@example.com:P@ssw0Rd"
export SERVICE_APP_NAME="acams-ai-agent"
```

4) Initial deployment of the app. Define your runtime capacity (compute/memory) as required.
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

## Web Chat App AWS
This section deploys the FastAPI app in `app/` to AWS App Runner and configures it to use Amazon Bedrock Knowledge
Bases for retrieval and generation (RAG). The Google path remains unchanged; AWS is enabled via environment variables.

Prerequisites
- You have run `bash deploy.sh embedding` and all previous steps and created the Bedrock Knowledge Base, 
OpenSearch Serverless collection, and optionally completed 
fine‑tuning with ```bash deploy.sh model``` and created Provisioned Throughput (PT).
- Your account has the required IAM permissions for App Runner, ECR, and Bedrock.

Scripted App Deployment
If you run ```bash deploy.sh all``` the web application will automatically be created for you, if the 
creation of provisioned throughput fails however you can simply run ```bash deploy.sh app```
to create a fresh AppRunner service. The service URL will be displayed at the end of the deployment.


Manual App Deployment
To deploy each of the web app components manuall follow the steps below.
1) Discover IDs from the deployed stack
```bash
export REGION=AWS_REGION
export PROFILE=AWS_ACCOUNT_PROFILE
export STACK=ofac-rag-kb

# Get KB ID
KB_ID=$(aws cloudformation list-stack-resources --stack-name "$STACK" --region "$REGION" --profile "$PROFILE" \
  | jq -r '.StackResourceSummaries[]|select(.ResourceType=="AWS::Bedrock::KnowledgeBase")|.PhysicalResourceId' | head -n1)
echo "KB_ID=$KB_ID"

# Optional: resolve the custom model + provisioned throughput (PT) created by deploy.sh
CUSTOM_MODEL_NAME=$(date +"ofac-nova-custom-%Y%m%d")
PT_ARN=$(aws bedrock list-provisioned-model-throughputs --region "$REGION" --profile "$PROFILE" \
  | jq -r --arg n "${CUSTOM_MODEL_NAME}-pt" '.provisionedModelSummaries[]?|select(.provisionedModelName==$n)|.provisionedModelArn' | head -n1)
echo "PT_ARN=$PT_ARN"

# Base model to fall back to when PT is absent
BASE_MODEL_ID="amazon.nova-micro-v1:0"
```

2) Build and push the container image to ECR
```bash
APP_IMAGE_NAME=fccassistant-app
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION" --profile "$PROFILE")
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$APP_IMAGE_NAME:latest"

aws ecr describe-repositories --repository-names "$APP_IMAGE_NAME" --region "$REGION" --profile "$PROFILE" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$APP_IMAGE_NAME" --region "$REGION" --profile "$PROFILE"

aws ecr get-login-password --region "$REGION" --profile "$PROFILE" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
docker build -t "$ECR_URI" ./app
docker push "$ECR_URI"
```

3) Create an App Runner instance role with Bedrock permissions (one‑time)
```bash
ROLE_NAME=apprunner-bedrock-exec-role
TRUST=$(jq -n '{Version:"2012-10-17",Statement:[{Effect:"Allow",Principal:{Service:"tasks.apprunner.amazonaws.com"},Action:"sts:AssumeRole"}]}')
POL=$(jq -n '{Version:"2012-10-17",Statement:[
  {Effect:"Allow",Action:["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream","bedrock:Retrieve","bedrock:RetrieveAndGenerate","bedrock:RetrieveAndGenerateStream"],Resource:"*"},
  {Effect:"Allow",Action:["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],Resource:"*"}
]}')
aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" --region "$REGION" --profile "$PROFILE" || true
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name BedrockInvoke --policy-document "$POL" --region "$REGION" --profile "$PROFILE"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --region "$REGION" --profile "$PROFILE" | jq -r .Role.Arn)
```

4) Create or update the App Runner service
```bash
SERVICE_NAME=fccassistant-web

# Choose model source: prefer PT if present
if [[ -n "$PT_ARN" ]]; then
  BEDROCK_MODEL_ARN="$PT_ARN"
  echo "Using Provisioned Throughput for inference: $BEDROCK_MODEL_ARN"
else
  BEDROCK_MODEL_ARN=""
  echo "No PT detected; app will use base model: $BASE_MODEL_ID"
fi

ENV_VARS="AWS_BEDROCK_MODE=1,AWS_REGION=$REGION,BEDROCK_KB_ID=$KB_ID,BEDROCK_MODEL_ARN=$BEDROCK_MODEL_ARN,BEDROCK_BASE_MODEL_ID=$BASE_MODEL_ID,ALLOW_CORS_ALL=1,ALLOWED_USERS=demo@example.com:P@ssw0Rd"

# Create the service (first run). Subsequent updates can use update-service.
aws apprunner create-service \
  --service-name "$SERVICE_NAME" \
  --region "$REGION" --profile "$PROFILE" \
  --source-configuration ImageRepository="{imageIdentifier=\"$ECR_URI\",imageRepositoryType=ECR,port=\"8080\"}" \
  --instance-configuration "{InstanceRoleArn=\"$ROLE_ARN\"}" \
  --health-check-configuration "{Protocol=HTTP,Path=\"/api/health\"}" \
  --tags Key=app,Value=fccassistant \
  --runtime-configuration EnvironmentVariables="[{Name=AWS_BEDROCK_MODE,Value=1},{Name=AWS_REGION,Value=$REGION},{Name=BEDROCK_KB_ID,Value=$KB_ID},{Name=BEDROCK_MODEL_ARN,Value=$BEDROCK_MODEL_ARN},{Name=BEDROCK_BASE_MODEL_ID,Value=$BASE_MODEL_ID},{Name=ALLOW_CORS_ALL,Value=1},{Name=ALLOWED_USERS,Value=demo@example.com:P@ssw0Rd}]"

# Wait and fetch the service URL
SERVICE_ARN=$(aws apprunner list-services --region "$REGION" --profile "$PROFILE" | jq -r ".ServiceSummaryList[]|select(.ServiceName==\"$SERVICE_NAME\")|.ServiceArn")
aws apprunner wait service-running --service-arn "$SERVICE_ARN" --region "$REGION" --profile "$PROFILE"
APP_URL=$(aws apprunner describe-service --service-arn "$SERVICE_ARN" --region "$REGION" --profile "$PROFILE" | jq -r .Service.ServiceUrl)
echo "App Runner URL: https://$APP_URL"
```

Notes
- The app auto‑selects the PT ARN when provided; otherwise it constructs a foundation model ARN from `BEDROCK_BASE_MODEL_ID` and region.
- Keep the KB role created by the stack; the app only needs Bedrock runtime permissions (granted via the App Runner instance role above).
- To update after code changes: build and push the image again, then run `aws apprunner update-service --service-arn $SERVICE_ARN --source-configuration ImageRepository={imageIdentifier=...}`.

Environment variables used by the app in AWS mode
- `AWS_BEDROCK_MODE=1` enables the AWS path.
- `AWS_REGION` (e.g., us-east-1).
- `BEDROCK_KB_ID` — Knowledge Base ID from the stack.
- `BEDROCK_MODEL_ARN` — Provisioned Throughput ARN (optional). If omitted, the app uses `BEDROCK_BASE_MODEL_ID`.
- `BEDROCK_BASE_MODEL_ID` — e.g., `amazon.nova-micro-v1:0`.
- `ALLOWED_USERS` — comma‑separated email:password pairs for login.
