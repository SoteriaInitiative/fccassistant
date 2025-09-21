#!/bin/bash

set -euo pipefail

# Dependency check
log() { printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { echo "Error: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found. Please install it."; }

require aws
require jq

# Error context surfaced on failure
CURRENT_STAGE="init"
trap 'echo "[ERROR] Stage=$CURRENT_STAGE region=$region profile=$profile — last command failed. See AWS error above." >&2' ERR

stage_info() { # $1=name, $2=extra
  echo "[stage:$1] region=$region profile=$profile ${2:-}";
}

# Resolve a valid customizable base model ID for SFT in this region/account
resolve_customizable_model_id() {
  if [[ -n "${BASE_MODEL_OVERRIDE:-}" ]]; then
    echo "$BASE_MODEL_OVERRIDE"
    return 0
  fi
  local ids chosen
  ids=$(aws bedrock list-foundation-models --region "$region" --profile "$profile" \
        --by-customization-type FINE_TUNING \
        | jq -r '.modelSummaries[]?.modelId' 2>/dev/null)
  if [[ -z "$ids" ]]; then
    echo "ERROR: No customizable base models reported by Bedrock in $region for this account." >&2
    echo "       Try another region or set BASE_MODEL_OVERRIDE to a known-good ID (e.g., amazon.nova-micro-v1:0)." >&2
    return 1
  fi
  # Prefer Nova; otherwise first customizable model
  chosen=$(echo "$ids" | awk '/nova/ {print; exit} END{if(NR==0){exit 1}}')
  if [[ -z "$chosen" ]]; then
    chosen=$(echo "$ids" | head -n1)
  fi
  # Strip token-window suffix (e.g., :128k) if present
  echo "$chosen" | sed -E 's/:128k$//'
}

preflight_model_id() {
  local mid="$1"
  if ! aws bedrock get-foundation-model --region "$region" --profile "$profile" --model-identifier "$mid" >/dev/null 2>&1; then
    echo "ERROR: Base model identifier '$mid' is not valid or not accessible in region $region." >&2
    return 1
  fi
  return 0
}

# Create Provisioned Throughput (handles both legacy and new CLI flag names)
create_pt() {
  local name="$1" model_arn="$2"; shift || true; shift || true
  # Try legacy flags first
  local out rc
  out=$(aws bedrock create-provisioned-model-throughput \
    --provisioned-model-name "$name" \
    --model-arn "$model_arn" \
    --desired-model-units 1 \
    --no-cli-pager --output json \
    --region "$region" --profile "$profile" 2>&1)
  rc=$?
  if (( rc != 0 )) && echo "$out" | grep -qiE 'model-units|model-id|the following arguments are required'; then
    # Retry with new flag names
    out=$(aws bedrock create-provisioned-model-throughput \
      --provisioned-model-name "$name" \
      --model-id "$model_arn" \
      --model-units 1 \
      --no-cli-pager --output json \
      --region "$region" --profile "$profile" 2>&1)
    rc=$?
  fi
  # Defensive: if CLI printed nothing and rc==0, treat as failure so caller reports
  if [[ -z "$out" && $rc -eq 0 ]]; then
    out="(no output from AWS CLI)"; rc=1
  fi
  printf '%s' "$out"
  return $rc
}

# Parameters and user inputs (edit or pass via env)
stack_name="${stack_name:-ofac-rag-kb}"
source_docs_s3_bucket="${source_docs_s3_bucket:-ofac-rag-docs}"
source_doc_local_dir="${source_doc_local_dir:-data}"
region="${region:-us-east-1}"
profile="${profile:-FCCAssistant}"
params_file="${params_file:-./params.json}"
LOCAL_ARTIFACT_DIR="${artifacts_local_dir:-artifacts}"
LOCAL_TEMPLATES_DIR="${templates_local_dir:-templates}"
# Pre-chunk corpus bucket/prefix for Bedrock KB (when using pre-chunked files)
corpus_s3_bucket="${corpus_s3_bucket:-ofac-rag-corpus}"
corpus_prefix="${corpus_prefix:-corpus}"
# Base model for tuning
BASE_MODEL_ID="amazon.nova-micro-v1:0:128k"
# Tuning bucket/prefix for SFT assets
tuning_s3_bucket="${tuning_s3_bucket:-ofac-tuning}"
tuning_prefix="${tuning_prefix:-tuning}"
# App Runner service name (used by status/app stages)
apprunner_service_name="${apprunner_service_name:-fccassistant-web}"
# App Runner exec role name (used by app/status)
apprunner_exec_role_name="${apprunner_exec_role_name:-apprunner-bedrock-exec-role}"
# App Runner ECR access role (for pulling from private ECR)
apprunner_ecr_access_role_name="${apprunner_ecr_access_role_name:-apprunner-ecr-access-role}"
# Web app auth (demo credentials by default)
allowed_user_pw="${ALLOWED_USER_PW:-demo@example.com:P@ssw0Rd}"

echo "[*] Verifying deployment parameters..."
echo "[X] Profile Name: ${profile}"
echo "[X] Region: ${region}"
echo "NOTE: Account must have AdministratorAccess & AmazonBedrockFullAccess permissions!"

# Normalize an ID by taking the last token after '/' or ':'.
trim_id() {
  local s="$1"
  s="${s%/}"
  echo "$s" | sed -E 's#.*/##; s#.*:##'
}

# For "KBID|DSID": split and normalize both sides. Prints "KBID DSID".
normalize_ds_pair() {
  local pair="$1"
  local left="${pair%%|*}"
  local right="${pair#*|}"
  left="$(trim_id "$left")"
  right="$(trim_id "$right")"
  printf "%s %s\n" "$left" "$right"
}

list_resources_page() {
  local stack="$1" token="${2:-}"
  if [[ -n "$token" ]]; then
    aws cloudformation list-stack-resources --region "$region" --profile "$profile"\
      --stack-name "$stack" --starting-token "$token"
  else
    aws cloudformation list-stack-resources --region "$region" --profile "$profile"\
      --stack-name "$stack"
  fi
}

# BFS over nested stacks; prints JSONL with: path, depth, logical, physical, type, status
expand_stack() {
  local stack="$1" path="$2" depth="$3"
  local next="" out=""
  local -a children
  children=()

  # paginate resources of this stack
  while :; do
    if [[ -n "$next" ]]; then
      out=$(aws cloudformation list-stack-resources --region "$region" --profile "$profile" --stack-name "$stack" --starting-token "$next")
    else
      out=$(aws cloudformation list-stack-resources --region "$region" --profile "$profile" --stack-name "$stack")
    fi

    # print this page's resources
    echo "$out" | jq -c --arg path "$path" --argjson depth "$depth" '
      .StackResourceSummaries[]
      | {
          path: $path,
          depth: $depth,
          logical: .LogicalResourceId,
          physical: (.PhysicalResourceId // ""),
          type: .ResourceType,
          status: .ResourceStatus
        }'

    # collect child stacks from this page
    while read -r child; do
      [[ -n "$child" ]] && children+=("$child")
    done < <(echo "$out" | jq -r '
      .StackResourceSummaries[]?
      | select(.ResourceType=="AWS::CloudFormation::Stack")
      | .PhysicalResourceId // empty')

    next=$(echo "$out" | jq -r '.NextToken // empty')
    [[ -z "$next" ]] && break
  done

  # recurse into child stacks (if any)
  if ((${#children[@]} > 0)); then
    local child
    for child in "${children[@]}"; do
      expand_stack "$child" "$path/$child" "$((depth+1))"
    done
  fi
}

gather_all_resources() { jq -s 'flatten' < <(expand_stack "$1" "$1" 0); }

start_ingestion_jobs() {
  local root="$1"
  log "Scanning stack hierarchy for Bedrock KnowledgeBases and DataSources…"
  local all
  all="$(gather_all_resources "$root")"

  IFS=$'\n' read -r -d '' -a kb_ids_raw < <(echo "$all" | jq -r '.[] | select(.type=="AWS::Bedrock::KnowledgeBase") | .physical | select(length>0)' && printf '\0') || true
  kb_ids=()
  for k in "${kb_ids_raw[@]:-}"; do
    [[ -z "$k" ]] && continue
    k="${k%%|*}"
    kb_ids+=("$(trim_id "$k")")
  done

  IFS=$'\n' read -r -d '' -a ds_pairs_raw < <(echo "$all" | jq -r '.[] | select(.type=="AWS::Bedrock::DataSource") | .physical | select(length>0)' && printf '\0') || true
  ds_pairs=("${ds_pairs_raw[@]:-}")

  if ((${#kb_ids[@]}==0)); then
    log "No AWS::Bedrock::KnowledgeBase resources found."
    return 0
  fi

  log "Found ${#kb_ids[@]} KnowledgeBase id(s): ${kb_ids[*]}"
  log "Found ${#ds_pairs[@]:-0} DataSource pair(s)."

  local any_started=0
  local pair kb_id ds_id

  for pair in "${ds_pairs[@]:-}"; do
    [[ -z "$pair" ]] && continue
    read kb_id ds_id < <(normalize_ds_pair "$pair")
    if printf '%s\n' "${kb_ids[@]}" | grep -qx "$kb_id"; then
      log "Starting ingestion job for KB=$kb_id, DataSource=$ds_id …"
      aws bedrock-agent start-ingestion-job \
        --region "$region" --profile "$profile" \
        --knowledge-base-id "$kb_id" \
        --data-source-id "$ds_id" >/dev/null
      any_started=1
    fi
  done

  if [[ "$any_started" -eq 1 ]]; then
    log "Ingestion job(s) started. To list jobs:"
    for pair in "${ds_pairs[@]:-}"; do
      [[ -z "$pair" ]] && continue
      read kb_id ds_id < <(normalize_ds_pair "$pair")
      if printf '%s\n' "${kb_ids[@]}" | grep -qx "$kb_id"; then
        echo "aws bedrock-agent list-ingestion-jobs --region $region --profile $profile --knowledge-base-id $kb_id --data-source-id $ds_id"
      fi
    done
  else
    log "No matching KnowledgeBase/DataSource pairs detected to start ingestion."
  fi
}

main() {
  CURRENT_STAGE="all:prepare"
  stage_info "all:prepare" "stack=$stack_name"
  # Deploy CloudFormation templates
  echo "Starting CloudFormation template deployment..."
  AWS="aws --output=text --region ${region} --profile ${profile}"
  ACCOUNT_ID=$(aws --output=text --region ${region} --profile ${profile} sts get-caller-identity --query 'Account')
  DEPLOYMENT_BUCKET="${DEPLOYMENT_BUCKET:-ofac-rag-deployment-${ACCOUNT_ID}-${region}}"
  # Create deployment bucket if needed, with proper LocationConstraint
  if ! aws --region "${region}" --profile "${profile}" s3api head-bucket --bucket "${DEPLOYMENT_BUCKET}" >/dev/null 2>&1; then
    if [[ "${region}" == "us-east-1" ]]; then
      aws --region "${region}" --profile "${profile}" s3api create-bucket --bucket "${DEPLOYMENT_BUCKET}"
    else
      aws --region "${region}" --profile "${profile}" s3api create-bucket --bucket "${DEPLOYMENT_BUCKET}" --create-bucket-configuration LocationConstraint="${region}"
    fi
  else
    log "Deployment bucket already exists: ${DEPLOYMENT_BUCKET}"
  fi
  aws s3 cp ${LOCAL_ARTIFACT_DIR} s3://${DEPLOYMENT_BUCKET}/${LOCAL_ARTIFACT_DIR}/ --recursive --profile ${profile} --region ${region}
  # Update the templates with the deployment bucket path...
  yml_file_contents=$(cat templates/main-template-out-tmp.yml)
  updated_yml_file_contents=$(sed "s/pDeploymentBucket/${DEPLOYMENT_BUCKET}/g"<<< "$yml_file_contents")
  echo "$updated_yml_file_contents" >templates/main-template-out.yml
  yml_file_contents=$(cat templates/oss-infra-template-tmp.template)
  updated_yml_file_contents=$(sed "s/pDeploymentBucket/${DEPLOYMENT_BUCKET}/g"<<< "$yml_file_contents")
  echo "$updated_yml_file_contents" >templates/oss-infra-template.template
  aws s3 cp ${LOCAL_TEMPLATES_DIR} s3://${DEPLOYMENT_BUCKET}/${LOCAL_TEMPLATES_DIR}/ --recursive --profile ${profile} --region ${region}
  template_s3_url="https://${DEPLOYMENT_BUCKET}.s3.${region}.amazonaws.com/templates/main-template-out.yml"
  echo "Completed CloudFormation template deployment."

  # Create and copy source data bucket
  echo "Creating source document storage: s3://$source_docs_s3_bucket"
  if ! aws --region "${region}" --profile "${profile}" s3api head-bucket --bucket "${source_docs_s3_bucket}" >/dev/null 2>&1; then
    if [[ "${region}" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "$source_docs_s3_bucket" --region "$region" --profile "$profile"
    else
      aws s3api create-bucket --bucket "$source_docs_s3_bucket" --region "$region" --profile "$profile" --create-bucket-configuration LocationConstraint="$region"
    fi
  else
    log "Source docs bucket already exists: ${source_docs_s3_bucket}"
  fi
  aws s3 cp ./$source_doc_local_dir s3://$source_docs_s3_bucket --recursive --recursive --exclude "*" --include "*.pdf" --region $region --profile $profile

  # Create corpus bucket for pre-chunked files and run chunker in AWS mode
  echo "Creating corpus storage for pre-chunked files: s3://$corpus_s3_bucket"
  if ! aws --region "${region}" --profile "${profile}" s3api head-bucket --bucket "${corpus_s3_bucket}" >/dev/null 2>&1; then
    if [[ "${region}" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "$corpus_s3_bucket" --region "$region" --profile "$profile"
    else
      aws s3api create-bucket --bucket "$corpus_s3_bucket" --region "$region" --profile "$profile" --create-bucket-configuration LocationConstraint="$region"
    fi
  else
    log "Corpus bucket already exists: ${corpus_s3_bucket}"
  fi
  echo "Running pdf_ingest in AWS mode to produce pre-chunked .txt files"
  AWS_PROFILE="$profile" AWS_REGION="$region" python -m model.pdf_ingest aws \
    --aws-input-bucket "$source_docs_s3_bucket" \
    --aws-input-prefix "" \
    --aws-output-bucket "$corpus_s3_bucket" \
    --aws-output-prefix "$corpus_prefix" \
    --aws-region "$region" \
    --aws-profile "$profile" \
    --upload-mode files

  echo "NOTE: To have KB ingest pre-chunked files, set Q01pInputBucketName to '$corpus_s3_bucket' and Q04pChunkingStrategy to 'No chunking' in params.json before stack creation."

  echo "Validating template: $template_s3_url"
  aws cloudformation validate-template \
    --template-url "$template_s3_url" \
    --region "$region" \
    --profile "$profile"
  echo "Template is syntactically valid."

  echo "Creating stack: $stack_name in $region"
  CURRENT_STAGE="all:stack-create"
  aws cloudformation create-stack \
    --stack-name "$stack_name" \
    --template-url "$template_s3_url" \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
    --parameters "file://$params_file" \
    --region "$region" \
    --profile "$profile"

  echo "Waiting for stack to reach CREATE_COMPLETE…"
  CURRENT_STAGE="all:stack-wait"
  aws cloudformation wait stack-create-complete \
    --stack-name "$stack_name" \
    --region "$region" \
    --profile "$profile"

  echo "SUCCESS. Stack outputs:"
  aws cloudformation describe-stacks \
    --stack-name "$stack_name" \
    --query "Stacks[0].Outputs" \
    --output table \
    --region "$region" \
    --profile "$profile"

  start_ingestion_jobs "$stack_name"

  # -----------------------
  # Supervised fine-tuning
  # -----------------------
  echo "Creating tuning storage bucket: s3://$tuning_s3_bucket"
  stage_info "all:model" "train_uri=s3://${tuning_s3_bucket}/${tuning_prefix}/tuning_dataset_contents.jsonl base_model=amazon.nova-micro-v1:0"
  if ! aws --region "${region}" --profile "${profile}" s3api head-bucket --bucket "${tuning_s3_bucket}" >/dev/null 2>&1; then
    if [[ "${region}" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "$tuning_s3_bucket" --region "$region" --profile "$profile"
    else
      aws s3api create-bucket --bucket "$tuning_s3_bucket" --region "$region" --profile "$profile" --create-bucket-configuration LocationConstraint="$region"
    fi
  else
    log "Tuning bucket already exists: ${tuning_s3_bucket}"
  fi

  echo "Generating tuning data (JSONL) and uploading to s3://$tuning_s3_bucket/$tuning_prefix"
  AWS_PROFILE="$profile" AWS_REGION="$region" python -m model.generate_tuning_data aws \
    --aws-output-bucket "$tuning_s3_bucket" \
    --aws-output-prefix "$tuning_prefix" \
    --aws-region "$region" \
    --aws-profile "$profile"

  echo "Creating the Bedrock service role for supervised fine tuning (SFT) job execution"
  SFT_ROLE_NAME="ofac-bedrock-sft-role"
  SFT_POLICY_NAME="SFTAccess"
  TRUST=$(jq -n --arg acct "$ACCOUNT_ID" --arg region "$region" '{Version:"2012-10-17",Statement:
  [{Effect:"Allow",Principal:{Service:"bedrock.amazonaws.com"},Action:"sts:AssumeRole",Condition:{StringEquals:{"aws:SourceAccount":$acct},ArnLike:{"aws:SourceArn":
  ("arn:aws:bedrock:"+$region+":"+$acct+":model-customization-job/*")}}}]}');
  POLICY=$(jq -n --arg b "$tuning_s3_bucket" --arg p "$tuning_prefix" '{Version:"2012-10-17",Statement:[{Effect:"Allow",Action:["s3:ListBucket"],Resource:("arn:aws:s3:::"+$b),Condition:{StringLike:{"s3:prefix":[($p+"/*"),$p]}}},{Effect:"Allow",Action:
  ["s3:GetObject"],Resource:("arn:aws:s3:::"+$b+"/"+$p+"/*")},{Effect:"Allow",Action:["s3:PutObject"],Resource:("arn:aws:s3:::"+$b+"/"+$p+"/outputs/*")}]}');
  aws iam create-role --role-name "$SFT_ROLE_NAME" \
    --assume-role-policy-document "$TRUST" \
    --description "Bedrock SFT role" \
    --profile "$profile" --region "$region" >/dev/null
  aws iam put-role-policy --role-name "$SFT_ROLE_NAME" \
    --policy-name "$SFT_POLICY_NAME" \
    --policy-document "$POLICY" \
    --profile "$profile" --region "$region"
  echo "SFT Role created: ${SFT_ROLE_NAME} with:"
  #echo "ROLE POLICY:$TRUST"
  #echo "ACCESS POLICY: $POLICY"

  echo "Waiting role creation..."
  CURRENT_STAGE="all:wait-sft-role"; aws iam wait role-exists --role-name "$SFT_ROLE_NAME" --profile "$profile" --region "$region"
  sleep 20

  echo "Submitting Bedrock SFT job (adjust base model and role ARN as needed)"
  TRAINING_URI="s3://${tuning_s3_bucket}/${tuning_prefix}/tuning_dataset_contents.jsonl"
  OUTPUT_URI="s3://${tuning_s3_bucket}/${tuning_prefix}/outputs/"
  JOB_NAME="ofac-sft-$(date +%Y%m%d-%H%M%S)"
  CUSTOM_MODEL_NAME="ofac-nova-custom-$(date +%Y%m%d)"
  #echo "Resolving customizable base model ID…"
  #BASE_MODEL_ID="$(resolve_customizable_model_id)" || die "No customizable base model available; set BASE_MODEL_OVERRIDE to a valid model ID (e.g., amazon.nova-micro-v1:0)"
  echo "  Using base model: $BASE_MODEL_ID"
  preflight_model_id "$BASE_MODEL_ID" || die "Invalid base model ID: $BASE_MODEL_ID"
  BEDROCK_SFT_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SFT_ROLE_NAME}"
  CURRENT_STAGE="all:sft-submit"; SFT_CREATE_JSON=$(AWS_PROFILE="$profile" AWS_REGION="$region" python -m model.tune_nova \
    --job-name "$JOB_NAME" \
    --custom-model-name "$CUSTOM_MODEL_NAME" \
    --role-arn "$BEDROCK_SFT_ROLE_ARN" \
    --base-model-id "$BASE_MODEL_ID" \
    --training-s3-uri "$TRAINING_URI" \
    --output-s3-uri "$OUTPUT_URI" \
    --region "$region" \
    --profile "$profile")
  echo "$SFT_CREATE_JSON"

  JOB_ARN=$(echo "$SFT_CREATE_JSON" | jq -r '.jobArn // empty')
  if [[ -z "$JOB_ARN" ]]; then
    echo "Could not parse jobArn from response; attempting lookup by name…"
    JOB_ARN=$(aws bedrock list-model-customization-jobs --region "$region" --profile "$profile" \
      | jq -r --arg n "$JOB_NAME" '.modelCustomizationJobSummaries[]?|select(.jobName==$n)|.jobArn' | head -n1)
  fi
  if [[ -z "$JOB_ARN" ]]; then
    echo "ERROR: Unable to determine SFT job ARN for $JOB_NAME" >&2
    return 1
  fi

  echo "Waiting for SFT job completion (jobArn=$JOB_ARN)…"
  CUSTOM_MODEL_ARN=""
  tries=0
  CURRENT_STAGE="all:sft-wait"; while :; do
    JOB_JSON=$(aws bedrock get-model-customization-job --job-identifier "$JOB_ARN" --region "$region" --profile "$profile") || JOB_JSON='{}'
    JOB_STATUS=$(echo "$JOB_JSON" | jq -r '.status // ""')
    VAL_STATUS=$(echo "$JOB_JSON" | jq -r '.statusDetails.validationDetails.status // "-"')
    TRN_STATUS=$(echo "$JOB_JSON" | jq -r '.statusDetails.trainingDetails.status // "-"')
    CUSTOM_MODEL_ARN=$(echo "$JOB_JSON" | jq -r '.customModelArn // ""')
    printf "[SFT status=%s validation=%s training=%s]\n" "$JOB_STATUS" "$VAL_STATUS" "$TRN_STATUS"
    case "$JOB_STATUS" in
      Completed)
        echo "SFT Completed. Custom model name: $CUSTOM_MODEL_NAME"
        break;;
      Failed|Stopped|Stopping)
        echo "SFT job ended with status $JOB_STATUS" >&2
        break;;
      *)
        ((tries++)) || true
        if (( tries > 360 )); then
          echo "ERROR: SFT did not complete within ~3 hours." >&2
          break
        fi
        sleep 30;;
    esac
  done

  # After SFT, optionally create Provisioned Throughput (1 MU, no commitment)
  PT_ARN=""
  if [[ "$JOB_STATUS" == "Completed" ]]; then
    # Ensure we have the custom model ARN; if not, fetch by name
    if [[ -z "$CUSTOM_MODEL_ARN" ]]; then
      CUSTOM_MODEL_ARN=$(aws bedrock list-custom-models --region "$region" --profile "$profile" \
        | jq -r --arg n "$CUSTOM_MODEL_NAME" '.modelSummaries[]?|select(.modelName==$n)|.modelArn' | head -n1)
    fi
    if [[ -z "$CUSTOM_MODEL_ARN" ]]; then
      echo "WARN: Could not resolve custom model ARN for $CUSTOM_MODEL_NAME; skipping provisioning." >&2
    else
      read -r -p "Provision 1 MU for '$CUSTOM_MODEL_NAME' now? This incurs cost. [y/N] " ans
      ans_lc=$(printf '%s' "$ans" | tr '[:upper:]' '[:lower:]')
      if [[ "$ans_lc" == "y" || "$ans_lc" == "yes" ]]; then
        PT_NAME="${CUSTOM_MODEL_NAME}-pt"
        echo "Creating Provisioned Throughput (no commitment) named $PT_NAME …"
        CURRENT_STAGE="all:pt-create"; PT_RESP=$(create_pt "$PT_NAME" "$CUSTOM_MODEL_ARN" ); PT_RC=$?
        PT_ARN=$(echo "$PT_RESP" | jq -r '.provisionedModelArn // empty' 2>/dev/null || true)
        if [[ $PT_RC -eq 0 && -n "$PT_ARN" ]]; then
          echo "Provisioned model ARN: $PT_ARN"
          tries=0
          CURRENT_STAGE="all:pt-wait"; while :; do
            PT_JSON=$(aws bedrock get-provisioned-model-throughput --provisioned-model-arn "$PT_ARN" --region "$region" --profile "$profile" || true)
            PT_STATUS=$(echo "$PT_JSON" | jq -r '.status // ""')
            printf "[PT status=%s]\n" "$PT_STATUS"
            case "$PT_STATUS" in
              InService)
                echo "Provisioned throughput is ready."
                break;;
              Failed)
                echo "Provisioned throughput failed." >&2
                PT_ARN=""
                break;;
              *)
                ((tries++)) || true
                if (( tries > 120 )); then
                  echo "WARN: Provisioned throughput not InService after ~10 minutes; continuing without it." >&2
                  PT_ARN=""
                  break
                fi
                sleep 5;;
            esac
          done
        else
          echo "ERROR: Failed to create provisioned throughput. AWS says:" >&2
          echo "$PT_RESP" >&2
        fi
      fi
    fi
  fi

  # Run test query against KB using PT (if available) else base model
  TEST_Q='Given the following ownership structure: Co A is 3% owned by SDN A Co A is 47% owned by Co B Co A is 50% owned by Co C Co B is 42% owned by SDN B Co B is 52% owned by Co D Co D is 50% owned by Person C Co D is 50% owned by SDN D Co C is 52% owned by Trust Co C is 48% owned by Co E Co E is 19% owned by Person E Co E is 81% owned by SDN F Trust is managed by Settlor Trust is managed by Trustee Trust is managed by Beneficiary  Further given that Company A Ltd (Co A) is registered in Country X (15% BO threshold), resolve: Task 1: Identify all beneficial owners (FATF Recs 24/25). Task 2: SDN A/B/D/F are on OFAC SDN List. Is Co A subject to US sanctions, and which intermediate/owners contribute? Refer to OFAC FAQ 401 for guidance.'
  CURRENT_STAGE="all:test-query"; ALL_RES=$(gather_all_resources "$stack_name")
  KB_ID=$(echo "$ALL_RES" | jq -r '.[]|select(.type=="AWS::Bedrock::KnowledgeBase")|.physical' | head -n1)
  if [[ -z "$KB_ID" ]]; then
    echo "WARN: Could not locate Knowledge Base ID; skipping test query." >&2
  else
    if [[ -n "$PT_ARN" ]]; then
      MODEL_FOR_QUERY="$PT_ARN"
      echo "Running test query via Provisioned Throughput ($MODEL_FOR_QUERY) …"
    else
      MODEL_FOR_QUERY="arn:aws:bedrock:${region}::foundation-model/${BASE_MODEL_ID}"
      echo "Running test query via base model ($MODEL_FOR_QUERY) …"
    fi
    REQ_FILE=$(mktemp)
    jq -n --arg kb "$KB_ID" --arg model "$MODEL_FOR_QUERY" --arg q "$TEST_Q" '{knowledgeBaseId:$kb, modelArn:$model, retrievalQuery:{text:$q}}' > "$REQ_FILE"
    aws bedrock-agent-runtime retrieve-and-generate \
      --region "$region" --profile "$profile" \
      --input file://"$REQ_FILE" \
      | jq -r '.output?.text // .citations // .'
    rm -f "$REQ_FILE"
  fi
}

# -----------------------
# Targeted stages (optional)
# -----------------------
stage_data() {
  echo "[stage:data] Upload PDFs and pre-chunk to corpus…"
  # Create and copy source data bucket
  if ! aws --region "${region}" --profile "${profile}" s3api head-bucket --bucket "${source_docs_s3_bucket}" >/dev/null 2>&1; then
    if [[ "${region}" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "$source_docs_s3_bucket" --region "$region" --profile "$profile"
    else
      aws s3api create-bucket --bucket "$source_docs_s3_bucket" --region "$region" --profile "$profile" --create-bucket-configuration LocationConstraint="$region"
    fi
  fi
  aws s3 cp ./$source_doc_local_dir s3://$source_docs_s3_bucket --recursive --exclude "*" --include "*.pdf" --region $region --profile $profile
  # Corpus bucket and chunking
  if ! aws --region "${region}" --profile "${profile}" s3api head-bucket --bucket "${corpus_s3_bucket}" >/dev/null 2>&1; then
    if [[ "${region}" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "$corpus_s3_bucket" --region "$region" --profile "$profile"
    else
      aws s3api create-bucket --bucket "$corpus_s3_bucket" --region "$region" --profile "$profile" --create-bucket-configuration LocationConstraint="$region"
    fi
  fi
  AWS_PROFILE="$profile" AWS_REGION="$region" python -m model.pdf_ingest aws \
    --aws-input-bucket "$source_docs_s3_bucket" \
    --aws-input-prefix "" \
    --aws-output-bucket "$corpus_s3_bucket" \
    --aws-output-prefix "$corpus_prefix" \
    --aws-region "$region" \
    --aws-profile "$profile" \
    --upload-mode files
}

stage_embedding() {
  echo "[stage:embedding] Deploy KB + OSS and start ingestion…"
  # Prepare templates and deployment bucket
  ACCOUNT_ID=$(aws --output=text --region ${region} --profile ${profile} sts get-caller-identity --query 'Account')
  DEPLOYMENT_BUCKET="${DEPLOYMENT_BUCKET:-ofac-rag-deployment-${ACCOUNT_ID}-${region}}"
  if ! aws --region "${region}" --profile "${profile}" s3api head-bucket --bucket "${DEPLOYMENT_BUCKET}" >/dev/null 2>&1; then
    if [[ "${region}" == "us-east-1" ]]; then
      aws --region "${region}" --profile "${profile}" s3api create-bucket --bucket "${DEPLOYMENT_BUCKET}"
    else
      aws --region "${region}" --profile "${profile}" s3api create-bucket --bucket "${DEPLOYMENT_BUCKET}" --create-bucket-configuration LocationConstraint="${region}"
    fi
  fi
  aws s3 cp ${LOCAL_ARTIFACT_DIR} s3://${DEPLOYMENT_BUCKET}/${LOCAL_ARTIFACT_DIR}/ --recursive --profile ${profile} --region ${region}
  updated_yml_file_contents=$(sed "s/pDeploymentBucket/${DEPLOYMENT_BUCKET}/g" templates/main-template-out-tmp.yml)
  echo "$updated_yml_file_contents" >templates/main-template-out.yml
  updated_yml_file_contents=$(sed "s/pDeploymentBucket/${DEPLOYMENT_BUCKET}/g" templates/oss-infra-template-tmp.template)
  echo "$updated_yml_file_contents" >templates/oss-infra-template.template
  aws s3 cp ${LOCAL_TEMPLATES_DIR} s3://${DEPLOYMENT_BUCKET}/${LOCAL_TEMPLATES_DIR}/ --recursive --profile ${profile} --region ${region}
  template_s3_url="https://${DEPLOYMENT_BUCKET}.s3.${region}.amazonaws.com/templates/main-template-out.yml"
  echo "Validating template: $template_s3_url"
  aws cloudformation validate-template --template-url "$template_s3_url" --region "$region" --profile "$profile" >/dev/null 2>&1

  echo "Creating stack: $stack_name in $region"
  aws cloudformation create-stack \
    --stack-name "$stack_name" \
    --template-url "$template_s3_url" \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
    --parameters "file://$params_file" \
    --region "$region" \
    --profile "$profile"
  echo "Waiting for stack to reach CREATE_COMPLETE…"
  aws cloudformation wait stack-create-complete --stack-name "$stack_name" --region "$region" --profile "$profile"
  echo "SUCCESS. Stack outputs:"
  aws cloudformation describe-stacks --stack-name "$stack_name" --query "Stacks[0].Outputs" --output table --region "$region" --profile "$profile"
  start_ingestion_jobs "$stack_name"
}

stage_model_only() {
  echo "[stage:model] Generate tuning data and submit SFT job…"
  ACCOUNT_ID=$(aws --output=text --region ${region} --profile ${profile} sts get-caller-identity --query 'Account')
  # Create tuning bucket if needed
  echo "Creating tuning storage bucket: s3://$tuning_s3_bucket"
  if ! aws --region "${region}" --profile "${profile}" s3api head-bucket --bucket "${tuning_s3_bucket}" >/dev/null 2>&1; then
    if [[ "${region}" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "$tuning_s3_bucket" --region "$region" --profile "$profile"
    else
      aws s3api create-bucket --bucket "$tuning_s3_bucket" --region "$region" --profile "$profile" --create-bucket-configuration LocationConstraint="$region"
    fi
  else
    log "Tuning bucket already exists: ${tuning_s3_bucket}"
  fi

  echo "Generating tuning data (JSONL) and uploading to s3://$tuning_s3_bucket/$tuning_prefix"
  CURRENT_STAGE="model:generate"; AWS_PROFILE="$profile" AWS_REGION="$region" python -m model.generate_tuning_data aws \
    --aws-output-bucket "$tuning_s3_bucket" \
    --aws-output-prefix "$tuning_prefix" \
    --aws-region "$region" \
    --aws-profile "$profile"

  echo "Creating the Bedrock service role for supervised fine tuning (SFT) job execution"
  SFT_ROLE_NAME="ofac-bedrock-sft-role"
  SFT_POLICY_NAME="SFTAccess"
  TRUST=$(jq -n --arg acct "$ACCOUNT_ID" '{Version:"2012-10-17",Statement:[{Effect:"Allow",Principal:{Service:"bedrock.amazonaws.com"},Action:"sts:AssumeRole",Condition:{StringEquals:{"aws:SourceAccount":$acct}}}]}')
  POLICY=$(jq -n --arg b "$tuning_s3_bucket" --arg p "$tuning_prefix" '{Version:"2012-10-17",Statement:[{Effect:"Allow",Action:["s3:ListBucket"],Resource:("arn:aws:s3:::"+$b),Condition:{StringLike:{"s3:prefix":[($p+"/*"),$p]}}},{Effect:"Allow",Action:["s3:GetObject"],Resource:("arn:aws:s3:::"+$b+"/"+$p+"/*")},{Effect:"Allow",Action:["s3:PutObject"],Resource:("arn:aws:s3:::"+$b+"/"+$p+"/outputs/*")}]}')
  aws iam create-role --role-name "$SFT_ROLE_NAME" --assume-role-policy-document "$TRUST" --description "Bedrock SFT role" --profile "$profile" --region "$region" >/dev/null 2>&1 || true
  aws iam put-role-policy --role-name "$SFT_ROLE_NAME" --policy-name "$SFT_POLICY_NAME" --policy-document "$POLICY" --profile "$profile" --region "$region"
  echo "Waiting role creation..."
  CURRENT_STAGE="model:wait-role"; aws iam wait role-exists --role-name "$SFT_ROLE_NAME" --profile "$profile" --region "$region"; sleep 10

  echo "Submitting Bedrock SFT job"
  TRAINING_URI="s3://${tuning_s3_bucket}/${tuning_prefix}/tuning_dataset_contents.jsonl"
  OUTPUT_URI="s3://${tuning_s3_bucket}/${tuning_prefix}/outputs/"
  JOB_NAME="ofac-sft-$(date +%Y%m%d-%H%M%S)"
  CUSTOM_MODEL_NAME="ofac-nova-custom-$(date +%Y%m%d)"
  #echo "Resolving customizable base model ID…"
  #BASE_MODEL_ID="$(resolve_customizable_model_id)" || die "No customizable base model available; set BASE_MODEL_OVERRIDE to a valid model ID (e.g., amazon.nova-micro-v1:0)"
  echo "  Using base model: $BASE_MODEL_ID"
  preflight_model_id "$BASE_MODEL_ID" || die "Invalid base model ID: $BASE_MODEL_ID"
  BEDROCK_SFT_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SFT_ROLE_NAME}"
  CURRENT_STAGE="model:sft-submit"; SFT_CREATE_JSON=$(AWS_PROFILE="$profile" AWS_REGION="$region" python -m model.tune_nova \
    --job-name "$JOB_NAME" \
    --custom-model-name "$CUSTOM_MODEL_NAME" \
    --role-arn "$BEDROCK_SFT_ROLE_ARN" \
    --base-model-id "$BASE_MODEL_ID" \
    --training-s3-uri "$TRAINING_URI" \
    --output-s3-uri "$OUTPUT_URI" \
    --region "$region" \
    --profile "$profile")
  echo "$SFT_CREATE_JSON"
  JOB_ARN=$(echo "$SFT_CREATE_JSON" | jq -r '.jobArn // empty')
  if [[ -z "$JOB_ARN" ]]; then
    JOB_ARN=$(aws bedrock list-model-customization-jobs --region "$region" --profile "$profile" | jq -r --arg n "$JOB_NAME" '.modelCustomizationJobSummaries[]?|select(.jobName==$n)|.jobArn' | head -n1)
  fi
  if [[ -z "$JOB_ARN" ]]; then
    echo "ERROR: Unable to determine SFT job ARN for $JOB_NAME" >&2
    return 1
  fi

  echo "Waiting for SFT job completion (jobArn=$JOB_ARN)…"
  tries=0
  CURRENT_STAGE="model:sft-wait"; while :; do
    JOB_JSON=$(aws bedrock get-model-customization-job --job-identifier "$JOB_ARN" --region "$region" --profile "$profile") || JOB_JSON='{}'
    JOB_STATUS=$(echo "$JOB_JSON" | jq -r '.status // ""')
    VAL_STATUS=$(echo "$JOB_JSON" | jq -r '.statusDetails.validationDetails.status // "-"')
    TRN_STATUS=$(echo "$JOB_JSON" | jq -r '.statusDetails.trainingDetails.status // "-"')
    CUSTOM_MODEL_ARN=$(echo "$JOB_JSON" | jq -r '.customModelArn // ""')
    printf "[SFT status=%s validation=%s training=%s]\n" "$JOB_STATUS" "$VAL_STATUS" "$TRN_STATUS"
    case "$JOB_STATUS" in
      Completed)
        echo "SFT Completed. Custom model: ${CUSTOM_MODEL_ARN:-$CUSTOM_MODEL_NAME}"
        break;;
      Failed|Stopped|Stopping)
        echo "SFT job ended with status $JOB_STATUS" >&2
        return 1;;
      *)
        ((tries++)) || true
        if (( tries > 360 )); then
          echo "ERROR: SFT did not complete within ~3 hours." >&2
          return 1
        fi
        sleep 30;;
    esac
  done
}

stage_endpoint_only() {
  echo "[stage:endpoint] Create PT for latest custom model…"
  CUSTOM_MODEL_NAME="${CUSTOM_MODEL_NAME:-$(aws bedrock list-custom-models --region "$region" --profile "$profile" | jq -r '.modelSummaries[]|select(.modelName|startswith("ofac-nova-custom-"))|.modelName' | sort | tail -n1)}"
  if [[ -z "$CUSTOM_MODEL_NAME" ]]; then die "No custom model found (prefix ofac-nova-custom-)"; fi
  CUSTOM_MODEL_ARN=$(aws bedrock list-custom-models --region "$region" --profile "$profile" | jq -r --arg n "$CUSTOM_MODEL_NAME" '.modelSummaries[]?|select(.modelName==$n)|.modelArn' | head -n1)
  echo "  Model name: $CUSTOM_MODEL_NAME"
  echo "  Model ARN:  ${CUSTOM_MODEL_ARN:-missing}"
  if [[ -z "$CUSTOM_MODEL_ARN" ]]; then
    echo "ERROR: Could not resolve custom model ARN; cannot create PT." >&2
    die "Missing custom model ARN"
  fi
  read -r -p "Provision 1 MU (no-commitment) for '$CUSTOM_MODEL_NAME'? [y/N] " ans
  ans_lc=$(printf '%s' "$ans" | tr '[:upper:]' '[:lower:]')
  [[ "$ans_lc" == y || "$ans_lc" == yes ]] || { echo "Skipping."; exit 0; }
  PT_NAME="${CUSTOM_MODEL_NAME}-pt"
  echo "  Creating PT: $PT_NAME"
  set +e
  PT_RESP=$(create_pt "$PT_NAME" "$CUSTOM_MODEL_ARN")
  PT_RC=$?
  set -e
  PT_ARN=$(echo "$PT_RESP" | jq -r '.provisionedModelArn // empty' 2>/dev/null || true)
  if [[ -z "$PT_ARN" ]]; then
    echo "  PT ARN not returned; checking list for up to 30s…"
    tries=0; while (( tries < 6 )); do
      FOUND=$(aws bedrock list-provisioned-model-throughputs --no-cli-pager --output json --region "$region" --profile "$profile" \
        | jq -r --arg n "$PT_NAME" '.provisionedModelSummaries[]?|select(.provisionedModelName==$n)|.provisionedModelArn' | head -n1)
      if [[ -n "$FOUND" ]]; then PT_ARN="$FOUND"; break; fi
      sleep 5; ((tries++))
    done
  fi
  if [[ $PT_RC -ne 0 || -z "$PT_ARN" ]]; then echo "ERROR: Failed to create provisioned throughput. AWS says:" >&2; echo "$PT_RESP" >&2; die "PT creation failed"; fi
  echo "Waiting for PT InService… $PT_ARN"
  tries=0; while :; do
    s=$(aws bedrock get-provisioned-model-throughput --provisioned-model-arn "$PT_ARN" --region "$region" --profile "$profile" | jq -r .status)
    echo "[PT=$s]"; [[ "$s" == InService ]] && break; [[ "$s" == Failed ]] && die "PT failed"; ((tries++)); [[ $tries -gt 120 ]] && break; sleep 5;
  done
  # Verify presence in listing by name
  FOUND=$(aws bedrock list-provisioned-model-throughputs --region "$region" --profile "$profile" \
    | jq -r --arg n "$PT_NAME" '.provisionedModelSummaries[]?|select(.provisionedModelName==$n)|.provisionedModelArn' | head -n1)
  if [[ -z "$FOUND" ]]; then echo "WARN: PT not listed yet for name '$PT_NAME' (eventual consistency)." >&2; else echo "Verified PT present: $FOUND"; fi
}

stage_app_only() {
  echo "[stage:app] Build/push ECR (local Docker or AWS CodeBuild) and deploy App Runner…"
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$region" --profile "$profile")
  ECR_URI="$ACCOUNT_ID.dkr.ecr.$region.amazonaws.com/fccassistant-app:latest"
  aws ecr describe-repositories --repository-names fccassistant-app --region "$region" --profile "$profile" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name fccassistant-app --region "$region" --profile "$profile"

  if command -v docker >/dev/null 2>&1 && [[ "${DOCKER_LOCAL:-0}" == "1" ]]; then
    echo "  Using local Docker daemon to build"
    aws ecr get-login-password --region "$region" --profile "$profile" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$region.amazonaws.com"
    docker build -t "$ECR_URI" ./app
    docker push "$ECR_URI"
  else
    echo "  Using AWS CodeBuild to build the container image (no local Docker required)"
    # Ensure deployment bucket exists (should be from earlier steps); use it to store source zip
    DEPLOYMENT_BUCKET="${DEPLOYMENT_BUCKET:-ofac-rag-deployment-${ACCOUNT_ID}-${region}}"
    if ! aws s3api head-bucket --bucket "$DEPLOYMENT_BUCKET" --region "$region" --profile "$profile" >/dev/null 2>&1; then
      if [[ "$region" == "us-east-1" ]]; then
        aws s3api create-bucket --bucket "$DEPLOYMENT_BUCKET" --region "$region" --profile "$profile"
      else
        aws s3api create-bucket --bucket "$DEPLOYMENT_BUCKET" --region "$region" --profile "$profile" --create-bucket-configuration LocationConstraint="$region"
      fi
    fi
    # Zip source
    SRC_ZIP="/tmp/app-source-$(date +%s).zip"
    (cd app && zip -rq "$SRC_ZIP" .)
    S3_KEY="app-source/latest.zip"
    aws s3 cp "$SRC_ZIP" "s3://$DEPLOYMENT_BUCKET/$S3_KEY" --region "$region" --profile "$profile"
    rm -f "$SRC_ZIP"
    # Create CodeBuild service role
    CB_ROLE_NAME="fccassistant-codebuild-role"
    CB_TRUST=$(jq -n '{Version:"2012-10-17",Statement:[{Effect:"Allow",Principal:{Service:"codebuild.amazonaws.com"},Action:"sts:AssumeRole"}]}')
    CB_POL=$(jq -n --arg b "$DEPLOYMENT_BUCKET" '{Version:"2012-10-17",Statement:[
      {Effect:"Allow",Action:["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],Resource:"*"},
      {Effect:"Allow",Action:["ecr:GetAuthorizationToken"],Resource:"*"},
      {Effect:"Allow",Action:["ecr:BatchCheckLayerAvailability","ecr:CompleteLayerUpload","ecr:InitiateLayerUpload","ecr:PutImage","ecr:UploadLayerPart"],Resource:"*"},
      {Effect:"Allow",Action:["s3:GetObject"],Resource:("arn:aws:s3:::"+$b+"/app-source/*")}
    ]}')
    aws iam create-role --role-name "$CB_ROLE_NAME" --assume-role-policy-document "$CB_TRUST" --region "$region" --profile "$profile" >/dev/null 2>&1 || true
    aws iam put-role-policy --role-name "$CB_ROLE_NAME" --policy-name CBAccess --policy-document "$CB_POL" --region "$region" --profile "$profile"
    CB_ROLE_ARN=$(aws iam get-role --role-name "$CB_ROLE_NAME" --region "$region" --profile "$profile" | jq -r .Role.Arn)
    echo "Waiting for CodeBuild role creation ..."
    sleep 20
    # Create or update CodeBuild project
    CB_NAME="fccassistant-app-build"
    CB_ENV=$(jq -n '{type:"LINUX_CONTAINER",image:"aws/codebuild/standard:7.0",computeType:"BUILD_GENERAL1_SMALL",privilegedMode:true}')
    # batch-get-projects returns 0 even when project is missing; inspect result
    EXISTING_NAME=$(aws codebuild batch-get-projects --names "$CB_NAME" --region "$region" --profile "$profile" --query 'projects[0].name' --output text 2>/dev/null || echo "")
    if [[ -z "$EXISTING_NAME" || "$EXISTING_NAME" == "None" ]]; then
      aws codebuild create-project \
        --name "$CB_NAME" \
        --source type=S3,location="$DEPLOYMENT_BUCKET/$S3_KEY" \
        --artifacts type=NO_ARTIFACTS \
        --environment "$(echo $CB_ENV)" \
        --service-role "$CB_ROLE_ARN" \
        --region "$region" --profile "$profile" >/dev/null
    else
      aws codebuild update-project \
        --name "$CB_NAME" \
        --source type=S3,location="$DEPLOYMENT_BUCKET/$S3_KEY" \
        --environment "$(echo $CB_ENV)" \
        --service-role "$CB_ROLE_ARN" \
        --region "$region" --profile "$profile" >/dev/null
    fi
    # Start build with buildspec override: login + build + push
    BUILD_SPEC=$(cat <<'YAML'
version: 0.2
phases:
  pre_build:
    commands:
      - aws --version
      - ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
  build:
    commands:
      - docker build -t $ECR_URI .
  post_build:
    commands:
      - docker push $ECR_URI
YAML
)
    BID=$(aws codebuild start-build --project-name "$CB_NAME" \
      --source-location-override "$DEPLOYMENT_BUCKET/$S3_KEY" \
      --environment-variables-override name=ECR_URI,value="$ECR_URI" \
      --buildspec-override "$BUILD_SPEC" \
      --region "$region" --profile "$profile" | jq -r .build.id)
    echo "  CodeBuild started: $BID"
    # Wait for build to complete
    while :; do
      BJSON=$(aws codebuild batch-get-builds --ids "$BID" --region "$region" --profile "$profile")
      BST=$(echo "$BJSON" | jq -r '.builds[0].buildStatus // ""')
      echo "  [CodeBuild=$BST]"; [[ "$BST" == SUCCEEDED ]] && break; [[ "$BST" == FAILED || "$BST" == FAULT || "$BST" == TIMED_OUT ]] && die "CodeBuild failed"; sleep 5;
    done
  fi
  KB_ID=$(gather_all_resources "$stack_name" | jq -r '.[]|select(.type=="AWS::Bedrock::KnowledgeBase")|.physical' | head -n1)
  CUSTOM_MODEL_NAME="${CUSTOM_MODEL_NAME:-$(aws bedrock list-custom-models --region "$region" --profile "$profile" | jq -r '.modelSummaries[]|select(.modelName|startswith("ofac-nova-custom-"))|.modelName' | sort | tail -n1)}"
  PT_ARN=$(aws bedrock list-provisioned-model-throughputs --region "$region" --profile "$profile" | jq -r --arg n "${CUSTOM_MODEL_NAME}-pt" '.provisionedModelSummaries[]?|select(.provisionedModelName==$n)|.provisionedModelArn' | head -n1)
  #BASE_MODEL_ID="amazon.nova-micro-v1:0"

  # Fallback model ARN if no Provisioned Throughput ARN is available
  FM_ARN="arn:aws:bedrock:${region}::foundation-model/${BASE_MODEL_ID/:128k/}"
  MODEL_ARN="${PT_ARN:-$FM_ARN}"

  echo "Creating App Runner service role..."
  ROLE_NAME="apprunner-bedrock-exec-role"
  TRUST=$(jq -n '{Version:"2012-10-17",Statement:[{Effect:"Allow",Principal:{Service:"tasks.apprunner.amazonaws.com"},Action:"sts:AssumeRole"}]}')
  POL=$(jq -n '{Version:"2012-10-17",Statement:[{Effect:"Allow",Action:["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream","bedrock:Retrieve","bedrock:RetrieveAndGenerate","bedrock:RetrieveAndGenerateStream"],Resource:"*"}]}')
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" --region "$region" --profile "$profile" >/dev/null 2>&1 || true
  aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name BedrockInvoke --policy-document "$POL" --region "$region" --profile "$profile"
  ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --region "$region" --profile "$profile" | jq -r .Role.Arn)

  # ECR access role for App Runner to pull private ECR images
  echo "Creating ECR service role..."
  ECR_TRUST=$(jq -n '{Version:"2012-10-17",Statement:[{Effect:"Allow",Principal:{Service:"build.apprunner.amazonaws.com"},Action:"sts:AssumeRole"}]}')
  aws iam create-role --role-name "$apprunner_ecr_access_role_name" --assume-role-policy-document "$ECR_TRUST" --region "$region" --profile "$profile" >/dev/null 2>&1 || true
  aws iam attach-role-policy --role-name "$apprunner_ecr_access_role_name" --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess --region "$region" --profile "$profile" >/dev/null 2>&1 || true
  ECR_ACCESS_ROLE_ARN=$(aws iam get-role --role-name "$apprunner_ecr_access_role_name" --region "$region" --profile "$profile" | jq -r .Role.Arn)

  echo "Creating App Runner service endpoint..."
  SERVICE_NAME="fccassistant-web"
  # Build SourceConfiguration JSON with runtime env vars
  echo "Creating App Runner service endpoint ..."
  SC_JSON=$(mktemp)
  jq -n \
    --arg img   "$ECR_URI" \
    --arg port  "8080" \
    --arg region "$region" \
    --arg kb     "$KB_ID" \
    --arg pt     "$PT_ARN" \
    --arg modelArn "$MODEL_ARN" \
    --arg base   "$BASE_MODEL_ID" \
    --arg users  "$allowed_user_pw" \
    '{
      ImageRepository: {
        ImageIdentifier: $img,
        ImageRepositoryType: "ECR",
        ImageConfiguration: {
          Port: $port,
          RuntimeEnvironmentVariables: {
            AWS_BEDROCK_MODE: "1",
            AWS_REGION: $region,
            BEDROCK_KB_ID: $kb,
            BEDROCK_MODEL_ARN: $modelArn,
            BEDROCK_BASE_MODEL_ID: $base,
            ALLOW_CORS_ALL: "1",
            ALLOWED_USERS: $users
          }
        }
      },
      AuthenticationConfiguration: {
        AccessRoleArn: "PLACEHOLDER_ECR_ACCESS_ROLE"
      }
    }' > "$SC_JSON"
  # Inject ECR AccessRoleArn string safely
  sed -i '' "s#PLACEHOLDER_ECR_ACCESS_ROLE#$ECR_ACCESS_ROLE_ARN#g" "$SC_JSON"

  echo "... with confing {$SC_JSON}..."
  SERVICE_ARN=$(aws apprunner list-services --region "$region" --profile "$profile" | jq -r ".ServiceSummaryList[]|select(.ServiceName==\"$SERVICE_NAME\")|.ServiceArn")
  if [[ -z "$SERVICE_ARN" ]]; then
    set +e
    CREATE_OUT=$(aws apprunner create-service \
      --service-name "$SERVICE_NAME" \
      --source-configuration file://"$SC_JSON" \
      --instance-configuration InstanceRoleArn="$ROLE_ARN" \
      --health-check-configuration Protocol=HTTP,Path=/api/health \
      --region "$region" --profile "$profile" 2>&1)
    RC=$?
    set -e
    if (( RC != 0 )); then
      echo "ERROR: App Runner create-service failed:" >&2
      echo "$CREATE_OUT" >&2
      die "App Runner create failed"
    fi
    SERVICE_ARN=$(echo "$CREATE_OUT" | jq -r .Service.ServiceArn 2>/dev/null || echo "")
    if [[ -z "$SERVICE_ARN" ]]; then
      SERVICE_ARN=$(aws apprunner list-services --region "$region" --profile "$profile" | jq -r ".ServiceSummaryList[]|select(.ServiceName==\"$SERVICE_NAME\")|.ServiceArn")
    fi
  else
    set +e
    UPDATE_OUT=$(aws apprunner update-service --service-arn "$SERVICE_ARN" --source-configuration file://"$SC_JSON" --region "$region" --profile "$profile" 2>&1)
    RC=$?
    set -e
    if (( RC != 0 )); then
      echo "ERROR: App Runner update-service failed:" >&2
      echo "$UPDATE_OUT" >&2
      die "App Runner update failed"
    fi
  fi
  rm -f "$SC_JSON"
  echo "Waiting for App Runner service to become RUNNING…"
  tries=0
  while :; do
    SRV_JSON=$(aws apprunner describe-service --service-arn "$SERVICE_ARN" --region "$region" --profile "$profile" 2>/dev/null || echo '{}')
    STATUS=$(echo "$SRV_JSON" | jq -r '.Service.Status // ""')
    URL=$(echo "$SRV_JSON" | jq -r '.Service.ServiceUrl // empty')
    echo "  [AppRunner=$STATUS]"
    if [[ "$STATUS" == "RUNNING" ]]; then
      [[ -n "$URL" ]] && echo "App Runner URL: https://$URL" || echo "App Runner is RUNNING (no URL yet)"
      break
    fi
    if [[ "$STATUS" == "CREATE_FAILED" || "$STATUS" == "DELETED" || "$STATUS" == "PAUSED" ]]; then
      echo "ERROR: App Runner service entered failure state: $STATUS" >&2
      exit 1
    fi
    ((tries++)) || true
    if (( tries > 120 )); then
      echo "ERROR: Timed out waiting for App Runner service to become RUNNING" >&2
      exit 1
    fi
    sleep 5
  done
}

stage_status_only() {
  echo "[stage:status] region=$region profile=$profile stack=$stack_name"
  echo "- CloudFormation stack:"
  if aws cloudformation describe-stacks --stack-name "$stack_name" --region "$region" --profile "$profile" >/dev/null 2>&1; then
    aws cloudformation describe-stacks --stack-name "$stack_name" --region "$region" --profile "$profile" \
      | jq -r '.Stacks[0] | "  Name: \(.StackName)\n  Status: \(.StackStatus)\n  Created: \(.CreationTime)"'
  else
    echo "  not found"
  fi

  echo "- Knowledge Base:"
  KB_ID=$(gather_all_resources "$stack_name" | jq -r '.[]|select(.type=="AWS::Bedrock::KnowledgeBase")|.physical' | head -n1)
  if [[ -n "$KB_ID" ]]; then
    if aws bedrock-agent get-knowledge-base --knowledge-base-id "$KB_ID" --region "$region" --profile "$profile" >/dev/null 2>&1; then
      aws bedrock-agent get-knowledge-base --knowledge-base-id "$KB_ID" --region "$region" --profile "$profile" \
        | jq -r --arg id "$KB_ID" '"  ID: \($id)\n  Name: \(.knowledgeBase.name)\n  Status: \(.knowledgeBase.status)"'
    else
      echo "  ID: $KB_ID (details unavailable)"
    fi
  else
    echo "  not found"
  fi

  echo "- Custom model (prefix ofac-nova-custom-):"
  CM_NAME=$(aws bedrock list-custom-models --region "$region" --profile "$profile" 2>/dev/null \
    | jq -r '.modelSummaries[]?|select(.modelName|startswith("ofac-nova-custom-"))|.modelName' | sort | tail -n1)
  if [[ -n "$CM_NAME" ]]; then
    CM_ARN=$(aws bedrock list-custom-models --region "$region" --profile "$profile" | jq -r --arg n "$CM_NAME" '.modelSummaries[]?|select(.modelName==$n)|.modelArn' | head -n1)
    echo "  Name: $CM_NAME"
    echo "  ARN:  ${CM_ARN:-unknown}"
  else
    echo "  none"
  fi

  echo "- Provisioned Throughput:"
  if [[ -n "${CM_NAME:-}" ]]; then
    PT_NAME="${CM_NAME}-pt"
    PT_JSON=$(aws bedrock list-provisioned-model-throughputs --region "$region" --profile "$profile" 2>/dev/null || echo '{}')
    PT_ARN=$(echo "$PT_JSON" | jq -r --arg n "$PT_NAME" '.provisionedModelSummaries[]?|select(.provisionedModelName==$n)|.provisionedModelArn' | head -n1)
    if [[ -n "$PT_ARN" ]]; then
      PT_STATUS=$(aws bedrock get-provisioned-model-throughput --provisioned-model-arn "$PT_ARN" --region "$region" --profile "$profile" 2>/dev/null | jq -r '.status // ""')
      echo "  Name: $PT_NAME"
      echo "  ARN:  $PT_ARN"
      echo "  Status: ${PT_STATUS:-unknown}"
    else
      echo "  none for $PT_NAME"
    fi
  else
    echo "  none"
  fi

  echo "- App Runner service:"
  SERVICE_ARN=$(aws apprunner list-services --region "$region" --profile "$profile" 2>/dev/null | jq -r ".ServiceSummaryList[]|select(.ServiceName==\"$apprunner_service_name\")|.ServiceArn")
  if [[ -n "$SERVICE_ARN" ]]; then
    SRV_JSON=$(aws apprunner describe-service --service-arn "$SERVICE_ARN" --region "$region" --profile "$profile" 2>/dev/null || echo '{}')
    URL=$(echo "$SRV_JSON" | jq -r '.Service.ServiceUrl // empty')
    STATUS=$(echo "$SRV_JSON" | jq -r '.Service.Status // empty')
    echo "  Name: $apprunner_service_name"
    echo "  Status: ${STATUS:-unknown}"
    [[ -n "$URL" ]] && echo "  URL: https://$URL"
  else
    echo "  not found"
  fi

  echo "- Roles:"
  if aws iam get-role --role-name "ofac-bedrock-sft-role" --region "$region" --profile "$profile" >/dev/null 2>&1; then
    echo "  SFT role: ofac-bedrock-sft-role (present)"
  else
    echo "  SFT role: ofac-bedrock-sft-role (missing)"
  fi
  if aws iam get-role --role-name "$apprunner_exec_role_name" --region "$region" --profile "$profile" >/dev/null 2>&1; then
    echo "  AppRunner exec role: $apprunner_exec_role_name (present)"
  else
    echo "  AppRunner exec role: $apprunner_exec_role_name (missing)"
  fi

  echo "- Buckets:"
  # Source PDFs
  if aws s3api head-bucket --bucket "$source_docs_s3_bucket" --region "$region" --profile "$profile" >/dev/null 2>&1; then
    SRC_COUNT=$(aws s3api list-objects-v2 --bucket "$source_docs_s3_bucket" --region "$region" --profile "$profile" --query 'length(Contents[])' --output text 2>/dev/null || echo 0)
    echo "  s3://$source_docs_s3_bucket present (objects: ${SRC_COUNT:-0})"
  else
    echo "  s3://$source_docs_s3_bucket missing"
  fi
  # Corpus chunks
  if aws s3api head-bucket --bucket "$corpus_s3_bucket" --region "$region" --profile "$profile" >/dev/null 2>&1; then
    CHUNK_PREFIX="${corpus_prefix}/chunks/"
    CHK_COUNT=$(aws s3api list-objects-v2 --bucket "$corpus_s3_bucket" --prefix "$CHUNK_PREFIX" --region "$region" --profile "$profile" --query 'length(Contents[])' --output text 2>/dev/null || echo 0)
    echo "  s3://$corpus_s3_bucket present (chunks under $CHUNK_PREFIX: ${CHK_COUNT:-0})"
  else
    echo "  s3://$corpus_s3_bucket missing"
  fi
  # Tuning dataset
  if aws s3api head-bucket --bucket "$tuning_s3_bucket" --region "$region" --profile "$profile" >/dev/null 2>&1; then
    TUNE_KEY="${tuning_prefix}/tuning_dataset_contents.jsonl"
    if aws s3api head-object --bucket "$tuning_s3_bucket" --key "$TUNE_KEY" --region "$region" --profile "$profile" >/dev/null 2>&1; then
      echo "  s3://$tuning_s3_bucket present (training: $TUNE_KEY)"
    else
      echo "  s3://$tuning_s3_bucket present (training: missing)"
    fi
  else
    echo "  s3://$tuning_s3_bucket missing"
  fi
}
if [[ "${1:-all}" != "all" ]]; then
  case "${1}" in
    data) stage_data ; exit 0;;
    embedding) stage_embedding ; exit 0;;
    model) stage_model_only ; exit 0;;
    endpoint) stage_endpoint_only ; exit 0;;
    app) stage_app_only ; exit 0;;
    status) stage_status_only ; exit 0;;
    *) die "Unknown target '${1}'. Use: all|data|embedding|model|endpoint|app|status";;
  esac
fi

main "$@"
