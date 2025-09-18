#!/bin/bash

set -euo pipefail

# Dependency check
log() { printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { echo "Error: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found. Please install it."; }

require aws
require jq

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
# Tuning bucket/prefix for SFT assets
tuning_s3_bucket="${tuning_s3_bucket:-ofac-tuning}"
tuning_prefix="${tuning_prefix:-tuning}"

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
    --profile "$profile" >/dev/null
  echo "Template is syntactically valid."

  echo "Creating stack: $stack_name in $region"
  aws cloudformation create-stack \
    --stack-name "$stack_name" \
    --template-url "$template_s3_url" \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
    --parameters "file://$params_file" \
    --region "$region" \
    --profile "$profile"

  echo "Waiting for stack to reach CREATE_COMPLETE…"
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
  echo "ROLE POLICY:$TRUST"
  echo "ACCESS POLICY: $POLICY"

  echo "Waiting role creation..."
  aws iam wait role-exists --role-name "$SFT_ROLE_NAME" --profile "$profile" --region "$region"
  sleep 20

  echo "Submitting Bedrock SFT job (adjust base model and role ARN as needed)"
  TRAINING_URI="s3://${tuning_s3_bucket}/${tuning_prefix}/tuning_dataset_contents.jsonl"
  OUTPUT_URI="s3://${tuning_s3_bucket}/${tuning_prefix}/outputs/"
  JOB_NAME="ofac-sft-$(date +%Y%m%d-%H%M%S)"
  CUSTOM_MODEL_NAME="ofac-nova-custom-$(date +%Y%m%d)"
  BASE_MODEL_ID="amazon.nova-micro-v1:0:128k"
  BEDROCK_SFT_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SFT_ROLE_NAME}"
  SFT_CREATE_JSON=$(AWS_PROFILE="$profile" AWS_REGION="$region" python -m model.tune_nova \
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
  while :; do
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
        PT_RESP=$(aws bedrock create-provisioned-model-throughput \
          --provisioned-model-name "$PT_NAME" \
          --model-arn "$CUSTOM_MODEL_ARN" \
          --desired-model-units 1 \
          --region "$region" --profile "$profile" 2>/dev/null || true)
        PT_ARN=$(echo "$PT_RESP" | jq -r '.provisionedModelArn // ""')
        if [[ -n "$PT_ARN" ]]; then
          echo "Provisioned model ARN: $PT_ARN"
          tries=0
          while :; do
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
          echo "ERROR: Failed to create provisioned throughput." >&2
        fi
      fi
    fi
  fi

  # Run test query against KB using PT (if available) else base model
  TEST_Q='Given the following ownership structure: Co A is 3% owned by SDN A Co A is 47% owned by Co B Co A is 50% owned by Co C Co B is 42% owned by SDN B Co B is 52% owned by Co D Co D is 50% owned by Person C Co D is 50% owned by SDN D Co C is 52% owned by Trust Co C is 48% owned by Co E Co E is 19% owned by Person E Co E is 81% owned by SDN F Trust is managed by Settlor Trust is managed by Trustee Trust is managed by Beneficiary  Further given that Company A Ltd (Co A) is registered in Country X (15% BO threshold), resolve: Task 1: Identify all beneficial owners (FATF Recs 24/25). Task 2: SDN A/B/D/F are on OFAC SDN List. Is Co A subject to US sanctions, and which intermediate/owners contribute? Refer to OFAC FAQ 401 for guidance.'
  ALL_RES=$(gather_all_resources "$stack_name")
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
    cat > "$REQ_FILE" <<JSON
{
  "knowledgeBaseId": "${KB_ID}",
  "modelArn": "${MODEL_FOR_QUERY}",
  "retrievalQuery": { "text": ${TEST_Q@Q} }
}
JSON
    aws bedrock-agent-runtime retrieve-and-generate \
      --region "$region" --profile "$profile" \
      --input file://"$REQ_FILE" \
      | jq -r '.output?.text // .citations // .'
    rm -f "$REQ_FILE"
  fi
}

main "$@"
