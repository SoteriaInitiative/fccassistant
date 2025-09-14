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
  DEPLOYMENT_BUCKET="${DEPLOYMENT_BUCKET:-e2e-rag-deployment-${ACCOUNT_ID}-${region}}"
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
}

main "$@"
