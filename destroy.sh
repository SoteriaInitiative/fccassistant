#!/bin/bash

set -euo pipefail

log() { printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { echo "Error: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found. Please install it."; }

require aws
require jq

# Error context surfaced on failure
CURRENT_STAGE="init"
trap 'echo "[ERROR] Stage=$CURRENT_STAGE region=$region profile=$profile — last command failed. See AWS error above." >&2' ERR

stage_info(){ echo "[stage:$1] region=$region profile=$profile ${2:-}"; }

# -------- Parameters (env-overridable) --------
stack_name="${stack_name:-ofac-rag-kb}"
region="${region:-us-east-1}"
profile="${profile:-FCCAssistant}"

# Buckets used by deploy.sh (defaults here must match deploy.sh)
source_docs_s3_bucket="${source_docs_s3_bucket:-ofac-rag-docs}"
corpus_s3_bucket="${corpus_s3_bucket:-ofac-rag-corpus}"
tuning_s3_bucket="${tuning_s3_bucket:-ofac-tuning}"

# Deployment bucket pattern as in deploy.sh
ACCOUNT_ID=$(aws --output=text --region "${region}" --profile "${profile}" sts get-caller-identity --query 'Account')
DEPLOYMENT_BUCKET="${DEPLOYMENT_BUCKET:-ofac-rag-deployment-${ACCOUNT_ID}-${region}}"

# Name patterns for resources created by templates/scripts
KB_NAME="${KB_NAME:-e2e-rag-knowledgebase}"
OSS_COLLECTION_NAME="${OSS_COLLECTION_NAME:-e2e-rag-collection}"
SFT_ROLE_NAME="${SFT_ROLE_NAME:-ofac-bedrock-sft-role}"
# Custom model name prefix used by deployment (controls which PT + models to remove)
CUSTOM_MODEL_PREFIX="${custom_model_prefix:-ofac-nova-custom-}"
APP_RUNNER_SERVICE_NAME="${apprunner_service_name:-fccassistant-web}"
APP_RUNNER_ROLE_NAME="${apprunner_exec_role_name:-apprunner-bedrock-exec-role}"
# App image/ECR + CodeBuild resources created by deploy.sh app
APP_ECR_REPO="${app_image_name:-fccassistant-app}"
CB_PROJECT_NAME="${cb_project_name:-fccassistant-app-build}"
CB_ROLE_NAME="${cb_role_name:-fccassistant-codebuild-role}"
APP_ECR_ACCESS_ROLE_NAME="${apprunner_ecr_access_role_name:-apprunner-ecr-access-role}"

confirm() {
  local msg=${1:-Are you sure?}
  read -r -p "$msg [y/N] " ans || true
  local ans_lc
  ans_lc=$(printf '%s' "$ans" | tr '[:upper:]' '[:lower:]')
  [[ "$ans_lc" == y || "$ans_lc" == yes ]]
}

delete_stack() {
  CURRENT_STAGE="destroy:stack"
  local name="$1"
  if aws cloudformation describe-stacks --stack-name "$name" --region "$region" --profile "$profile" >/dev/null 2>&1; then
    log "Deleting CloudFormation stack: $name"
    aws cloudformation delete-stack --stack-name "$name" --region "$region" --profile "$profile"
    log "Waiting for stack deletion to complete…"
    if ! aws cloudformation wait stack-delete-complete --stack-name "$name" --region "$region" --profile "$profile"; then
      die "CloudFormation stack deletion did not complete successfully for '$name'. Aborting further cleanup."
    fi
  else
    log "Stack not found (skipping): $name"
  fi
}

empty_and_remove_bucket() {
  local b="$1"
  [[ -z "$b" ]] && return 0
  if aws --region "$region" --profile "$profile" s3api head-bucket --bucket "$b" >/dev/null 2>&1; then
    log "Removing bucket and all contents: s3://$b"
    # aws s3 rb --force will delete all objects (and versions) then remove the bucket
    aws --region "$region" --profile "$profile" s3 rb "s3://$b" --force || {
      log "Failed 's3 rb --force' for $b, attempting manual empty + delete"
      aws --region "$region" --profile "$profile" s3 rm "s3://$b" --recursive || true
      aws --region "$region" --profile "$profile" s3api delete-bucket --bucket "$b" || true
    }
    # Wait up to ~150s for bucket to disappear
    local tries=0
    while aws --region "$region" --profile "$profile" s3api head-bucket --bucket "$b" >/dev/null 2>&1; do
      ((tries++)) || true
      if (( tries > 30 )); then
        log "[WARN] Bucket still present after wait: s3://$b"
        break
      fi
      sleep 5
    done
  else
    log "Bucket not found (skipping): s3://$b"
  fi
}

delete_kbs_and_data_sources() {
  CURRENT_STAGE="destroy:kbs"
  log "Scanning for Bedrock Knowledge Bases named '$KB_NAME'…"
  local kbs_json kb_id kb_name
  if ! kbs_json=$(aws bedrock-agent list-knowledge-bases --region "$region" --profile "$profile" 2>/dev/null || true); then
    return 0
  fi
  echo "$kbs_json" | jq -r '.knowledgeBaseSummaries[]?|[.knowledgeBaseId,.name]|@tsv' | while IFS=$'\t' read -r kb_id kb_name; do
    if [[ "$kb_name" == "$KB_NAME" ]]; then
      log "Found KB=$kb_id ($kb_name). Deleting its data sources…"
      local ds_json
      ds_json=$(aws bedrock-agent list-data-sources --knowledge-base-id "$kb_id" --region "$region" --profile "$profile" 2>/dev/null || true)
      echo "$ds_json" | jq -r '.dataSourceSummaries[]?|.dataSourceId' | while read -r ds_id; do
        [[ -z "$ds_id" ]] && continue
        log "Deleting DataSource=$ds_id from KB=$kb_id"
        aws bedrock-agent delete-data-source --knowledge-base-id "$kb_id" --data-source-id "$ds_id" --region "$region" --profile "$profile" || true
      done
      log "Deleting KnowledgeBase=$kb_id"
      aws bedrock-agent delete-knowledge-base --knowledge-base-id "$kb_id" --region "$region" --profile "$profile" || true
      # Wait for KB to be gone
      local tries=0
      while aws bedrock-agent get-knowledge-base --knowledge-base-id "$kb_id" --region "$region" --profile "$profile" >/dev/null 2>&1; do
        ((tries++)) || true
        if (( tries > 30 )); then
          log "[WARN] KB $kb_id still present after wait"
          break
        fi
        sleep 5
      done
    fi
  done
}

delete_custom_models_and_pt() {
  CURRENT_STAGE="destroy:models-pt"
  log "Scanning for Bedrock PT + custom models with prefix '$CUSTOM_MODEL_PREFIX' and deleting them in order…"

  # Build list of custom model ARNs that match our prefix (used to filter PTs)
  local models_json model_arns
  models_json=$(aws bedrock list-custom-models --region "$region" --profile "$profile" 2>/dev/null || echo '{}')
  model_arns=$(echo "$models_json" | jq -r --arg p "$CUSTOM_MODEL_PREFIX" '[.modelSummaries[]?|select(.modelName|startswith($p))|.modelArn]')

  # Delete PTs associated with those models, or whose PT name starts with the prefix (or ends with -pt)
  if aws bedrock list-provisioned-model-throughputs --region "$region" --profile "$profile" >/dev/null 2>&1; then
    aws bedrock list-provisioned-model-throughputs --region "$region" --profile "$profile" \
      | jq -r --arg p "$CUSTOM_MODEL_PREFIX" --argjson marr "$model_arns" '
          .provisionedModelSummaries[]?
          | select((.provisionedModelName|startswith($p))
                   or (.provisionedModelName|endswith("-pt"))
                   or (.modelArn != null and ($marr | index(.modelArn) != null)))
          | .provisionedModelArn' \
      | while read -r pt_arn; do
          [[ -z "$pt_arn" ]] && continue
          log "Deleting Provisioned Throughput: $pt_arn"
          aws bedrock delete-provisioned-model-throughput --provisioned-model-arn "$pt_arn" --region "$region" --profile "$profile" || true
          # Wait until PT no longer resolvable
          local tries=0
          while aws bedrock get-provisioned-model-throughput --provisioned-model-arn "$pt_arn" --region "$region" --profile "$profile" >/dev/null 2>&1; do
            ((tries++)) || true
            if (( tries > 30 )); then
              log "[WARN] Provisioned Throughput still present after wait: $pt_arn"
              break
            fi
            sleep 5
          done
        done
  fi

  # Delete custom models matching our prefix (after PTs are removed)
  echo "$models_json" \
    | jq -r --arg p "$CUSTOM_MODEL_PREFIX" '.modelSummaries[]?|select(.modelName|startswith($p))|.modelArn' \
    | while read -r m_arn; do
        [[ -z "$m_arn" ]] && continue
        log "Deleting Custom Model: $m_arn"
        aws bedrock delete-custom-model --model-identifier "$m_arn" --region "$region" --profile "$profile" || true
        # Wait until model is gone
        local tries=0
        while aws bedrock get-custom-model --model-identifier "$m_arn" --region "$region" --profile "$profile" >/dev/null 2>&1; do
          ((tries++)) || true
          if (( tries > 30 )); then
            log "[WARN] Custom model still present after wait: $m_arn"
            break
          fi
          sleep 5
        done
      done
}

delete_aoss_collections() {
  CURRENT_STAGE="destroy:aoss"
  log "Checking for OpenSearch Serverless collections named '$OSS_COLLECTION_NAME'…"
  if aws opensearchserverless list-collections --region "$region" --profile "$profile" >/dev/null 2>&1; then
    local cols
    cols=$(aws opensearchserverless list-collections --region "$region" --profile "$profile" \
      --collection-filters name="$OSS_COLLECTION_NAME" 2>/dev/null || true)
    echo "$cols" | jq -r '.collectionSummaries[]?|.id + "\t" + .name' | while IFS=$'\t' read -r cid cname; do
      [[ -z "$cid" ]] && continue
      log "Deleting AOSS collection id=$cid name=$cname"
      aws opensearchserverless delete-collection --id "$cid" --region "$region" --profile "$profile" || true
      # Wait until collection gone
      local tries=0
      while aws opensearchserverless get-collection --id "$cid" --region "$region" --profile "$profile" >/dev/null 2>&1; do
        ((tries++)) || true
        if (( tries > 30 )); then
          log "[WARN] AOSS collection still present after wait: $cid"
          break
        fi
        sleep 5
      done
    done
  fi
}

delete_sft_role() {
  CURRENT_STAGE="destroy:role"
  local role="$SFT_ROLE_NAME"
  if aws iam get-role --role-name "$role" --region "$region" --profile "$profile" >/dev/null 2>&1; then
    log "Deleting IAM role: $role (removing inline policies first)"
    for pol in $(aws iam list-role-policies --role-name "$role" --region "$region" --profile "$profile" --query 'PolicyNames[]' --output text || true); do
      aws iam delete-role-policy --role-name "$role" --policy-name "$pol" --region "$region" --profile "$profile" || true
    done
    aws iam delete-role --role-name "$role" --region "$region" --profile "$profile" || true
    # Wait for role disappearance (up to ~150s)
    local tries=0
    while aws iam get-role --role-name "$role" --region "$region" --profile "$profile" >/dev/null 2>&1; do
      ((tries++)) || true
      if (( tries > 30 )); then
        log "[WARN] IAM role still present after wait: $role"
        break
      fi
      sleep 5
    done
  else
    log "SFT role not found (skipping): $role"
  fi
}

verify_cleanup() {
  log "Verifying cleanup status…"
  echo "- S3 buckets (should be absent): $source_docs_s3_bucket, $corpus_s3_bucket, $tuning_s3_bucket, $DEPLOYMENT_BUCKET"
  for b in "$source_docs_s3_bucket" "$corpus_s3_bucket" "$tuning_s3_bucket" "$DEPLOYMENT_BUCKET"; do
    if aws s3api head-bucket --bucket "$b" --region "$region" --profile "$profile" >/dev/null 2>&1; then
      echo "  [!] Bucket still exists: s3://$b"
    else
      echo "  [+] Bucket removed: s3://$b"
    fi
  done

  echo "- Bedrock Knowledge Bases named '$KB_NAME':"
  aws bedrock-agent list-knowledge-bases --region "$region" --profile "$profile" \
    | jq -r --arg n "$KB_NAME" '.knowledgeBaseSummaries[]?|select(.name==$n)|.knowledgeBaseId' || true

  echo "- Bedrock custom models (prefix ofac-nova-custom-):"
  aws bedrock list-custom-models --region "$region" --profile "$profile" \
    | jq -r '.modelSummaries[]?|select(.modelName|startswith("ofac-nova-custom-"))|.modelArn' || true

  echo "- AOSS collections named '$OSS_COLLECTION_NAME':"
  aws opensearchserverless list-collections --collection-filters name="$OSS_COLLECTION_NAME" --region "$region" --profile "$profile" \
    | jq -r '.collectionSummaries[]?|.id+"\t"+.name' || true
}

main() {
  log "Destroy starting for stack=$stack_name, region=$region, profile=$profile"

  # Targeted deletion support
  target="${1:-all}"
  case "$target" in
    all)
      if ! confirm "Proceed to delete CloudFormation stack '$stack_name' and related resources?"; then die "Aborted."; fi
      delete_stack "$stack_name"
      delete_kbs_and_data_sources || true
      delete_aoss_collections || true
      delete_custom_models_and_pt || true
      empty_and_remove_bucket "$DEPLOYMENT_BUCKET" || true
      empty_and_remove_bucket "$source_docs_s3_bucket" || true
      empty_and_remove_bucket "$corpus_s3_bucket" || true
      empty_and_remove_bucket "$tuning_s3_bucket" || true
      delete_sft_role || true
      # App Runner service (if exists)
      if aws apprunner list-services --region "$region" --profile "$profile" >/dev/null 2>&1; then
        SRN=$(aws apprunner list-services --region "$region" --profile "$profile" | jq -r ".ServiceSummaryList[]|select(.ServiceName==\"$APP_RUNNER_SERVICE_NAME\")|.ServiceArn")
        if [[ -n "$SRN" ]]; then
          log "Deleting App Runner service: $APP_RUNNER_SERVICE_NAME"
          aws apprunner delete-service --service-arn "$SRN" --region "$region" --profile "$profile" || true
        fi
      fi
      verify_cleanup
      ;;
    data)
      empty_and_remove_bucket "$source_docs_s3_bucket" || true
      empty_and_remove_bucket "$corpus_s3_bucket" || true
      ;;
    embedding)
      if ! confirm "Delete stack '$stack_name' (KB/OSS infra)?"; then die "Aborted."; fi
      delete_stack "$stack_name"
      delete_kbs_and_data_sources || true
      delete_aoss_collections || true
      ;;
    model)
      delete_custom_models_and_pt || true
      delete_sft_role || true
      ;;
    endpoint)
      # Only PTs
      log "Deleting provisioned throughputs related to prefix '$CUSTOM_MODEL_PREFIX'"
      if aws bedrock list-provisioned-model-throughputs --region "$region" --profile "$profile" >/dev/null 2>&1; then
        aws bedrock list-provisioned-model-throughputs --region "$region" --profile "$profile" \
          | jq -r --arg p "$CUSTOM_MODEL_PREFIX" '.provisionedModelSummaries[]?|select(.provisionedModelName|startswith($p) or (.provisionedModelName|endswith("-pt")))|.provisionedModelArn' \
          | while read -r pt; do
              [[ -z "$pt" ]] && continue
              aws bedrock delete-provisioned-model-throughput --provisioned-model-arn "$pt" --region "$region" --profile "$profile" || true
            done
      fi
      ;;
    app)
      if aws apprunner list-services --region "$region" --profile "$profile" >/dev/null 2>&1; then
        SRN=$(aws apprunner list-services --region "$region" --profile "$profile" | jq -r ".ServiceSummaryList[]|select(.ServiceName==\"$APP_RUNNER_SERVICE_NAME\")|.ServiceArn")
        if [[ -n "$SRN" ]]; then
          log "Deleting App Runner service: $APP_RUNNER_SERVICE_NAME"
          aws apprunner delete-service --service-arn "$SRN" --region "$region" --profile "$profile" || true
        else
          log "App Runner service not found: $APP_RUNNER_SERVICE_NAME"
        fi
      fi
      # Optional: remove App Runner exec role
      if aws iam get-role --role-name "$APP_RUNNER_ROLE_NAME" --region "$region" --profile "$profile" >/dev/null 2>&1; then
        for pol in $(aws iam list-role-policies --role-name "$APP_RUNNER_ROLE_NAME" --region "$region" --profile "$profile" --query 'PolicyNames[]' --output text || true); do
          aws iam delete-role-policy --role-name "$APP_RUNNER_ROLE_NAME" --policy-name "$pol" --region "$region" --profile "$profile" || true
        done
        aws iam delete-role --role-name "$APP_RUNNER_ROLE_NAME" --region "$region" --profile "$profile" || true
      fi
      # Remove ECR access role for App Runner
      if aws iam get-role --role-name "$APP_ECR_ACCESS_ROLE_NAME" --region "$region" --profile "$profile" >/dev/null 2>&1; then
        # Detach managed policy first
        aws iam detach-role-policy --role-name "$APP_ECR_ACCESS_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess --region "$region" --profile "$profile" || true
        for pol in $(aws iam list-role-policies --role-name "$APP_ECR_ACCESS_ROLE_NAME" --region "$region" --profile "$profile" --query 'PolicyNames[]' --output text || true); do
          aws iam delete-role-policy --role-name "$APP_ECR_ACCESS_ROLE_NAME" --policy-name "$pol" --region "$region" --profile "$profile" || true
        done
        aws iam delete-role --role-name "$APP_ECR_ACCESS_ROLE_NAME" --region "$region" --profile "$profile" || true
      fi
      # Delete CodeBuild project and its role
      if aws codebuild batch-get-projects --names "$CB_PROJECT_NAME" --region "$region" --profile "$profile" >/dev/null 2>&1; then
        log "Deleting CodeBuild project: $CB_PROJECT_NAME"
        aws codebuild delete-project --name "$CB_PROJECT_NAME" --region "$region" --profile "$profile" || true
      fi
      if aws iam get-role --role-name "$CB_ROLE_NAME" --region "$region" --profile "$profile" >/dev/null 2>&1; then
        for pol in $(aws iam list-role-policies --role-name "$CB_ROLE_NAME" --region "$region" --profile "$profile" --query 'PolicyNames[]' --output text || true); do
          aws iam delete-role-policy --role-name "$CB_ROLE_NAME" --policy-name "$pol" --region "$region" --profile "$profile" || true
        done
        aws iam delete-role --role-name "$CB_ROLE_NAME" --region "$region" --profile "$profile" || true
      fi
      # Delete ECR repo (remove images first)
      if aws ecr describe-repositories --repository-names "$APP_ECR_REPO" --region "$region" --profile "$profile" >/dev/null 2>&1; then
        log "Deleting ECR repository images and repo: $APP_ECR_REPO"
        IMG_IDS=$(aws ecr list-images --repository-name "$APP_ECR_REPO" --region "$region" --profile "$profile" --query 'imageIds' --output json)
        if [[ "$IMG_IDS" != "[]" && -n "$IMG_IDS" ]]; then
          aws ecr batch-delete-image --repository-name "$APP_ECR_REPO" --image-ids "$IMG_IDS" --region "$region" --profile "$profile" || true
        fi
        aws ecr delete-repository --repository-name "$APP_ECR_REPO" --force --region "$region" --profile "$profile" || true
      fi
      # Remove app source objects in deployment bucket (but not the bucket itself)
      if aws s3api head-bucket --bucket "$DEPLOYMENT_BUCKET" --region "$region" --profile "$profile" >/dev/null 2>&1; then
        log "Cleaning S3 app-source/* from s3://$DEPLOYMENT_BUCKET"
        aws s3 rm "s3://$DEPLOYMENT_BUCKET/app-source/" --recursive --region "$region" --profile "$profile" || true
      fi
      ;;
    *)
      die "Unknown target '$target'. Use: all|data|embedding|model|endpoint|app" ;;
  esac
  log "Destroy stage '$target' completed."
}

main "$@"
