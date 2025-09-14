#!/usr/bin/env bash
set -euo pipefail

# Build all deployment artifacts from source under ./lambdas and pinned deps.
# Outputs go to ./artifacts matching the CloudFormation template keys.

# Versions for Lambda Layer
OPENSEARCH_PY_VERSION="2.5.0"
REQ_AWS4AUTH_VERSION="1.2.3"
# opensearch-py 2.5.0 requires urllib3 < 2
URLLIB3_VERSION="1.26.18"
# Pin transitive deps for deterministic builds
REQUESTS_VERSION="2.31.0"
IDNA_VERSION="3.6"
CERTIFI_VERSION="2024.2.2"
CHARSET_NORMALIZER_VERSION="3.3.2"

ARTIFACTS_DIR="artifacts"
BUILD_DIR="build"

log() { printf "[%s] %s\n" "$(date '+%F %T')" "$*"; }
require() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }

require zip
require unzip
require python3
python3 -m pip --version >/dev/null || true

# Determinism and clean packaging on macOS/Linux
export COPYFILE_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${ARTIFACTS_DIR}"

# Always start from fresh ZIPs to avoid CRC/update warnings
rm -f "${ARTIFACTS_DIR}/custom-resource-lambda.zip" \
      "${ARTIFACTS_DIR}/provider-event-handler.zip" \
      "${ARTIFACTS_DIR}/opensearchpy-layer.zip"

# 1) Custom resource Lambda (Python)
log "Building ${ARTIFACTS_DIR}/custom-resource-lambda.zip"
zip -X -9 -q -j "${ARTIFACTS_DIR}/custom-resource-lambda.zip" \
  lambdas/oss_handler.py \
  lambdas/oss_utils.py \
  lambdas/client_utils.py

# 2) Provider event handler (NodeJS)
log "Building ${ARTIFACTS_DIR}/provider-event-handler.zip"
zip -X -9 -q -j "${ARTIFACTS_DIR}/provider-event-handler.zip" \
  lambdas/provider_event_handler/framework.js \
  lambdas/provider_event_handler/consts.js \
  lambdas/provider_event_handler/outbound.js \
  lambdas/provider_event_handler/util.js \
  lambdas/provider_event_handler/cfn-response.js

# 3) OpenSearch Python layer (pure Python deps)
log "Building ${ARTIFACTS_DIR}/opensearchpy-layer.zip"
LAYER_ROOT="${BUILD_DIR}/layer"
mkdir -p "${LAYER_ROOT}/python"

have_docker() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }
if have_docker; then
  log "Using Docker (Lambda python:3.10 image) to build layer"
  docker run --rm -v "$PWD":/var/task public.ecr.aws/lambda/python:3.10 \
    bash -lc "pip install -t '${LAYER_ROOT}/python' --no-compile --no-cache-dir \
      opensearch-py==${OPENSEARCH_PY_VERSION} \
      requests-aws4auth==${REQ_AWS4AUTH_VERSION} \
      urllib3==${URLLIB3_VERSION} \
      requests==${REQUESTS_VERSION} \
      idna==${IDNA_VERSION} \
      certifi==${CERTIFI_VERSION} \
      charset-normalizer==${CHARSET_NORMALIZER_VERSION}"
else
  log "Docker unavailable or not running; using cross-platform wheel fallback"
  WHEELHOUSE="${BUILD_DIR}/wheelhouse"
  mkdir -p "${WHEELHOUSE}"
  python3 -m pip download \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.10 \
    --abi cp310 \
    -d "${WHEELHOUSE}" \
    "opensearch-py==${OPENSEARCH_PY_VERSION}" \
    "requests-aws4auth==${REQ_AWS4AUTH_VERSION}" \
    "urllib3==${URLLIB3_VERSION}" \
    "requests==${REQUESTS_VERSION}" \
    "idna==${IDNA_VERSION}" \
    "certifi==${CERTIFI_VERSION}" \
    "charset-normalizer==${CHARSET_NORMALIZER_VERSION}"
  (cd "${LAYER_ROOT}/python" && for whl in "../../wheelhouse"/*.whl; do unzip -q -o "$whl"; done)
fi

# Prune caches to reduce layer size and improve determinism
find "${LAYER_ROOT}/python" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${LAYER_ROOT}/python" -type f -name "*.pyc" -delete
find "${LAYER_ROOT}/python" -type f -name ".DS_Store" -delete

(cd "${LAYER_ROOT}" && zip -X -9 -q -r "../../${ARTIFACTS_DIR}/opensearchpy-layer.zip" python)

# 4) Verify contents
log "Verifying ZIP contents"
unzip -l "${ARTIFACTS_DIR}/custom-resource-lambda.zip" | sed -n '1,50p' || true
unzip -l "${ARTIFACTS_DIR}/provider-event-handler.zip" | sed -n '1,50p' || true
unzip -l "${ARTIFACTS_DIR}/opensearchpy-layer.zip" | sed -n '1,60p' || true

# 5) Checksums
if command -v shasum >/dev/null; then
  log "Checksums:"
  shasum "${ARTIFACTS_DIR}/custom-resource-lambda.zip" \
         "${ARTIFACTS_DIR}/provider-event-handler.zip" \
         "${ARTIFACTS_DIR}/opensearchpy-layer.zip"
fi

rm -r build

log "Build complete."
