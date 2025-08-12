import json
from pathlib import Path
from urllib.parse import quote_plus

import google.auth
from google.auth.transport.requests import Request
import requests

# ---- Config (read everything from config.py; provide safe defaults) ----
import config as C

PROJECT_ID           = getattr(C, "PROJECT_ID", None)
LOCATION             = getattr(C, "LOCATION", "us-central1")

# Pick a Gemini 2.x model that supports supervised tuning (e.g., "gemini-2.5-flash", "gemini-2.0-flash")
TUNING_BASE_MODEL    = getattr(C, "TUNING_BASE_MODEL", "gemini-2.5-flash")
TUNING_DISPLAY_NAME  = getattr(C, "TUNING_DISPLAY_NAME", "gemini25-sft")

# Epochs (your config already has this; e.g., 3)
EPOCHS               = int(getattr(C, "EPOCHS", 3))

# Where your training set already lives (bucket may be "gs://bucket" or "bucket[/prefix]")
GCS_CORPUS_BUCKET    = getattr(C, "GCS_CORPUS_BUCKET", None)  # e.g., "gs://fcc-assistant-84"
TUNING_PREFIX        = getattr(C, "TUNING_PREFIX", "tuning")
TRAINING_FILENAME    = getattr(C, "OUT_NAME", "tuning_dataset_contents.jsonl")  # produced by your generator

# Local validation file & local data dir from config
DATA_DIR             = Path(getattr(C, "DATA_DIR", "data"))
VALIDATION_FILENAME  = getattr(C, "VALIDATION_FILENAME", "validation_contents.jsonl")

# ------------------------------------------------------------------------

def _require(name, value):
    if not value:
        raise SystemExit(f"❌ Missing required config: {name}")
    return value

def _bucket_and_prefix(gs_or_name: str):
    v = gs_or_name.strip()
    if v.startswith("gs://"):
        b, *rest = v[5:].split("/", 1)
        return b, (rest[0] if rest else "")
    return v, ""

def _join_gcs(bucket: str, *parts: str) -> str:
    norm = "/".join(p.strip("/").strip() for p in parts if p)
    return f"gs://{bucket}/{norm}"

def _console_link(project: str, region: str, job_name: str) -> str:
    # job_name: projects/{project}/locations/{region}/tuningJobs/{ID}
    jid = job_name.split("/")[-1]
    return f"https://console.cloud.google.com/vertex-ai/training/tuning-jobs/{quote_plus(jid)}/details?project={project}&region={region}"

def _adc_token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not creds.valid:
        creds.refresh(Request())
    return creds.token

def _upload_validation_to_gcs(local_path: Path, gcs_bucket_conf: str, prefix: str, dest_name: str) -> str:
    from google.cloud import storage
    bucket_name, base = _bucket_and_prefix(gcs_bucket_conf)
    object_path = "/".join([p for p in [base, prefix, dest_name] if p])
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{object_path}"

def main():
    # Sanity: required config
    _require("PROJECT_ID", PROJECT_ID)
    _require("GCS_CORPUS_BUCKET", GCS_CORPUS_BUCKET)

    # Compute training & validation URIs
    bucket_name, base_prefix = _bucket_and_prefix(GCS_CORPUS_BUCKET)
    training_uri = _join_gcs(bucket_name, base_prefix, TUNING_PREFIX, TRAINING_FILENAME)

    # Validation: read local file and upload to the same prefix
    validation_local = DATA_DIR / VALIDATION_FILENAME
    if not validation_local.exists():
        raise SystemExit(f"❌ Local validation file not found: {validation_local}")

    print(f"Uploading validation set to GCS…")
    validation_uri = _upload_validation_to_gcs(validation_local, GCS_CORPUS_BUCKET, TUNING_PREFIX, VALIDATION_FILENAME)
    print(f"   ✔ {validation_uri}")

    # Build tuning job request
    endpoint = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/tuningJobs"
    body = {
        "baseModel": TUNING_BASE_MODEL,
        "tunedModelDisplayName": TUNING_DISPLAY_NAME,
        "supervisedTuningSpec": {
            "trainingDatasetUri": training_uri,
            "validationDatasetUri": validation_uri,
            "hyperParameters": {
                "epochCount": int(EPOCHS),
                # Optional: "learningRateMultiplier": 1.0,
                # Optional: "adapterSize": "ADAPTER_SIZE_FOUR",
            },
        },
    }

    print("POST", endpoint)
    print("Request:", json.dumps(body, indent=2))

    try:
        token = _adc_token()
    except Exception as e:
        raise SystemExit(f"❌ Could not obtain ADC token. Run:\n"
                         f"   gcloud auth application-default login --account=<your-account>\n"
                         f"   gcloud auth application-default set-quota-project {PROJECT_ID}\n{e}")

    resp = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        data=json.dumps(body),
        timeout=300,
    )

    if resp.status_code >= 300:
        print("❌ Tuning create failed:", resp.status_code, resp.text)
        print("\nTroubleshooting:")
        print(" • Ensure training & validation files are in Generate-Content JSONL schema (top-level `contents`, roles user/model).")
        print(" • Confirm the chosen base model supports supervised tuning and your project can access it.")
        print(" • Check the region/model combo and that Vertex AI API is enabled + billing active.")
        return

    job = resp.json()
    name = job.get("name", "")
    print("\n🚀 Tuning job created:", name)
    print("🔗 Track progress:", _console_link(PROJECT_ID, LOCATION, name))

if __name__ == "__main__":
    main()
