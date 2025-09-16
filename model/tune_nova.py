import argparse
import json
import time
from typing import Optional


def _session(profile: Optional[str] = None, region: Optional[str] = None):
    import boto3
    return boto3.Session(profile_name=profile, region_name=region) if (profile or region) else boto3.Session()


def create_sft_job(
    job_name: str,
    custom_model_name: str,
    role_arn: str,
    base_model_id: str,
    training_s3_uri: str,
    output_s3_uri: str,
    profile: Optional[str] = None,
    region: Optional[str] = None,
):
    """
    Submit a Bedrock supervised fine-tuning job using CreateModelCustomizationJob.
    Expects training_s3_uri to be an S3 URI to a JSONL dataset (Generate Content JSONL schema).
    """
    sess = _session(profile, region)
    bedrock = sess.client("bedrock")

    # Minimal hyperparameters; adjust as supported by the target model
    hyper = {
        # Examples; tune as allowed by base model
        # "epochs": "3",
        # "learning_rate": "1e-5",
        # "batch_size": "4",
    }

    req = {
        "jobName": job_name,
        "customModelName": custom_model_name,
        "roleArn": role_arn,
        "baseModelIdentifier": base_model_id,
        "trainingDataConfig": {
            "s3Uri": training_s3_uri,
        },
        "outputDataConfig": {
            "s3Uri": output_s3_uri,
        },
        "hyperParameters": hyper,
    }

    resp = bedrock.create_model_customization_job(**req)
    return resp


def main(argv=None):
    p = argparse.ArgumentParser(description="Kick off Bedrock SFT for an Amazon Nova base model")
    p.add_argument("--job-name", required=True)
    p.add_argument("--custom-model-name", required=True)
    p.add_argument("--role-arn", required=True, help="IAM role with bedrock:CreateModelCustomizationJob and S3 access")
    p.add_argument("--base-model-id", required=True, help="e.g., amazon.nova-micro-v1 or another supported ID")
    p.add_argument("--training-s3-uri", required=True, help="s3://bucket/prefix/file.jsonl")
    p.add_argument("--output-s3-uri", required=True, help="s3://bucket/prefix/")
    p.add_argument("--region", default=None)
    p.add_argument("--profile", default=None)
    args = p.parse_args(argv)

    r = create_sft_job(
        job_name=args.job_name,
        custom_model_name=args.custom_model_name,
        role_arn=args.role_arn,
        base_model_id=args.base_model_id,
        training_s3_uri=args.training_s3_uri,
        output_s3_uri=args.output_s3_uri,
        profile=args.profile,
        region=args.region,
    )
    print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()

