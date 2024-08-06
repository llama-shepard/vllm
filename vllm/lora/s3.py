import os
import time
from datetime import timedelta
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from vllm.logger import init_logger

if TYPE_CHECKING:
    from boto3.resources.factory.s3 import Bucket

logger = init_logger(__name__)
S3_PREFIX = "s3://"

def get_bucket_and_model_id(model_id: str) -> Tuple[str, str]:
    if model_id.startswith(S3_PREFIX):
        model_id_no_protocol = model_id[len(S3_PREFIX) :]
        if "/" not in model_id_no_protocol:
            raise ValueError(
                f"Invalid model_id {model_id}. " f"model_id should be of the form `s3://bucket_name/model_id`"
            )
        bucket_name, model_id = model_id_no_protocol.split("/", 1)
        return bucket_name, model_id

def get_bucket_resource(bucket_name: str) -> "Bucket":
    """Get the s3 client"""
    config = Config(
        retries=dict(
            max_attempts=5,
            mode="standard",
        )
    )

    S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", None)
    R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", None)

    if R2_ACCOUNT_ID:
        s3 = boto3.resource("s3", endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com", config=config)
        return s3.Bucket(bucket_name)
    elif S3_ENDPOINT_URL:
        s3 = boto3.resource("s3", endpoint_url=f"{S3_ENDPOINT_URL}", config=config)
        return s3.Bucket(bucket_name)
    else:
        s3 = boto3.resource("s3", config=config)
        return s3.Bucket(bucket_name)

def get_s3_model_local_dir(model_id: str):
    object_id = model_id.replace("/", "--")
    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / 's3_adapters' / f"{object_id}"
    return repo_cache

def get_s3_model_path(model_id: str):
    s3_model_path = get_s3_model_local_dir(model_id)
    if s3_model_path.exists():
        return s3_model_path
    else:
        return None

def download_files_from_s3(
    bucket: Any,
    filenames: List[str],
    model_id: str,
    revision: str = "",
) -> List[Path]:
    """Download the safetensors files from the s3"""

    def download_file(filename):
        repo_cache = get_s3_model_local_dir(model_id)
        
        logger.info(f"Download file: {filename}")
        start_time = time.time()
        local_file_path = repo_cache / filename
        # ensure cache dir exists and create it if needed
        repo_cache.mkdir(parents=True, exist_ok=True)
        model_id_path = Path(model_id)
        bucket_file_name = model_id_path / filename
        logger.info(f"Downloading file {bucket_file_name} to {local_file_path}")
        bucket.download_file(str(bucket_file_name), str(local_file_path))
        # TODO: add support for revision
        logger.info(f"Downloaded {local_file_path} in {timedelta(seconds=int(time.time() - start_time))}.")
        if not local_file_path.is_file():
            raise FileNotFoundError(f"File {local_file_path} not found")
        return local_file_path

    start_time = time.time()
    files = []
    for i, filename in enumerate(filenames):
        # TODO: clean up name creation logic
        if not filename:
            continue
        file = download_file(filename)

        elapsed = timedelta(seconds=int(time.time() - start_time))
        remaining = len(filenames) - (i + 1)
        eta = (elapsed / (i + 1)) * remaining if remaining > 0 else 0

        logger.info(f"Download: [{i + 1}/{len(filenames)}] -- ETA: {eta}")
        files.append(file)

    return files

def download_model_from_s3(bucket: Any, model_id: str):
    model_path = get_s3_model_local_dir(model_id)
    try:
        model_files = bucket.objects.filter(Prefix=model_id)
        # ensure that only one model is retrieved by filtering on the first dir of the path
        total_models = set([Path(f.key).parts[0] for f in model_files])
        if len(total_models) > 1:
            raise ValueError(f"Multiple models found for model_id {model_id}")

        # need to filter out the empty name
        filenames = [f.key.removeprefix(model_id).lstrip("/") for f in model_files]
        logger.info(filenames)
        download_files_from_s3(bucket, filenames, model_id)
        logger.info(f"Downloaded {len(filenames)} files")

        logger.info(f"Contents of the cache folder: {os.listdir(model_path)}")

        logger.info(f"Downloaded adapter from S3: {model_id}")
        return str(model_path)
    except:
        logger.info(f"Error downloading adapter from S3: {model_id}",exc_info=True)
        shutil.rmtree(model_path, ignore_errors=True)

def download_s3_model(lora_path):
    bucket_name, model_id = get_bucket_and_model_id(lora_path)
    bucket = get_bucket_resource(bucket_name)
    try:
        # Check if the adapter is already downloaded to 's3_adapters'
        s3_model_path = get_s3_model_path(model_id)
        if s3_model_path:
            logger.info(f"Adapter already downloaded: {s3_model_path}")
            return s3_model_path

        # If not found locally, attempt to download from S3
        model_path = download_model_from_s3(bucket, model_id)
        return model_path  # Return the path of the model
    except Exception as e:
        logger.exception(f"Error downloading from S3: {e}")
        return None