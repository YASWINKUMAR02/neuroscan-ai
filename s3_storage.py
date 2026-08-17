"""
s3_storage.py — AWS S3 Cloud Object Storage Layer for NeuroScan AI
Manages persistent cloud storage for:
1. High-resolution Brain MRI Scans (in `scans/` folder)
2. DICOM-grade Clinical PDF Diagnostic Reports (in `reports/` folder)
3. Secure Presigned Download URLs for Doctor and Patient access
"""

import os
import io
import datetime
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def get_s3_credentials():
    """Retrieve S3 credentials from Streamlit Secrets or Environment Variables."""
    aws_key = None
    aws_secret = None
    aws_region = "ap-south-1"
    s3_bucket = "neuroscan-storage-2026"

    # Try Streamlit Secrets first
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "aws" in st.secrets:
            aws_key = st.secrets["aws"].get("aws_access_key_id")
            aws_secret = st.secrets["aws"].get("aws_secret_access_key")
            aws_region = st.secrets["aws"].get("region_name", "ap-south-1")
            s3_bucket = st.secrets["aws"].get("s3_bucket", "neuroscan-storage-2026")
    except Exception:
        pass

    # Fallback to Environment Variables
    if not aws_key or not aws_secret:
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "ap-south-1")
        s3_bucket = os.getenv("S3_BUCKET_NAME", "neuroscan-storage-2026")

    return aws_key, aws_secret, aws_region, s3_bucket



def get_s3_client():
    """Create and return an authenticated boto3 S3 client."""
    if not BOTO3_AVAILABLE:
        return None, None

    aws_key, aws_secret, aws_region, s3_bucket = get_s3_credentials()
    if not aws_key or not aws_secret:
        return None, None

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
        return s3_client, s3_bucket
    except Exception as e:
        print(f"[S3 Client Error]: {e}")
        return None, None


def is_s3_available():
    """Verify if AWS S3 client and bucket are accessible."""
    if not BOTO3_AVAILABLE:
        return False, "boto3 library not installed (run `pip install boto3`)"

    s3_client, bucket_name = get_s3_client()
    if not s3_client or not bucket_name:
        return False, "AWS S3 credentials missing or invalid"

    try:
        # Check bucket existence with head_bucket
        s3_client.head_bucket(Bucket=bucket_name)
        _, _, region, _ = get_s3_credentials()
        return True, f"Online ({bucket_name} @ {region})"
    except Exception as e:
        return False, str(e)


def upload_pdf_to_s3(pdf_bytes: bytes, filename: str, folder: str = "reports") -> tuple:
    """
    Upload a generated clinical PDF report to S3.
    Returns: (s3_key, presigned_url) or (None, None) on failure.
    """
    s3_client, bucket_name = get_s3_client()
    if not s3_client or not bucket_name or not pdf_bytes:
        return None, None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_fn = "".join(c for c in filename if c.isalnum() or c in ("_", "-", "."))
    s3_key = f"{folder}/{timestamp}_{clean_fn}"

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=pdf_bytes,
            ContentType="application/pdf"
        )
        # Generate presigned download URL valid for 24 hours
        url = generate_presigned_url(s3_key, expires_in=86400)
        return s3_key, url
    except Exception as e:
        print(f"[S3 PDF Upload Error]: {e}")
        return None, None


def upload_mri_to_s3(pil_image, filename: str, folder: str = "scans") -> tuple:
    """
    Upload an MRI scan image to S3.
    Returns: (s3_key, presigned_url) or (None, None) on failure.
    """
    s3_client, bucket_name = get_s3_client()
    if not s3_client or not bucket_name or pil_image is None:
        return None, None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_fn = "".join(c for c in filename if c.isalnum() or c in ("_", "-", "."))
    s3_key = f"{folder}/{timestamp}_{clean_fn}"

    try:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="image/png"
        )
        url = generate_presigned_url(s3_key, expires_in=86400)
        return s3_key, url
    except Exception as e:
        print(f"[S3 Scan Upload Error]: {e}")
        return None, None


def generate_presigned_url(s3_key: str, expires_in: int = 86400) -> str:
    """
    Generate a secure presigned URL for direct S3 download.
    Default expiry: 24 hours (86400 seconds).
    """
    s3_client, bucket_name = get_s3_client()
    if not s3_client or not bucket_name or not s3_key:
        return None

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_key},
            ExpiresIn=expires_in
        )
        return url
    except Exception as e:
        print(f"[S3 Presigned URL Error]: {e}")
        return None
