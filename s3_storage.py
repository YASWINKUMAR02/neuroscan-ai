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


def upload_mri_to_s3(image_or_bytes, filename: str, patient_name: str = "", folder: str = "scans") -> tuple:
    """
    Upload an MRI scan image or NIfTI file to S3.
    Supports PIL Image objects or raw bytes.
    Returns: (s3_key, presigned_url) or (None, None) on failure.
    """
    s3_client, bucket_name = get_s3_client()
    if not s3_client or not bucket_name or image_or_bytes is None:
        return None, None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p_tag = "".join(c for c in (patient_name or "").lower() if c.isalnum() or c == "_")
    prefix = f"{p_tag}_" if p_tag else ""
    clean_fn = "".join(c for c in filename if c.isalnum() or c in ("_", "-", "."))
    s3_key = f"{folder}/{timestamp}_{prefix}{clean_fn}"

    try:
        if isinstance(image_or_bytes, (bytes, bytearray)):
            body_data = bytes(image_or_bytes)
            content_type = "application/gzip" if clean_fn.endswith(".gz") else ("application/octet-stream" if clean_fn.endswith(".nii") else "image/png")
        else:
            buffer = io.BytesIO()
            image_or_bytes.save(buffer, format="PNG")
            body_data = buffer.getvalue()
            content_type = "image/png"

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=body_data,
            ContentType=content_type
        )
        url = generate_presigned_url(s3_key, expires_in=86400)
        return s3_key, url
    except Exception as e:
        print(f"[S3 Scan Upload Error]: {e}")
        return None, None


def download_bytes_from_s3(s3_key: str):
    """
    Download raw object bytes from S3.
    """
    if not s3_key:
        return None
    s3_client, bucket_name = get_s3_client()
    if s3_client and bucket_name:
        try:
            resp = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            return resp["Body"].read()
        except Exception as e:
            print(f"[S3 Download Bytes Error via boto3]: {e}")

    # Fallback via presigned URL
    try:
        import urllib.request
        url = generate_presigned_url(s3_key)
        if url:
            with urllib.request.urlopen(url, timeout=15) as response:
                return response.read()
    except Exception as e:
        print(f"[S3 Download Bytes Error via URL]: {e}")
    return None


def download_mri_from_s3(s3_key: str):
    """
    Download an MRI scan image from S3 and return as a PIL Image.
    """
    raw_data = download_bytes_from_s3(s3_key)
    if raw_data:
        try:
            return Image.open(io.BytesIO(raw_data)).convert("RGB")
        except Exception as e:
            print(f"[PIL Image Load Error from S3 bytes]: {e}")
    return None


def find_patient_scans_in_s3(patient_name: str = "", mrn: str = "") -> list:
    """
    Search S3 scans/ folder for MRI images belonging to a patient.
    Returns a list of dicts with {s3_key, filename, last_modified, size, s3_url}.
    """
    s3_client, bucket_name = get_s3_client()
    if not s3_client or not bucket_name:
        return []

    try:
        resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="scans/")
        if "Contents" not in resp:
            return []

        results = []
        p_clean = patient_name.lower().replace(" ", "").replace("_", "") if patient_name else ""
        mrn_clean = mrn.lower().replace("-", "") if mrn else ""

        for obj in resp["Contents"]:
            k = obj["Key"]
            k_lower = k.lower().replace(" ", "").replace("_", "").replace("-", "")
            match = False
            if p_clean and p_clean in k_lower:
                match = True
            elif mrn_clean and mrn_clean in k_lower:
                match = True
            elif not p_clean and not mrn_clean:
                match = True

            if match:
                url = generate_presigned_url(k)
                results.append({
                    "s3_key": k,
                    "filename": k.split("/")[-1],
                    "last_modified": obj["LastModified"],
                    "size": obj["Size"],
                    "s3_url": url
                })

        # Sort by most recent
        results.sort(key=lambda x: x["last_modified"], reverse=True)
        return results
    except Exception as e:
        print(f"[S3 List Scans Error]: {e}")
        return []


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
