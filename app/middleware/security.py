import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, UploadFile

from app.config.config import settings
from app.utils.exceptions import InvalidInputException, UnsupportedFormatException

# Path traversal prevention pattern
SAFE_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.\/]+$")


def validate_secure_filename(filename: str) -> str:
    """
    Sanitizes filename and prevents directory traversal attacks.
    """
    # Remove any absolute path indicators or directory traversal sequences
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "", clean_name)
    if not clean_name:
        raise InvalidInputException("filename", filename, "Sanitized filename is empty.")
    return clean_name


def validate_uploaded_file(file: UploadFile) -> None:
    """
    Enforces maximum upload size and allowed file format extensions.
    """
    # 1. Validate Extension
    filename = file.filename or ""
    file_ext = filename.split(".")[-1].lower() if "." in filename else ""
    allowed_list = settings.get_allowed_extensions_list()

    if file_ext not in allowed_list:
        raise UnsupportedFormatException(file_ext, allowed_list)

    # 2. Check File Size (requires checking the file stream size dynamically or content-length headers)
    # We can check size by reading a chunk or reading fully and seeking back
    try:
        file.file.seek(0, 2)  # Seek to end
        size_bytes = file.file.tell()
        file.file.seek(0)  # Seek back to start
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read upload stream size: {e}")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB. Upload size: {size_mb:.2f}MB.",
        )


def cleanup_temp_file(path: str) -> None:
    """
    Safely deletes temporary uploaded file streams.
    """
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            # Non-blocking log
            print(f"[SECURITY] Warning: Could not clean up path {path}: {e}")


# Enterprise JWT Ready Foundation
def create_jwt_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Generates a secure JSON Web Token (JWT) for authentication.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRY_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Validates and decodes a JWT token. Raises HTTPException if expired or invalid.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Authentication token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
