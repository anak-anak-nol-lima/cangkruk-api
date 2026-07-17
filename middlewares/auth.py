import base64
import secrets
import os

from fastapi import Request, HTTPException, Response
from starlette.responses import JSONResponse


async def basic_auth_middleware(request: Request, call_next):
    public_paths = ["/ping", "/docs", "/favicon.ico", "/openapi.json"]

    if request.url.path in public_paths:
        return await call_next(request)


    auth_header = request.headers.get("Authorization")

    SECRET_USERNAME = os.environ.get("SECRET_USERNAME")
    SECRET_PASSWORD = os.environ.get("SECRET_PASSWORD")

    if not auth_header:
        return create_unauthorized_response()

    try:
        scheme, credentials = auth_header.split(" ", 1)
        if scheme.lower() != "basic":
            return create_unauthorized_response()

        decoded_credentials = base64.b64decode(credentials)
        decoded_str = decoded_credentials.decode("utf-8")

        username, password = decoded_str.split(":", 1)

        is_username_valid = secrets.compare_digest(username, SECRET_USERNAME)
        is_password_valid = secrets.compare_digest(password, SECRET_PASSWORD)

        if is_username_valid and is_password_valid:
            return await call_next(request)
        else:
            return create_unauthorized_response()
    except:
        return create_unauthorized_response()

def create_unauthorized_response() -> Response:
    """Helper to return a proper 401 Unauthorized status with browser prompt headers."""
    return JSONResponse(
        status_code=401,
        content={"detail": "Unauthorized"},
        headers={"WWW-Authenticate": 'Basic realm="Restricted Area"'}
    )
