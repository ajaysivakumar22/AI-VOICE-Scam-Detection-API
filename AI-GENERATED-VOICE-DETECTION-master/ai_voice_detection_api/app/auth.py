from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from app.config import API_KEY

# Define the API key header scheme for Swagger UI
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def validate_api_key(x_api_key: str = Security(api_key_header)):
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )
    return x_api_key
