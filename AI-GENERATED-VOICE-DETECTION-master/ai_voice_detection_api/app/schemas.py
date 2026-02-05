from pydantic import BaseModel, Field
from typing import Literal


class VoiceRequest(BaseModel):
    """Request body for voice detection using Base64 encoded audio."""
    
    language: str = Field(
        ..., 
        json_schema_extra={"example": "Tamil"},
        description="Language of the audio. Must be one of: Tamil, English, Hindi, Malayalam, Telugu"
    )
    audioFormat: str = Field(
        ..., 
        json_schema_extra={"example": "mp3"},
        description="Audio format. Must be 'mp3'"
    )
    audioBase64: str = Field(
        ...,
        description="Base64-encoded MP3 audio data"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
            }
        }


class VoiceResponse(BaseModel):
    """Response containing voice classification results."""
    
    status: str = Field(
        ...,
        json_schema_extra={"example": "success"},
        description="API response status: 'success' or 'error'"
    )
    language: str = Field(
        ...,
        json_schema_extra={"example": "Tamil"},
        description="Language of the analyzed audio"
    )
    classification: Literal["HUMAN", "AI_GENERATED"] = Field(
        ...,
        json_schema_extra={"example": "AI_GENERATED"},
        description="Classification result: HUMAN or AI_GENERATED"
    )
    confidenceScore: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        json_schema_extra={"example": 0.91},
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        ...,
        json_schema_extra={"example": "Unnatural pitch consistency and robotic speech patterns detected"},
        description="Short reason for the decision"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.91,
                "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
            }
        }
