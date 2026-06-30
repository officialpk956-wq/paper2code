from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from typing import Optional, List
from datetime import datetime

class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=10, max_length=128)

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    avatar_url: Optional[str] = None
    points: int
    weekly_points: int = 0
    storage_bytes_used: int = 0
    streak: int
    xp_level: int
    last_active: Optional[datetime] = None
    is_admin: bool = False
    is_verified: bool
    mfa_enabled: bool
    is_email_verified: bool = False
    email_verification_sent: Optional[bool] = None

    model_config = ConfigDict(from_attributes=True)

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse

class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., max_length=2048)

class RefreshResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str = Field(..., max_length=2048)
    new_password: str = Field(..., min_length=10, max_length=128)

class VerifyEmailRequest(BaseModel):
    token: str = Field(..., max_length=2048)

class ResendVerificationRequest(BaseModel):
    email: EmailStr

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., max_length=128)
    new_password: str = Field(..., min_length=10, max_length=128)

class ChangeEmailRequest(BaseModel):
    new_email: EmailStr
    password: str = Field(..., max_length=128)

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    avatar_url: Optional[str] = Field(None, max_length=2048)

    @field_validator("avatar_url")
    @classmethod
    def avatar_url_no_xss(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        lower = v.lower().lstrip()
        for dangerous in ("javascript:", "data:", "vbscript:", "<script"):
            if lower.startswith(dangerous) or dangerous in lower:
                raise ValueError("avatar_url contains a disallowed URI scheme or script tag")
        return v

class SessionResponse(BaseModel):
    id: str
    browser: Optional[str] = None
    os: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: datetime
    last_used_at: datetime
    is_current: bool = False

    model_config = ConfigDict(from_attributes=True)

class EnableMFAResponse(BaseModel):
    secret: str
    qr_code_data_uri: str
    backup_codes: List[str]

class VerifyMFARequest(BaseModel):
    code: str = Field(..., max_length=16)

class DisableMFARequest(BaseModel):
    password: str = Field(..., max_length=128)
