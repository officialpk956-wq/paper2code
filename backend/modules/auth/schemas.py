from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=10)

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

    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse

class RefreshRequest(BaseModel):
    refresh_token: str

class RefreshResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=10)

class VerifyEmailRequest(BaseModel):
    token: str

class ResendVerificationRequest(BaseModel):
    email: EmailStr

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=10)

class ChangeEmailRequest(BaseModel):
    new_email: EmailStr
    password: str

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    avatar_url: Optional[str] = None

class SessionResponse(BaseModel):
    id: str
    browser: Optional[str] = None
    os: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: datetime
    last_used_at: datetime
    is_current: bool = False

    class Config:
        from_attributes = True

class EnableMFAResponse(BaseModel):
    secret: str
    qr_code_data_uri: str
    backup_codes: List[str]

class VerifyMFARequest(BaseModel):
    code: str

class DisableMFARequest(BaseModel):
    password: str
