import re
import bcrypt
import hmac
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Password Policy Configuration
MIN_PASSWORD_LENGTH = 10
REQUIRE_UPPERCASE = True
REQUIRE_LOWERCASE = True
REQUIRE_NUMBER = True
REQUIRE_SPECIAL = True

# Argon2id Configuration (Default parameters from argon2)
argon2_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16
)

def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password according to policy. Returns (is_valid, error_message)."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
    if REQUIRE_UPPERCASE and not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if REQUIRE_LOWERCASE and not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if REQUIRE_NUMBER and not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit."
    if REQUIRE_SPECIAL and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character."
    return True, ""

def hash_password(password: str) -> str:
    """Hash password using Argon2id."""
    return argon2_hasher.hash(password)

def verify_password_and_needs_rehash(plain_password: str, hashed_password: str) -> tuple[bool, bool]:
    """
    Verify password. Handles both Argon2id and legacy bcrypt hashes.
    Returns (is_verified, needs_rehash).
    """
    if not hashed_password:
        return False, False

    # Check if it looks like an Argon2 hash
    if hashed_password.startswith("$argon2"):
        try:
            verified = argon2_hasher.verify(hashed_password, plain_password)
            # Check if parameters match current settings (needs_rehash check)
            needs_rehash = argon2_hasher.check_needs_rehash(hashed_password)
            return True, needs_rehash
        except VerifyMismatchError:
            return False, False
        except Exception:
            return False, False
    
    # Try legacy bcrypt verification
    try:
        # constant-time verification for bcrypt
        is_bcrypt = bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        if is_bcrypt:
            # Bcrypt works, but we should upgrade this to Argon2id immediately!
            return True, True
    except Exception:
        pass

    return False, False

def verify_constant_time(val1: str, val2: str) -> bool:
    """Compare two strings in constant time (e.g. tokens)."""
    return hmac.compare_digest(val1.encode("utf-8"), val2.encode("utf-8"))
