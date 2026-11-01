import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Set, Callable
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

# Import from unified database core
from dotenv import load_dotenv
load_dotenv()
from database_core import get_db, User, UserRole

# --- CRYPTOGRAPHIC SECRET KEY MANAGEMENT ---
ENV = os.getenv("APP_ENV", "development").lower()
RAW_SECRET = os.getenv("SECRET_KEY", "").strip()

if not RAW_SECRET or RAW_SECRET == "replace-with-a-secure-random-64-character-hex-key":
    if ENV in ("production", "prod", "staging"):
        raise RuntimeError(
            "[CRITICAL SECURITY] Insecure or missing SECRET_KEY in production environment! "
            "Set a cryptographically secure 64-character hex key in environment variables."
        )
    # In development/test mode, generate a secure ephemeral 256-bit key
    SECRET_KEY = secrets.token_hex(32)
    print(f"[AUTH SECURITY] Generated ephemeral 256-bit development SECRET_KEY.")
else:
    SECRET_KEY = RAW_SECRET

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 480))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# In-Memory Revoked Token Blacklist with TTL cleanup
REVOKED_TOKENS: Set[str] = set()

# --- PYDANTIC SCHEMAS ---
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    user_id: int

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: EmailStr
    role: str
    is_active: bool

    class Config:
        from_attributes = True

# --- CORE PASSWORD & CRYPTO FUNCTIONS ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a bcrypt hash, catching malformed strings safely."""
    if not plain_password or not hashed_password:
        return False
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"[AUTH WARNING] Verification failed due to malformed hash in DB: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generates standard bcrypt hash with salt."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Encodes standard JWT claims with exp, iat, and jti."""
    to_encode = data.copy()
    now = datetime.utcnow()
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({
        "exp": expire,
        "iat": now,
        "jti": secrets.token_hex(16)
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict:
    """Decodes and validates token against revocation blacklist and expiration."""
    if token in REVOKED_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def revoke_token(token: str):
    """Adds token to revocation blacklist."""
    if token:
        REVOKED_TOKENS.add(token)

# --- DEPENDENCY INJECTION: EXTRACT TOKEN (BEARER OR COOKIE) ---
def get_token_from_request(
    request: Request,
    bearer_token: Optional[str] = Depends(oauth2_scheme)
) -> str:
    """Extracts access token from Authorization header or HttpOnly cookie."""
    if bearer_token:
        return bearer_token
    # Fallback to cookie
    cookie_token = request.cookies.get("aura_access_token")
    if cookie_token:
        return cookie_token
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Missing Bearer token or session cookie.",
        headers={"WWW-Authenticate": "Bearer"},
    )

def get_current_user(
    token: str = Depends(get_token_from_request),
    db: Session = Depends(get_db)
) -> User:
    """Derives and validates authenticated User record strictly from server-side JWT claims."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is deactivated")
    return user

# --- ROLE-BASED ACCESS CONTROL (RBAC) DEPENDENCY FACTORY ---
def require_roles(*allowed_roles: UserRole) -> Callable:
    """Enforces that the current authenticated user possesses at least one of the allowed roles."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role in {[r.value for r in allowed_roles]}, but user has {current_user.role.value}"
            )
        return current_user
    return role_checker

# Pre-defined convenience role guards
get_admin_user = require_roles(UserRole.ADMIN)
get_faculty_user = require_roles(UserRole.FACULTY, UserRole.ADMIN)
get_any_authenticated_user = get_current_user