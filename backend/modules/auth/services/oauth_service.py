from sqlalchemy.orm import Session

from backend.models import User
from backend.modules.auth.oauth.provider import GitHubProvider, GoogleProvider
from backend.modules.auth.services.audit_service import AuditService
from backend.repositories.user_repository import UserRepository


class OAuthService:
    def __init__(self, db: Session):
        self.db = db
        self.user_repo = UserRepository(db)
        self.audit_service = AuditService(db)
        self.google_provider = GoogleProvider()
        self.github_provider = GitHubProvider()

    async def authenticate_oauth(
        self, provider_name: str, token: str, ip_address: str | None, user_agent: str | None
    ) -> User | None:
        # 1. Resolve provider
        if provider_name == "google":
            info = await self.google_provider.get_user_info(token)
        elif provider_name == "github":
            info = await self.github_provider.get_user_info(token)
        else:
            return None

        if not info or not info.email:
            return None

        # 2. Prevent duplicate accounts by fetching by email
        user = self.user_repo.get_by_email(info.email)

        if user:
            # Account exists. We link it or sync the avatar/name
            # Email is verified via OAuth so we can set is_verified = True
            user.is_verified = True
            if info.avatar_url and not user.avatar_url:
                user.avatar_url = info.avatar_url
            self.db.commit()

            self.audit_service.log(
                action="oauth_login",
                user_id=user.id,
                ip_address=ip_address,
                device=user_agent,
                metadata_dict={"provider": provider_name},
            )
            return user

        # 3. Register new user
        user = self.user_repo.create(email=info.email, name=info.name, avatar_url=info.avatar_url)
        user.is_verified = True
        self.db.commit()

        self.audit_service.log(
            action="oauth_register",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent,
            metadata_dict={"provider": provider_name},
        )
        return user
