import datetime
import secrets

from sqlalchemy.orm import Session

from backend.models import User
from backend.modules.auth.services.audit_service import AuditService
from backend.modules.authz.models import Team
from backend.modules.authz.repositories.team_repository import TeamRepository


class TeamService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = TeamRepository(db)
        self.audit = AuditService(db)

    def create_team(
        self,
        org_id: int,
        name: str,
        slug: str,
        creator_id: int,
        parent_id: int | None = None,
        ip_address: str | None = None,
        device: str | None = None,
    ) -> Team:
        # If parent_id is specified, verify it belongs to the same org
        if parent_id:
            parent = self.repo.get_team_by_id(parent_id)
            if not parent or parent.organization_id != org_id:
                raise ValueError("Parent team must belong to the same organization")

        team = self.repo.create_team(org_id, name, slug, parent_id)
        # Add creator as a team member (role: Admin)
        self.repo.add_member(team.id, creator_id, "Admin")
        self.db.commit()

        self.audit.log(
            "team_created",
            user_id=creator_id,
            ip_address=ip_address,
            device=device,
            metadata_dict={"team_id": team.id, "organization_id": org_id, "parent_id": parent_id},
        )
        return team

    def invite_team_member(
        self,
        team_id: int,
        inviting_user_id: int,
        email: str,
        role: str = "Member",
        ip_address: str | None = None,
        device: str | None = None,
    ) -> str:
        token = secrets.token_hex(32)
        token_hash = secrets.token_hex(32)
        expires = datetime.datetime.utcnow() + datetime.timedelta(days=7)

        self.repo.create_invitation(team_id, email, role, token_hash, expires)
        self.db.commit()

        self.audit.log(
            "team_invitation_sent",
            user_id=inviting_user_id,
            ip_address=ip_address,
            device=device,
            metadata_dict={"team_id": team_id, "email": email, "role": role},
        )
        return token

    def accept_team_invitation(
        self, token_hash: str, user: User, ip_address: str | None = None, device: str | None = None
    ) -> bool:
        inv = self.repo.get_invitation_by_token(token_hash)
        if not inv or inv.accepted or inv.expires_at < datetime.datetime.utcnow():
            return False

        # Add user as member
        self.repo.add_member(inv.team_id, user.id, inv.role)
        inv.accepted = True
        self.db.commit()

        self.audit.log(
            "team_invitation_accepted",
            user_id=user.id,
            ip_address=ip_address,
            device=device,
            metadata_dict={"team_id": inv.team_id, "invitation_id": inv.id},
        )
        return True
