import secrets
import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from backend.models import User
from backend.modules.authz.models import Organization, OrganizationMember, OrganizationInvitation
from backend.modules.authz.repositories.org_repository import OrgRepository
from backend.modules.auth.services.audit_service import AuditService

class OrgService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = OrgRepository(db)
        self.audit = AuditService(db)

    def create_organization(self, name: str, slug: str, owner_id: int, ip_address: Optional[str] = None, device: Optional[str] = None) -> Organization:
        org = self.repo.create_org(name, slug, owner_id)
        # Register the owner as an Admin/Owner in members table
        self.repo.add_member(org.id, owner_id, "Owner")
        self.db.commit()
        
        self.audit.log("org_created", user_id=owner_id, ip_address=ip_address, device=device, metadata_dict={"org_id": org.id, "name": name})
        return org

    def transfer_ownership(self, org_id: int, current_owner_id: int, new_owner_id: int, ip_address: Optional[str] = None, device: Optional[str] = None) -> bool:
        org = self.repo.get_org_by_id(org_id)
        if not org or org.owner_id != current_owner_id:
            return False

        # Verify new owner is a member
        new_owner_member = self.repo.get_member(org_id, new_owner_id)
        if not new_owner_member:
            return False

        # Transfer ownership
        org.owner_id = new_owner_id
        new_owner_member.role = "Owner"
        
        # Demote old owner to Admin
        old_owner_member = self.repo.get_member(org_id, current_owner_id)
        if old_owner_member:
            old_owner_member.role = "Admin"

        self.db.commit()
        
        self.audit.log("ownership_transferred", user_id=current_owner_id, ip_address=ip_address, device=device, metadata_dict={
            "org_id": org_id,
            "old_owner": current_owner_id,
            "new_owner": new_owner_id
        })
        return True

    def invite_member(self, org_id: int, inviting_user_id: int, email: str, role: str, ip_address: Optional[str] = None, device: Optional[str] = None) -> str:
        token = secrets.token_hex(32)
        token_hash = secrets.token_hex(32)  # For db hashing/comparison
        expires = datetime.datetime.utcnow() + datetime.timedelta(days=7)
        
        self.repo.create_invitation(org_id, email, role, token_hash, expires)
        self.db.commit()

        self.audit.log("org_invitation_sent", user_id=inviting_user_id, ip_address=ip_address, device=device, metadata_dict={
            "org_id": org_id,
            "email": email,
            "role": role
        })
        return token  # In real life this token is sent via email; we return it for testing

    def accept_invitation(self, token_hash: str, user: User, ip_address: Optional[str] = None, device: Optional[str] = None) -> bool:
        inv = self.repo.get_invitation_by_token(token_hash)
        if not inv or inv.accepted or inv.expires_at < datetime.datetime.utcnow():
            return False

        # Add user as member
        self.repo.add_member(inv.organization_id, user.id, inv.role)
        inv.accepted = True
        self.db.commit()

        self.audit.log("org_invitation_accepted", user_id=user.id, ip_address=ip_address, device=device, metadata_dict={
            "org_id": inv.organization_id,
            "invitation_id": inv.id
        })
        return True
