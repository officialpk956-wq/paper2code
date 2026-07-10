from sqlalchemy.orm import Session

from backend.modules.authz.models import Organization, OrganizationInvitation, OrganizationMember


class OrgRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_org(
        self, name: str, slug: str, owner_id: int, subscription_tier: str = "Free"
    ) -> Organization:
        org = Organization(
            name=name, slug=slug, owner_id=owner_id, subscription_tier=subscription_tier
        )
        self.db.add(org)
        self.db.flush()
        return org

    def add_member(self, organization_id: int, user_id: int, role: str) -> OrganizationMember:
        member = OrganizationMember(organization_id=organization_id, user_id=user_id, role=role)
        self.db.add(member)
        self.db.flush()
        return member

    def get_org_by_id(self, org_id: int) -> Organization | None:
        return self.db.get(Organization, org_id)

    def get_org_by_slug(self, slug: str) -> Organization | None:
        return self.db.query(Organization).filter_by(slug=slug).first()

    def get_member(self, organization_id: int, user_id: int) -> OrganizationMember | None:
        return (
            self.db.query(OrganizationMember)
            .filter_by(organization_id=organization_id, user_id=user_id)
            .first()
        )

    def remove_member(self, organization_id: int, user_id: int) -> bool:
        member = self.get_member(organization_id, user_id)
        if member:
            self.db.delete(member)
            self.db.flush()
            return True
        return False

    def create_invitation(
        self, organization_id: int, email: str, role: str, token_hash: str, expires_at
    ) -> OrganizationInvitation:
        inv = OrganizationInvitation(
            organization_id=organization_id,
            email=email,
            role=role,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self.db.add(inv)
        self.db.flush()
        return inv

    def get_invitation_by_token(self, token_hash: str) -> OrganizationInvitation | None:
        return self.db.query(OrganizationInvitation).filter_by(token_hash=token_hash).first()
