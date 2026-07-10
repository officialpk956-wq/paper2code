from sqlalchemy.orm import Session

from backend.modules.authz.models import Team, TeamInvitation, TeamMember


class TeamRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_team(
        self, organization_id: int, name: str, slug: str, parent_id: int | None = None
    ) -> Team:
        team = Team(organization_id=organization_id, name=name, slug=slug, parent_id=parent_id)
        self.db.add(team)
        self.db.flush()
        return team

    def add_member(self, team_id: int, user_id: int, role: str = "Member") -> TeamMember:
        member = TeamMember(team_id=team_id, user_id=user_id, role=role)
        self.db.add(member)
        self.db.flush()
        return member

    def get_team_by_id(self, team_id: int) -> Team | None:
        return self.db.get(Team, team_id)

    def get_member(self, team_id: int, user_id: int) -> TeamMember | None:
        return self.db.query(TeamMember).filter_by(team_id=team_id, user_id=user_id).first()

    def remove_member(self, team_id: int, user_id: int) -> bool:
        member = self.get_member(team_id, user_id)
        if member:
            self.db.delete(member)
            self.db.flush()
            return True
        return False

    def create_invitation(
        self, team_id: int, email: str, role: str, token_hash: str, expires_at
    ) -> TeamInvitation:
        inv = TeamInvitation(
            team_id=team_id, email=email, role=role, token_hash=token_hash, expires_at=expires_at
        )
        self.db.add(inv)
        self.db.flush()
        return inv

    def get_invitation_by_token(self, token_hash: str) -> TeamInvitation | None:
        return self.db.query(TeamInvitation).filter_by(token_hash=token_hash).first()
