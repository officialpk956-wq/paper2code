from typing import Any

from sqlalchemy.orm import Session

from backend.models import User
from backend.modules.authz.models import ApiKey, OrganizationMember, ResourceShare, TeamMember
from backend.modules.authz.roles import has_role_permission


def check_scoped_api_key(
    db: Session, key_hash: str, required_scope: str
) -> tuple[bool, User | None]:
    """Verify API key hash, check scopes, track usage, and return user."""
    import datetime

    api_key = db.query(ApiKey).filter_by(key_hash=key_hash, revoked=False).first()
    if not api_key:
        return False, None

    if api_key.expires_at and api_key.expires_at < datetime.datetime.utcnow():
        return False, None

    # Verify scopes
    scopes = api_key.scopes or []
    if required_scope not in scopes and "*" not in scopes:
        return False, None

    # Track usage
    api_key.usage_count += 1
    api_key.last_used_at = datetime.datetime.utcnow()
    db.commit()

    return True, api_key.user


def authorize(
    db: Session,
    user: User,
    action: str,
    resource_type: str | None = None,
    resource_id: Any | None = None,
    org_id: int | None = None,
) -> bool:
    """
    Centralized authorization engine.
    Checks permissions, resource ownership, team/org membership, sharing levels, and Super Admin bypass.
    """
    # 0. Global platform admin bypass (User.is_admin), independent of org context
    if getattr(user, "is_admin", False):
        return True

    # 1. Super Admin bypass
    # First check if the user is a global Super Admin (e.g. org membership or role check)
    if org_id:
        member = (
            db.query(OrganizationMember).filter_by(organization_id=org_id, user_id=user.id).first()
        )
        if member and member.role == "Super Admin":
            return True

    # 2. Check Org-level role permission if org context is provided
    if org_id and not resource_type:
        member = (
            db.query(OrganizationMember).filter_by(organization_id=org_id, user_id=user.id).first()
        )
        if member:
            return has_role_permission(member.role, action)
        return False

    # 3. Check Resource Ownership & Sharing
    if resource_type and resource_id is not None:
        # Check direct ownership first
        # Dynamic lookup on db entity
        from sqlalchemy import text

        # We query the resource using table mapping
        # E.g. to see if resource table has owner_id or user_id
        table_name = (
            f"{resource_type}s" if not resource_type.endswith("y") else f"{resource_type[:-1]}ies"
        )
        if resource_type == "project" or resource_type == "paper":
            table_name = f"{resource_type}s"

        owner_columns = (
            ["uploaded_by", "owner_id", "user_id"]
            if resource_type == "paper" or table_name == "papers"
            else ["owner_id", "user_id", "uploaded_by"]
        )
        for owner_col in owner_columns:
            try:
                sql = text(f"SELECT {owner_col} FROM {table_name} WHERE id = :id")
                res = db.execute(sql, {"id": resource_id}).scalar_one_or_none()
                if res is not None and res == user.id:
                    # User is the owner of the resource
                    return True
            except Exception:
                continue

        # Check explicit resource sharing rules
        # First, query all shares for this resource
        shares = (
            db.query(ResourceShare)
            .filter_by(resource_type=resource_type, resource_id=str(resource_id))
            .all()
        )
        for share in shares:
            # Access levels mapping
            access_levels = {
                "owner": ["read", "comment", "edit", "admin", "owner"],
                "admin": ["read", "comment", "edit", "admin"],
                "edit": ["read", "comment", "edit"],
                "comment": ["read", "comment"],
                "read": ["read"],
            }
            allowed_actions = access_levels.get(share.access_level.lower(), [])

            # Map action prefix to share permissions
            required_access = "read"
            if "delete" in action:
                required_access = "admin"
            elif "write" in action or "edit" in action:
                required_access = "edit"
            elif "comment" in action:
                required_access = "comment"

            if required_access in allowed_actions:
                if share.shared_with_type == "public":
                    return True
                elif share.shared_with_type == "user" and share.shared_with_id == str(user.id):
                    return True
                elif share.shared_with_type == "organization":
                    # Check if user belongs to this organization
                    org_member = (
                        db.query(OrganizationMember)
                        .filter_by(organization_id=int(share.shared_with_id), user_id=user.id)
                        .first()
                    )
                    if org_member:
                        return True
                elif share.shared_with_type == "team":
                    # Check if user belongs to this team
                    team_member = (
                        db.query(TeamMember)
                        .filter_by(team_id=int(share.shared_with_id), user_id=user.id)
                        .first()
                    )
                    if team_member:
                        return True

    return False
