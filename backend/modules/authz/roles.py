from typing import Dict, Set

# Permission constants
PROJECT_READ = "project.read"
PROJECT_CREATE = "project.create"
PROJECT_WRITE = "project.write"
PROJECT_DELETE = "project.delete"
PROJECT_SHARE = "project.share"

PAPER_READ = "paper.read"
PAPER_CREATE = "paper.create"
PAPER_WRITE = "paper.write"
PAPER_DELETE = "paper.delete"

TEAM_MANAGE = "team.manage"
ORG_MANAGE = "org.manage"
ORG_DELETE = "org.delete"
OWNERSHIP_TRANSFER = "ownership.transfer"

# Set of all available permissions
ALL_PERMISSIONS = {
    PROJECT_READ, PROJECT_CREATE, PROJECT_WRITE, PROJECT_DELETE, PROJECT_SHARE,
    PAPER_READ, PAPER_CREATE, PAPER_WRITE, PAPER_DELETE,
    TEAM_MANAGE, ORG_MANAGE, ORG_DELETE, OWNERSHIP_TRANSFER
}

# Role definitions mapping to direct permissions
ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    "Guest": {
        PROJECT_READ
    },
    "Viewer": {
        PROJECT_READ, PAPER_READ
    },
    "Member": {
        PROJECT_READ, PAPER_READ,
        PROJECT_CREATE, PROJECT_WRITE,
        PAPER_CREATE, PAPER_WRITE
    },
    "Editor": {
        PROJECT_READ, PAPER_READ,
        PROJECT_CREATE, PROJECT_WRITE,
        PAPER_CREATE, PAPER_WRITE,
        PROJECT_SHARE
    },
    "Maintainer": {
        PROJECT_READ, PAPER_READ,
        PROJECT_CREATE, PROJECT_WRITE,
        PAPER_CREATE, PAPER_WRITE,
        PROJECT_SHARE, PROJECT_DELETE,
        TEAM_MANAGE
    },
    "Admin": {
        PROJECT_READ, PAPER_READ,
        PROJECT_CREATE, PROJECT_WRITE,
        PAPER_CREATE, PAPER_WRITE,
        PROJECT_SHARE, PROJECT_DELETE,
        TEAM_MANAGE, ORG_MANAGE
    },
    "Owner": {
        PROJECT_READ, PAPER_READ,
        PROJECT_CREATE, PROJECT_WRITE,
        PAPER_CREATE, PAPER_WRITE,
        PROJECT_SHARE, PROJECT_DELETE,
        TEAM_MANAGE, ORG_MANAGE,
        ORG_DELETE, OWNERSHIP_TRANSFER
    },
    "Super Admin": ALL_PERMISSIONS
}

# Role hierarchy inheritance list: elements later in list inherit from earlier ones
ROLE_ORDER = ["Guest", "Viewer", "Member", "Editor", "Maintainer", "Admin", "Owner", "Super Admin"]

def get_effective_permissions(role: str) -> Set[str]:
    """Get all permissions for a role including inherited permissions."""
    if role not in ROLE_PERMISSIONS:
        return set()
        
    # Standard roles have simple prefix hierarchy:
    # A role has all permissions of itself and any role below it in hierarchy order.
    try:
        idx = ROLE_ORDER.index(role)
        effective = set()
        for i in range(idx + 1):
            r = ROLE_ORDER[i]
            effective.update(ROLE_PERMISSIONS[r])
        return effective
    except ValueError:
        # Custom role fallback
        return ROLE_PERMISSIONS.get(role, set())

def has_role_permission(role: str, permission: str) -> bool:
    """Check if a role has the specified permission."""
    return permission in get_effective_permissions(role)
