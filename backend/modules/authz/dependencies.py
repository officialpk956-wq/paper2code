from typing import Optional, Any
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.modules.auth.dependencies import get_current_user
from backend.modules.authz.engine import authorize

def get_route_param(request: Request, name: str) -> Optional[Any]:
    """Helper to extract parameter from path, query or json body."""
    val = request.path_params.get(name)
    if val is not None:
        return val
    val = request.query_params.get(name)
    if val is not None:
        return val
    return None

def check_permission(
    action: str,
    resource_type: Optional[str] = None,
    resource_id_param: Optional[str] = None
):
    """
    FastAPI dependency to enforce action permission, ownership, or sharing level checks on routes.
    """
    def dependency(
        request: Request,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        # Resolve org_id parameter if present
        org_id_val = get_route_param(request, "org_id")
        org_id = int(org_id_val) if org_id_val is not None else None

        # Resolve resource_id parameter if present
        resource_id = None
        if resource_id_param:
            resource_id_val = get_route_param(request, resource_id_param)
            if resource_id_val is not None:
                try:
                    resource_id = int(resource_id_val)
                except ValueError:
                    resource_id = resource_id_val

        # Execute check
        is_allowed = authorize(
            db=db,
            user=user,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            org_id=org_id
        )
        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        return True

    return dependency
