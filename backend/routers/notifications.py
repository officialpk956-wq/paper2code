"""
backend/routers/notifications.py

In-app notification endpoints for authenticated users.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import Notification

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


def _notif_to_dict(n: Notification) -> dict:
    return {
        "id": n.id,
        "type": n.type,
        "title": n.title,
        "body": n.body,
        "is_read": n.is_read,
        "created_at": n.created_at.isoformat() if n.created_at else None,
        "payload": n.payload,
    }


@router.get("")
def list_notifications(
    unread_only: bool = Query(False),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the current user's notifications, newest first."""
    query = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id)
    )
    if unread_only:
        query = query.filter(Notification.is_read == False)

    total = query.count()
    items = query.order_by(Notification.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "unread": db.query(Notification)
            .filter(Notification.user_id == current_user.id, Notification.is_read == False)
            .count(),
        "notifications": [_notif_to_dict(n) for n in items],
    }


@router.post("/{notification_id}/read")
def mark_read(
    notification_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark a single notification as read."""
    notif = db.query(Notification).filter_by(id=notification_id).first()
    if not notif:
        raise HTTPException(404, "Notification not found")
    if notif.user_id != current_user.id:
        raise HTTPException(403, "Not your notification")
    notif.is_read = True
    db.commit()
    return _notif_to_dict(notif)


@router.post("/read-all")
def mark_all_read(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark all of the current user's notifications as read."""
    updated = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id, Notification.is_read == False)
        .update({"is_read": True})
    )
    db.commit()
    return {"marked_read": updated}
