import hashlib
from typing import Optional
from sqlalchemy.orm import Session
from backend.models import User
from backend.modules.authz.models import OrganizationMember, Organization

class FlagService:
    def __init__(self, db: Session):
        self.db = db

    def is_feature_enabled(self, feature_name: str, user: User, org_id: Optional[int] = None) -> bool:
        """
        Check feature gating. Supports tier checks, beta features, internal check, and rollout percentage.
        """
        # Internal features: only allow if user has .example.com or .paper2code.com email
        if feature_name.startswith("internal."):
            return user.email.endswith("@paper2code.com") or user.email.endswith("@example.com")

        # Rollout percentage based on consistent hashing of user ID
        # E.g. "project.beta-editor:40" means 40% rollout
        if ":" in feature_name:
            name_part, pct_part = feature_name.split(":")
            try:
                pct = int(pct_part)
                # Compute hash modulo 100
                h = hashlib.md5(f"{name_part}:{user.id}".encode("utf-8")).hexdigest()
                score = int(h, 16) % 100
                return score < pct
            except ValueError:
                pass

        # Subscription Tier check
        # E.g. "subscription.pro" requires Pro, Team, or Enterprise
        # E.g. "subscription.enterprise" requires Enterprise
        if feature_name.startswith("subscription."):
            required_tier = feature_name.split(".")[1].lower()
            tier_ranks = {"free": 0, "pro": 1, "team": 2, "enterprise": 3}
            required_rank = tier_ranks.get(required_tier, 0)

            # Get user's active organization tier if org_id is provided
            user_rank = 0
            if org_id:
                org = self.db.get(Organization, org_id)
                if org:
                    user_rank = tier_ranks.get(org.subscription_tier.lower(), 0)
            
            return user_rank >= required_rank

        return True
