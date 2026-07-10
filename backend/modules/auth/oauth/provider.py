import logging

import httpx

logger = logging.getLogger(__name__)


class OAuthUserInfo:
    def __init__(self, provider: str, uid: str, email: str, name: str, avatar_url: str | None):
        self.provider = provider
        self.uid = uid
        self.email = email
        self.name = name
        self.avatar_url = avatar_url


class OAuthProvider:
    async def get_user_info(self, token: str) -> OAuthUserInfo | None:
        raise NotImplementedError()


class GoogleProvider(OAuthProvider):
    async def get_user_info(self, token: str) -> OAuthUserInfo | None:
        """
        Verify Google id_token using Google TokenInfo API.
        """
        url = f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=5.0)
                if resp.status_code != 200:
                    logger.error(f"Google TokenInfo verification failed: {resp.text}")
                    return None
                data = resp.json()

                # Check email verification
                email_verified = (
                    data.get("email_verified") == "true" or data.get("email_verified") is True
                )
                if not email_verified:
                    logger.error("Google account email is not verified")
                    return None

                return OAuthUserInfo(
                    provider="google",
                    uid=data.get("sub", ""),
                    email=data.get("email", ""),
                    name=data.get("name", "Google User"),
                    avatar_url=data.get("picture"),
                )
        except Exception as e:
            logger.exception(f"Error validating Google OAuth token: {e}", exc_info=True)
            return None


class GitHubProvider(OAuthProvider):
    async def get_user_info(self, token: str) -> OAuthUserInfo | None:
        """
        Verify GitHub access token using GitHub User API.
        """
        url = "https://api.github.com/user"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        try:
            async with httpx.AsyncClient() as client:
                # 1. Fetch main user profile
                resp = await client.get(url, headers=headers, timeout=5.0)
                if resp.status_code != 200:
                    logger.error(f"GitHub user API verification failed: {resp.text}")
                    return None
                user_data = resp.json()

                email = user_data.get("email")

                # 2. If email is private/null, fetch emails list
                if not email:
                    email_resp = await client.get(
                        "https://api.github.com/user/emails", headers=headers, timeout=5.0
                    )
                    if email_resp.status_code == 200:
                        emails = email_resp.json()
                        # Get primary verified email
                        for entry in emails:
                            if entry.get("primary") and entry.get("verified"):
                                email = entry.get("email")
                                break
                        # Fallback to any verified email if primary is not found
                        if not email:
                            for entry in emails:
                                if entry.get("verified"):
                                    email = entry.get("email")
                                    break

                if not email:
                    logger.error("No verified email found for GitHub user")
                    return None

                return OAuthUserInfo(
                    provider="github",
                    uid=str(user_data.get("id", "")),
                    email=email,
                    name=user_data.get("name") or user_data.get("login") or "GitHub User",
                    avatar_url=user_data.get("avatar_url"),
                )
        except Exception as e:
            logger.exception(f"Error validating GitHub OAuth token: {e}", exc_info=True)
            return None
