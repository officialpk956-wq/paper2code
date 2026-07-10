def parse_user_agent(user_agent_str: str) -> tuple[str, str]:
    """
    Simple, robust parser to extract (browser, os) from a user agent string.
    """
    if not user_agent_str:
        return "Unknown", "Unknown"

    ua = user_agent_str.lower()

    # Detect OS
    if "windows" in ua:
        os = "Windows"
    elif "macintosh" in ua or "mac os" in ua:
        os = "macOS"
    elif "android" in ua:
        os = "Android"
    elif "iphone" in ua or "ipad" in ua:
        os = "iOS"
    elif "linux" in ua:
        os = "Linux"
    else:
        os = "Unknown"

    # Detect Browser
    if "edg/" in ua or "edge" in ua:
        browser = "Edge"
    elif "chrome" in ua or "crios" in ua:
        browser = "Chrome"
    elif "firefox" in ua or "fxios" in ua:
        browser = "Firefox"
    elif "safari" in ua and "chrome" not in ua and "chromium" not in ua:
        browser = "Safari"
    elif "msie" in ua or "trident" in ua:
        browser = "Internet Explorer"
    else:
        browser = "Other"

    return browser, os
