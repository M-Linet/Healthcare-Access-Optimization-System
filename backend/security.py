"""
security.py
-------------
Production-hardening additions for the FastAPI backend. Every function
here is additive and self-contained -- drop this file into backend/ and
wire it in with a few lines in main.py (see MAIN_PY_SECURITY_PATCH.txt).

WHAT THIS ADDS, AND WHY EACH ONE MATTERS:

1. Security response headers (add_security_headers middleware)
   Browsers respect these headers to reduce entire classes of attack:
     - X-Content-Type-Options: nosniff
         Stops browsers "guessing" a file's type and executing something
         unexpected (e.g. treating a JSON response as HTML/JS).
     - X-Frame-Options: DENY
         Stops the app being embedded in an <iframe> on someone else's
         site for clickjacking attacks (invisible overlay tricking users
         into clicking your buttons).
     - Referrer-Policy: strict-origin-when-cross-origin
         Stops leaking full URLs (which could contain query params) to
         third-party sites when a user clicks an outbound link.
     - Permissions-Policy
         Explicitly denies access to camera/microphone/geolocation at the
         browser API level for embedded content, since this app doesn't
         need those from anything except its own first-party JS.
     - Content-Security-Policy
         Restricts which domains scripts/styles/images can load from --
         set to allow only the specific CDNs this app actually uses
         (fonts.googleapis.com, cdnjs.cloudflare.com, OpenStreetMap tiles)
         plus itself. This is the single biggest lever against XSS: even
         if an attacker somehow injected a `<script src="evil.com">` tag,
         the browser would refuse to load it.

2. Locked-down CORS (get_cors_origins)
   The current setup allows origin "*" -- any website on the internet can
   call this API directly from a user's browser. That's fine for a public
   read-only demo, but it means another site could silently piggyback off
   your API (using your rate limit budget, your OSRM routing budget) or
   scrape your model. This reads allowed origins from an environment
   variable (ALLOWED_ORIGINS) with a safe localhost-only default, so the
   student can explicitly whitelist their real deployed domain later
   without needing another code change.

3. Global exception handler (add_safe_exception_handler)
   FastAPI's default behavior can leak internal details (file paths,
   library versions, stack traces) in error responses if DEBUG-style
   errors aren't caught. This catches any *unhandled* exception, logs the
   real error server-side (so you can still debug it), and returns a
   generic, safe message to the client -- the same pattern real production
   APIs use ("Something went wrong" + a reference ID, not a traceback).

4. Request body size guard (RequestSizeLimitMiddleware)
   Without this, nothing stops someone from POSTing a 500MB symptom list
   to /api/predict and tying up server memory. Caps request bodies at a
   sane size (default 100KB -- this API's largest legitimate payload is a
   few hundred bytes of JSON).

Usage: see MAIN_PY_SECURITY_PATCH.txt for the exact lines to add to main.py.
"""

import logging
import os
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("hao.security")


# ---------------------------------------------------------------------------
# 1. Security headers
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(self)"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https://*.tile.openstreetmap.org; "
            "connect-src 'self' https://router.project-osrm.org;"
        )
        # Only send HSTS if we're actually behind HTTPS (a reverse proxy in
        # production, e.g. nginx/Caddy) -- sending it over plain HTTP during
        # local development would be actively wrong.
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response


def add_security_headers(app: FastAPI):
    app.add_middleware(SecurityHeadersMiddleware)


# ---------------------------------------------------------------------------
# 2. CORS origin allowlist (environment-configurable, safe default)
# ---------------------------------------------------------------------------
def get_cors_origins():
    """
    Reads ALLOWED_ORIGINS from the environment as a comma-separated list,
    e.g.  ALLOWED_ORIGINS=https://yourdomain.example,https://www.yourdomain.example

    Falls back to localhost-only if unset, which is the correct default
    for local development and prevents accidentally shipping a wildcard
    "*" origin to production.
    """
    raw = os.environ.get("ALLOWED_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


# ---------------------------------------------------------------------------
# 3. Global exception handler -- never leak internals to the client
# ---------------------------------------------------------------------------
def add_safe_exception_handler(app: FastAPI):
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        error_id = str(uuid.uuid4())[:8]
        # Full detail goes to the server log ONLY -- this is what you look
        # at to actually debug the problem.
        logger.error(f"[{error_id}] Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
        # The client gets a generic, safe message plus a reference ID they
        # can quote if they report the issue -- never a stack trace, file
        # path, or library version.
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Something went wrong on our end. Please try again.",
                "error_id": error_id,
            },
        )


# ---------------------------------------------------------------------------
# 4. Request body size guard
# ---------------------------------------------------------------------------
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = 100_000):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (max {self.max_bytes} bytes)."},
            )
        return await call_next(request)


def add_request_size_limit(app: FastAPI, max_bytes: int = 100_000):
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_bytes)


# ---------------------------------------------------------------------------
# 5. Basic request timing + slow-request logging (observability, not just security)
# ---------------------------------------------------------------------------
class RequestTimingMiddleware(BaseHTTPMiddleware):
    SLOW_REQUEST_THRESHOLD_MS = 2000

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        if duration_ms > self.SLOW_REQUEST_THRESHOLD_MS:
            logger.warning(f"Slow request: {request.method} {request.url.path} took {duration_ms:.0f}ms")
        return response


def add_request_timing(app: FastAPI):
    app.add_middleware(RequestTimingMiddleware)
