"""arXiv Digest & QA Agent package.

On import we hook Python's TLS into the operating system's native trust store
via ``truststore``. This makes HTTPS calls (arXiv, LLM providers, PDF downloads)
trust the same root CAs the OS does — including the custom root certificates
that corporate TLS-inspection proxies install in the system keychain but that
Python's bundled ``certifi`` store does not know about. Without this, requests to
arxiv.org fail with ``CERTIFICATE_VERIFY_FAILED`` on many managed machines.

It is a no-op (silently skipped) when ``truststore`` is unavailable or the
platform has no supported native store, so nothing breaks in plain environments.
"""
from __future__ import annotations

try:  # pragma: no cover - environment dependent
    import truststore

    truststore.inject_into_ssl()
except Exception:
    # truststore not installed or unsupported platform: fall back to certifi.
    pass
