"""
Minimal Vercel Python entrypoint.

This repository's primary UI is Streamlit, which is typically run as:
    streamlit run RAG-layer-of-startup-judging/frontend.py

Vercel expects a Python web entrypoint (app.py/index.py/etc). This WSGI app
provides a valid entrypoint so deployment succeeds and explains how to run the
full Streamlit experience.
"""

from __future__ import annotations


def app(environ, start_response):
    body = (
        "InvestorSave backend is reachable.\n\n"
        "This project uses Streamlit for the full UI:\n"
        "  streamlit run RAG-layer-of-startup-judging/frontend.py\n"
    ).encode("utf-8")
    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]
