"""
Backward-compatible entry point that re-exports the server module.
"""

from server.app import app, main

__all__ = ["app", "main"]
