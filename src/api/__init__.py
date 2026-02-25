from __future__ import annotations

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name == "app":
        from .app_ner import app

        return app
    if name == "create_app":
        from .app_ner import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
