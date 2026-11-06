"""HTTP service for FGAO RAG (install ``fgao[server]``)."""
from .config import build_from_config, load_config


def create_app(*args, **kwargs):
    from .app import create_app as _create

    return _create(*args, **kwargs)


__all__ = ["create_app", "build_from_config", "load_config"]
