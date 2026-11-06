"""Command line: ``fgao serve``, ``fgao ask``, ``fgao check``.

    fgao serve --config fgao.yaml --host 0.0.0.0 --port 8000 [--workers 1]
    fgao serve --factory myapp.rag:build              # build() returns an FGAORAG
    fgao ask   --config fgao.yaml "What is ...?"        # one query, JSON to stdout
    fgao check --config fgao.yaml                       # build the pipeline and print its layout
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def _load(config: Optional[str], factory: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
    from .config import import_object, load_config

    if factory:
        return import_object(factory)(), {}
    if not config:
        raise SystemExit("give --config FILE or --factory module:function")
    return load_config(config)


def app_from_env():
    """uvicorn factory (used for --workers > 1): reads FGAO_CONFIG / FGAO_FACTORY."""
    from .app import create_app
    from .config import server_kwargs

    rag, settings = _load(os.environ.get("FGAO_CONFIG"), os.environ.get("FGAO_FACTORY"))
    return create_app(rag, **server_kwargs(settings))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fgao", description="FGAO orchestration service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "ask", "check"):
        p = sub.add_parser(name)
        p.add_argument("--config", help="YAML or JSON configuration file")
        p.add_argument("--factory", help="module:function returning an FGAORAG")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--log-level", default="info")
            p.add_argument("--keep-alive", type=int, default=75,
                           help="idle keep-alive seconds; keep it above your load balancer's idle timeout")
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--budget", type=float)
            p.add_argument("--explain", action="store_true", help="print the decision trace instead of JSON")
    a = ap.parse_args(argv)

    if a.cmd == "check":
        rag, settings = _load(a.config, a.factory)
        print(json.dumps({"sources": [{"name": s.name, "k": s.k, "cost": s.cost, "family": s.family,
                                       "rewrite": s.rewrite, "retriever": type(s.retriever).__name__}
                                      for s in rag.sources],
                          "llm": type(rag.llm).__name__ if rag.llm is not None else None,
                          "budget": rag.budget, "on_error": rag.on_error,
                          "operations": [str(x) for x in rag.app.actions], "server": settings}, indent=2))
        return 0
    if a.cmd == "ask":
        rag, _ = _load(a.config, a.factory)
        r = asyncio.run(rag.ainvoke(a.question, budget=a.budget))
        print(r.explain() if a.explain else json.dumps(r.to_dict(), indent=2))
        return 0

    import uvicorn

    logging.basicConfig(level=a.log_level.upper(), format="%(message)s")
    if a.workers > 1:
        if a.config:
            os.environ["FGAO_CONFIG"] = os.path.abspath(a.config)
        if a.factory:
            os.environ["FGAO_FACTORY"] = a.factory
        uvicorn.run("fgao.server.cli:app_from_env", factory=True, host=a.host, port=a.port, workers=a.workers,
                    log_level=a.log_level, timeout_keep_alive=a.keep_alive)
        return 0
    from .app import create_app
    from .config import server_kwargs

    rag, settings = _load(a.config, a.factory)
    uvicorn.run(create_app(rag, **server_kwargs(settings)), host=a.host, port=a.port, log_level=a.log_level,
                timeout_keep_alive=a.keep_alive)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
