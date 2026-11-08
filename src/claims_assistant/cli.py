"""Command line: ``claims-assistant {chat,eval,serve}``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


def _quiet_logs() -> None:
    for name in ("httpx", "httpx2", "mcp", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def _chat(username: str, provider: str) -> None:
    from .config import Settings
    from .system import System

    async with System(Settings(model_provider=provider)) as s:
        session = s.login(username)
        conversation_id = None
        name = s.backends.customers.by_username(username).display_name
        print(f"Signed in as {name}. Type a message; '/upload <path>' attaches a file; '/quit' exits.\n")
        pending_uploads: list[str] = []
        while True:
            try:
                line = (await asyncio.to_thread(input, "you> ")).strip()
            except EOFError:
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line.startswith("/upload "):
                path = Path(line.split(" ", 1)[1]).expanduser()  # noqa: ASYNC240 (no I/O)
                media = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg"}.get(
                    path.suffix, "text/plain"
                )
                content = await asyncio.to_thread(path.read_bytes)
                pending_uploads.append(s.upload(session, path.name, media, content))
                print(f"(attached {path.name})")
                continue
            result = await s.chat.handle_turn(session, conversation_id, line, pending_uploads)
            pending_uploads = []
            conversation_id = result.conversation_id
            print(f"assistant> {result.reply}\n")
            if result.step_up_required:
                session = s.step_up(session)
                print("(development IdP: one-time code sent and verified automatically; send your message again)\n")
        print(json.dumps(s.ledger.report(), indent=2))


def _serve(host: str, port: int) -> None:
    import uvicorn

    from .api.app import create_app

    uvicorn.run(create_app(), host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claims-assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="talk to the assistant in the terminal")
    p_chat.add_argument("--user", default="maya", choices=["maya", "tom", "arjun", "priya"])
    p_chat.add_argument("--provider", default="scripted", choices=["scripted", "anthropic", "bedrock"])

    p_eval = sub.add_parser("eval", help="run the evaluation gate")
    p_eval.add_argument("--provider", default="scripted", choices=["scripted", "anthropic", "bedrock"])
    p_eval.add_argument("--runs", type=int, default=None, help="runs per case (default: per case, usually 3)")
    p_eval.add_argument("--case", action="append", help="case id or scenario (repeatable)")
    p_eval.add_argument(
        "--route", action="append", default=[], help="override a route, e.g. extract_incident=claude-haiku-5-5"
    )
    p_eval.add_argument("--threshold", type=float, default=0.9)
    p_eval.add_argument("--judge", action="store_true", help="grade replies with the judge model (real providers)")
    p_eval.add_argument("--out", default="evals/reports")

    p_serve = sub.add_parser("serve", help="run the chat API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    _quiet_logs()

    if args.cmd == "chat":
        asyncio.run(_chat(args.user, args.provider))
        return 0
    if args.cmd == "serve":
        _serve(args.host, args.port)
        return 0

    from .evals.runner import render_markdown, run_suite

    routes = dict(r.split("=", 1) for r in args.route)
    report = asyncio.run(
        run_suite(
            provider=args.provider,
            runs=args.runs,
            selected=args.case,
            routes=routes,
            threshold=args.threshold,
            judge=args.judge,
        )
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report-{args.provider}.json").write_text(json.dumps(report, indent=2))
    md = render_markdown(report)
    (out / f"report-{args.provider}.md").write_text(md)
    print(md)
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
