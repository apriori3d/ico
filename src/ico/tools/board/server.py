"""Live Board HTTP server: the board page and its JSON endpoints.

Standard library only: a ``ThreadingHTTPServer`` in a daemon thread, bound
to ``127.0.0.1`` on a free port by default (or a given host and port, e.g.
for a LAN proxy), with a token generated per board in the URL. Endpoints (all need ``?token=``):

- ``GET /``: the page (``board.html``, no external scripts);
- ``GET /api/runs``: ``{"runs": [RunView, ...]}``, newest first;
- ``GET /api/plan?run=<id>``: the ``plan_view`` of a run, or ``null``.
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ico.tools.board.runs import BoardSource

HOST = "127.0.0.1"


class LiveBoard:
    """A board over a :class:`BoardSource`.

    Example:
        >>> board = LiveBoard(runs)
        >>> print(board.start())  # http://127.0.0.1:<port>/?token=...
        >>> board.stop()
    """

    def __init__(self, source: BoardSource, *, host: str = HOST, port: int = 0) -> None:
        """Args:
        source: The runs to show.
        host: The address to bind; ``0.0.0.0`` serves every interface (the
            URL then names this machine). Anyone who can reach it and has
            the URL's token sees the runs.
        port: The port; ``0`` takes a free one at every start.
        """
        self.source = source
        self.host = host
        self.port = port
        self.token = secrets.token_urlsafe(16)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str | None:
        """The page URL while the board is started."""
        if self._server is None:
            return None
        port = int(self._server.server_address[1])
        host = socket.gethostname() if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{port}/?token={self.token}"

    def start(self) -> str:
        """Start serving (once) and return the page URL."""
        if self._server is None:
            self._server = ThreadingHTTPServer((self.host, self.port), _handler(self))
            self._server.daemon_threads = True
            self._thread = threading.Thread(
                target=self._server.serve_forever, name="ico-live-board", daemon=True
            )
            self._thread.start()
        url = self.url
        assert url is not None
        return url

    def stop(self) -> None:
        """Stop serving; the board can be started again (on a new port unless
        one was given)."""
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join()


def _page() -> bytes:
    return (files("ico.tools.board") / "board.html").read_bytes()


def _handler(board: LiveBoard) -> type[BaseHTTPRequestHandler]:
    page = _page()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            token = query.get("token", [""])[0]
            if not secrets.compare_digest(token, board.token):
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            elif url.path == "/":
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self._json({"runs": board.source.runs()})
            elif url.path == "/api/plan":
                self._json(board.source.plan(query.get("run", [""])[0]))
            elif url.path == "/api/profile":
                self._json(board.source.profile(query.get("run", [""])[0]))
            elif url.path == "/api/resources":
                self._json(board.source.resources(query.get("run", [""])[0]))
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def _json(self, value: Any) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # no request log on stderr

    return Handler
