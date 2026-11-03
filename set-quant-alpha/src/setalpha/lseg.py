"""LSEG Workspace (desktop session) access for research pulls.

Optional: needs the `lseg-data` package (`uv sync --group lseg`; included in the Docker images), LSEG Workspace running
and logged in on this machine (or on the Docker host), and LSEG_APP_KEY in set-quant-alpha/.env. The pipeline never
depends on LSEG; scripts/fetch_lseg.py caches everything to parquet.

    from setalpha import lseg
    with lseg.session() as ld:
        df = ld.get_data(["PTT.BK"], ["TR.CompanyMarketCap"])

Inside Docker, set LSEG_PROXY_HOST=host.docker.internal (docker-compose.yml does). Workspace's API proxy listens on the
host's 127.0.0.1:9000 and rejects requests whose Host header is not localhost, so a small relay on the container's
127.0.0.1:9000 forwards the bytes to the host and the library connects as if Workspace ran locally.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
from contextlib import contextmanager

from setalpha import config  # noqa: F401  (loads .env)

PORT = 9000  # Workspace API proxy


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while data := src.recv(65536):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            with contextlib.suppress(OSError):
                s.shutdown(socket.SHUT_RDWR)


def _start_relay(host: str, port: int = PORT) -> None:
    """Forward 127.0.0.1:port to host:port in background threads (no-op if the local port is already taken)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("127.0.0.1", port))
    except OSError:
        srv.close()
        return
    srv.listen(32)

    def serve() -> None:
        while True:
            client, _ = srv.accept()
            try:
                upstream = socket.create_connection((host, port), timeout=10)
                upstream.settimeout(None)
            except OSError:
                client.close()
                continue
            threading.Thread(target=_pipe, args=(client, upstream), daemon=True).start()
            threading.Thread(target=_pipe, args=(upstream, client), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()


def unavailable() -> str | None:
    """Why LSEG cannot be used right now (None = ready): missing package, missing key, or Workspace not reachable."""
    try:
        import lseg.data  # noqa: F401
    except ImportError:
        return "the lseg-data package is not installed (uv sync --group lseg)"
    if not os.getenv("LSEG_APP_KEY"):
        return "LSEG_APP_KEY is not set in set-quant-alpha/.env"
    host = os.getenv("LSEG_PROXY_HOST", "127.0.0.1")
    try:
        socket.create_connection((host, PORT), timeout=5).close()
    except OSError:
        return f"LSEG Workspace is not reachable at {host}:{PORT} (start Workspace and log in)"
    return None


@contextmanager
def session():
    import lseg.data as ld

    key = os.getenv("LSEG_APP_KEY")
    if not key:
        raise RuntimeError("LSEG_APP_KEY is not set in set-quant-alpha/.env")
    host = os.getenv("LSEG_PROXY_HOST")
    if host and host not in ("127.0.0.1", "localhost"):
        _start_relay(host)
    ld.open_session(name="desktop.workspace", app_key=key)
    try:
        yield ld
    finally:
        ld.close_session()
