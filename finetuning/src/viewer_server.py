from __future__ import annotations

import argparse
import socket
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "viewer"


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        pass


def free_port(preferred: int, tries: int = 20) -> int:
    for port in range(preferred, preferred + tries):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit(f"Ports {preferred}-{preferred + tries - 1} are all busy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the 3D pocket viewer on localhost.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not (VIEWER / "data" / "manifest.json").is_file():
        raise SystemExit(
            f"No viewer data under {VIEWER / 'data'}.\n"
            "Rebuild it with scripts/dump_viewer_poses.py, then "
            "scripts/score_viewer_poses.py and scripts/fetch_full_proteins.py."
        )

    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(Handler, directory=str(VIEWER))
    )
    print(f"Viewer -> {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
