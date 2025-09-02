#!/usr/bin/env python3
"""
Minimal HTTP server to serve Cube2 scramble and solution moves.

Endpoints:
  GET /health
    -> {"status":"ok"}

  GET /solve?mode=random&scramble_len=20&model_dir=saved_models/cube2/current&weight=1.0&batch_size=64&nnet_batch_size=
  GET /solve?mode=moves&scramble_moves=U1,R-1,F1&model_dir=...&weight=...&batch_size=...
  GET /solve?mode=state&state_colors=0,1,2,...,23&model_dir=...&weight=...&batch_size=...

Response (application/json):
  {
    "scramble_moves_idx": [..] | null,
    "scramble_moves_human": ["U1", ...] | null,
    "solution_moves_idx": [..],
    "solution_moves_human": ["..."],
    "elapsed": float_seconds,
    "nodes_generated": int
  }

Usage:
  python scripts/cube2_http_server.py --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import json
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Optional, List

from environments.cube2 import Cube2, Cube2State
from scripts.solve_one_cube2 import (
    scramble_random, scramble_with_moves, solve_one,
    parse_moves, parse_state_colors
)


def bool_env(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "y", "on")


class Cube2Handler(BaseHTTPRequestHandler):
    server_version = "Cube2HTTP/0.1"
    env = Cube2()

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                return self._send_json(200, {"status": "ok"})
            if parsed.path != "/solve":
                return self._send_json(404, {"error": "not found"})

            q = parse_qs(parsed.query)
            mode = (q.get("mode", ["random"]) or ["random"])[0]
            model_dir = (q.get("model_dir", ["saved_models/cube2/current"]) or [None])[0]
            weight = float((q.get("weight", ["1.0"]) or ["1.0"])[0])
            batch_size = int((q.get("batch_size", ["64"]) or ["64"])[0])
            nnet_bs_raw = (q.get("nnet_batch_size", [None]) or [None])[0]
            nnet_batch_size = int(nnet_bs_raw) if nnet_bs_raw not in (None, "", "none", "null") else None
            verbose = bool_env((q.get("verbose", ["0"]) or ["0"])[0], False)

            scramble_moves_idx: Optional[List[int]] = None
            if mode == "random":
                scr_len = int((q.get("scramble_len", ["20"]) or ["20"])[0])
                state, scramble_moves_idx = scramble_random(self.env, scr_len)
            elif mode == "moves":
                smoves = (q.get("scramble_moves", [""]) or [""])[0].replace(" ", ",")
                if not smoves:
                    return self._send_json(400, {"error": "scramble_moves required for mode=moves"})
                tokens = smoves.replace(",", " ")
                moves_idx = parse_moves(self.env, tokens)
                state = scramble_with_moves(self.env, moves_idx)
                scramble_moves_idx = moves_idx
            elif mode == "state":
                scolors = (q.get("state_colors", [None]) or [None])[0]
                if not scolors:
                    return self._send_json(400, {"error": "state_colors required for mode=state"})
                colors = parse_state_colors(scolors)
                state = Cube2State(colors)
            else:
                return self._send_json(400, {"error": f"unknown mode '{mode}'"})

            soln, elapsed, nodes_generated, _ = solve_one(
                state, self.env, model_dir, weight, batch_size, nnet_batch_size, verbose
            )

            resp = {
                "scramble_moves_idx": scramble_moves_idx,
                "scramble_moves_human": [self.env.moves[m] for m in scramble_moves_idx] if scramble_moves_idx else None,
                "solution_moves_idx": soln,
                "solution_moves_human": [self.env.moves[m] for m in soln],
                "elapsed": elapsed,
                "nodes_generated": nodes_generated,
            }
            return self._send_json(200, resp)
        except Exception as e:  # broad catch to simplify ops
            return self._send_json(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(description="Cube2 HTTP server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Cube2Handler)
    print(f"[Cube2HTTP] Listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
