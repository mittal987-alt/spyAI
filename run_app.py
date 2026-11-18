#!/usr/bin/env python3
"""
ModelAutopsy AI — Unified Development & Demonstration Launcher.

Boots both the FastAPI diagnostics backend and the React Linear-design frontend
with concurrent subprocess orchestration, dependency verification, and graceful shutdown.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

BANNER = r"""
========================================================================
           __  __           _      _     _         _
          |  \/  |         | |    | |   / \  _   _| |_ ___  _ __  ___ _   _
          | |\/| | ___   __| | ___| |  / _ \| | | | __/ _ \| '_ \/ __| | | |
          | |  | |/ _ \ / _` |/ _ \ | / ___ \ |_| | || (_) | |_) \__ \ |_| |
          |_|  |_|\___/ \__,_|\___/_|/_/   \_\__,_|\__\___/| .__/|___/\__, |
                                                           |_|        |___/
                     Autonomous Model Diagnostics & Telemetry
========================================================================
"""


def check_prerequisites() -> None:
    """Validate python and npm dependencies before booting."""
    print("[1/3] Verifying environment & system dependencies...")

    # Python version check
    if sys.version_info < (3, 10):
        print(f"Error: Python 3.10+ is required. Found: {sys.version.split()[0]}")
        sys.exit(1)

    # Check uvicorn
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError as exc:
        print(f"Missing backend dependency ({exc}). Run: pip install -r requirements.txt")
        sys.exit(1)

    # Check node / npm
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    try:
        subprocess.run([npm_cmd, "--version"], capture_output=True, check=True)
    except Exception:
        print("Warning: npm not found in system PATH. Ensure Node.js is installed.")

    # Check frontend directory
    if not FRONTEND_DIR.exists():
        print(f"Error: Frontend directory not found at {FRONTEND_DIR}")
        sys.exit(1)

    print("      ✓ Python & FastAPI runtime verified.")
    print("      ✓ Frontend directory located.")


def initialize_databases() -> None:
    """Pre-initialize SQLite schemas so tables exist immediately."""
    print("[2/3] Initializing diagnostic ledger databases...")
    try:
        from app.monitoring.alerts import initialize_alert_db
        from app.models.registry import initialize_registry_db
        from app.diagnostics.golden_dataset import initialize_golden_db
        from app.monitoring.alert_rules import initialize_rules_db
        from app.monitoring.llm_tracer import initialize_traces_db
        from app.agents.retraining_trigger import initialize_retraining_db
        from app.diagnostics.vector_store import initialize_vector_store_db

        initialize_alert_db()
        initialize_registry_db()
        initialize_golden_db()
        initialize_rules_db()
        initialize_traces_db()
        initialize_retraining_db()
        initialize_vector_store_db()
        print("      ✓ Alert, Model Registry, Golden Dataset, Rules, Traces, Retraining & Vector DBs active.")
    except Exception as exc:
        print(f"      ! Database pre-initialization note: {exc}")


def run_platform() -> None:
    """Start backend and frontend processes concurrently."""
    check_prerequisites()
    initialize_databases()

    print("[3/3] Launching ModelAutopsy AI platform...")
    print(BANNER)

    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    # Start FastAPI Backend via uvicorn
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]

    # Start Vite Frontend
    frontend_cmd = [npm_cmd, "run", "dev"]

    backend_proc = None
    frontend_proc = None

    try:
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(ROOT_DIR),
        )

        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=str(FRONTEND_DIR),
        )

        time.sleep(1.5)
        print("\n  Status: ALL ENGINES OPERATIONAL")
        print("  ─────────────────────────────────────────────────────────────────")
        print("  ● Linear Web UI:        http://localhost:5173")
        print("  ● Backend REST API:     http://127.0.0.1:8000")
        print("  ● Interactive Swagger:  http://127.0.0.1:8000/docs")
        print("  ● Live SSE Event Stream: http://127.0.0.1:8000/api/v1/stream/events")
        print("  ─────────────────────────────────────────────────────────────────")
        print("  Press Ctrl+C to terminate both servers safely.\n")

        # Wait on either process
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend process exited.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend process exited.")
                break

    except KeyboardInterrupt:
        print("\n\nShutting down ModelAutopsy AI platform...")
    finally:
        for proc, name in [(backend_proc, "Backend"), (frontend_proc, "Frontend")]:
            if proc and proc.poll() is None:
                try:
                    if sys.platform == "win32":
                        proc.terminate()
                    else:
                        proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=3)
                    print(f"✓ {name} stopped.")
                except Exception:
                    proc.kill()
                    print(f"✓ {name} terminated.")
        print("Done. Goodbye!")


if __name__ == "__main__":
    run_platform()
