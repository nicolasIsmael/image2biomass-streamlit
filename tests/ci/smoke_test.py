"""Smoke test: confirms app.py starts under `streamlit run` without crashing.

Not a pytest test on purpose - it launches a real Streamlit server process, so it
runs as a standalone CI step (see .github/workflows/ci.yml) rather than under pytest.
"""

import subprocess
import sys
import time
import urllib.request

PORT = 8599
STARTUP_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 1


def main() -> int:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            f"--server.port={PORT}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url = f"http://localhost:{PORT}/_stcore/health"
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    started = False
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        started = True
                        break
            except Exception:
                time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        proc.terminate()
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate()

    if not started:
        print("Smoke test FAILED: app did not report healthy in time.")
        print(output)
        return 1

    print("Smoke test PASSED: app started and reported healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
