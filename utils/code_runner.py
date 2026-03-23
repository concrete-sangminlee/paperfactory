import subprocess
import tempfile
import os
import logging
import sys

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
DATA_DIR = os.path.join(OUTPUTS_DIR, "data")


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def run_python_code(code: str, timeout: int = 600) -> dict:
    ensure_dirs()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=OUTPUTS_DIR, delete=False
    ) as f:
        header = f"""
import sys, os
sys.path.insert(0, {repr(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))})
os.chdir({repr(OUTPUTS_DIR)})
FIGURES_DIR = {repr(FIGURES_DIR)}
DATA_DIR = {repr(DATA_DIR)}
"""
        f.write(header + "\n" + code)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=OUTPUTS_DIR,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "script_path": script_path,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Script timed out after {timeout}s",
            "script_path": script_path,
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def list_generated_figures() -> list[str]:
    ensure_dirs()
    extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    figures = []
    for fname in sorted(os.listdir(FIGURES_DIR)):
        if os.path.splitext(fname)[1].lower() in extensions:
            figures.append(os.path.join(FIGURES_DIR, fname))
    return figures


def list_generated_data() -> list[str]:
    ensure_dirs()
    extensions = {".csv", ".json", ".xlsx", ".npy", ".pkl"}
    files = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if os.path.splitext(fname)[1].lower() in extensions:
            files.append(os.path.join(DATA_DIR, fname))
    return files
