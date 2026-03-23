import subprocess
import json
import os
import logging

logger = logging.getLogger(__name__)


def call_claude(prompt: str, system: str = "", max_turns: int = 5, timeout: int = 300) -> str:
    cmd = ["claude", "--print", "--output-format", "text", "--max-turns", str(max_turns)]
    if system:
        cmd += ["--system-prompt", system]
    cmd += ["-p", prompt]

    env = os.environ.copy()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode != 0:
            logger.error(f"Claude CLI error: {result.stderr}")
            raise RuntimeError(f"Claude CLI failed: {result.stderr}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out after {timeout}s")


def call_claude_with_file(prompt: str, file_path: str, system: str = "", timeout: int = 300) -> str:
    with open(file_path, "r") as f:
        file_content = f.read()
    full_prompt = f"{prompt}\n\n---FILE CONTENT ({file_path})---\n{file_content}"
    return call_claude(full_prompt, system=system, timeout=timeout)


def call_claude_json(prompt: str, system: str = "", timeout: int = 300) -> dict:
    system_with_json = system + "\n\nYou MUST respond with valid JSON only. No markdown, no code blocks, no explanation."
    raw = call_claude(prompt, system=system_with_json, timeout=timeout)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    return json.loads(cleaned)
