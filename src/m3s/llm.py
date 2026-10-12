"""Chat via the opencode CLI plus an OpenAI-compatible embedding client."""
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

CHAT_PREFIXES = ("VLM_", "RETRIEVAL_LLM_")
EMBED_PREFIXES = ("EMBED_",)

# Chat goes through the opencode CLI by default; the HTTP path stays available
# for hosts without opencode installed (M3S_CHAT_BACKEND=http).
CHAT_BACKEND = (os.environ.get("M3S_CHAT_BACKEND") or "opencode").strip().lower()
OPENCODE_BIN = os.environ.get("M3S_OPENCODE_BIN") or "opencode"
OPENCODE_MODEL = os.environ.get("M3S_OPENCODE_MODEL") or "volcengine-plan/ark-code-latest"
OPENCODE_AGENT = os.environ.get("M3S_OPENCODE_AGENT") or ""


def setting(name, prefixes, default=None):
    """Return the first non-empty environment value across the given prefixes."""
    for prefix in prefixes:
        value = os.environ.get(prefix + name)
        if value and value.strip():
            return value.strip()
    return default


def _post(url, payload, api_key, timeout):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"The model endpoint returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Cannot connect to the model endpoint: {exc}") from exc


def strip_thinking(text):
    """Drop reasoning blocks so only the user-facing answer remains."""
    cleaned = re.sub(r"<(thinking|think)>.*?</\1>", "", text or "", flags=re.DOTALL | re.I)
    return cleaned.strip()


def _opencode_workdir():
    """Return a scratch directory so project AGENTS.md never enters the prompt."""
    path = os.environ.get("M3S_OPENCODE_WORKDIR")
    if not path:
        path = os.path.join(tempfile.gettempdir(), "m3s_opencode_run")
    os.makedirs(path, exist_ok=True)
    return path


def _opencode_text(stream):
    """Collect assistant text parts from the `--format json` event stream."""
    chunks = []
    for line in stream.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part") or {}
        if event.get("type") == "text" and part.get("type") == "text":
            chunks.append(part.get("text") or "")
    return chunks[-1] if chunks else ""


def chat_opencode(prompt, timeout=300):
    """Run one prompt through the opencode CLI and return the assistant text."""
    command = [OPENCODE_BIN, "run", "--pure", "--format", "json",
               "--dir", _opencode_workdir(), "-m", OPENCODE_MODEL]
    if OPENCODE_AGENT:
        command += ["--agent", OPENCODE_AGENT]
    command += ["--", prompt]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Cannot find opencode ({OPENCODE_BIN}); install it or set M3S_OPENCODE_BIN.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"opencode timed out after {timeout}s.") from exc
    text = strip_thinking(_opencode_text(done.stdout))
    if not text:
        detail = (done.stderr or done.stdout or "").strip()[:500]
        raise RuntimeError(f"opencode returned no text (exit code {done.returncode}): {detail}")
    return text


def chat_http(prompt, temperature=0.7, timeout=300):
    """Send one user prompt to the OpenAI-compatible endpoint."""
    model, url = setting("MODEL", CHAT_PREFIXES), setting("URL", CHAT_PREFIXES)
    if not model or not url:
        raise RuntimeError("Missing chat model settings: set VLM_MODEL/VLM_URL or RETRIEVAL_LLM_MODEL/RETRIEVAL_LLM_URL in .env.")
    payload = {"model": model, "temperature": temperature, "stream": False,
               "messages": [{"role": "user", "content": prompt}]}
    thinking = os.getenv("M3S_CHAT_ENABLE_THINKING")
    if thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking.lower() == "true"}
    max_tokens = os.getenv("M3S_CHAT_MAX_TOKENS")
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    endpoint = url.rstrip("/") + "/chat/completions"
    key = setting("API_KEY", CHAT_PREFIXES, "no-key")
    data = _post(endpoint, payload, key, timeout)
    return strip_thinking(data["choices"][0]["message"]["content"])


def chat(prompt, temperature=0.7, timeout=300, retries=3):
    """Send one user prompt and return the assistant text."""
    last = None
    for _ in range(max(1, retries)):
        try:
            if CHAT_BACKEND == "opencode":
                return chat_opencode(prompt, timeout=timeout)
            return chat_http(prompt, temperature=temperature, timeout=timeout)
        except (RuntimeError, KeyError, IndexError, TypeError) as exc:
            last = exc
    raise RuntimeError(f"Chat model call failed: {last}")


def chat_json(prompt, timeout=300, retries=3):
    """Return the parsed JSON object from a model reply, or {} when unusable."""
    for _ in range(max(1, retries)):
        try:
            text = chat(prompt, temperature=0, timeout=timeout, retries=1)
        except RuntimeError:
            continue
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        candidate = match.group(1) if match else text
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def embed(text, timeout=300, retries=3):
    """Return the embedding vector for one piece of text."""
    model, url = setting("MODEL", EMBED_PREFIXES), setting("URL", EMBED_PREFIXES)
    if not model or not url:
        raise RuntimeError("Missing embedding model settings: set EMBED_MODEL/EMBED_URL in .env.")
    endpoint = url.rstrip("/") + "/embeddings"
    key = setting("API_KEY", EMBED_PREFIXES, "no-key")
    last = None
    for _ in range(max(1, retries)):
        try:
            data = _post(endpoint, {"model": model, "input": text}, key, timeout)
            return data["data"][0]["embedding"]
        except (RuntimeError, KeyError, IndexError, TypeError) as exc:
            last = exc
    raise RuntimeError(f"Embedding model call failed: {last}")
