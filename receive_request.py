# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn", "python-dotenv", "httpx"]

import os
import re
import asyncio
import traceback
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

# ----- Config -----
SECRET_KEY = os.getenv("SECRET_KEY")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
LLM_MODEL = "openai/gpt-4.1-nano"

if not SECRET_KEY or not AIPIPE_TOKEN:
    # do not crash on startup; print a warning for logs
    print("WARNING: SECRET_KEY or AIPIPE_TOKEN not set in environment")

# ----- App & global async client -----
app = FastAPI()
http_client = httpx.AsyncClient(timeout=60.0)  # global async client


# ----- Helpers -----
def strip_markdown_code(text: str) -> str:
    """
    Extract the first fenced code block if present, otherwise return the text.
    """
    # find ```...``` blocks
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


async def call_aipipe(prompt: str, retries: int = 2, backoff: float = 1.0) -> Dict[str, Any]:
    """
    Call the AIPipe OpenRouter endpoint (async), with simple retry/backoff.
    Returns parsed JSON response or raises an exception.
    """
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You generate short, correct Python scripts (no explanation)."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            resp = await http_client.post(AIPIPE_URL, headers=headers, json=payload)
            # raise_for_status will allow us to inspect body on non-2xx
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            # log minimal info and retry
            print(f"AIPipe call attempt {attempt} failed: {repr(e)}")
            if attempt <= retries:
                await asyncio.sleep(backoff * attempt)
            else:
                break
    raise last_exc


async def exec_generated_code(code: str, timeout: float = 100.0):
    """
    Execute generated code in-process. The generated script SHOULD define
    an async function: async def main(): ... which will be awaited.
    This avoids subprocess and heavy dependencies.
    """
    # prepare isolated namespaces (limited)
    global_ns: Dict[str, Any] = {"__name__": "__generated__"}
    local_ns: Dict[str, Any] = {}

    try:
        # compile first to raise syntax errors early
        compiled = compile(code, "<generated>", "exec")
        exec(compiled, global_ns, local_ns)
    except Exception:
        print("EXECUTION - compile/exec error:\n", traceback.format_exc())
        return {"status": "error", "reason": "compile_error"}

    # run async main() if present
    main = local_ns.get("main") or global_ns.get("main")
    if main and asyncio.iscoroutinefunction(main):
        try:
            result = await asyncio.wait_for(main(), timeout=timeout)
            return {"status": "ok", "result": result}
        except asyncio.TimeoutError:
            print("EXECUTION - generated script timed out")
            return {"status": "error", "reason": "timeout"}
        except Exception:
            print("EXECUTION - runtime error:\n", traceback.format_exc())
            return {"status": "error", "reason": "runtime_error"}
    else:
        # no async main - attempt to call sync main() if exists
        if callable(main):
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, main)
                return {"status": "ok", "result": result}
            except Exception:
                print("EXECUTION - sync main() error:\n", traceback.format_exc())
                return {"status": "error", "reason": "sync_main_error"}
        else:
            print("EXECUTION - no main() found in generated script")
            return {"status": "error", "reason": "no_main"}


# ----- Core background task -----
async def process_request(data: Dict[str, Any]):
    """
    Background worker that:
    - builds a concise robust prompt instructing the LLM to return ONLY Python code
    - calls AIPipe to generate the solver script
    - executes the generated code in-process (expects async main)
    """
    try:
        start_url = data.get("url")
        email = data.get("email")
        secret = data.get("secret")

        prompt = f"""
You are an autonomous quiz-solving agent. Output ONLY a complete Python script (no explanation, no markdown).
The script MUST start with these imports exactly:
import httpx
import asyncio
import base64
from bs4 import BeautifulSoup
import json
import re

Constraints:
- Use only Python standard library and httpx (async).
- The script MUST define: `async def main():` which performs the complete solve & returns the final submission result (a JSON-serializable object or value).
- The script MUST fetch the starting URL: {start_url}, extract the quiz question (handle base64-encoded HTML embedded in <script> tags), compute the answer, find the submit URL in the page, POST the JSON:
  {{ "email": "{email}", "secret": "{secret}", "url": "<current_quiz_url>", "answer": <answer> }}
- If the server returns a new 'url', the script MUST follow it and repeat until no new url is provided.
- The script MUST use httpx.AsyncClient() and async/await.
- No playwright, no selenium, no external binaries. Detect and decode base64 fragments and HTML in scripts.
- Keep the script concise and deterministic; avoid long commentary and do not call external LLMs.
- Print minimal progress and return the final server response (the last POST response) from main().

Start script now.
"""

        # call AIPipe
        print("Calling AIPipe to generate solver script...")
        resp_json = await call_aipipe(prompt)
        print("AIPipe response received (raw).")

        # debug: if choices missing, print entire response and abort
        if not isinstance(resp_json, dict) or "choices" not in resp_json:
            print("AIPipe response missing 'choices':", resp_json)
            return

        # Extract code from response
        try:
            content = resp_json["choices"][0]["message"]["content"]
        except Exception:
            # older / different shape? try alternative keys
            try:
                content = resp_json["choices"][0]["text"]
            except Exception:
                print("Unexpected AIPipe response shape:", resp_json)
                return

        script_code = strip_markdown_code(content)
        print("Generated script length:", len(script_code))

        # Very small safety gate: ensure no heavy imports are present
        banned = ["playwright", "selenium", "subprocess", "os.system", "pexpect"]
        for b in banned:
            if b in script_code:
                print(f"Generated script contains banned token '{b}'. Aborting execution.")
                return

        # Execute the generated script in-process
        print("Executing generated script in-process (awaiting main()) ...")
        exec_result = await exec_generated_code(script_code, timeout=120.0)
        print("Execution result:", exec_result)

    except Exception:
        print("process_request unexpected error:\n", traceback.format_exc())


# ----- API endpoints -----
@app.post("/receive_request")
async def receive_request(request: Request, background_tasks: BackgroundTasks):
    # parse json
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # secret checks
    if data.get("secret") is None:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if data.get("secret") != SECRET_KEY:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    # required fields
    if not data.get("url") or not data.get("email"):
        return JSONResponse({"error": "Missing required fields (url/email)"}, status_code=400)

    # accept and run background task
    background_tasks.add_task(process_request, data)
    return JSONResponse({"message": "Request accepted"}, status_code=200)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"service": "IITM Quiz Solver", "endpoint": "/receive_request"}


# ----- cleanup on shutdown -----
@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


# ----- if run directly -----
if __name__ == "__main__":
    import uvicorn

    print("Starting IITM Quiz Solver (async, in-process execution)")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
