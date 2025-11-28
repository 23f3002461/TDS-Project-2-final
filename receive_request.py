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
    Background worker:
    - Builds strict prompt
    - Calls AIPipe LLM
    - Sanitizes + auto-fixes generated code
    - Executes script in-process by awaiting main()
    """
    try:
        start_url = data.get("url")
        email = data.get("email")
        secret = data.get("secret")

        # -------------------------------------------------------------
        # STRICT, SAFE, F-STRING-COMPATIBLE PROMPT
        # -------------------------------------------------------------
        prompt = f"""
You are an autonomous quiz-solving agent.
You MUST output ONLY valid Python code. No markdown, no backticks, no explanation.

CRITICAL RULES (FAIL IF VIOLATED):

1. Your script MUST begin with EXACTLY these imports:

import httpx
import asyncio
import base64
from bs4 import BeautifulSoup
import json
import re

THESE IMPORTS MUST BE THE FIRST LINES OF THE SCRIPT.
If any import is missing or reordered, your answer is INVALID.

2. You MUST define:

async def main():

3. main() MUST:
   - Fetch the quiz page at: {start_url}
   - Extract base64-encoded HTML inside <script> tags containing atob("...")
   - Decode with base64.b64decode(...)
   - Parse decoded HTML with BeautifulSoup
   - Extract the quiz question text
   - Compute the answer (number, string, boolean, or dict)
   - Extract the submit URL via regex
   - POST:

     {{
       "email": "{email}",
       "secret": "{secret}",
       "url": "<CURRENT_URL>",
       "answer": ANSWER_VALUE
     }}

   - If the server returns a new "url", recursively follow it.
   - Stop when no new URL exists.
   - Return the FINAL submission response from main().

4. STRICTLY FORBIDDEN:
   - asyncio.run(...)
   - if __name__ == "__main__"
   - subprocess / os.system / playwright / selenium
   - Any external LLM calls

5. DO NOT call main() yourself.
   The platform will call main() automatically.

6. Output ONLY Python code. No extra text.

Start writing the script NOW:
"""

        # -------------------------------------------------------------
        # CALL AIPipe LLM
        # -------------------------------------------------------------
        print("Calling AIPipe to generate solver script...")
        resp_json = await call_aipipe(prompt)
        print("AIPipe response received (raw).")

        # Extract LLM code
        try:
            raw_content = resp_json["choices"][0]["message"]["content"]
        except Exception:
            try:
                raw_content = resp_json["choices"][0]["text"]
            except Exception:
                print("ERROR: Unrecognized AIPipe response:", resp_json)
                return

        script_code = strip_markdown_code(raw_content)
        print("Generated script length:", len(script_code))

        # -------------------------------------------------------------
        # FAILSAFE #1 — Auto-inject imports if missing
        # -------------------------------------------------------------
        required_imports = (
            "import httpx\n"
            "import asyncio\n"
            "import base64\n"
            "from bs4 import BeautifulSoup\n"
            "import json\n"
            "import re\n"
        )

        if not script_code.lstrip().startswith("import httpx"):
            print("Auto-injecting required imports...")
            script_code = required_imports + "\n" + script_code

        # -------------------------------------------------------------
        # FAILSAFE #2 — Strip forbidden patterns
        # -------------------------------------------------------------
        forbidden_patterns = [
            "asyncio.run(",
            "if __name__ == '__main__'",
            "if __name__ == \"__main__\"",
        ]

        for pat in forbidden_patterns:
            if pat in script_code:
                print("Sanitizer: removing forbidden pattern:", pat)
                script_code = script_code.replace(pat, f"# REMOVED_BY_SANITIZER {pat}")

        # -------------------------------------------------------------
        # FAILSAFE #3 — Prevent "main()" from being called inline
        # -------------------------------------------------------------
        # Remove patterns like: main()
        # But keep "async def main():"
        script_code = re.sub(
            r"(?<!def )main\(\)",
            "# REMOVED_BY_SANITIZER main()",
            script_code
        )

        # -------------------------------------------------------------
        # EXECUTE CLEAN SCRIPT
        # -------------------------------------------------------------
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
