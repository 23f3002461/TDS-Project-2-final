# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn", "python-dotenv", "httpx", "beautifulsoup4"]

import os
import re
import asyncio
import traceback
import json
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import base64

load_dotenv()

# ----- Config -----
SECRET_KEY = os.getenv("SECRET_KEY")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
# AIPipe OpenRouter endpoint (confirmed)
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
LLM_MODEL = "openai/gpt-4.1-nano"

if not SECRET_KEY or not AIPIPE_TOKEN:
    print("WARNING: SECRET_KEY or AIPIPE_TOKEN not set in environment")

# ----- App & global async client -----
app = FastAPI()
http_client = httpx.AsyncClient(timeout=60.0)


# ----- Helpers -----
def safe_json_load(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


async def call_aipipe_for_answer(question_text: str, retries: int = 2, timeout: float = 30.0) -> Any:
    """
    Ask AIPipe to compute the answer for the given question_text.
    Returns the parsed answer (number/string/bool/object) or raises an exception.
    This function strictly requests a JSON response: {"answer": <value>}
    """
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    # Strict prompt: produce a single JSON object with key "answer"
    prompt = (
        "You are an assistant that reads a single quiz question (plaintext) and RETURNS ONLY a JSON object "
        "with exactly one key 'answer'. Do NOT include any extra text, explanation, or markdown. The value "
        "should be the computed answer: a number, string, boolean, or JSON object. Example: {\"answer\": 123}\n\n"
        "QUESTION:\n"
        f"{question_text}\n\n"
        "Return only: {\"answer\": ... }\n"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You MUST output only valid JSON with a single key named 'answer'."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            resp = await http_client.post(AIPIPE_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            j = resp.json()
            # Expecting choices[0].message.content (OpenRouter/Chat-like)
            content = None
            try:
                content = j["choices"][0]["message"]["content"]
            except Exception:
                # fallback older shapes
                content = j.get("choices", [{}])[0].get("text") if isinstance(j.get("choices"), list) else None

            if content is None:
                raise ValueError(f"No content in AIPipe response: {j}")

            # Trim whitespace and try to extract a JSON substring
            text = content.strip()

            # If content contains code fences, strip them
            # e.g. ```json\n{"answer": 5}\n```
            m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
            if m:
                json_text = m.group(1)
            else:
                # Try to find first { ... } block
                m2 = re.search(r"(\{.*\})", text, re.DOTALL)
                json_text = m2.group(1) if m2 else text

            parsed = safe_json_load(json_text)
            if parsed is None or "answer" not in parsed:
                # If parsing failed, try the whole text as a bare value (e.g., "42" or "true")
                stripped = text.strip().strip('"').strip("'")
                # attempt to convert to int/float/bool
                if re.fullmatch(r"-?\d+", stripped):
                    return int(stripped)
                if re.fullmatch(r"-?\d+\.\d+", stripped):
                    return float(stripped)
                if stripped.lower() in ("true", "false"):
                    return stripped.lower() == "true"
                # Otherwise raise to retry
                raise ValueError(f"Could not parse JSON answer from LLM output: {text}")

            return parsed["answer"]

        except Exception as e:
            last_exc = e
            print(f"AIPipe attempt {attempt} failed: {repr(e)}")
            if attempt <= retries:
                await asyncio.sleep(1.0 * attempt)
            else:
                break

    raise last_exc


def extract_base64_from_html(html: str) -> Optional[str]:
    """
    Search for atob("...") pattern and return the base64 string or None.
    """
    # match atob("...") or atob('...')
    m = re.search(r'atob\(\s*["\']([^"\']+)["\']\s*\)', html)
    if m:
        return m.group(1)
    return None


def decode_base64_to_html(b64: str) -> Optional[str]:
    try:
        b = base64.b64decode(b64)
        return b.decode("utf-8", errors="replace")
    except Exception:
        return None

def find_submit_url_in_html(html: str, base_url: str) -> Optional[str]:
    """
    Extract submit URL from HTML. Supports both absolute and relative URLs.
    """

    # 1. Absolute URL
    m = re.search(r"https?://[^\s'\"<>]+/submit[^\s'\"<>]*", html)
    if m:
        return m.group(0)

    # 2. JSON-like "url":"...something..."
    m2 = re.search(r'"url"\s*:\s*"([^"]+)"', html)
    if m2:
        url_candidate = m2.group(1).strip()

        # Already absolute?
        if url_candidate.startswith("http://") or url_candidate.startswith("https://"):
            return url_candidate

        # Starts with "/", relative path
        if url_candidate.startswith("/"):
            from urllib.parse import urljoin
            return urljoin(base_url, url_candidate)

        # Bare relative path → also join
        from urllib.parse import urljoin
        return urljoin(base_url, "/" + url_candidate)

    # 3. Hard fallback: search for "/submit"
    m3 = re.search(r"/submit[^\s'\"<>]*", html)
    if m3:
        from urllib.parse import urljoin
        return urljoin(base_url, m3.group(0))

    return None



def extract_question_text(decoded_html: str) -> str:
    """
    Extract human-readable question text from decoded HTML.
    We look at #result, or first <pre>, or body text.
    """
    soup = BeautifulSoup(decoded_html, "html.parser")
    # Prefer element with id="result"
    el = soup.find(id="result")
    if el and el.get_text(strip=True):
        return el.get_text(separator="\n", strip=True)
    # fallback to first <pre>
    pre = soup.find("pre")
    if pre and pre.get_text(strip=True):
        return pre.get_text(separator="\n", strip=True)
    # fallback to body text
    body = soup.body
    if body:
        return body.get_text(separator="\n", strip=True)
    return decoded_html.strip()


# ----- Core background task (SAFE MODE) -----
async def process_request(data: Dict[str, Any]):
    """
    SAFE MODE:
    - The server performs fetching, decoding, parsing and submission.
    - The LLM is used ONLY to compute the answer for a single question_text via call_aipipe_for_answer().
    - This avoids asking the LLM to generate code; it only returns an answer value.
    """
    try:
        start_url = data.get("url")
        email = data.get("email")
        secret = data.get("secret")

        if not start_url or not start_url.startswith(("http://", "https://")):
            print("Invalid URL:", start_url)
            return

        # set an overall deadline for the whole job (e.g., 150 seconds)
        overall_deadline = asyncio.get_event_loop().time() + 150

        url = start_url
        last_result = None

        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                if asyncio.get_event_loop().time() > overall_deadline:
                    print("Overall deadline exceeded; aborting.")
                    break

                try:
                    resp = await client.get(url)
                except Exception as e:
                    print("HTTP GET failed for", url, repr(e))
                    break

                html = resp.text or ""
                # 1) try to extract base64 embedded content
                b64 = extract_base64_from_html(html)
                decoded_html = None
                if b64:
                    decoded_html = decode_base64_to_html(b64)
                # if no decoded HTML found, treat the original html as the page content
                page_to_parse = decoded_html if decoded_html else html

                # 2) extract question text
                question_text = extract_question_text(page_to_parse)
                if not question_text:
                    print("No question text found on page:", url)
                    break

                # 3) find submit URL inside decoded content or original html
                submit_url = find_submit_url_in_html(page_to_parse, url)
                if not submit_url:
                    print("No submit URL found on page; aborting. Page snippet:", page_to_parse[:400])
                    break

                # 4) compute answer using LLM (safe single-question call)
                try:
                    answer = await call_aipipe_for_answer(question_text)
                except Exception as e:
                    print("LLM failed to compute answer:", repr(e))
                    break

                # 5) prepare payload and submit
                payload = {
                        "email": email,
                        "secret": secret,
                        "url": url,
                        "answer": answer,
                    }

                try:
                    post_resp = await client.post(submit_url, json=payload, timeout=60.0)
                except Exception as e:
                    print("POST to submit_url failed:", submit_url, repr(e))
                    break

                # Try parsing JSON. If not JSON → assume quiz complete.
                try:
                    post_json = post_resp.json()
                except Exception:
                    text = (post_resp.text or "").strip()
                    print("Submit response was not JSON; treating as final:", text[:200])
                    last_result = {"final": True, "raw_response": text}
                    break

                print("Submit response:", post_json)
                last_result = post_json


                # If there's a next url, follow it; else finish
                next_url = None
                if isinstance(post_json, dict) and "url" in post_json and post_json["url"]:
                    next_url = post_json["url"]

                if not next_url:
                    break

                # protect against infinite loops
                if next_url == url:
                    print("Next URL same as current; aborting loop to avoid infinite recursion.")
                    break

                url = next_url

        print("Background task finished. Last result:", last_result)
        return

    except Exception:
        print("process_request unexpected error:\n", traceback.format_exc())
        return


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

    print("Starting IITM Quiz Solver (SAFE MODE TEMPLATE)")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
