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

# ===== CONFIG =====
SECRET_KEY = os.getenv("SECRET_KEY")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
LLM_MODEL = "openai/gpt-4.1-nano"

app = FastAPI()
http_client = httpx.AsyncClient(timeout=60.0)


# ===== HELPERS =====

def safe_json_load(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except:
        return None


async def call_aipipe_for_answer(question_text: str) -> Any:
    """
    Strict JSON-only LLM call.
    """
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    prompt = (
        "Read the quiz question below and return ONLY a JSON object with a single key 'answer'.\n"
        "No markdown. No explanation. No extra text.\n"
        "Examples:\n"
        '{"answer": 123}\n'
        '{"answer": "hello"}\n'
        '{"answer": true}\n\n'
        "QUESTION:\n" + question_text
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return ONLY a JSON with a single key 'answer'."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    resp = await http_client.post(AIPIPE_URL, headers=headers, json=payload)
    resp.raise_for_status()

    j = resp.json()
    content = j["choices"][0]["message"]["content"].strip()

    # Extract JSON object
    m = re.search(r"{.*}", content, re.DOTALL)
    if not m:
        raise ValueError("LLM did not return JSON: " + content)

    parsed = safe_json_load(m.group(0))
    if not parsed or "answer" not in parsed:
        raise ValueError("LLM JSON missing 'answer': " + content)

    return parsed["answer"]


def extract_base64(html: str) -> Optional[str]:
    m = re.search(r'atob\(["\']([^"\']+)["\']\)', html)
    return m.group(1) if m else None


def decode_b64(b64: str) -> Optional[str]:
    try:
        return base64.b64decode(b64).decode("utf-8", errors="replace")
    except:
        return None


def extract_question_text(decoded_html: str) -> str:
    soup = BeautifulSoup(decoded_html, "html.parser")

    # #result
    r = soup.find(id="result")
    if r:
        return r.get_text(separator="\n", strip=True)

    # <pre>
    p = soup.find("pre")
    if p:
        return p.get_text(separator="\n", strip=True)

    # fallback body
    if soup.body:
        return soup.body.get_text(separator="\n", strip=True)

    return decoded_html.strip()


def find_submit_url(html: str, base_url: str) -> Optional[str]:
    """
    Robust extraction of the REAL submit URL.
    It ONLY returns URLs ending with '/submit' and ignores broken HTML.
    """

    # 1. Absolute correct submit URL
    m = re.search(r"https?://[^\s\"'>]+/submit\b", html)
    if m:
        return m.group(0)

    # 2. JSON-like "url": "/submit"
    m2 = re.search(r'"url"\s*:\s*"(/submit[^"]*)"', html)
    if m2:
        from urllib.parse import urljoin
        return urljoin(base_url, m2.group(1))

    # 3. ANY text containing /submit but NOT inside HTML tags
    # Prevent matching things like "<span class...."
    m3 = re.search(r"(?<!<)[/][s]ubmit[^\s\"'>]*", html, re.IGNORECASE)
    if m3:
        from urllib.parse import urljoin
        return urljoin(base_url, m3.group(0))

    return None



# ===== CORE QUIZ PROCESSOR =====

async def process_request(data: Dict[str, Any]):
    start_url = data["url"]
    email = data["email"]
    secret = data["secret"]

    url = start_url
    last_result = None

    print("====================================")
    print("PROCESS REQUEST START")
    print("Email:", email)
    print("URL:", start_url)
    print("====================================")

    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            print(f"\n--- Fetching Quiz Page: {url}")
            try:
                resp = await client.get(url)
            except Exception as e:
                print("Fetch error:", repr(e))
                break

            html = resp.text or ""

            # 1️⃣ decode base64 (if present) else use html as-is
            decoded_html = None
            b64 = extract_base64(html)
            if b64:
                decoded_html = decode_b64(b64)

            page = decoded_html if decoded_html else html

            # 2️⃣ extract question
            question = extract_question_text(page)

            # ⭐ IMPORTANT FIX:
            if not question:
                print("No question found — THIS IS EXPECTED FOR FIRST PAGE.")
                question = ""   # just send dummy or blank; LLM won't be used yet

            # 3️⃣ submit URL
            submit_url = find_submit_url(page, url)
            if not submit_url:
                print("❌ No submit URL; stopping.")
                break

            # 4️⃣ COMPUTE ANSWER
            if question.strip():
                # real question → ask LLM
                try:
                    answer = await call_aipipe_for_answer(question)
                except Exception as e:
                    print("LLM compute error:", repr(e))
                    break
            else:
                # first page → answer doesn't matter
                answer = "start"

            # 5️⃣ submit
            payload = {
                "email": email,
                "secret": secret,
                "url": url,
                "answer": answer
            }

            print(f"Submitting → {submit_url}")
            try:
                p = await client.post(submit_url, json=payload)
            except Exception as e:
                print("POST failed:", repr(e))
                break

            # 6️⃣ parse JSON OR finish
            try:
                j = p.json()
            except Exception:
                print("Submit returned non-JSON → finished.")
                last_result = {"final": True, "raw": p.text}
                break

            print("Submit response:", j)
            last_result = j

            # next URL?
            nxt = j.get("url")
            if not nxt:
                break

            url = nxt

    print("===== FINAL RESULT =====")
    print(last_result)
    print("========================")


# ===== API ENDPOINTS =====

@app.post("/receive_request")
async def receive_request(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, 400)

    if data.get("secret") != SECRET_KEY:
        return JSONResponse({"error": "Forbidden"}, 403)

    if not data.get("url") or not data.get("email"):
        return JSONResponse({"error": "Missing required fields"}, 400)

    background_tasks.add_task(process_request, data)
    return {"message": "Request accepted"}


@app.get("/")
async def root():
    return {"service": "IITM Quiz Solver", "endpoint": "/receive_request"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
