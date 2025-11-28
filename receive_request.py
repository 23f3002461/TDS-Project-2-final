# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn", "python-dotenv", "httpx"]

import os
import httpx
import subprocess
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
SECRET_KEY = os.getenv("SECRET_KEY")

app = FastAPI()


async def process_request(data):
    """Process the IITM quiz in background."""

    prompt = f"""
You are an autonomous quiz-solving agent. Generate a complete, runnable Python script.
This Python script will be executed directly by my server. You MUST output ONLY Python code.

============================================================
                   GLOBAL REQUIREMENTS
============================================================
You MUST generate a Python script that:

1. Fetches the quiz page at: {data.get("url")}
2. Extracts the question text (HTML, JS-rendered, or base64-encoded).
3. Solves the question using Python only.
4. Finds the submit URL inside the quiz page.
5. POSTs this JSON to the submit URL:

{{
  "email": "{data["email"]}",
  "secret": "{data["secret"]}",
  "url": "{data["url"]}",
  "answer": <your_answer>
}}

6. Reads the server response. If a new quiz URL is provided, solve recursively.
7. Continue until no new URL is returned.
8. Print the final response.
9. Use only Python standard libraries + httpx.
10. Output ONLY Python code. No explanations.

============================================================
Token for optional use inside the generated script:
{AIPIPE_TOKEN}
============================================================
Return ONLY the Python code now:
"""

    # Call AIPipe API
    llm_response = httpx.post(
        AIPIPE_URL,
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {"role": "system", "content": "You generate Python scripts."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2000
        }
    )

    result = llm_response.json()
    print("AIPipe Raw Response:", result)
    script_code = result["choices"][0]["message"]["content"]

    # save script
    with open("generated_script.py", "w") as f:
        f.write(script_code)

    # run script
    proc = subprocess.run(
        ["python3", "generated_script.py"],
        capture_output=True, text=True
    )

    print("Generated script output:", proc.stdout)
    print("Generated script error:", proc.stderr)


@app.post("/receive_request")
async def receive_request(request: Request, background_tasks: BackgroundTasks):
    """Entry point IITM server will call."""

    try:
        data = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    if data.get("secret") != SECRET_KEY:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    # run quiz in background
    background_tasks.add_task(process_request, data)

    return {"message": "Request accepted"}
