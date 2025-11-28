# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn", "python-dotenv", "httpx"]
import os
import httpx
import subprocess
import re
import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
SECRET_KEY = os.getenv("SECRET_KEY")

app = FastAPI()

def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Try to find code block with ```python or ```
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # If no code block found, return the text as-is
    return text.strip()

async def process_request(data):
    """Process the IITM quiz in background."""
    
    prompt = f"""You are an autonomous quiz-solving agent. Generate a COMPLETE, RUNNABLE Python script.

CRITICAL REQUIREMENTS:
1. This script will be executed directly via subprocess - it must be 100% self-contained
2. The script must handle JavaScript-rendered pages (use playwright for headless browser)
3. Extract questions from base64-encoded content using atob() decoding
4. Solve the quiz question using analysis/calculation
5. Find the submit URL from within the page content
6. Submit answer and handle recursive quiz chains
7. Continue until no new URL is returned

INITIAL PARAMETERS:
- Email: {data["email"]}
- Secret: {data["secret"]}
- Starting URL: {data["url"]}

YOUR SCRIPT MUST:
1. Install/import: playwright, httpx, beautifulsoup4, base64, json, re
2. Use playwright to render JavaScript (handle atob() decoding)
3. Parse the rendered HTML to extract the question text
4. Solve the question (do calculations, data processing, etc.)
5. Find submit URL in the page (look for patterns like "https://*/submit")
6. POST the answer with format:
   {{
     "email": "{data["email"]}",
     "secret": "{data["secret"]}",
     "url": "<current_quiz_url>",
     "answer": <your_computed_answer>
   }}
7. Check response for "correct" field and new "url" field
8. If new URL exists, recursively solve that quiz
9. Continue until no new URL (quiz chain complete)
10. Print detailed progress and final result

IMPORTANT:
- Handle file downloads if question mentions files (.pdf, .csv, etc.)
- Answer can be: number, string, boolean, or JSON object
- Must complete within 3 minutes
- Use async/await properly with playwright
- Add error handling and timeouts

OUTPUT ONLY VALID PYTHON CODE - NO EXPLANATIONS OR MARKDOWN:
"""

    try:
        # Call LLM API with async client
        async with httpx.AsyncClient(timeout=60.0) as client:
            llm_response = await client.post(
                AIPIPE_URL,
                headers={
                    "Authorization": f"Bearer {AIPIPE_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",  # Using more capable model
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an expert Python developer. Generate complete, executable Python scripts. Output ONLY the Python code with no markdown formatting, explanations, or comments outside the code."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4000,  # Increased for longer scripts
                    "temperature": 0.3
                }
            )
            llm_response.raise_for_status()
            
    except httpx.TimeoutException:
        print("ERROR: LLM API request timed out")
        return
    except httpx.HTTPStatusError as e:
        print(f"ERROR: LLM API returned status {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return
    except Exception as e:
        print(f"ERROR calling LLM API: {e}")
        return
    
    try:
        result = llm_response.json()
        print("\n" + "="*60)
        print("LLM API Response received")
        print("="*60)
        
        # Extract the script content
        raw_content = result["choices"][0]["message"]["content"]
        script_code = extract_code_from_markdown(raw_content)
        
        # Save script
        script_path = "generated_script.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)
        
        print(f"\n✓ Script saved to {script_path}")
        print(f"✓ Script length: {len(script_code)} characters")
        print("\nScript preview (first 500 chars):")
        print("-" * 60)
        print(script_code[:500])
        print("-" * 60)
        
        # Check if playwright is available
        try:
            import playwright
            print("\n✓ Playwright is installed")
        except ImportError:
            print("\n⚠ WARNING: Playwright not installed. Script may fail.")
            print("  Run: pip install playwright && playwright install chromium")
        
        # Run the generated script
        print(f"\n{'='*60}")
        print("EXECUTING GENERATED SCRIPT")
        print("="*60 + "\n")
        
        proc = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )
        
        print("\n" + "="*60)
        print("SCRIPT EXECUTION COMPLETE")
        print("="*60)
        
        if proc.stdout:
            print("\n--- SCRIPT OUTPUT ---")
            print(proc.stdout)
        
        if proc.stderr:
            print("\n--- SCRIPT ERRORS ---")
            print(proc.stderr)
        
        if proc.returncode == 0:
            print(f"\n✓ Script executed successfully (exit code: 0)")
        else:
            print(f"\n✗ Script failed with exit code: {proc.returncode}")
            
    except subprocess.TimeoutExpired:
        print("\n✗ ERROR: Generated script timed out (>3 minutes)")
    except KeyError as e:
        print(f"\n✗ ERROR: Unexpected API response format - missing key {e}")
        print(f"Full response: {result}")
    except Exception as e:
        print(f"\n✗ ERROR processing script: {e}")
        import traceback
        traceback.print_exc()

@app.post("/receive_request")
async def receive_request(request: Request, background_tasks: BackgroundTasks):
    """Entry point IITM server will call."""
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(
            {"error": f"Invalid JSON: {str(e)}"}, 
            status_code=400
        )
    
    # Validate secret first
    if not data.get("secret"):
        return JSONResponse(
            {"error": "Missing secret"}, 
            status_code=400
        )
    
    if data.get("secret") != SECRET_KEY:
        return JSONResponse(
            {"error": "Forbidden"}, 
            status_code=403
        )
    
    # Validate other required fields
    required_fields = ["url", "email"]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        return JSONResponse(
            {"error": f"Missing required fields: {', '.join(missing)}"}, 
            status_code=400
        )
    
    print("\n" + "="*60)
    print(f"REQUEST ACCEPTED")
    print("="*60)
    print(f"Email: {data.get('email')}")
    print(f"URL: {data.get('url')}")
    print("="*60 + "\n")
    
    # Run quiz solving in background
    background_tasks.add_task(process_request, data)
    
    return JSONResponse(
        {"message": "Request accepted"}, 
        status_code=200
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "IITM Quiz Solver",
        "status": "running",
        "endpoint": "/receive_request",
        "method": "POST"
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("IITM QUIZ SOLVER SERVICE")
    print("="*60)
    print(f"Secret Key: {'✓ Set' if SECRET_KEY else '✗ NOT SET'}")
    print(f"API Token: {'✓ Set' if AIPIPE_TOKEN else '✗ NOT SET'}")
    print("="*60 + "\n")
    
    if not SECRET_KEY or not AIPIPE_TOKEN:
        print("⚠ WARNING: Environment variables not set properly!")
        print("  Set SECRET_KEY and AIPIPE_TOKEN in your .env file\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
