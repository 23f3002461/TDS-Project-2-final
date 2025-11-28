# TDS-Project-2-final
minimal app for LLM deployment project
# IITM LLM Quiz Solver – Safe Mode Architecture

This repository contains a fully working **FastAPI-based quiz solver** designed for the IITM LLM Analysis Project.  
Your API endpoint receives a POST request from the IITM server containing:

```json
{
  "email": "your_email@example.com",
  "secret": "your_secret",
  "url": "https://example-quiz-url"
}
Your backend then:

Fetches the quiz page from the url.

Extracts any Base64-encoded HTML containing the question.

Parses and extracts the readable question using BeautifulSoup.

Sends only the question text to the AIPipe (OpenRouter) LLM to compute the answer.

Submits the answer to the quiz’s submit URL found inside the page.

Follows recursive quiz chains (if new quiz URLs are returned).

Stops when the quiz chain completes.

This architecture follows Safe Mode, meaning:

The LLM never generates or runs arbitrary code.

Only the server fetches pages, parses HTML, computes answers and submits.

The LLM is used only to compute the final answer for each question.

🚀 Live Endpoint

Your public POST endpoint (for IITM):

https://web-production-b0da9.up.railway.app/receive_request


This is the URL you must submit in the IITM Google Form.

📦 Features

Fully asynchronous httpx.AsyncClient

Safe HTML parsing using BeautifulSoup

Automatic Base64 decoding of JavaScript-rendered quiz content

Robust URL extraction for submit endpoints

Retry logic for LLM requests

Clean Safe Mode (no subprocess, no dynamic code execution)

3-minute overall runtime guard

Works on Railway / Render / any cloud

📁 Project Structure
.
├── receive_request.py    # Main FastAPI server (safe-mode solver)
├── requirements.txt      # All dependencies
├── README.md             # Documentation
└── LICENSE               # MIT License

🔧 Environment Variables

Create .env:

SECRET_KEY=your_secret_here
AIPIPE_TOKEN=your_openrouter_aipipe_token


Both must be added to Railway → Variables.

🧪 Testing With CURL
curl -X POST "https://web-production-b0da9.up.railway.app/receive_request" \
-H "Content-Type: application/json" \
-d '{
  "email": "your_email@example.com",
  "secret": "your_secret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}'


Expected response:
{ "message": "Request accepted" }

📜 How It Works Internally
/receive_request validates the secret.
Background task process_request() starts execution.
Solver fetches the URL.
Extracts Base64 → decodes → parses question.
Sends question to AIPipe using a strict JSON-only prompt.
Receives answer.
Submits JSON to the quiz’s submit endpoint.
If response contains "url", it follows the next quiz.
Stops when submission endpoint returns non-JSON.
All steps logged with helpful prints for debugging.

⚙️ Deployment Instructions (Railway)
Create a Python service.
Upload these files.
Add environment variables:
SECRET_KEY
AIPIPE_TOKEN

Set Start Command:
uvicorn receive_request:app --host 0.0.0.0 --port $PORT
Deploy.

📦 requirements.txt
fastapi
uvicorn
python-dotenv
httpx
beautifulsoup4
