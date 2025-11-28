# IITM LLM Quiz Solver – Safe Mode Architecture

This repository contains a fully working **FastAPI-based quiz solver** designed for the IITM LLM Analysis Project.

## 📡 API Interface

Your API endpoint receives a `POST` request from the IITM server containing the following payload:

```json
{
  "email": "your_email@example.com",
  "secret": "your_secret",
  "url": "https://example-quiz-url"
}
