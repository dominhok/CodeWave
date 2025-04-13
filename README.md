# CodeWave - Rapid & Personalized Disaster Safety Alert System

## Project Goal

**To provide rapid and personalized safety information via SMS and voice calls during disasters, considering user location and vulnerability type.**

Leveraging AI technology, CodeWave delivers essential evacuation guidance and identifies users requesting help via voice, automatically connecting them to relevant authorities.

## Overview

CodeWave is an application integrating various AI and communication technologies to deliver tailored safety alerts in disaster situations. It utilizes RAG (Retrieval-Augmented Generation) and LLMs (Large Language Models) to generate contextually relevant safety information and interpret voice responses to identify users needing urgent assistance.

## Key Features

### 1. Personalized Alerting

*   **User Profile:** Register users with location (address geocoded to coordinates), vulnerability type (e.g., visually impaired, elderly), preferred language, contact details, etc.
*   **Location-Based Targeting:** Sends alerts only to users within the radius of a simulated disaster location.
*   **AI-Powered Custom Messages:**
    *   Generates **core action guidance** (SMS/Voice) based on disaster type using RAG with safety manuals (from `how-to-s/` directory).
    *   Adjusts message nuances based on user vulnerability type (via LLM prompts).

### 2. Interactive Communication & Emergency Support

*   **Multi-Channel Notifications:** Sends SMS and voice (TTS) alerts via Twilio API.
*   **Voice Response Recognition:** Collects user voice responses and performs STT (Speech-to-Text) via Twilio.
    *   Provides **Hints** to improve STT accuracy (e.g., "help", "report", "safe", "need assistance").
*   **AI Response Analysis (Claude 3.5 Sonnet):** Analyzes collected voice messages to determine if urgent support/reporting is needed.
*   **Automatic Emergency Call:** If support is needed, automatically initiates a call to the pre-configured `EMERGENCY_PHONE_NUMBER`.
    *   Relays **user information (phone number, location)** and the **voice message content** to the operator.

### 3. Information Access

*   **Real-time Disaster Info:** Provides nearby shelter information via integration with the Safety Data Portal API (proxy endpoint).
*   **Address Geocoding:** Converts user addresses to latitude/longitude coordinates using Kakao API.
*   **Interactive Map:**
    *   Sends a map link via SMS (`/map/disaster_map/{simulation_id}`).
    *   Displays disaster epicenter, radius, user location, nearest shelter, and directions link on the map (Naver Maps API).
*   **(Experimental) Dashboard Summary:** Summarizes user voice reports by region using an LLM (`/api/dashboard/summary`).

## Technology Stack

*   **Backend:** Python, FastAPI
*   **AI/LLM:**
    *   Langchain (Orchestration)
    *   Anthropic Claude 3.5 Sonnet (Voice Interpretation, RAG Summarization)
    *   Upstage (Embeddings, Translation KO->EN)
    *   FAISS (Vector Store)
*   **Communication:** Twilio (SMS, Voice, TTS, STT)
*   **Database:** SQLAlchemy (Default: SQLite)
*   **APIs & Mapping:** Kakao Geocoding, Safety Data Portal (Korea), Naver Maps API
*   **Infra & Tools:** Uvicorn, Pydantic, python-dotenv, httpx, Ngrok (for Development Webhooks)

## Quick Start

### Prerequisites

*   Python 3.10+
*   `pip`
*   API Keys/Accounts: Twilio, Anthropic, Upstage, Kakao (REST & JS), Safety Data Portal (Korea)
*   Ngrok (Optional, for local webhook testing)

### Installation & Setup

1.  **Clone:** `git clone <repo-url> && cd <repo-dir>`
2.  **Virtual Env:** `python -m venv venv && source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
3.  **Install Deps:** `pip install -r requirements.txt`
4.  **Environment Variables:**
    *   `cp .env.example .env`
    *   Edit `.env` and **fill in all required API keys and phone numbers**.
5.  **RAG Data:** Place relevant safety manuals (`.txt` files) in the `how-to-s/` directory. The vector store will be built automatically on first run.

### Running

1.  **Start Server:**
    ```bash
    uvicorn flask_twilio_demo.app:app --reload --host 0.0.0.0 --port 30000
    ```
2.  **Ngrok (if needed for webhooks):**
    *   `ngrok http 30000 --domain <your-reserved-ngrok-domain.ngrok.app>` (Replace with your actual reserved domain)
    *   Update Twilio webhook URLs in the Twilio console to point to your ngrok HTTPS URL (e.g., `https://<your-domain>.ngrok.app/twilio/sms`).
    *   **(Important for Map Links):** Ensure the `hardcoded_base_url` variable in `app.py` (around line 1091) matches your Ngrok URL **OR** preferably, set `BASE_URL` in your `.env` file and modify `app.py` to use `os.getenv("BASE_URL", "default_url")`.
3.  **Access:**
    *   **API Docs:** `http://localhost:30000/docs`
    *   **Map (after simulation):** Link sent via SMS (e.g., `https://<your-ngrok-domain>.ngrok.app/map/disaster_map/{sim_id}`)

## Key Endpoints

*   `/api/users/` (POST): Register a new user.
*   `/api/simulate_disaster/` (POST): Trigger a disaster simulation and send alerts.
*   `/map/disaster_map/{simulation_id}` (GET): View the interactive map.
*   `/twilio/sms` & `/twilio/voice`: Webhook endpoints for Twilio.

## Project Structure (Key Files/Dirs)

```
.
├── flask_twilio_demo/
│   ├── __init__.py
│   ├── app.py          # Main FastAPI application logic, API endpoints
│   ├── models.py       # SQLAlchemy database models
│   ├── static/         # Static files (CSS, JS, Frontend Build Output)
│   │   ├── static/     # (Potentially nested static dir from frontend build)
│   │   └── maps/       # HTML map templates (e.g., disaster_map.html)
│   ├── .env            # Environment variables (sensitive keys) - DO NOT COMMIT
│   └── users.db        # Default SQLite database file
├── how-to-s/           # Directory for .txt safety manuals used for RAG
├── .env.example        # Example environment variable file
├── README.md           # This file
└── requirements.txt    # Python dependencies
``` 