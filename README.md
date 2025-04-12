# CodeWave - Disaster Safety Alert System

## Overview

This project is a FastAPI-based application designed to provide targeted safety alerts during disaster scenarios. It leverages various AI and communication technologies to deliver personalized notifications via SMS and voice calls, understand user voice responses, and provide relevant safety information including interactive maps.

The system aims to cater alerts based on user vulnerability types and location, utilizing Retrieval-Augmented Generation (RAG) for contextually relevant safety instructions and Large Language Models (LLMs) for interpreting user needs expressed via voice.

## Key Features

*   **User Registration:** Allows users to register with their vulnerability type, address (geocoded to lat/lon), phone number, guardian details, and notification preferences (SMS/Call, Language).
*   **Disaster Simulation:** An endpoint (`/api/simulate_disaster/`) triggers alert notifications to users within a specified radius of a simulated disaster event.
*   **Targeted Notifications:** Filters users based on proximity to the disaster location.
*   **RAG-Powered Alerts:** Utilizes Upstage AI LLM and embeddings with a FAISS vector store (built from local `.txt` manuals) to generate context-aware SMS and voice alert messages tailored to the disaster type and potentially user vulnerability.
*   **Twilio Integration:**
    *   Sends SMS alerts, including a link to an interactive map.
    *   Initiates outbound voice calls with safety information using Text-to-Speech (TTS).
    *   Handles incoming voice calls, gathers user speech input.
*   **Voice Response Interpretation:** Uses Anthropic's Claude model to analyze user voice responses during alert calls to determine if urgent assistance or reporting is required.
*   **Emergency Contact Notification:** If Claude interprets a user response as requiring assistance, it automatically initiates a voice call to a pre-configured emergency phone number, relaying the user's details and voice message.
*   **External API Integrations:**
    *   Kakao Geocoding API: Converts user addresses to latitude/longitude coordinates.
    *   Safety Data Portal (Korea): Fetches nearby shelter information via a proxy endpoint.
*   **Interactive Map:** Serves an HTML map page (`/map/disaster_map/{simulation_id}`) using Naver Maps API, displaying the user's location (requires browser permission), the disaster epicenter and radius, and the nearest identified shelter with a directions link.
*   **Dashboard Summary (Basic):** An endpoint (`/api/dashboard/summary`) uses an LLM to summarize user voice reports clustered by region.
*   **Multi-language Support (Basic):** Handles basic translation (KO -> EN using Upstage) for messages based on user preference.

## Technology Stack

*   **Backend:** Python, FastAPI
*   **Communication:** Twilio (SMS, Voice)
*   **AI/LLM:**
    *   Upstage AI (Embeddings, LLM for RAG and Translation)
    *   Anthropic (Claude LLM for Voice Interpretation)
    *   Langchain (Orchestration, Vector Store, Prompts)
*   **Database:** SQLAlchemy (with SQLite as default)
*   **Vector Store:** FAISS
*   **Geocoding:** Kakao REST API
*   **Mapping:** Naver Maps API (via JavaScript)
*   **Other:** Uvicorn (ASGI Server), Pydantic (Data Validation), python-dotenv (Environment Variables), httpx (HTTP Client)

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

## Setup and Installation

### Prerequisites

*   Python 3.10+ (Developed with 3.12)
*   `pip` (Python package installer)
*   An `ngrok` account (or similar tunneling service) for local development if testing Twilio webhooks. A paid ngrok plan is recommended to avoid interstitial pages that can interfere with API calls from certain browsers (like iOS Safari).
*   Accounts and API Keys for:
    *   Twilio
    *   Upstage AI
    *   Anthropic
    *   Kakao (REST API Key for Geocoding, JavaScript App Key for Naver Maps via Kakao)
    *   Safety Data Portal (Korea)

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, you might need to install packages listed in `app.py`'s imports manually or generate it using `pip freeze > requirements.txt` after manual installation.)*

4.  **Set Up Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and fill in your actual API keys and configuration values obtained from the respective services (Twilio, Upstage, Anthropic, Kakao, Safety Data Portal). Pay special attention to `TWILIO_PHONE_NUMBER` (must be a Twilio number you own) and `EMERGENCY_PHONE_NUMBER` (the number to call when a user reports an emergency).
    *   Set the `DATABASE_URL` if you want to use a different database than the default SQLite file (`sqlite:///./flask_twilio_demo/users.db`).

5.  **Prepare RAG Data:**
    *   Ensure the `how-to-s/` directory exists in the project root.
    *   Place relevant safety manuals or instruction documents as `.txt` files inside the `how-to-s/` directory. These will be used to build the FAISS vector store on the first run (or if the store doesn't exist).

6.  **Database Setup:**
    *   The application uses SQLAlchemy and Alembic (if configured) or `Base.metadata.create_all()` to manage the database schema.
    *   When the application starts for the first time, it should automatically create the necessary tables in the database specified by `DATABASE_URL` (default: `users.db` in `flask_twilio_demo`).

7.  **(Optional) Frontend Setup:**
    *   The `app.py` includes a catch-all route to serve a frontend application (likely React, based on typical setups) from the `flask_twilio_demo/static/` directory.
    *   If a separate frontend build process is required, follow its specific instructions (e.g., `npm install` and `npm run build`) to generate the static assets in the correct directory.

## Running the Application

1.  **Start the FastAPI Server:**
    ```bash
    uvicorn flask_twilio_demo.app:app --reload --host 0.0.0.0 --port 30000
    ```
    *   `--reload`: Enables auto-reloading during development (remove for production).
    *   `--host 0.0.0.0`: Makes the server accessible on your local network.
    *   `--port 30000`: Specifies the port number (adjust if needed).

2.  **Set up a Tunnel (for Webhooks):**
    *   If testing Twilio webhooks locally (e.g., incoming calls/SMS, voice responses), you need a tunneling service like `ngrok`.
    *   Start ngrok for the port your FastAPI app is running on:
        ```bash
        ngrok http 30000
        ```
    *   Ngrok will provide a public HTTPS URL (e.g., `https://<unique-id>.ngrok.io`).
    *   **Crucially:** Update the `hardcoded_base_url` variable within the `simulate_disaster` function in `app.py` to match this ngrok URL. (Ideally, move this to the `.env` file as `BASE_URL`).
    *   Configure your Twilio phone number's webhook URLs (for incoming SMS and Voice) in the Twilio console to point to your ngrok URL + the respective webhook paths (e.g., `https://<unique-id>.ngrok.io/twilio/sms`, `https://<unique-id>.ngrok.io/twilio/voice`).

3.  **Access the Application:**
    *   **API Documentation (Swagger UI):** Open your browser to `http://localhost:30000/docs` (or your ngrok URL + `/docs`).
    *   **Map Interface (Example):** After running a simulation, access the map link sent via SMS, which will look like `https://<your-ngrok-url>/map/disaster_map/{simulation_id}`.
    *   **Frontend:** Access `http://localhost:30000` (or your ngrok URL).

## Usage Examples

1.  **Register a User:** Use the Swagger UI (`/docs`) to send a POST request to `/api/users/` with the required user details in the request body (using the specified aliases like `personType`, `phone`, etc.).
2.  **Simulate a Disaster:** Send a POST request to `/api/simulate_disaster/` with `DisasterAlertData` in the request body. This will trigger SMS/voice alerts to registered users near the disaster location.

## Important Considerations

*   **In-Memory Simulation Data:** Active simulation data (used for map links) is stored in an in-memory dictionary (`app.state.active_simulations`). This data will be lost if the server restarts. For persistence, consider using a database or cache (like Redis).
*   **ngrok and Webhooks:** A tunneling service like ngrok is essential for Twilio webhooks to reach your local development server. Remember to update the `hardcoded_base_url` in `app.py` or configure it via `.env`. Using a paid ngrok plan is recommended to avoid issues with interstitial pages.
*   **API Keys:** Keep all your API keys secure in the `.env` file and **do not** commit the `.env` file to version control. Use `.env.example` as a template.
*   **Error Handling:** While some error handling is present, further robustness could be added, especially around external API calls and LLM interactions.
*   **Scalability:** The current setup (in-memory storage, default SQLite) is suitable for development but may need adjustments (e.g., PostgreSQL, Redis, Celery for background tasks) for production scaling.
*   **RAG Data Quality:** The effectiveness of the generated alert messages heavily depends on the quality and relevance of the `.txt` files provided in the `how-to-s` directory. 