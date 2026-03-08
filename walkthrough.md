# Sensorless Cycling Coach Walkthrough

This walkthrough guides you through setting up and running the Sensorless Cycling Coach, a two-model agentic AI application that generates personalized cycling training plans.

## 1. Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** installed.
- **NVIDIA NIM API Key**: For the Planner agent (Kimi).
- **Mistral API Key**: For the Executor agent (Mistral).

## 2. Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd agentic-ai-demo
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Configuration

1.  **Set up environment variables**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```

2.  **Edit `.env`**:
    Open the `.env` file and add your API keys:
    ```env
    NVIDIA_API_KEY=nvapi-...
    MISTRAL_API_KEY=...
    ```

## 4. Running the Application

You need to run both the FastAPI backend and the Streamlit frontend. It's best to use two separate terminal windows.

### Terminal 1: Start the Backend

Start the FastAPI server which hosts the LangGraph agent:

```bash
python3 main.py
```
*The backend will start at `http://localhost:8000`.*

### Terminal 2: Start the Frontend

In a new terminal (ensure your virtual environment is activated), start the Streamlit UI:

```bash
python3 -m streamlit run streamlit_app.py
```
*The Streamlit app will open in your browser at `http://localhost:8501`.*

## 5. Usage

1.  Open the Streamlit app in your browser.
2.  Enter your training goal in the text area (e.g., "Plan my training week. I want to ride 100km total.").
3.  Click **🚀 Generate Plan**.
4.  Watch the real-time execution:
    - **Planner (Kimi)**: Analyzes your request and creates a high-level strategy.
    - **Executor (Mistral)**: Takes the plan, checks weather (mock tool), and generates a detailed schedule.
5.  Review the final plan and schedule displayed on the screen.

## Troubleshooting

- **Connection Error**: If the Streamlit app says "Cannot connect to the backend", ensure `python3 main.py` is running and accessible at `http://localhost:8000`.
- **API Errors**: Check the terminal output of `main.py` for detailed error logs if generation fails. Ensure your API keys in `.env` are correct.
