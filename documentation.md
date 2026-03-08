# 🚴 Sensorless Cycling Coach — Agentic AI Demo

## Comprehensive Project Documentation

---

## 1. What the App Does

This application is an **educational demonstration** of agentic AI architecture applied to a real-world use case: **personalised cycling training plan generation**.

A user selects a rider profile, types a training goal (e.g. *"Plan my training week. I want to ride 100km total."*), and the system runs a **6-node intelligent pipeline** that:

1. **Loads the rider's profile** — ride history, FTP, preferences, health notes
2. **Analyses fitness** — uses an LLM to assess training load, fatigue, strengths
3. **Checks weather** — 7-day forecast for the rider's location
4. **Plans strategy** — LLM creates a high-level weekly approach
5. **Executes schedule** — a second LLM turns strategy into a day-by-day plan
6. **Quality checks** — an LLM validates the schedule and can **loop back** to re-plan

The core educational value is that the **same logic runs through 3 different orchestration approaches**, letting you compare them directly:

| Mode | Framework | Control Flow |
|---|---|---|
| **LangGraph** | `langgraph.StateGraph` | Graph with conditional edges — quality checker loops back to planner |
| **LangChain** | `langchain LCEL` | Sequential chain (`RunnableLambda \| RunnableLambda`) — inline retry if quality fails |
| **Plain Python** | None | Direct function calls + `if` statement for retry |

---

## 2. Technology Stack & Libraries

### Core Backend

| Library | Version | Purpose |
|---|---|---|
| **FastAPI** | Latest | REST API server with SSE (Server-Sent Events) streaming |
| **Uvicorn** | Latest | ASGI server to run FastAPI |
| **python-dotenv** | Latest | Load `.env` file for API keys |

### AI / LLM Libraries

| Library | Purpose |
|---|---|
| **LangChain** (`langchain`) | Core abstractions: messages, runnables, prompt templates |
| **langchain-nvidia-ai-endpoints** | `ChatNVIDIA` class — connects to NVIDIA NIM API for Kimi k2 |
| **langchain-mistralai** | `ChatMistralAI` class — connects to Mistral API for Mistral Small |
| **LangGraph** (`langgraph`) | `StateGraph` — graph-based agent orchestration with conditional edges |
| **langchain-community** | Community-maintained integrations |

### Frontend

| Library | Purpose |
|---|---|
| **Streamlit** | Interactive web UI with real-time updates |
| **Requests** | HTTP client for calling the FastAPI backend |
| **sseclient-py** | Parse Server-Sent Events stream |

### LLM Models Used

| Model | Provider | Role | Why This Model |
|---|---|---|---|
| **Kimi k2 Instruct** (`moonshotai/kimi-k2-instruct`) | NVIDIA NIM | Planner, Fitness Analyser, Quality Checker | Deep reasoning, large context window (128k tokens), strong multi-step logic — ideal for strategy and analysis |
| **Mistral Small Latest** | Mistral API | Executor | Fast inference, instruction-following optimised, lower latency — ideal for converting plans into structured output |

**Why two different models?**
This demonstrates a real-world pattern: use a **heavyweight reasoning model** for planning and a **lightweight fast model** for execution. This reduces cost and latency while maintaining quality where it matters.

---

## 3. Architecture Deep Dive

### Project Structure

```
agentic-ai-demo/
├── main.py                          # FastAPI server + SSE streaming
├── streamlit_app.py                 # Streamlit frontend (5-tab UI)
├── requirements.txt                 # Python dependencies
├── .env                             # API keys (NVIDIA_API_KEY, MISTRAL_API_KEY)
├── app/
│   ├── agents/
│   │   ├── scheduler_graph.py       # LangGraph pipeline (6 nodes + conditional edge)
│   │   ├── langchain_pipeline.py    # LangChain LCEL pipeline (6 steps + inline retry)
│   │   └── simple_pipeline.py       # Plain Python pipeline (6 steps + if-statement)
│   ├── core/
│   │   ├── llm_factory.py           # LLM instantiation (Kimi k2, Mistral Small)
│   │   └── telemetry.py             # Token counting, thinking logs, state snapshots
│   ├── data/
│   │   └── user_history.py          # 3 dummy rider profiles with ride history
│   └── services/
│       └── weather.py               # Mock 7-day weather forecast
```

### Data Flow

```
User Input ──→ FastAPI ──→ Pipeline (LangGraph / LangChain / Python)
                  │                          │
                  │         ┌────────────────┤
                  │         ↓                ↓
                  │    Load Profile     Analyse Fitness (Kimi k2)
                  │         │                │
                  │         ↓                ↓
                  │    Check Weather    Plan Strategy (Kimi k2)
                  │         │                │
                  │         ↓                ↓
                  │    Execute Schedule (Mistral Small)
                  │         │
                  │         ↓
                  │    Quality Check (Kimi k2) ──→ Loop back if REVISE
                  │         │
                  ↓         ↓
              SSE Stream ←──┘
                  │
                  ↓
              Streamlit UI (5 tabs: Execution, Flow, State, Code, Telemetry)
```

---

## 4. The Three Approaches — Compared

### 4.1 LangGraph (Graph-Based Orchestration)

**How it works:**
- Uses `StateGraph(AgentState)` — a directed graph where each node is a Python function
- Nodes are connected with `.add_edge()` and `.add_conditional_edges()`
- A shared `AgentState` (TypedDict) flows through the graph, accumulating context
- The graph is **compiled** into a runnable: `compiled_graph = workflow.compile()`
- Execution via `compiled_graph.stream(inputs)` yields events per node

**Quality checker loop:**
```python
workflow.add_conditional_edges("quality_checker", should_revise)
# should_revise returns "planner" if verdict is "revise", otherwise END
```

**What shines:**
- ✅ **True conditional control flow** — the graph can loop, branch, and merge
- ✅ **State management is built in** — `AgentState` is automatically passed between nodes
- ✅ **Streaming is native** — `compiled_graph.stream()` yields events as nodes complete
- ✅ **Best for complex, evolving workflows** — adding new nodes or edges is a config change
- ✅ **Visualisable** — the graph topology can be rendered as a diagram

**Where it struggles:**
- ⚠️ **Heavier setup** — defining `TypedDict`, graph construction, compilation
- ⚠️ **Steeper learning curve** — requires understanding graph concepts
- ⚠️ **Overhead for simple tasks** — overkill if your pipeline is strictly linear

**Best for:** Production agentic systems with dynamic routing, retry loops, human-in-the-loop, or multi-agent collaboration.

---

### 4.2 LangChain LCEL (Sequential Chain)

**How it works:**
- Uses `RunnableLambda` to wrap each step function
- Steps are composed with the pipe operator: `step1 | step2 | step3`
- A plain `dict` is passed through the chain as shared state
- Each step receives the dict, modifies it, and returns it

**Quality checker "loop":**
```python
# Chains CANNOT loop — quality check does inline retry
if state["quality_check_result"] == "revise":
    state = step_planner(state, tel, is_revision=True)  # explicit re-call
    state = step_executor(state, tel)
```

**What shines:**
- ✅ **Composable and readable** — the pipe syntax is intuitive: `A | B | C`
- ✅ **Rich ecosystem** — integrates with LangChain's tools, retrievers, output parsers
- ✅ **Good for linear pipelines** — simple left-to-right data flow
- ✅ **Lower overhead than LangGraph** — no graph compilation needed

**Where it struggles:**
- ⚠️ **No native loops or branches** — conditional logic requires breaking out of the chain
- ⚠️ **Retry is manual** — you must explicitly call functions again (inline)
- ⚠️ **State management is manual** — you manage the dict yourself
- ⚠️ **Less suitable for evolving workflows** — adding a branch means rewriting the chain

**Best for:** Straightforward prompt chains, RAG pipelines, or sequential transformations where each step feeds the next.

---

### 4.3 Plain Python (No Framework)

**How it works:**
- Each step is a standalone function: `load_user_profile()`, `analyse_fitness()`, etc.
- Functions are called sequentially in a generator function
- State is managed with local variables
- Retry logic uses a plain `if` statement

**Quality checker "loop":**
```python
if verdict == "revise":
    plan = create_plan(query, user_summary, fitness, weather_summary)
    final = execute_schedule(user_summary, fitness, weather_summary, plan)
```

**What shines:**
- ✅ **Maximum transparency** — every line of control flow is visible
- ✅ **No dependencies** — no framework to learn, install, or debug
- ✅ **Easiest to debug** — standard Python debugging tools work perfectly
- ✅ **Fastest startup** — no compilation, no graph construction

**Where it struggles:**
- ⚠️ **No reusable abstractions** — every new workflow is bespoke code
- ⚠️ **Hardcoded control flow** — every branch must be anticipated and written
- ⚠️ **No streaming for free** — you must manually yield events
- ⚠️ **Scales poorly** — 20-node pipelines become spaghetti code

**Best for:** Prototyping, simple scripts, educational demos, or when you need full control and minimal dependencies.

---

## 5. Performance Comparison: Time & Tokens

> **Note:** Actual numbers vary per run. These are representative ranges based on testing.

### Execution Time (Typical Run)

| Node | LangGraph | LangChain | Plain Python | Notes |
|---|---|---|---|---|
| Profile Loader | ~0.1s | ~0.1s | ~0.1s | Local data lookup — no LLM involved |
| Fitness Analyzer | ~3-8s | ~3-8s | ~3-8s | Kimi k2 LLM call — network-bound |
| Weather Checker | ~0.1s | ~0.1s | ~0.1s | Mock tool — instant |
| Planner | ~5-12s | ~5-12s | ~5-12s | Kimi k2 LLM call — longest reasoning |
| Executor | ~2-5s | ~2-5s | ~2-5s | Mistral Small — fast instruction following |
| Quality Checker | ~2-6s | ~2-6s | ~2-6s | Kimi k2 — evaluating output |
| **Total (no revision)** | **~12-30s** | **~12-30s** | **~12-30s** | LLM latency dominates — framework overhead is negligible |
| **Total (with revision)** | **~20-45s** | **~20-45s** | **~20-45s** | Adds planner + executor re-run |

**Key insight:** Framework overhead is **< 0.5 seconds** across all modes. The dominant cost is **LLM inference latency** (network round-trip + model computation). This means your choice of framework should be based on **developer experience and maintainability**, not raw speed.

### Token Usage (Typical Run)

| Node | Tokens In | Tokens Out | Model |
|---|---|---|---|
| Fitness Analyzer | ~400-600 | ~200-400 | Kimi k2 |
| Planner | ~800-1200 | ~400-800 | Kimi k2 |
| Executor | ~1000-1500 | ~500-1000 | Mistral Small |
| Quality Checker | ~600-900 | ~50-200 | Kimi k2 |
| **Total** | **~2800-4200** | **~1150-2400** | |

Token counts are **identical across all three modes** because they invoke the same LLMs with the same prompts. The framework does not change what gets sent to the model.

---

## 6. Streamlit UI — 5 Tabs Explained

| Tab | What It Shows | Educational Value |
|---|---|---|
| 🚀 **Execution** | Real-time node tracker, "Thought → Action → Observation" logs, output expanders | Watch the agent "think" step by step — shows intermediate reasoning |
| 🔀 **Flow Diagram** | Mermaid diagrams: graph (LangGraph), chain (LangChain), linear (Python) | Visual comparison of control flow topology — loops vs chains vs functions |
| 📦 **State Inspector** | JSON snapshots of agent state after each node | See how context accumulates — each node enriches the shared state |
| 💻 **Source Code** | Actual Python source of each pipeline with syntax highlighting | Compare how 30 lines of LangGraph = 50 lines of LangChain = 70 lines of Python |
| 📊 **Telemetry** | Token counts, API calls, timing per node | Prove that framework overhead is negligible — LLM cost dominates |

---

## 7. Dummy Rider Profiles

The app includes 3 pre-built profiles to test different coaching scenarios:

| Rider | Level | FTP | Location | Goal | Interesting Edge Cases |
|---|---|---|---|---|---|
| **Alex** | Beginner | 180W | Bangalore | Complete first century ride | Low FTP, needs conservative plans |
| **Priya** | Intermediate | 240W | Hyderabad | Improve FTP, race criteriums | Race-focused, needs structured intensity |
| **Ravi** | Advanced | 210W | Chennai | 500km multi-day coastal tour | Endurance-focused, needs multi-day nutrition strategy |

Each profile includes **5+ rides** with detailed metrics (distance, duration, power, heart rate, elevation, notes).

---

## 8. Curveball Prompts — Stress-Testing Adaptability

The sidebar includes challenging prompts that expose framework differences:

| Prompt | What It Tests |
|---|---|
| *"Plan my week but if it rains, switch to indoor training"* | Weather-conditional branching — does the agent adapt the plan mid-stream? |
| *"I can only ride Tuesday 30min + weekends. Hit VO2max Tuesday."* | Multi-constraint handling — specific days, specific training zones |
| *"Build a recovery week — I crashed and my knee hurts"* | Health-aware planning — should reduce load, avoid impact |
| *"I have a 50km race in 4 weeks"* | Periodisation — build volume then taper |

---

## 9. Future Approaches & Enhancements

### 9.1 Short-Term Improvements

| Enhancement | Description |
|---|---|
| **Real weather API** | Replace mock weather with OpenWeatherMap or WeatherAPI for live data |
| **Persistent user profiles** | Store profiles in SQLite/Firebase instead of hardcoded dicts |
| **Token cost tracking** | Calculate actual API cost per run ($/ per 1M tokens) |
| **Parallel node execution** | In LangGraph, run `fitness_analyzer` and `weather_checker` concurrently (they don't depend on each other) |

### 9.2 Medium-Term Architecture

| Enhancement | Description |
|---|---|
| **Multi-agent collaboration** | Separate "Coach Agent" (strategy) and "Nutrition Agent" (fueling plan) that communicate via LangGraph's multi-agent patterns |
| **Human-in-the-loop** | After the planner step, pause and ask the user for feedback before executing — LangGraph supports this natively with `interrupt_before` |
| **Tool calling** | Use LLM tool-calling (function calling) so the model decides *when* to check weather vs. look up routes, rather than hardcoding the sequence |
| **RAG integration** | Add a vector database of cycling training articles — the planner can retrieve relevant knowledge before creating a strategy |

### 9.3 Long-Term Vision

| Enhancement | Description |
|---|---|
| **Streaming LLM output** | Stream tokens from the LLM as they're generated, not just the final response — gives a ChatGPT-like experience |
| **LangSmith observability** | Add LangSmith tracing for production monitoring: latency, token costs, LLM errors, prompt versioning |
| **CrewAI / AutoGen comparison** | Add two more orchestration modes to compare additional multi-agent frameworks |
| **Fine-tuned models** | Train a domain-specific cycling coach model on training plan datasets — faster and cheaper than general LLMs |
| **Mobile app** | Flutter frontend (already partially built) connecting to the same FastAPI backend |
| **Adaptive learning** | Track which plans the user actually follows, and feed completion data back into future plans |

### 9.4 Framework Decision Matrix

When to use each approach in production:

| Scenario | Recommended | Why |
|---|---|---|
| Simple chatbot | Plain Python | One LLM call, no orchestration needed |
| RAG pipeline | LangChain LCEL | Linear: query → retrieve → augment → generate |
| Autonomous agent | LangGraph | Needs loops: plan → act → observe → re-plan |
| Multi-agent system | LangGraph | Agents need to route to each other dynamically |
| Prototype / POC | Plain Python | Fastest to iterate, no framework lock-in |
| Production at scale | LangGraph + LangSmith | Observability, persistence, human-in-the-loop |

---

## 10. How to Run

```bash
# 1. Clone & enter directory
cd agentic-ai-demo

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API keys in .env
echo "NVIDIA_API_KEY=your-key-here" > .env
echo "MISTRAL_API_KEY=your-key-here" >> .env

# 5. Start backend (Terminal 1)
python main.py

# 6. Start frontend (Terminal 2)
python -m streamlit run streamlit_app.py
```

Open **http://localhost:8501** → select a mode → select a rider → type a goal → hit Generate Plan.

---

## 11. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /plan/stream` | POST | Stream pipeline execution as SSE events |
| `GET /source/{mode}` | GET | Get raw Python source code for a pipeline |
| `GET /users` | GET | List all available rider profiles |
| `GET /health` | GET | Health check |

### SSE Event Format

Each streamed event includes:

```json
{
  "node": "fitness_analyzer",
  "status": "Fitness Analyzer (Kimi) finished",
  "thinking": [
    {"thought": "Analysing 5 rides for training load", "action": "Invoke Kimi LLM"},
    {"observation": "Analysis complete: moderate load, ready for progression"}
  ],
  "state_snapshot": {"user_id": "rider_001", "fitness_analysis": "..."},
  "telemetry": {"tokens_in": 450, "tokens_out": 280, "api_calls": 1},
  "elapsed": 4.2
}
```

---

*Built as an educational demo for comparing agentic AI orchestration approaches. Not intended for production cycling coaching.*
