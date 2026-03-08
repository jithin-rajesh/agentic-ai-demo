import streamlit as st
import requests
import json
import time

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚴 Sensorless Cycling Coach",
    page_icon="🚴",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem; color: white;
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; color: white; }
.main-header p { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; color: #e0e0ff; }

.mode-badge {
    display: inline-block; padding: 0.35rem 1rem; border-radius: 20px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.5px;
}
.mode-langgraph { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.4); }
.mode-langchain { background: rgba(236,72,153,0.2); color: #f472b6; border: 1px solid rgba(236,72,153,0.4); }
.mode-none { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }

.arch-badge {
    display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin-right: 0.5rem;
}
.kimi-badge { background: rgba(167,139,250,0.2); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.mistral-badge { background: rgba(110,231,183,0.2); color: #6ee7b7; border: 1px solid rgba(110,231,183,0.3); }
.tool-badge { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.qc-badge { background: rgba(244,114,182,0.2); color: #f472b6; border: 1px solid rgba(244,114,182,0.3); }

.node-active { border-left: 4px solid #fbbf24; padding-left: 1rem; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
.node-done { border-left: 4px solid #34d399; padding-left: 1rem; }
.node-waiting { border-left: 4px solid #6b7280; padding-left: 1rem; opacity: 0.5; }

.sidebar-info {
    background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.2);
    border-radius: 12px; padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem;
}
.user-card {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; font-size: 0.82rem; line-height: 1.5;
}

.thinking-entry {
    background: rgba(99,102,241,0.05); border-left: 3px solid #818cf8;
    padding: 0.5rem 0.8rem; margin: 0.3rem 0; border-radius: 0 8px 8px 0;
    font-size: 0.82rem; font-family: 'Fira Code', monospace;
}
.thinking-thought { color: #818cf8; }
.thinking-action { color: #fbbf24; }
.thinking-observation { color: #34d399; }

div.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 12px;
    padding: 0.7rem 2rem; font-size: 1rem; font-weight: 600;
    width: 100%; transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.4);
}

.manual-qc-card {
    background: rgba(251,191,36,0.08); border: 2px solid rgba(251,191,36,0.4);
    border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
}
.human-qc-badge {
    display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
    background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid rgba(52,211,153,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────
NODES = [
    {"key": "user_profile_loader", "label": "Profile Loader",  "icon": "👤", "sub": "Load rider history & preferences",  "type": "tool"},
    {"key": "fitness_analyzer",    "label": "Fitness Analyzer", "icon": "📊", "sub": "Analyse training load & readiness",  "type": "kimi"},
    {"key": "weather_checker",     "label": "Weather Checker",  "icon": "🌤️", "sub": "7-day forecast for location",       "type": "tool"},
    {"key": "planner",             "label": "Planner",          "icon": "🧠", "sub": "High-level training strategy",       "type": "kimi"},
    {"key": "executor",            "label": "Executor",         "icon": "⚡", "sub": "Final structured schedule",          "type": "mistral"},
    {"key": "quality_checker",     "label": "Quality Check",    "icon": "✅", "sub": "Validate & conditionally re-plan",   "type": "qc"},
]
NODE_ORDER = [n["key"] for n in NODES]

TYPE_BADGES = {
    "kimi":    '<span class="arch-badge kimi-badge">Kimi</span>',
    "mistral": '<span class="arch-badge mistral-badge">Mistral</span>',
    "tool":    '<span class="arch-badge tool-badge">Tool</span>',
    "qc":      '<span class="arch-badge qc-badge">QC</span>',
}

MODE_API_MAP = {"LangGraph": "langgraph", "LangChain": "langchain", "None (Plain Python)": "none"}
MODE_CLASS   = {"LangGraph": "mode-langgraph", "LangChain": "mode-langchain", "None (Plain Python)": "mode-none"}
MODE_DESCRIPTIONS = {
    "LangGraph": "Graph-based orchestration — nodes + edges + conditional loops.",
    "LangChain": "Sequential LCEL chain — RunnableLambda steps piped with |.",
    "None (Plain Python)": "No framework — plain functions + if-statements.",
}

USER_INFO = {
    "rider_001": {"name": "Alex",  "level": "Beginner",     "ftp": 180, "location": "Bangalore", "goal": "Complete first century ride"},
    "rider_002": {"name": "Priya", "level": "Intermediate", "ftp": 240, "location": "Hyderabad", "goal": "Improve FTP, race criteriums"},
    "rider_003": {"name": "Ravi",  "level": "Advanced",     "ftp": 210, "location": "Chennai",   "goal": "500km multi-day coastal tour"},
}

MERMAID_DIAGRAMS = {
    "LangGraph": """```mermaid
graph TD
    START((Start)) --> A["👤 Profile Loader<br/><small>Tool</small>"]
    A --> B["📊 Fitness Analyzer<br/><small>Kimi LLM</small>"]
    B --> C["🌤️ Weather Checker<br/><small>Tool</small>"]
    C --> D["🧠 Planner<br/><small>Kimi LLM</small>"]
    D --> E["⚡ Executor<br/><small>Mistral LLM</small>"]
    E --> F{"✅ Quality Check<br/><small>Kimi LLM</small>"}
    F -->|PASS| END_NODE((End))
    F -->|REVISE| D

    style F fill:#f472b6,stroke:#ec4899,color:#1a1a2e
    style D fill:#818cf8,stroke:#6366f1,color:#1a1a2e
    style E fill:#6ee7b7,stroke:#34d399,color:#1a1a2e
    style START fill:#667eea,stroke:#764ba2,color:white
    style END_NODE fill:#34d399,stroke:#059669,color:#1a1a2e
```""",
    "LangChain": """```mermaid
graph LR
    A["👤 Profile Loader"] -->|pipe| B["📊 Fitness Analyzer"]
    B -->|pipe| C["🌤️ Weather Checker"]
    C -->|pipe| D["🧠 Planner"]
    D -->|pipe| E["⚡ Executor"]
    E -->|pipe| F["✅ Quality Check"]
    F -.->|"inline retry<br/>(no loop)"| D2["🧠 Re-plan"]
    D2 -.-> E2["⚡ Re-execute"]

    style A fill:#f472b6,stroke:#ec4899,color:#1a1a2e
    style B fill:#f472b6,stroke:#ec4899,color:#1a1a2e
    style C fill:#f472b6,stroke:#ec4899,color:#1a1a2e
    style D fill:#f472b6,stroke:#ec4899,color:#1a1a2e
    style E fill:#f472b6,stroke:#ec4899,color:#1a1a2e
    style F fill:#f472b6,stroke:#ec4899,color:#1a1a2e
    style D2 fill:#f472b6,stroke:#ec4899,color:#1a1a2e,stroke-dasharray: 5 5
    style E2 fill:#f472b6,stroke:#ec4899,color:#1a1a2e,stroke-dasharray: 5 5
```""",
    "None (Plain Python)": """```mermaid
graph TD
    A["👤 load_user_profile()"] --> B["📊 analyse_fitness()"]
    B --> C["🌤️ fetch_weather()"]
    C --> D["🧠 create_plan()"]
    D --> E["⚡ execute_schedule()"]
    E --> F["✅ quality_check()"]
    F --> G{"if verdict == 'revise':"}
    G -->|True| D2["🧠 create_plan() again"]
    D2 --> E2["⚡ execute_schedule() again"]
    G -->|False| END_NODE["return result"]

    style G fill:#fbbf24,stroke:#f59e0b,color:#1a1a2e
    style D2 fill:#fbbf24,stroke:#f59e0b,color:#1a1a2e,stroke-dasharray: 5 5
    style E2 fill:#fbbf24,stroke:#f59e0b,color:#1a1a2e,stroke-dasharray: 5 5
```""",
}

# ── Helpers ──────────────────────────────────────────────────────────────
def render_thinking_html(thinking_entries):
    if not thinking_entries:
        return ""
    parts = []
    for entry in thinking_entries:
        lines = []
        if "thought" in entry:
            lines.append(f'<span class="thinking-thought">💭 Thought:</span> {entry["thought"]}')
        if "action" in entry:
            lines.append(f'<span class="thinking-action">🔧 Action:</span> {entry["action"]}')
        if "observation" in entry:
            lines.append(f'<span class="thinking-observation">👁️ Observation:</span> {entry["observation"]}')
        parts.append('<div class="thinking-entry">' + "<br>".join(lines) + "</div>")
    return "".join(parts)


def render_node_tracker(placeholders, active_key, completed):
    for i, node in enumerate(NODES):
        key = node["key"]
        if key in completed:
            cls, icon = "node-done", "✅"
        elif key == active_key:
            cls, icon = "node-active", "⏳"
        else:
            cls, icon = "node-waiting", "⬜"
        badge = TYPE_BADGES.get(node["type"], "")
        placeholders[i].markdown(
            f'<div class="{cls}">{icon} <strong>{node["label"]}</strong> {badge}'
            f'<br><small>{node["sub"]}</small></div>',
            unsafe_allow_html=True,
        )


# ── Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚴 Sensorless Cycling Coach</h1>
    <p>Multi-Model Agentic Architecture — Compare LangGraph, LangChain &amp; Plain Python</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔀 Pipeline Mode")
    mode = st.select_slider("Choose orchestration framework",
        options=["LangGraph", "LangChain", "None (Plain Python)"], value="LangGraph")
    mc = MODE_CLASS[mode]
    st.markdown(f'<span class="mode-badge {mc}">{mode.upper()}</span>', unsafe_allow_html=True)
    st.caption(MODE_DESCRIPTIONS[mode])

    manual_qc = st.toggle("🧑‍⚖️ Manual QC", value=False,
        help="Replace automatic LLM quality check with human-in-the-loop review")
    if manual_qc:
        st.caption("Pipeline will pause after Executor for your review.")
    st.markdown("---")

    st.markdown("### 👤 Select Rider")
    user_labels = {uid: f"{info['name']} ({info['level']})" for uid, info in USER_INFO.items()}
    selected_user = st.selectbox("Rider profile", options=list(USER_INFO.keys()),
        format_func=lambda uid: user_labels[uid])
    u = USER_INFO[selected_user]
    st.markdown(f'<div class="user-card"><strong>{u["name"]}</strong> — {u["level"]}<br>'
                f'📍 {u["location"]} &nbsp; ⚡ FTP {u["ftp"]}W<br>🎯 {u["goal"]}</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ Architecture")
    st.markdown("""
    <div class="sidebar-info">
        <span class="arch-badge kimi-badge">🧠 Planner</span> <strong>Kimi k2</strong><br>
        <small>Reasoning &amp; strategy via NVIDIA NIM</small>
    </div>
    <div class="sidebar-info">
        <span class="arch-badge mistral-badge">⚡ Executor</span> <strong>Mistral Small</strong><br>
        <small>Fast execution via Mistral API</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    api_url = st.text_input("Backend URL", value="http://localhost:8000")
    st.markdown("---")

    st.markdown("### 💡 Example Prompts")
    examples = [
        "Plan my training week. I want to ride 100km total.",
        "I have a 50km race in 4 weeks. Build me a plan.",
        "Plan my week but if it rains, switch to indoor training.",
        "I can only ride Tuesday 30min + weekends. Hit VO2max Tuesday.",
        "Build a recovery week — I crashed and my knee hurts.",
    ]
    for ex in examples:
        if st.button(f"📝 {ex[:50]}...", key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

# ── Main Input ───────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area("What's your training goal?",
        value=st.session_state.get("query_input", ""),
        height=100, placeholder="e.g. Plan my training week. I want to ride 100km total.")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("🚀 Generate Plan", use_container_width=True)

# ── Execution ────────────────────────────────────────────────────────────
if submit and query:
    st.markdown("---")
    mode_api = MODE_API_MAP[mode]
    st.markdown(f'### 🔄 Pipeline — <span class="mode-badge {mc}">{mode}</span>'
                + (' &nbsp; <span class="human-qc-badge">🧑‍⚖️ Manual QC</span>' if manual_qc else ''),
                unsafe_allow_html=True)

    # ── Create ALL tabs and ALL containers in a SINGLE pass ──────────
    tab_exec, tab_flow, tab_state, tab_code, tab_telemetry = st.tabs([
        "🚀 Execution", "🔀 Flow Diagram", "📦 State Inspector", "💻 Source Code", "📊 Telemetry"
    ])

    # Flow Diagram tab — static content
    with tab_flow:
        st.markdown("#### Logic Flow — " + mode)
        st.markdown(MERMAID_DIAGRAMS[mode])
        if mode == "LangGraph":
            st.info("🔄 **LangGraph** uses a compiled StateGraph with conditional edges. The quality checker can loop back to the planner — true graph control flow.")
        elif mode == "LangChain":
            st.warning("⛓️ **LangChain** chains are sequential. Retry is done *inline* by re-calling functions — no native loop.")
        else:
            st.warning("🐍 **Plain Python** uses explicit `if` statements. Simple but inflexible — you must hard-code every branch.")

    # Source Code tab — static content
    with tab_code:
        st.markdown(f"#### Source Code — `{mode_api}` Pipeline")
        try:
            src_resp = requests.get(f"{api_url}/source/{mode_api}", timeout=5)
            if src_resp.status_code == 200:
                st.code(src_resp.text, language="python", line_numbers=True)
            else:
                st.error("Could not fetch source code.")
        except Exception:
            st.error("Backend not reachable for source code.")

    # State Inspector tab — placeholder
    with tab_state:
        st.markdown("#### State Snapshots")
        state_tab_placeholder = st.empty()
        state_tab_placeholder.info("⏳ Waiting for pipeline execution…")

    # Telemetry tab — placeholder
    with tab_telemetry:
        st.markdown("#### 📊 Telemetry Dashboard")
        telemetry_placeholder = st.empty()
        telemetry_placeholder.info("⏳ Waiting for pipeline execution…")

    # Execution tab — ALL containers created in one block
    with tab_exec:
        graph_col, detail_col = st.columns([1, 2])
        with graph_col:
            node_placeholders = [st.empty() for _ in NODES]
            done_placeholder = st.empty()
        with detail_col:
            log_container = st.empty()
            thinking_container = st.empty()
            detail_containers = {n["key"]: st.empty() for n in NODES}
            # Container for Manual QC decision area
            manual_qc_container = st.empty()
            timing_container = st.empty()

    # ── Initialize node tracker ──────────────────────────────────────
    completed_nodes = set()
    render_node_tracker(node_placeholders, None, completed_nodes)
    done_placeholder.markdown(
        '<div class="node-waiting">⬜ <strong>Complete</strong><br><small>Return response</small></div>',
        unsafe_allow_html=True)

    # ── Data accumulators ────────────────────────────────────────────
    logs = []
    thinking_html = []
    node_times = {}
    all_telemetry = {}
    state_snapshots = []
    node_details = []  # Store node output data for replay across reruns
    paused_pipeline_state = None  # Will be set if manual_qc_pause event arrives
    pause_telemetry_summary = None

    # ── Helper: finalise the pipeline UI ─────────────────────────────
    def finalise_pipeline(total, tel_summary, qc_label=None):
        """Common 'done' logic — fill timing table, state inspector, telemetry tab."""
        if qc_label:
            logs.append(f"**{total}s** — 🧑‍⚖️ {qc_label}")
        logs.append(f"**{total}s** — ✅ Pipeline complete")

        completed_nodes.update(NODE_ORDER)
        render_node_tracker(node_placeholders, None, completed_nodes)
        done_placeholder.markdown(
            '<div class="node-done">✅ <strong>Complete</strong><br><small>Return response</small></div>',
            unsafe_allow_html=True)
        log_container.success("\n\n".join(logs))

        # Timing table
        prev = 0
        rows = []
        for n in NODES:
            t = node_times.get(n["key"], 0)
            delta = round(t - prev, 1)
            model = {"kimi": "Kimi k2", "mistral": "Mistral Small", "tool": "Tool/Data", "qc": "Kimi k2"}[n["type"]]
            t_in = all_telemetry.get(n["key"], {}).get("tokens_in", 0)
            t_out = all_telemetry.get(n["key"], {}).get("tokens_out", 0)
            calls = all_telemetry.get(n["key"], {}).get("api_calls", 0)
            rows.append(f"| {n['icon']} {n['label']} | {model} | **{delta}s** | {t_in} | {t_out} | {calls} |")
            prev = t
        timing_container.markdown(
            "---\n#### ⏱️ Execution Summary\n"
            "| Node | Model | Time | Tok In | Tok Out | API |\n|---|---|---|---|---|---|\n"
            + "\n".join(rows)
            + f"\n| **Total** | | **{total}s** | **{tel_summary.get('total_tokens_in', 0)}** | **{tel_summary.get('total_tokens_out', 0)}** | **{tel_summary.get('total_api_calls', 0)}** |"
        )

        # State inspector
        if state_snapshots:
            md_parts = []
            for snap in state_snapshots:
                md_parts.append(f"**After `{snap['node']}`:**")
                md_parts.append(f"```json\n{json.dumps(snap['snapshot'], indent=2, default=str)}\n```")
            state_tab_placeholder.markdown("\n\n".join(md_parts))
        else:
            state_tab_placeholder.info("No state snapshots available.")

        # Telemetry
        if tel_summary and tel_summary.get("per_node"):
            per = tel_summary["per_node"]
            t_md = (
                f"**Total Tokens:** {tel_summary.get('total_tokens', 0):,} &nbsp;&nbsp; "
                f"**API Calls:** {tel_summary.get('total_api_calls', 0)} &nbsp;&nbsp; "
                f"**Time:** {total}s\n\n---\n\n"
                "| Node | Tokens In | Tokens Out | API Calls |\n|---|---|---|---|\n"
            )
            for nn in per:
                t_md += f"| {nn} | {per[nn]['tokens_in']} | {per[nn]['tokens_out']} | {per[nn]['api_calls']} |\n"
            telemetry_placeholder.markdown(t_md)
        else:
            telemetry_placeholder.info("No telemetry data available.")

    # ── Helper: process a regular node event ──────────────────────────
    def process_node_event(data, node, status, elapsed, thinking, snapshot, tel):
        node_times[node] = elapsed
        completed_nodes.add(node)
        if tel:
            all_telemetry[node] = tel
        if snapshot:
            state_snapshots.append({"node": node, "snapshot": snapshot})
        # Save node output data for replay across reruns
        node_details.append({"node": node, **data})

        # Next active node
        try:
            idx = NODE_ORDER.index(node)
            next_active = NODE_ORDER[idx + 1] if idx + 1 < len(NODE_ORDER) else None
        except ValueError:
            next_active = None

        node_info = next((n for n in NODES if n["key"] == node), None)
        icon = node_info["icon"] if node_info else "🔧"
        logs.append(f"**{elapsed}s** — {icon} {status}")

        if thinking:
            label = node_info["label"] if node_info else node
            thinking_html.append(f"<strong>{icon} {label}</strong>")
            thinking_html.append(render_thinking_html(thinking))

        render_node_tracker(node_placeholders, next_active, completed_nodes)
        log_container.info("\n\n".join(logs))
        if thinking_html:
            thinking_container.markdown("".join(thinking_html), unsafe_allow_html=True)

        # Output expanders
        container = detail_containers.get(node)
        if container:
            if node == "user_profile_loader":
                with container.expander("👤 **User Profile Loaded**", expanded=False):
                    st.markdown(data.get("detail", ""))
            elif node == "fitness_analyzer":
                with container.expander("📊 **Fitness Analysis (Kimi)**", expanded=True):
                    st.markdown(data.get("detail", ""))
            elif node == "weather_checker":
                with container.expander("🌤️ **Weather Forecast**", expanded=False):
                    st.markdown(data.get("detail", ""))
            elif node == "planner":
                with container.expander("🧠 **Planner Output (Kimi)**", expanded=True):
                    st.markdown(data.get("plan", ""))
            elif node == "executor":
                with container.expander("⚡ **Executor Output (Mistral)**", expanded=True):
                    st.markdown(data.get("final_response", ""))
            elif node == "quality_checker":
                qr = data.get("quality_result", "pass")
                with container.expander(f"✅ **Quality Check: {qr.upper()}**", expanded=False):
                    st.markdown(data.get("detail", ""))

    # ── Stream from backend ──────────────────────────────────────────
    try:
        with requests.post(
            f"{api_url}/plan/stream",
            json={"user_id": selected_user, "query": query, "mode": mode_api,
                  "manual_qc": manual_qc},
            stream=True, timeout=300,
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])
                node = data.get("node", "")
                status = data.get("status", "")
                elapsed = data.get("elapsed", 0)
                thinking = data.get("thinking", [])
                snapshot = data.get("state_snapshot", {})
                tel = data.get("telemetry", {})

                if node == "done":
                    finalise_pipeline(elapsed, data.get("telemetry_summary", {}))

                elif node == "manual_qc_pause":
                    # Pipeline paused — store state for PASS/REVISE decision
                    paused_pipeline_state = data.get("pipeline_state", {})
                    pause_telemetry_summary = data.get("telemetry_summary", {})
                    logs.append(f"**{elapsed}s** — ⏸️ Paused for human QC review")
                    log_container.warning("\n\n".join(logs))
                    # Update node tracker: show QC as the next active
                    render_node_tracker(node_placeholders, "quality_checker", completed_nodes)

                    # Persist everything to session_state so UI survives reruns
                    st.session_state["qc_paused"] = {
                        "mode": mode,
                        "manual_qc": manual_qc,
                        "query": query,
                        "selected_user": selected_user,
                        "api_url": api_url,
                        "mode_api": mode_api,
                        "logs": list(logs),
                        "thinking_html": list(thinking_html),
                        "node_times": dict(node_times),
                        "all_telemetry": dict(all_telemetry),
                        "state_snapshots": list(state_snapshots),
                        "completed_nodes": set(completed_nodes),
                        "node_details": list(node_details),
                        "pipeline_state": paused_pipeline_state,
                        "pause_telemetry_summary": pause_telemetry_summary,
                    }

                elif node == "error":
                    log_container.error(f"❌ Error: {status}")

                else:
                    process_node_event(data, node, status, elapsed, thinking, snapshot, tel)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend. Is `python3 main.py` running?")
    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

    # ── Manual QC decision UI ────────────────────────────────────────
    if paused_pipeline_state is not None:
        with manual_qc_container.container():
            st.markdown('<div class="manual-qc-card">', unsafe_allow_html=True)
            st.markdown("#### 🧑‍⚖️ Human Quality Check")
            st.markdown("The Executor has produced a training schedule. Review it above, then decide:")

            # Educational callout
            st.info(
                "🎓 **How this works behind the scenes:**\n\n"
                "- **LangGraph**: Would use `interrupt_before('quality_checker')` for "
                "native pause/resume with a single persistent graph session\n"
                "- **LangChain / Python**: Requires manual state serialisation and "
                "two separate API calls (what we're simulating here)"
            )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                pass_clicked = st.button("✅ PASS — Schedule looks good", use_container_width=True, key="qc_pass")
            with btn_col2:
                revise_clicked = st.button("🔄 REVISE — Needs improvement", use_container_width=True, key="qc_revise")
            st.markdown('</div>', unsafe_allow_html=True)

            if pass_clicked:
                # Mark QC as passed by human
                completed_nodes.add("quality_checker")
                # Update the QC node in the tracker
                node_placeholders[5].markdown(
                    '<div class="node-done">✅ <strong>Quality Check</strong> '
                    '<span class="human-qc-badge">🧑‍⚖️ Human</span>'
                    '<br><small>Human QC: PASS</small></div>',
                    unsafe_allow_html=True)
                manual_qc_container.empty()
                finalise_pipeline(
                    max(node_times.values()) if node_times else 0,
                    pause_telemetry_summary or {},
                    qc_label="Human QC: PASS ✅",
                )

            elif revise_clicked:
                manual_qc_container.empty()
                # Show revision in progress
                logs.append("**…** — 🔄 Revising plan (human requested)")
                log_container.warning("\n\n".join(logs))

                # Call /plan/revise endpoint
                try:
                    with requests.post(
                        f"{api_url}/plan/revise",
                        json={
                            "user_id": selected_user,
                            "query": query,
                            "mode": mode_api,
                            "pipeline_state": paused_pipeline_state,
                        },
                        stream=True, timeout=300,
                    ) as rev_resp:
                        rev_resp.raise_for_status()
                        for rev_line in rev_resp.iter_lines(decode_unicode=True):
                            if not rev_line or not rev_line.startswith("data: "):
                                continue
                            rev_data = json.loads(rev_line[6:])
                            rev_node = rev_data.get("node", "")
                            rev_status = rev_data.get("status", "")
                            rev_elapsed = rev_data.get("elapsed", 0)
                            rev_thinking = rev_data.get("thinking", [])
                            rev_snapshot = rev_data.get("state_snapshot", {})
                            rev_tel = rev_data.get("telemetry", {})

                            if rev_node == "done":
                                # Mark QC as passed after revision
                                completed_nodes.add("quality_checker")
                                node_placeholders[5].markdown(
                                    '<div class="node-done">✅ <strong>Quality Check</strong> '
                                    '<span class="human-qc-badge">🧑‍⚖️ Human</span>'
                                    '<br><small>Human QC: PASS (after revision)</small></div>',
                                    unsafe_allow_html=True)
                                finalise_pipeline(
                                    max(node_times.values()) if node_times else 0,
                                    rev_data.get("telemetry_summary", pause_telemetry_summary or {}),
                                    qc_label="Human QC: PASS (after revision) ✅",
                                )
                            elif rev_node == "error":
                                log_container.error(f"❌ Revision Error: {rev_status}")
                            else:
                                process_node_event(rev_data, rev_node, rev_status, rev_elapsed, rev_thinking, rev_snapshot, rev_tel)

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend for revision.")
                except Exception as e:
                    st.error(f"❌ Revision error: {e}")

elif submit and not query:
    st.warning("Please enter a training goal first.")

# ── Persistent Manual QC UI (survives Streamlit reruns) ──────────────────
# When PASS/REVISE is clicked, Streamlit reruns the script. submit is False,
# so the if-block above is skipped. This elif catches that case by reading
# the pipeline results we saved into session_state.
elif "qc_paused" in st.session_state:
    saved = st.session_state["qc_paused"]
    s_mode = saved["mode"]
    s_mc = MODE_CLASS[s_mode]
    s_mode_api = saved["mode_api"]

    # Detect which button was clicked on the previous run
    pass_clicked = st.session_state.pop("qc_pass", False)
    revise_clicked = st.session_state.pop("qc_revise", False)

    # ── Reconstruct the full pipeline UI from saved state ────────────
    st.markdown("---")
    st.markdown(f'### 🔄 Pipeline — <span class="mode-badge {s_mc}">{s_mode}</span>'
                ' &nbsp; <span class="human-qc-badge">🧑‍⚖️ Manual QC</span>',
                unsafe_allow_html=True)

    tab_exec, tab_flow, tab_state, tab_code, tab_telemetry = st.tabs([
        "🚀 Execution", "🔀 Flow Diagram", "📦 State Inspector", "💻 Source Code", "📊 Telemetry"
    ])

    with tab_flow:
        st.markdown("#### Logic Flow — " + s_mode)
        st.markdown(MERMAID_DIAGRAMS[s_mode])

    with tab_code:
        st.markdown(f"#### Source Code — `{s_mode_api}` Pipeline")
        try:
            src_resp = requests.get(f"{saved['api_url']}/source/{s_mode_api}", timeout=5)
            if src_resp.status_code == 200:
                st.code(src_resp.text, language="python", line_numbers=True)
            else:
                st.error("Could not fetch source code.")
        except Exception:
            st.error("Backend not reachable for source code.")

    with tab_state:
        st.markdown("#### State Snapshots")
        state_tab_placeholder = st.empty()
        if saved["state_snapshots"]:
            md_parts = []
            for snap in saved["state_snapshots"]:
                md_parts.append(f"**After `{snap['node']}`:**")
                md_parts.append(f"```json\n{json.dumps(snap['snapshot'], indent=2, default=str)}\n```")
            state_tab_placeholder.markdown("\n\n".join(md_parts))
        else:
            state_tab_placeholder.info("No state snapshots available.")

    with tab_telemetry:
        st.markdown("#### 📊 Telemetry Dashboard")
        telemetry_placeholder = st.empty()
        telemetry_placeholder.info("⏳ Telemetry will be finalized after QC decision…")

    with tab_exec:
        graph_col, detail_col = st.columns([1, 2])
        with graph_col:
            node_placeholders = [st.empty() for _ in NODES]
            done_placeholder = st.empty()
        with detail_col:
            log_container = st.empty()
            thinking_container = st.empty()
            detail_containers = {n["key"]: st.empty() for n in NODES}
            manual_qc_container = st.empty()
            timing_container = st.empty()

    # Restore accumulators from saved state (mutable references for helpers)
    logs = saved["logs"]
    thinking_html = saved["thinking_html"]
    node_times = saved["node_times"]
    all_telemetry = saved["all_telemetry"]
    state_snapshots = saved["state_snapshots"]
    completed_nodes = saved["completed_nodes"]
    node_details = saved["node_details"]

    # ── Render restored node tracker ─────────────────────────────────
    render_node_tracker(node_placeholders, "quality_checker", completed_nodes)
    done_placeholder.markdown(
        '<div class="node-waiting">⬜ <strong>Complete</strong><br><small>Return response</small></div>',
        unsafe_allow_html=True)
    log_container.warning("\n\n".join(logs))
    if thinking_html:
        thinking_container.markdown("".join(thinking_html), unsafe_allow_html=True)

    # ── Replay node detail expanders ─────────────────────────────────
    for nd in node_details:
        nk = nd["node"]
        container = detail_containers.get(nk)
        if container:
            if nk == "user_profile_loader":
                with container.expander("👤 **User Profile Loaded**", expanded=False):
                    st.markdown(nd.get("detail", ""))
            elif nk == "fitness_analyzer":
                with container.expander("📊 **Fitness Analysis (Kimi)**", expanded=False):
                    st.markdown(nd.get("detail", ""))
            elif nk == "weather_checker":
                with container.expander("🌤️ **Weather Forecast**", expanded=False):
                    st.markdown(nd.get("detail", ""))
            elif nk == "planner":
                with container.expander("🧠 **Planner Output (Kimi)**", expanded=False):
                    st.markdown(nd.get("plan", ""))
            elif nk == "executor":
                with container.expander("⚡ **Executor Output (Mistral)**", expanded=True):
                    st.markdown(nd.get("final_response", ""))

    # ── Finalise helper (scoped for this block) ──────────────────────
    def finalise_qc(total, tel_summary, qc_label):
        logs.append(f"**{total}s** — 🧑‍⚖️ {qc_label}")
        logs.append(f"**{total}s** — ✅ Pipeline complete")
        completed_nodes.update(NODE_ORDER)
        render_node_tracker(node_placeholders, None, completed_nodes)
        done_placeholder.markdown(
            '<div class="node-done">✅ <strong>Complete</strong><br><small>Return response</small></div>',
            unsafe_allow_html=True)
        log_container.success("\n\n".join(logs))
        # Timing table
        prev = 0
        rows = []
        for n in NODES:
            t = node_times.get(n["key"], 0)
            delta = round(t - prev, 1)
            model = {"kimi": "Kimi k2", "mistral": "Mistral Small", "tool": "Tool/Data", "qc": "Kimi k2"}[n["type"]]
            t_in = all_telemetry.get(n["key"], {}).get("tokens_in", 0)
            t_out = all_telemetry.get(n["key"], {}).get("tokens_out", 0)
            calls = all_telemetry.get(n["key"], {}).get("api_calls", 0)
            rows.append(f"| {n['icon']} {n['label']} | {model} | **{delta}s** | {t_in} | {t_out} | {calls} |")
            prev = t
        timing_container.markdown(
            "---\n#### ⏱️ Execution Summary\n"
            "| Node | Model | Time | Tok In | Tok Out | API |\n|---|---|---|---|---|---|\n"
            + "\n".join(rows)
            + f"\n| **Total** | | **{total}s** | **{tel_summary.get('total_tokens_in', 0)}** | **{tel_summary.get('total_tokens_out', 0)}** | **{tel_summary.get('total_api_calls', 0)}** |"
        )
        if state_snapshots:
            md_parts = []
            for snap in state_snapshots:
                md_parts.append(f"**After `{snap['node']}`:**")
                md_parts.append(f"```json\n{json.dumps(snap['snapshot'], indent=2, default=str)}\n```")
            state_tab_placeholder.markdown("\n\n".join(md_parts))
        if tel_summary and tel_summary.get("per_node"):
            per = tel_summary["per_node"]
            t_md = (
                f"**Total Tokens:** {tel_summary.get('total_tokens', 0):,} &nbsp;&nbsp; "
                f"**API Calls:** {tel_summary.get('total_api_calls', 0)} &nbsp;&nbsp; "
                f"**Time:** {total}s\n\n---\n\n"
                "| Node | Tokens In | Tokens Out | API Calls |\n|---|---|---|---|\n"
            )
            for nn in per:
                t_md += f"| {nn} | {per[nn]['tokens_in']} | {per[nn]['tokens_out']} | {per[nn]['api_calls']} |\n"
            telemetry_placeholder.markdown(t_md)
        # Clean up session state
        st.session_state.pop("qc_paused", None)

    # ── Handle PASS ──────────────────────────────────────────────────
    if pass_clicked:
        completed_nodes.add("quality_checker")
        node_placeholders[5].markdown(
            '<div class="node-done">✅ <strong>Quality Check</strong> '
            '<span class="human-qc-badge">🧑‍⚖️ Human</span>'
            '<br><small>Human QC: PASS</small></div>',
            unsafe_allow_html=True)
        finalise_qc(
            max(node_times.values()) if node_times else 0,
            saved.get("pause_telemetry_summary", {}),
            "Human QC: PASS ✅",
        )

    # ── Handle REVISE ────────────────────────────────────────────────
    elif revise_clicked:
        logs.append("**…** — 🔄 Revising plan (human requested)")
        log_container.warning("\n\n".join(logs))

        def process_revision_event(data, node, status, elapsed, thinking, snapshot, tel):
            node_times[node] = elapsed
            completed_nodes.add(node)
            if tel:
                all_telemetry[node] = tel
            if snapshot:
                state_snapshots.append({"node": node, "snapshot": snapshot})
            try:
                idx = NODE_ORDER.index(node)
                next_active = NODE_ORDER[idx + 1] if idx + 1 < len(NODE_ORDER) else None
            except ValueError:
                next_active = None
            node_info = next((n for n in NODES if n["key"] == node), None)
            icon = node_info["icon"] if node_info else "🔧"
            logs.append(f"**{elapsed}s** — {icon} {status}")
            if thinking:
                label = node_info["label"] if node_info else node
                thinking_html.append(f"<strong>{icon} {label}</strong>")
                thinking_html.append(render_thinking_html(thinking))
            render_node_tracker(node_placeholders, next_active, completed_nodes)
            log_container.info("\n\n".join(logs))
            if thinking_html:
                thinking_container.markdown("".join(thinking_html), unsafe_allow_html=True)
            container = detail_containers.get(node)
            if container:
                if node == "planner":
                    with container.expander("🧠 **Planner Output (Kimi) — Revised**", expanded=True):
                        st.markdown(data.get("plan", ""))
                elif node == "executor":
                    with container.expander("⚡ **Executor Output (Mistral) — Revised**", expanded=True):
                        st.markdown(data.get("final_response", ""))

        try:
            with requests.post(
                f"{saved['api_url']}/plan/revise",
                json={
                    "user_id": saved["selected_user"],
                    "query": saved["query"],
                    "mode": s_mode_api,
                    "pipeline_state": saved["pipeline_state"],
                },
                stream=True, timeout=300,
            ) as rev_resp:
                rev_resp.raise_for_status()
                for rev_line in rev_resp.iter_lines(decode_unicode=True):
                    if not rev_line or not rev_line.startswith("data: "):
                        continue
                    rev_data = json.loads(rev_line[6:])
                    rev_node = rev_data.get("node", "")
                    if rev_node == "done":
                        completed_nodes.add("quality_checker")
                        node_placeholders[5].markdown(
                            '<div class="node-done">✅ <strong>Quality Check</strong> '
                            '<span class="human-qc-badge">🧑‍⚖️ Human</span>'
                            '<br><small>Human QC: PASS (after revision)</small></div>',
                            unsafe_allow_html=True)
                        finalise_qc(
                            max(node_times.values()) if node_times else 0,
                            rev_data.get("telemetry_summary", saved.get("pause_telemetry_summary", {})),
                            "Human QC: PASS (after revision) ✅",
                        )
                    elif rev_node == "error":
                        log_container.error(f"❌ Revision Error: {rev_data.get('status', '')}")
                        st.session_state.pop("qc_paused", None)
                    else:
                        process_revision_event(
                            rev_data, rev_node, rev_data.get("status", ""),
                            rev_data.get("elapsed", 0), rev_data.get("thinking", []),
                            rev_data.get("state_snapshot", {}), rev_data.get("telemetry", {}),
                        )
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend for revision.")
            st.session_state.pop("qc_paused", None)
        except Exception as e:
            st.error(f"❌ Revision error: {e}")
            st.session_state.pop("qc_paused", None)

    # ── Neither button clicked yet — show QC buttons ─────────────────
    else:
        with manual_qc_container.container():
            st.markdown('<div class="manual-qc-card">', unsafe_allow_html=True)
            st.markdown("#### 🧑‍⚖️ Human Quality Check")
            st.markdown("The Executor has produced a training schedule. Review it above, then decide:")
            st.info(
                "🎓 **How this works behind the scenes:**\n\n"
                "- **LangGraph**: Would use `interrupt_before('quality_checker')` for "
                "native pause/resume with a single persistent graph session\n"
                "- **LangChain / Python**: Requires manual state serialisation and "
                "two separate API calls (what we're simulating here)"
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.button("✅ PASS — Schedule looks good", use_container_width=True, key="qc_pass")
            with btn_col2:
                st.button("🔄 REVISE — Needs improvement", use_container_width=True, key="qc_revise")
            st.markdown('</div>', unsafe_allow_html=True)
