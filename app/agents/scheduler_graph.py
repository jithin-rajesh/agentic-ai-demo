"""
Enhanced LangGraph pipeline with 6 nodes + conditional re-evaluation.

Graph:
  user_profile_loader → fitness_analyzer → weather_checker → planner → executor
      → quality_checker  ──(pass)──→ END
                         └─(revise)─→ planner  (max 1 revision)

Each node populates telemetry: thinking logs, token counts, state snapshots.
"""

import json
import logging
import operator
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.core.llm_factory import get_planner_llm, get_executor_llm
from app.core.telemetry import Telemetry
from app.data.user_history import get_user_profile, get_user_summary
from app.services.weather import check_weather

logger = logging.getLogger("scheduler_graph")

# Module-level telemetry — replaced per invocation via run_langgraph_pipeline()
_telemetry: Telemetry = Telemetry()


# ── State ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: str
    user_id: str
    user_profile: dict
    user_summary: str
    fitness_analysis: str
    weather_data: dict
    weather_summary: str
    route_suggestions: str
    quality_check_result: str   # "pass" or "revise"
    revision_count: int


# ── Node 1: User Profile Loader ─────────────────────────────────────────
def user_profile_loader(state: AgentState) -> dict:
    _telemetry.start_node("user_profile_loader")
    _telemetry.add_thinking(
        thought="I need to load the rider's profile and ride history",
        action="Call get_user_profile() from data store",
    )

    user_id = state.get("user_id", "rider_001")
    profile = get_user_profile(user_id)
    summary = get_user_summary(user_id)

    _telemetry.add_thinking(
        observation=f"Loaded {profile['name']}: {profile['experience_level']}, FTP {profile['ftp_watts']}W, {len(profile.get('ride_history', []))} rides"
    )

    result = {
        "user_profile": profile,
        "user_summary": summary,
        "messages": [AIMessage(content=f"[Profile Loaded] {profile['name']} — {profile['experience_level']} rider, FTP {profile['ftp_watts']}W")],
    }
    _telemetry.snapshot_state({**state, **result})
    return result


# ── Node 2: Fitness Analyzer (LLM) ──────────────────────────────────────
def fitness_analyzer(state: AgentState) -> dict:
    _telemetry.start_node("fitness_analyzer")

    profile = state.get("user_profile", {})
    history = profile.get("ride_history", [])

    _telemetry.add_thinking(
        thought=f"Analysing {len(history)} recent rides for training load and fatigue",
        action="Invoke Kimi LLM with ride history data",
    )

    llm = get_planner_llm()
    history_text = "\n".join(
        f"  {r['date']}: {r['distance_km']}km, {r['duration_min']}min, "
        f"{r['avg_power_w']}W avg, {r['avg_hr']}bpm, {r['elevation_m']}m elev — {r.get('notes', '')}"
        for r in history
    )

    system_msg = SystemMessage(content=(
        "You are a cycling fitness analyst. Analyse the rider's recent ride "
        "history and produce a concise fitness assessment (3-5 bullet points). "
        "Cover: current training load (low/moderate/high), fatigue level, "
        "strengths, areas for improvement, and readiness for the coming week. "
        "Be specific with numbers where possible."
    ))
    user_msg = HumanMessage(content=(
        f"Rider: {profile.get('name', 'Unknown')}, "
        f"FTP: {profile.get('ftp_watts', 'N/A')}W, "
        f"Weight: {profile.get('weight_kg', 'N/A')}kg, "
        f"Level: {profile.get('experience_level', 'N/A')}\n"
        f"Weekly target: {profile.get('weekly_volume_target_km', 'N/A')}km\n"
        f"Health: {profile.get('health_notes', 'None')}\n\n"
        f"Recent rides:\n{history_text}"
    ))

    response = llm.invoke([system_msg, user_msg])
    _telemetry.record_llm_call(response)
    _telemetry.add_thinking(observation=f"Analysis complete: {response.content[:120]}...")

    result = {
        "fitness_analysis": response.content,
        "messages": [AIMessage(content=f"[Fitness Analysis]\n{response.content}")],
    }
    _telemetry.snapshot_state({**state, **result})
    return result


# ── Node 3: Weather Checker ─────────────────────────────────────────────
def weather_checker(state: AgentState) -> dict:
    _telemetry.start_node("weather_checker")

    profile = state.get("user_profile", {})
    location = profile.get("location", "Bangalore")

    _telemetry.add_thinking(
        thought=f"Need 7-day weather forecast for {location}",
        action=f"Call check_weather('{location}', days=7)",
    )

    weather = check_weather(location, days=7)
    lines = [f"7-day forecast for {location}:"]
    for day in weather.get("forecast", []):
        lines.append(
            f"  Day {day['day']}: {day['condition']}, "
            f"{day['temperature_c']}°C, Wind {day['wind_speed_kmh']}km/h"
        )
    weather_summary = "\n".join(lines)

    rain_days = sum(1 for d in weather.get("forecast", []) if d["condition"] == "Rain")
    _telemetry.add_thinking(
        observation=f"Forecast retrieved: {rain_days} rainy day(s) out of 7"
    )

    result = {
        "weather_data": weather,
        "weather_summary": weather_summary,
        "messages": [AIMessage(content=f"[Weather]\n{weather_summary}")],
    }
    _telemetry.snapshot_state({**state, **result})
    return result


# ── Node 4: Planner (Kimi LLM) ──────────────────────────────────────────
def planner_node(state: AgentState) -> dict:
    _telemetry.start_node("planner")

    revision = state.get("revision_count", 0)
    is_revision = revision > 0

    _telemetry.add_thinking(
        thought="Re-planning after quality check feedback" if is_revision
                else "Creating high-level strategy from all context",
        action="Invoke Kimi LLM with profile + fitness + weather + user request",
    )

    planner = get_planner_llm()
    user_summary = state.get("user_summary", "")
    fitness = state.get("fitness_analysis", "")
    weather = state.get("weather_summary", "")

    revision_note = ""
    if is_revision:
        revision_note = (
            "\n\nIMPORTANT: The previous plan was reviewed and found incomplete. "
            "Please create a more detailed and comprehensive plan this time."
        )

    system_msg = SystemMessage(content=(
        "You are a Cycling Coach Planner.\n"
        "You have the rider's profile, fitness analysis, and weather forecast.\n"
        "Create a high-level weekly training strategy that:\n"
        "1. Respects the rider's fitness level and fatigue\n"
        "2. Accounts for weather conditions\n"
        "3. Aligns with their goals and preferred days\n"
        "4. Specifies approximate distances, intensities, and workout types per day\n"
        "Output a clear plan for the Executor to implement as a detailed schedule."
        + revision_note
    ))

    context = (
        f"RIDER PROFILE:\n{user_summary}\n\n"
        f"FITNESS ANALYSIS:\n{fitness}\n\n"
        f"WEATHER FORECAST:\n{weather}\n\n"
        f"USER REQUEST:\n{state['messages'][0].content}"
    )

    response = planner.invoke([system_msg, HumanMessage(content=context)])
    _telemetry.record_llm_call(response)
    _telemetry.add_thinking(observation=f"Strategy created: {response.content[:120]}...")

    result = {"plan": response.content, "messages": [response]}
    _telemetry.snapshot_state({**state, **result})
    return result


# ── Node 5: Executor (Mistral LLM) ──────────────────────────────────────
def executor_node(state: AgentState) -> dict:
    _telemetry.start_node("executor")

    _telemetry.add_thinking(
        thought="Translating high-level plan into detailed day-by-day schedule",
        action="Invoke Mistral LLM with plan + all context",
    )

    executor = get_executor_llm()
    plan = state.get("plan", "")
    user_summary = state.get("user_summary", "")
    weather = state.get("weather_summary", "")
    fitness = state.get("fitness_analysis", "")

    system_msg = SystemMessage(content=(
        "You are the Executor Agent for a cycling coach app.\n"
        "You have received a high-level plan, rider context, and weather data.\n"
        "Produce the FINAL weekly training schedule in a clear, user-friendly format.\n"
        "Include:\n"
        "- Specific days with workout names\n"
        "- Distances, intensities (easy/moderate/hard/threshold)\n"
        "- Duration and target power zones if applicable\n"
        "- Rest day recommendations\n"
        "- Weather-adjusted advice (indoor alternatives if bad weather)\n"
        "Format neatly with headers and bullet points."
    ))

    user_content = (
        f"RIDER PROFILE:\n{user_summary}\n\n"
        f"FITNESS ANALYSIS:\n{fitness}\n\n"
        f"WEATHER:\n{weather}\n\n"
        f"PLAN FROM PLANNER:\n{plan}"
    )

    response = executor.invoke([system_msg, HumanMessage(content=user_content)])
    _telemetry.record_llm_call(response)
    _telemetry.add_thinking(observation=f"Schedule produced: {response.content[:120]}...")

    result = {"messages": [response]}
    _telemetry.snapshot_state({**state, **result})
    return result


# ── Node 6: Quality Checker (Kimi LLM) ──────────────────────────────────
def quality_checker(state: AgentState) -> dict:
    _telemetry.start_node("quality_checker")

    revision_count = state.get("revision_count", 0)

    _telemetry.add_thinking(
        thought="Evaluating if the schedule is complete, safe, and aligned with rider goals",
        action="Invoke Kimi LLM to review executor output",
    )

    # If we've already revised once, auto-pass to prevent infinite loops
    if revision_count >= 1:
        _telemetry.add_thinking(
            observation="Already revised once — auto-passing to prevent loop"
        )
        result = {
            "quality_check_result": "pass",
            "messages": [AIMessage(content="[Quality Check] PASS (max revisions reached)")],
        }
        _telemetry.snapshot_state({**state, **result})
        return result

    llm = get_planner_llm()
    schedule = state["messages"][-1].content if state.get("messages") else ""
    user_summary = state.get("user_summary", "")

    system_msg = SystemMessage(content=(
        "You are a quality reviewer for cycling training plans.\n"
        "Review the schedule below and determine if it is:\n"
        "1. Complete (covers all requested days)\n"
        "2. Safe (respects rider's health notes and fitness level)\n"
        "3. Aligned with the rider's goals\n\n"
        "Respond with EXACTLY one word on the first line:\n"
        "  PASS — if the schedule is good\n"
        "  REVISE — if the schedule needs improvement\n\n"
        "Then on subsequent lines, briefly explain why."
    ))

    user_content = (
        f"RIDER:\n{user_summary}\n\n"
        f"SCHEDULE TO REVIEW:\n{schedule}"
    )

    response = llm.invoke([system_msg, HumanMessage(content=user_content)])
    _telemetry.record_llm_call(response)

    answer = response.content.strip()
    first_line = answer.split("\n")[0].strip().upper()
    verdict = "revise" if "REVISE" in first_line else "pass"

    _telemetry.add_thinking(
        observation=f"Quality verdict: {verdict.upper()}. {answer[:150]}"
    )

    result = {
        "quality_check_result": verdict,
        "revision_count": revision_count + 1,
        "messages": [AIMessage(content=f"[Quality Check] {verdict.upper()}: {answer}")],
    }
    _telemetry.snapshot_state({**state, **result})
    return result


# ── Conditional edge: route based on quality check ──────────────────────
def should_revise(state: AgentState) -> str:
    if state.get("quality_check_result", "pass") == "revise":
        return "planner"
    return END


# ── Graph ────────────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("user_profile_loader", user_profile_loader)
workflow.add_node("fitness_analyzer", fitness_analyzer)
workflow.add_node("weather_checker", weather_checker)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("quality_checker", quality_checker)

workflow.set_entry_point("user_profile_loader")
workflow.add_edge("user_profile_loader", "fitness_analyzer")
workflow.add_edge("fitness_analyzer", "weather_checker")
workflow.add_edge("weather_checker", "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "quality_checker")
workflow.add_conditional_edges("quality_checker", should_revise)

compiled_graph = workflow.compile()


def run_langgraph_pipeline(user_id: str, query: str):
    """
    Generator that streams SSE-compatible dicts with enriched telemetry.
    """
    global _telemetry
    _telemetry = Telemetry()

    inputs = {
        "messages": [HumanMessage(content=query)],
        "plan": "",
        "user_id": user_id,
        "user_profile": {},
        "user_summary": "",
        "fitness_analysis": "",
        "weather_data": {},
        "weather_summary": "",
        "route_suggestions": "",
        "quality_check_result": "",
        "revision_count": 0,
    }

    for event in compiled_graph.stream(inputs):
        for node_name, node_output in event.items():
            telemetry_data = _telemetry.get_node_data(node_name)
            payload = {
                "node": node_name,
                "status": _get_status(node_name, node_output),
                **telemetry_data,
            }

            # Add content-specific fields
            if node_name == "user_profile_loader":
                msgs = node_output.get("messages", [])
                payload["detail"] = msgs[-1].content if msgs else ""

            elif node_name == "fitness_analyzer":
                payload["detail"] = node_output.get("fitness_analysis", "")

            elif node_name == "weather_checker":
                payload["detail"] = node_output.get("weather_summary", "")

            elif node_name == "planner":
                payload["plan"] = node_output.get("plan", "")

            elif node_name == "executor":
                msgs = node_output.get("messages", [])
                payload["final_response"] = msgs[-1].content if msgs else ""

            elif node_name == "quality_checker":
                payload["quality_result"] = node_output.get("quality_check_result", "")
                msgs = node_output.get("messages", [])
                payload["detail"] = msgs[-1].content if msgs else ""

            yield payload

    # Final summary
    yield {
        "node": "done",
        "status": "Complete",
        "telemetry_summary": _telemetry.get_summary(),
    }


def run_langgraph_pipeline_until_executor(user_id: str, query: str):
    """
    Generator that runs nodes 1–5 (Profile→Fitness→Weather→Planner→Executor)
    then yields a manual_qc_pause sentinel with serialised pipeline state.
    Used for Manual QC mode.
    """
    global _telemetry
    _telemetry = Telemetry()

    # Build a 5-node graph (no quality_checker)
    wf = StateGraph(AgentState)
    wf.add_node("user_profile_loader", user_profile_loader)
    wf.add_node("fitness_analyzer", fitness_analyzer)
    wf.add_node("weather_checker", weather_checker)
    wf.add_node("planner", planner_node)
    wf.add_node("executor", executor_node)
    wf.set_entry_point("user_profile_loader")
    wf.add_edge("user_profile_loader", "fitness_analyzer")
    wf.add_edge("fitness_analyzer", "weather_checker")
    wf.add_edge("weather_checker", "planner")
    wf.add_edge("planner", "executor")
    wf.add_edge("executor", END)
    short_graph = wf.compile()

    inputs = {
        "messages": [HumanMessage(content=query)],
        "plan": "",
        "user_id": user_id,
        "user_profile": {},
        "user_summary": "",
        "fitness_analysis": "",
        "weather_data": {},
        "weather_summary": "",
        "route_suggestions": "",
        "quality_check_result": "",
        "revision_count": 0,
    }

    accumulated_state = dict(inputs)

    for event in short_graph.stream(inputs):
        for node_name, node_output in event.items():
            accumulated_state.update(node_output)
            telemetry_data = _telemetry.get_node_data(node_name)
            payload = {
                "node": node_name,
                "status": _get_status(node_name, node_output),
                **telemetry_data,
            }

            if node_name == "user_profile_loader":
                msgs = node_output.get("messages", [])
                payload["detail"] = msgs[-1].content if msgs else ""
            elif node_name == "fitness_analyzer":
                payload["detail"] = node_output.get("fitness_analysis", "")
            elif node_name == "weather_checker":
                payload["detail"] = node_output.get("weather_summary", "")
            elif node_name == "planner":
                payload["plan"] = node_output.get("plan", "")
            elif node_name == "executor":
                msgs = node_output.get("messages", [])
                payload["final_response"] = msgs[-1].content if msgs else ""

            yield payload

    # Yield pause sentinel with serialised state for the frontend
    yield {
        "node": "manual_qc_pause",
        "status": "Waiting for human QC decision",
        "pipeline_state": {
            "user_id": user_id,
            "query": query,
            "user_summary": accumulated_state.get("user_summary", ""),
            "fitness_analysis": accumulated_state.get("fitness_analysis", ""),
            "weather_summary": accumulated_state.get("weather_summary", ""),
            "plan": accumulated_state.get("plan", ""),
            "final_response": accumulated_state.get("messages", [])[-1].content
                if accumulated_state.get("messages") else "",
        },
        "telemetry_summary": _telemetry.get_summary(),
    }


def run_langgraph_revision(user_id: str, query: str, pipeline_state: dict):
    """
    Generator that re-runs Planner→Executor with revision context.
    Used after a human REVISE decision in Manual QC mode.
    """
    global _telemetry
    _telemetry = Telemetry()

    # Rebuild a 2-node graph: planner → executor
    wf = StateGraph(AgentState)
    wf.add_node("planner", planner_node)
    wf.add_node("executor", executor_node)
    wf.set_entry_point("planner")
    wf.add_edge("planner", "executor")
    wf.add_edge("executor", END)
    revision_graph = wf.compile()

    inputs = {
        "messages": [HumanMessage(content=query)],
        "plan": pipeline_state.get("plan", ""),
        "user_id": user_id,
        "user_profile": {},
        "user_summary": pipeline_state.get("user_summary", ""),
        "fitness_analysis": pipeline_state.get("fitness_analysis", ""),
        "weather_data": {},
        "weather_summary": pipeline_state.get("weather_summary", ""),
        "route_suggestions": "",
        "quality_check_result": "",
        "revision_count": 1,  # Triggers is_revision logic in planner_node
    }

    for event in revision_graph.stream(inputs):
        for node_name, node_output in event.items():
            telemetry_data = _telemetry.get_node_data(node_name)
            payload = {
                "node": node_name,
                "status": _get_status(node_name, node_output),
                **telemetry_data,
            }
            if node_name == "planner":
                payload["plan"] = node_output.get("plan", "")
                payload["status"] = "Planner (Kimi) revised strategy [REVISION]"
            elif node_name == "executor":
                msgs = node_output.get("messages", [])
                payload["final_response"] = msgs[-1].content if msgs else ""
                payload["status"] = "Executor (Mistral) revised schedule [REVISION]"
            yield payload

    yield {
        "node": "done",
        "status": "Complete",
        "telemetry_summary": _telemetry.get_summary(),
    }


def _get_status(node_name: str, node_output: dict) -> str:
    statuses = {
        "user_profile_loader": "User profile loaded",
        "fitness_analyzer": "Fitness Analyzer (Kimi) finished",
        "weather_checker": "Weather data fetched",
        "planner": "Planner (Kimi) finished strategy",
        "executor": "Executor (Mistral) finished schedule",
        "quality_checker": f"Quality Check: {node_output.get('quality_check_result', 'pass').upper()}",
    }
    return statuses.get(node_name, f"{node_name} completed")
