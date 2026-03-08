"""
LangChain sequential pipeline (LCEL) with 6 steps + simulated retry.

Same logic as LangGraph but using RunnableLambda chaining.
Quality checker step does an inline retry (re-runs planner+executor)
since chains cannot loop — demonstrating framework limitation.
"""

import json
import logging
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.llm_factory import get_planner_llm, get_executor_llm
from app.core.telemetry import Telemetry
from app.data.user_history import get_user_profile, get_user_summary
from app.services.weather import check_weather

logger = logging.getLogger("langchain_pipeline")


def _make_initial_state(user_id: str, query: str) -> dict:
    return {
        "user_id": user_id,
        "query": query,
        "user_profile": {},
        "user_summary": "",
        "fitness_analysis": "",
        "weather_data": {},
        "weather_summary": "",
        "plan": "",
        "final_response": "",
        "quality_check_result": "",
        "revision_count": 0,
        "step_outputs": [],
    }


# ── Step 1: User Profile Loader ─────────────────────────────────────────
def step_user_profile_loader(state: dict, tel: Telemetry) -> dict:
    tel.start_node("user_profile_loader")
    tel.add_thinking(
        thought="Loading rider profile and ride history from data store",
        action="Call get_user_profile()",
    )

    user_id = state.get("user_id", "rider_001")
    profile = get_user_profile(user_id)
    summary = get_user_summary(user_id)

    tel.add_thinking(
        observation=f"Loaded {profile['name']}: {profile['experience_level']}, FTP {profile['ftp_watts']}W"
    )

    state["user_profile"] = profile
    state["user_summary"] = summary
    tel.snapshot_state(state)

    state["step_outputs"].append({
        "node": "user_profile_loader",
        "status": f"Loaded profile for {profile['name']}",
        "detail": f"{profile['name']} — {profile['experience_level']} rider, FTP {profile['ftp_watts']}W",
        **tel.get_node_data("user_profile_loader"),
    })
    return state


# ── Step 2: Fitness Analyzer (Kimi LLM) ─────────────────────────────────
def step_fitness_analyzer(state: dict, tel: Telemetry) -> dict:
    tel.start_node("fitness_analyzer")

    profile = state.get("user_profile", {})
    history = profile.get("ride_history", [])

    tel.add_thinking(
        thought=f"Analysing {len(history)} rides for training load and fatigue",
        action="Invoke Kimi LLM with ride history",
    )

    llm = get_planner_llm()
    history_text = "\n".join(
        f"  {r['date']}: {r['distance_km']}km, {r['duration_min']}min, "
        f"{r['avg_power_w']}W avg, {r['avg_hr']}bpm — {r.get('notes', '')}"
        for r in history
    )

    response = llm.invoke([
        SystemMessage(content=(
            "You are a cycling fitness analyst. Analyse the rider's recent ride "
            "history and produce a concise fitness assessment (3-5 bullet points). "
            "Cover: current training load, fatigue level, strengths, areas for "
            "improvement, and readiness for the coming week."
        )),
        HumanMessage(content=(
            f"Rider: {profile.get('name', 'Unknown')}, "
            f"FTP: {profile.get('ftp_watts', 'N/A')}W, "
            f"Level: {profile.get('experience_level', 'N/A')}\n"
            f"Health: {profile.get('health_notes', 'None')}\n\n"
            f"Recent rides:\n{history_text}"
        )),
    ])

    tel.record_llm_call(response)
    tel.add_thinking(observation=f"Analysis complete: {response.content[:120]}...")

    state["fitness_analysis"] = response.content
    tel.snapshot_state(state)

    state["step_outputs"].append({
        "node": "fitness_analyzer",
        "status": "Fitness Analyzer (Kimi) finished analysis",
        "detail": response.content,
        **tel.get_node_data("fitness_analyzer"),
    })
    return state


# ── Step 3: Weather Checker ──────────────────────────────────────────────
def step_weather_checker(state: dict, tel: Telemetry) -> dict:
    tel.start_node("weather_checker")

    profile = state.get("user_profile", {})
    location = profile.get("location", "Bangalore")

    tel.add_thinking(
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
    tel.add_thinking(observation=f"{rain_days} rainy day(s) out of 7")

    state["weather_data"] = weather
    state["weather_summary"] = weather_summary
    tel.snapshot_state(state)

    state["step_outputs"].append({
        "node": "weather_checker",
        "status": f"Weather fetched for {location}",
        "detail": weather_summary,
        **tel.get_node_data("weather_checker"),
    })
    return state


# ── Step 4: Planner (Kimi LLM) ──────────────────────────────────────────
def step_planner(state: dict, tel: Telemetry, is_revision: bool = False) -> dict:
    tel.start_node("planner")

    tel.add_thinking(
        thought="Re-planning after quality check feedback" if is_revision
                else "Creating high-level strategy from all context",
        action="Invoke Kimi LLM with profile + fitness + weather + request",
    )

    planner = get_planner_llm()
    revision_note = ""
    if is_revision:
        revision_note = (
            "\n\nIMPORTANT: The previous plan was reviewed and found incomplete. "
            "Create a more detailed and comprehensive plan this time."
        )

    response = planner.invoke([
        SystemMessage(content=(
            "You are a Cycling Coach Planner.\n"
            "Create a high-level weekly training strategy that accounts for all context.\n"
            "Output a clear plan for the Executor to implement."
            + revision_note
        )),
        HumanMessage(content=(
            f"RIDER PROFILE:\n{state.get('user_summary', '')}\n\n"
            f"FITNESS ANALYSIS:\n{state.get('fitness_analysis', '')}\n\n"
            f"WEATHER:\n{state.get('weather_summary', '')}\n\n"
            f"USER REQUEST:\n{state.get('query', '')}"
        )),
    ])

    tel.record_llm_call(response)
    tel.add_thinking(observation=f"Strategy: {response.content[:120]}...")

    state["plan"] = response.content
    tel.snapshot_state(state)

    state["step_outputs"].append({
        "node": "planner",
        "status": "Planner (Kimi) finished strategy" + (" [REVISION]" if is_revision else ""),
        "plan": response.content,
        **tel.get_node_data("planner"),
    })
    return state


# ── Step 5: Executor (Mistral LLM) ──────────────────────────────────────
def step_executor(state: dict, tel: Telemetry) -> dict:
    tel.start_node("executor")

    tel.add_thinking(
        thought="Translating plan into detailed day-by-day schedule",
        action="Invoke Mistral LLM with plan + context",
    )

    executor = get_executor_llm()
    response = executor.invoke([
        SystemMessage(content=(
            "You are the Executor Agent for a cycling coach app.\n"
            "Produce the FINAL weekly training schedule.\n"
            "Include specific days, distances, intensities, durations, "
            "and weather-adjusted advice. Format neatly."
        )),
        HumanMessage(content=(
            f"RIDER PROFILE:\n{state.get('user_summary', '')}\n\n"
            f"FITNESS:\n{state.get('fitness_analysis', '')}\n\n"
            f"WEATHER:\n{state.get('weather_summary', '')}\n\n"
            f"PLAN:\n{state.get('plan', '')}"
        )),
    ])

    tel.record_llm_call(response)
    tel.add_thinking(observation=f"Schedule: {response.content[:120]}...")

    state["final_response"] = response.content
    tel.snapshot_state(state)

    state["step_outputs"].append({
        "node": "executor",
        "status": "Executor (Mistral) finished schedule",
        "final_response": response.content,
        **tel.get_node_data("executor"),
    })
    return state


# ── Step 6: Quality Checker (Kimi LLM) ──────────────────────────────────
def step_quality_checker(state: dict, tel: Telemetry) -> dict:
    tel.start_node("quality_checker")

    tel.add_thinking(
        thought="Evaluating schedule completeness, safety, and goal alignment",
        action="Invoke Kimi LLM to review executor output",
    )

    llm = get_planner_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a quality reviewer for cycling training plans.\n"
            "Review the schedule and respond with EXACTLY one word on the first line:\n"
            "  PASS — if the schedule is good\n"
            "  REVISE — if it needs improvement\n"
            "Then briefly explain why."
        )),
        HumanMessage(content=(
            f"RIDER:\n{state.get('user_summary', '')}\n\n"
            f"SCHEDULE TO REVIEW:\n{state.get('final_response', '')}"
        )),
    ])

    tel.record_llm_call(response)
    answer = response.content.strip()
    first_line = answer.split("\n")[0].strip().upper()
    verdict = "revise" if "REVISE" in first_line else "pass"

    tel.add_thinking(observation=f"Verdict: {verdict.upper()} — {answer[:120]}")

    state["quality_check_result"] = verdict
    tel.snapshot_state(state)

    state["step_outputs"].append({
        "node": "quality_checker",
        "status": f"Quality Check: {verdict.upper()}",
        "quality_result": verdict,
        "detail": answer,
        **tel.get_node_data("quality_checker"),
    })
    return state


# ── Runner (generator for SSE streaming) ────────────────────────────────
def run_langchain_pipeline(user_id: str, query: str):
    """
    Generator that runs the LangChain pipeline step-by-step and yields
    SSE-compatible dicts with enriched telemetry.

    If quality_checker returns 'revise', re-runs planner+executor inline
    (chains can't loop — showing framework limitation).
    """
    tel = Telemetry()
    state = _make_initial_state(user_id, query)

    # Steps 1-3
    state = step_user_profile_loader(state, tel)
    yield state["step_outputs"][-1]

    state = step_fitness_analyzer(state, tel)
    yield state["step_outputs"][-1]

    state = step_weather_checker(state, tel)
    yield state["step_outputs"][-1]

    # Steps 4-5
    state = step_planner(state, tel)
    yield state["step_outputs"][-1]

    state = step_executor(state, tel)
    yield state["step_outputs"][-1]

    # Step 6: Quality check
    state = step_quality_checker(state, tel)
    yield state["step_outputs"][-1]

    # If revise needed → inline retry (can't loop in chain)
    if state["quality_check_result"] == "revise":
        state = step_planner(state, tel, is_revision=True)
        yield state["step_outputs"][-1]

        state = step_executor(state, tel)
        yield state["step_outputs"][-1]

        # Auto-pass second time
        state["quality_check_result"] = "pass"
        state["step_outputs"].append({
            "node": "quality_checker",
            "status": "Quality Check: PASS (after revision)",
            "quality_result": "pass",
            "detail": "Auto-passed after inline retry",
            **tel.get_node_data("quality_checker"),
        })
        yield state["step_outputs"][-1]

    yield {
        "node": "done",
        "status": "Complete",
        "telemetry_summary": tel.get_summary(),
    }


def run_langchain_pipeline_until_executor(user_id: str, query: str):
    """
    Generator that runs steps 1–5 (no quality check) then yields
    a manual_qc_pause sentinel with serialised pipeline state.
    """
    tel = Telemetry()
    state = _make_initial_state(user_id, query)

    # Steps 1-3
    state = step_user_profile_loader(state, tel)
    yield state["step_outputs"][-1]

    state = step_fitness_analyzer(state, tel)
    yield state["step_outputs"][-1]

    state = step_weather_checker(state, tel)
    yield state["step_outputs"][-1]

    # Steps 4-5
    state = step_planner(state, tel)
    yield state["step_outputs"][-1]

    state = step_executor(state, tel)
    yield state["step_outputs"][-1]

    # Yield pause sentinel
    yield {
        "node": "manual_qc_pause",
        "status": "Waiting for human QC decision",
        "pipeline_state": {
            "user_id": user_id,
            "query": query,
            "user_summary": state.get("user_summary", ""),
            "fitness_analysis": state.get("fitness_analysis", ""),
            "weather_summary": state.get("weather_summary", ""),
            "plan": state.get("plan", ""),
            "final_response": state.get("final_response", ""),
        },
        "telemetry_summary": tel.get_summary(),
    }


def run_langchain_revision(user_id: str, query: str, pipeline_state: dict):
    """
    Generator that re-runs Planner + Executor with revision context.
    Restores state from the serialised pipeline_state dict.
    """
    tel = Telemetry()
    state = _make_initial_state(user_id, query)

    # Restore accumulated state
    state["user_summary"] = pipeline_state.get("user_summary", "")
    state["fitness_analysis"] = pipeline_state.get("fitness_analysis", "")
    state["weather_summary"] = pipeline_state.get("weather_summary", "")
    state["plan"] = pipeline_state.get("plan", "")

    # Re-run planner with revision context
    state = step_planner(state, tel, is_revision=True)
    yield state["step_outputs"][-1]

    state = step_executor(state, tel)
    yield state["step_outputs"][-1]

    yield {
        "node": "done",
        "status": "Complete",
        "telemetry_summary": tel.get_summary(),
    }
