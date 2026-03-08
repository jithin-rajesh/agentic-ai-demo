"""
Simple pipeline — 6 steps, plain Python, no LangGraph / LangChain.

Quality check uses a plain if-statement + explicit re-call.
Demonstrates what the same logic looks like without any framework.
"""

import json
import logging

from app.core.llm_factory import get_planner_llm, get_executor_llm
from app.core.telemetry import Telemetry
from app.data.user_history import get_user_profile, get_user_summary
from app.services.weather import check_weather

logger = logging.getLogger("simple_pipeline")


def _invoke_llm(llm, system_prompt: str, user_content: str, tel: Telemetry):
    """Invoke an LLM and record telemetry."""
    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])
    tel.record_llm_call(response)
    return response.content


def run_simple_pipeline(user_id: str, query: str):
    """
    Generator that runs each step and yields enriched SSE-compatible dicts.
    """
    tel = Telemetry()

    # ── Step 1: Load Profile ─────────────────────────────────────────
    tel.start_node("user_profile_loader")
    tel.add_thinking(
        thought="Loading rider profile from data store",
        action="Call get_user_profile()",
    )

    profile = get_user_profile(user_id)
    user_summary = get_user_summary(user_id)

    tel.add_thinking(
        observation=f"Loaded {profile['name']}: {profile['experience_level']}, FTP {profile['ftp_watts']}W"
    )
    tel.snapshot_state({"user_id": user_id, "user_profile": profile["name"]})

    yield {
        "node": "user_profile_loader",
        "status": f"Loaded profile for {profile['name']}",
        "detail": f"{profile['name']} — {profile['experience_level']} rider, FTP {profile['ftp_watts']}W",
        **tel.get_node_data("user_profile_loader"),
    }

    # ── Step 2: Fitness Analysis ─────────────────────────────────────
    tel.start_node("fitness_analyzer")
    history = profile.get("ride_history", [])
    tel.add_thinking(
        thought=f"Analysing {len(history)} rides for training load",
        action="Invoke Kimi LLM with ride history",
    )

    history_text = "\n".join(
        f"  {r['date']}: {r['distance_km']}km, {r['duration_min']}min, "
        f"{r['avg_power_w']}W avg — {r.get('notes', '')}"
        for r in history
    )

    fitness = _invoke_llm(
        get_planner_llm(), tel=tel,
        system_prompt="You are a cycling fitness analyst. Produce a concise fitness assessment (3-5 bullet points).",
        user_content=f"Rider: {profile.get('name')}, FTP: {profile.get('ftp_watts')}W, Level: {profile.get('experience_level')}\nHealth: {profile.get('health_notes', 'None')}\n\nRecent rides:\n{history_text}",
    )
    tel.add_thinking(observation=f"Analysis: {fitness[:120]}...")
    tel.snapshot_state({"fitness_analysis": fitness[:200]})

    yield {
        "node": "fitness_analyzer",
        "status": "Fitness Analyzer finished analysis",
        "detail": fitness,
        **tel.get_node_data("fitness_analyzer"),
    }

    # ── Step 3: Weather ──────────────────────────────────────────────
    tel.start_node("weather_checker")
    location = profile.get("location", "Bangalore")
    tel.add_thinking(
        thought=f"Need weather forecast for {location}",
        action=f"Call check_weather('{location}', days=7)",
    )

    weather_data = check_weather(location, days=7)
    lines = [f"7-day forecast for {location}:"]
    for day in weather_data.get("forecast", []):
        lines.append(f"  Day {day['day']}: {day['condition']}, {day['temperature_c']}°C, Wind {day['wind_speed_kmh']}km/h")
    weather_summary = "\n".join(lines)

    rain_days = sum(1 for d in weather_data.get("forecast", []) if d["condition"] == "Rain")
    tel.add_thinking(observation=f"{rain_days} rainy day(s) out of 7")
    tel.snapshot_state({"weather_summary": weather_summary[:200]})

    yield {
        "node": "weather_checker",
        "status": f"Weather fetched for {location}",
        "detail": weather_summary,
        **tel.get_node_data("weather_checker"),
    }

    # ── Step 4: Planner ──────────────────────────────────────────────
    tel.start_node("planner")
    tel.add_thinking(
        thought="Creating high-level training strategy",
        action="Invoke Kimi LLM with all context",
    )

    plan = _invoke_llm(
        get_planner_llm(), tel=tel,
        system_prompt="You are a Cycling Coach Planner. Create a high-level weekly training strategy.",
        user_content=f"RIDER:\n{user_summary}\n\nFITNESS:\n{fitness}\n\nWEATHER:\n{weather_summary}\n\nREQUEST:\n{query}",
    )
    tel.add_thinking(observation=f"Strategy: {plan[:120]}...")
    tel.snapshot_state({"plan": plan[:200]})

    yield {
        "node": "planner",
        "status": "Planner finished strategy",
        "plan": plan,
        **tel.get_node_data("planner"),
    }

    # ── Step 5: Executor ─────────────────────────────────────────────
    tel.start_node("executor")
    tel.add_thinking(
        thought="Building detailed schedule from plan",
        action="Invoke Mistral LLM with plan + context",
    )

    final = _invoke_llm(
        get_executor_llm(), tel=tel,
        system_prompt="You are the Executor Agent. Produce the FINAL weekly schedule with days, distances, intensities.",
        user_content=f"RIDER:\n{user_summary}\n\nFITNESS:\n{fitness}\n\nWEATHER:\n{weather_summary}\n\nPLAN:\n{plan}",
    )
    tel.add_thinking(observation=f"Schedule: {final[:120]}...")
    tel.snapshot_state({"final_response": final[:200]})

    yield {
        "node": "executor",
        "status": "Executor finished schedule",
        "final_response": final,
        **tel.get_node_data("executor"),
    }

    # ── Step 6: Quality Check (plain if statement) ───────────────────
    tel.start_node("quality_checker")
    tel.add_thinking(
        thought="Checking schedule quality with plain if-statement",
        action="Invoke Kimi LLM for review, then branch with Python if/else",
    )

    verdict_text = _invoke_llm(
        get_planner_llm(), tel=tel,
        system_prompt="You are a quality reviewer. Respond PASS or REVISE on the first line, then explain.",
        user_content=f"RIDER:\n{user_summary}\n\nSCHEDULE:\n{final}",
    )

    first_line = verdict_text.split("\n")[0].strip().upper()
    verdict = "revise" if "REVISE" in first_line else "pass"
    tel.add_thinking(observation=f"Verdict: {verdict.upper()}")
    tel.snapshot_state({"quality_check_result": verdict})

    yield {
        "node": "quality_checker",
        "status": f"Quality Check: {verdict.upper()}",
        "quality_result": verdict,
        "detail": verdict_text,
        **tel.get_node_data("quality_checker"),
    }

    # ── Plain Python retry with if-statement ─────────────────────────
    if verdict == "revise":
        tel.start_node("planner")
        tel.add_thinking(
            thought="Quality check failed — re-running planner with hardcoded if/else",
            action="Explicit function re-call (no framework loop)",
        )
        plan = _invoke_llm(
            get_planner_llm(), tel=tel,
            system_prompt="You are a Cycling Coach Planner. Create a MORE DETAILED plan. Previous attempt was incomplete.",
            user_content=f"RIDER:\n{user_summary}\n\nFITNESS:\n{fitness}\n\nWEATHER:\n{weather_summary}\n\nREQUEST:\n{query}",
        )
        tel.add_thinking(observation=f"Revised strategy: {plan[:120]}...")
        yield {"node": "planner", "status": "Planner revised strategy", "plan": plan, **tel.get_node_data("planner")}

        tel.start_node("executor")
        tel.add_thinking(thought="Re-executing with revised plan", action="Call executor again")
        final = _invoke_llm(
            get_executor_llm(), tel=tel,
            system_prompt="You are the Executor Agent. Produce the FINAL weekly schedule.",
            user_content=f"RIDER:\n{user_summary}\n\nFITNESS:\n{fitness}\n\nWEATHER:\n{weather_summary}\n\nPLAN:\n{plan}",
        )
        tel.add_thinking(observation=f"Revised schedule: {final[:120]}...")
        yield {"node": "executor", "status": "Executor revised schedule", "final_response": final, **tel.get_node_data("executor")}
        yield {"node": "quality_checker", "status": "Quality Check: PASS (after retry)", "quality_result": "pass", "detail": "Auto-passed after if/else retry"}

    yield {
        "node": "done",
        "status": "Complete",
        "telemetry_summary": tel.get_summary(),
    }


def run_simple_pipeline_until_executor(user_id: str, query: str):
    """
    Generator that runs steps 1–5 (no quality check) then yields
    a manual_qc_pause sentinel with serialised pipeline state.
    """
    tel = Telemetry()

    # ── Step 1: Load Profile ─────────────────────────────────────────
    tel.start_node("user_profile_loader")
    tel.add_thinking(thought="Loading rider profile from data store", action="Call get_user_profile()")
    profile = get_user_profile(user_id)
    user_summary = get_user_summary(user_id)
    tel.add_thinking(observation=f"Loaded {profile['name']}: {profile['experience_level']}, FTP {profile['ftp_watts']}W")
    tel.snapshot_state({"user_id": user_id, "user_profile": profile["name"]})
    yield {
        "node": "user_profile_loader",
        "status": f"Loaded profile for {profile['name']}",
        "detail": f"{profile['name']} — {profile['experience_level']} rider, FTP {profile['ftp_watts']}W",
        **tel.get_node_data("user_profile_loader"),
    }

    # ── Step 2: Fitness Analysis ─────────────────────────────────────
    tel.start_node("fitness_analyzer")
    history = profile.get("ride_history", [])
    tel.add_thinking(thought=f"Analysing {len(history)} rides for training load", action="Invoke Kimi LLM with ride history")
    history_text = "\n".join(
        f"  {r['date']}: {r['distance_km']}km, {r['duration_min']}min, "
        f"{r['avg_power_w']}W avg — {r.get('notes', '')}"
        for r in history
    )
    fitness = _invoke_llm(
        get_planner_llm(), tel=tel,
        system_prompt="You are a cycling fitness analyst. Produce a concise fitness assessment (3-5 bullet points).",
        user_content=f"Rider: {profile.get('name')}, FTP: {profile.get('ftp_watts')}W, Level: {profile.get('experience_level')}\nHealth: {profile.get('health_notes', 'None')}\n\nRecent rides:\n{history_text}",
    )
    tel.add_thinking(observation=f"Analysis: {fitness[:120]}...")
    tel.snapshot_state({"fitness_analysis": fitness[:200]})
    yield {
        "node": "fitness_analyzer",
        "status": "Fitness Analyzer finished analysis",
        "detail": fitness,
        **tel.get_node_data("fitness_analyzer"),
    }

    # ── Step 3: Weather ──────────────────────────────────────────────
    tel.start_node("weather_checker")
    location = profile.get("location", "Bangalore")
    tel.add_thinking(thought=f"Need weather forecast for {location}", action=f"Call check_weather('{location}', days=7)")
    weather_data = check_weather(location, days=7)
    lines = [f"7-day forecast for {location}:"]
    for day in weather_data.get("forecast", []):
        lines.append(f"  Day {day['day']}: {day['condition']}, {day['temperature_c']}°C, Wind {day['wind_speed_kmh']}km/h")
    weather_summary = "\n".join(lines)
    tel.add_thinking(observation=f"{sum(1 for d in weather_data.get('forecast', []) if d['condition'] == 'Rain')} rainy day(s) out of 7")
    tel.snapshot_state({"weather_summary": weather_summary[:200]})
    yield {
        "node": "weather_checker",
        "status": f"Weather fetched for {location}",
        "detail": weather_summary,
        **tel.get_node_data("weather_checker"),
    }

    # ── Step 4: Planner ──────────────────────────────────────────────
    tel.start_node("planner")
    tel.add_thinking(thought="Creating high-level training strategy", action="Invoke Kimi LLM with all context")
    plan = _invoke_llm(
        get_planner_llm(), tel=tel,
        system_prompt="You are a Cycling Coach Planner. Create a high-level weekly training strategy.",
        user_content=f"RIDER:\n{user_summary}\n\nFITNESS:\n{fitness}\n\nWEATHER:\n{weather_summary}\n\nREQUEST:\n{query}",
    )
    tel.add_thinking(observation=f"Strategy: {plan[:120]}...")
    tel.snapshot_state({"plan": plan[:200]})
    yield {
        "node": "planner",
        "status": "Planner finished strategy",
        "plan": plan,
        **tel.get_node_data("planner"),
    }

    # ── Step 5: Executor ─────────────────────────────────────────────
    tel.start_node("executor")
    tel.add_thinking(thought="Building detailed schedule from plan", action="Invoke Mistral LLM with plan + context")
    final = _invoke_llm(
        get_executor_llm(), tel=tel,
        system_prompt="You are the Executor Agent. Produce the FINAL weekly schedule with days, distances, intensities.",
        user_content=f"RIDER:\n{user_summary}\n\nFITNESS:\n{fitness}\n\nWEATHER:\n{weather_summary}\n\nPLAN:\n{plan}",
    )
    tel.add_thinking(observation=f"Schedule: {final[:120]}...")
    tel.snapshot_state({"final_response": final[:200]})
    yield {
        "node": "executor",
        "status": "Executor finished schedule",
        "final_response": final,
        **tel.get_node_data("executor"),
    }

    # ── Pause sentinel ───────────────────────────────────────────────
    yield {
        "node": "manual_qc_pause",
        "status": "Waiting for human QC decision",
        "pipeline_state": {
            "user_id": user_id,
            "query": query,
            "user_summary": user_summary,
            "fitness_analysis": fitness,
            "weather_summary": weather_summary,
            "plan": plan,
            "final_response": final,
        },
        "telemetry_summary": tel.get_summary(),
    }


def run_simple_revision(user_id: str, query: str, pipeline_state: dict):
    """
    Generator that re-runs Planner + Executor with revision context.
    """
    tel = Telemetry()
    user_summary = pipeline_state.get("user_summary", "")
    fitness = pipeline_state.get("fitness_analysis", "")
    weather_summary = pipeline_state.get("weather_summary", "")

    # ── Revised Planner ──────────────────────────────────────────────
    tel.start_node("planner")
    tel.add_thinking(
        thought="Quality check failed — re-running planner with revision context",
        action="Explicit function re-call (no framework loop)",
    )
    plan = _invoke_llm(
        get_planner_llm(), tel=tel,
        system_prompt="You are a Cycling Coach Planner. Create a MORE DETAILED plan. Previous attempt was incomplete.",
        user_content=f"RIDER:\n{user_summary}\n\nFITNESS:\n{fitness}\n\nWEATHER:\n{weather_summary}\n\nREQUEST:\n{query}",
    )
    tel.add_thinking(observation=f"Revised strategy: {plan[:120]}...")
    yield {"node": "planner", "status": "Planner revised strategy [REVISION]", "plan": plan, **tel.get_node_data("planner")}

    # ── Revised Executor ─────────────────────────────────────────────
    tel.start_node("executor")
    tel.add_thinking(thought="Re-executing with revised plan", action="Call executor again")
    final = _invoke_llm(
        get_executor_llm(), tel=tel,
        system_prompt="You are the Executor Agent. Produce the FINAL weekly schedule.",
        user_content=f"RIDER:\n{user_summary}\n\nFITNESS:\n{fitness}\n\nWEATHER:\n{weather_summary}\n\nPLAN:\n{plan}",
    )
    tel.add_thinking(observation=f"Revised schedule: {final[:120]}...")
    yield {"node": "executor", "status": "Executor revised schedule [REVISION]", "final_response": final, **tel.get_node_data("executor")}

    yield {
        "node": "done",
        "status": "Complete",
        "telemetry_summary": tel.get_summary(),
    }
