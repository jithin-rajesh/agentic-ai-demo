"""
Telemetry tracker for pipeline execution.

Collects per-node metrics: token counts, API calls, thinking logs,
and state snapshots. Each pipeline instantiates one Telemetry object
and calls its methods as nodes execute.
"""

import copy
import json
from dataclasses import dataclass, field


@dataclass
class NodeTelemetry:
    """Metrics for a single pipeline node."""
    node: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    api_calls: int = 0
    thinking: list = field(default_factory=list)
    state_snapshot: dict = field(default_factory=dict)


class Telemetry:
    """
    Accumulates telemetry across all nodes in a pipeline run.
    Thread-safe enough for single-request use.
    """

    def __init__(self):
        self.nodes: dict[str, NodeTelemetry] = {}
        self._current_node: str = ""

    def start_node(self, node_name: str):
        self._current_node = node_name
        self.nodes[node_name] = NodeTelemetry(node=node_name)

    def add_thinking(self, thought: str = "", action: str = "", observation: str = ""):
        """Record a Thought → Action → Observation step."""
        nt = self.nodes.get(self._current_node)
        if not nt:
            return
        entry = {}
        if thought:
            entry["thought"] = thought
        if action:
            entry["action"] = action
        if observation:
            entry["observation"] = observation
        nt.thinking.append(entry)

    def record_llm_call(self, response):
        """
        Extract token usage from a LangChain LLM response.
        Works with ChatNVIDIA and ChatMistralAI responses.
        """
        nt = self.nodes.get(self._current_node)
        if not nt:
            return
        nt.api_calls += 1

        # Try to extract token usage from response metadata
        usage = {}
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata or {}
            usage = meta.get("token_usage", meta.get("usage", {}))
        elif hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "input_tokens", 0),
                "completion_tokens": getattr(um, "output_tokens", 0),
            }

        nt.tokens_in += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        nt.tokens_out += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

    # Keys to exclude from snapshots (cause circular refs or are too large)
    _SKIP_KEYS = {"step_outputs", "messages", "ride_history", "forecast"}

    def snapshot_state(self, state: dict):
        """Save a serializable copy of the current state."""
        nt = self.nodes.get(self._current_node)
        if not nt:
            return

        safe = {}
        for k, v in state.items():
            if k in self._SKIP_KEYS:
                continue
            try:
                # Quick circular-ref check: try serialisation
                serialised = json.dumps(v, default=str)
                # Only keep reasonably sized values
                if len(serialised) < 2000:
                    safe[k] = json.loads(serialised)
                else:
                    safe[k] = str(v)[:500] + "…"
            except (TypeError, ValueError, RecursionError):
                safe[k] = str(v)[:500]
        nt.state_snapshot = safe

    def get_node_data(self, node_name: str) -> dict:
        """Return telemetry dict for a node, suitable for SSE payload."""
        nt = self.nodes.get(node_name)
        if not nt:
            return {}
        return {
            "thinking": nt.thinking,
            "telemetry": {
                "tokens_in": nt.tokens_in,
                "tokens_out": nt.tokens_out,
                "api_calls": nt.api_calls,
            },
            "state_snapshot": nt.state_snapshot,
        }

    def get_summary(self) -> dict:
        """Return aggregate telemetry across all nodes."""
        total_in = sum(n.tokens_in for n in self.nodes.values())
        total_out = sum(n.tokens_out for n in self.nodes.values())
        total_calls = sum(n.api_calls for n in self.nodes.values())
        return {
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_tokens": total_in + total_out,
            "total_api_calls": total_calls,
            "per_node": {
                name: {
                    "tokens_in": nt.tokens_in,
                    "tokens_out": nt.tokens_out,
                    "api_calls": nt.api_calls,
                }
                for name, nt in self.nodes.items()
            },
        }
