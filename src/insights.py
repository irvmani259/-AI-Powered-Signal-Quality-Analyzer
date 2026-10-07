"""
insights.py
-----------
IBM Granite integration via watsonx.ai for AI-generated root-cause analysis
and actionable recommendations.

Behaviour
---------
- Reads WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL from environment.
- When credentials are present  → calls the real watsonx.ai REST API.
- When credentials are absent   → returns a deterministic mock response so
  the application runs fully offline without any keys.

Response schema (returned by generate_insights)
------------------------------------------------
{
  "root_cause":        str,
  "confidence":        float  (0-1),
  "affected_nodes":    list[str],
  "time_window":       {"start": ISO str, "end": ISO str},
  "recommendations":   list[{"priority": int, "action": str, "rationale": str}],
  "plain_summary":     str,
  "source":            "granite" | "mock"
}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# watsonx.ai endpoint template
# ---------------------------------------------------------------------------
_DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"
_GENERATE_PATH = "/ml/v1/text/generation?version=2023-05-29"
_IAM_URL = "https://iam.cloud.ibm.com/identity/token"
_MODEL_ID = "ibm/granite-13b-instruct-v2"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_insights(
    anomaly_summary: list[dict[str, Any]],
    interference_events: list[dict[str, Any]],
    degradation_events: list[dict[str, Any]],
    node_health: pd.DataFrame,
    horizon_hours: int = 12,
) -> dict[str, Any]:
    """
    Generate AI insights from detection results.

    Parameters
    ----------
    anomaly_summary      : list of anomaly dicts from detection.DetectionResult.summary
    interference_events  : list of interference event dicts
    degradation_events   : list of degradation event dicts
    node_health          : per-node health DataFrame
    horizon_hours        : forecast horizon used (for context in prompt)

    Returns
    -------
    Structured insights dict (see module docstring for schema).
    """
    api_key = os.getenv("WATSONX_API_KEY", "").strip()
    project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
    base_url = os.getenv("WATSONX_URL", _DEFAULT_URL).strip()

    context = _build_context(
        anomaly_summary, interference_events, degradation_events, node_health
    )

    if api_key and project_id:
        logger.info("Calling watsonx.ai Granite for insights")
        try:
            return _call_granite(api_key, project_id, base_url, context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Granite API call failed (%s); falling back to mock.", exc)

    logger.info("Using mock insights (no credentials configured)")
    return _mock_insights(context)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(
    anomaly_summary: list[dict[str, Any]],
    interference_events: list[dict[str, Any]],
    degradation_events: list[dict[str, Any]],
    node_health: pd.DataFrame,
) -> dict[str, Any]:
    """Condense detection results into a compact context block for the prompt."""
    # Aggregate anomaly counts
    from collections import Counter
    severity_counts = Counter(a.get("severity", "Unknown") for a in anomaly_summary)
    affected_nodes = sorted({a["node_id"] for a in anomaly_summary})

    # Top-5 worst anomalies
    critical_anomalies = [
        a for a in anomaly_summary if a.get("severity") in ("Critical", "High")
    ][:5]

    # Node health summary
    health_summary: list[dict] = []
    if not node_health.empty:
        for _, row in node_health.iterrows():
            health_summary.append({
                "node": row["node_id"],
                "health": row["health_score"],
                "mean_snr": row["mean_snr"],
                "mean_ber": float(row["mean_ber"]),
                "mean_latency": row["mean_latency"],
            })

    # Time window
    if anomaly_summary:
        timestamps = [a["timestamp"] for a in anomaly_summary if "timestamp" in a]
        time_window = {
            "start": min(timestamps) if timestamps else "N/A",
            "end": max(timestamps) if timestamps else "N/A",
        }
    else:
        time_window = {"start": "N/A", "end": "N/A"}

    return {
        "severity_counts": dict(severity_counts),
        "total_anomalies": len(anomaly_summary),
        "affected_nodes": affected_nodes,
        "time_window": time_window,
        "critical_anomalies": critical_anomalies,
        "interference_events": interference_events[:3],
        "degradation_events": degradation_events[:3],
        "node_health": health_summary,
    }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(context: dict[str, Any]) -> str:
    ctx_json = json.dumps(context, indent=2, default=str)
    return f"""You are an expert telecommunications signal quality engineer and AI assistant.

Analyze the following signal quality monitoring data and provide structured insights.

SIGNAL QUALITY CONTEXT:
{ctx_json}

Based on this data, respond with a JSON object containing exactly these fields:
{{
  "root_cause": "<concise root cause explanation, 1-2 sentences>",
  "confidence": <float 0.0-1.0>,
  "affected_nodes": ["list", "of", "node", "ids"],
  "recommendations": [
    {{
      "priority": <1=highest>,
      "action": "<specific action>",
      "rationale": "<why this helps>"
    }}
  ],
  "plain_summary": "<engineer-friendly paragraph summary, 3-5 sentences>"
}}

Prioritize the most critical issues. Recommendations should be specific, actionable,
and ordered by urgency. Respond with ONLY the JSON object, no other text.
"""


# ---------------------------------------------------------------------------
# watsonx.ai API call
# ---------------------------------------------------------------------------

def _get_iam_token(api_key: str) -> str:
    """Exchange IBM Cloud API key for a short-lived IAM bearer token."""
    resp = requests.post(
        _IAM_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _call_granite(
    api_key: str,
    project_id: str,
    base_url: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    token = _get_iam_token(api_key)
    prompt = _build_prompt(context)

    payload = {
        "model_id": _MODEL_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 800,
            "min_new_tokens": 50,
            "stop_sequences": [],
            "repetition_penalty": 1.05,
        },
        "project_id": project_id,
    }

    url = base_url.rstrip("/") + _GENERATE_PATH
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    raw_text = resp.json()["results"][0]["generated_text"].strip()

    # Parse the JSON response from Granite
    try:
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = "\n".join(raw_text.split("\n")[1:])
            raw_text = raw_text.rsplit("```", 1)[0].strip()
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("Granite response was not valid JSON; wrapping as plain summary")
        parsed = {
            "root_cause": "Unable to parse structured response.",
            "confidence": 0.5,
            "affected_nodes": context.get("affected_nodes", []),
            "recommendations": [],
            "plain_summary": raw_text,
        }

    parsed["time_window"] = context.get("time_window", {})
    parsed["source"] = "granite"
    return parsed


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def _mock_insights(context: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic mock that generates plausible insights from the context
    without any external API call.
    """
    affected = context.get("affected_nodes", [])
    severity_counts = context.get("severity_counts", {})
    interference = context.get("interference_events", [])
    degradation = context.get("degradation_events", [])
    health = context.get("node_health", [])

    # Determine dominant issue
    has_interference = len(interference) > 0
    has_degradation = len(degradation) > 0
    critical_count = severity_counts.get("Critical", 0)
    high_count = severity_counts.get("High", 0)

    if critical_count > 0 and has_interference:
        root_cause = (
            "Critical signal degradation is likely caused by co-channel interference "
            "or external RF interference sources. Correlated SNR drops and BER spikes "
            "across multiple nodes suggest a shared interference source."
        )
        confidence = 0.82
    elif has_degradation and not has_interference:
        root_cause = (
            "Gradual signal quality degradation detected across monitored nodes. "
            "Sustained downward SNR trends combined with rising BER suggest physical "
            "layer issues such as connector corrosion, cable degradation, or antenna misalignment."
        )
        confidence = 0.74
    elif has_interference:
        root_cause = (
            "Periodic interference bursts detected. The regular pattern suggests "
            "a scheduled process, industrial equipment, or co-channel transmission "
            "from a neighboring cell site."
        )
        confidence = 0.78
    elif critical_count > 0 or high_count > 0:
        root_cause = (
            "Sudden signal quality drops indicate transient hardware faults, "
            "power supply instability, or brief environmental obstructions."
        )
        confidence = 0.65
    else:
        root_cause = (
            "Minor signal quality variations observed within acceptable operational bounds. "
            "No critical root cause identified at this time."
        )
        confidence = 0.55

    # Build prioritized recommendations
    recommendations: list[dict[str, Any]] = []
    priority = 1

    if critical_count > 0:
        recommendations.append({
            "priority": priority,
            "action": f"Immediately inspect hardware on nodes: {', '.join(affected[:3])}",
            "rationale": "Critical severity anomalies require urgent on-site verification.",
        })
        priority += 1

    if has_interference:
        recommendations.append({
            "priority": priority,
            "action": "Perform RF spectrum scan and identify interference sources near affected nodes",
            "rationale": "Periodic or correlated interference patterns indicate an external RF source.",
        })
        priority += 1
        recommendations.append({
            "priority": priority,
            "action": "Consider frequency reuse plan adjustment or channel reassignment",
            "rationale": "Co-channel interference can be mitigated by reassigning frequency bands.",
        })
        priority += 1

    if has_degradation:
        recommendations.append({
            "priority": priority,
            "action": "Inspect and clean antenna connectors, cables, and feeder lines on degrading nodes",
            "rationale": "Gradual SNR decline is a classic symptom of physical layer degradation.",
        })
        priority += 1
        recommendations.append({
            "priority": priority,
            "action": "Increase transmit power by 2-3 dB on nodes showing SNR below 15 dB",
            "rationale": "Compensates for path loss increase while root cause is being addressed.",
        })
        priority += 1

    # Low-health nodes
    low_health_nodes = [h["node"] for h in health if h.get("health", 100) < 60]
    if low_health_nodes:
        recommendations.append({
            "priority": priority,
            "action": f"Schedule preventive maintenance for low-health nodes: {', '.join(low_health_nodes[:3])}",
            "rationale": "Proactive maintenance prevents escalation to service-affecting failures.",
        })
        priority += 1

    recommendations.append({
        "priority": priority,
        "action": "Enable enhanced monitoring (1-minute polling) for affected nodes for the next 24 hours",
        "rationale": "Higher polling frequency will capture transient events and improve anomaly detection accuracy.",
    })

    # Plain summary
    node_list = ", ".join(affected[:5]) or "no nodes"
    total_anomalies = context.get("total_anomalies", 0)
    plain_summary = (
        f"Signal quality analysis identified {total_anomalies} anomalous measurements "
        f"across nodes: {node_list}. "
        f"{root_cause} "
        f"The analysis detected {len(interference)} interference event(s) and "
        f"{len(degradation)} degradation trend(s). "
        f"Immediate attention is recommended for the {len(recommendations)} prioritized actions above, "
        f"starting with the highest-priority items to minimize service impact."
    )

    return {
        "root_cause": root_cause,
        "confidence": confidence,
        "affected_nodes": affected,
        "time_window": context.get("time_window", {}),
        "recommendations": recommendations,
        "plain_summary": plain_summary,
        "source": "mock",
    }
