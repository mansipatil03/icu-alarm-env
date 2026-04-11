"""
inference.py — Baseline inference script for ICU Alarm Fatigue Reducer
Uses OpenAI-compatible client to run agent through all 3 task levels.

Required environment variables:
  API_KEY_TOKEN     - Your API key
  API_BASE_URL      - API base URL (e.g. https://api.openai.com/v1)
  MODEL_NAME        - Model to use (e.g. gpt-4o-mini)
  HF_SPACE          - Your HuggingFace space URL
  HF_TOKEN          - Your HuggingFace token
"""

import os
import json
import requests
from openai import OpenAI

# ── Config from environment variables ──────────────────────────────────
API_KEY   = os.environ.get("API_KEY_TOKEN", "")
BASE_URL  = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL     = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_SPACE  = os.environ.get("HF_SPACE", "http://localhost:7860")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """You are an expert ICU monitoring AI assistant.
Your job is to analyze patient vital signs and alarm history, then classify each alarm as either a REAL emergency or a FALSE alarm.

Always respond in this exact JSON format:
{
  "classification": "real" or "false",
  "confidence": 0.0 to 1.0,
  "explanation": "your clinical reasoning here",
  "recommended_action": "what the nurse should do"
}

Be medically accurate. Missing a real emergency is the worst possible outcome."""


def call_env(endpoint: str, method: str = "POST", data: dict = None, params: dict = None):
    """Helper to call the HuggingFace space API."""
    url = f"{HF_SPACE.rstrip('/')}/{endpoint}"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    if method == "POST":
        resp = requests.post(url, json=data or {}, params=params or {}, headers=headers, timeout=30)
    else:
        resp = requests.get(url, params=params or {}, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_agent(observation: dict) -> dict:
    """Run the LLM agent on a given observation."""
    prompt = observation.get("prompt", "")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if model doesn't return valid JSON
        if "real" in raw.lower():
            return {"classification": "real", "confidence": 0.5, "explanation": raw, "recommended_action": "Notify physician"}
        return {"classification": "false", "confidence": 0.5, "explanation": raw, "recommended_action": "Continue monitoring"}


def run_episode(task_level: str, session_id: str) -> dict:
    """Run one full episode for a given task level."""
    print(f"\n{'='*50}")
    print(f"  Running task: {task_level.upper()}")
    print(f"{'='*50}")

    # Reset
    obs = call_env("reset", params={"task_level": task_level, "session_id": session_id})
    print(f"  Patient: {obs.get('patient_id', 'unknown')}")
    print(f"  Prompt preview: {obs.get('prompt', '')[:120]}...")

    # Agent action
    agent_output = run_agent(obs)
    print(f"  Agent classification: {agent_output.get('classification')}")
    print(f"  Agent confidence: {agent_output.get('confidence')}")

    # Step
    action = {
        "classification": agent_output.get("classification", "false"),
        "confidence": agent_output.get("confidence", 0.5),
        "explanation": agent_output.get("explanation", ""),
        "recommended_action": agent_output.get("recommended_action", "")
    }
    result = call_env("step", data=action, params={"session_id": session_id})
    score = result.get("reward", 0.0)
    info = result.get("info", {})

    print(f"  Score: {score}")
    print(f"  Result: {info.get('result', 'unknown')}")
    print(f"  Ground truth: {info.get('ground_truth', 'unknown')}")

    return {
        "task_level": task_level,
        "prompt": obs.get("prompt", ""),
        "steps": [action],
        "score": score,
        "info": info
    }


def main():
    print("\n🏥 ICU Alarm Fatigue Reducer — Baseline Inference")
    print(f"   Model: {MODEL}")
    print(f"   Space: {HF_SPACE}")

    results = []
    total_score = 0.0

    for i, level in enumerate(["easy", "medium", "hard"]):
        session_id = f"baseline_{level}"
        episode = run_episode(level, session_id)
        results.append(episode)
        total_score += episode["score"]

    avg_score = total_score / 3
    print(f"\n{'='*50}")
    print(f"  BASELINE RESULTS")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['task_level'].upper():8s} → score: {r['score']:.3f}  [{r['info'].get('result','?')}]")
    print(f"  {'AVERAGE':8s} → score: {avg_score:.3f}")
    print(f"{'='*50}\n")

    # Output in required format
    output = {
        "prompt": "ICU Alarm Fatigue Reducer — 3-level baseline evaluation",
        "steps": results,
        "score": round(avg_score, 3)
    }
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
