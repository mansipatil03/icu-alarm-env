"""
inference.py — Baseline inference script for ICU Alarm Fatigue Reducer
Uses OpenAI-compatible client to run agent through all 3 task levels.

Required environment variables:
  API_KEY_TOKEN  - Your API key
  API_BASE_URL   - API base URL (e.g. https://api.openai.com/v1)
  MODEL_NAME     - Model to use (e.g. gpt-4o-mini)
  HF_SPACE       - Your HuggingFace space URL
  HF_TOKEN       - Your HuggingFace token
"""

import os
import json
import requests
from openai import OpenAI

API_KEY  = os.environ.get("API_KEY_TOKEN", "")
BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL    = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_SPACE = os.environ.get("HF_SPACE", "http://localhost:7860")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """You are an expert ICU monitoring AI assistant.
Classify each ICU alarm as REAL emergency or FALSE alarm.
Always respond in this exact JSON format:
{
  "classification": "real" or "false",
  "confidence": 0.0 to 1.0,
  "explanation": "your clinical reasoning here",
  "recommended_action": "what the nurse should do"
}"""


def call_env(endpoint, method="POST", data=None, params=None):
    url = f"{HF_SPACE.rstrip('/')}/{endpoint}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    if method == "POST":
        resp = requests.post(url, json=data or {}, params=params or {}, headers=headers, timeout=30)
    else:
        resp = requests.get(url, params=params or {}, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_agent(observation):
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
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if "real" in raw.lower():
            return {"classification": "real", "confidence": 0.5,
                    "explanation": raw, "recommended_action": "Notify physician"}
        return {"classification": "false", "confidence": 0.5,
                "explanation": raw, "recommended_action": "Continue monitoring"}


def run_episode(task_level, session_id):
    obs = call_env("reset", params={"task_level": task_level, "session_id": session_id})
    prompt = obs.get("prompt", "")

    agent_output = run_agent(obs)

    action = {
        "classification": agent_output.get("classification", "false"),
        "confidence": agent_output.get("confidence", 0.5),
        "explanation": agent_output.get("explanation", ""),
        "recommended_action": agent_output.get("recommended_action", "")
    }

    result = call_env("step", data=action, params={"session_id": session_id})
    score = result.get("reward", 0.0)

    return {
        "task_level": task_level,
        "prompt": prompt,
        "action": action,
        "score": score,
        "info": result.get("info", {})
    }


def main():
    # [START] required by competition
    print(json.dumps({
        "event": "START",
        "model": MODEL,
        "space": HF_SPACE,
        "tasks": ["easy", "medium", "hard"]
    }))

    results = []
    total_score = 0.0

    for level in ["easy", "medium", "hard"]:
        session_id = f"baseline_{level}"
        episode = run_episode(level, session_id)
        results.append(episode)
        total_score += episode["score"]

        # [STEP] required by competition
        print(json.dumps({
            "event": "STEP",
            "task_level": level,
            "prompt": episode["prompt"][:200],
            "action": episode["action"],
            "score": episode["score"],
            "result": episode["info"].get("result", "unknown")
        }))

    avg_score = round(total_score / 3, 3)

    # [END] required by competition
    print(json.dumps({
        "event": "END",
        "prompt": "ICU Alarm Fatigue Reducer — 3-level baseline evaluation",
        "steps": results,
        "score": avg_score
    }))

    return avg_score


if __name__ == "__main__":
    main()
