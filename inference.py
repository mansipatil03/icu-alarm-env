"""
inference.py — Corrected Multi-Level Evaluation Script
"""

import os
import json
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────
API_KEY   = os.environ.get("API_KEY_TOKEN", "")
BASE_URL  = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL     = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_SPACE  = os.environ.get("HF_SPACE", "http://localhost:7860")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")

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
    
    # Use json= for Body data and params= for URL queries
    if method == "POST":
        resp = requests.post(url, json=data, params=params, headers=headers, timeout=30)
    else:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
    
    resp.raise_for_status()
    return resp.json()

def run_agent(observation):
    # Pass the full observation JSON so the agent sees vitals and history
    obs_text = json.dumps(observation)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text}
        ],
        response_format={"type": "json_object"}, # Forces JSON output
        temperature=0.1
    )
    
    return json.loads(response.choices[0].message.content)

def run_episode(task_level, session_id):
    # 1. Reset
    obs = call_env("reset", method="POST", params={"task_level": task_level, "session_id": session_id})
    
    # 2. Agent Inference
    agent_output = run_agent(obs)

    # 3. Step
    action = {
        "classification": agent_output.get("classification", "false"),
        "confidence": agent_output.get("confidence", 0.5),
        "explanation": agent_output.get("explanation", ""),
        "recommended_action": agent_output.get("recommended_action", "")
    }

    result = call_env("step", method="POST", data=action, params={"session_id": session_id})
    
    return {
        "task_level": task_level,
        "prompt": obs.get("prompt", ""),
        "action": action,
        "score": result.get("reward", 0.0),
        "info": result.get("info", {})
    }

def main():
    # [START] Required logging
    print(json.dumps({
        "event": "START",
        "model": MODEL,
        "tasks": ["easy", "medium", "hard"]
    }))

    results = []
    total_score = 0.0

    # The Loop: Ensure we hit all three levels
    for level in ["easy", "medium", "hard"]:
        try:
            session_id = f"eval_{level}"
            episode = run_episode(level, session_id)
            results.append(episode)
            total_score += episode["score"]

            # [STEP] Required logging
            print(json.dumps({
                "event": "STEP",
                "task_level": level,
                "score": episode["score"],
                "result": episode["info"].get("result", "unknown")
            }))
        except Exception as e:
            print(f"Error on {level}: {str(e)}")

    avg_score = round(total_score / 3, 3)

    # [END] Required logging
    print(json.dumps({
        "event": "END",
        "score": avg_score,
        "details": results
    }))

if __name__ == "__main__":
    main()
