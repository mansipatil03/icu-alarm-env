import sys
import os
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from models import Action, StepResult, EnvState
from env import ICUAlarmEnv
import uvicorn

app = FastAPI(
    title="ICU Alarm Fatigue Reducer — OpenEnv",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_envs: Dict[str, ICUAlarmEnv] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(body: dict = Body(...)):
    session_id = body.get("session_id", "default")
    task_level = body.get("task_level", "easy")

    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(400, "task_level must be easy, medium, or hard")

    env = ICUAlarmEnv(task_level=task_level)
    _envs[session_id] = env

    return env.reset()


@app.post("/step")
def step(action: Action, session_id: str = "default") -> StepResult:
    if session_id not in _envs:
        raise HTTPException(400, "Call /reset first")

    try:
        return _envs[session_id].step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/state")
def state(session_id: str = "default") -> EnvState:
    if session_id not in _envs:
        raise HTTPException(400, "Call /reset first")

    return _envs[session_id].state()


# ✅ REQUIRED by OpenEnv
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


# ✅ REQUIRED by OpenEnv
if __name__ == "__main__":
    main()
