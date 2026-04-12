import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Action, Observation, StepResult, EnvState
from env import ICUAlarmEnv
import uvicorn

main = FastAPI(
    title="ICU Alarm Fatigue Reducer — OpenEnv",
    version="1.0.0"
)

main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_envs = {}

@main.get("/health")
def health():
    return {"status": "ok"}

@main.post("/reset")
def reset(session_id: str = "default", task_level: str = "easy"):
    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(400, "task_level must be easy, medium, or hard")
    _envs[session_id] = ICUAlarmEnv(task_level=task_level)
    return _envs[session_id].reset()

@main.post("/step")
def step(action: Action, session_id: str = "default"):
    if session_id not in _envs:
        raise HTTPException(400, "Call /reset first")
    try:
        return _envs[session_id].step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

@main.get("/state")
def state(session_id: str = "default"):
    if session_id not in _envs:
        raise HTTPException(400, "Call /reset first")
    return _envs[session_id].state()

if __name__ == "__main__":
    uvicorn.run("server.app:main", host="0.0.0.0", port=7860)
    
