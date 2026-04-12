"""
FastAPI server exposing OpenEnv-compatible endpoints for ICU Alarm Fatigue Reducer
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Action, Observation, StepResult, EnvState
from env import ICUAlarmEnv, verify_score
import uvicorn

# The validator specifically looks for 'main'
main = FastAPI(
    title="ICU Alarm Fatigue Reducer — OpenEnv",
    description="An AI environment where agents learn to distinguish real ICU emergencies from false alarms.",
    version="1.0.0"
)

main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_envs: dict[str, ICUAlarmEnv] = {}

@main.get("/")
def root():
    return {"message": "ICU Alarm Fatigue Reducer API is running 🚑", "docs": "/docs"}

@main.get("/health")
def health():
    return {"status": "ok"}

@main.post("/reset", response_model=Observation)
def reset(session_id: str = "default", task_level: str = "easy"):
    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(400, "task_level must be easy, medium, or hard")
    _envs[session_id] = ICUAlarmEnv(task_level=task_level)
    return _envs[session_id].reset()

@main.post("/step", response_model=StepResult)
def step(action: Action, session_id: str = "default"):
    if session_id not in _envs:
        raise HTTPException(400, "No active session. Call /reset first.")
    try:
        return _envs[session_id].step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

@main.get("/state", response_model=EnvState)
def state(session_id: str = "default"):
    if session_id not in _envs:
        raise HTTPException(400, "No active session. Call /reset first.")
    return _envs[session_id].state()

@main.get("/verify_score")
def verify(score: float):
    return {"score": score, "valid": verify_score(score)}

def start():
    # This matches the folder structure (server) and the variable name (main)
    uvicorn.run("server.app:main", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    start()
