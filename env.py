"""
ICU Alarm Fatigue Reducer - OpenEnv Environment
An AI agent that reviews ICU patient vitals + alarm history
and decides: real emergency or false alarm?
"""

import random
import json
from typing import Any
from models import Observation, Action, StepResult, EnvState
from tasks import TASKS, get_task


class ICUAlarmEnv:
    """
    OpenAI Gym-compatible environment for ICU Alarm Fatigue Reduction.
    The agent must classify ICU alarms as real emergencies or false alarms.
    """

    def __init__(self, task_level: str = "easy"):
        assert task_level in ["easy", "medium", "hard"], \
            "task_level must be 'easy', 'medium', or 'hard'"
        self.task_level = task_level
        self.task = get_task(task_level)
        self._current_obs = None
        self._done = False
        self._step_count = 0
        self._max_steps = 5

    def reset(self) -> Observation:
        """Reset the environment and return a new observation."""
        self._done = False
        self._step_count = 0
        scenario = self.task.generate_scenario()
        self._current_obs = Observation(
            task_level=self.task_level,
            patient_id=scenario["patient_id"],
            prompt=scenario["prompt"],
            vitals=scenario["vitals"],
            alarm_history=scenario["alarm_history"],
            metadata=scenario.get("metadata", {})
        )
        self._ground_truth = scenario["ground_truth"]
        return self._current_obs

    def step(self, action: Action) -> StepResult:
        """
        Take an action (agent's classification) and return reward + next state.
        action.classification: 'real' or 'false'
        action.explanation: agent's reasoning (optional, gives bonus)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        reward, info = self.task.grade(
            action=action,
            ground_truth=self._ground_truth,
            observation=self._current_obs
        )
        self._done = True  # single-step task per episode

        return StepResult(
            observation=self._current_obs,
            reward=round(reward, 3),
            done=self._done,
            info=info
        )

    def state(self) -> EnvState:
        """Return the current environment state."""
        return EnvState(
            task_level=self.task_level,
            step_count=self._step_count,
            done=self._done,
            current_observation=self._current_obs
        )


def verify_score(score: float) -> bool:
    """Verify that score is in valid range 0.0-1.0"""
    return 0.0 <= score <= 1.0
