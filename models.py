"""
Pydantic models for ICU Alarm Fatigue Reducer OpenEnv
"""

from pydantic import BaseModel, Field
from typing import Optional, Any


class VitalSigns(BaseModel):
    heart_rate: float = Field(..., description="Heart rate in BPM")
    systolic_bp: float = Field(..., description="Systolic blood pressure mmHg")
    diastolic_bp: float = Field(..., description="Diastolic blood pressure mmHg")
    spo2: float = Field(..., description="Oxygen saturation %")
    respiratory_rate: float = Field(..., description="Breaths per minute")
    temperature: Optional[float] = Field(None, description="Body temperature Celsius")


class AlarmEvent(BaseModel):
    timestamp: str = Field(..., description="Time of alarm e.g. 14:32")
    alarm_type: str = Field(..., description="Type of alarm triggered")
    value: float = Field(..., description="The value that triggered alarm")
    threshold: float = Field(..., description="Threshold that was breached")
    was_acknowledged: bool = Field(..., description="Was alarm acknowledged by nurse")


class Observation(BaseModel):
    task_level: str = Field(..., description="easy | medium | hard")
    patient_id: str = Field(..., description="Anonymous patient identifier")
    prompt: str = Field(..., description="Natural language task description for agent")
    vitals: VitalSigns = Field(..., description="Current patient vital signs")
    alarm_history: list[AlarmEvent] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict, description="Extra context like medications, age, etc.")


class Action(BaseModel):
    classification: str = Field(..., description="'real' or 'false'")
    confidence: Optional[float] = Field(None, description="Agent confidence 0.0-1.0")
    explanation: Optional[str] = Field(None, description="Agent reasoning - gives bonus score")
    recommended_action: Optional[str] = Field(None, description="What nurse should do (hard task)")


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(..., description="Score 0.0-1.0")
    done: bool
    info: dict = Field(default_factory=dict)


class EnvState(BaseModel):
    task_level: str
    step_count: int
    done: bool
    current_observation: Optional[Observation] = None
