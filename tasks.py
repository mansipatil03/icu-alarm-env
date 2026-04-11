"""
Three graded tasks for ICU Alarm Fatigue Reducer:
- Easy: Single vital spike classification
- Medium: Multi-vital pattern analysis
- Hard: Full 24hr patient history analysis
"""

import random
from models import Observation, Action, VitalSigns, AlarmEvent


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATORS
# ─────────────────────────────────────────────

PATIENT_IDS = [f"PT-{str(i).zfill(4)}" for i in range(1000, 9999)]

def random_patient_id():
    return random.choice(PATIENT_IDS)

def normal_vitals():
    return VitalSigns(
        heart_rate=round(random.uniform(60, 100), 1),
        systolic_bp=round(random.uniform(110, 130), 1),
        diastolic_bp=round(random.uniform(70, 85), 1),
        spo2=round(random.uniform(95, 99), 1),
        respiratory_rate=round(random.uniform(12, 20), 1),
        temperature=round(random.uniform(36.5, 37.5), 1)
    )

def abnormal_vitals(alarm_type: str):
    v = normal_vitals()
    if alarm_type == "tachycardia":
        v.heart_rate = round(random.uniform(105, 145), 1)
    elif alarm_type == "bradycardia":
        v.heart_rate = round(random.uniform(35, 55), 1)
    elif alarm_type == "hypoxia":
        v.spo2 = round(random.uniform(85, 92), 1)
    elif alarm_type == "hypertension":
        v.systolic_bp = round(random.uniform(160, 200), 1)
    elif alarm_type == "hypotension":
        v.systolic_bp = round(random.uniform(75, 95), 1)
    return v

TIMES = ["06:00","06:30","07:00","07:15","08:00","09:00","10:00","11:30",
         "12:00","13:00","14:00","15:30","16:00","17:00","18:00","19:30",
         "20:00","21:00","22:00","23:00","00:30","01:00","02:00","03:30"]


# ─────────────────────────────────────────────
# EASY TASK
# ─────────────────────────────────────────────

class EasyTask:
    """
    Single vital spike — is this one reading dangerous or a sensor glitch?
    """
    name = "single_vital_spike"
    description = "Classify a single abnormal vital sign reading as a real emergency or false alarm."

    def generate_scenario(self):
        is_real = random.choice([True, False])
        alarm_types = ["tachycardia", "bradycardia", "hypoxia", "hypertension", "hypotension"]
        alarm_type = random.choice(alarm_types)

        if is_real:
            vitals = abnormal_vitals(alarm_type)
            context = "Patient has been consistently showing this reading for the past 10 minutes."
        else:
            vitals = abnormal_vitals(alarm_type)
            context = "This is an isolated spike. Previous and subsequent readings were completely normal. Patient appears calm and comfortable."

        alarm_history = [
            AlarmEvent(
                timestamp=random.choice(TIMES),
                alarm_type=alarm_type,
                value=vitals.heart_rate if "cardia" in alarm_type else vitals.spo2,
                threshold=100.0 if alarm_type == "tachycardia" else 95.0,
                was_acknowledged=not is_real
            )
        ]

        prompt = f"""You are an ICU alarm monitoring AI assistant.

Patient {random_patient_id()} has triggered a {alarm_type.upper()} alarm.

Current Vitals:
- Heart Rate: {vitals.heart_rate} BPM
- SpO2: {vitals.spo2}%
- Blood Pressure: {vitals.systolic_bp}/{vitals.diastolic_bp} mmHg
- Respiratory Rate: {vitals.respiratory_rate} breaths/min

Context: {context}

Is this a REAL emergency or a FALSE alarm?
Respond with classification: 'real' or 'false', and optionally provide an explanation."""

        return {
            "patient_id": random_patient_id(),
            "prompt": prompt,
            "vitals": vitals,
            "alarm_history": alarm_history,
            "ground_truth": "real" if is_real else "false",
            "metadata": {"alarm_type": alarm_type, "context": context}
        }

    def grade(self, action: Action, ground_truth: str, observation: Observation):
        score = 0.0
        info = {"ground_truth": ground_truth, "predicted": action.classification}

        if action.classification == ground_truth:
            score = 0.8
            # Bonus for explanation
            if action.explanation and len(action.explanation) > 20:
                score = min(1.0, score + 0.2)
            info["result"] = "correct"
        else:
            # Penalize missing real alarm more heavily (patient safety)
            if ground_truth == "real" and action.classification == "false":
                score = 0.0
                info["result"] = "dangerous_miss"
            else:
                score = 0.1
                info["result"] = "false_positive"

        info["score"] = score
        return score, info


# ─────────────────────────────────────────────
# MEDIUM TASK
# ─────────────────────────────────────────────

class MediumTask:
    """
    Multi-vital pattern — 3 alarms in 10 mins, real emergency or noise?
    """
    name = "multi_vital_pattern"
    description = "Analyze multiple alarm events across different vital signs to classify overall patient status."

    def generate_scenario(self):
        is_real = random.choice([True, False])
        alarm_types = random.sample(["tachycardia", "hypoxia", "hypertension", "hypotension"], 2)

        if is_real:
            vitals = abnormal_vitals(alarm_types[0])
            vitals.spo2 = round(random.uniform(86, 92), 1)
            alarm_history = [
                AlarmEvent(timestamp=TIMES[i], alarm_type=alarm_types[i % 2],
                           value=vitals.heart_rate, threshold=100.0,
                           was_acknowledged=False)
                for i in range(3)
            ]
            pattern_desc = "Alarms are escalating — each reading is worse than the last."
        else:
            vitals = normal_vitals()
            alarm_history = [
                AlarmEvent(timestamp=TIMES[i], alarm_type=alarm_types[i % 2],
                           value=round(random.uniform(98, 103), 1), threshold=100.0,
                           was_acknowledged=True)
                for i in range(3)
            ]
            pattern_desc = "All alarms were brief and self-resolved. Nurse checked and found patient stable."

        prompt = f"""You are an ICU alarm monitoring AI assistant.

Patient has triggered {len(alarm_history)} alarms in the past 10 minutes.

Current Vitals:
- Heart Rate: {vitals.heart_rate} BPM
- SpO2: {vitals.spo2}%
- Blood Pressure: {vitals.systolic_bp}/{vitals.diastolic_bp} mmHg
- Respiratory Rate: {vitals.respiratory_rate} breaths/min
- Temperature: {vitals.temperature}°C

Alarm Pattern: {pattern_desc}

Alarm History:
{chr(10).join([f"  [{a.timestamp}] {a.alarm_type} = {a.value} (threshold: {a.threshold}) - {'Acknowledged' if a.was_acknowledged else 'UNACKNOWLEDGED'}" for a in alarm_history])}

Analyze the PATTERN of alarms. Is this a REAL emergency or a FALSE alarm?
Respond with classification and explanation."""

        return {
            "patient_id": random_patient_id(),
            "prompt": prompt,
            "vitals": vitals,
            "alarm_history": alarm_history,
            "ground_truth": "real" if is_real else "false",
            "metadata": {"alarm_types": alarm_types, "pattern": pattern_desc}
        }

    def grade(self, action: Action, ground_truth: str, observation: Observation):
        score = 0.0
        info = {"ground_truth": ground_truth, "predicted": action.classification}

        if action.classification == ground_truth:
            score = 0.7
            if action.explanation and len(action.explanation) > 30:
                score = min(1.0, score + 0.2)
            if action.confidence and action.confidence >= 0.7:
                score = min(1.0, score + 0.1)
            info["result"] = "correct"
        else:
            if ground_truth == "real" and action.classification == "false":
                score = 0.0
                info["result"] = "dangerous_miss"
            else:
                score = 0.15
                info["result"] = "false_positive"

        info["score"] = score
        return score, info


# ─────────────────────────────────────────────
# HARD TASK
# ─────────────────────────────────────────────

class HardTask:
    """
    Full 24hr patient history — classify + explain + recommend nurse action.
    """
    name = "full_patient_history"
    description = "Analyze 24-hour patient history with medications, vitals trends, and alarm logs to classify alarm and recommend action."

    MEDICATIONS = ["Metoprolol", "Furosemide", "Heparin", "Morphine", "Dopamine", "Norepinephrine"]
    DIAGNOSES = ["Post-cardiac surgery", "Septic shock", "Acute respiratory failure",
                 "Hypertensive crisis", "Pulmonary embolism", "Congestive heart failure"]

    def generate_scenario(self):
        is_real = random.choice([True, False])
        diagnosis = random.choice(self.DIAGNOSES)
        meds = random.sample(self.MEDICATIONS, 2)
        age = random.randint(45, 85)

        if is_real:
            vitals = abnormal_vitals(random.choice(["tachycardia", "hypoxia"]))
            alarm_history = [
                AlarmEvent(timestamp=TIMES[i*3], alarm_type=random.choice(["tachycardia","hypoxia"]),
                           value=round(random.uniform(105, 140), 1), threshold=100.0,
                           was_acknowledged=False)
                for i in range(5)
            ]
            trend = "Vitals have been progressively deteriorating over the past 6 hours."
            urgency = "HIGH"
            recommended = "Immediate physician notification required."
        else:
            vitals = normal_vitals()
            alarm_history = [
                AlarmEvent(timestamp=TIMES[i*2], alarm_type="tachycardia",
                           value=round(random.uniform(100, 105), 1), threshold=100.0,
                           was_acknowledged=True)
                for i in range(3)
            ]
            trend = "Patient has been stable. Alarms correlate with patient movement and physiotherapy sessions."
            urgency = "LOW"
            recommended = "No immediate action needed. Continue monitoring."

        prompt = f"""You are an advanced ICU alarm monitoring AI assistant.

=== PATIENT SUMMARY ===
Age: {age} years | Diagnosis: {diagnosis}
Current Medications: {', '.join(meds)}

=== CURRENT VITALS ===
- Heart Rate: {vitals.heart_rate} BPM
- SpO2: {vitals.spo2}%
- Blood Pressure: {vitals.systolic_bp}/{vitals.diastolic_bp} mmHg
- Respiratory Rate: {vitals.respiratory_rate} breaths/min
- Temperature: {vitals.temperature}°C

=== 24-HOUR ALARM LOG ({len(alarm_history)} alarms) ===
{chr(10).join([f"  [{a.timestamp}] {a.alarm_type} — value: {a.value} — {'⚠ UNACKNOWLEDGED' if not a.was_acknowledged else '✓ Acknowledged'}" for a in alarm_history])}

=== CLINICAL TREND ===
{trend}

Your task:
1. Classify this alarm: 'real' emergency or 'false' alarm
2. Provide detailed clinical explanation
3. Recommend specific nurse action
4. State urgency level: LOW / MEDIUM / HIGH

Respond with your classification, explanation, and recommended_action."""

        return {
            "patient_id": random_patient_id(),
            "prompt": prompt,
            "vitals": vitals,
            "alarm_history": alarm_history,
            "ground_truth": "real" if is_real else "false",
            "metadata": {
                "diagnosis": diagnosis,
                "medications": meds,
                "age": age,
                "urgency": urgency,
                "recommended": recommended
            }
        }

    def grade(self, action: Action, ground_truth: str, observation: Observation):
        score = 0.0
        info = {"ground_truth": ground_truth, "predicted": action.classification}

        if action.classification == ground_truth:
            score = 0.6

            # Bonus: explanation quality
            if action.explanation and len(action.explanation) > 50:
                score += 0.15

            # Bonus: recommended action provided
            if action.recommended_action and len(action.recommended_action) > 20:
                score += 0.15

            # Bonus: confidence calibration
            if action.confidence and action.confidence >= 0.7:
                score += 0.1

            score = min(1.0, score)
            info["result"] = "correct"
        else:
            if ground_truth == "real" and action.classification == "false":
                score = 0.0
                info["result"] = "dangerous_miss — missed real emergency"
            else:
                score = 0.1
                info["result"] = "false_positive"

        info["score"] = score
        return score, info


# ─────────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────────

TASKS = {
    "easy": EasyTask(),
    "medium": MediumTask(),
    "hard": HardTask()
}

def get_task(level: str):
    return TASKS[level]
