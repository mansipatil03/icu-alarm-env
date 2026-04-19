Try it here: https://huggingface.co/spaces/mansi-patil/icu-alarm-env

---
title: ICU Alarm Fatigue Reducer

colorFrom: blue
colorTo: pink
sdk: docker
app_file: app.py
pinned: false
---
# 🏥 ICU Alarm Fatigue Reducer — OpenEnv

> *Every 2 minutes, an ICU nurse responds to an alarm. 99% are false. Our agent learns to tell the difference — so nurses focus on the 1% that matters.*

---

## 🌍 Real-World Problem

**Alarm fatigue** is one of the most dangerous issues in modern ICUs. Nurses are bombarded with thousands of alarms per day — most of which are false positives caused by patient movement, sensor artifacts, or overly sensitive thresholds.

This desensitization leads to delayed responses to **real emergencies**, contributing to preventable patient deaths. This OpenEnv environment trains AI agents to solve this problem.

---

## 🎯 Environment Overview

An AI agent receives ICU patient data and must classify each alarm:
- `"real"` → Genuine medical emergency requiring immediate nurse response
- `"false"` → False alarm that can be safely deprioritized

### Three Task Levels

| Level | Task | What Agent Must Do |
|-------|------|--------------------|
| 🟢 Easy | Single vital spike | Classify one abnormal reading with context clues |
| 🟡 Medium | Multi-vital pattern | Analyze 3 alarms across different vitals in 10 min |
| 🔴 Hard | Full 24hr history | Classify + explain + recommend nurse action |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server locally
```bash
python app.py
```
Server starts at `http://localhost:7860`

### 3. Test with curl
```bash
# Reset (get a new patient scenario)
curl -X POST "http://localhost:7860/reset?task_level=easy&session_id=test1"

# Submit agent action
curl -X POST "http://localhost:7860/step?session_id=test1" \
  -H "Content-Type: application/json" \
  -d '{"classification": "real", "confidence": 0.9, "explanation": "HR is critically elevated and persistent"}'
```

### 4. Run baseline inference
```bash
export API_KEY_TOKEN=your_openai_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_SPACE=http://localhost:7860
export HF_TOKEN=your_hf_token

python inference.py
```

### 5. Run validation
```bash
python validate.py --space-url http://localhost:7860
```

---

## 🐳 Docker

```bash
# Build
docker build -t icu-alarm-env .

# Run
docker run -p 7860:7860 icu-alarm-env
```

---

## 📡 API Reference

### `POST /reset`
Start a new episode with a fresh patient scenario.

**Query params:**
- `task_level`: `easy` | `medium` | `hard`
- `session_id`: any string identifier

**Returns:** `Observation` object with patient vitals, alarm history, and task prompt.

---

### `POST /step`
Submit agent's classification and get reward.

**Body:**
```json
{
  "classification": "real",
  "confidence": 0.85,
  "explanation": "Patient showing progressive SpO2 decline over 6 hours",
  "recommended_action": "Notify attending physician immediately"
}
```

**Returns:** `StepResult` with reward (0.0–1.0), done flag, and info.

---

### `GET /state`
Get current environment state.

---

### `GET /health`
Health check — returns `{"status": "ok"}`

---

## 🎯 Reward Structure

| Outcome | Score |
|---------|-------|
| Correct classification | 0.6–0.8 base |
| + Quality explanation | +0.15–0.2 |
| + Recommended action | +0.15 |
| + Confidence calibration | +0.1 |
| Miss a real emergency | **0.0** (patient safety) |
| False positive | 0.1 |
| **Maximum** | **1.0** |

---

## 📊 Baseline Scores

| Task | Baseline Score (gpt-4o-mini) |
|------|------------------------------|
| Easy | ~0.85 |
| Medium | ~0.78 |
| Hard | ~0.72 |
| **Average** | **~0.78** |

---

## 🧠 Observation Space

```yaml
task_level: easy | medium | hard
patient_id: string
prompt: string (natural language task for agent)
vitals:
  heart_rate: float (BPM)
  systolic_bp: float (mmHg)
  diastolic_bp: float (mmHg)
  spo2: float (%)
  respiratory_rate: float (breaths/min)
  temperature: float (°C)
alarm_history: list of AlarmEvent
metadata: dict (medications, diagnosis, age for hard task)
```

## ⚡ Action Space

```yaml
classification: "real" | "false"   # required
confidence: float [0.0, 1.0]       # optional, bonus
explanation: string                 # optional, bonus
recommended_action: string          # optional, bonus (required for full score on hard)
```

---

## 📁 Project Structure

```
icu-alarm-env/
├── app.py          # FastAPI server
├── env.py          # Main OpenEnv environment
├── tasks.py        # 3 graded tasks with scoring
├── models.py       # Pydantic data models
├── inference.py    # Baseline agent script
├── validate.py     # Pre-submission checker
├── openenv.yaml    # OpenEnv spec file
├── Dockerfile      # Container config
├── requirements.txt
└── README.md
```

---

## 📄 License
MIT
