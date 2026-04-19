
title: ICU Alarm Fatigue Reducer
emoji: 🚑
colorFrom: blue
colorTo: pink
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
---

<div align="center">

# 🏥 ICU Alarm Fatigue Reducer
### An OpenEnv-Compatible AI Training Environment

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace-orange?style=for-the-badge)](https://huggingface.co/spaces/mansi-patil/icu-alarm-env)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/mansipatil03/icu-alarm-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=for-the-badge)](https://huggingface.co/spaces/mansi-patil/icu-alarm-env)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> *Every 2 minutes, an ICU nurse responds to an alarm. **99% are false.***
> *Nurses become desensitized — and miss the 1% that kills.*
> *This environment trains AI agents to tell the difference.*

</div>

---

## 🌍 The Real-World Problem

**Alarm fatigue** is one of the most critical safety issues in modern healthcare:

- 🔔 ICU patients trigger **~288 alarms per day** on average
- ❌ **99% of these alarms are false positives** — caused by patient movement, sensor artifacts, or overly sensitive thresholds
- 😔 Nurses become desensitized and begin **ignoring alarms altogether**
- ☠️ This leads to **delayed responses to real emergencies** — contributing to preventable deaths

This OpenEnv environment trains AI agents to **accurately classify ICU alarms**, so nurses can focus on what truly matters.

---

## 🎯 Environment Overview

An AI agent receives ICU patient data and must classify each alarm:

| Label | Meaning |
|-------|---------|
| `"real"` | Genuine medical emergency — nurse must respond immediately |
| `"false"` | False alarm — safe to deprioritize |

### 🗂️ Three Task Levels

| Level | Task | Difficulty | Agent Must |
|-------|------|-----------|-----------|
| 🟢 **Easy** | Single Vital Spike | Straightforward | Classify one abnormal reading using context clues |
| 🟡 **Medium** | Multi-Vital Pattern | Moderate | Analyze 3 alarms across vitals in a 10-min window |
| 🔴 **Hard** | Full 24-Hour History | Complex | Classify + explain + recommend specific nurse action |

---

## ✨ Features

- ✅ **Full OpenEnv spec compliance** — `step()`, `reset()`, `state()`, typed models, `openenv.yaml`
- ✅ **3 difficulty levels** with deterministic grading (0.0–1.0)
- ✅ **Realistic synthetic patient data** — vitals, alarm history, medications, diagnoses
- ✅ **Rich reward function** — partial credit for explanations and confidence calibration
- ✅ **Patient safety penalty** — missing a real emergency scores 0.0
- ✅ **Interactive web UI** — live demo with animated vitals display
- ✅ **Docker containerized** — one command deployment
- ✅ **Baseline inference script** — GPT-4o-mini agent included

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/mansipatil03/icu-alarm-env.git
cd icu-alarm-env
pip install -r requirements.txt
```

### 2. Run Locally
```bash
python app.py
# Server starts at http://localhost:7860
```

### 3. Test the API
```bash
# Get a new patient scenario
curl -X POST "http://localhost:7860/reset?task_level=easy&session_id=test1"

# Submit agent classification
curl -X POST "http://localhost:7860/step?session_id=test1" \
  -H "Content-Type: application/json" \
  -d '{
    "classification": "real",
    "confidence": 0.9,
    "explanation": "HR critically elevated and persistent — not a sensor artifact",
    "recommended_action": "Notify attending physician immediately"
  }'
```

### 4. Run Baseline Inference
```bash
export API_KEY_TOKEN=your_openai_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_SPACE=http://localhost:7860
export HF_TOKEN=your_hf_token

python inference.py
```

### 5. Validate
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

**Returns:** `Observation` with patient vitals, alarm history, and task prompt.

---

### `POST /step`
Submit agent classification and receive reward.

**Request body:**
```json
{
  "classification": "real",
  "confidence": 0.85,
  "explanation": "Patient showing progressive SpO2 decline — pattern consistent with respiratory failure",
  "recommended_action": "Notify attending physician, prepare oxygen support"
}
```

**Returns:** `StepResult` with reward (0.0–1.0), done flag, and detailed info.

---

### `GET /state`
Returns current environment state and step count.

### `GET /health`
Health check — returns `{"status": "ok"}`

### `GET /docs`
Full interactive Swagger API documentation.

---

## 🎯 Reward Structure

| Outcome | Score | Notes |
|---------|-------|-------|
| ✅ Correct classification | `0.6 – 0.8` | Base reward |
| 💬 + Quality explanation | `+0.15 – 0.2` | Clinical reasoning bonus |
| 🏥 + Recommended action | `+0.15` | Nurse action bonus |
| 📊 + Confidence ≥ 0.7 | `+0.10` | Calibration bonus |
| ☠️ Missed real emergency | `0.0` | Patient safety penalty |
| ⚠️ False positive | `0.1` | Low penalty |
| 🏆 **Maximum** | **`1.0`** | Perfect classification |

---

## 📊 Baseline Scores

Tested with `gpt-4o-mini` as the baseline agent:

| Task | Score | Notes |
|------|-------|-------|
| 🟢 Easy | `~0.85` | Strong single-vital reasoning |
| 🟡 Medium | `~0.78` | Good pattern recognition |
| 🔴 Hard | `~0.72` | Complex history analysis |
| **Average** | **`~0.78`** | Solid baseline performance |

---

## 🧠 Observation Space

```yaml
task_level: easy | medium | hard
patient_id: string
prompt: string          # Natural language task for the agent
vitals:
  heart_rate: float     # BPM (normal: 60–100)
  systolic_bp: float    # mmHg (normal: 90–140)
  diastolic_bp: float   # mmHg
  spo2: float           # % (critical if <95)
  respiratory_rate: float  # breaths/min (normal: 12–20)
  temperature: float    # °C
alarm_history:          # List of past alarm events
  - timestamp: string
    alarm_type: string
    value: float
    threshold: float
    was_acknowledged: bool
metadata:               # Hard task extras
  diagnosis: string
  medications: list
  age: int
```

## ⚡ Action Space

```yaml
classification: "real" | "false"    # Required
confidence: float [0.0, 1.0]        # Optional — bonus score
explanation: string                  # Optional — bonus score
recommended_action: string           # Optional — bonus (required for max score on hard)
```

---

## 📁 Project Structure

```
icu-alarm-env/
├── app.py            # FastAPI server + Web UI
├── env.py            # Main OpenEnv environment class
├── tasks.py          # 3 graded tasks with scoring logic
├── models.py         # Pydantic data models
├── inference.py      # Baseline agent script (gpt-4o-mini)
├── validate.py       # Pre-submission validation checker
├── openenv.yaml      # OpenEnv spec file
├── pyproject.toml    # Project metadata
├── uv.lock           # Dependency lock file
├── Dockerfile        # Container configuration
├── requirements.txt  # Python dependencies
├── server/
│   ├── __init__.py
│   └── app.py        # Multi-mode server entry point
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Containerization | Docker |
| Deployment | HuggingFace Spaces |
| Baseline Agent | OpenAI GPT-4o-mini |
| Environment Spec | OpenEnv |

---

## 👥 Team

**Team DuoNexus** — Built for the OpenEnv Hackathon by Scaler × HuggingFace

| Member | Role |
|--------|------|
| Mansi Patil | Team Lead · Backend · Environment Design |
| Pooja Patil | Contributor · Testing · Documentation |

*MCA Students — MET Institute of Computer Science, Bhujbal Knowledge City*

---

## 📄 License

MIT © 2026 Team DuoNexus
