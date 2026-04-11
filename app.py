"""
FastAPI server exposing OpenEnv-compatible endpoints for ICU Alarm Fatigue Reducer
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from models import Action, Observation, StepResult, EnvState
from env import ICUAlarmEnv, verify_score
import uvicorn

app = FastAPI(
    title="ICU Alarm Fatigue Reducer — OpenEnv",
    description="An AI environment where agents learn to distinguish real ICU emergencies from false alarms.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_envs: dict[str, ICUAlarmEnv] = {}

def get_or_create_env(session_id: str, task_level: str = "easy") -> ICUAlarmEnv:
    if session_id not in _envs:
        _envs[session_id] = ICUAlarmEnv(task_level=task_level)
    return _envs[session_id]

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ICU Alarm Fatigue Reducer — OpenEnv</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #050a0f;
    --surface: #0b1623;
    --card: #0f1e2e;
    --border: #1a3048;
    --accent: #00d4ff;
    --accent2: #ff4d6d;
    --accent3: #00ff9d;
    --text: #e0eaf5;
    --muted: #5a7a94;
    --font-display: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated grid background */
  body::before {
    content:'';
    position:fixed;
    inset:0;
    background-image:
      linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events:none;
    z-index:0;
  }

  .container { max-width: 1100px; margin: 0 auto; padding: 0 24px; position:relative; z-index:1; }

  /* HEADER */
  header {
    padding: 48px 0 32px;
    border-bottom: 1px solid var(--border);
    animation: fadeDown 0.7s ease both;
  }
  .header-top { display:flex; align-items:center; gap:16px; margin-bottom:12px; }
  .badge {
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent);
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 2px;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .badge.live {
    background: rgba(0,255,157,0.08);
    border-color: rgba(0,255,157,0.3);
    color: var(--accent3);
    display:flex; align-items:center; gap:6px;
  }
  .pulse {
    width:7px; height:7px; border-radius:50%;
    background: var(--accent3);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%,100%{ opacity:1; transform:scale(1); }
    50%{ opacity:0.4; transform:scale(1.4); }
  }
  h1 {
    font-family: var(--font-display);
    font-size: clamp(28px, 5vw, 52px);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -1px;
  }
  h1 span { color: var(--accent); }
  .tagline {
    margin-top: 16px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
    max-width: 600px;
    border-left: 2px solid var(--accent2);
    padding-left: 14px;
  }
  .tagline em { color: var(--accent2); font-style: normal; font-weight:700; }

  /* STATS ROW */
  .stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    margin: 32px 0;
    border: 1px solid var(--border);
    animation: fadeUp 0.6s 0.2s ease both;
  }
  .stat {
    background: var(--card);
    padding: 20px;
    text-align: center;
  }
  .stat-value {
    font-family: var(--font-display);
    font-size: 28px;
    font-weight: 800;
    color: var(--accent);
  }
  .stat-label { font-size: 10px; color: var(--muted); letter-spacing: 1.5px; text-transform:uppercase; margin-top:4px; }

  /* TASKS */
  .section-title {
    font-family: var(--font-display);
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
  }
  .tasks { display:grid; grid-template-columns: repeat(3,1fr); gap:16px; margin-bottom:32px; animation: fadeUp 0.6s 0.3s ease both; }
  .task-card {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
  }
  .task-card::before {
    content:'';
    position:absolute;
    top:0; left:0; right:0;
    height:2px;
  }
  .task-card.easy::before { background: var(--accent3); }
  .task-card.medium::before { background: #ffd166; }
  .task-card.hard::before { background: var(--accent2); }
  .task-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .task-level {
    font-size: 10px;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:8px;
    font-weight:700;
  }
  .task-card.easy .task-level { color: var(--accent3); }
  .task-card.medium .task-level { color: #ffd166; }
  .task-card.hard .task-level { color: var(--accent2); }
  .task-name { font-family: var(--font-display); font-size:15px; font-weight:700; margin-bottom:8px; }
  .task-desc { font-size:11px; color: var(--muted); line-height:1.6; }

  /* REWARD TABLE */
  .reward-grid { margin-bottom:32px; animation: fadeUp 0.6s 0.4s ease both; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th {
    text-align:left; padding:10px 14px;
    background: var(--surface);
    color: var(--muted);
    font-size:10px; letter-spacing:1.5px; text-transform:uppercase;
    border-bottom: 1px solid var(--border);
  }
  td { padding:10px 14px; border-bottom: 1px solid rgba(26,48,72,0.5); }
  tr:hover td { background: rgba(0,212,255,0.03); }
  .score-good { color: var(--accent3); font-weight:700; }
  .score-bad { color: var(--accent2); font-weight:700; }
  .score-warn { color: #ffd166; font-weight:700; }

  /* ENDPOINTS */
  .endpoints { margin-bottom:32px; animation: fadeUp 0.6s 0.5s ease both; }
  .endpoint {
    display:grid; grid-template-columns: 70px 220px 1fr;
    align-items:center; gap:16px;
    padding:12px 16px;
    background: var(--card);
    border: 1px solid var(--border);
    border-bottom: none;
    font-size:12px;
    transition: background 0.15s;
  }
  .endpoint:last-child { border-bottom: 1px solid var(--border); }
  .endpoint:hover { background: rgba(0,212,255,0.04); }
  .method {
    font-size:10px; font-weight:700; letter-spacing:1px;
    padding:3px 8px; border-radius:2px; text-align:center;
  }
  .method.post { background:rgba(0,255,157,0.1); color:var(--accent3); border:1px solid rgba(0,255,157,0.2); }
  .method.get  { background:rgba(0,212,255,0.1); color:var(--accent);  border:1px solid rgba(0,212,255,0.2); }
  .endpoint-path { color: var(--accent); font-family: var(--font-mono); }
  .endpoint-desc { color: var(--muted); font-size:11px; }

  /* TRY IT */
  .try-it { margin-bottom:48px; animation: fadeUp 0.6s 0.6s ease both; }
  .try-box {
    background: var(--card);
    border: 1px solid var(--border);
    padding:24px;
  }
  .controls { display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap; }
  select, button {
    font-family: var(--font-mono);
    font-size: 12px;
    padding: 10px 16px;
    border-radius: 2px;
    cursor: pointer;
  }
  select {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-reset {
    background: rgba(0,212,255,0.1);
    color: var(--accent);
    border: 1px solid rgba(0,212,255,0.4);
    transition: all 0.2s;
  }
  .btn-reset:hover { background: rgba(0,212,255,0.2); }
  .btn-step {
    background: rgba(0,255,157,0.1);
    color: var(--accent3);
    border: 1px solid rgba(0,255,157,0.4);
    transition: all 0.2s;
  }
  .btn-step:hover { background: rgba(0,255,157,0.2); }
  .output {
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 16px;
    font-size: 11px;
    color: var(--accent);
    min-height: 120px;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.7;
    max-height: 320px;
    overflow-y: auto;
  }
  .output.error { color: var(--accent2); }
  .output.success { color: var(--accent3); }

  /* FOOTER */
  footer {
    border-top: 1px solid var(--border);
    padding: 24px 0;
    text-align:center;
    color: var(--muted);
    font-size:11px;
    letter-spacing:1px;
  }

  @keyframes fadeDown { from{ opacity:0; transform:translateY(-16px); } to{ opacity:1; transform:none; } }
  @keyframes fadeUp   { from{ opacity:0; transform:translateY(16px);  } to{ opacity:1; transform:none; } }

  @media(max-width:700px) {
    .stats { grid-template-columns:repeat(2,1fr); }
    .tasks { grid-template-columns:1fr; }
    .endpoint { grid-template-columns:60px 1fr; }
    .endpoint-desc { display:none; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="header-top">
      <span class="badge">OpenEnv</span>
      <span class="badge live"><span class="pulse"></span>Live</span>
    </div>
    <h1>ICU Alarm<br/><span>Fatigue Reducer</span></h1>
    <p class="tagline">
      Every 2 minutes, an ICU nurse responds to an alarm. <em>99% are false.</em><br/>
      This environment trains AI agents to tell the difference — so nurses focus on the 1% that matters.
    </p>
  </header>

  <div class="stats">
    <div class="stat"><div class="stat-value">3</div><div class="stat-label">Task Levels</div></div>
    <div class="stat"><div class="stat-value">99%</div><div class="stat-label">ICU False Alarm Rate</div></div>
    <div class="stat"><div class="stat-value">1.0</div><div class="stat-label">Max Reward</div></div>
    <div class="stat"><div class="stat-value">0.0</div><div class="stat-label">Missed Emergency</div></div>
  </div>

  <p class="section-title">// Task Levels</p>
  <div class="tasks">
    <div class="task-card easy">
      <div class="task-level">🟢 Easy</div>
      <div class="task-name">Single Vital Spike</div>
      <div class="task-desc">Classify one abnormal vital reading. Context clues indicate sensor artifact vs genuine alert.</div>
    </div>
    <div class="task-card medium">
      <div class="task-level">🟡 Medium</div>
      <div class="task-name">Multi-Vital Pattern</div>
      <div class="task-desc">Analyze 3 alarms across different vitals in a 10-minute window. Identify the pattern.</div>
    </div>
    <div class="task-card hard">
      <div class="task-level">🔴 Hard</div>
      <div class="task-name">Full 24hr History</div>
      <div class="task-desc">Complete patient record: medications, diagnoses, vitals trend + alarm log. Classify + recommend action.</div>
    </div>
  </div>

  <p class="section-title">// Reward Structure</p>
  <div class="reward-grid">
    <table>
      <thead><tr><th>Outcome</th><th>Score</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td>Correct classification</td><td class="score-good">0.6 – 0.8</td><td>Base reward</td></tr>
        <tr><td>+ Quality explanation</td><td class="score-good">+0.15 – 0.2</td><td>Clinical reasoning bonus</td></tr>
        <tr><td>+ Recommended action</td><td class="score-good">+0.15</td><td>Nurse action bonus</td></tr>
        <tr><td>+ Confidence calibration</td><td class="score-good">+0.1</td><td>When confidence ≥ 0.7</td></tr>
        <tr><td>Missed real emergency</td><td class="score-bad">0.0</td><td>Patient safety penalty</td></tr>
        <tr><td>False positive</td><td class="score-warn">0.1</td><td>Low penalty</td></tr>
        <tr><td>Maximum score</td><td class="score-good">1.0</td><td>Perfect classification</td></tr>
      </tbody>
    </table>
  </div>

  <p class="section-title">// API Endpoints</p>
  <div class="endpoints">
    <div class="endpoint">
      <span class="method post">POST</span>
      <span class="endpoint-path">/reset</span>
      <span class="endpoint-desc">Start new episode — returns patient vitals + alarm scenario</span>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span>
      <span class="endpoint-path">/step</span>
      <span class="endpoint-desc">Submit agent classification — returns reward (0.0–1.0)</span>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <span class="endpoint-path">/state</span>
      <span class="endpoint-desc">Current environment state</span>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <span class="endpoint-path">/health</span>
      <span class="endpoint-desc">Health check — returns {"status": "ok"}</span>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <span class="endpoint-path">/docs</span>
      <span class="endpoint-desc">Full interactive Swagger API documentation</span>
    </div>
  </div>

  <p class="section-title">// Try It Live</p>
  <div class="try-it">
    <div class="try-box">
      <div class="controls">
        <select id="taskLevel">
          <option value="easy">🟢 Easy — Single Vital Spike</option>
          <option value="medium">🟡 Medium — Multi-Vital Pattern</option>
          <option value="hard">🔴 Hard — Full 24hr History</option>
        </select>
        <button class="btn-reset" onclick="doReset()">▶ Reset / New Patient</button>
        <button class="btn-step" onclick="doStep()">⚡ Submit: classify as "real"</button>
      </div>
      <div class="output" id="output">// Click "Reset / New Patient" to load a patient scenario...</div>
    </div>
  </div>

  <footer>
    ICU Alarm Fatigue Reducer · OpenEnv · mansi-patil · 2026
    &nbsp;·&nbsp;
    <a href="/docs" style="color:var(--accent); text-decoration:none;">API Docs →</a>
  </footer>

</div>

<script>
  let sessionId = 'ui_' + Math.random().toString(36).slice(2);
  const out = document.getElementById('output');

  function display(data, type='') {
    out.className = 'output' + (type ? ' ' + type : '');
    out.textContent = JSON.stringify(data, null, 2);
  }

  async function doReset() {
    const level = document.getElementById('taskLevel').value;
    sessionId = 'ui_' + Math.random().toString(36).slice(2);
    out.className = 'output';
    out.textContent = '// Loading patient scenario...';
    try {
      const r = await fetch(`/reset?task_level=${level}&session_id=${sessionId}`, { method:'POST' });
      const data = await r.json();
      display(data);
    } catch(e) {
      display({ error: e.message }, 'error');
    }
  }

  async function doStep() {
    out.className = 'output';
    out.textContent = '// Submitting classification...';
    try {
      const r = await fetch(`/step?session_id=${sessionId}`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          classification: 'real',
          confidence: 0.85,
          explanation: 'Manually submitted via UI demo — classifying as real emergency.',
          recommended_action: 'Notify attending physician immediately.'
        })
      });
      const data = await r.json();
      display(data, data.reward >= 0.7 ? 'success' : 'error');
    } catch(e) {
      display({ error: e.message }, 'error');
    }
  }
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_UI


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(session_id: str = "default", task_level: str = "easy"):
    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(400, "task_level must be easy, medium, or hard")
    _envs[session_id] = ICUAlarmEnv(task_level=task_level)
    obs = _envs[session_id].reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action, session_id: str = "default"):
    if session_id not in _envs:
        raise HTTPException(400, "No active session. Call /reset first.")
    env = _envs[session_id]
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/state", response_model=EnvState)
def state(session_id: str = "default"):
    if session_id not in _envs:
        raise HTTPException(400, "No active session. Call /reset first.")
    return _envs[session_id].state()


@app.get("/verify_score")
def verify(score: float):
    return {"score": score, "valid": verify_score(score)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()

