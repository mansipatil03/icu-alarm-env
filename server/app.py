"""
FastAPI server - ICU Alarm Fatigue Reducer OpenEnv
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

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_envs: dict[str, ICUAlarmEnv] = {}

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
  --amber: #ffb800;
  --text: #e0eaf5;
  --muted: #5a7a94;
  --font-display: 'Syne', sans-serif;
  --font-mono: 'Space Mono', monospace;
}
* { margin:0; padding:0; box-sizing:border-box; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-mono);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Animated grid */
body::before {
  content:'';
  position:fixed; inset:0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  animation: gridShift 30s linear infinite;
  pointer-events:none; z-index:0;
}
@keyframes gridShift { 0%{background-position:0 0;} 100%{background-position:40px 40px;} }

/* Ambient glow blobs */
.blob {
  position:fixed; border-radius:50%;
  filter:blur(100px); opacity:0.08;
  pointer-events:none; z-index:0;
  animation: blobFloat 12s ease-in-out infinite;
}
.blob1 { width:500px;height:500px;background:var(--accent);top:-150px;right:-100px; }
.blob2 { width:400px;height:400px;background:var(--accent2);bottom:-100px;left:-100px;animation-delay:4s; }
.blob3 { width:300px;height:300px;background:var(--accent3);top:50%;left:40%;animation-delay:8s; }
@keyframes blobFloat { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(30px,-30px) scale(1.1);} }

.container { max-width:1200px; margin:0 auto; padding:0 32px; position:relative; z-index:1; }

/* ── HEADER ── */
header {
  padding: 48px 0 40px;
  border-bottom: 1px solid var(--border);
  animation: fadeDown 0.6s ease both;
}
.header-top {
  display:flex; justify-content:space-between; align-items:center;
  margin-bottom: 32px;
}
.badges { display:flex; gap:10px; flex-wrap:wrap; }
.badge {
  font-size:10px; letter-spacing:2px; text-transform:uppercase;
  padding:5px 12px; border-radius:2px; border:1px solid;
}
.badge-openenv { background:rgba(0,212,255,0.1); border-color:rgba(0,212,255,0.3); color:var(--accent); }
.badge-live {
  background:rgba(0,255,157,0.08); border-color:rgba(0,255,157,0.3); color:var(--accent3);
  display:flex; align-items:center; gap:6px;
}
.badge-tag { background:rgba(255,184,0,0.08); border-color:rgba(255,184,0,0.3); color:var(--amber); }
.pulse { width:7px;height:7px;border-radius:50%;background:var(--accent3);animation:pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:0.3;transform:scale(1.5);} }

.nav-links { display:flex; gap:20px; align-items:center; }
.nav-links a {
  font-size:11px; letter-spacing:1px; color:var(--muted);
  text-decoration:none; text-transform:uppercase;
  transition:color 0.2s;
}
.nav-links a:hover { color:var(--accent); }

/* Hero */
.hero { display:grid; grid-template-columns:1fr 1fr; gap:40px; align-items:center; }
.hero-left {}
h1 {
  font-family: var(--font-display);
  font-size: clamp(36px,5vw,64px);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -1px;
  margin-bottom: 24px;
}
h1 .line2 { color: var(--accent); display:block; }
.tagline {
  font-size: 13px; line-height: 1.9;
  color: var(--muted);
  border-left: 2px solid var(--accent2);
  padding-left: 16px;
  margin-bottom: 32px;
}
.tagline .highlight { color: var(--accent2); font-weight:700; }

/* Alarm counter animation */
.hero-right {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 32px;
  position:relative;
  overflow:hidden;
}
.hero-right::before {
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent2),var(--accent),var(--accent3));
}
.alarm-display {
  text-align:center; margin-bottom:24px;
}
.alarm-num {
  font-family:var(--font-display);
  font-size:72px; font-weight:800;
  color:var(--accent2);
  line-height:1;
  animation:countUp 0s;
}
.alarm-label { font-size:10px;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-top:4px; }
.alarm-meta { font-size:11px;color:var(--muted);text-align:center;margin-bottom:20px; }
.alarm-meta span { color:var(--accent3); }

.mini-vitals { display:grid;grid-template-columns:1fr 1fr;gap:8px; }
.mini-vital {
  background:var(--surface);border:1px solid var(--border);
  padding:10px 12px;display:flex;justify-content:space-between;align-items:center;
}
.mv-name { font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted); }
.mv-val { font-family:var(--font-display);font-size:20px;font-weight:700; }
.mv-normal { color:var(--accent3); }
.mv-warn { color:var(--amber); }
.mv-crit { color:var(--accent2);animation:blink 0.8s infinite; }
@keyframes blink { 0%,100%{opacity:1;}50%{opacity:0.3;} }

/* ── STATS ── */
.stats-row {
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:1px;background:var(--border);
  border:1px solid var(--border);
  margin:40px 0;
  animation: fadeUp 0.6s 0.1s ease both;
}
.stat {
  background:var(--card);padding:24px 20px;text-align:center;
  transition:background 0.2s;
}
.stat:hover { background:rgba(0,212,255,0.05); }
.stat-val {
  font-family:var(--font-display);font-size:36px;font-weight:800;
  color:var(--accent);
  transition:all 0.3s;
}
.stat-lbl { font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-top:6px; }

/* ── SECTION TITLES ── */
.section-title {
  font-size:10px;letter-spacing:3px;text-transform:uppercase;
  color:var(--muted);margin-bottom:20px;
  display:flex;align-items:center;gap:10px;
}
.section-title::before { content:'//'; color:var(--accent);font-weight:700; }
.section-title::after { content:'';flex:1;height:1px;background:var(--border); }

/* ── TASKS ── */
.tasks {
  display:grid;grid-template-columns:repeat(3,1fr);
  gap:16px;margin-bottom:40px;
  animation: fadeUp 0.6s 0.2s ease both;
}
.task-card {
  background:var(--card);border:1px solid var(--border);
  padding:24px;position:relative;overflow:hidden;
  cursor:pointer;transition:all 0.25s;
}
.task-card::before {
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  transition:height 0.2s;
}
.task-card.easy::before { background:var(--accent3); }
.task-card.medium::before { background:var(--amber); }
.task-card.hard::before { background:var(--accent2); }
.task-card:hover { transform:translateY(-4px);border-color:var(--accent); box-shadow:0 8px 32px rgba(0,212,255,0.1); }
.task-card:hover::before { height:4px; }
.task-card.selected { border-color:var(--accent);background:rgba(0,212,255,0.05); }

.task-icon { font-size:28px;margin-bottom:12px; }
.task-level { font-size:9px;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:6px; }
.easy .task-level { color:var(--accent3); }
.medium .task-level { color:var(--amber); }
.hard .task-level { color:var(--accent2); }
.task-name { font-family:var(--font-display);font-size:17px;font-weight:700;margin-bottom:10px; }
.task-desc { font-size:11px;color:var(--muted);line-height:1.7; }
.task-badge {
  display:inline-block;margin-top:14px;
  font-size:9px;letter-spacing:1px;padding:3px 8px;border-radius:2px;
}
.easy .task-badge { background:rgba(0,255,157,0.1);color:var(--accent3);border:1px solid rgba(0,255,157,0.2); }
.medium .task-badge { background:rgba(255,184,0,0.1);color:var(--amber);border:1px solid rgba(255,184,0,0.2); }
.hard .task-badge { background:rgba(255,77,109,0.1);color:var(--accent2);border:1px solid rgba(255,77,109,0.2); }

/* ── INTERACTIVE DEMO ── */
.demo-grid {
  display:grid;grid-template-columns:1fr 1.4fr;
  gap:16px;margin-bottom:40px;
  animation: fadeUp 0.6s 0.3s ease both;
}

.demo-controls { background:var(--card);border:1px solid var(--border);padding:28px; }
.level-select { display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap; }
.level-btn {
  flex:1;min-width:80px;
  font-family:var(--font-mono);font-size:10px;letter-spacing:1px;
  text-transform:uppercase;padding:10px 8px;
  border:1px solid var(--border);background:transparent;
  color:var(--muted);cursor:pointer;transition:all 0.2s;border-radius:2px;
}
.level-btn:hover { border-color:var(--accent);color:var(--accent); }
.level-btn.active-easy { border-color:var(--accent3);color:var(--accent3);background:rgba(0,255,157,0.08); }
.level-btn.active-medium { border-color:var(--amber);color:var(--amber);background:rgba(255,184,0,0.08); }
.level-btn.active-hard { border-color:var(--accent2);color:var(--accent2);background:rgba(255,77,109,0.08); }

.action-btns { display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px; }
.btn {
  font-family:var(--font-mono);font-size:11px;letter-spacing:1px;
  text-transform:uppercase;padding:13px 16px;
  border:1px solid;cursor:pointer;transition:all 0.2s;border-radius:2px;
  font-weight:700;
}
.btn-new { background:rgba(0,212,255,0.08);color:var(--accent);border-color:rgba(0,212,255,0.3); }
.btn-new:hover { background:rgba(0,212,255,0.15);box-shadow:0 0 20px rgba(0,212,255,0.15); }
.btn-real { background:rgba(255,77,109,0.08);color:var(--accent2);border-color:rgba(255,77,109,0.3); grid-column:span 2; }
.btn-real:hover { background:rgba(255,77,109,0.15);box-shadow:0 0 20px rgba(255,77,109,0.15); }
.btn-false { background:rgba(0,255,157,0.08);color:var(--accent3);border-color:rgba(0,255,157,0.3); grid-column:span 2; }
.btn-false:hover { background:rgba(0,255,157,0.15);box-shadow:0 0 20px rgba(0,255,157,0.15); }

/* Patient card */
.patient-card {
  background:var(--surface);border:1px solid var(--border);
  padding:16px;margin-bottom:16px;display:none;
}
.patient-id { font-size:10px;letter-spacing:2px;color:var(--muted);margin-bottom:8px; }
.patient-prompt { font-size:11px;line-height:1.7;color:var(--text); }

/* Vitals mini */
.vitals-mini { display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:16px;display:none; }
.vm {
  background:var(--surface);border:1px solid var(--border);
  padding:8px 10px;display:flex;flex-direction:column;gap:2px;
}
.vm-k { font-size:8px;letter-spacing:1px;text-transform:uppercase;color:var(--muted); }
.vm-v { font-family:var(--font-display);font-size:18px;font-weight:700; }
.normal { color:var(--accent3); }
.warning { color:var(--amber); }
.critical { color:var(--accent2); }

/* Score display */
.score-display { display:none; }
.score-bar-wrap { background:var(--surface);border:1px solid var(--border);height:28px;position:relative;margin-bottom:8px; }
.score-fill { height:100%;width:0%;transition:width 0.6s ease;background:linear-gradient(90deg,var(--accent2),var(--amber),var(--accent3)); }
.score-num { position:absolute;right:10px;top:50%;transform:translateY(-50%);font-weight:700;font-size:13px; }
.score-verdict { font-size:11px;text-align:center;padding:8px;background:var(--surface);border:1px solid var(--border); }

/* Live feed */
.live-feed {
  background:#000;border:1px solid rgba(0,212,255,0.15);
  padding:0;display:flex;flex-direction:column;
}
.feed-header {
  padding:12px 16px;
  border-bottom:1px solid rgba(0,212,255,0.1);
  display:flex;justify-content:space-between;align-items:center;
}
.feed-title { font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--accent); }
.feed-blink { width:6px;height:6px;border-radius:50%;background:var(--accent3);animation:pulse 1s infinite; }
.feed-body {
  flex:1;padding:16px;overflow-y:auto;
  min-height:350px;max-height:500px;
  font-size:11px;line-height:2;
}
.feed-line { display:flex;gap:10px;align-items:flex-start;margin-bottom:2px; }
.feed-time { color:var(--muted);flex-shrink:0;font-size:10px;margin-top:2px; }
.feed-ok { color:var(--accent3); }
.feed-err { color:var(--accent2); }
.feed-warn { color:var(--amber); }
.feed-info { color:var(--accent); }
.feed-dim { color:var(--muted); }
.feed-cursor { display:inline-block;width:7px;height:13px;background:var(--accent);animation:pulse 1s infinite;vertical-align:middle; }

/* ── REWARD TABLE ── */
.reward-section { margin-bottom:40px;animation: fadeUp 0.6s 0.4s ease both; }
.reward-table { width:100%;border-collapse:collapse;font-size:12px; }
.reward-table th {
  text-align:left;padding:12px 16px;
  background:var(--surface);
  color:var(--muted);font-size:9px;letter-spacing:2px;text-transform:uppercase;
  border-bottom:1px solid var(--border);
}
.reward-table td { padding:12px 16px;border-bottom:1px solid rgba(26,48,72,0.4); }
.reward-table tr:hover td { background:rgba(0,212,255,0.03); }
.g { color:var(--accent3);font-weight:700; }
.r { color:var(--accent2);font-weight:700; }
.a { color:var(--amber);font-weight:700; }
.b { color:var(--accent);font-weight:700; }

/* ── API ENDPOINTS ── */
.endpoints-section { margin-bottom:40px; animation: fadeUp 0.6s 0.5s ease both; }
.endpoint-row {
  display:grid;grid-template-columns:64px 200px 1fr 80px;
  align-items:center;gap:16px;
  padding:14px 16px;
  background:var(--card);border:1px solid var(--border);
  border-bottom:none;font-size:12px;transition:background 0.15s;
}
.endpoint-row:last-child { border-bottom:1px solid var(--border); }
.endpoint-row:hover { background:rgba(0,212,255,0.03); }
.method {
  font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
  padding:3px 8px;border-radius:2px;text-align:center;
}
.post { background:rgba(0,255,157,0.1);color:var(--accent3);border:1px solid rgba(0,255,157,0.2); }
.get  { background:rgba(0,212,255,0.1);color:var(--accent);border:1px solid rgba(0,212,255,0.2); }
.ep-path { color:var(--accent);font-family:var(--font-mono);font-size:12px; }
.ep-desc { color:var(--muted);font-size:11px; }
.ep-try {
  font-size:9px;letter-spacing:1px;padding:4px 10px;
  background:var(--surface);border:1px solid var(--border);
  color:var(--muted);cursor:pointer;transition:all 0.2s;
  font-family:var(--font-mono);
}
.ep-try:hover { border-color:var(--accent);color:var(--accent); }

/* ── FOOTER ── */
footer {
  border-top:1px solid var(--border);
  padding:32px 0;
  display:flex;justify-content:space-between;align-items:center;
  font-size:10px;color:var(--muted);letter-spacing:1px;
}
footer a { color:var(--accent);text-decoration:none; }
footer a:hover { color:var(--text); }

@keyframes fadeDown { from{opacity:0;transform:translateY(-16px);} to{opacity:1;transform:none;} }
@keyframes fadeUp   { from{opacity:0;transform:translateY(16px);} to{opacity:1;transform:none;} }

@media(max-width:900px) {
  .hero,.demo-grid { grid-template-columns:1fr; }
  .stats-row,.tasks { grid-template-columns:repeat(2,1fr); }
  .action-btns { grid-template-columns:1fr; }
  .btn-real,.btn-false { grid-column:span 1; }
  .endpoint-row { grid-template-columns:64px 1fr; }
  .ep-desc,.ep-try { display:none; }
}
</style>
</head>
<body>
<div class="blob blob1"></div>
<div class="blob blob2"></div>
<div class="blob blob3"></div>

<div class="container">

<!-- ── HEADER ── -->
<header>
  <div class="header-top">
    <div class="badges">
      <span class="badge badge-openenv">OpenEnv</span>
      <span class="badge badge-live"><span class="pulse"></span>Live</span>
      <span class="badge badge-tag">Medical AI</span>
      <span class="badge badge-tag">RL Environment</span>
    </div>
    <nav class="nav-links">
      <a href="/docs">API Docs</a>
      <a href="/health">Health</a>
      <a href="https://github.com/mansipatil03/icu-alarm-env" target="_blank">GitHub</a>
    </nav>
  </div>

  <div class="hero">
    <div class="hero-left">
      <h1>ICU Alarm<span class="line2">Fatigue Reducer</span></h1>
      <p class="tagline">
        Every 2 minutes, an ICU nurse responds to an alarm.
        <span class="highlight">99% are false.</span>
        Nurses become desensitized — and miss the 1% that kills.<br/>
        This environment trains AI agents to tell the difference.
      </p>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <div style="background:var(--card);border:1px solid var(--border);padding:10px 16px;font-size:11px;color:var(--muted);">
          🧠 <span style="color:var(--text);">OpenAI Gym Compatible</span>
        </div>
        <div style="background:var(--card);border:1px solid var(--border);padding:10px 16px;font-size:11px;color:var(--muted);">
          🏥 <span style="color:var(--text);">Real Clinical Data Patterns</span>
        </div>
        <div style="background:var(--card);border:1px solid var(--border);padding:10px 16px;font-size:11px;color:var(--muted);">
          ⚡ <span style="color:var(--text);">FastAPI + Docker</span>
        </div>
      </div>
    </div>

    <div class="hero-right">
      <div class="alarm-display">
        <div class="alarm-num" id="alarmTicker">288</div>
        <div class="alarm-label">ICU Alarms per Day (avg)</div>
      </div>
      <div class="alarm-meta">Only <span>~3</span> require immediate action</div>
      <div class="mini-vitals" id="demoVitals">
        <div class="mini-vital"><div class="mv-name">Heart Rate</div><div class="mv-val mv-warn" id="dv-hr">102</div></div>
        <div class="mini-vital"><div class="mv-name">SpO2</div><div class="mv-val mv-crit" id="dv-spo2">91%</div></div>
        <div class="mini-vital"><div class="mv-name">Sys BP</div><div class="mv-val mv-normal" id="dv-bp">118</div></div>
        <div class="mini-vital"><div class="mv-name">Resp Rate</div><div class="mv-val mv-normal" id="dv-rr">16</div></div>
      </div>
      <div style="text-align:center;font-size:10px;color:var(--muted);margin-top:8px;letter-spacing:1px;">
        ↑ LIVE DEMO PATIENT — values update every 3s
      </div>
    </div>
  </div>
</header>

<!-- ── STATS ── -->
<div class="stats-row">
  <div class="stat"><div class="stat-val">3</div><div class="stat-lbl">Task Levels</div></div>
  <div class="stat"><div class="stat-val">99%</div><div class="stat-lbl">ICU False Alarm Rate</div></div>
  <div class="stat"><div class="stat-val">1.0</div><div class="stat-lbl">Max Reward</div></div>
  <div class="stat"><div class="stat-val">0.0</div><div class="stat-lbl">Missed Emergency</div></div>
</div>

<!-- ── TASKS ── -->
<p class="section-title">Task Levels</p>
<div class="tasks">
  <div class="task-card easy selected" onclick="setLevel('easy',this)">
    <div class="task-icon">🟢</div>
    <div class="task-level">Easy · Single Vital</div>
    <div class="task-name">Vital Spike Detection</div>
    <div class="task-desc">Classify one abnormal vital sign reading. Context clues indicate whether it's a sensor artifact or genuine alert.</div>
    <div class="task-badge">Threshold: 0.8</div>
  </div>
  <div class="task-card medium" onclick="setLevel('medium',this)">
    <div class="task-icon">🟡</div>
    <div class="task-level">Medium · Pattern Analysis</div>
    <div class="task-name">Multi-Vital Pattern</div>
    <div class="task-desc">Analyze 3 alarm events across different vital signs in a 10-minute window. Identify escalating patterns.</div>
    <div class="task-badge">Threshold: 0.7</div>
  </div>
  <div class="task-card hard" onclick="setLevel('hard',this)">
    <div class="task-icon">🔴</div>
    <div class="task-level">Hard · Full History</div>
    <div class="task-name">24-Hour Patient Review</div>
    <div class="task-desc">Complete patient record — medications, diagnoses, 24hr vitals trend, alarm log. Classify + recommend nurse action.</div>
    <div class="task-badge">Threshold: 0.6</div>
  </div>
</div>

<!-- ── INTERACTIVE DEMO ── -->
<p class="section-title">Live Demo — Try It</p>
<div class="demo-grid">
  <div class="demo-controls">
    <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:10px;">Select Difficulty</div>
    <div class="level-select">
      <button class="level-btn active-easy" id="lb-easy" onclick="setLevel('easy',null)">🟢 Easy</button>
      <button class="level-btn" id="lb-medium" onclick="setLevel('medium',null)">🟡 Medium</button>
      <button class="level-btn" id="lb-hard" onclick="setLevel('hard',null)">🔴 Hard</button>
    </div>

    <div class="patient-card" id="patientCard">
      <div class="patient-id" id="patientId">PATIENT —</div>
      <div class="patient-prompt" id="patientPrompt"></div>
    </div>

    <div class="vitals-mini" id="vitalsMini"></div>

    <div class="action-btns">
      <button class="btn btn-new" onclick="doReset()">▶ New Patient</button>
      <button class="btn btn-new" style="background:rgba(90,122,148,0.1);color:var(--muted);border-color:var(--border);" onclick="clearFeed()">⌫ Clear</button>
      <button class="btn btn-real" onclick="doClassify('real')">🚨 Classify: REAL Emergency</button>
      <button class="btn btn-false" onclick="doClassify('false')">✓ Classify: FALSE Alarm</button>
    </div>

    <div class="score-display" id="scoreDisplay">
      <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;margin-top:16px;">Last Score</div>
      <div class="score-bar-wrap">
        <div class="score-fill" id="scoreFill"></div>
        <div class="score-num" id="scoreNum">—</div>
      </div>
      <div class="score-verdict" id="scoreVerdict"></div>
    </div>

    <!-- session stats -->
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:16px;">
      <div style="background:var(--surface);border:1px solid var(--border);padding:12px;text-align:center;">
        <div style="font-family:var(--font-display);font-size:24px;color:var(--accent);" id="sTotal">0</div>
        <div style="font-size:8px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);">Tested</div>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);padding:12px;text-align:center;">
        <div style="font-family:var(--font-display);font-size:24px;color:var(--accent3);" id="sCorrect">0</div>
        <div style="font-size:8px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);">Correct</div>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);padding:12px;text-align:center;">
        <div style="font-family:var(--font-display);font-size:24px;color:var(--amber);" id="sAvg">—</div>
        <div style="font-size:8px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);">Avg Score</div>
      </div>
    </div>
  </div>

  <!-- LIVE FEED -->
  <div class="live-feed">
    <div class="feed-header">
      <div class="feed-title">// System Feed</div>
      <div class="feed-blink"></div>
    </div>
    <div class="feed-body" id="feed">
      <div class="feed-line"><span class="feed-time">00:00</span><span class="feed-dim">ICU Alarm Fatigue Reducer — OpenEnv v1.0</span></div>
      <div class="feed-line"><span class="feed-time">00:00</span><span class="feed-dim">Environment initialized. 3 task levels loaded.</span></div>
      <div class="feed-line"><span class="feed-time">00:00</span><span class="feed-info">Select a difficulty and click NEW PATIENT to begin.</span></div>
      <div class="feed-line"><span class="feed-time">00:00</span><span class="feed-dim">Awaiting agent action... <span class="feed-cursor"></span></span></div>
    </div>
  </div>
</div>

<!-- ── REWARD TABLE ── -->
<div class="reward-section">
  <p class="section-title">Reward Structure</p>
  <table class="reward-table">
    <thead>
      <tr><th>Outcome</th><th>Base Score</th><th>Bonus</th><th>Notes</th></tr>
    </thead>
    <tbody>
      <tr><td>Correct classification</td><td class="g">0.6 – 0.8</td><td>—</td><td style="color:var(--muted);font-size:11px;">Base reward</td></tr>
      <tr><td>+ Quality explanation</td><td>—</td><td class="g">+0.15 – 0.2</td><td style="color:var(--muted);font-size:11px;">Clinical reasoning bonus</td></tr>
      <tr><td>+ Recommended nurse action</td><td>—</td><td class="g">+0.15</td><td style="color:var(--muted);font-size:11px;">Action specificity bonus</td></tr>
      <tr><td>+ Confidence ≥ 0.7</td><td>—</td><td class="g">+0.10</td><td style="color:var(--muted);font-size:11px;">Calibration bonus</td></tr>
      <tr><td>Missed real emergency</td><td class="r">0.0</td><td class="r">—</td><td style="color:var(--muted);font-size:11px;">⚠ Patient safety penalty</td></tr>
      <tr><td>False positive</td><td class="a">0.1</td><td>—</td><td style="color:var(--muted);font-size:11px;">Low penalty</td></tr>
      <tr><td><strong>Maximum possible</strong></td><td class="b">1.0</td><td class="b">—</td><td style="color:var(--muted);font-size:11px;">Perfect classification</td></tr>
    </tbody>
  </table>
</div>

<!-- ── ENDPOINTS ── -->
<div class="endpoints-section">
  <p class="section-title">API Endpoints</p>
  <div class="endpoint-row">
    <span class="method post">POST</span>
    <span class="ep-path">/reset</span>
    <span class="ep-desc">Start new episode — returns patient vitals + alarm scenario</span>
    <button class="ep-try" onclick="doReset()">Try →</button>
  </div>
  <div class="endpoint-row">
    <span class="method post">POST</span>
    <span class="ep-path">/step</span>
    <span class="ep-desc">Submit agent classification — returns reward (0.0–1.0)</span>
    <button class="ep-try" onclick="doClassify('real')">Try →</button>
  </div>
  <div class="endpoint-row">
    <span class="method get">GET</span>
    <span class="ep-path">/state</span>
    <span class="ep-desc">Current environment state and step count</span>
    <button class="ep-try" onclick="tryState()">Try →</button>
  </div>
  <div class="endpoint-row">
    <span class="method get">GET</span>
    <span class="ep-path">/health</span>
    <span class="ep-desc">Health check — returns status ok</span>
    <button class="ep-try" onclick="window.open('/health')">Try →</button>
  </div>
  <div class="endpoint-row">
    <span class="method get">GET</span>
    <span class="ep-path">/docs</span>
    <span class="ep-desc">Full interactive Swagger API documentation</span>
    <button class="ep-try" onclick="window.open('/docs')">Open →</button>
  </div>
</div>

<footer>
  <span>ICU Alarm Fatigue Reducer · OpenEnv · mansi-patil · 2026</span>
  <span>
    <a href="/docs">API Docs</a> &nbsp;·&nbsp;
    <a href="https://github.com/mansipatil03/icu-alarm-env" target="_blank">GitHub</a> &nbsp;·&nbsp;
    <a href="/health">Health</a>
  </span>
</footer>

</div><!-- /container -->

<script>
// ── STATE ──
let level = 'easy';
let sessionId = null;
let currentObs = null;
let total = 0, correct = 0, scoreSum = 0;

// ── DEMO VITALS ANIMATION ──
const demoVitals = [
  {id:'dv-hr', vals:[[96,'mv-normal'],[102,'mv-warn'],[88,'mv-normal'],[115,'mv-warn'],[72,'mv-normal']]},
  {id:'dv-spo2', vals:['98%','91%','96%','93%','99%']},
  {id:'dv-bp', vals:['118','142','108','125','116']},
  {id:'dv-rr', vals:['14','22','16','18','12']},
];
let demoIdx = 0;
setInterval(() => {
  demoIdx = (demoIdx+1) % 5;
  const hr = demoVitals[0].vals[demoIdx];
  const hrEl = document.getElementById('dv-hr');
  if (hrEl) { hrEl.textContent = hr[0]; hrEl.className = 'mv-val '+hr[1]; }
  const spo2El = document.getElementById('dv-spo2');
  if (spo2El) { const v = demoVitals[1].vals[demoIdx]; spo2El.textContent = v; spo2El.className = 'mv-val ' + (parseFloat(v)<95?'mv-crit':'mv-normal'); }
  const bpEl = document.getElementById('dv-bp');
  if (bpEl) { const v = demoVitals[2].vals[demoIdx]; bpEl.textContent = v; bpEl.className = 'mv-val '+(parseInt(v)>135?'mv-warn':'mv-normal'); }
  const rrEl = document.getElementById('dv-rr');
  if (rrEl) { const v = demoVitals[3].vals[demoIdx]; rrEl.textContent = v; rrEl.className = 'mv-val '+(parseInt(v)>20||parseInt(v)<12?'mv-warn':'mv-normal'); }
}, 3000);

// ── FEED ──
function getTime() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}
function feed(type, msg) {
  const f = document.getElementById('feed');
  const cursors = f.querySelectorAll('.feed-cursor');
  cursors.forEach(c => c.closest('.feed-line').remove());
  const line = document.createElement('div');
  line.className = 'feed-line';
  const cls = {ok:'feed-ok',err:'feed-err',warn:'feed-warn',info:'feed-info',dim:'feed-dim'}[type]||'feed-dim';
  line.innerHTML = `<span class="feed-time">${getTime()}</span><span class="${cls}">${msg}</span>`;
  f.appendChild(line);
  const cur = document.createElement('div');
  cur.className = 'feed-line';
  cur.innerHTML = `<span class="feed-time">${getTime()}</span><span class="feed-dim"><span class="feed-cursor"></span></span>`;
  f.appendChild(cur);
  f.scrollTop = f.scrollHeight;
}
function clearFeed() {
  const f = document.getElementById('feed');
  f.innerHTML = `<div class="feed-line"><span class="feed-time">${getTime()}</span><span class="feed-dim">Feed cleared. Ready.<span class="feed-cursor"></span></span></div>`;
}

// ── LEVEL SELECT ──
function setLevel(l, cardEl) {
  level = l;
  document.querySelectorAll('.task-card').forEach(c => c.classList.remove('selected'));
  if (cardEl) cardEl.classList.add('selected');
  ['easy','medium','hard'].forEach(x => {
    const b = document.getElementById('lb-'+x);
    b.className = 'level-btn' + (x===l ? ` active-${l}` : '');
  });
  feed('info', `Task level → ${l.toUpperCase()}`);
}

// ── RESET ──
async function doReset() {
  sessionId = 'ui_' + Math.random().toString(36).slice(2);
  feed('info', `Generating ${level.toUpperCase()} patient scenario...`);
  try {
    const r = await fetch(`/reset?task_level=${level}&session_id=${sessionId}`, {method:'POST'});
    currentObs = await r.json();
    document.getElementById('patientCard').style.display = 'block';
    document.getElementById('patientId').textContent = `PATIENT // ${currentObs.patient_id}`;
    document.getElementById('patientPrompt').textContent = (currentObs.prompt||'').slice(0,250) + '...';
    // Show mini vitals
    if (currentObs.vitals) {
      const vm = document.getElementById('vitalsMini');
      vm.style.display = 'grid';
      const v = currentObs.vitals;
      const checks = { heart_rate: x=>x<60||x>100, systolic_bp: x=>x<90||x>160, spo2: x=>x<95 };
      const labels = { heart_rate:'HR', systolic_bp:'SBP', diastolic_bp:'DBP', spo2:'SpO2', respiratory_rate:'RR', temperature:'Temp' };
      vm.innerHTML = Object.entries(v).map(([k,val]) => {
        const cls = checks[k] ? (checks[k](val)?'critical':'normal') : 'normal';
        return `<div class="vm"><div class="vm-k">${labels[k]||k}</div><div class="vm-v ${cls}">${val}</div></div>`;
      }).join('');
    }
    feed('ok', `✓ Patient ${currentObs.patient_id} loaded`);
    if (currentObs.alarm_history?.length) {
      feed('warn', `⚠ ${currentObs.alarm_history.length} alarm(s) in history`);
    }
    feed('info', 'Classify: REAL emergency or FALSE alarm?');
    document.getElementById('scoreDisplay').style.display = 'none';
  } catch(e) {
    feed('err', `Error loading patient: ${e.message}`);
  }
}

// ── CLASSIFY ──
async function doClassify(classification) {
  if (!currentObs) { feed('err', 'No patient loaded — click NEW PATIENT first!'); return; }
  feed('info', `Submitting: ${classification.toUpperCase()}...`);
  try {
    const r = await fetch(`/step?session_id=${sessionId}`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        classification,
        confidence: 0.85,
        explanation: `Manual classification: marked as ${classification} alarm based on vital pattern analysis.`,
        recommended_action: classification==='real' ? 'Notify attending physician immediately' : 'Continue standard monitoring'
      })
    });
    const result = await r.json();
    const score = result.reward;
    const info = result.info || {};
    // Update stats
    total++; scoreSum += score;
    if (score >= 0.6) correct++;
    document.getElementById('sTotal').textContent = total;
    document.getElementById('sCorrect').textContent = correct;
    document.getElementById('sAvg').textContent = (scoreSum/total).toFixed(2);
    // Score bar
    document.getElementById('scoreDisplay').style.display = 'block';
    document.getElementById('scoreFill').style.width = (score*100)+'%';
    document.getElementById('scoreNum').textContent = score.toFixed(3);
    // Verdict
    const v = document.getElementById('scoreVerdict');
    if (info.result === 'dangerous_miss') {
      v.textContent = '☠ DANGEROUS MISS — That was a real emergency!';
      v.style.color = 'var(--accent2)';
      feed('err', `☠ DANGEROUS MISS! Ground truth: ${info.ground_truth}. Score: ${score}`);
    } else if (score >= 0.7) {
      v.textContent = `✓ CORRECT — ${info.result || 'Well done!'}`;
      v.style.color = 'var(--accent3)';
      feed('ok', `✓ Correct! Score: ${score.toFixed(3)}`);
    } else {
      v.textContent = `✗ Incorrect — Ground truth: ${info.ground_truth}`;
      v.style.color = 'var(--amber)';
      feed('warn', `✗ Wrong. Ground truth: ${info.ground_truth}. Score: ${score.toFixed(3)}`);
    }
    currentObs = null;
    document.getElementById('patientCard').style.display = 'none';
    document.getElementById('vitalsMini').style.display = 'none';
  } catch(e) {
    feed('err', `Step error: ${e.message}`);
  }
}

async function tryState() {
  if (!sessionId) { feed('warn', 'No active session. Load a patient first.'); return; }
  const r = await fetch(`/state?session_id=${sessionId}`);
  const d = await r.json();
  feed('info', `State: step=${d.step_count} done=${d.done} level=${d.task_level}`);
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
    return _envs[session_id].reset()

@app.post("/step", response_model=StepResult)
def step(action: Action, session_id: str = "default"):
    if session_id not in _envs:
        raise HTTPException(400, "No active session. Call /reset first.")
    try:
        return _envs[session_id].step(action)
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

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
