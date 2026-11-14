"""
Strata-Coder Web GUI, Monitor Dashboard, OpenAI API Server & CLI for Grawnch/Kimi-K3.1-Pro-Coder.

Usage:
  1. Start the Web GUI + Monitor Dashboard + OpenAI Server on http://127.0.0.1:8080:
       python -m strata_coder.serve --model /path/to/Kimi-K3.1-Pro-Coder --port 8080

  2. Or chat directly in the terminal:
       python -m strata_coder.serve --model /path/to/Kimi-K3.1-Pro-Coder --cli
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List
import torch
from .engine import KimiCoderEngine


_HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Strata Monitor & Chat · Kimi-K3.1-Pro-Coder</title>
<style>
  :root { --bg:#0a0c0f; --card:#12151b; --border:#222731; --text:#f1f5f9; --muted:#8b95a5; --green:#22c55e; --cyan:#00e5ff; --amber:#f59e0b; }
  * { box-sizing:border-box; margin:0; padding:0; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  body { background:var(--bg); color:var(--text); padding:18px 24px; }
  .top-bar { display:flex; align-items:center; justify-content:space-between; background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 18px; margin-bottom:16px; }
  .badges { display:flex; align-items:center; gap:12px; font-size:13px; }
  .badge { padding:4px 10px; border-radius:999px; font-weight:600; font-size:12px; background:#1a202c; color:var(--muted); }
  .badge.active { background:rgba(34,197,94,0.16); color:var(--green); border:1px solid rgba(34,197,94,0.4); }
  .grid { display:grid; grid-template-columns:repeat(6, 1fr); gap:12px; margin-bottom:16px; }
  @media(max-width:1100px){ .grid { grid-template-columns:repeat(3, 1fr); } }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
  .card-label { font-size:12px; color:var(--muted); margin-bottom:6px; display:flex; justify-content:space-between; }
  .card-val { font-size:26px; font-weight:700; letter-spacing:-0.5px; }
  .card-unit { font-size:13px; color:var(--muted); font-weight:400; margin-left:4px; }
  .card-sub { font-size:11.5px; color:var(--muted); margin-top:4px; }
  .bar-track { height:6px; background:#1e2430; border-radius:4px; overflow:hidden; margin-top:10px; }
  .bar-fill { height:100%; background:var(--green); border-radius:4px; transition:width 0.3s; }
  .main-split { display:grid; grid-template-columns:360px 1fr; gap:16px; }
  @media(max-width:960px){ .main-split { grid-template-columns:1fr; } }
  .chat-box { display:flex; flex-direction:column; height:520px; background:var(--card); border:1px solid var(--border); border-radius:12px; overflow:hidden; }
  .chat-msgs { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:12px; }
  .msg { padding:10px 14px; border-radius:10px; max-width:88%; line-height:1.5; font-size:14px; white-space:pre-wrap; }
  .msg.user { align-self:flex-end; background:#1e293b; color:#f8fafc; border:1px solid #334155; }
  .msg.assistant { align-self:flex-start; background:#161b22; border:1px solid var(--border); }
  .think-box { font-size:12.5px; color:var(--muted); border-left:2px solid var(--cyan); padding-left:10px; margin-bottom:8px; }
  .chat-input { display:flex; gap:8px; padding:12px; border-top:1px solid var(--border); background:#0e1116; }
  .chat-input input { flex:1; background:#161b22; border:1px solid var(--border); color:var(--text); padding:10px 14px; border-radius:8px; font-size:14px; outline:none; }
  .chat-input button { background:var(--green); color:#051b0d; font-weight:700; border:none; padding:0 18px; border-radius:8px; cursor:pointer; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }
  th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; }
</style>
</head>
<body>
  <div class="top-bar">
    <div class="badges">
      <strong>Model state</strong>
      <span id="st-badge" class="badge active">Idle</span>
      <span id="st-model" style="color:var(--muted)">Grawnch/Kimi-K3.1-Pro-Coder</span>
    </div>
    <div id="st-summary" style="font-size:12.5px;color:var(--muted)">Strata-Coder 3-Tier Engine</div>
  </div>
  <div class="grid">
    <div class="card"><div class="card-label"><span>Speed</span><span>Engine</span></div><div class="card-val"><span id="m-tps">0.0</span><span class="card-unit">t/s</span></div><div class="card-sub">Prefill: <span id="m-prefill">0.0</span> t/s</div></div>
    <div class="card"><div class="card-label"><span>GPU load</span><span>Sensor</span></div><div class="card-val"><span id="m-gpu">0</span><span class="card-unit">%</span></div><div class="card-sub" id="m-gpuname">CUDA Device</div><div class="bar-track"><div id="b-gpu" class="bar-fill" style="width:0%"></div></div></div>
    <div class="card"><div class="card-label"><span>VRAM</span><span>3-Tier Hot</span></div><div class="card-val"><span id="m-vram">0.0</span><span class="card-unit">/ <span id="m-vram-tot">0</span> GB</span></div><div class="card-sub"><span id="m-vram-exp">0</span> experts cached</div><div class="bar-track"><div id="b-vram" class="bar-fill" style="width:0%"></div></div></div>
    <div class="card"><div class="card-label"><span>System RAM</span><span>3-Tier Warm</span></div><div class="card-val"><span id="m-ram">0.0</span><span class="card-unit">/ <span id="m-ram-tot">0</span> GB</span></div><div class="card-sub"><span id="m-ram-exp">0</span> warm experts pinned</div><div class="bar-track"><div id="b-ram" class="bar-fill" style="width:0%"></div></div></div>
    <div class="card"><div class="card-label"><span>GPU temp</span><span>Sensor</span></div><div class="card-val"><span id="m-temp">--</span><span class="card-unit">°C</span></div><div class="card-sub">Thermal sensor</div></div>
    <div class="card"><div class="card-label"><span>Power</span><span>Sensor</span></div><div class="card-val"><span id="m-pwr">--</span><span class="card-unit">W</span></div><div class="card-sub" id="m-plim">GPU power draw</div></div>
  </div>
  <div class="main-split">
    <div class="card">
      <div class="card-label"><strong style="color:var(--text);font-size:14px">3-Tier Residency & Recent Requests</strong></div>
      <div style="margin-top:12px;font-size:13px;display:flex;justify-content:space-between"><span>Experts in VRAM (Tier 1)</span><span id="t1-lbl" style="color:var(--muted)">0</span></div>
      <div class="bar-track"><div id="b-t1" class="bar-fill" style="width:0%"></div></div>
      <div style="margin-top:12px;font-size:13px;display:flex;justify-content:space-between"><span>Experts in System RAM (Tier 2)</span><span id="t2-lbl" style="color:var(--muted)">0</span></div>
      <div class="bar-track"><div id="b-t2" class="bar-fill" style="width:0%;background:var(--cyan)"></div></div>
      <div style="margin-top:18px"><strong style="font-size:13px">Recent requests</strong>
        <table><thead><tr><th>Time</th><th>Out</th><th>Decode t/s</th><th>Duration</th></tr></thead><tbody id="req-tb"></tbody></table>
      </div>
    </div>
    <div class="chat-box">
      <div class="chat-msgs" id="chat-msgs"><div class="msg assistant">Strata-Coder ready. Ask a coding or math question below!</div></div>
      <form class="chat-input" onsubmit="sendMsg(event)">
        <input id="chat-in" placeholder="Message Kimi-K3.1-Pro-Coder..." autocomplete="off"/>
        <button type="submit">Send</button>
      </form>
    </div>
  </div>
<script>
let hist = [];
async function pollMonitor(){
  try {
    const r = await fetch('/api/monitor');
    const d = await r.json();
    document.getElementById('st-badge').textContent = d.state;
    document.getElementById('st-model').textContent = d.model;
    document.getElementById('m-tps').textContent = d.decode_tps.toFixed(1);
    document.getElementById('m-prefill').textContent = d.prefill_tps.toFixed(1);
    document.getElementById('m-gpu').textContent = d.gpu_util;
    document.getElementById('b-gpu').style.width = Math.min(100, d.gpu_util) + '%';
    document.getElementById('m-gpuname').textContent = d.gpu_name;
    document.getElementById('m-vram').textContent = d.vram_used_gb.toFixed(1);
    document.getElementById('m-vram-tot').textContent = d.vram_total_gb.toFixed(0);
    document.getElementById('m-vram-exp').textContent = d.vram_experts.toLocaleString();
    document.getElementById('b-vram').style.width = Math.min(100, (d.vram_used_gb / Math.max(1, d.vram_total_gb))*100) + '%';
    document.getElementById('m-ram').textContent = d.ram_used_gb.toFixed(1);
    document.getElementById('m-ram-tot').textContent = d.ram_total_gb.toFixed(0);
    document.getElementById('m-ram-exp').textContent = d.ram_experts.toLocaleString();
    document.getElementById('b-ram').style.width = Math.min(100, (d.ram_used_gb / Math.max(1, d.ram_total_gb))*100) + '%';
    document.getElementById('m-temp').textContent = d.gpu_temp ?? '--';
    document.getElementById('m-pwr').textContent = d.gpu_power ?? '--';
    document.getElementById('m-plim').textContent = d.gpu_power_limit ? ('of ' + d.gpu_power_limit + ' W limit') : 'GPU power draw';
    document.getElementById('t1-lbl').textContent = d.vram_experts.toLocaleString() + ' · ' + d.vram_expert_gb.toFixed(1) + ' GB';
    document.getElementById('t2-lbl').textContent = d.ram_experts.toLocaleString() + ' · ' + d.ram_expert_gb.toFixed(1) + ' GB';
    document.getElementById('b-t1').style.width = Math.min(100, (d.vram_expert_gb / Math.max(1, d.vram_total_gb))*100) + '%';
    document.getElementById('b-t2').style.width = Math.min(100, (d.ram_expert_gb / Math.max(1, d.ram_total_gb))*100) + '%';
    document.getElementById('req-tb').innerHTML = (d.requests || []).slice(0,6).map(x =>
      `<tr><td>${x.time}</td><td>${x.tokens}</td><td>${x.tps.toFixed(1)}</td><td>${x.dur.toFixed(1)}s</td></tr>`
    ).join('');
  } catch(e){}
}
setInterval(pollMonitor, 2000); pollMonitor();
async function sendMsg(ev){
  ev.preventDefault();
  const inp = document.getElementById('chat-in');
  const txt = inp.value.trim();
  if(!txt) return;
  inp.value = '';
  const box = document.getElementById('chat-msgs');
  const uDiv = document.createElement('div'); uDiv.className = 'msg user'; uDiv.textContent = txt; box.appendChild(uDiv);
  hist.push({role:'user', content:txt});
  const aDiv = document.createElement('div'); aDiv.className = 'msg assistant';
  const tDiv = document.createElement('div'); tDiv.className = 'think-box'; tDiv.style.display = 'none';
  const cDiv = document.createElement('div');
  aDiv.appendChild(tDiv); aDiv.appendChild(cDiv); box.appendChild(aDiv);
  box.scrollTop = box.scrollHeight;
  let fullContent = '';
  try {
    const resp = await fetch('/v1/chat/completions', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:hist, stream:true, max_tokens:512})
    });
    if(!resp.ok){
      const errTxt = await resp.text();
      cDiv.textContent = '[HTTP ' + resp.status + '] ' + errTxt;
      hist.pop();
      pollMonitor();
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if(payload === '[DONE]') break;
        try {
          const j = JSON.parse(payload);
          const delta = j.choices[0].delta || {};
          if(delta.reasoning_content){ tDiv.style.display = 'block'; tDiv.textContent += delta.reasoning_content; }
          if(delta.content){ fullContent += delta.content; cDiv.textContent = fullContent; }
          box.scrollTop = box.scrollHeight;
        } catch(e){}
      }
    }
    if(fullContent && !fullContent.includes('[Server Error:')){
      hist.push({role:'assistant', content:fullContent});
    } else {
      if(!fullContent) cDiv.textContent = '[Empty response from server — check server log]';
      hist.pop();
    }
  } catch(e){
    cDiv.textContent = fullContent || ('[Connection Error: ' + e + ']');
    hist.pop();
  }
  pollMonitor();
}
</script>
</body>
</html>
"""


def _query_nvidia_smi() -> Dict[str, Any]:
    info: Dict[str, Any] = {"gpu_name": "CUDA Device", "gpu_util": 0, "gpu_temp": None, "gpu_power": None, "gpu_power_limit": None}
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"],
            timeout=1.5, text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        info["gpu_util"] = int(float(parts[0]))
        info["gpu_temp"] = int(float(parts[1]))
        info["gpu_power"] = int(float(parts[2]))
        info["gpu_power_limit"] = int(float(parts[3]))
    except Exception:
        pass
    return info


def run_cli(engine: KimiCoderEngine, max_tokens: int, temperature: float):
    print("\n========================================================")
    print("  Strata-Coder Interactive Terminal (Kimi-K3.1-Pro-Coder)")
    print("  Type '/reset' to clear history, '/quit' to exit.")
    print("========================================================")
    messages = []
    while True:
        try:
            user_in = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_in in ("/quit", "/exit"):
            break
        if user_in == "/reset":
            messages = []
            print("(conversation reset)")
            continue
        if not user_in:
            continue
        messages.append({"role": "user", "content": user_in})
        print("model> ", end="", flush=True)
        t0 = time.time()
        pieces = []
        for tok_str in engine.generate_stream(messages, max_new_tokens=max_tokens, temperature=temperature):
            print(tok_str, end="", flush=True)
            pieces.append(tok_str)
        dt = max(time.time() - t0, 1e-6)
        print(f"\n[{len(pieces)} tokens in {dt:.1f}s ({len(pieces)/dt:.2f} tok/s)]")
        messages.append({"role": "assistant", "content": "".join(pieces)})
def make_handler(engine: KimiCoderEngine, default_max_tokens: int):
    state: Dict[str, Any] = {
        "phase": "Idle",
        "decode_tps": 0.0,
        "prefill_tps": 0.0,
        "requests": [],
    }

    def build_monitor_payload() -> Dict[str, Any]:
        smi = _query_nvidia_smi()
        ps = getattr(engine, "peer_stats", None) or {}
        vram_exp = sum(len(m.vram_cache) for m in engine.layer_managers.values()) + int(ps.get("vram_experts", 0))
        ram_exp = sum(len(m.ram_cache) for m in engine.layer_managers.values()) + int(ps.get("ram_experts", 0))
        vram_exp_gb = sum(m.vram_used_bytes for m in engine.layer_managers.values()) / (1024 ** 3) + float(ps.get("vram_used_gb", 0.0))
        ram_exp_gb = sum(m.ram_used_bytes for m in engine.layer_managers.values()) / (1024 ** 3) + float(ps.get("ram_used_gb", 0.0))
        vram_alloc = (
            sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )
        m_raw = str(engine.model_id_or_dir)
        m_disp = Path(m_raw).name if os.path.exists(m_raw) else m_raw
        return {
            "state": state["phase"],
            "model": m_disp,
            "decode_tps": state["decode_tps"],
            "prefill_tps": state["prefill_tps"],
            "vram_used_gb": max(vram_alloc, vram_exp_gb),
            "vram_total_gb": engine.total_vram_gb + float(ps.get("vram_total_gb", 0.0)),
            "vram_experts": vram_exp,
            "vram_expert_gb": vram_exp_gb,
            "ram_used_gb": ram_exp_gb,
            "ram_total_gb": engine.total_ram_gb + float(ps.get("ram_total_gb", 0.0)),
            "ram_experts": ram_exp,
            "ram_expert_gb": ram_exp_gb,
            "requests": state["requests"],
            **smi,
        }

    class ChatHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html", "/monitor"):
                raw = _HTML_DASHBOARD.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path.startswith("/api/monitor"):
                raw = json.dumps(build_monitor_payload()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/health":
                payload = {
                    "service": "strata",
                    "model": "kimi-sc-lut-3tier",
                    "max_context": 262144,
                    "images": False,
                    "loaded": True,
                    "api_key": False,
                }
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/status":
                payload = {
                    "busy": state["phase"] == "Generating",
                    "phase": state["phase"].lower(),
                    "tokens_per_s": state["decode_tps"],
                }
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/v1/status":
                mon = build_monitor_payload()
                payload = {
                    "engine": "strata-coder-3tier",
                    "activity": {"requests": len(state["requests"])},
                    "machine": {
                        "gpu": {
                            "used_mib": round(mon["vram_used_gb"] * 1024),
                            "total_mib": round(mon["vram_total_gb"] * 1024),
                        },
                        "ram": {
                            "used_gib": round(mon["ram_used_gb"], 1),
                            "total_gib": round(mon["ram_total_gb"], 1),
                        },
                    },
                }
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path.startswith("/v1/models"):
                payload = {"object": "list", "data": [{"id": "kimi-k3.1-pro-coder", "object": "model"}]}
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages", [])
            stream = bool(body.get("stream", True))
            max_tokens = int(body.get("max_tokens", default_max_tokens))
            temperature = float(body.get("temperature", 0.0))
            reasoning_effort = str(body.get("reasoning_effort", "none"))

            state["phase"] = "Generating"
            t0 = time.time()
            t_first = None
            tok_cnt = 0
            headers_sent = False

            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    headers_sent = True
                    in_think = False
                    for piece in engine.generate_stream(
                        messages, max_new_tokens=max_tokens, temperature=temperature, reasoning_effort=reasoning_effort
                    ):
                        now = time.time()
                        if t_first is None:
                            t_first = now
                            state["prefill_tps"] = round(32.0 / max(t_first - t0, 1e-3), 1)
                        tok_cnt += 1
                        state["decode_tps"] = round(tok_cnt / max(now - (t_first or t0), 1e-3), 2)
                        if "<think>" in piece:
                            in_think = True
                            piece = piece.replace("<think>", "")
                        if "</think>" in piece:
                            in_think = False
                            piece = piece.replace("</think>", "")
                        if not piece:
                            continue
                        delta = {"reasoning_content": piece} if in_think else {"content": piece}
                        chunk = {"choices": [{"index": 0, "delta": delta}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    pieces = list(engine.generate_stream(
                        messages, max_new_tokens=max_tokens, temperature=temperature, reasoning_effort=reasoning_effort
                    ))
                    tok_cnt = len(pieces)
                    out_text = "".join(p.replace("<think>", "").replace("</think>", "") for p in pieces)
                    resp = {
                        "id": "chatcmpl-strata",
                        "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": out_text}, "finish_reason": "stop"}],
                    }
                    raw = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    headers_sent = True
                    self.wfile.write(raw)

                dur = max(time.time() - t0, 1e-3)
                tps = round(tok_cnt / max(time.time() - (t_first or t0), 1e-3), 2)
                state["decode_tps"] = tps
                state["requests"].insert(0, {"time": time.strftime("%H:%M:%S"), "tokens": tok_cnt, "tps": tps, "dur": dur})
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                for mgr in getattr(engine, "layer_managers", {}).values():
                    if hasattr(mgr, "release_transient_gpu_cache"):
                        mgr.release_transient_gpu_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                err_msg = f"{type(exc).__name__}: {exc}"
                try:
                    if stream and headers_sent:
                        err_chunk = {"choices": [{"index": 0, "delta": {"content": f"\n[Server Error: {err_msg}]"}}]}
                        self.wfile.write(f"data: {json.dumps(err_chunk)}\n\n".encode("utf-8"))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        self.close_connection = True
                    elif not headers_sent:
                        raw = json.dumps({"error": {"message": err_msg, "type": type(exc).__name__}}).encode("utf-8")
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                except OSError:
                    pass
            finally:
                state["phase"] = "Idle"

    return ChatHandler


def main():
    ap = argparse.ArgumentParser(description="Strata-Coder Web GUI, Server & CLI for Kimi-K3.1-Pro-Coder")
    ap.add_argument("--model", default="Grawnch/Kimi-K3.1-Pro-Coder", help="Local shard directory or HF repo ID")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-layers", type=int, default=93)
    ap.add_argument("--start-layer", type=int, default=0, help="First layer index on this node (inclusive)")
    ap.add_argument("--end-layer", type=int, default=None, help="Last layer index on this node (exclusive)")
    ap.add_argument("--peer", type=str, default=None, help="Next pipeline node host:port over 100G DAC (e.g. 10.10.10.2:9100)")
    ap.add_argument("--worker-port", type=int, default=None, help="Run as 100G DAC pipeline RPC worker listening on this TCP port")
    ap.add_argument("--vram-gb", type=float, default=None)
    ap.add_argument("--ram-gb", type=float, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cli", action="store_true", help="Run interactive terminal chat directly")
    ap.add_argument("--prompt", type=str, default=None, help="Run a single prompt and exit")
    args = ap.parse_args()

    print(
        f"[Strata-Coder] Initializing KimiCoderEngine ({args.model}, layers={args.start_layer}..{args.end_layer or args.max_layers})...",
        flush=True,
    )
    engine = KimiCoderEngine(
        model_id_or_dir=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        max_layers=args.max_layers,
        total_vram_gb=args.vram_gb,
        total_ram_gb=args.ram_gb,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        peer_addr=args.peer,
    )
    print(
        f"[Strata-Coder] Ready | Device: {engine.device} | Compute Dtype: {engine.out_dtype} | "
        f"Layers: {engine.start_layer}..{engine.end_layer - 1} | Peer: {engine.peer_addr or 'None'} | "
        f"Detected VRAM: {engine.total_vram_gb:.1f} GB | Detected RAM: {engine.total_ram_gb:.1f} GB | "
        f"Resident Layer Caches: {len(engine.layer_managers)}",
        flush=True,
    )

    if args.worker_port is not None:
        engine.serve_rpc_worker(host=args.host if args.host != "127.0.0.1" else "0.0.0.0", port=args.worker_port)
        return

    if args.prompt is not None:
        for tok_str in engine.generate_stream(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature):
            print(tok_str, end="", flush=True)
        print(flush=True)
        return

    if args.cli:
        run_cli(engine, max_tokens=args.max_tokens, temperature=args.temperature)
        return

    server = None
    bound_port = args.port
    for candidate_port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer((args.host, candidate_port), make_handler(engine, args.max_tokens))
            bound_port = candidate_port
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError(f"Could not bind HTTP server on {args.host}:{args.port}..{args.port + 19}")
    print(f"[Strata-Coder] Web GUI & Monitor Dashboard: http://{args.host}:{bound_port}/", flush=True)
    print(f"[Strata-Coder] OpenAI-compatible API:       http://{args.host}:{bound_port}/v1/chat/completions", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
