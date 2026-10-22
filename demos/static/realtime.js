import {
  LANG, applyStrings, canvas2d, cssVar, flash, fmtClock, makeRow, setStatus, showServerInfo, socketUrl,
} from "/static/common.js";

const T = {
  ja: {
    title: "リアルタイム文字起こしデモ", navLongform: "長時間音声", navRealtime: "リアルタイム",
    mic: "マイクで話す", pick: "音声ファイルを再生…", stop: "停止",
    delay: "遅延（エンドツーエンド・中央値）", model: "モデル処理", audioLen: "音声の長さ",
    idle: "待機中", recording: "録音中", playing: "再生中", stopped: "停止",
    settled: "確定", pending: "認識中", now: "現在",
    empty: "マイクで話すか音声ファイルを再生すると、ここに書き起こしが表示されます",
    connected: "接続済み", disconnected: "切断", unavailable: "サーバーに接続できません", micDenied: "マイクを使用できません",
  },
  en: {
    title: "Real-time transcription demo", navLongform: "Long-form", navRealtime: "Real-time",
    mic: "Use microphone", pick: "Play an audio file…", stop: "Stop",
    delay: "End-to-end delay (median)", model: "Model time", audioLen: "Audio length",
    idle: "Idle", recording: "Recording", playing: "Playing", stopped: "Stopped",
    settled: "Settled", pending: "Pending", now: "now",
    empty: "Speak into the microphone or play an audio file; the transcript appears here",
    connected: "connected", disconnected: "disconnected", unavailable: "server unavailable", micDenied: "microphone unavailable",
  },
}[LANG];
applyStrings(T);

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), mic: $("mic"), micLabel: $("mic-label"), pick: $("pick"), pickLabel: $("pick-label"),
  file: $("file"), delay: $("m-delay"), model: $("m-model"), audio: $("m-audio"), canvas: $("wave"),
  axis: $("axis"), state: $("state"), stateText: $("state-text"), transcript: $("transcript"),
};

const TARGET_RATE = 16000, FRAME_MS = 100;
const WINDOW_S = 20, BAR_S = 0.05; // waveform: the last 20 s, one bar per 50 ms
const SPEECH_RMS = 0.012; // the server's speech threshold (SegmenterConfig.silence_rms)
let ws = null, running = null, starting = false;
let delays = [], modelTimes = [], levels = [], settleMarks = [];
let shownPhrases = 0;

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};
const elapsed = () => (running ? running.ctx.currentTime - running.t0 : 0);

function setState(key, active) {
  els.stateText.textContent = T[key];
  els.state.className = "state" + (active ? " on" : " off");
}

// ---------------------------------------------------------------------------- waveform

function drawWave() {
  const [g, w, h] = canvas2d(els.canvas);
  const bars = Math.round(WINDOW_S / BAR_S), step = w / bars, now = elapsed();
  g.fillStyle = cssVar("--rule");
  g.fillRect(0, h / 2, w, 1);
  const shown = levels.slice(-bars);
  shown.forEach(([level, speech], i) => {
    const x = w - (shown.length - i) * step;
    const amp = Math.max(1, Math.min(1, level * 7) * (h / 2 - 4));
    g.fillStyle = speech ? cssVar("--green") : cssVar("--wave");
    g.fillRect(x, h / 2 - amp, Math.max(1, step * 0.6), amp * 2);
  });
  g.fillStyle = cssVar("--ink"); // a line where each phrase was settled
  for (const mark of settleMarks) {
    const x = w - ((now - mark) / WINDOW_S) * w;
    if (x >= 0 && x <= w) g.fillRect(Math.round(x), 0, 1.5, h);
  }
}

function drawAxis() {
  els.axis.replaceChildren();
  for (let s = WINDOW_S; s >= 0; s -= 5) {
    const label = document.createElement("span");
    label.style.left = `${((WINDOW_S - s) / WINDOW_S) * 100}%`;
    label.textContent = s ? `−${s}s` : T.now;
    label.className = s === WINDOW_S ? "first" : s === 0 ? "last" : "";
    els.axis.append(label);
  }
}

function sampleLevel() {
  if (!running) return;
  const buffer = new Float32Array(running.analyser.fftSize);
  running.analyser.getFloatTimeDomainData(buffer);
  let sum = 0;
  for (const x of buffer) sum += x * x;
  const rms = Math.sqrt(sum / buffer.length);
  levels.push([rms, rms > SPEECH_RMS]);
  if (levels.length > 2000) levels.splice(0, 1000);
  els.audio.textContent = fmtClock(elapsed());
  drawWave();
}

// ---------------------------------------------------------------------------- transcript

function render(phrases, pending) {
  if (!shownPhrases && (phrases.length || pending)) els.transcript.replaceChildren();
  for (const old of els.transcript.querySelectorAll(".row.pending")) old.remove();
  for (const phrase of phrases.slice(shownPhrases)) {
    const row = makeRow(phrase.start, phrase.text);
    els.transcript.append(row);
    flash(row);
    settleMarks.push(elapsed());
  }
  shownPhrases = phrases.length;
  if (pending) {
    const row = makeRow(pending.start, pending.text);
    row.classList.add("pending");
    els.transcript.append(row);
  }
  els.transcript.scrollTo({ top: els.transcript.scrollHeight, behavior: "smooth" });
}

function handle(msg) {
  if (msg.type === "error") {
    setStatus(els.status, msg.message, "error");
    stop();
  } else if (msg.type === "update") {
    render(msg.phrases, msg.pending);
    if (msg.latency) {
      const { stamp, model_ms: modelMs } = msg.latency;
      // End to end: the newest audio leaving the audio thread -> this text on screen.
      requestAnimationFrame(() => {
        delays.push(performance.now() - stamp);
        modelTimes.push(modelMs);
        if (delays.length > 60) { delays.shift(); modelTimes.shift(); }
        els.delay.textContent = Math.round(median(delays.slice(-20)));
        els.model.textContent = Math.round(median(modelTimes.slice(-20)));
      });
    }
  }
}

function connect() {
  ws = new WebSocket(socketUrl("/ws/realtime"));
  ws.binaryType = "arraybuffer";
  ws.onopen = () => showServerInfo(els.status, T);
  ws.onclose = () => {
    setStatus(els.status, T.disconnected, "error");
    stop();
    setTimeout(connect, 1500);
  };
  ws.onmessage = (event) => handle(JSON.parse(event.data));
}

// ---------------------------------------------------------------------------- capture

async function start(file) {
  if (!ws || ws.readyState !== WebSocket.OPEN || running || starting) return;
  starting = true; // `running` is only set once capture begins, after several awaits
  try {
    await capture(file);
  } finally {
    starting = false;
  }
}

async function capture(file) {
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/static/capture-worklet.js");
  let source, stream = null;
  if (file) {
    source = ctx.createBufferSource();
    source.buffer = await ctx.decodeAudioData(await file.arrayBuffer());
    source.connect(ctx.destination); // play it aloud
    source.onended = () => setTimeout(() => running && running.source === source && stop(), 900);
  } else {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch {
      setStatus(els.status, T.micDenied, "error");
      ctx.close();
      return;
    }
    source = ctx.createMediaStreamSource(stream);
  }
  const node = new AudioWorkletNode(ctx, "capture", {
    processorOptions: { targetRate: TARGET_RATE, frameMs: FRAME_MS },
  });
  // Each frame carries the time it left the audio thread (8-byte float64, this
  // page's clock); the server echoes the newest one with every update.
  node.port.onmessage = (event) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const frame = new Uint8Array(8 + event.data.byteLength);
    new DataView(frame.buffer).setFloat64(0, performance.now(), true);
    frame.set(new Uint8Array(event.data), 8);
    ws.send(frame.buffer);
  };
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(node);
  source.connect(analyser);
  const silent = ctx.createGain();
  silent.gain.value = 0;
  node.connect(silent).connect(ctx.destination); // keep the graph running without echoing the mic

  ws.send(JSON.stringify({ type: "start" }));
  delays = []; modelTimes = []; levels = []; settleMarks = []; shownPhrases = 0;
  els.delay.textContent = els.model.textContent = "–";
  els.transcript.replaceChildren();
  running = { ctx, node, stream, source, analyser, t0: ctx.currentTime };
  running.timer = setInterval(sampleLevel, BAR_S * 1000);
  if (file) source.start();
  const [button, label] = file ? [els.pick, els.pickLabel] : [els.mic, els.micLabel];
  button.classList.add("on");
  label.textContent = T.stop;
  (file ? els.mic : els.pick).disabled = true;
  setState(file ? "playing" : "recording", true);
}

function stop() {
  const r = running;
  running = null;
  if (!r) return;
  clearInterval(r.timer);
  try { r.source.stop?.(); } catch { /* already stopped */ }
  r.node.port.onmessage = null;
  r.node.disconnect();
  r.stream?.getTracks().forEach((track) => track.stop());
  r.ctx.close();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
  for (const [button, label, key] of [[els.mic, els.micLabel, "mic"], [els.pick, els.pickLabel, "pick"]]) {
    button.classList.remove("on");
    button.disabled = false;
    label.textContent = T[key];
  }
  setState("stopped", false);
}

els.mic.addEventListener("click", () => (running ? stop() : start(null)));
els.pick.addEventListener("click", () => (running ? stop() : els.file.click()));
els.file.addEventListener("change", () => {
  if (els.file.files[0]) start(els.file.files[0]);
  els.file.value = "";
});
window.addEventListener("resize", drawWave);
drawAxis();
drawWave();
connect();
