import {
  LANG, applyStrings, canvas2d, cssVar, flash, fmtClock, makeRow, setStatus, showServerInfo, socketUrl,
} from "/static/common.js";

const T = {
  ja: {
    title: "長時間音声の書き起こしデモ", navLongform: "長時間音声", navRealtime: "リアルタイム",
    pick: "音声ファイルを選択…", batch: "バッチ", audioLen: "音声の長さ", elapsed: "処理時間",
    empty: "音声ファイルを選ぶか、ここにドロップすると書き起こしが表示されます",
    connected: "接続済み", disconnected: "切断", unavailable: "サーバーに接続できません", uploading: "アップロード中", decoding: "音声を読み込み中",
    transcribing: "書き起こし中", chunks: "チャンク",
    done: (audio, seconds) => `${audio}の音声を ${seconds} 秒で書き起こしました`,
    duration: (s) => {
      const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = Math.round(s % 60);
      if (h) return `${h}時間${m ? m + "分" : ""}`;
      return m ? `${m}分${r ? r + "秒" : ""}` : `${r}秒`;
    },
  },
  en: {
    title: "Long-form transcription demo", navLongform: "Long-form", navRealtime: "Real-time",
    pick: "Choose an audio file…", batch: "Batch", audioLen: "Audio length", elapsed: "Processing time",
    empty: "Choose an audio file, or drop one here, and the transcript appears here",
    connected: "connected", disconnected: "disconnected", unavailable: "server unavailable", uploading: "uploading", decoding: "reading audio",
    transcribing: "transcribing", chunks: "chunks",
    done: (audio, seconds) => `${audio} of audio transcribed in ${seconds} s`,
    duration: (s) => {
      const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = Math.round(s % 60);
      if (h) return `${h} h${m ? ` ${m} min` : ""}`;
      return m ? `${m} min${r ? ` ${r} s` : ""}` : `${r} s`;
    },
  },
}[LANG];
applyStrings(T);

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), pick: $("pick"), file: $("file"), filename: $("filename"), batch: $("batch"),
  audio: $("s-audio"), elapsed: $("s-elapsed"), canvas: $("wave"), axis: $("axis"),
  play: $("play"), ptime: $("ptime"), phase: $("phase"), headline: $("headline"),
  transcript: $("transcript"), player: $("player"),
};

const SLICE = 1 << 20; // upload in 1 MB frames
let ws = null, busy = false;
let plan = null, doneUntil = 0, clockStart = 0, clockOn = false;
let pending = [], flushQueued = false, playingRow = null;

// ---------------------------------------------------------------------------- waveform

function drawWave() {
  const [g, w, h] = canvas2d(els.canvas);
  if (!plan) {
    g.fillStyle = cssVar("--rule");
    g.fillRect(0, h / 2, w, 1);
    return;
  }
  const { duration, peaks, chunks } = plan, step = 3;
  g.fillStyle = cssVar("--green-s"); // finished chunks get a pale cell behind the waveform
  for (const [a, b] of chunks) if (b <= doneUntil + 1e-6) g.fillRect((a / duration) * w, 0, ((b - a) / duration) * w, h);
  for (let x = 0; x < w; x += step) {
    const i0 = Math.floor((x / w) * peaks.length);
    const i1 = Math.max(i0 + 1, Math.floor(((x + step) / w) * peaks.length));
    let peak = 0;
    for (let i = i0; i < i1 && i < peaks.length; i++) peak = Math.max(peak, peaks[i]);
    const amp = Math.max(1, (peak / 255) * (h / 2 - 4));
    g.fillStyle = ((x + step / 2) / w) * duration <= doneUntil ? cssVar("--green") : cssVar("--wave");
    g.fillRect(x, h / 2 - amp, 2, amp * 2);
  }
  g.fillStyle = cssVar("--rule2"); // chunk boundaries
  for (const [, b] of chunks.slice(0, -1)) g.fillRect(Math.round((b / duration) * w), 0, 1, h);
  if (els.player.src && (els.player.currentTime > 0 || !els.player.paused)) {
    g.fillStyle = cssVar("--ink"); // playhead
    g.fillRect(Math.round((els.player.currentTime / duration) * w) - 1, 0, 2, h);
  }
}

function drawAxis() {
  els.axis.replaceChildren();
  if (!plan) return;
  const total = plan.duration;
  const step = [60, 120, 300, 600, 900, 1800, 3600].find((s) => total / s <= 7) || 7200;
  for (let t = 0; t <= total; t += step) {
    const label = document.createElement("span");
    label.style.left = `${(t / total) * 100}%`;
    label.textContent = fmtClock(t);
    if (t === 0) label.className = "first";
    else if (t / total > 0.97) label.className = "last";
    els.axis.append(label);
  }
}
window.addEventListener("resize", drawWave);

// ---------------------------------------------------------------------------- run

function tick() {
  if (!clockOn) return;
  els.elapsed.textContent = ((performance.now() - clockStart) / 1000).toFixed(2);
  requestAnimationFrame(tick);
}

// Results arrive a batch at a time; draw at most once per frame so the page
// keeps up and the clock stays honest.
function flushRows() {
  flushQueued = false;
  if (!pending.length) return;
  for (const item of pending) {
    const row = makeRow(item.start, item.text);
    row.dataset.start = item.start;
    els.transcript.append(row);
    flash(row);
    doneUntil = Math.max(doneUntil, item.end);
  }
  pending = [];
  drawWave();
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function setBusy(state) {
  busy = state;
  els.pick.disabled = els.batch.disabled = state;
}

function handle(msg) {
  if (msg.type === "error") {
    setStatus(els.status, msg.message, "error");
    clockOn = false;
    setBusy(false);
  } else if (msg.type === "progress") {
    els.phase.textContent = `${T.uploading} ${Math.round((msg.received / msg.total) * 100)}%`;
    if (msg.received >= msg.total) els.phase.textContent = T.decoding;
  } else if (msg.type === "plan") {
    plan = msg;
    doneUntil = 0;
    els.audio.textContent = fmtClock(msg.duration);
    els.phase.textContent = `${T.transcribing} · 0 / ${msg.chunks.length} ${T.chunks}`;
    els.transcript.replaceChildren();
    drawAxis();
    drawWave();
    clockStart = performance.now();
    clockOn = true;
    tick();
  } else if (msg.type === "chunks") {
    pending.push(...msg.items);
    els.phase.textContent = `${T.transcribing} · ${msg.done} / ${plan.chunks.length} ${T.chunks}`;
    if (!flushQueued) { flushQueued = true; requestAnimationFrame(flushRows); }
  } else if (msg.type === "done") {
    const seconds = ((performance.now() - clockStart) / 1000).toFixed(2); // one clock, no jump
    clockOn = false;
    flushRows();
    doneUntil = msg.duration;
    drawWave();
    els.elapsed.textContent = seconds;
    els.headline.textContent = T.done(T.duration(msg.duration), seconds);
    els.phase.textContent = `${msg.chunks} ${T.chunks} · batch ${els.batch.value}`;
    els.play.disabled = false;
    els.canvas.classList.add("seekable");
    els.transcript.classList.add("seekable");
    els.ptime.textContent = `0:00 / ${fmtClock(msg.duration)}`;
    els.transcript.scrollTop = 0;
    setBusy(false);
  }
}

function connect() {
  ws = new WebSocket(socketUrl("/ws/longform"));
  ws.binaryType = "arraybuffer";
  ws.onopen = () => showServerInfo(els.status, T);
  ws.onclose = () => {
    setStatus(els.status, T.disconnected, "error");
    setBusy(false);
    clockOn = false;
    setTimeout(connect, 1500);
  };
  ws.onmessage = (event) => handle(JSON.parse(event.data));
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function transcribe(file) {
  if (busy || !ws || ws.readyState !== WebSocket.OPEN) return;
  setBusy(true);
  stopPlayback();
  plan = null;
  drawWave();
  drawAxis();
  els.audio.textContent = els.elapsed.textContent = "–";
  els.headline.textContent = els.ptime.textContent = "";
  els.play.disabled = true;
  els.canvas.classList.remove("seekable");
  els.transcript.classList.remove("seekable");
  els.filename.textContent = file.name;
  if (els.player.src) URL.revokeObjectURL(els.player.src);
  els.player.src = URL.createObjectURL(file);
  // Upload on this socket only: stop if it closes (a reconnect is a new socket)
  // or the server refuses the file (an error ends `busy`).
  const socket = ws;
  const uploading = () => busy && socket === ws && socket.readyState === WebSocket.OPEN;
  socket.send(JSON.stringify({ type: "start", bytes: file.size, batch_size: Number(els.batch.value) }));
  // send() only queues; pace on bufferedAmount so the progress shown is real.
  for (let offset = 0; offset < file.size && uploading(); offset += SLICE) {
    const data = await file.slice(offset, offset + SLICE).arrayBuffer();
    while (socket.bufferedAmount > 4 * SLICE && uploading()) await sleep(15);
    if (uploading()) socket.send(data);
  }
}

// ---------------------------------------------------------------------------- playback

function seek(seconds) {
  els.player.currentTime = Math.max(0, seconds);
  els.player.play();
}

function stopPlayback() {
  els.player.pause();
  els.play.classList.remove("on");
  if (playingRow) playingRow.classList.remove("playing");
  playingRow = null;
}

function followPlayback() {
  const t = els.player.currentTime;
  els.ptime.textContent = `${fmtClock(t)} / ${fmtClock(plan ? plan.duration : els.player.duration || 0)}`;
  let current = null;
  for (const row of els.transcript.querySelectorAll(".row")) {
    if (Number(row.dataset.start) <= t + 0.05) current = row;
    else break;
  }
  if (current !== playingRow) {
    if (playingRow) playingRow.classList.remove("playing");
    playingRow = current;
    if (current) {
      current.classList.add("playing");
      const box = els.transcript;
      if (current.offsetTop < box.scrollTop || current.offsetTop + current.offsetHeight > box.scrollTop + box.clientHeight) {
        box.scrollTo({ top: current.offsetTop - 12, behavior: "smooth" });
      }
    }
  }
  drawWave();
  if (!els.player.paused) requestAnimationFrame(followPlayback);
}

els.player.addEventListener("play", () => { els.play.classList.add("on"); requestAnimationFrame(followPlayback); });
els.player.addEventListener("pause", () => { els.play.classList.remove("on"); followPlayback(); });
els.play.addEventListener("click", () => (els.player.paused ? els.player.play() : els.player.pause()));
els.transcript.addEventListener("click", (event) => {
  const row = event.target.closest(".row");
  if (row && plan && !busy) seek(Number(row.dataset.start));
});
els.canvas.addEventListener("click", (event) => {
  if (!plan || busy) return;
  const box = els.canvas.getBoundingClientRect();
  seek(((event.clientX - box.left) / box.width) * plan.duration);
});

// ---------------------------------------------------------------------------- input

els.pick.addEventListener("click", () => !busy && els.file.click());
els.file.addEventListener("change", () => els.file.files[0] && transcribe(els.file.files[0]));
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => {
  event.preventDefault();
  if (event.dataTransfer.files[0]) transcribe(event.dataTransfer.files[0]);
});

drawWave();
connect();
