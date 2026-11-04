// RST client — plain ES module, no build step.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};

const SKILLS = [
  { id: "rookie", label: "Rookie", pips: 1, desc: "Thinks slowly, reaches for loose links, and now and then blurts out a banned-letter word." },
  { id: "casual", label: "Casual", pips: 2, desc: "A friendly table guest. Solid links, the occasional slip." },
  { id: "sharp", label: "Sharp", pips: 3, desc: "Quick and reliable. Rarely makes a mistake." },
  { id: "pro", label: "Pro", pips: 4, desc: "Fast, and nearly always plays the strongest association." },
  { id: "grandmaster", label: "Grandmaster", pips: 5, desc: "Never slips. Picks words whose best answers start with a banned letter — leaving you stuck." },
];
const AVATAR_COLORS = ["#C2512F", "#2F6F62", "#B5832A", "#3B5B8C", "#7A4E8A", "#5E7D33", "#A23E5A", "#8A6A3B"];
const REASONS = {
  "already used": "Already played this game.",
  "not in dictionary": "Not a word the judge knows.", "not related enough": "The judge didn't buy that link.",
  "time's up": "Ran out of time.", "letters only": "Letters only, please.", "too short": "Too short.",
};

const S = {
  ws: null, code: null, me: null, state: null, skill: "casual", lastWordShown: null,
  typing: {}, offset: 0, pendingSoloBots: 0, retry: 0, leaving: false,
};

// ---------------------------------------------------------------- theme
const root = document.documentElement;
const savedTheme = store.get("rst.theme");
if (savedTheme) root.dataset.theme = savedTheme;
$("#themeBtn").onclick = () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  store.set("rst.theme", root.dataset.theme);
};

// ---------------------------------------------------------------- utils
function toast(msg, ms = 2400) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), ms);
}
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function colorFor(id) { let h = 0; for (const c of id) h = (h * 31 + c.charCodeAt(0)) >>> 0; return AVATAR_COLORS[h % AVATAR_COLORS.length]; }
function avatar(p) { return `<span class="avatar ${p.isBot ? "bot" : ""}" style="--c:${colorFor(p.id)}">${esc(p.name[0]?.toUpperCase() || "?")}</span>`; }
function player(id) { return S.state?.players.find(p => p.id === id); }
function skillLabel(id) { return SKILLS.find(s => s.id === id)?.label || id; }
function show(view) {
  for (const v of ["home", "lobby", "game", "over"]) $(`#view-${v}`).hidden = v !== view;
  $("#roomChip").hidden = view === "home";
}

// ---------------------------------------------------------------- home
const nameInput = $("#nameInput"), codeInput = $("#codeInput");
nameInput.value = store.get("rst.name", "");
const pathCode = location.pathname.match(/^\/r\/([A-Za-z0-9]{5})/);
if (pathCode) { codeInput.value = pathCode[1].toUpperCase(); setTimeout(() => (nameInput.value ? $("#joinBtn") : nameInput).focus(), 50); }
codeInput.addEventListener("input", () => { codeInput.value = codeInput.value.toUpperCase().replace(/[^A-Z0-9]/g, ""); });

function needName() {
  const n = nameInput.value.trim();
  if (!n) { $("#homeError").textContent = "Pick a name first."; nameInput.focus(); return null; }
  store.set("rst.name", n); $("#homeError").textContent = ""; return n;
}
async function createRoom(soloBots = 0) {
  const n = needName(); if (!n) return;
  try {
    const r = await fetch("/api/rooms", { method: "POST" });
    const { code } = await r.json();
    S.pendingSoloBots = soloBots;
    connect(code, n);
  } catch { $("#homeError").textContent = "Couldn't reach the server."; }
}
$("#createBtn").onclick = () => createRoom(0);
$("#soloBtn").onclick = () => createRoom(2);
$("#joinBtn").onclick = async () => {
  const n = needName(); if (!n) return;
  const code = codeInput.value.trim().toUpperCase();
  if (code.length !== 5) { $("#homeError").textContent = "Room codes are 5 characters."; return; }
  const r = await fetch(`/api/rooms/${code}`);
  if (!r.ok) { $("#homeError").textContent = "No room with that code."; return; }
  const info = await r.json();
  if (info.phase !== "lobby" && !store.get(`rst.tok.${code}`)) { $("#homeError").textContent = "That game has already started."; return; }
  if (info.players >= info.maxPlayers && !store.get(`rst.tok.${code}`)) { $("#homeError").textContent = "That room is full."; return; }
  connect(code, n);
};
[nameInput, codeInput].forEach(el => el.addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); (codeInput.value.length === 5 ? $("#joinBtn") : $("#createBtn")).click(); }
}));

// ---------------------------------------------------------------- socket
function connect(code, name) {
  clearTimeout(S.retryTimer);
  if (S.ws && S.ws.readyState <= 1) { const old = S.ws; S.ws = null; old.close(); }
  S.code = code; S.leaving = false;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${code}`);
  S.ws = ws;
  ws.onopen = () => { S.retry = 0; ws.send(JSON.stringify({ name, token: store.get(`rst.tok.${code}`) })); };
  ws.onmessage = e => { if (S.ws === ws) onMessage(JSON.parse(e.data)); };
  ws.onclose = () => {
    if (S.leaving || S.ws !== ws) return;
    if (S.retry++ < 6) {
      toast("Connection lost — reconnecting…");
      S.retryTimer = setTimeout(() => { if (!S.leaving && S.ws === ws) connect(code, name); }, 600 * S.retry);
    } else { toast("Disconnected."); goHome(); }
  };
  history.replaceState(null, "", `/r/${code}`);
  $("#roomChip").textContent = `Room ${code}`;
}
function send(type, data = {}) { if (S.ws?.readyState === 1) S.ws.send(JSON.stringify({ type, ...data })); }
function goHome() {
  clearTimeout(S.retryTimer);
  S.leaving = true; const ws = S.ws; S.ws = null; ws?.close(); S.state = null; S.code = null;
  history.replaceState(null, "", "/"); show("home");
}

function onMessage(m) {
  switch (m.type) {
    case "welcome":
      S.me = m.playerId; store.set(`rst.tok.${S.code}`, m.token); break;
    case "state": {
      const prev = S.state; S.state = m.state; S.offset = m.state.serverTime - Date.now() / 1000;
      if (S.pendingSoloBots && m.state.hostId === S.me && m.state.phase === "lobby") {
        for (let i = 0; i < S.pendingSoloBots; i++) send("add_bot", { difficulty: i ? "sharp" : "casual" });
        S.pendingSoloBots = 0;
      }
      render(prev); break;
    }
    case "move": onMove(m.move); break;
    case "typing": S.typing[m.playerId] = m.text; renderTyping(); break;
    case "error": toast(m.message); if (!S.state) { $("#homeError").textContent = m.message; goHome(); } break;
    case "kicked": toast("The host removed you from the room."); store.del(`rst.tok.${S.code}`); goHome(); break;
  }
}

// ---------------------------------------------------------------- render
function render(prev) {
  const st = S.state; if (!st) return;
  $("#roomChip").textContent = `Room ${st.code}`;
  if (st.phase === "lobby") { show("lobby"); renderLobby(); }
  else if (st.phase === "playing") { show("game"); renderGame(prev); }
  else { show("over"); renderOver(prev); }
}

function renderLobby() {
  const st = S.state, host = st.hostId === S.me;
  $("#bigCode").textContent = st.code;
  $("#seatCount").textContent = `${st.players.length} / ${st.maxPlayers}`;
  const seats = st.players.map(p => `
    <li class="seat">
      ${avatar(p)}
      <div class="who"><div class="name">${esc(p.name)}</div>
        <div class="tag">${p.isBot ? `AI · ${skillLabel(p.difficulty)}` : p.id === st.hostId ? "host" : "player"}${p.id === S.me ? " · you" : ""}${!p.connected ? " · away" : ""}</div></div>
      ${host && p.id !== S.me ? `<button class="x" data-kick="${p.id}" aria-label="Remove ${esc(p.name)}">×</button>` : ""}
    </li>`);
  for (let i = st.players.length; i < st.maxPlayers; i++) seats.push(`<li class="seat empty">open seat</li>`);
  $("#seatList").innerHTML = seats.join("");
  $$("[data-kick]").forEach(b => b.onclick = () => send("remove", { playerId: b.dataset.kick }));

  $("#botPanel").hidden = !host;
  $("#addBotBtn").disabled = st.players.length >= st.maxPlayers;
  for (const seg of $$(".seg[data-key]")) {
    const v = String(st.settings[seg.dataset.key]);
    seg.classList.toggle("locked", !host);
    for (const b of $$("button", seg)) { b.classList.toggle("on", b.dataset.v === v); b.disabled = !host && b.dataset.v !== v; }
  }
  renderLetters(host);
  $("#hostHint").textContent = host ? "You're the host — you set the rules." : `${esc(player(st.hostId)?.name || "The host")} sets the rules.`;
  const start = $("#startBtn");
  start.hidden = !host;
  start.disabled = st.players.length < 2;
  start.textContent = st.players.length < 2 ? "Need 2+ players" : `Start game · ${st.players.length} players`;
}

// skill picker
$(".skill-picker").innerHTML = SKILLS.map(s => `
  <button class="skill" role="radio" aria-checked="${s.id === S.skill}" data-skill="${s.id}">
    <span class="pips">${[1, 2, 3, 4, 5].map(i => `<i class="${i <= s.pips ? "on" : ""}"></i>`).join("")}</span>${s.label}
  </button>`).join("");
function pickSkill(id) {
  S.skill = id;
  $$(".skill").forEach(b => b.setAttribute("aria-checked", b.dataset.skill === id));
  $("#skillDesc").textContent = SKILLS.find(s => s.id === id).desc;
}
$$(".skill").forEach(b => b.onclick = () => pickSkill(b.dataset.skill));
pickSkill(S.skill);
$("#addBotBtn").onclick = () => send("add_bot", { difficulty: S.skill });
$$(".seg[data-key]").forEach(seg => seg.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b || seg.classList.contains("locked")) return;
  const v = seg.dataset.key === "strictness" ? b.dataset.v : Number(b.dataset.v);
  send("settings", { settings: { [seg.dataset.key]: v } });
}));
// banned letters
const LETTER_PRESETS = ["rst", "xyz", "aei", "cps"];
const letterInputs = $$("#letterPick input");
function renderLetters(host) {
  const st = S.state, set = st.settings.banned;
  // While the host is typing letters, don't overwrite the boxes with the server's sorted copy.
  const editing = letterInputs.includes(document.activeElement);
  letterInputs.forEach((inp, i) => {
    if (!editing) inp.value = set[i].toUpperCase();
    inp.disabled = !host;
  });
  $$("#letterPresets button").forEach(b => {
    b.classList.toggle("on", b.dataset.set === set);
    b.disabled = !host && b.dataset.set !== set;
    b.hidden = !host && b.dataset.set === "random";
  });
  $("#letterPresets").classList.toggle("locked", !host);
  const share = st.bannedShare ?? 0, pct = Math.round(share * 100);
  const level = share < 0.08 ? "easy" : share < 0.2 ? "medium" : share < 0.27 ? "hard" : "brutal";
  $("#blockFill").style.width = `${Math.min(100, share / 0.35 * 100)}%`;
  $("#blockFill").classList.toggle("hard", share >= 0.2);
  $("#blockLabel").textContent = `blocks ${pct}% of words · ${level}`;
}
function sendLetters(set) { send("settings", { settings: { banned: set } }); }
$("#letterPresets").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b || $("#letterPresets").classList.contains("locked")) return;
  let set = b.dataset.set;
  if (set === "random") {
    const pool = "abcdefghijklmnopqrstuvwxyz".split("").filter(c => !"xqz".includes(c));
    do { set = [...pool].sort(() => Math.random() - .5).slice(0, 3).sort().join(""); } while (LETTER_PRESETS.includes(set));
  }
  sendLetters(set);
});
function onLetterChange(i) {
  const set = letterInputs.map(x => x.value.toLowerCase()).join("");
  if (letterInputs[i].value && i < 2) letterInputs[i + 1].focus();
  if (set.length === 3) {
    if (new Set(set).size < 3) { toast("Pick three different letters."); return; }
    sendLetters(set);
  }
}
letterInputs.forEach((inp, i) => {
  // A letter key always replaces the box's letter (a full maxlength=1 box would ignore it).
  inp.addEventListener("keydown", e => {
    if (/^[a-z]$/i.test(e.key) && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault(); inp.value = e.key.toUpperCase(); onLetterChange(i);
    }
  });
  // Mobile keyboards and paste don't always send usable key events.
  inp.addEventListener("input", () => {
    inp.value = inp.value.replace(/[^a-z]/gi, "").slice(-1).toUpperCase();
    onLetterChange(i);
  });
  inp.addEventListener("focus", () => inp.select());
  inp.addEventListener("blur", () => setTimeout(() => {
    if (S.state?.phase === "lobby" && !letterInputs.includes(document.activeElement)) renderLetters(S.state.hostId === S.me);
  }, 0));
});
$("#startBtn").onclick = () => send("start");
$("#leaveBtn").onclick = () => { store.del(`rst.tok.${S.code}`); goHome(); };
$("#copyLink").onclick = async () => {
  const url = `${location.origin}/r/${S.code}`;
  try { await navigator.clipboard.writeText(url); toast("Invite link copied."); } catch { toast(url, 5000); }
};

// game
const moveInput = $("#moveInput"), moveForm = $("#moveForm");
function renderGame(prev) {
  const st = S.state, cur = player(st.currentId), mine = st.currentId === S.me;
  const word = st.chain.at(-1)?.word || "—";
  const cw = $("#currentWord");
  if (S.lastWordShown !== word) { cw.textContent = word; cw.classList.remove("enter"); void cw.offsetWidth; cw.classList.add("enter"); S.lastWordShown = word; S.typing = {}; }
  $("#bannedStrip").innerHTML = st.settings.banned.toUpperCase().split("").map(c => `<b>${c}</b>`).join("");
  $("#bannedStrip").title = `Banned first letters: ${st.settings.banned.toUpperCase().split("").join(", ")}`;
  $("#endBtn").hidden = st.hostId !== S.me;
  $("#turnLabel").textContent = mine ? "Your turn — link a word to" : `${cur?.name || "…"} is thinking${cur?.isBot ? " (AI)" : ""}`;

  $("#chain").innerHTML = st.chain.slice(0, -1).map((m, i) =>
    `<span class="link ${i === 0 ? "seed" : ""}" title="${esc(player(m.playerId)?.name || "start")}">${esc(m.word)}</span><span class="sep">→</span>`).join("");
  $("#chain").scrollLeft = 1e6;

  $("#players").innerHTML = st.players.map(p => `
    <li class="pl ${p.id === S.me ? "me" : ""} ${p.id === st.currentId ? "turn" : ""} ${p.lives <= 0 ? "out" : ""}">
      ${avatar(p)}
      <div class="who">
        <div class="name">${esc(p.name)} ${p.id === S.me ? '<span class="badge you">you</span>' : ""}${p.isBot ? `<span class="badge">${skillLabel(p.difficulty)}</span>` : ""}${!p.connected && !p.isBot ? '<span class="badge">away</span>' : ""}</div>
        <div class="meta"><span class="lives">${Array.from({ length: st.settings.lives }, (_, i) => `<i class="${i < p.lives ? "" : "gone"}"></i>`).join("")}</span>
          ${p.id === st.currentId && !p.isBot && p.id !== S.me ? '<span class="typing-dot"></span>' : ""}</div>
      </div>
      <span class="score">${p.score}</span>
    </li>`).join("");

  renderFeed();
  moveForm.classList.toggle("waiting", !mine);
  moveInput.disabled = !mine; $("#moveBtn").disabled = !mine;
  const becameMine = mine && (!prev || prev.currentId !== S.me || prev.phase !== "playing");
  if (becameMine) { moveInput.value = ""; updateWarn(); moveInput.focus(); }
  if (!mine) moveInput.placeholder = "Wait for your turn…"; else moveInput.placeholder = "Your word…";
  renderTyping();
  startFuse();
}

function renderFeed() {
  const st = S.state;
  $("#feed").innerHTML = st.history.slice().reverse().map(m => {
    const p = player(m.playerId);
    const pct = m.score == null ? 0 : Math.round(m.score * 100);
    return `<li class="${m.ok ? "ok" : "bad"}">
      <span class="w">${esc(m.word)}</span>
      <span class="verdict">${m.ok ? `${pct}% link` : esc(m.reason || "rejected")}</span>
      <span class="by">${esc(p?.name || "")}</span>
      ${m.ok || m.score != null ? `<span class="meter"><i style="width:${pct}%;${m.ok ? "" : "background:var(--accent)"}"></i></span>` : ""}
    </li>`;
  }).join("") || `<li class="by">The first word is on the table. Moves appear here.</li>`;
}

function renderTyping() {
  const st = S.state; if (!st || st.phase !== "playing") return;
  const cur = st.currentId, txt = S.typing[cur];
  $("#typing").textContent = cur !== S.me && txt ? `${player(cur)?.name}: ${txt}` : "";
}

function onMove(m) {
  if (!m.ok && m.playerId === S.me) {
    $("#moveWrap").classList.remove("shake"); void $("#moveWrap").offsetWidth; $("#moveWrap").classList.add("shake");
    const banned = /^starts with ([A-Z])$/.exec(m.reason || "");
    toast(banned ? `Starts with ${banned[1]} — banned letter.` : REASONS[m.reason] || m.reason || "Rejected", 2800);
  } else if (!m.ok) {
    const p = player(m.playerId); toast(`${p?.name || "Someone"} lost a life — ${m.reason}`, 2000);
  }
}

function updateWarn() {
  const v = moveInput.value.trim().toLowerCase();
  const bad = !!v && (S.state?.settings.banned || "rst").includes(v[0]);
  $("#moveWrap").classList.toggle("warn", bad);
  $("#moveWarn").textContent = bad ? `Starts with ${v[0].toUpperCase()}` : "";
}
let typingT = 0;
moveInput.addEventListener("input", () => {
  updateWarn();
  const now = Date.now();
  if (now - typingT > 250) { typingT = now; send("typing", { text: moveInput.value.slice(0, 32) }); }
});
moveForm.addEventListener("submit", e => {
  e.preventDefault();
  const w = moveInput.value.trim();
  if (!w || S.state?.currentId !== S.me) return;
  send("submit", { word: w }); moveInput.value = ""; updateWarn();
});

let fuseRaf = 0;
function startFuse() {
  cancelAnimationFrame(fuseRaf);
  const fill = $("#fuse");
  const tick = () => {
    const st = S.state; if (!st || st.phase !== "playing" || !st.deadline) return;
    const left = st.deadline - (Date.now() / 1000 + S.offset);
    const frac = Math.max(0, Math.min(1, left / (st.turnLength || st.settings.turnSeconds)));
    fill.style.transform = `scaleX(${frac})`;
    fill.classList.toggle("hot", left < 4);
    if (left > -1) fuseRaf = requestAnimationFrame(tick);
  };
  tick();
}

// over
function renderOver(prev) {
  const st = S.state, w = player(st.winnerId);
  const how = st.endedBy === "host" ? " on points" : "";
  $("#winnerLine").innerHTML = w ? (w.id === S.me ? `You <em>win</em>${how}.` : `${esc(w.name)} <em>wins</em>${how}.`) : "No one survived.";
  $("#finalChain").innerHTML = st.chain.map((m, i) => `<span><b>${esc(m.word)}</b>${i < st.chain.length - 1 ? " → " : ""}</span>`).join(" ");
  const ranked = (st.standings || []).map(player).filter(Boolean);
  $("#standings").innerHTML = ranked.map((p, i) => `<li><span class="rank">${i + 1}</span>${avatar(p)}<span class="name">${esc(p.name)}${p.isBot ? ` <span class="badge">${skillLabel(p.difficulty)}</span>` : ""}<span class="sub">${p.lives > 0 ? `${p.lives} ${p.lives === 1 ? "life" : "lives"} left` : "out"} · ${p.misses} ${p.misses === 1 ? "miss" : "misses"}</span></span><span class="score">${p.score}</span></li>`).join("");
  $("#againBtn").hidden = st.hostId !== S.me;
  if (prev?.phase === "playing" && w?.id === S.me) confetti();
}
$("#againBtn").onclick = () => send("lobby");
$("#endBtn").onclick = () => {
  if (confirm("End the game for everyone? The leader on points wins.")) send("end");
};
$("#homeBtn").onclick = () => { store.del(`rst.tok.${S.code}`); goHome(); };

function confetti() {
  const cols = ["var(--accent)", "var(--ink)", "var(--gold)", "var(--ok)"];
  for (let i = 0; i < 90; i++) {
    const c = document.createElement("i");
    c.className = "confetti";
    c.style.left = Math.random() * 100 + "vw";
    c.style.background = cols[i % cols.length];
    c.style.animationDuration = 1.8 + Math.random() * 1.8 + "s";
    c.style.animationDelay = Math.random() * .4 + "s";
    c.style.borderRadius = i % 3 ? "2px" : "50%";
    document.body.appendChild(c);
    setTimeout(() => c.remove(), 4200);
  }
}

// ---------------------------------------------------------------- boot
fetch("/api/health").then(r => r.json()).then(h => { $("#modelTag").textContent = h.model || ""; }).catch(() => {});
show("home");
// Rejoin a room after a refresh if we still hold a seat token.
if (pathCode && store.get(`rst.tok.${pathCode[1].toUpperCase()}`) && nameInput.value) connect(pathCode[1].toUpperCase(), nameInput.value);
